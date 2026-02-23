#!/usr/bin/env python3
"""
law_checker_pipelined — Pipelined sandbox compliance checker using Qwen3Guard-Gen-8B (8 replicas).

Pipeline:
  0. Start sglang_qwen3_model.sh on GPU device=2 with 8 replicas (ports 30000-30007)
  1. Load compiled laws from --law <path> (single JSONL file), law by law:
       - compiled.type == "CHECK"           → whitelist
       - compiled.type == "OBEY" or "DENY"  → blacklist
  2. For each request (up to --max-req) in --req <path>:
       a. Strip "policy" from metadata
       b. Blacklist phase: send all blacklist checkers in parallel across 8 replicas.
          If ANY returns compliant=false → DENIED, skip whitelist.
       c. Whitelist phase: distribute whitelist across 8 replicas (each ~1/8).
          If ALL return compliant=true → PERMITTED.
  3. Score:
       - permitted  & policy ∈ {PERMIT, AMBIGUOUS}  → +1
       - denied     & policy ∈ {FORBID, AMBIGUOUS}  → +1
  4. Save results + metrics to logs/; print final score.
  5. Stop sglang.
"""

import argparse
import json
import re
import time
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx
from openai import OpenAI

PROJECT_DIR      = Path(__file__).parent.parent.parent
SCRIPTS_DIR      = PROJECT_DIR / "scripts"
DEFAULT_LAW      = PROJECT_DIR / "datasets" / "compiled" / "HIPAA.jsonl"
DEFAULT_REQ      = PROJECT_DIR / "datasets" / "req" / "req.jsonl"
DEFAULT_LOG_DIR  = PROJECT_DIR / "experiments" / "law_checker_pipelined"
DEFAULT_REPLICAS = 8


# ── sglang lifecycle ──────────────────────────────────────────────────────────

def replica_urls(base_port: int, n: int) -> list[str]:
    return [f"http://localhost:{base_port + i}/v1" for i in range(n)]


def start_sglang(device: str, base_port: int, n_replicas: int, max_total_tokens: int) -> None:
    print(f"[sglang] Starting Qwen3 {n_replicas}x replicas on GPU device={device}, "
          f"ports {base_port}-{base_port+n_replicas-1} "
          f"(max-total-tokens={max_total_tokens} per replica) ...")
    subprocess.run(
        [str(SCRIPTS_DIR / "sglang_qwen3_model.sh"), "start",
         "--device", device, "--port", str(base_port), "--instances", str(n_replicas),
         "--max-total-tokens", str(max_total_tokens)],
        check=True,
    )
    print("[sglang] Server ready.")


def stop_sglang(base_port: int) -> None:
    print("[sglang] Stopping replicas ...")
    subprocess.run(
        [str(SCRIPTS_DIR / "sglang_qwen3_model.sh"), "stop",
         "--port", str(base_port)],
        check=False,
    )
    print("[sglang] Stopped.")


# ── data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_compiled_law(law_path: Path) -> tuple[list[dict], list[dict]]:
    """Load a single compiled JSONL; split into (blacklist, whitelist) law-by-law."""
    blacklist: list[dict] = []
    whitelist: list[dict] = []
    for record in load_jsonl(law_path):
        compiled = record.get("compiled")
        if not compiled:
            continue
        ctype = compiled.get("type", "").upper()
        if ctype in ("OBEY", "DENY"):
            blacklist.append(compiled)
        elif ctype == "CHECK":
            whitelist.append(compiled)
    return blacklist, whitelist


def remove_policy(req: dict) -> dict:
    copy = dict(req)
    if "metadata" in copy:
        copy["metadata"] = {k: v for k, v in copy["metadata"].items() if k != "policy"}
    return copy


def extract_policy_type(policy: str) -> str:
    p = policy.upper()
    if "PERMIT"    in p: return "PERMIT"
    if "FORBID"    in p: return "FORBID"
    if "AMBIGUOUS" in p: return "AMBIGUOUS"
    return "UNKNOWN"


def load_system_prompt() -> str:
    return (PROJECT_DIR / "prompts" / "law_checker.txt").read_text()


# ── per-call result ───────────────────────────────────────────────────────────

@dataclass
class CallResult:
    checker: str
    phase: str          # "blacklist" | "whitelist"
    replica: int
    compliant: bool
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    raw_response: str


# ── LLM query ─────────────────────────────────────────────────────────────────

def query_checker(client: OpenAI, model: str, system_prompt: str,
                  checker: dict, req_clean: dict,
                  replica_idx: int, phase: str) -> CallResult:
    payload = {**checker, **req_clean}
    user_content = json.dumps(payload) + "\n/no_think"

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    elapsed = time.perf_counter() - t0

    raw = response.choices[0].message.content.strip()
    usage = response.usage
    in_tok  = usage.prompt_tokens     if usage else 0
    out_tok = usage.completion_tokens if usage else 0

    # Strip <think>...</think> tags if present
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        result = json.loads(text)
        compliant = bool(result.get("compliant", True))
    except json.JSONDecodeError:
        compliant = text.lower().startswith("true") or '"compliant": true' in text.lower()

    return CallResult(
        checker=checker.get("checker", ""),
        phase=phase,
        replica=replica_idx,
        compliant=compliant,
        input_tokens=in_tok,
        output_tokens=out_tok,
        elapsed_s=round(elapsed, 3),
        raw_response=text,
    )


def check_batch(replica_url: str, replica_idx: int, model: str, system_prompt: str,
                checkers: list[dict], req_clean: dict, phase: str) -> list[CallResult]:
    """Run a batch of checkers sequentially on one replica. Stop on first failure."""
    client = OpenAI(base_url=replica_url, api_key="dummy")
    results = []
    for checker in checkers:
        r = query_checker(client, model, system_prompt, checker, req_clean, replica_idx, phase)
        results.append(r)
        if not r.compliant:
            break
    return results


# ── blacklist phase ───────────────────────────────────────────────────────────

def run_blacklist(model: str, system_prompt: str, blacklist: list[dict],
                  req_clean: dict, urls: list[str]) -> tuple[bool, list[CallResult]]:
    if not blacklist:
        return True, []

    n = len(urls)
    chunks: list[list[dict]] = [[] for _ in range(n)]
    for i, checker in enumerate(blacklist):
        chunks[i % n].append(checker)

    all_results: list[CallResult] = []
    passed = True

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(check_batch, urls[i], i, model, system_prompt,
                        chunks[i], req_clean, "blacklist"): i
            for i in range(n) if chunks[i]
        }
        for fut in as_completed(futures):
            batch = fut.result()
            all_results.extend(batch)
            if not all(r.compliant for r in batch):
                passed = False
                for f in futures:
                    f.cancel()
                break

    return passed, all_results


# ── whitelist phase ───────────────────────────────────────────────────────────

def run_whitelist(model: str, system_prompt: str, whitelist: list[dict],
                  req_clean: dict, urls: list[str]) -> tuple[bool, list[CallResult]]:
    if not whitelist:
        return True, []

    n = len(urls)
    size = (len(whitelist) + n - 1) // n
    chunks = [whitelist[i * size: (i + 1) * size] for i in range(n)]

    all_results: list[CallResult] = []
    all_passed = True

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(check_batch, urls[i], i, model, system_prompt,
                        chunks[i], req_clean, "whitelist"): i
            for i in range(n) if chunks[i]
        }
        for fut in as_completed(futures):
            batch = fut.result()
            all_results.extend(batch)
            if not all(r.compliant for r in batch):
                all_passed = False
                for f in futures:
                    f.cancel()
                break

    return all_passed, all_results


# ── scoring ───────────────────────────────────────────────────────────────────

def score(permitted: bool, policy_type: str) -> int:
    if permitted     and policy_type in ("PERMIT",  "AMBIGUOUS"): return 1
    if not permitted and policy_type in ("FORBID",  "AMBIGUOUS"): return 1
    return 0


# ── logging ───────────────────────────────────────────────────────────────────

def save_results(log_dir: Path, run_id: str,
                 request_logs: list[dict], summary: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = log_dir / f"{run_id}_results.jsonl"
    summary_path = log_dir / f"{run_id}_summary.json"

    with results_path.open("w") as f:
        for entry in request_logs:
            f.write(json.dumps(entry) + "\n")

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[log] Results  → {results_path}")
    print(f"[log] Summary  → {summary_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Law checker pipelined (Qwen3 8x replicas)")
    parser.add_argument("--device",   default="1,2",          help="GPU device ids, comma-separated (default: 1,2)")
    parser.add_argument("--port",     type=int, default=30000, help="Base port for replicas (default: 30000)")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS,
                        help=f"Number of replicas (default: {DEFAULT_REPLICAS})")
    parser.add_argument("--max-total-tokens", type=int, default=4096,
                        help="Max KV cache tokens per replica (default: 4096); caps memory for co-located instances")
    parser.add_argument("--law",      default=str(DEFAULT_LAW), help="Compiled law JSONL file")
    parser.add_argument("--req",      default=str(DEFAULT_REQ), help="Request JSONL file")
    parser.add_argument("--max-req",  type=int, default=None,  help="Max requests to process")
    parser.add_argument("--log-dir",  default=str(DEFAULT_LOG_DIR), help="Output log directory")
    parser.add_argument("--no-start", action="store_true",    help="Skip starting sglang")
    parser.add_argument("--no-stop",  action="store_true",    help="Skip stopping sglang")
    args = parser.parse_args()

    base_port = args.port
    n_replicas = args.replicas
    urls = replica_urls(base_port, n_replicas)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)

    # 0. Start sglang
    if not args.no_start:
        start_sglang(args.device, base_port, n_replicas, args.max_total_tokens)

    try:
        resp = httpx.get(f"http://localhost:{base_port}/v1/models", timeout=10)
        resp.raise_for_status()
        model_id = resp.json()["data"][0]["id"]
        print(f"[info] Model: {model_id}")

        system_prompt = load_system_prompt()

        # 1. Load compiled laws (law by law from single file)
        blacklist, whitelist = load_compiled_law(Path(args.law))
        print(f"[info] Law file: {args.law}")
        print(f"[info] Blacklist: {len(blacklist)} laws, Whitelist: {len(whitelist)} laws")

        reqs = load_jsonl(Path(args.req))
        if args.max_req is not None:
            reqs = reqs[: args.max_req]
        print(f"[info] Requests to check: {len(reqs)}\n")

        total_score    = 0
        total_requests = 0
        total_in_tok   = 0
        total_out_tok  = 0
        total_elapsed  = 0.0
        request_logs: list[dict] = []

        run_t0 = time.perf_counter()

        for i, req in enumerate(reqs):
            policy      = req.get("metadata", {}).get("policy", "")
            policy_type = extract_policy_type(policy)

            if policy_type == "UNKNOWN":
                print(f"[{i+1}/{len(reqs)}] SKIP  unknown policy='{policy}'")
                continue

            req_clean = remove_policy(req)
            req_t0 = time.perf_counter()

            # 2. Blacklist phase
            bl_passed, bl_calls = run_blacklist(model_id, system_prompt, blacklist, req_clean, urls)

            if not bl_passed:
                permitted = False
                wl_calls: list[CallResult] = []
                phase_str = "BL=FAIL"
            else:
                # 3. Whitelist phase
                wl_passed, wl_calls = run_whitelist(model_id, system_prompt, whitelist, req_clean, urls)
                permitted = wl_passed
                phase_str = f"BL=pass WL={'pass' if wl_passed else 'FAIL'}"

            req_elapsed = time.perf_counter() - req_t0
            all_calls = bl_calls + wl_calls
            req_in_tok  = sum(c.input_tokens  for c in all_calls)
            req_out_tok = sum(c.output_tokens for c in all_calls)

            total_in_tok   += req_in_tok
            total_out_tok  += req_out_tok
            total_elapsed  += req_elapsed

            gained = score(permitted, policy_type)
            total_score    += gained
            total_requests += 1

            verdict = "PERMIT" if permitted else "DENY"
            print(
                f"[{i+1:3d}/{len(reqs)}] policy={policy_type:<9s} verdict={verdict:<6s} "
                f"[{phase_str}]  calls={len(all_calls):3d} "
                f"tok={req_in_tok}+{req_out_tok} "
                f"t={req_elapsed:.2f}s  +{gained}"
            )

            request_logs.append({
                "req_index": i + 1,
                "policy": policy,
                "policy_type": policy_type,
                "verdict": verdict,
                "score": gained,
                "elapsed_s": round(req_elapsed, 3),
                "input_tokens": req_in_tok,
                "output_tokens": req_out_tok,
                "calls": [asdict(c) for c in all_calls],
            })

        total_run_elapsed = time.perf_counter() - run_t0

        # 4. Final summary
        print()
        accuracy = (100 * total_score / total_requests) if total_requests else 0.0
        print(f"=== Score: {total_score}/{total_requests} ({accuracy:.1f}% accuracy) ===")
        print(f"    Total input tokens : {total_in_tok}")
        print(f"    Total output tokens: {total_out_tok}")
        print(f"    Total check time   : {total_elapsed:.2f}s (wall: {total_run_elapsed:.2f}s)")
        if total_requests:
            print(f"    Avg time/request   : {total_elapsed/total_requests:.2f}s")

        summary = {
            "run_id": run_id,
            "model": model_id,
            "law_file": args.law,
            "req_file": args.req,
            "base_port": base_port,
            "max_req": args.max_req,
            "blacklist_count": len(blacklist),
            "whitelist_count": len(whitelist),
            "total_requests": total_requests,
            "total_score": total_score,
            "accuracy_pct": round(accuracy, 2),
            "total_input_tokens": total_in_tok,
            "total_output_tokens": total_out_tok,
            "total_check_elapsed_s": round(total_elapsed, 3),
            "wall_elapsed_s": round(total_run_elapsed, 3),
        }

        save_results(log_dir, run_id, request_logs, summary)

    finally:
        # 5. Stop sglang
        if not args.no_stop:
            stop_sglang(base_port)


if __name__ == "__main__":
    main()
