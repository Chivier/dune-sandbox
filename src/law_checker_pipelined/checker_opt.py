#!/usr/bin/env python3
"""
law_checker_pipelined (Opt) — Optimized pipelined checker with batch inference + prefix caching.

Key optimizations vs checker.py:
  - 4 replicas (2/GPU) with large KV cache for aggressive prefix caching
  - Async batch inference: all blacklist/whitelist checks fire concurrently per request
  - Minimal output: true/false only (max_tokens=16), no reason field
  - Shared system prompt cached by sglang RadixAttention across all calls
"""

import argparse
import asyncio
import json
import re
import time
import datetime
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
from openai import AsyncOpenAI

PROJECT_DIR      = Path(__file__).parent.parent.parent
SCRIPTS_DIR      = PROJECT_DIR / "scripts"
DEFAULT_LAW      = PROJECT_DIR / "datasets" / "compiled" / "HIPAA.jsonl"
DEFAULT_REQ      = PROJECT_DIR / "datasets" / "req" / "req.jsonl"
DEFAULT_LOG_DIR  = PROJECT_DIR / "experiments" / "law_checker_pipelined"
DEFAULT_REPLICAS = 4


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
    phase: str
    replica: int
    compliant: bool
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    raw_response: str


# ── async LLM query ──────────────────────────────────────────────────────────

async def query_one(client: AsyncOpenAI, model: str, system_prompt: str,
                    checker: dict, req_clean: dict,
                    replica_idx: int, phase: str) -> CallResult:
    payload = {**checker, **req_clean}
    user_content = json.dumps(payload) + "\n/no_think"

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    elapsed = time.perf_counter() - t0

    raw = response.choices[0].message.content.strip()
    usage = response.usage
    in_tok  = usage.prompt_tokens     if usage else 0
    out_tok = usage.completion_tokens if usage else 0

    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    compliant = text.lower().startswith("true")

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


# ── batch blacklist / whitelist ───────────────────────────────────────────────

async def run_blacklist(clients: list[AsyncOpenAI], model: str, system_prompt: str,
                        blacklist: list[dict], req_clean: dict) -> tuple[bool, list[CallResult]]:
    if not blacklist:
        return True, []

    n = len(clients)
    tasks = [
        query_one(clients[i % n], model, system_prompt, checker, req_clean, i % n, "blacklist")
        for i, checker in enumerate(blacklist)
    ]
    results = await asyncio.gather(*tasks)
    passed = all(r.compliant for r in results)
    return passed, list(results)


async def run_whitelist(clients: list[AsyncOpenAI], model: str, system_prompt: str,
                        whitelist: list[dict], req_clean: dict) -> tuple[bool, list[CallResult]]:
    if not whitelist:
        return True, []

    n = len(clients)
    tasks = [
        query_one(clients[i % n], model, system_prompt, checker, req_clean, i % n, "whitelist")
        for i, checker in enumerate(whitelist)
    ]
    results = await asyncio.gather(*tasks)
    passed = all(r.compliant for r in results)
    return passed, list(results)


# ── scoring ───────────────────────────────────────────────────────────────────

def score(permitted: bool, policy_type: str) -> int:
    if permitted     and policy_type in ("PERMIT",  "AMBIGUOUS"): return 1
    if not permitted and policy_type in ("FORBID",  "AMBIGUOUS"): return 1
    return 0


# ── logging ───────────────────────────────────────────────────────────────────

def save_results(log_dir: Path, run_id: str,
                 request_logs: list[dict], summary: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = log_dir / f"{run_id}_opt_results.jsonl"
    summary_path = log_dir / f"{run_id}_opt_summary.json"

    with results_path.open("w") as f:
        for entry in request_logs:
            f.write(json.dumps(entry) + "\n")

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[log] Results  → {results_path}")
    print(f"[log] Summary  → {summary_path}")


# ── async main ────────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    base_port = args.port
    n_replicas = args.replicas
    urls = replica_urls(base_port, n_replicas)

    resp = httpx.get(f"http://localhost:{base_port}/v1/models", timeout=10)
    resp.raise_for_status()
    model_id = resp.json()["data"][0]["id"]
    print(f"[info] Model: {model_id}")

    system_prompt = load_system_prompt()
    clients = [AsyncOpenAI(base_url=url, api_key="dummy") for url in urls]

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
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)

    run_t0 = time.perf_counter()

    for i, req in enumerate(reqs):
        policy      = req.get("metadata", {}).get("policy", "")
        policy_type = extract_policy_type(policy)

        if policy_type == "UNKNOWN":
            print(f"[{i+1}/{len(reqs)}] SKIP  unknown policy='{policy}'")
            continue

        req_clean = remove_policy(req)
        req_t0 = time.perf_counter()

        # Blacklist: fire all concurrently
        bl_passed, bl_calls = await run_blacklist(clients, model_id, system_prompt, blacklist, req_clean)

        if not bl_passed:
            permitted = False
            wl_calls: list[CallResult] = []
            phase_str = "BL=FAIL"
        else:
            # Whitelist: fire all concurrently
            wl_passed, wl_calls = await run_whitelist(clients, model_id, system_prompt, whitelist, req_clean)
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
        "variant": "opt_pipelined",
        "model": model_id,
        "law_file": args.law,
        "req_file": args.req,
        "base_port": base_port,
        "n_replicas": n_replicas,
        "max_total_tokens": args.max_total_tokens,
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Law checker Opt_Pipelined (batch + prefix cache)")
    parser.add_argument("--device",   default="1,2",          help="GPU device ids (default: 1,2)")
    parser.add_argument("--port",     type=int, default=40000, help="Base port (default: 40000)")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS,
                        help=f"Number of replicas (default: {DEFAULT_REPLICAS})")
    parser.add_argument("--max-total-tokens", type=int, default=65536,
                        help="KV cache tokens per replica (default: 65536, aggressive for prefix caching)")
    parser.add_argument("--law",      default=str(DEFAULT_LAW), help="Compiled law JSONL file")
    parser.add_argument("--req",      default=str(DEFAULT_REQ), help="Request JSONL file")
    parser.add_argument("--max-req",  type=int, default=None,  help="Max requests to process")
    parser.add_argument("--log-dir",  default=str(DEFAULT_LOG_DIR), help="Output log directory")
    parser.add_argument("--no-start", action="store_true",    help="Skip starting sglang")
    parser.add_argument("--no-stop",  action="store_true",    help="Skip stopping sglang")
    args = parser.parse_args()

    if not args.no_start:
        start_sglang(args.device, args.port, args.replicas, args.max_total_tokens)

    try:
        asyncio.run(async_main(args))
    finally:
        if not args.no_stop:
            stop_sglang(args.port)


if __name__ == "__main__":
    main()
