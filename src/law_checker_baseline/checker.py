#!/usr/bin/env python3
"""
law_checker_baseline — Baseline sandbox compliance checker.

This script can run in two modes:
  - OpenAI API (default): use GPT-5.2 (or another OpenAI model) over the network.
  - Local sglang: start a local model server and query it via the OpenAI-compatible API.

Examples:
  # OpenAI (GPT-5.2)
  OPENAI_API_KEY=... python src/law_checker_baseline/checker.py

  # OpenAI (custom model)
  OPENAI_API_KEY=... python src/law_checker_baseline/checker.py --model gpt-5.2-mini

  # Local sglang (keeps previous behavior)
  python src/law_checker_baseline/checker.py --backend sglang --device 3 --port 30000

Steps:
  1. (sglang only) Start sglang model server (1 replica)
  2. Split law JSONL into N token-budget parts (each fits within context window)
  3. For each request in req.jsonl:
       - Strip "policy" from metadata
       - Send N prompts (one per law part) sequentially; parse true/false
       - Overall compliance = AND of all part results
  4. Score against ground-truth policy label:
       - LLM=true  & policy∈{PERMIT, AMBIGUOUS}  → +1
       - LLM=false & policy∈{FORBID, AMBIGUOUS}  → +1
       - otherwise                                → 0
  5. Print final score / number of requests; save results + logs
  6. Stop sglang server
"""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI

PROJECT_DIR  = Path(__file__).parent.parent.parent
SCRIPTS_DIR  = PROJECT_DIR / "scripts"
RESULTS_DIR  = PROJECT_DIR / "experiments" / "law_checker_baseline"
DEFAULT_LAW  = PROJECT_DIR / "datasets" / "law" / "HIPAA.jsonl"
DEFAULT_REQ  = PROJECT_DIR / "datasets" / "req" / "req.jsonl"
DEFAULT_OPENAI_MODEL = "gpt-5.2"


# ── misc helpers ──────────────────────────────────────────────────────────────

def safe_tag(value: str) -> str:
    """Make a string safe to embed in filenames/run-ids."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


# ── sglang lifecycle ──────────────────────────────────────────────────────────

def start_sglang(device: str, port: int) -> None:
    print(f"[sglang] Starting 70b model on GPU device={device}, port={port} ...")
    subprocess.run(
        [str(SCRIPTS_DIR / "sglang_70b_model.sh"), "start",
         "--device", device, "--port", str(port)],
        check=True,
    )
    print("[sglang] Server ready.")


def stop_sglang(port: int) -> None:
    print("[sglang] Stopping server ...")
    subprocess.run(
        [str(SCRIPTS_DIR / "sglang_70b_model.sh"), "stop",
         "--port", str(port)],
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


def split_laws(laws: list[dict], max_chars_per_part: int = 6000) -> list[str]:
    """Split law records into parts that each fit within max_chars_per_part.

    Individual laws longer than the limit are truncated with a notice.
    """
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for item in laws:
        text = item["law"]
        # Truncate oversized individual laws
        if len(text) > max_chars_per_part:
            text = text[:max_chars_per_part] + "\n[... truncated ...]"
        # If adding this law would exceed the budget, flush current part first
        if current and current_len + len(text) + 2 > max_chars_per_part:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(text)
        current_len += len(text) + 2  # +2 for "\n\n" separator

    if current:
        parts.append("\n\n".join(current))

    return parts


def remove_policy(req: dict) -> dict:
    """Return a shallow copy of req with 'policy' removed from metadata."""
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


# ── LLM query ─────────────────────────────────────────────────────────────────

def query_part(client: OpenAI, model: str, law_part: str, req_json: str, api: str) -> dict:
    """Query the LLM for one law part. Returns dict with result + metrics."""
    prompt = (
        f"Check if this user obey all laws in:\n{law_part}\n\n"
        f"User: {req_json}\n\n"
        f"Only return true/false."
    )
    t0 = time.perf_counter()
    if api == "responses":
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # OpenAI Responses API enforces a minimum (currently 16).
            max_output_tokens=16,
        )
        raw_text = (response.output_text or "").strip()
        input_tokens  = response.usage.input_tokens  if response.usage else None
        output_tokens = response.usage.output_tokens if response.usage else None
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw_text = (response.choices[0].message.content or "").strip()
        input_tokens  = response.usage.prompt_tokens     if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None
    elapsed = time.perf_counter() - t0

    answer = raw_text.lower()

    return {
        "compliant":      answer.startswith("true"),
        "raw_answer":     raw_text,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "time_s":         round(elapsed, 3),
    }


# ── scoring ───────────────────────────────────────────────────────────────────

def calc_score(llm_result: bool, policy_type: str) -> int:
    if llm_result     and policy_type in ("PERMIT",  "AMBIGUOUS"): return 1
    if not llm_result and policy_type in ("FORBID",  "AMBIGUOUS"): return 1
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Law checker baseline")
    parser.add_argument("--backend",  choices=("openai", "sglang"), default="openai",
                        help="Which backend to use: openai (default) or sglang (local server)")
    parser.add_argument("--model",    default=None,
                        help=f"Model name. Default: {DEFAULT_OPENAI_MODEL} for openai; auto-detected for sglang.")
    parser.add_argument("--api",      choices=("responses", "chat"), default=None,
                        help="Which OpenAI-style API to use. Default: responses for openai, chat for sglang.")
    parser.add_argument("--api-key",  default=None,
                        help="API key for the OpenAI client. Defaults to OPENAI_API_KEY for openai; dummy for sglang.")
    parser.add_argument("--base-url", default=None,
                        help="Override base URL for OpenAI-compatible endpoints. "
                             "For sglang default is http://localhost:<port>/v1.")
    parser.add_argument("--device",   default="3",
                        help="GPU device id (sglang only; default: 3)")
    parser.add_argument("--port",     type=int, default=30000,
                        help="sglang server port (sglang only; default: 30000)")
    parser.add_argument("--law",      default=str(DEFAULT_LAW), help="Law JSONL file")
    parser.add_argument("--req",      default=str(DEFAULT_REQ), help="Request JSONL file")
    parser.add_argument("--max-req",  type=int, default=None,   help="Max number of requests to process")
    parser.add_argument("--no-start", action="store_true",      help="Skip starting sglang (sglang only)")
    parser.add_argument("--no-stop",  action="store_true",      help="Skip stopping sglang (sglang only)")
    args = parser.parse_args()

    backend = args.backend
    api = args.api or ("responses" if backend == "openai" else "chat")
    if backend == "sglang" and api == "responses":
        raise SystemExit("sglang backend does not support the Responses API. Use --api chat (default).")

    # 1. Start sglang (optional)
    if backend == "sglang" and not args.no_start:
        start_sglang(args.device, args.port)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    law_tag = Path(args.law).stem  # e.g. "HIPAA" or "GDPR"

    try:
        # Resolve model + client
        if backend == "sglang":
            base_url = args.base_url or f"http://localhost:{args.port}/v1"
            client = OpenAI(base_url=base_url, api_key=args.api_key or "dummy")

            if args.model:
                model_id = args.model
            else:
                # Resolve model name from running server
                resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=10)
                resp.raise_for_status()
                model_id = resp.json()["data"][0]["id"]
        else:
            if not args.api_key and not os.getenv("OPENAI_API_KEY"):
                raise SystemExit(
                    "OPENAI_API_KEY is not set. Set it in your environment or pass --api-key."
                )

            client_kwargs: dict = {}
            if args.base_url:
                client_kwargs["base_url"] = args.base_url
            if args.api_key:
                client_kwargs["api_key"] = args.api_key
            client = OpenAI(**client_kwargs)
            model_id = args.model or DEFAULT_OPENAI_MODEL

        resolved_base_url = str(client.base_url)
        run_id = (
            f"{run_ts}_{law_tag}_gpu{safe_tag(args.device)}"
            if backend == "sglang"
            else f"{run_ts}_{law_tag}_{safe_tag(model_id)}"
        )

        print(f"[info] Backend: {backend} (api={api})")
        print(f"[info] Base URL: {resolved_base_url}")
        print(f"[info] Model:   {model_id}")
        print(f"[info] Run ID:  {run_id}")

        # 2. Load & split laws
        laws  = load_jsonl(Path(args.law))
        parts = split_laws(laws)
        part_chars = [len(p) for p in parts]
        print(f"[info] {len(laws)} laws → {len(parts)} parts (chars: min={min(part_chars)} max={max(part_chars)})")

        # Load requests (apply --max-req cap)
        all_reqs = load_jsonl(Path(args.req))
        reqs     = all_reqs[:args.max_req] if args.max_req else all_reqs
        print(f"[info] {len(reqs)} requests to check (of {len(all_reqs)} total)")
        print()

        # Prepare output paths
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / f"{run_id}_results.jsonl"
        log_path     = RESULTS_DIR / f"{run_id}_run.log"
        log_lines    = []

        def log(msg: str) -> None:
            print(msg)
            log_lines.append(msg)

        # 3–4. Process each request
        total_score    = 0
        total_requests = 0
        agg_in_tok     = 0
        agg_out_tok    = 0
        agg_time       = 0.0

        with results_path.open("w") as rf:
            for i, req in enumerate(reqs):
                policy      = req.get("metadata", {}).get("policy", "")
                policy_type = extract_policy_type(policy)

                if policy_type == "UNKNOWN":
                    log(f"[{i+1:3d}/{len(reqs)}] SKIP  unknown policy='{policy}'")
                    continue

                req_clean = remove_policy(req)
                req_json  = json.dumps(req_clean)

                # Send 4 prompts one by one, collect metrics
                part_results = []
                req_in_tok   = 0
                req_out_tok  = 0
                req_time     = 0.0

                for p_idx, part_text in enumerate(parts):
                    qr = query_part(client, model_id, part_text, req_json, api=api)
                    part_results.append(qr)
                    req_in_tok  += qr["input_tokens"]  or 0
                    req_out_tok += qr["output_tokens"] or 0
                    req_time    += qr["time_s"]

                agg_in_tok  += req_in_tok
                agg_out_tok += req_out_tok
                agg_time    += req_time

                overall = all(qr["compliant"] for qr in part_results)
                gained  = calc_score(overall, policy_type)
                total_score    += gained
                total_requests += 1

                parts_str = " ".join(
                    f"P{j+1}={'T' if qr['compliant'] else 'F'}"
                    for j, qr in enumerate(part_results)
                )
                log(
                    f"[{i+1:3d}/{len(reqs)}] {policy_type:<9s} "
                    f"overall={'true ' if overall else 'false'} "
                    f"[{parts_str}]  +{gained}  "
                    f"tok={req_in_tok}+{req_out_tok}  t={req_time:.2f}s"
                )

                # Write per-request result record
                record = {
                    "request_idx":   i,
                    "policy":        policy,
                    "policy_type":   policy_type,
                    "prompt":        req.get("prompt", ""),
                    "overall":       overall,
                    "score":         gained,
                    "parts": [
                        {
                            "part":         j + 1,
                            "compliant":    qr["compliant"],
                            "raw_answer":   qr["raw_answer"],
                            "input_tokens": qr["input_tokens"],
                            "output_tokens":qr["output_tokens"],
                            "time_s":       qr["time_s"],
                        }
                        for j, qr in enumerate(part_results)
                    ],
                    "total_input_tokens":  req_in_tok,
                    "total_output_tokens": req_out_tok,
                    "total_time_s":        round(req_time, 3),
                }
                rf.write(json.dumps(record) + "\n")

        # 5. Summary
        accuracy = 100 * total_score / total_requests if total_requests else 0.0
        avg_time = agg_time / total_requests if total_requests else 0.0

        summary_lines = [
            "",
            "=" * 60,
            f"  Score:         {total_score}/{total_requests}  ({accuracy:.1f}% accuracy)",
            f"  Total tokens:  {agg_in_tok} input + {agg_out_tok} output",
            f"  Total time:    {agg_time:.1f}s  (avg {avg_time:.2f}s/req)",
            f"  Results:       {results_path}",
            f"  Log:           {log_path}",
            "=" * 60,
        ]
        for line in summary_lines:
            log(line)

        # Write summary to log
        summary_path = RESULTS_DIR / f"{run_id}_summary.json"
        summary_path.write_text(json.dumps({
            "run_id":         run_id,
            "backend":        backend,
            "api":            api,
            "model":          model_id,
            "base_url":       resolved_base_url,
            "law_file":       str(args.law),
            "req_file":       str(args.req),
            "max_req":        args.max_req,
            "num_requests":   total_requests,
            "score":          total_score,
            "accuracy":       round(accuracy, 4),
            "total_input_tokens":  agg_in_tok,
            "total_output_tokens": agg_out_tok,
            "total_time_s":   round(agg_time, 3),
            "avg_time_s":     round(avg_time, 3),
        }, indent=2) + "\n")

        log_path.write_text("\n".join(log_lines) + "\n")

    finally:
        # 6. Stop sglang
        if backend == "sglang" and not args.no_stop:
            stop_sglang(args.port)


if __name__ == "__main__":
    main()
