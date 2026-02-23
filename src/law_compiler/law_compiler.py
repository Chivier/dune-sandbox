#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

# Load .env manually (no python-dotenv dependency)
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in environment", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "law_compiler.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

MODEL = "x-ai/grok-4.1-fast"


def extract_json(content: str) -> dict:
    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block via regex
    m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"No valid JSON found in response: {content[:200]}")


def compile_law(law_text: str, metadata: dict) -> dict:
    user_message = json.dumps({
        "original_text": law_text,
        "metadata": metadata,
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()
    return extract_json(content)


def main():
    parser = argparse.ArgumentParser(description="Compile laws using LLM")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--rerun-errors", action="store_true",
                        help="If output exists, only rerun entries where compiled is null")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if rerunning errors
    existing = {}
    if args.rerun_errors and output_path.exists():
        for line in output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            section = rec.get("metadata", {}).get("section")
            if section:
                existing[section] = rec

    lines = input_path.read_text().splitlines()
    total = len([l for l in lines if l.strip()])
    print(f"Processing {total} laws from {input_path}")

    with output_path.open("w") as out_f:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            law_text = record["law"]
            metadata = record.get("metadata", {})
            section = metadata.get("section", "?")

            # If rerunning errors, skip successful entries
            if args.rerun_errors and section in existing:
                if existing[section].get("compiled") is not None:
                    out_f.write(json.dumps(existing[section]) + "\n")
                    print(f"[{i+1}/{total}] {section} ... SKIP")
                    continue

            print(f"[{i+1}/{total}] {section}", end=" ... ", flush=True)
            try:
                result = compile_law(law_text, metadata)
                output_record = {
                    "metadata": metadata,
                    "compiled": result,
                }
                out_f.write(json.dumps(output_record) + "\n")
                print(result.get("type", "?"))
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                output_record = {
                    "metadata": metadata,
                    "compiled": None,
                    "error": str(e),
                }
                out_f.write(json.dumps(output_record) + "\n")

    print(f"Done. Output written to {output_path}")


if __name__ == "__main__":
    main()
