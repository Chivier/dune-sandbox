import ast
import json
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), "req", "req.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "req_converted.jsonl")

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        req = ast.literal_eval(parts[0])
        policy = parts[1] if len(parts) > 1 else ""
        records.append({
            "prompt": req["user_input"],
            "metadata": {
                "sender_role": req["sender_role"],
                "consent_obtained": req["consent_obtained"],
                "authorization_obtained": req["authorization_obtained"],
                "policy": policy,
            }
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} records to {OUTPUT_FILE}")
