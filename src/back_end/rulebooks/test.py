import re
import json
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, "HIPAA_extracted.txt"), "r", encoding="utf-8") as f:
    lines = f.readlines()

d = {}

i = 0
for line in lines:
    # print(f"{i}: {line.strip()}")
    line = line.strip()
    assert len(line.split('\t')) == 2
    a, b = line.split('\t')
    # print(f"ID: {a}, Text: {b}")

    m = re.match(r'[0-9\.]+.[0-9\.]+', a)
    if m is None:
        # print(f"ID {a} Text {b} does not match the expected format.")
        continue
    
    if m.group(0) not in d:
        d[m.group(0)] = []
    d[m.group(0)].append(b)

with open(os.path.join(BASE_DIR, "HIPAA_extracted.json"), "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4)
