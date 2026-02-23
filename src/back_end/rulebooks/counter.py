# token counter
# model to use: qwen3-0.6b, 1.7b, 4b; ROPE methods testing

import transformers
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(BASE_DIR, "HIPAA_extracted.txt")
model_name = 'Qwen/Qwen3-0.6B'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

with open(filepath, 'r') as f:
    lines = f.read()

tokens = tokenizer(lines)
input_ids = tokens['input_ids']
sum_token_count = len(input_ids)

print(f"Total tokens: {sum_token_count}")
