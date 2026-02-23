from typing import Tuple
from openai import OpenAI
import os
import time
import json
import threading


rulebook_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "compiled", "HIPAA.jsonl")
)
num_leaves = 10
overlap_size = 0.2

think_template = "</think>"

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="testkey"
)

def interleaver(text: str) -> list[str]:
    leaves = []

    total_length = len(text)
    leaves_count = num_leaves - overlap_size * (num_leaves - 1)
    leaf_size = int(total_length // leaves_count)

    for i in range(num_leaves):
        start = int(i * (leaf_size * (1 - overlap_size)))
        end = start + leaf_size
        if i == num_leaves - 1:
            end = total_length
        print(f"Leaf {i+1}: start={start}, end={end}")
        leaves.append(text[start:end])

    return leaves

def context_filter(rulebook: str, previous_context: str, tool_call: str) -> Tuple[bool, dict]:

    starting_time = time.time()
    intermeasure_time = time.time()

    _cxt = list(filter(lambda x: x['role'] != "system", previous_context))

    with open(rulebook_path, 'r') as f:
        rulebook = f.read()

    minilog = {}

    prompt = """
        - Rulebook: {rulebook}
        - Previous Context: {previous_context}
        - Tool Call: {tool_call}
        You are a context filter that decides whether to allow or block a tool call made by an AI agent.
        The decision should be based on the following inputs, 
        ONLY say "ALLOW" or "FORBIDDEN BY <certain law index>", 
        like "FORBIDDEN BY HIPAA Section 164.xxx".
    """

    halt = False
    def task(_leaf, __cxt, _tool_call, log):
        nonlocal halt

        log['start_time'] = time.time()

        formatted = prompt.format(
            rulebook=_leaf,
            previous_context=__cxt,
            tool_call=_tool_call
        )

        response = client.chat.completions.create(
            model="sglang-gpt-oss-20b",
            messages=[
            {"role": "user", "content": formatted}
            ],
            stream=True,
        )
        
        assistant_text_parts = []

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                assistant_text_parts.append(delta.content)
                # print(delta.content, end='', flush=True)
                if halt:
                    log['msg'] = "Aborted due to halt signal."
                    log['end_time'] = time.time()
                    response.close()
                    return
        log['msg'] = "Completed without halt."

        assistant_text = "".join(assistant_text_parts).strip()

        ind = assistant_text.find(think_template)
        if ind != -1:
            print("Found think template, extracting decision...")
            assistant_text = assistant_text[ind + len(think_template):].strip()

        # if assistant_text.lower() != "allow":
        #     halt = True
        #     log['msg'] = f"Tool call forbidden by rule: {assistant_text}"
        log['msg'] = f"Tool call decision: {assistant_text}"
        log['end_time'] = time.time()

        return assistant_text, response

    ts = []

    for i, leaf in enumerate(interleaver(rulebook)):

        minilog[f"leaf_{i}"] = {}

        # assistant_text, response = task(leaf, _cxt, tool_call)
        t = threading.Thread(target=task, args=(leaf, _cxt, tool_call, minilog[f"leaf_{i}"]))
        ts.append(t)
    
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    
    return halt, minilog

if __name__ == "__main__":

    test_context_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_cont.json"))
    if os.path.exists(test_context_path):
        with open(test_context_path, "r", encoding="utf-8") as f:
            test_context = json.load(f)
    else:
        # Fallback sample context so this module can be run standalone.
        test_context = [{"role": "user", "content": "The user asked about HIPAA compliance dates."}]

    res = context_filter(
        rulebook="",
        previous_context=test_context,
        tool_call="Fetch HIPAA compliance dates from the rulebook."
    )
    print(res)
