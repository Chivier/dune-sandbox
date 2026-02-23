# a system prompt
# a set of tools that user could specify
# a tool calling chain
# a filter that could filter out compliance violations
# a final response generator
# a metrics collector
# a logging module that put all of the important checkpoints into logs

# switch of reading mode and interactive mode

import argparse
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional
from openai import OpenAI
import os
import requests
from filters.trivial_filter import trivial_filter
from filters.context_filter import context_filter
from filters.strict_filter import strict_filter
import time
import json
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
_RULEBOOK_CACHE = {}

# arg parse
@dataclass
class parsed_args:
    mode: str = "read"  # read or interactive
    file: str = "datasets/200_list.txt"  # input file for read mode
    out: str = "out/output.json"  # output file for read mode
    filter: str = "trivial"  # filter type: trivial, context, strict
    quick: bool = False  # quick mode for testing
    model: str = None  # model name, if None, use local model

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
log_dir = os.path.join(REPO_ROOT, "log")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

think_template = "</think>"

def _format_tool_call_event(http_req: str) -> dict:
    text = (http_req or "").strip()
    if not text:
        return {"call": "", "method": "", "url": ""}

    first_line = text.splitlines()[0].strip()
    parts = first_line.split(None, 1)
    method = parts[0].upper() if parts else ""
    url = parts[1].strip() if len(parts) > 1 else ""

    payload_len = None
    if method == "POST":
        lines = text.splitlines()
        if len(lines) > 1:
            payload = "\n".join(lines[1:]).strip()
            payload_len = len(payload) if payload else 0

    call_display = f"{method} {url}".strip()
    if payload_len is not None:
        call_display = f"{call_display} (payload {payload_len} chars)"

    return {
        "call": call_display,
        "method": method,
        "url": url,
        "payload_len": payload_len,
    }

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _load_compiled_rulebook_text(path: str) -> str:
    """
    Load a dune-sandbox compiled rulebook from JSONL and flatten it into
    line-oriented text suitable for LLM prompting.

    Expected JSONL shape per line:
      {"metadata": {...}, "compiled": {"type": "...", "checker": "..."}}.
    """
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = (raw or "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                # If a line isn't valid JSON, keep it verbatim.
                lines.append(raw)
                continue

            meta = obj.get("metadata") or {}
            compiled = obj.get("compiled") or {}

            source = str(meta.get("source") or "").strip()
            section = str(meta.get("section") or "").strip()
            article = str(meta.get("article") or "").strip()
            title = str(meta.get("title") or "").strip()

            ctype = str(compiled.get("type") or "").strip()
            checker = str(compiled.get("checker") or "").strip()

            header = ""
            if article:
                header = f"{source} {article}".strip()
                if title:
                    header = f"{header}: {title}"
            elif section:
                header = f"{source} {section}".strip()
            else:
                header = source

            body = " ".join(part for part in (ctype, checker) if part)
            if header and body:
                lines.append(f"{header}\t{body}")
            elif header:
                lines.append(header)
            elif body:
                lines.append(body)

    return "\n".join(lines).strip()

def _rulebook_path_for(rulebook_id: str) -> str:
    rid = (rulebook_id or "").strip().lower()
    if rid in ("hipaa", ""):
        # Use the *compiled* dataset representation as the rulebook to keep the
        # Rotunda runtime fully dependent on dune-sandbox artifacts.
        return os.path.join(DATASETS_DIR, "compiled", "HIPAA.jsonl")
    raise ValueError(f"Unknown builtin rulebook_id: {rulebook_id}")

def _load_gdpr_rulebook_text() -> str:
    # Use the compiled GDPR representation from the dune-sandbox dataset.
    return _load_compiled_rulebook_text(os.path.join(DATASETS_DIR, "compiled", "GDPR.jsonl"))

def load_rulebook_text(rulebook_id: str = "hipaa", rulebook_text: str = "") -> str:
    """
    Returns the rulebook text used by compliance filters.

    - hipaa: loads the compiled HIPAA rules from `datasets/compiled/HIPAA.jsonl`.
    - gdpr: loads the compiled GDPR rules from `datasets/compiled/GDPR.jsonl`.
    - custom: uses the user-provided `rulebook_text`.
    """
    rid = (rulebook_id or "hipaa").strip().lower()
    if rid == "custom":
        if rulebook_text and str(rulebook_text).strip():
            return str(rulebook_text)
        raise ValueError("custom rulebook selected but rulebook_text is empty")
    if rid == "gdpr":
        cached = _RULEBOOK_CACHE.get("gdpr")
        if cached is None:
            cached = _load_gdpr_rulebook_text()
            _RULEBOOK_CACHE["gdpr"] = cached
        return cached

    if rid not in ("hipaa", ""):
        raise ValueError(f"unknown rulebook_id: {rulebook_id}")

    # Default: HIPAA.
    cached = _RULEBOOK_CACHE.get("hipaa")
    if cached is None:
        cached = _load_compiled_rulebook_text(_rulebook_path_for("hipaa"))
        _RULEBOOK_CACHE["hipaa"] = cached
    return cached

# parsing system prompt
# returns a template that could fill with question
def read_system_prompt(file_path: str, base_url: str) -> "function[str, str]":
    resolved = file_path
    if not os.path.isabs(resolved):
        resolved = os.path.join(REPO_ROOT, file_path)
    template = _read_text(resolved)

    funcs = _read_text(os.path.join(PROMPTS_DIR, "tool_spec.json"))

    def temp(question: str):

        return template.format(api_base=base_url, functions=funcs, question=question)

    return temp

def tool_calling(
    http_req: str,
    history: list,
    rulebook: str,
    timeline: list = None,
    on_tool_call: Optional[Callable[[dict], None]] = None,
) -> str:
    # parse the response to see if there is any tool calling
    # if yes, call the tool and get the result
    # return the result

    # sometimes the rule book exceeds the token limit, do be careful


    if parsed_args.filter == 'trivial':
        result, minilog = trivial_filter(rulebook, history, http_req)
    elif parsed_args.filter == 'context':
        result, minilog = context_filter(rulebook, history, http_req)
    elif parsed_args.filter == 'strict':
        result, minilog = strict_filter(rulebook, history, http_req)
    else:
        raise ValueError("Invalid filter type specified.")
    if timeline is not None:
        timeline.append(
            {'time': time.time(), 'role': 'context filter', 'content': f"Result: {result}, Minilog: {minilog}"})

    if not result:
        print(parsed_args.filter)
        raise PermissionError(next((m.get("response") for m in reversed(minilog or []) if isinstance(m, dict) and m.get("response")), "Tool call blocked by filter."))

    if on_tool_call is not None:
        try:
            on_tool_call(_format_tool_call_event(http_req))
        except Exception:
            # Don't fail the agent run if the trace stream fails.
            pass

    if http_req.startswith("GET"):
        url = http_req[4:].strip()
        resp = requests.get(url)
        return resp.text
    elif http_req.startswith("POST"):
        print("POST tool calling detected.", http_req)
        url, playload = http_req[5:].strip().split('\n', 1)
        # resp = requests.post(url, data=playload)
        resp = requests.post(url, data=playload, headers={"Content-Type": "application/json"})
        return resp.text
        # raise NotImplementedError("POST tool calling is not implemented yet.")

    raise ValueError("No valid tool calling found in the response.")

def one_liner(
    inputer: str,
    rulebook_id: str = "hipaa",
    rulebook_text: str = "",
    on_tool_call: Optional[Callable[[dict], None]] = None,
):
    logger.info("Entering file read mode.")
    result = []
    rulebook = load_rulebook_text(rulebook_id=rulebook_id, rulebook_text=rulebook_text)

    if parsed_args.model is None:
        client = OpenAI(
            base_url="http://localhost:30000/v1",
            api_key=os.getenv("OPENAI_API_KEY", "testkey")
        )
        caller = lambda history: client.chat.completions.create(
            messages=history,
            model="sglang-gpt-oss-20b"
        )
    else:
        assert os.getenv("OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY environment variable for external model."
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        caller = lambda history: client.chat.completions.create(
            messages=history,
            model=parsed_args.model
        )
    
    question_template = read_system_prompt(
        "prompts/sys_text_new.txt", "http://localhost:9090/fhir/")

    line = inputer


    case_dict = {}

    lits = line.strip().split('\t')

    user_input = lits[0]

    case_dict["input"] = user_input
    case_dict["args"] = lits[1:]
    case_dict["rulebook_id"] = rulebook_id
    case_dict["timeline"] = [
        {'time': time.time(), 'role': 'user', 'content': user_input}]

    logger.info(f"Processing input: {user_input}")
    
    history = []

    init_time = time.time()
    print(init_time)

    logger.info("Received user input.")
    logger.info(f"User input: {user_input}")
    # Process user input here

    history.append({"role": "system", "content": question_template(user_input)})

    resp = caller(history)

    case_dict["timeline"].append(
        {'time': time.time(), 
            'role': 'system', 
            'content': question_template(user_input)})


    # print(question_template(user_input))

    assistant_text = (resp.choices[0].message.content or "").strip()

    ind = assistant_text.find(think_template)
    if ind != -1:
        assistant_text = assistant_text[ind + len(think_template):].strip()

    print(assistant_text)
    history.append({"role": "assistant", "content": assistant_text})

    logger.info("OpenAI responded successfully.")

    print(f"Processed: {assistant_text}")
    print()

    tool_call_step = 0
    while True:
        if assistant_text.startswith("POST") or assistant_text.startswith("GET"):
            
            try:
                tool_call_step += 1
                tool_call_callback = None
                if on_tool_call is not None:
                    def tool_call_callback(event):  # noqa: E306 - intentional nesting for closure
                        payload = dict(event or {})
                        payload.setdefault("step", tool_call_step)
                        on_tool_call(payload)
                case_dict["timeline"].append({'time': time.time(), 'role': 'tool calling', 'content': assistant_text})
                tool_response = tool_calling(
                    assistant_text,
                    history,
                    rulebook,
                    case_dict['timeline'],
                    on_tool_call=tool_call_callback,
                )
                case_dict["timeline"].append({'time': time.time(), 'role': 'tool response', 'content': tool_response})
                print(time.time() - init_time)
            except Exception as e:
                tool_response = str(e) if isinstance(e, PermissionError) else f"Tool calling failed: {e}"
                logger.error(tool_response)
                case_dict["timeline"].append({"role": "final response", "content": tool_response})
                print(time.time() - init_time)
                break

            history.append({"role": "user", "content": tool_response})
            logger.info(tool_response[:1000])  # log first 1000 chars

            resp = caller(history)

            case_dict["timeline"].append({'time': time.time(), 'role': 'system', 'content': (resp.choices[0].message.content or "").strip()})
            assistant_text = (resp.choices[0].message.content or "").strip()
            ind = assistant_text.find(think_template)
            if ind != -1:
                assistant_text = assistant_text[ind + len(think_template):].strip()
            history.append({"role": "assistant", "content": assistant_text})
            # logger.info(history)

        elif assistant_text.startswith("FINISH"):
            logger.info("Final response generated.")
            logger.info(f"Final Response: {assistant_text[len('FINISH'):].strip()}")
            case_dict["timeline"].append({"role": "final response", "content": assistant_text})
            break

        else:
            logger.info("No tool calling detected.")
            logger.info(f"Final Response: {assistant_text}")
            case_dict["timeline"].append({"role": "final response", "content": assistant_text})
            break

    print(time.time() - init_time)
    case_dict['timeline'].append({'time': time.time(), 'role': 'eval', 'content': 'end of processing'})
    result.append(case_dict)
    logger.info("Processed user input.")
    
    return case_dict
