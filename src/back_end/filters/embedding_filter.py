
def embed_filter(rulebook: str, previous_context: str, tool_call: str) -> bool:
    # A trivial filter that allows all tool calls
    return True