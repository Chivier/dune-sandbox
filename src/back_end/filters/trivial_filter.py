# for any filter, you have only the rulebook, the previous context
# and the current tool call the model is going to make.
# You need to decide whether to allow the tool call or not.

def trivial_filter(rulebook: str, previous_context: str, tool_call: str) -> bool:
    # A trivial filter that allows all tool calls
    return True, None