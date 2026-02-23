from openai import OpenAI

cot_instruction_template = {
    "instruction": "Please assess the applicability of the HIPAA Privacy Rule to the case through the following steps: Step 1: Annotate the message characteristics [Sender, Sender Role, Recipient, Recipient Role, About, About Role, Type] about the flow of private information in the case as a list. Step 2: Determine whether the HIPAA Privacy Rule is applicable to the case. Read the case: {case} Step 1:",
    "response": "Step 1: {step1} Step 2: {step2}"
}

def goldcoin_filter(rulebook: str, previous_context: str, tool_call: str) -> tuple[bool, str]:
    prompt = cot_instruction_template["instruction"].format(case=previous_context)
    client = OpenAI(
        base_url="http://localhost:30000/v1",
        api_key="testkey"
    )
    response = client.chat.completions.create(
        model="sglang-gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
