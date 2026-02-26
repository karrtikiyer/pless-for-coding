def is_instruct_model(model_id: str) -> bool:
    """Check if a model ID indicates an instruct/chat model."""
    lower = model_id.lower()
    return "instruct" in lower or "chat" in lower


def format_prompt_base(task: dict) -> tuple[str, str]:
    """Format a completion-style prompt for base models.

    HumanEval's `prompt` field already contains the function signature + docstring,
    so we pass it directly. The model completes the function body.

    Returns (prompt_text, code_prefix) where code_prefix is the prompt itself
    so that prefix + generated_body = complete executable code.
    """
    prompt = task["prompt"]
    return prompt, prompt


def format_prompt_instruct(task: dict, tokenizer) -> tuple[str, str]:
    """Format a chat-template prompt for instruct models.

    Returns (full_prompt, code_prefix). For instruct models the code_prefix
    is empty since the model generates the complete response.
    """
    user_msg = (
        f"Complete the following Python function. "
        f"Provide only the function implementation, no explanations.\n\n"
        f"{task['prompt']}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), ""
