import re


def _extract_function_name(test_list: list[str]) -> str | None:
    """Extract function name from assert statements like 'assert func_name(...)'."""
    for test in test_list:
        m = re.search(r"assert\s+(\w+)\s*\(", test)
        if m:
            return m.group(1)
    return None


def format_prompt_base(task: dict) -> tuple[str, str]:
    """Format a completion-style prompt for base models.

    Returns (full_prompt, code_prefix) where code_prefix is the part
    that should be prepended to the generated output to form complete code.
    e.g. code_prefix = "def func_name(" so that prefix + completion = full function.
    """
    desc = task["prompt"]
    tests = task["test_list"]
    func_name = _extract_function_name(tests)

    lines = [f'"""{desc}"""']
    for test in tests:
        lines.append(test)
    lines.append("")

    code_prefix = ""
    if func_name:
        code_prefix = f"def {func_name}("
        lines.append(code_prefix)

    return "\n".join(lines), code_prefix


def format_prompt_instruct(task: dict, tokenizer) -> tuple[str, str]:
    """Format a chat-template prompt for instruct models.

    Returns (full_prompt, code_prefix). For instruct models the code_prefix
    is empty since the model generates the complete function.
    """
    test_lines = "\n".join(task["test_list"])
    user_msg = (
        f"Write a Python function to solve the following task.\n\n"
        f"Task: {task['prompt']}\n\n"
        f"Your solution must pass these tests:\n```\n{test_lines}\n```\n\n"
        f"Provide only the function implementation, no explanations."
    )
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": user_msg},
    ]
    # Most tokenizers handle special tokens correctly when encoding the
    # template string, so default to the text path (preserves BOS behaviour).
    # Old Qwen tokenizers are the exception: <|im_start|>/<|im_end|> get
    # split into subwords by encode(), producing a garbled prompt.  For
    # those we tokenize directly (return_dict=False gives list[int] in
    # transformers 5.x which defaults to BatchEncoding otherwise).
    if getattr(tokenizer, '_qwen_direct_tokenize', False):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return prompt, ""


def is_instruct_model(model_id: str) -> bool:
    """Check if a model ID indicates an instruct/chat model."""
    lower = model_id.lower()
    return "instruct" in lower or "chat" in lower
