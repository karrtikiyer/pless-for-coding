# Standard HumanEval stop sequences (from Codex/HumanEval paper)
# These prevent base models from generating past the target function.
# Trailing spaces/parens avoid false positives on substrings like def_ or class_name.
HUMANEVAL_STOP_SEQUENCES = ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint("]


def is_instruct_model(model_id: str) -> bool:
    """Check if a model ID indicates an instruct/chat model."""
    lower = model_id.lower()
    if "instruct" in lower or "chat" in lower:
        return True
    # Qwen3 series are instruction-tuned by default (separate -Base variants exist)
    if "qwen3-" in lower and "-base" not in lower:
        return True
    return False


def format_prompt_base(task: dict) -> tuple[str, str]:
    """Format a completion-style prompt for base models.

    HumanEval's `prompt` field already contains the function signature + docstring,
    so we pass it directly. The model completes the function body.

    Returns (prompt_text, code_prefix) where code_prefix is the prompt itself
    so that prefix + generated_body = complete executable code.
    """
    prompt = task["prompt"]
    return prompt, prompt


def format_prompt_instruct(task: dict, tokenizer, enable_thinking: bool = False) -> tuple[str, str]:
    """Format a chat-template prompt for instruct models.

    Returns (full_prompt, code_prefix). For instruct models the code_prefix
    is empty since the model generates the complete response.

    Args:
        enable_thinking: Pass enable_thinking to the chat template (Qwen3 thinking mode).
                         Always passed explicitly — Qwen3 defaults to thinking ON, so we
                         must send False to suppress it when not wanted.  Non-Qwen3
                         templates silently ignore the unknown kwarg (Jinja passthrough).
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
    extra_kwargs = {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **extra_kwargs
    ), ""
