"""APPS prompt formatting.

APPS problems are full competitive-programming statements (848-7,440 chars on
the test split). The paper's protocol asks for a *complete program* that reads
from stdin and writes to stdout — fundamentally different from MBPP, where the
target is a single function.

We do not include few-shot examples (the problem statement is already long)
and we do not append the human reference solution (we want generation, not
imitation). For instruct models we wrap the statement in a chat template that
mirrors ``bench.prompts.format_prompt_instruct`` so the tokenizer-direct path
for old-Qwen tokenizers also works.
"""

from __future__ import annotations

from bench.apps.dataset import AppsProblem


def _user_message(problem: AppsProblem) -> str:
    parts = [
        "Solve the following programming problem in Python. The program must "
        "read input from standard input and write its answer to standard "
        "output. Provide only the complete Python program in a single "
        "```python ... ``` code block, with no surrounding explanation.",
        "",
        "Problem:",
        problem.question.strip(),
    ]
    if problem.starter_code.strip():
        parts += [
            "",
            "Starter code (use this as your starting point):",
            "```python",
            problem.starter_code.strip(),
            "```",
        ]
    return "\n".join(parts)


def format_prompt_apps_instruct(
    problem: AppsProblem,
    tokenizer,
    enable_thinking: bool = False,
) -> tuple[str | list[int], str]:
    """Format a chat-template prompt for instruct models on an APPS problem.

    Returns ``(prompt, code_prefix)``. ``code_prefix`` is always empty for
    instruct models. ``prompt`` is either a string (default) or a list of
    token ids (for old-Qwen tokenizers that mangle their own template).

    Mirrors :func:`bench.prompts.format_prompt_instruct`'s tokenize-direct
    handling for old-Qwen tokenizers so we never feed model.generate a
    string that the encoder will split into subwords (would corrupt
    ``<|im_start|>`` and similar).
    """
    user_msg = _user_message(problem)
    messages = [
        {"role": "system",
         "content": "You are a helpful coding assistant. Write clean, correct "
                    "Python programs."},
        {"role": "user", "content": user_msg},
    ]
    extra_kwargs = {"enable_thinking": enable_thinking}
    if getattr(tokenizer, "_qwen_direct_tokenize", False):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, **extra_kwargs,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **extra_kwargs,
        )
    return prompt, ""
