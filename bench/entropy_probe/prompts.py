"""Prompt formatters for the cross-domain entropy probe.

For non-code domains we **explicitly forbid code** in the system message,
so a coder-tuned model (e.g. Qwen2.5-Coder) doesn't slip into writing
Python programs for a math word problem. If a model ignored that
instruction, its output structure would mimic code, the entropy
distribution would look bimodal for output-format reasons rather than
domain reasons, and the experiment would be invalidated. The probe
runner inspects a few generations after each run; if the constraint is
visibly violated, that model/dataset cell is flagged.

For the MBPP code baseline we use a standard programmer prompt (no
no-code constraint, since the task *is* code generation).
"""
from __future__ import annotations


GSM8K_SYSTEM = (
    "You are a careful math tutor. Solve the problem step by step in "
    "plain English. Do not write any code, do not use markdown code "
    "fences, and do not use any programming syntax (no 'def', no "
    "'print', no parentheses around function calls). Show your "
    "arithmetic in standard mathematical notation. End your response "
    "with 'The answer is X.' where X is the final numeric answer."
)

MATH_SYSTEM = (
    "You are a careful math tutor. Solve the competition math problem "
    "step by step in plain English, using LaTeX for equations and "
    "mathematical expressions. Do not write any code, do not use "
    "markdown code fences, and do not use any programming syntax "
    "(no 'def', no 'print', no Python). End your response with "
    "\\boxed{ANSWER} where ANSWER is the final answer."
)

MBPP_SYSTEM = (
    "You are an expert Python programmer. Write a Python function "
    "that solves the problem. Return only the function definition "
    "in a single ```python ... ``` code block, with no explanation."
)


SYSTEM_BY_DATASET = {
    "gsm8k": GSM8K_SYSTEM,
    "math": MATH_SYSTEM,
    "mbpp": MBPP_SYSTEM,
}


def format_prompt(dataset: str, problem: str, tokenizer) -> str:
    """Build a chat-templated prompt for ``dataset``'s system+user pair.

    Routes through the model's own ``tokenizer.apply_chat_template`` so
    each model gets its native chat syntax wrapping. The *content* of
    the system message is held fixed per dataset; cross-model
    comparisons stay clean.
    """
    if dataset not in SYSTEM_BY_DATASET:
        raise ValueError(
            f"Unknown dataset {dataset!r}; valid: {list(SYSTEM_BY_DATASET)}"
        )
    messages = [
        {"role": "system", "content": SYSTEM_BY_DATASET[dataset]},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
