"""LiveCodeBench prompt builder — the benchmark's OWN canonical code-generation
prompt (lcb_runner/prompts/code_generation.py), applied through our chat-template
+ enable_thinking machinery. Branches on starter_code: empty => stdin/stdout
instructions (AtCoder/CodeForces), non-empty => complete-the-function (LeetCode).

Mirrors bench.apps.prompts.format_prompt_apps_instruct's (prompt, code_prefix)
contract and old-Qwen tokenize-direct handling.
"""
from __future__ import annotations

from bench.livecodebench.dataset import LcbProblem

_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)

_FMT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
_FMT_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT."
)


def _user_message(problem: LcbProblem) -> str:
    p = f"### Question:\n{problem.question}\n\n"
    if problem.starter_code:
        p += f"### Format: {_FMT_WITH_STARTER}\n"
        p += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        p += f"### Format: {_FMT_STDIN}\n"
        p += "```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return p


_SYSTEM_COT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification). Reason carefully before writing code, then "
    "generate a correct Python program that matches the specification and "
    "passes all tests."
)


def _user_message_cot(problem: LcbProblem) -> str:
    """User message for the induced-CoT (``<think>`` prefill) prompt.

    Same ``### Question``/``### Format``/``### Answer`` scaffold as
    :func:`_user_message`, with a reasoning instruction inserted before the
    format block. Mirrors ``bench.apps.prompts._user_message_cot``: asks the
    model to *close* the reasoning block with ``</think>`` but not to *open*
    one — the opening ``<think>`` is supplied by the prefill in
    :func:`format_prompt_lcb_cot_prefill`. The closing-tag instruction is
    load-bearing for the same downstream machinery APPS's variant relies on
    (``cot_efficiency.extract_think_span`` -> ``generator._strip_think_content``).
    """
    p = f"### Question:\n{problem.question}\n\n"
    p += (
        "First reason through your approach: identify the algorithm, the "
        "input/output format, and the edge cases. When your reasoning is "
        "complete, write </think> on its own line, then provide your "
        "solution in the format below.\n\n"
    )
    if problem.starter_code:
        p += f"### Format: {_FMT_WITH_STARTER}\n"
        p += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        p += f"### Format: {_FMT_STDIN}\n"
        p += "```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return p


def format_prompt_lcb_cot_prefill(
    problem: LcbProblem,
    tokenizer,
) -> tuple[str, str]:
    """Induce a chain-of-thought from an *instruct* (non-reasoning) model via a
    ``<think>`` prefill, DeepSeek-R1-Distill style — LCB counterpart to
    :func:`bench.apps.prompts.format_prompt_apps_cot_prefill`.

    The rendered chat prompt ends with ``<think>\\n`` so generation starts
    mid-reasoning; :func:`_user_message_cot` tells the model to close with
    ``</think>`` then write the program.

    Returns ``(prompt, code_prefix)``; ``code_prefix`` is always ``""``.

    Only the modern string-template path is supported — old-Qwen tokenizers
    (the ``_qwen_direct_tokenize`` flag) are not used by the instruct models
    this mode targets, and a token-id prefill would need different handling.
    """
    if getattr(tokenizer, "_qwen_direct_tokenize", False):
        raise NotImplementedError(
            "cot-prefill prompt format does not support the old-Qwen "
            "tokenize-direct path (string prefill only)."
        )
    messages = [
        {"role": "system", "content": _SYSTEM_COT},
        {"role": "user", "content": _user_message_cot(problem)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt += "<think>\n"
    return prompt, ""


def format_prompt_lcb_instruct(
    problem: LcbProblem,
    tokenizer,
    enable_thinking: bool = False,
    system_prompt: str | None = None,
) -> tuple[str | list[int], str]:
    """Chat-template prompt for instruct models on an LCB problem.

    Returns ``(prompt, code_prefix)``; code_prefix is always empty. ``prompt`` is a
    string (default) or token-id list (old-Qwen tokenize-direct).

    ``system_prompt`` overrides the default ``_SYSTEM`` message when given —
    needed for model families whose reasoning mode is triggered by specific
    system-prompt content rather than a chat-template kwarg (e.g. Ministral-3
    Reasoning's official ``SYSTEM_PROMPT.txt``, which our own ``_SYSTEM``
    would otherwise silently replace, since ``enable_thinking`` is a Qwen3
    template variable that Mistral's template doesn't consume at all).
    """
    messages = [
        {"role": "system", "content": system_prompt if system_prompt is not None else _SYSTEM},
        {"role": "user", "content": _user_message(problem)},
    ]
    extra_kwargs = {"enable_thinking": enable_thinking}
    if getattr(tokenizer, "_qwen_direct_tokenize", False):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, **extra_kwargs,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra_kwargs,
        )
    return prompt, ""
