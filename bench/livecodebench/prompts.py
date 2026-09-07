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


def format_prompt_lcb_instruct(
    problem: LcbProblem,
    tokenizer,
    enable_thinking: bool = False,
) -> tuple[str | list[int], str]:
    """Chat-template prompt for instruct models on an LCB problem.

    Returns ``(prompt, code_prefix)``; code_prefix is always empty. ``prompt`` is a
    string (default) or token-id list (old-Qwen tokenize-direct)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
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
