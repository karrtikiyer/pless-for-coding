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


def format_prompt_apps_bigcode_default(
    problem: AppsProblem,
) -> tuple[str, str]:
    """Format an APPS prompt exactly as bigcode-evaluation-harness does.

    Reproduces the ``get_prompt`` method of
    ``bigcode_eval/tasks/apps.py::GeneralAPPS`` byte-for-byte. This is the
    prompt format the paper authors (Lee et al. 2025; arXiv:2503.00691)
    started from before applying their own per-model chat-template wrapping.

    Using this format lets us run the SAME prompt across HF transformers
    and vLLM backends and isolate the "backend choice" effect from the
    "prompt format" effect when comparing pass@k.

    Returns ``(prompt, code_prefix)``. ``code_prefix`` is always ``""`` —
    the bigcode protocol expects the model to write the full answer after
    ``ANSWER:\\n`` with no per-prompt boilerplate.

    Key properties (asserted in ``tests/test_apps_bigcode_prompt.py``):
      * No chat template applied
      * No system prompt
      * First character is ``"\\n"``; first line is ``"QUESTION:"``
      * No ``### Instruction`` / ``### Response`` wrapping
      * For CODEFORCES problems (no ``fn_name``): trailing
        ``"\\nUse Standard Input format\\nANSWER:\\n"``
      * For function-call problems: trailing
        ``"\\nUse Call-Based format\\nANSWER:\\n"``

    Reference:
        https://github.com/bigcode-project/bigcode-evaluation-harness/
        blob/main/bigcode_eval/tasks/apps.py
    """
    starter_code = problem.starter_code if problem.starter_code else ""
    fn_name = problem.fn_name  # None for stdin/stdout problems

    prompt = "\nQUESTION:\n"
    prompt += problem.question
    if starter_code:
        prompt += starter_code
    if not fn_name:
        call_format = "\nUse Standard Input format"
    else:
        call_format = "\nUse Call-Based format"
    prompt += call_format
    prompt += "\nANSWER:\n"
    return prompt, ""


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
