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


def format_prompt_apps_bigcode_chat(
    problem: AppsProblem,
    tokenizer,
) -> tuple[str, str]:
    """Wrap bigcode-eval-harness's bare APPS prompt in the model's chat
    template — the modification paper authors most plausibly applied to
    make bigcode-eval-harness work on chat-tuned models like
    Deepseek-Coder-Instruct.

    The bigcode prompt (``QUESTION:/Use Standard Input format/ANSWER:``)
    becomes the *user-message content*. ``tokenizer.apply_chat_template``
    then adds the model-specific system prompt + role wrappers
    (``### Instruction:`` / ``### Response:`` for Deepseek; varies by
    model). The model sees its expected chat framing on the outside and
    bigcode's QUESTION/ANSWER markers on the inside.

    Why this exists: bigcode's bare prompt fails on instruct models
    (output drifts to C++, off-topic, or model collapse — verified
    empirically in the smoke for this experiment). Wrapping with chat
    template re-activates the model's instruct-tuning prior (Python
    by default) without modifying bigcode's task-specific content.

    Returns ``(prompt, code_prefix)`` like the other formatters.
    ``code_prefix`` is always ``""``.
    """
    bare_prompt, _ = format_prompt_apps_bigcode_default(problem)
    messages = [{"role": "user", "content": bare_prompt}]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return wrapped, ""


def _user_message_cot(problem: AppsProblem) -> str:
    """User message for the induced-CoT (``<think>`` prefill) prompt.

    Unlike :func:`_user_message`, this asks the model to *reason first* and then
    emit the program. Crucially it instructs the model to **close** the reasoning
    block with ``</think>`` but does NOT ask it to *open* one — the opening
    ``<think>`` is supplied by the prefill in
    :func:`format_prompt_apps_cot_prefill` (asking for it again risks a doubled
    ``<think><think>``). The closing-tag instruction is load-bearing: everything
    downstream (``cot_efficiency.extract_think_span`` →
    ``generator._strip_think_content``) keys on the literal ``</think>`` string.
    """
    parts = [
        "Solve the following programming problem in Python. The program must "
        "read input from standard input and write its answer to standard "
        "output.",
        "",
        "First reason through your approach: identify the algorithm, the input "
        "format, and the edge cases. When your reasoning is complete, write "
        "</think> on its own line, then provide the complete Python program as "
        "a single ```python ... ``` code block and nothing after it.",
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


def format_prompt_apps_cot_prefill(
    problem: AppsProblem,
    tokenizer,
) -> tuple[str, str]:
    """Induce a chain-of-thought from an *instruct* (non-reasoning) model via a
    ``<think>`` prefill, DeepSeek-R1-Distill style.

    The rendered chat prompt ends with ``<think>\\n`` so generation starts
    *mid-reasoning* — the model never decides whether to open a reasoning block,
    it is already inside one. The user message (see :func:`_user_message_cot`)
    tells it to close with ``</think>`` then write the program.

    This mirrors how R1-Distill emits CoT (opening tag in the prompt, only the
    closing ``</think>`` in the output), so the repo's text-based CoT machinery
    reuses with zero changes:
      * ``cot_efficiency.extract_think_span`` — ``start == -1`` branch.
      * ``generator._strip_think_content`` — returns text after the last
        ``</think>``.

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
        {"role": "system",
         "content": "You are an expert competitive programmer. Reason "
                    "carefully before writing code."},
        {"role": "user", "content": _user_message_cot(problem)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt += "<think>\n"
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
