"""Tests for the bigcode-eval-harness-compatible APPS prompt formatter.

Goal: reproduce bigcode-evaluation-harness's `bigcode_eval/tasks/apps.py::
get_prompt()` output verbatim. Used to isolate "backend effect" (HF vs vLLM)
from "prompt effect" — we feed both backends identical bigcode-style
prompts and measure pass@k divergence.

Reference (bigcode-evaluation-harness/bigcode_eval/tasks/apps.py main branch):

    def get_prompt(self, doc):
        starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        except ValueError:
            fn_name = None
        prompt = "\\nQUESTION:\\n"
        prompt += doc["question"]
        if starter_code:
            prompt += starter_code
        if not fn_name:
            call_format = "\\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\\nUse Call-Based format"
            prompt += call_format
        prompt += "\\nANSWER:\\n"
        return prompt
"""
from __future__ import annotations

import pytest


def _make_problem(*, question, starter_code="", fn_name=None):
    """Build an AppsProblem with the minimal fields needed by the formatter."""
    from bench.apps.dataset import AppsProblem
    return AppsProblem(
        problem_id=4000,
        source="CODEFORCES",
        difficulty="interview",
        question=question,
        starter_code=starter_code,
        fn_name=fn_name,
    )


def test_module_exposes_format_prompt_apps_bigcode_default():
    """Public API surface must exist."""
    from bench.apps.prompts import format_prompt_apps_bigcode_default
    assert callable(format_prompt_apps_bigcode_default)


def test_bigcode_prompt_codeforces_style_no_starter_no_fn_name():
    """Standard CODEFORCES problem: no starter_code, no fn_name → uses
    'Use Standard Input format'. The exact string must match bigcode's
    get_prompt() byte-for-byte."""
    from bench.apps.prompts import format_prompt_apps_bigcode_default
    q = "Given an array, find its maximum element."
    p = _make_problem(question=q)
    prompt, code_prefix = format_prompt_apps_bigcode_default(p)
    expected = "\nQUESTION:\n" + q + "\nUse Standard Input format\nANSWER:\n"
    assert prompt == expected, (
        f"Prompt does not match bigcode's get_prompt() output.\n"
        f"Expected: {expected!r}\n"
        f"Got:      {prompt!r}"
    )
    assert code_prefix == ""


def test_bigcode_prompt_with_starter_code():
    """starter_code (when non-empty) gets appended BEFORE the call_format
    string. Tests the prompt += starter_code branch."""
    from bench.apps.prompts import format_prompt_apps_bigcode_default
    q = "Implement the function below."
    sc = "\ndef solve(arr):\n    pass\n"
    p = _make_problem(question=q, starter_code=sc)
    prompt, _ = format_prompt_apps_bigcode_default(p)
    expected = "\nQUESTION:\n" + q + sc + "\nUse Standard Input format\nANSWER:\n"
    assert prompt == expected


def test_bigcode_prompt_call_based_when_fn_name_set():
    """When fn_name is non-None, the format string is 'Use Call-Based
    format' (rare for CODEFORCES, common for LeetCode-style buckets)."""
    from bench.apps.prompts import format_prompt_apps_bigcode_default
    q = "Write a function to compute the factorial."
    p = _make_problem(question=q, fn_name="factorial")
    prompt, _ = format_prompt_apps_bigcode_default(p)
    expected = "\nQUESTION:\n" + q + "\nUse Call-Based format\nANSWER:\n"
    assert prompt == expected


def test_bigcode_prompt_does_NOT_add_chat_template():
    """Critical: bigcode-eval-harness treats the model as a base completion
    model. NO system prompt, NO ### Instruction / ### Response wrappers,
    NO BOS token (the tokenizer adds BOS at its own step if at all).
    First char of the returned prompt should be '\\n'."""
    from bench.apps.prompts import format_prompt_apps_bigcode_default
    p = _make_problem(question="anything")
    prompt, _ = format_prompt_apps_bigcode_default(p)
    assert prompt.startswith("\nQUESTION:\n")
    # Must NOT contain Deepseek's chat-template artifacts
    assert "### Instruction:" not in prompt
    assert "### Response:" not in prompt
    assert "<|begin" not in prompt
    assert "<｜begin" not in prompt
    assert "AI programming assistant" not in prompt


def test_bigcode_prompt_returns_str_not_tuple_or_list():
    """Return shape: (prompt_str, code_prefix) — same as format_prompt_apps_instruct."""
    from bench.apps.prompts import format_prompt_apps_bigcode_default
    p = _make_problem(question="anything")
    result = format_prompt_apps_bigcode_default(p)
    assert isinstance(result, tuple)
    assert len(result) == 2
    prompt, code_prefix = result
    assert isinstance(prompt, str)
    assert isinstance(code_prefix, str)
