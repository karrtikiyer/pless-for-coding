"""Tests for the scaffold-injection APPS prompt formatter.

Experiment: feed Qwen3-8B (thinking OFF) an external Claude-Opus algorithm
scaffold and see whether it can now code tasks it never solved with its own
thinking ON. The load-bearing correctness properties tested here:

  * the scaffold text is injected into the user message,
  * ``enable_thinking`` is forwarded to ``apply_chat_template`` unchanged
    (the experiment requires thinking OFF), and
  * ``scaffold=None`` produces byte-identical output to
    ``format_prompt_apps_instruct`` (the control path must be regression-free).

Mirrors ``tests/test_apps_bigcode_prompt.py``: import-inside-test, a stub
tokenizer, exact-string assertions (avoids downloading a real tokenizer).
"""
from __future__ import annotations


def _make_problem(*, question, starter_code="", fn_name=None):
    from bench.apps.dataset import AppsProblem
    return AppsProblem(
        problem_id=4000,
        source="ATCODER",
        difficulty="interview",
        question=question,
        starter_code=starter_code,
        fn_name=fn_name,
    )


class _StubTokenizer:
    """Records the kwargs passed to apply_chat_template and renders the
    messages to a deterministic string so tests can assert on content."""

    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.last_kwargs = kw
        if tokenize:
            raise NotImplementedError("stub returns strings only")
        out = ""
        for m in messages:
            out += f"<|{m['role']}|>\n{m['content']}\n"
        if add_generation_prompt:
            out += "<|assistant|>\n"
        return out


def test_module_exposes_format_prompt_apps_scaffold():
    from bench.apps.prompts import format_prompt_apps_scaffold
    assert callable(format_prompt_apps_scaffold)


def test_scaffold_text_appears_in_user_message():
    from bench.apps.prompts import format_prompt_apps_scaffold
    scaffold = "1. Read n from stdin.\n2. Maintain a prefix-sum array `pre`.\n3. Output the answer."
    p = _make_problem(question="Compute the running maximum.")
    tok = _StubTokenizer()
    prompt, code_prefix = format_prompt_apps_scaffold(
        p, tok, scaffold=scaffold, enable_thinking=False,
    )
    assert scaffold in prompt
    assert "A correct high-level approach" in prompt
    # thinking must stay OFF — this is the whole experimental premise.
    assert tok.last_kwargs.get("enable_thinking") is False
    assert code_prefix == ""


def test_enable_thinking_is_forwarded_unchanged():
    """The formatter forwards whatever enable_thinking it is given."""
    from bench.apps.prompts import format_prompt_apps_scaffold
    p = _make_problem(question="anything")
    tok = _StubTokenizer()
    format_prompt_apps_scaffold(p, tok, scaffold="steps", enable_thinking=True)
    assert tok.last_kwargs.get("enable_thinking") is True


def test_scaffold_none_matches_instruct_baseline():
    """With scaffold=None the formatter delegates to
    format_prompt_apps_instruct → byte-identical control-path output."""
    from bench.apps.prompts import (
        format_prompt_apps_instruct,
        format_prompt_apps_scaffold,
    )
    p = _make_problem(question="Find the maximum.")
    scaffold_out, sc_prefix = format_prompt_apps_scaffold(
        p, _StubTokenizer(), scaffold=None, enable_thinking=False,
    )
    instruct_out, in_prefix = format_prompt_apps_instruct(
        p, _StubTokenizer(), enable_thinking=False,
    )
    assert scaffold_out == instruct_out
    assert sc_prefix == in_prefix == ""


def test_user_message_scaffold_builder_structure():
    """The bare user-message builder embeds the baseline body, the header,
    the scaffold, and a final 'implement it' instruction."""
    from bench.apps.prompts import _user_message, _user_message_scaffold
    p = _make_problem(question="Sort the array.")
    scaffold = "1. Read the array.\n2. Sort it.\n3. Print it."
    msg = _user_message_scaffold(p, scaffold)
    assert _user_message(p) in msg
    assert "A correct high-level approach" in msg
    assert scaffold in msg
    # must ask for a runnable stdin/stdout program at the end
    assert "standard input" in msg
    assert "```python" in msg
