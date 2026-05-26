"""Tests for bench.eval.apps_extractor — APPS-aware code extraction.

Motivation for the new prefix/window-strip rescue strategies:
on the Deepseek-6.7B-Instruct Phase A run, 35.3% of samples were
classified as ParsingError (vs 1.4% for the paper). A full-corpus
audit (2026-05-26) showed 94.1% of those are recoverable by simple
prefix/window stripping — the model frequently emits code without
markdown fences, prefixed by a token of garbage prose (e.g. '\\n\\nceed:\\n').

Test cases below codify the recovery behavior and the property that
the existing fence-based strategies continue to win when fences are
present (so we don't regress on cleanly-fenced models like Qwen).
"""
from __future__ import annotations

import pytest


# ─── Existing strategies must still work (regression guards) ────────────


def test_python_fence_winning_path():
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "Here's the code:\n```python\nprint('hello')\n```"
    r = extract_python_code_apps(txt)
    assert r.success
    assert r.strategy == "python_fence_asis"
    assert r.code == "print('hello')\n"


def test_plain_fence_falls_back_when_no_python_fence():
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "```\nprint('hi')\n```"
    r = extract_python_code_apps(txt)
    assert r.success
    assert r.strategy.startswith("plain_fence")


def test_raw_text_with_valid_code_compiles_as_is():
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "print('hello world')\n"  # bare code, no fence
    r = extract_python_code_apps(txt)
    assert r.success
    # raw text path was used (no fences anywhere)
    assert r.strategy.startswith("raw")


def test_empty_returns_failure():
    from bench.eval.apps_extractor import extract_python_code_apps
    r = extract_python_code_apps("")
    assert not r.success
    assert r.code == ""


# ─── New: prefix-strip rescue ────────────────────────────────────────────


def test_prefix_strip_recovers_real_deepseek_sample():
    """Real example from cell1 task_id=4000 sample_idx=0. The sample
    begins with bad prose ('\\n\\nceed:\\n') followed by valid Python.
    Without prefix-strip, this is ParsingError. With prefix-strip, the
    valid Python after the bad line should be recovered."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = (
        "\n\nceed:\ndef dfs(v, p):\n"
        "    size[v] = 1\n"
        "    for u in adj[v]:\n"
        "        if u != p:\n"
        "            dfs(u, v)\n"
        "            size[v] += size[u]\n"
        "\nn = int(input().strip())\n"
        "adj = [[] for _ in range(n+1)]\n"
        "size = [0] * (n+1)\n"
    )
    r = extract_python_code_apps(txt)
    assert r.success, f"expected recovery, got strategy={r.strategy}, reason={r.reason_if_failed!r}"
    # The recovered code should contain the def line
    assert "def dfs" in r.code
    # Should NOT contain the bad preamble
    assert "ceed:" not in r.code


def test_prefix_strip_strategy_name_exposed():
    """The strategy field should mark prefix-strip recoveries so
    diagnostics can attribute them."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "junk preamble line\ndef f():\n    return 1\n"
    r = extract_python_code_apps(txt)
    assert r.success
    # New strategy name we're adding
    assert "prefix" in r.strategy.lower(), f"got strategy={r.strategy}"


def test_prefix_strip_does_not_apply_when_fence_works():
    """When a ```python``` fence yields a compilable program, the fence
    strategy must win — we do NOT want to fall through to prefix-strip."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "preamble that doesn't compile alone\n```python\nprint(1)\n```\n"
    r = extract_python_code_apps(txt)
    assert r.success
    assert r.strategy.startswith("python_fence"), \
        f"fence should still win, got {r.strategy}"


# ─── New: window-strip rescue (drop leading AND trailing) ────────────────


def test_window_strip_recovers_bad_preamble_and_bad_suffix():
    """Sample has bad preamble + valid Python middle + bad trailing
    garbage. Neither prefix-strip alone nor suffix-strip alone works."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = (
        "ceed:\n"                                  # bad preamble
        "def add(a, b):\n    return a + b\n"       # valid middle
        "print(add(2, 3))\n"                       # valid middle
        "this is not python and will not compile!\n"  # bad suffix
    )
    r = extract_python_code_apps(txt)
    assert r.success, f"expected recovery, got strategy={r.strategy}, reason={r.reason_if_failed!r}"
    assert "def add" in r.code
    assert "ceed:" not in r.code
    assert "not python" not in r.code


def test_window_strip_strategy_name_exposed():
    """Recoveries that needed BOTH prefix-drop and suffix-drop should
    be attributable via a distinct strategy name."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "bad preamble\nprint('ok')\nbad suffix not python\n"
    r = extract_python_code_apps(txt)
    assert r.success
    # Either prefix or window — both are acceptable recovery names for
    # this case (prefix-strip alone may recover the print line by
    # dropping the bad suffix via the existing trim logic). We just want
    # one of the new names to appear.
    assert ("prefix" in r.strategy.lower()) or ("window" in r.strategy.lower())


# ─── Failure modes (nothing recoverable) ─────────────────────────────────


def test_pure_prose_is_unrecoverable():
    """Text with no Python anywhere should still fail extraction."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "I'm sorry but I cannot help with this problem. Please consult a textbook."
    r = extract_python_code_apps(txt)
    assert not r.success


def test_complete_garbage_is_unrecoverable():
    """Random non-Python tokens should fail extraction."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "@#$%^&* not valid syntax anywhere [{}] @@@"
    r = extract_python_code_apps(txt)
    assert not r.success


# ─── New: smart-dedent rescue (blank-line-aware) ─────────────────────────


def test_smart_dedent_recovers_blank_line_broken_indent():
    """Real Deepseek pattern: code at consistent 4-space indent but blank
    lines defeat textwrap.dedent (blank lines are 0-indent so common
    prefix collapses to 0). Smart-dedent ignores blank lines when
    computing common prefix."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = (
        "\n\n"  # leading blanks
        "    from collections import Counter\n"
        "\n"  # blank in middle — breaks textwrap.dedent
        "    n = int(input().strip())\n"
        "    arr = list(map(int, input().split()))\n"
        "    counter = Counter(arr)\n"
        "    if len(counter) > 1:\n"
        "        print(0)\n"
        "    else:\n"
        "        print(min(arr))\n"
    )
    r = extract_python_code_apps(txt)
    assert r.success, (
        f"smart-dedent should recover indented-code-with-blank-lines; "
        f"got strategy={r.strategy}, reason={r.reason_if_failed!r}"
    )
    assert "from collections" in r.code


def test_smart_dedent_handles_mixed_blank_and_indented_code_with_prose_suffix():
    """Variant that ALSO has prose at the end — recovery should still
    pick the compilable code-only portion."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = (
        "\n"
        "    def solve():\n"
        "\n"
        "        return 42\n"
        "\n"
        "    print(solve())\n"
        "\n"
        "This solution works by...\n"
    )
    r = extract_python_code_apps(txt)
    assert r.success
    assert "def solve" in r.code
    assert "solution works by" not in r.code


def test_smart_dedent_does_not_break_zero_indent_code():
    """If the code is already at 0-indent, smart-dedent should be a
    no-op and not change the result."""
    from bench.eval.apps_extractor import extract_python_code_apps
    txt = "def foo():\n    return 1\n"
    r = extract_python_code_apps(txt)
    assert r.success
    assert "def foo" in r.code
