"""Tests for bench/eval/cot_efficiency.py — CoT token-efficiency analysis helpers.

These tests cover the pure helpers (think-span extraction, length measurement,
sample-row join + classification, per-config decomposition) without needing the
real Qwen3 tokenizer or any on-disk data.
"""
import pytest

from pathlib import Path

from bench.eval.cot_efficiency import (
    _apps_label,
    aggregate_rows,
    build_sample_rows,
    config_meta,
    extract_think_span,
    measure_lengths,
)


class _StubTokenizer:
    """Token count == whitespace-word count (deterministic, offline)."""

    def encode(self, text, add_special_tokens=False):  # noqa: D401, ARG002
        return text.split()


# ── extract_think_span ──────────────────────────────────────────────────────

def test_extract_think_span_closed():
    swt = "<think>step one step two</think>\n```python\nx = 1\n```"
    think, closed = extract_think_span(swt)
    assert think.strip() == "step one step two"
    assert closed is True


def test_extract_think_span_unclosed_is_truncated():
    swt = "<think>reasoning that never finishes because it hit the cap"
    think, closed = extract_think_span(swt)
    assert "never finishes" in think
    assert closed is False


def test_extract_think_span_no_open_tag():
    swt = "```python\nx = 1\n```"
    think, closed = extract_think_span(swt)
    assert think == ""
    assert closed is False


# ── measure_lengths ─────────────────────────────────────────────────────────

def test_measure_lengths_with_tokenizer():
    n_tok, n_chars = measure_lengths("one two three", _StubTokenizer())
    assert n_tok == 3
    assert n_chars == len("one two three")


def test_measure_lengths_no_tokenizer():
    n_tok, n_chars = measure_lengths("one two three", None)
    assert n_tok is None
    assert n_chars == 13


# ── build_sample_rows (join + classification) ───────────────────────────────

def _record(task_id, swt_list, code_list):
    return {
        "task_id": task_id,
        "samples_with_thinking": swt_list,
        "samples": code_list,
    }


def _metrics(per_task):
    return {"per_task": per_task, "num_samples_per_task": 2}


def test_build_sample_rows_alignment_and_pass_join():
    records = [
        _record(1,
                ["<think>a</think>\n```python\nok\n```",
                 "<think>b</think>\n```python\nok\n```"],
                ["```python\nok\n```", "```python\nok\n```"]),
    ]
    metrics = _metrics([{"task_id": 1, "num_correct": 1,
                         "pass_results": [True, False]}])
    rows = build_sample_rows(records, metrics, _StubTokenizer(), max_tokens=8192)
    assert len(rows) == 2
    assert [r["passed"] for r in rows] == [True, False]
    assert all(r["completed"] for r in rows)       # closed + has code
    assert not any(r["truncated"] for r in rows)


def test_build_sample_rows_truncated_when_unclosed():
    records = [_record(1, ["<think>cut off"], [""])]
    metrics = _metrics([{"task_id": 1, "num_correct": 0, "pass_results": [False]}])
    rows = build_sample_rows(records, metrics, None, max_tokens=8192)
    assert rows[0]["truncated"] is True
    assert rows[0]["completed"] is False


def test_build_sample_rows_malformed_closed_but_no_code():
    records = [_record(1, ["<think>done</think>"], [""])]
    metrics = _metrics([{"task_id": 1, "num_correct": 0, "pass_results": [False]}])
    rows = build_sample_rows(records, metrics, None, max_tokens=8192)
    assert rows[0]["closed"] is True
    assert rows[0]["malformed"] is True
    assert rows[0]["completed"] is False


def test_build_sample_rows_raises_on_length_mismatch():
    # 2 thinking samples but only 1 pass label -> hard fail (alignment invariant)
    records = [_record(1, ["<think>a</think>x", "<think>b</think>y"], ["x", "y"])]
    metrics = _metrics([{"task_id": 1, "num_correct": 1, "pass_results": [True]}])
    with pytest.raises((ValueError, AssertionError)):
        build_sample_rows(records, metrics, None, max_tokens=8192)


# ── aggregate_rows (efficiency decomposition) ───────────────────────────────

def test_aggregate_decomposition():
    # 4 samples: 3 completed (2 pass), 1 truncated (fail)
    rows = [
        {"completed": True, "truncated": False, "malformed": False,
         "near_cap": False, "passed": True, "think_tokens": 100, "think_chars": 400},
        {"completed": True, "truncated": False, "malformed": False,
         "near_cap": False, "passed": True, "think_tokens": 200, "think_chars": 800},
        {"completed": True, "truncated": False, "malformed": False,
         "near_cap": False, "passed": False, "think_tokens": 300, "think_chars": 1200},
        {"completed": False, "truncated": True, "malformed": False,
         "near_cap": True, "passed": False, "think_tokens": 8192, "think_chars": 30000},
    ]
    agg = aggregate_rows(rows)
    assert agg["n_samples"] == 4
    assert agg["completion_rate"] == pytest.approx(0.75)
    assert agg["truncation_rate"] == pytest.approx(0.25)
    assert agg["near_cap_rate"] == pytest.approx(0.25)
    # conditional accuracy = pass among completed = 2/3
    assert agg["conditional_accuracy"] == pytest.approx(2 / 3)
    # mean think tokens over completed = (100+200+300)/3 = 200
    assert agg["mean_think_tokens_completed"] == pytest.approx(200.0)
    # mean over all = (100+200+300+8192)/4
    assert agg["mean_think_tokens"] == pytest.approx((100 + 200 + 300 + 8192) / 4)


# ── APPS support ────────────────────────────────────────────────────────────

def test_apps_label():
    assert _apps_label("temp", 1.0, 0, 0.6) == "temp 0.6 (unfiltered)"
    assert _apps_label("temp", 1.0, 20, 1.0) == "temp 1.0 (top_k 20)"
    assert _apps_label("temp", 0.95, 0, 1.0) == "temp 1.0 (top_p 0.95)"
    assert _apps_label("temp", 0.95, 20, 0.6) == "temp 0.6 (top_p 0.95 + top_k 20)"
    assert _apps_label("pless", 1.0, 0, 1.0) == "pless (t1.0)"


def test_config_meta_apps_reads_filters_and_budget():
    rec = {"method": "temp", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
           "source": "ATCODER", "difficulty": "interview"}
    m = config_meta(rec, Path("temp_p0.95_k20_think_t0.6_t0.6.jsonl"),
                    dataset="apps", max_tokens_override=16384)
    assert m["method"] == "temp"
    assert m["top_p"] == 0.95 and m["top_k"] == 20
    assert m["max_tokens"] == 16384
    assert m["difficulty"] == "interview"
    assert m["label"] == "temp 0.6 (top_p 0.95 + top_k 20)"


def test_build_sample_rows_apps_has_code_override():
    # extraction_success drives completed/malformed, not the code string.
    records = [_record(1,
                       ["<think>a</think>prog", "<think>b</think>"],
                       ["", ""])]  # samples empty — strip_code_fences would say no code
    metrics = _metrics([{"task_id": 1, "num_correct": 1,
                         "pass_results": [True, False]}])
    rows = build_sample_rows(records, metrics, None, max_tokens=16384,
                             has_code_by_task={1: [True, False]})
    assert rows[0]["completed"] is True   # extraction_success=True → has code
    assert rows[1]["malformed"] is True   # closed but extraction_success=False


def test_aggregate_handles_no_completed():
    rows = [
        {"completed": False, "truncated": True, "malformed": False,
         "near_cap": True, "passed": False, "think_tokens": 8192, "think_chars": 1},
    ]
    agg = aggregate_rows(rows)
    assert agg["completion_rate"] == 0.0
    assert agg["conditional_accuracy"] is None
    assert agg["mean_think_tokens_completed"] is None
