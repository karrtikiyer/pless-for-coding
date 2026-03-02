"""Tests for bench.eval.parse_humaneval."""

import json
import textwrap

import pytest

from bench.eval.parse_humaneval import compute_metrics_for_method, parse_detailed


# -- Fixtures ----------------------------------------------------------------

SAMPLE_CODE_BODY = """\
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

SAMPLE_CODE_BODY_ALT = """\
    sorted_nums = sorted(numbers)
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i + 1] - sorted_nums[i] < threshold:
            return True
    return False
"""


def _make_detailed(methods=None, n_tasks=3, n_samples=5, n_correct=3):
    """Build a minimal detailed JSON dict."""
    if methods is None:
        methods = ["greedy", "temp_0.2"]

    data = {}
    for method in methods:
        tasks = []
        for i in range(n_tasks):
            samples = []
            for j in range(n_samples):
                passed = j < n_correct
                code = SAMPLE_CODE_BODY if j % 2 == 0 else SAMPLE_CODE_BODY_ALT
                samples.append({"code": code, "passed": passed})
            tasks.append({
                "task_id": f"HumanEval/{i}",
                "n_samples": n_samples,
                "n_correct": n_correct,
                "samples": samples,
            })
        data[method] = tasks
    return data


# -- Tests -------------------------------------------------------------------


class TestParseDetailed:
    def test_basic_conversion(self):
        data = _make_detailed(methods=["greedy"], n_tasks=2, n_samples=4, n_correct=2)
        task_results, records = parse_detailed(data, "greedy")

        assert len(task_results) == 2
        assert len(records) == 2

        for tr in task_results:
            assert "task_id" in tr
            assert "num_correct" in tr
            assert "pass_results" in tr
            assert tr["num_correct"] == 2
            assert len(tr["pass_results"]) == 4
            assert tr["pass_results"] == [True, True, False, False]

    def test_records_have_dedented_code(self):
        data = _make_detailed(methods=["greedy"], n_tasks=1, n_samples=2, n_correct=1)
        _, records = parse_detailed(data, "greedy")

        for record in records:
            assert "task_id" in record
            assert "samples" in record
            for code in record["samples"]:
                # After dedent, the code should not start with spaces
                assert not code.startswith("    "), (
                    f"Code should be dedented but starts with spaces: {code[:40]!r}"
                )

    def test_task_ids_preserved(self):
        data = _make_detailed(methods=["p_less"], n_tasks=5)
        task_results, records = parse_detailed(data, "p_less")

        result_ids = [tr["task_id"] for tr in task_results]
        record_ids = [r["task_id"] for r in records]

        assert result_ids == record_ids
        assert result_ids == [f"HumanEval/{i}" for i in range(5)]


class TestComputeMetrics:
    def test_produces_expected_keys(self):
        data = _make_detailed(methods=["greedy"], n_tasks=3, n_samples=5, n_correct=3)
        task_results, records = parse_detailed(data, "greedy")

        metrics = compute_metrics_for_method(
            task_results, records,
            model="test-model", method="greedy",
            k_values=[1, 5], t_values=[0.1, 0.5],
        )

        assert metrics["model"] == "test-model"
        assert metrics["method"] == "greedy"
        assert metrics["dataset"] == "humaneval"
        assert metrics["num_tasks"] == 3
        assert metrics["num_samples_per_task"] == 5
        assert "pass_at_k" in metrics
        assert "cover_at_t" in metrics
        assert "cover_at_t_distinct" in metrics
        assert "per_task" in metrics
        assert "timestamp" in metrics

    def test_pass_at_k_values(self):
        # All samples correct -> pass@1 should be 1.0
        data = _make_detailed(methods=["greedy"], n_tasks=2, n_samples=5, n_correct=5)
        task_results, records = parse_detailed(data, "greedy")

        metrics = compute_metrics_for_method(
            task_results, records,
            model="test-model", method="greedy",
            k_values=[1, 5], t_values=[0.5],
        )

        assert metrics["pass_at_k"]["1"] == 1.0
        assert metrics["pass_at_k"]["5"] == 1.0

    def test_pass_at_k_zero_correct(self):
        data = _make_detailed(methods=["greedy"], n_tasks=2, n_samples=5, n_correct=0)
        task_results, records = parse_detailed(data, "greedy")

        metrics = compute_metrics_for_method(
            task_results, records,
            model="test-model", method="greedy",
            k_values=[1, 5], t_values=[0.5],
        )

        assert metrics["pass_at_k"]["1"] == 0.0

    def test_distinct_counts_populated(self):
        data = _make_detailed(methods=["greedy"], n_tasks=2, n_samples=5, n_correct=3)
        task_results, records = parse_detailed(data, "greedy")

        metrics = compute_metrics_for_method(
            task_results, records,
            model="test-model", method="greedy",
            k_values=[1], t_values=[0.1],
        )

        for task in metrics["per_task"]:
            assert "num_distinct_correct" in task

    def test_fingerprinting_dedented_code(self):
        """Verify that dedented function bodies can be AST-fingerprinted."""
        from bench.eval.fingerprint import ast_fingerprint

        dedented = textwrap.dedent(SAMPLE_CODE_BODY)
        fp = ast_fingerprint(dedented)
        assert fp is not None, "Dedented code body should be parseable by AST"

        dedented_alt = textwrap.dedent(SAMPLE_CODE_BODY_ALT)
        fp_alt = ast_fingerprint(dedented_alt)
        assert fp_alt is not None
        assert fp != fp_alt, "Different implementations should have different fingerprints"
