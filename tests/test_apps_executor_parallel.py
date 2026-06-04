"""Tests for the per-(task, sample) parallelization in evaluate_all_apps.

The behavior change: work items are now per-sample (not per-task), evaluated by
a ProcessPoolExecutor that yields results in arbitrary completion order. The
correctness invariant is that results are regrouped back to per-task lists in
ORIGINAL SAMPLE ORDER, so pass_results[i] stays aligned with samples[i].
"""
import random

from bench.apps.dataset import AppsProblem
from bench.eval.apps_executor import (
    AppsSampleResult,
    _regroup_results,
    evaluate_all_apps,
)
from bench.eval.apps_extractor import ExtractionResult


# ── the critical invariant: order preserved under arbitrary completion order ──

def test_regroup_preserves_sample_order():
    n_samples_by_task = {1: 3, 2: 2}
    # Tuples arrive scrambled (as ProcessPoolExecutor.as_completed would yield).
    completed = [
        (2, 1, "r2.1", "e2.1"),
        (1, 2, "r1.2", "e1.2"),
        (1, 0, "r1.0", "e1.0"),
        (2, 0, "r2.0", "e2.0"),
        (1, 1, "r1.1", "e1.1"),
    ]
    res, ext = _regroup_results(completed, n_samples_by_task)
    assert res[1] == ["r1.0", "r1.1", "r1.2"]
    assert res[2] == ["r2.0", "r2.1"]
    assert ext[1] == ["e1.0", "e1.1", "e1.2"]
    assert all(x is not None for v in res.values() for x in v)


def test_regroup_robust_to_random_shuffle():
    n = {7: 6}
    items = [(7, i, f"r{i}", f"e{i}") for i in range(6)]
    random.Random(0).shuffle(items)
    res, _ = _regroup_results(items, n)
    assert res[7] == [f"r{i}" for i in range(6)]


# ── full function via a stubbed per-sample evaluator (workers=1, in-process) ──

def _stub_problem(pid):
    return AppsProblem(problem_id=pid, source="ATCODER", difficulty="introductory",
                       question="q", starter_code="", fn_name=None,
                       inputs=[], outputs=[])


def _stub_sample_eval(sample, problem, per_test_timeout=10.0):
    """Sample passes iff it starts with 'P' — lets us check order alignment."""
    passed = sample.startswith("P")
    r = AppsSampleResult(status="Passed" if passed else "Failed",
                         n_tests_total=1, n_tests_passed=1 if passed else 0,
                         first_failing_idx=None if passed else 0)
    e = ExtractionResult(code=sample, success=True, strategy="stub",
                         n_candidates_seen=1)
    return r, e


def test_evaluate_all_apps_aligns_pass_results(monkeypatch):
    import bench.eval.apps_executor as ax
    monkeypatch.setattr(ax, "evaluate_apps_sample", _stub_sample_eval)

    records = [
        {"task_id": 1, "samples": ["P0", "F1", "P2"]},
        {"task_id": 2, "samples": ["F0", "P1"]},
    ]
    problems = {1: _stub_problem(1), 2: _stub_problem(2)}
    task_results, ext_diag, exec_diag = evaluate_all_apps(records, problems, workers=1)

    by_tid = {t.task_id: t for t in task_results}
    assert by_tid[1].pass_results == [True, False, True]   # aligned with samples
    assert by_tid[2].pass_results == [False, True]
    assert by_tid[1].num_correct == 2
    assert by_tid[2].num_correct == 1
    assert by_tid[1].extraction_success == [True, True, True]
    assert by_tid[1].extracted_codes == ["P0", "F1", "P2"]
    assert exec_diag["by_status"]["Passed"] == 3
    assert ext_diag["n_samples_total"] == 5
    assert [t.task_id for t in task_results] == [1, 2]  # deterministic order


def test_evaluate_all_apps_skips_missing_problem(monkeypatch):
    import bench.eval.apps_executor as ax
    monkeypatch.setattr(ax, "evaluate_apps_sample", _stub_sample_eval)
    records = [{"task_id": 1, "samples": ["P0"]},
               {"task_id": 99, "samples": ["P0"]}]   # 99 has no problem
    problems = {1: _stub_problem(1)}
    task_results, ext_diag, _ = evaluate_all_apps(records, problems, workers=1)
    assert [t.task_id for t in task_results] == [1]
    assert ext_diag["n_records_skipped_no_problem"] == 1
