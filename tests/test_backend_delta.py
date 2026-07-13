"""TDD for the HF-vs-vLLM backend-delta comparison logic (scripts/backend_delta.py).

Pure logic only (no GPU, no file IO): task pairing, paired pass@k gap + bootstrap CI,
and the truncation-rate counter. Scoring itself is NOT reimplemented — pass@k routes
through the canonical bench.eval.metrics.compute_pass_at_k (the same estimator used for
the α=2/α=5 columns), so this comparison stays apple-to-apple with the existing pipeline.
"""
import pytest

from scripts.backend_delta import (
    bootstrap_gap,
    index_by_task,
    paired_task_results,
    pass_at_k,
    truncation_rate,
)


def _pt(task_id, results):
    """A minimal per_task record: results is a list of pass/fail bools."""
    return {"task_id": task_id, "pass_results": list(results),
            "num_correct": sum(results)}


def test_index_by_task_detects_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        index_by_task([_pt(1, [True]), _pt(1, [False])])


def test_paired_aligns_and_orders_by_task_id():
    hf = [_pt(5, [True, False]), _pt(1, [True, True])]
    v = [_pt(1, [False, False]), _pt(5, [True, True])]
    ids, hfr, vr = paired_task_results(hf, v)
    assert ids == [1, 5]                                  # sorted, shared
    assert [r["task_id"] for r in hfr] == [1, 5]
    assert [r["task_id"] for r in vr] == [1, 5]


def test_paired_restricts_to_subset():
    hf = [_pt(1, [True]), _pt(2, [True]), _pt(3, [False])]
    v = [_pt(1, [False]), _pt(2, [True]), _pt(3, [True])]
    ids, _, _ = paired_task_results(hf, v, subset=[2, 3])
    assert ids == [2, 3]


def test_paired_missing_subset_task_raises():
    hf = [_pt(1, [True])]
    v = [_pt(1, [False])]
    with pytest.raises(ValueError, match="missing"):
        paired_task_results(hf, v, subset=[1, 99])


def test_paired_no_shared_raises():
    with pytest.raises(ValueError, match="no shared"):
        paired_task_results([_pt(1, [True])], [_pt(2, [True])])


def test_pass_at_1_matches_unbiased_estimator():
    # c/n per task, averaged. Task A: 1/2=0.5, Task B: 2/2=1.0 -> mean 0.75
    res = [_pt(1, [True, False]), _pt(2, [True, True])]
    assert pass_at_k(res, (1,))["1"] == pytest.approx(0.75)


def test_bootstrap_gap_point_equals_difference_of_passk():
    hf = [_pt(i, [True, True]) for i in range(8)]        # HF pass@1 = 1.0
    v = [_pt(i, [False, False]) for i in range(8)]       # vLLM pass@1 = 0.0
    point, lo, hi = bootstrap_gap(hf, v, k=1, iters=500, seed=0)
    assert point == pytest.approx(1.0)
    assert lo <= point <= hi
    # a perfect, uniform gap has zero bootstrap variance
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_gap_is_deterministic_for_a_seed():
    hf = [_pt(i, [True, False]) for i in range(10)]
    v = [_pt(i, [i % 2 == 0, False]) for i in range(10)]
    a = bootstrap_gap(hf, v, k=1, iters=300, seed=42)
    b = bootstrap_gap(hf, v, k=1, iters=300, seed=42)
    assert a == b                                        # reproducible


def test_bootstrap_gap_requires_equal_length():
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_gap([_pt(1, [True])], [_pt(1, [True]), _pt(2, [True])])


def test_truncation_rate_counts_missing_think_close():
    recs = [
        {"task_id": 1, "samples_with_thinking": ["...</think> code", "loops forever"]},
        {"task_id": 2, "samples_with_thinking": ["done</think>x"]},
    ]
    trunc, n, rate = truncation_rate(recs)
    assert (trunc, n) == (1, 3)
    assert rate == pytest.approx(1 / 3)


def test_truncation_rate_respects_subset():
    recs = [
        {"task_id": 1, "samples_with_thinking": ["loops"]},          # truncated
        {"task_id": 2, "samples_with_thinking": ["ok</think>"]},     # not
    ]
    trunc, n, rate = truncation_rate(recs, subset=[2])
    assert (trunc, n) == (0, 1)


def test_truncation_rate_empty_raises():
    with pytest.raises(ValueError, match="no samples"):
        truncation_rate([])
