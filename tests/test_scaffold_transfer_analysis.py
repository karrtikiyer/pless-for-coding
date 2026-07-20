"""Tests for the scaffold-transfer reduction (treatment vs control)."""
from __future__ import annotations


def _metrics(per_task, *, n=5, pass_at_1=0.0):
    return {
        "num_samples_per_task": n,
        "pass_at_k": {"1": pass_at_1},
        "per_task": per_task,
    }


def test_compute_transfer_identifies_newly_recovered_and_regressions():
    from bench.eval.scaffold_transfer_analysis import compute_transfer
    baseline = _metrics([
        {"task_id": 1, "num_correct": 0},
        {"task_id": 2, "num_correct": 2},
        {"task_id": 3, "num_correct": 0},
    ])
    treatment = _metrics([
        {"task_id": 1, "num_correct": 3},   # recovered
        {"task_id": 2, "num_correct": 0},   # regressed
        {"task_id": 3, "num_correct": 0},   # still failing
    ])
    r = compute_transfer(baseline, treatment)
    assert r["baseline_solved"] == {2}
    assert r["treatment_solved"] == {1}
    assert r["newly_recovered"] == {1}
    assert r["regressions"] == {2}
    assert r["n_tasks"] == 3


def test_compute_transfer_per_task_rows_carry_both_counts():
    from bench.eval.scaffold_transfer_analysis import compute_transfer
    baseline = _metrics([{"task_id": 10, "num_correct": 0}], n=5)
    treatment = _metrics([{"task_id": 10, "num_correct": 4}], n=5)
    r = compute_transfer(baseline, treatment)
    row = {x["task_id"]: x for x in r["rows"]}[10]
    assert row["baseline_num_correct"] == 0
    assert row["treatment_num_correct"] == 4
    assert row["baseline_pass_at_1"] == 0.0
    assert row["treatment_pass_at_1"] == 4 / 5
    assert row["newly_recovered"] is True
