"""TDD for the cross-task batching scheduler bookkeeping (scripts/cross_task_batch.py).
Pure logic; the anti-misattribution contract (a sample's result lands in its own slot,
completeness enforced) is the point of these tests. No GPU.
"""
import pytest

from scripts.cross_task_batch import flatten_workitems, chunk_items, regroup_by_task


def test_flatten_stable_order():
    assert flatten_workitems([10, 20], 3) == [
        (10, 0), (10, 1), (10, 2), (20, 0), (20, 1), (20, 2)]


def test_chunk_respects_max_seqs():
    items = flatten_workitems([1, 2, 3], 4)          # 12 items
    groups = chunk_items(items, 5)
    assert [len(g) for g in groups] == [5, 5, 2]
    assert sum(groups, []) == items                  # no loss/reorder


def test_chunk_can_split_a_task_across_groups():
    items = flatten_workitems([7], 10)               # one task, 10 samples
    groups = chunk_items(items, 4)
    assert [len(g) for g in groups] == [4, 4, 2]     # a task may straddle group boundaries


def test_chunk_bad_maxseqs_raises():
    with pytest.raises(ValueError):
        chunk_items([(1, 0)], 0)


def test_regroup_reassembles_in_sample_order_across_groups():
    # results arrive out of order and interleaved across tasks (as pooled batches would)
    flat = [{"task_id": 20, "sample": 1, "v": "b1"},
            {"task_id": 10, "sample": 0, "v": "a0"},
            {"task_id": 20, "sample": 0, "v": "b0"},
            {"task_id": 10, "sample": 1, "v": "a1"}]
    out = regroup_by_task(flat, n=2)
    assert [r["v"] for r in out[10]] == ["a0", "a1"]  # sample-ordered, correct task
    assert [r["v"] for r in out[20]] == ["b0", "b1"]


def test_regroup_detects_missing_sample():
    flat = [{"task_id": 5, "sample": 0}]             # sample 1 missing for n=2
    with pytest.raises(ValueError, match="missing samples"):
        regroup_by_task(flat, n=2)


def test_regroup_detects_duplicate():
    flat = [{"task_id": 5, "sample": 0}, {"task_id": 5, "sample": 0}]
    with pytest.raises(ValueError, match="duplicate"):
        regroup_by_task(flat, n=2)


def test_regroup_detects_out_of_range_sample():
    flat = [{"task_id": 5, "sample": 3}]
    with pytest.raises(ValueError, match="out of range"):
        regroup_by_task(flat, n=2)
