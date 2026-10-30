"""Tests for bench/mlx_difficulty_pools.py -- the pure scheduler for the
difficulty-pool design (docs/mlx_seq_groups_difficulty_pool_scheduling_proposal.md, todos E24).
"""
from collections import Counter

import pytest

from bench.mlx_difficulty_pools import PoolBatch, build_pool_batches, select_next_phase


def _tasks(prefix: str, difficulty: str, count: int, n: int = 10):
    return [(f"{prefix}{i}", difficulty, n) for i in range(count)]


def _rows_by_task(batches):
    c = Counter()
    for b in batches:
        for tid, n in b.rows:
            c[tid] += n
    return c


def _total(b):
    return sum(n for _, n in b.rows)


class TestEasyPhase:
    def test_easy_is_capped_not_one_giant_batch(self):
        # A single 200-row easy batch with backfill can pin a 32-row KV
        # tensor at a looping row's full length (independent review, 2026-10-30).
        easy = [b for b in build_pool_batches(_tasks("e", "easy", 20)) if b.phase == "easy"]
        assert len(easy) == 8
        assert all(_total(b) == 25 for b in easy)

    def test_easy_cap_is_configurable(self):
        easy = [b for b in build_pool_batches(_tasks("e", "easy", 3), easy_cap=10)
                if b.phase == "easy"]
        assert [_total(b) for b in easy] == [10, 10, 10]


class TestMediumPhase:
    def test_default_cap_16(self):
        med = [b for b in build_pool_batches(_tasks("m", "medium", 20)) if b.phase == "medium"]
        assert [_total(b) for b in med] == [16] * 12 + [8]

    def test_partial_shortfalls_pack_greedily_up_to_cap(self):
        tasks = [("a", "medium", 3), ("b", "medium", 4), ("c", "medium", 15)]
        med = [b for b in build_pool_batches(tasks) if b.phase == "medium"]
        assert [b.rows for b in med] == [[("a", 3), ("b", 4), ("c", 9)], [("c", 6)]]

    def test_medium_cap_is_configurable(self):
        med = [b for b in build_pool_batches(_tasks("m", "medium", 4), medium_cap=10)
               if b.phase == "medium"]
        assert len(med) == 4
        assert all(_total(b) == 10 for b in med)


class TestHardPhase:
    def test_one_task_per_batch_when_n_equals_cap(self):
        hard = [b for b in build_pool_batches(_tasks("h", "hard", 20)) if b.phase == "hard"]
        assert len(hard) == 20
        for b in hard:
            assert len(b.rows) == 1
            assert _total(b) == 10

    def test_task_larger_than_cap_splits(self):
        hard = [b for b in build_pool_batches([("h", "hard", 25)]) if b.phase == "hard"]
        assert [b.rows for b in hard] == [[("h", 10)], [("h", 10)], [("h", 5)]]

    def test_resume_shortfalls_pack_up_to_cap_not_mixed_beyond_it(self):
        tasks = [("x", "hard", 3), ("y", "hard", 3), ("z", "hard", 4), ("w", "hard", 5)]
        hard = [b for b in build_pool_batches(tasks) if b.phase == "hard"]
        assert [b.rows for b in hard] == [[("x", 3), ("y", 3), ("z", 4)], [("w", 5)]]


class TestInvariants:
    def _mixed(self):
        return (_tasks("e", "easy", 7) + _tasks("m", "medium", 7)
                + _tasks("h", "hard", 7))

    def test_phase_order_is_easy_then_medium_then_hard_regardless_of_input_order(self):
        phases = [b.phase for b in build_pool_batches(list(reversed(self._mixed())))]
        order = {"easy": 0, "medium": 1, "hard": 2}
        assert phases == sorted(phases, key=order.__getitem__)

    def test_every_row_preserved_per_task(self):
        tasks = self._mixed() + [("odd", "hard", 7)]
        expected = Counter({tid: n for tid, _, n in tasks})
        assert _rows_by_task(build_pool_batches(tasks)) == expected

    def test_no_batch_exceeds_its_phase_cap(self):
        caps = {"easy": 25, "medium": 16, "hard": 10}
        for b in build_pool_batches(self._mixed()):
            assert _total(b) <= caps[b.phase]

    def test_completion_batch_size_never_queues_a_row(self):
        # >= rows means mlx-lm's BatchGenerator never queues, so it can never
        # backfill past the cap.
        for b in build_pool_batches(self._mixed()):
            assert b.completion_batch_size >= _total(b)

    def test_zero_row_tasks_are_skipped(self):
        batches = build_pool_batches([("a", "easy", 0), ("b", "hard", 0), ("c", "medium", 4)])
        assert [b.rows for b in batches] == [[("c", 4)]]

    def test_empty_input_gives_no_batches(self):
        assert build_pool_batches([]) == []

    def test_returns_pool_batch_instances(self):
        assert all(isinstance(b, PoolBatch) for b in build_pool_batches(self._mixed()))


class TestInputValidation:
    def test_unknown_difficulty_raises_rather_than_dropping_task(self):
        with pytest.raises(ValueError, match="unknown difficulty"):
            build_pool_batches([("a", "extreme", 10)])

    @pytest.mark.parametrize("kwargs", [
        {"medium_cap": 0}, {"hard_cap": 0}, {"easy_cap": 0}, {"medium_cap": -1},
    ])
    def test_nonpositive_caps_raise(self, kwargs):
        with pytest.raises(ValueError):
            build_pool_batches(_tasks("m", "medium", 1), **kwargs)

    @pytest.mark.parametrize("n", [2.5, "10", None, True])
    def test_non_int_row_count_raises(self, n):
        with pytest.raises(ValueError, match="row count"):
            build_pool_batches([("a", "hard", n)])

    def test_negative_row_count_raises(self):
        with pytest.raises(ValueError, match="row count"):
            build_pool_batches([("a", "hard", -1)])

    def test_duplicate_task_id_raises(self):
        with pytest.raises(ValueError, match="duplicate task_id"):
            build_pool_batches([("a", "easy", 5), ("a", "hard", 5)])


class TestSelectNextPhase:
    """The runner restarts the process after each phase, so it runs only
    the batches of the first phase that still has work."""

    def test_returns_only_the_first_phase_with_work(self):
        batches = build_pool_batches(_tasks("e", "easy", 3) + _tasks("h", "hard", 2))
        nxt = select_next_phase(batches)
        assert {b.phase for b in nxt} == {"easy"}
        assert sum(_total(b) for b in nxt) == 30

    def test_skips_to_medium_when_easy_is_done(self):
        batches = build_pool_batches(_tasks("m", "medium", 3) + _tasks("h", "hard", 1))
        assert {b.phase for b in select_next_phase(batches)} == {"medium"}

    def test_empty_when_nothing_left(self):
        assert select_next_phase([]) == []

    def test_only_phase_filter(self):
        batches = build_pool_batches(_tasks("e", "easy", 1) + _tasks("m", "medium", 2))
        assert {b.phase for b in select_next_phase(batches, only_phase="medium")} == {"medium"}
        assert select_next_phase(batches, only_phase="hard") == []

    def test_only_phase_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown phase"):
            select_next_phase([], only_phase="extreme")
