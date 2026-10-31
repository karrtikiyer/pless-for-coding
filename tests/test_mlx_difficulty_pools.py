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
        # backfill past the cap (default: no easy backfill).
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


from bench.mlx_difficulty_pools import MLX_PREFILL_BATCH_SIZE, assign_row_ranges, batch_seed  # noqa: E402


class TestAssignRowRanges:
    def test_starts_at_banked_count_and_advances_across_batches(self):
        batches = build_pool_batches([("a", "easy", 10), ("b", "easy", 10), ("c", "easy", 10)])
        ranges = assign_row_ranges(batches, banked={"c": 0})
        assert ranges == [[("a", 0, 10), ("b", 0, 10), ("c", 0, 5)], [("c", 5, 5)]]

    def test_resume_offsets_by_banked(self):
        batches = build_pool_batches([("c", "easy", 5)])
        assert assign_row_ranges(batches, banked={"c": 5}) == [[("c", 5, 5)]]

    def test_shape_matches_batches(self):
        batches = build_pool_batches(_tasks("e", "easy", 3) + _tasks("h", "hard", 2))
        ranges = assign_row_ranges(batches, banked={})
        assert [[(t, n) for t, _, n in r] for r in ranges] == [b.rows for b in batches]


class TestBatchSeed:
    def test_deterministic_and_in_uint32_range(self):
        s = batch_seed(42, "gk_0_1", "easy", [("a", 0, 10)])
        assert s == batch_seed(42, "gk_0_1", "easy", [("a", 0, 10)])
        assert 0 <= s < 2**32

    def test_python_hash_salt_does_not_matter(self):
        import subprocess, sys
        code = ("from bench.mlx_difficulty_pools import batch_seed;"
                "print(batch_seed(42, 'gk_0_1', 'easy', [('a', 0, 10)]))")
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={"PYTHONHASHSEED": h, "PATH": ""}, cwd=".").stdout for h in ("1", "2")}
        assert len(outs) == 1 and outs != {""}

    @pytest.mark.parametrize("change", [
        dict(base=43), dict(config="gk_0_15"), dict(phase="hard"),
        dict(rows=[("a", 5, 10)]), dict(rows=[("a", 0, 9)]), dict(rows=[("b", 0, 10)]),
    ])
    def test_any_input_change_changes_seed(self, change):
        args = dict(base=42, config="gk_0_1", phase="easy", rows=[("a", 0, 10)])
        ref = batch_seed(args["base"], args["config"], args["phase"], args["rows"])
        args.update(change)
        assert batch_seed(args["base"], args["config"], args["phase"], args["rows"]) != ref

    def test_crash_resume_counterexample_gets_a_new_seed(self):
        # Reviewer's counterexample: X = A10, B10, C5 with C's 5 banked before the crash.
        # Content-only seeding rebuilt an identical batch and seed -> duplicate C samples.
        x = assign_row_ranges(build_pool_batches([("A", "easy", 10), ("B", "easy", 10), ("C", "easy", 10)]),
                              banked={})[0]
        resumed = assign_row_ranges(build_pool_batches([("A", "easy", 10), ("B", "easy", 10), ("C", "easy", 5)]),
                                    banked={"C": 5})[0]
        assert [(t, n) for t, _, n in x] == [(t, n) for t, _, n in resumed]
        assert batch_seed(42, "k", "easy", x) != batch_seed(42, "k", "easy", resumed)

    def test_low_cap_identical_batches_get_distinct_seeds(self):
        ranges = assign_row_ranges(build_pool_batches([("A", "hard", 10)], hard_cap=5), banked={})
        assert batch_seed(42, "k", "hard", ranges[0]) != batch_seed(42, "k", "hard", ranges[1])


class TestEasyBackfill:
    """Opt-in backfill for the easy phase only: each easy batch inserts up to
    easy_cap rows but at most easy_backfill run at once (mlx-lm queues the rest
    and backfills as rows finish). Medium/hard never backfill."""

    def test_easy_batches_backfill_medium_hard_do_not(self):
        tasks = _tasks("e", "easy", 30) + _tasks("m", "medium", 4) + _tasks("h", "hard", 2)
        for b in build_pool_batches(tasks, easy_cap=120, easy_backfill=12):
            if b.phase == "easy":
                assert _total(b) <= 120 and b.completion_batch_size == 12
            else:
                assert b.completion_batch_size >= _total(b)

    def test_easy_chunking_with_backfill(self):
        easy = [b for b in build_pool_batches(_tasks("e", "easy", 30), easy_cap=120, easy_backfill=12)
                if b.phase == "easy"]
        assert [_total(b) for b in easy] == [120, 120, 60]

    def test_small_easy_batch_does_not_queue(self):
        easy = [b for b in build_pool_batches([("e", "easy", 5)], easy_cap=120, easy_backfill=12)]
        assert easy[0].completion_batch_size >= _total(easy[0])

    def test_default_is_no_backfill(self):
        for b in build_pool_batches(_tasks("e", "easy", 5)):
            assert b.completion_batch_size >= _total(b)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonpositive_backfill_raises(self, bad):
        with pytest.raises(ValueError, match="easy_backfill"):
            build_pool_batches(_tasks("e", "easy", 3), easy_backfill=bad)

    def test_backfill_batches_prefill_one_row_at_a_time(self):
        # mlx-lm admits up to prefill_batch_size prompts while generation is below
        # completion_batch_size; at the default 8 a cap of 12 reached 14 resident
        # rows (measured). prefill_batch_size=1 held exactly 12.
        tasks = _tasks("e", "easy", 30) + _tasks("m", "medium", 2)
        for b in build_pool_batches(tasks, easy_cap=120, easy_backfill=12):
            backfills = b.completion_batch_size < _total(b)
            assert b.prefill_batch_size == (1 if backfills else MLX_PREFILL_BATCH_SIZE)

    def test_row_ranges_and_seeds_unchanged_by_backfill(self):
        tasks = _tasks("e", "easy", 30)
        a = assign_row_ranges(build_pool_batches(tasks, easy_cap=120), banked={})
        b = assign_row_ranges(build_pool_batches(tasks, easy_cap=120, easy_backfill=12), banked={})
        assert a == b


class TestMediumBackfill:
    """Opt-in backfill for medium, same mechanics as easy; hard never backfills."""

    def test_medium_backfill_8_hard_untouched(self):
        tasks = _tasks("e", "easy", 3) + _tasks("m", "medium", 30) + _tasks("h", "hard", 3)
        for b in build_pool_batches(tasks, easy_cap=120, medium_cap=120, easy_backfill=12,
                                    medium_backfill=8):
            if b.phase == "medium":
                assert _total(b) <= 120 and b.completion_batch_size == 8 and b.prefill_batch_size == 1
            elif b.phase == "hard":
                assert b.completion_batch_size >= _total(b)
                assert b.prefill_batch_size == MLX_PREFILL_BATCH_SIZE

    def test_medium_default_is_no_backfill(self):
        for b in build_pool_batches(_tasks("m", "medium", 30), medium_cap=120):
            assert b.completion_batch_size >= _total(b)

    def test_medium_chunk_not_larger_than_backfill_does_not_queue(self):
        b = build_pool_batches([("m", "medium", 5)], medium_cap=120, medium_backfill=8)[0]
        assert b.completion_batch_size >= _total(b)

    @pytest.mark.parametrize("bad", [0, -2])
    def test_nonpositive_medium_backfill_raises(self, bad):
        with pytest.raises(ValueError, match="medium_backfill"):
            build_pool_batches(_tasks("m", "medium", 3), medium_backfill=bad)

    def test_row_ranges_unchanged_by_medium_backfill(self):
        tasks = _tasks("m", "medium", 30)
        a = assign_row_ranges(build_pool_batches(tasks, medium_cap=120), banked={})
        b = assign_row_ranges(build_pool_batches(tasks, medium_cap=120, medium_backfill=8), banked={})
        assert a == b
