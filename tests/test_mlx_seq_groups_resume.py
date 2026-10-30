import json

import pytest

from bench.mlx_seq_groups_resume import (
    assemble_record,
    banked_sample_record,
    build_waves,
    load_banked_samples,
    load_jsonl_tolerant,
    remaining_samples_needed,
    truncate_to_last_newline,
)


def _write(path, lines):
    with path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


class TestTruncateToLastNewline:
    def test_missing_file_is_noop(self, tmp_path):
        p = tmp_path / "nope.jsonl"
        assert truncate_to_last_newline(p) == 0

    def test_clean_file_untouched(self, tmp_path):
        p = tmp_path / "clean.jsonl"
        _write(p, [{"a": 1}, {"a": 2}])
        before = p.read_bytes()
        assert truncate_to_last_newline(p) == 0
        assert p.read_bytes() == before

    def test_torn_trailing_line_is_dropped(self, tmp_path):
        p = tmp_path / "torn.jsonl"
        good = json.dumps({"a": 1}) + "\n"
        torn = '{"a": 2, "b": "unterm'  # no trailing newline, not valid json
        p.write_text(good + torn)
        dropped = truncate_to_last_newline(p)
        assert dropped == len(torn)
        assert p.read_text() == good
        # a subsequent append must not fuse with torn bytes
        with p.open("a") as f:
            f.write(json.dumps({"a": 3}) + "\n")
        records = load_jsonl_tolerant(p)
        assert records == [{"a": 1}, {"a": 3}]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert truncate_to_last_newline(p) == 0


class TestLoadJsonlTolerant:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_jsonl_tolerant(tmp_path / "nope.jsonl") == []

    def test_skips_malformed_middle_line(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text(
            json.dumps({"a": 1}) + "\n"
            + "{not json\n"
            + json.dumps({"a": 2}) + "\n"
        )
        assert load_jsonl_tolerant(p) == [{"a": 1}, {"a": 2}]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
        assert load_jsonl_tolerant(p) == [{"a": 1}, {"a": 2}]


class TestLoadBankedSamples:
    def test_groups_by_task_id(self, tmp_path):
        p = tmp_path / "partial.jsonl"
        recs = [
            banked_sample_record("t1", "<think>x</think>a", "a", "stop", 10, "attempt-1"),
            banked_sample_record("t2", "<think>y</think>b", "b", "stop", 12, "attempt-1"),
            banked_sample_record("t1", "<think>z</think>c", "c", "length", 25000, "attempt-1"),
        ]
        _write(p, recs)
        banked = load_banked_samples(p)
        assert set(banked.keys()) == {"t1", "t2"}
        assert len(banked["t1"]) == 2
        assert len(banked["t2"]) == 1

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert load_banked_samples(tmp_path / "nope.jsonl") == {}


class TestRemainingSamplesNeeded:
    def test_task_already_fully_done_needs_zero(self):
        assert remaining_samples_needed("t1", {"t1"}, {}, n_samples=10) == 0

    def test_task_not_banked_at_all_needs_full_n(self):
        assert remaining_samples_needed("t1", set(), {}, n_samples=10) == 10

    def test_task_partially_banked_needs_shortfall(self):
        banked = {"t1": [{"task_id": "t1"}] * 4}
        assert remaining_samples_needed("t1", set(), banked, n_samples=10) == 6

    def test_task_banked_at_or_above_n_needs_zero(self):
        banked = {"t1": [{"task_id": "t1"}] * 10}
        assert remaining_samples_needed("t1", set(), banked, n_samples=10) == 0
        banked_over = {"t1": [{"task_id": "t1"}] * 13}
        assert remaining_samples_needed("t1", set(), banked_over, n_samples=10) == 0

    def test_done_ids_takes_precedence_over_banked_count(self):
        # a full record already exists in the main file -- banked count is
        # irrelevant (and could even be 0 if the sidecar was pruned after a
        # successful write).
        assert remaining_samples_needed("t1", {"t1"}, {"t1": []}, n_samples=10) == 0


class TestAssembleRecord:
    def _sample(self, task_id, i):
        return banked_sample_record(task_id, f"think{i}", f"code{i}", "stop", 100 + i, f"attempt-{i}", seed=1000 + i)

    def test_combines_banked_then_fresh_exactly_n(self):
        banked = [self._sample("t1", i) for i in range(4)]
        fresh = [self._sample("t1", i) for i in range(4, 10)]
        rec = assemble_record("t1", banked, fresh, n_samples=10, base_fields={"model": "m"})
        assert rec["task_id"] == "t1"
        assert len(rec["samples"]) == 10
        assert len(rec["samples_with_thinking"]) == 10
        assert len(rec["attempt_ids"]) == 10
        assert rec["samples"] == [f"code{i}" for i in range(10)]
        assert rec["seeds"] == [1000 + i for i in range(10)]
        assert rec["model"] == "m"

    def test_seeds_distinguish_banked_vs_fresh_attempts(self):
        banked = [self._sample("t1", i) for i in range(4)]  # seeds 1000-1003
        fresh = [self._sample("t1", i) for i in range(4, 10)]  # seeds 1004-1009
        rec = assemble_record("t1", banked, fresh, n_samples=10, base_fields={})
        assert len(set(rec["seeds"])) == 10  # all distinct across the merge

    def test_all_from_banked_no_fresh(self):
        banked = [self._sample("t1", i) for i in range(10)]
        rec = assemble_record("t1", banked, [], n_samples=10, base_fields={})
        assert len(rec["samples"]) == 10

    def test_truncates_extra_banked_to_exactly_n(self):
        banked = [self._sample("t1", i) for i in range(13)]
        rec = assemble_record("t1", banked, [], n_samples=10, base_fields={})
        assert len(rec["samples"]) == 10

    def test_raises_if_not_enough_combined(self):
        banked = [self._sample("t1", i) for i in range(3)]
        fresh = [self._sample("t1", i) for i in range(3, 5)]
        with pytest.raises(ValueError, match="only 5 samples available"):
            assemble_record("t1", banked, fresh, n_samples=10, base_fields={})

    def test_does_not_mutate_base_fields(self):
        base = {"model": "m"}
        banked = [self._sample("t1", i) for i in range(10)]
        assemble_record("t1", banked, [], n_samples=10, base_fields=base)
        assert base == {"model": "m"}


class TestBuildWaves:
    def test_typical_group_deep_cap_8_interleaves_medium_and_hard(self):
        # real production shape: 1 easy + 1 medium + 1 hard, 10 samples each.
        # deep pool (medium+hard) = 20 rows, cap=8 -> 3 waves: 8, 8, 4.
        tasks = [("easy1", "easy", 10), ("med1", "medium", 10), ("hard1", "hard", 10)]
        waves = build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
        assert waves == [
            [("easy1", 10), ("med1", 8)],
            [("med1", 2), ("hard1", 6)],
            [("hard1", 4)],
        ]
        # every wave respects the combined deep-row cap
        deep_tids = {"med1", "hard1"}
        for w in waves:
            deep_total = sum(n for tid, n in w if tid in deep_tids)
            assert deep_total <= 8

    def test_no_deep_task_single_wave(self):
        tasks = [("easy1", "easy", 10), ("easy2", "easy", 10)]
        waves = build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
        assert waves == [[("easy1", 10), ("easy2", 10)]]

    def test_deep_shortfall_under_cap_single_wave(self):
        # resume case: hard task only needs 3 more samples, fits in one wave
        tasks = [("easy1", "easy", 2), ("hard1", "hard", 3)]
        waves = build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
        assert waves == [[("easy1", 2), ("hard1", 3)]]

    def test_single_deep_task_still_splits_above_cap(self):
        tasks = [("hard1", "hard", 10)]
        waves = build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
        assert waves == [[("hard1", 8)], [("hard1", 2)]]

    def test_empty_tasks_no_waves(self):
        assert build_waves([], deep_row_cap=8, max_rows_per_wave=32) == []

    def test_all_rows_preserved_across_waves(self):
        tasks = [("easy1", "easy", 10), ("med1", "medium", 10), ("hard1", "hard", 10)]
        waves = build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
        totals: dict[str, int] = {}
        for wave in waves:
            for tid, n in wave:
                totals[tid] = totals.get(tid, 0) + n
        assert totals == {"easy1": 10, "med1": 10, "hard1": 10}

    def test_every_wave_respects_combined_deep_cap_with_two_deep_tasks(self):
        tasks = [("med1", "medium", 10), ("hard1", "hard", 10)]
        waves = build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
        for w in waves:
            assert sum(n for _, n in w) <= 8
        totals: dict[str, int] = {}
        for wave in waves:
            for tid, n in wave:
                totals[tid] = totals.get(tid, 0) + n
        assert totals == {"med1": 10, "hard1": 10}

    def test_raises_if_wave_exceeds_max_rows(self):
        tasks = [("e1", "easy", 20), ("e2", "easy", 20)]
        with pytest.raises(ValueError, match="exceeds max_rows_per_wave"):
            build_waves(tasks, deep_row_cap=8, max_rows_per_wave=32)
