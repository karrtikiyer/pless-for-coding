"""Tests for the scaffold JSONL loader (bench.apps.scaffolds.load_scaffolds)."""
from __future__ import annotations

import pytest


def test_load_scaffolds_maps_int_task_id_to_scaffold(tmp_path):
    from bench.apps.scaffolds import load_scaffolds
    p = tmp_path / "scaffolds.jsonl"
    p.write_text(
        '{"task_id": 117, "scaffold": "algorithm A", "model": "claude-opus-4-8"}\n'
        '{"task_id": 257, "scaffold": "algorithm B", "model": "claude-opus-4-8"}\n'
    )
    mapping = load_scaffolds(p)
    assert mapping == {117: "algorithm A", 257: "algorithm B"}
    assert all(isinstance(k, int) for k in mapping)


def test_load_scaffolds_ignores_blank_lines(tmp_path):
    from bench.apps.scaffolds import load_scaffolds
    p = tmp_path / "scaffolds.jsonl"
    p.write_text(
        '{"task_id": 1, "scaffold": "a"}\n'
        '\n'
        '   \n'
        '{"task_id": 2, "scaffold": "b"}\n'
    )
    assert load_scaffolds(p) == {1: "a", 2: "b"}


def test_load_scaffolds_last_row_wins_on_duplicate(tmp_path):
    """Resume/re-request can append a second row for the same task_id;
    the most recent scaffold should win."""
    from bench.apps.scaffolds import load_scaffolds
    p = tmp_path / "scaffolds.jsonl"
    p.write_text(
        '{"task_id": 5, "scaffold": "old"}\n'
        '{"task_id": 5, "scaffold": "new"}\n'
    )
    assert load_scaffolds(p) == {5: "new"}


def test_load_scaffolds_missing_file_raises(tmp_path):
    """A wrong --scaffold-file path is a user error — fail loud, not silent."""
    from bench.apps.scaffolds import load_scaffolds
    with pytest.raises(FileNotFoundError):
        load_scaffolds(tmp_path / "does_not_exist.jsonl")
