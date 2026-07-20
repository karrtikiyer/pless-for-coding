"""Tests for the --scaffold-file runner flag (bench.apps.runner)."""
from __future__ import annotations

from pathlib import Path

_BASE = [
    "--model", "Qwen/Qwen3-8B",
    "--source", "ATCODER",
    "--difficulty", "interview",
    "--method", "temp",
]


def test_scaffold_file_defaults_to_none():
    from bench.apps.runner import _build_argparser
    args = _build_argparser().parse_args(_BASE)
    assert args.scaffold_file is None


def test_scaffold_file_parses_to_path():
    from bench.apps.runner import _build_argparser
    args = _build_argparser().parse_args(
        _BASE + ["--scaffold-file", "results/scaffold_transfer/scaffolds.jsonl"]
    )
    assert args.scaffold_file == Path("results/scaffold_transfer/scaffolds.jsonl")
