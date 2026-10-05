"""Tests for the canonical headline table generator.

Locks the column set + formatting once so future analyses can't drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.eval.headline_table import (
    CANONICAL_COLUMNS,
    TERMINAL_COLUMNS,
    headline_row,
    headline_table,
)


@pytest.fixture
def synthetic_metrics(tmp_path):
    """Write a minimal-but-complete metrics JSON to a temp file."""
    p = tmp_path / "synth_metrics.json"
    p.write_text(
        json.dumps(
            {
                "model": "TestModel",
                "method": "pless_alpha",
                "temperature": 1.0,
                "dataset": "mbpp",
                "num_tasks": 500,
                "num_samples_per_task": 10,
                "pass_at_k": {"1": 0.7708, "3": 0.8022, "5": 0.8126, "8": 0.8180, "10": 0.8200},
                "cover_at_t": {"0.1": 82.0, "0.3": 80.4, "0.5": 77.6, "0.7": 70.1},
                "cover_at_t_distinct": {"0.3": 60.4, "0.5": 50.6, "0.7": 40.1},
                "termination_pct": 69.5,
                "mean_token_size": 14311.0,
                "timeout_pct": 1.3,
                "structural_diversity": 0.0579,
                "codebleu_diversity": 0.1328,
                "per_task": [],
            }
        )
    )
    return p


def test_canonical_columns_locked():
    """The canonical column order is the load-bearing contract. Lock it.

    Order: N (completion), pass@1/3/5/8/10 (sample-efficiency), cov@0.3/0.5/0.7
    (coverage) + their distinct variants, term%/mean_tok/timeout% (reasoning-
    model diagnostics), then struct_div + cb_div (CPU-heavy diversity, docs only).
    """
    assert CANONICAL_COLUMNS == [
        "N",
        "pass@1",
        "pass@3",
        "pass@5",
        "pass@8",
        "pass@10",
        "cov@0.3",
        "cov@0.5",
        "cov@0.7",
        "cov@0.3d",
        "cov@0.5d",
        "cov@0.7d",
        "term%",
        "mean_tok",
        "timeout%",
        "struct_div",
        "cb_div",
    ]


def test_terminal_columns_excludes_diversity():
    """TERMINAL_COLUMNS is CANONICAL_COLUMNS minus the two CPU-heavy diversity
    columns — terminal/CLI reports shouldn't force a diversity pass."""
    assert TERMINAL_COLUMNS == CANONICAL_COLUMNS[:-2]
    assert "struct_div" not in TERMINAL_COLUMNS
    assert "cb_div" not in TERMINAL_COLUMNS


def test_headline_row_extracts_canonical_fields(synthetic_metrics):
    row = headline_row(synthetic_metrics, label="α=2.0")
    assert row["label"] == "α=2.0"
    assert row["N"] == "500"
    # pass@k as percentage with 2 decimals
    assert row["pass@1"] == "77.08%"
    assert row["pass@3"] == "80.22%"
    assert row["pass@5"] == "81.26%"
    assert row["pass@8"] == "81.80%"
    assert row["pass@10"] == "82.00%"
    # cov_at_t as percentage with 1 decimal (matches existing summaries)
    assert row["cov@0.3"] == "80.4%"
    assert row["cov@0.5"] == "77.6%"
    assert row["cov@0.7"] == "70.1%"
    assert row["cov@0.3d"] == "60.4%"
    assert row["cov@0.5d"] == "50.6%"
    assert row["cov@0.7d"] == "40.1%"
    # Reasoning-model diagnostics
    assert row["term%"] == "69.5%"
    assert row["mean_tok"] == "14311"
    assert row["timeout%"] == "1.3%"
    # Diversity as raw 4-decimal value
    assert row["struct_div"] == "0.0579"
    assert row["cb_div"] == "0.1328"


def test_headline_row_handles_missing_k(tmp_path):
    """If pass@k is missing a key (e.g., k=3 unavailable), display as '—'."""
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps(
            {
                "pass_at_k": {"1": 0.5, "10": 0.8},
                "cover_at_t": {"0.3": 50.0, "0.5": 40.0},
                "structural_diversity": 0.1,
                "codebleu_diversity": 0.2,
            }
        )
    )
    row = headline_row(p, label="x")
    assert row["pass@1"] == "50.00%"
    assert row["pass@10"] == "80.00%"
    assert row["pass@3"] == "—"
    assert row["pass@5"] == "—"
    assert row["pass@8"] == "—"


def test_headline_row_handles_missing_reasoning_diagnostics(tmp_path):
    """Non-reasoning-model runs (e.g. MBPP/HumanEval without a thinking
    phase) have no termination_pct/mean_token_size/timeout_pct — display
    as '—' rather than erroring."""
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps(
            {
                "num_tasks": 500,
                "pass_at_k": {"1": 0.5},
                "cover_at_t": {"0.3": 50.0},
                "cover_at_t_distinct": {"0.3": 40.0},
                "termination_pct": None,
                "mean_token_size": None,
                "timeout_pct": None,
                "structural_diversity": None,
                "codebleu_diversity": None,
            }
        )
    )
    row = headline_row(p, label="x")
    assert row["term%"] == "—"
    assert row["mean_tok"] == "—"
    assert row["timeout%"] == "—"


def test_headline_table_emits_markdown(synthetic_metrics, tmp_path):
    p2 = tmp_path / "m2.json"
    p2.write_text(
        json.dumps(
            {
                "num_tasks": 500,
                "pass_at_k": {"1": 0.80, "3": 0.85, "5": 0.87, "8": 0.89, "10": 0.90},
                "cover_at_t": {"0.3": 85.0, "0.5": 80.0, "0.7": 75.0},
                "cover_at_t_distinct": {"0.3": 65.0, "0.5": 60.0, "0.7": 55.0},
                "termination_pct": 72.0,
                "mean_token_size": 14010.0,
                "timeout_pct": 1.0,
                "structural_diversity": 0.15,
                "codebleu_diversity": 0.30,
            }
        )
    )
    table = headline_table(
        [(synthetic_metrics, "α=2.0"), (p2, "α=5.0")],
    )
    assert "| Config" in table
    assert "pass@1" in table
    assert "cb_div" in table
    # Both rows present
    assert "α=2.0" in table
    assert "α=5.0" in table
    # Values present (spot check)
    assert "77.08%" in table
    assert "0.0579" in table
    assert "90.00%" in table
    assert "0.3000" in table
    # Markdown table separator row present
    assert "|---" in table


def test_headline_table_terminal_columns_omits_diversity(synthetic_metrics):
    """headline_table(..., columns=TERMINAL_COLUMNS) must not render
    struct_div/cb_div at all, even when the metrics JSON has them."""
    table = headline_table([(synthetic_metrics, "test")], columns=TERMINAL_COLUMNS)
    assert "struct_div" not in table
    assert "cb_div" not in table
    assert "0.0579" not in table  # the struct_div value itself
    assert "term%" in table
    assert "69.5%" in table


def test_headline_table_column_count(synthetic_metrics):
    """Each data row has exactly len(CANONICAL_COLUMNS) + 1 columns (label +
    metrics)."""
    table = headline_table([(synthetic_metrics, "test")])
    # Find the data row (starts with "| test")
    data_row = [line for line in table.split("\n") if line.startswith("| test")][0]
    expected_pipes = len(CANONICAL_COLUMNS) + 2  # one before label, one per column
    assert data_row.count("|") == expected_pipes, f"Unexpected column count: {data_row}"


def test_headline_table_empty_list():
    """Empty input returns just the header + separator, no data rows."""
    table = headline_table([])
    assert "Config" in table
    assert "pass@1" in table
    # No data rows — strip header row + markdown separator row
    data_lines = [
        l for l in table.split("\n")
        if l.startswith("|")
        and "Config" not in l
        and ":---" not in l
        and "---:" not in l
    ]
    assert data_lines == []
