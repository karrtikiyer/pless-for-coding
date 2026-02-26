"""Tests for bench.eval.plots."""

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from bench.eval.plots import plot_aggregate_lines, plot_correctness_vs_diversity, load_metrics


def _make_metrics(model: str, method: str) -> dict:
    """Create a minimal metrics dict for testing."""
    return {
        "model": model,
        "method": method,
        "temperature": 1.0,
        "dataset": "mbpp",
        "num_tasks": 3,
        "num_samples_per_task": 10,
        "pass_at_k": {"1": 0.8, "3": 0.85, "5": 0.9, "10": 0.95},
        "cover_at_t": {"0.1": 90.0, "0.3": 70.0, "0.5": 50.0, "0.7": 30.0},
        "cover_at_t_distinct": {"0.1": 90.0, "0.3": 40.0, "0.5": 20.0, "0.7": 5.0},
        "per_task": [
            {"task_id": i, "num_correct": 7, "num_distinct_correct": 5, "pass_results": [True] * 7 + [False] * 3}
            for i in range(3)
        ],
    }


def _sample_metrics_list():
    return [
        _make_metrics("Qwen/Qwen2.5-7B", "pless"),
        _make_metrics("Qwen/Qwen2.5-7B", "pless_norm"),
        _make_metrics("Qwen/Qwen2.5-Coder-7B-Instruct", "pless"),
        _make_metrics("Qwen/Qwen2.5-Coder-7B-Instruct", "pless_norm"),
    ]


def test_load_metrics(tmp_path):
    data = _make_metrics("test/model", "pless")
    p = tmp_path / "m.json"
    p.write_text(json.dumps(data))
    result = load_metrics([p])
    assert len(result) == 1
    assert result[0]["model"] == "test/model"


def test_plot_aggregate_lines(tmp_path):
    out = tmp_path / "lines.png"
    plot_aggregate_lines(_sample_metrics_list(), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_correctness_vs_diversity(tmp_path):
    out = tmp_path / "bubble.png"
    plot_correctness_vs_diversity(_sample_metrics_list(), out)
    assert out.exists()
    assert out.stat().st_size > 0
