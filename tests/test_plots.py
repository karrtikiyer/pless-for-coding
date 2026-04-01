"""Tests for bench.eval.plots."""

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from bench.eval.plots import (
    plot_aggregate_lines,
    plot_aggregate_lines_faceted,
    plot_correctness_vs_diversity,
    plot_pareto_scatter,
    plot_method_heatmaps,
    load_metrics,
    _build_style_map,
)


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


def _multi_model_metrics_list():
    """4 models × 6 methods = 24 configs."""
    models = ["CodeLlama-7B", "Codestral-22B", "Qwen2.5-Coder-7B", "Qwen3-Coder-30B"]
    methods = ["greedy", "temp_0.2", "temp_0.7", "top_p_0.95", "p_less", "p_less_norm"]
    return [_make_metrics(m, method) for m in models for method in methods]


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


def test_plot_aggregate_lines_with_dataset_name(tmp_path):
    out = tmp_path / "lines_he.png"
    plot_aggregate_lines(_sample_metrics_list(), out, dataset_name="HumanEval")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_correctness_vs_diversity_with_dataset_name(tmp_path):
    out = tmp_path / "bubble_he.png"
    plot_correctness_vs_diversity(_sample_metrics_list(), out, dataset_name="HumanEval")
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_style_map_multi_model():
    metrics = _multi_model_metrics_list()
    style_map = _build_style_map(metrics)
    assert len(style_map) == 24
    # Each model should get a distinct colour
    colors_by_model = {}
    for (model, _method), style in style_map.items():
        colors_by_model.setdefault(model, set()).add(style["color"])
    # Within a model, all entries share the same colour
    for model, colors in colors_by_model.items():
        assert len(colors) == 1, f"Expected 1 color for {model}, got {colors}"
    # Across models, colours are distinct
    all_colors = {c for cs in colors_by_model.values() for c in cs}
    assert len(all_colors) == 4


def test_plot_aggregate_lines_24_configs(tmp_path):
    out = tmp_path / "lines_all.png"
    plot_aggregate_lines(_multi_model_metrics_list(), out, dataset_name="HumanEval")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_aggregate_lines_faceted(tmp_path):
    out = tmp_path / "lines_faceted.png"
    plot_aggregate_lines_faceted(_multi_model_metrics_list(), out, dataset_name="HumanEval")
    assert out.exists()
    assert out.stat().st_size > 0


def _make_metrics_with_diversity(model: str, method: str, pass1: float = 0.8, div: float = 0.3) -> dict:
    """Create metrics dict with diversity fields for Pareto/heatmap tests."""
    m = _make_metrics(model, method)
    m["pass_at_k"]["1"] = pass1
    m["structural_diversity"] = div
    m["codebleu_diversity"] = div * 0.9
    m["dataflow_match_diversity"] = div * 1.1
    return m


def _pareto_metrics_list():
    """4 models × 3 methods with varying pass@1 and diversity."""
    return [
        _make_metrics_with_diversity("ModelA", "greedy", 0.85, 0.0),
        _make_metrics_with_diversity("ModelA", "p_less", 0.82, 0.15),
        _make_metrics_with_diversity("ModelA", "temp_0.7", 0.78, 0.35),
        _make_metrics_with_diversity("ModelB", "greedy", 0.75, 0.0),
        _make_metrics_with_diversity("ModelB", "p_less", 0.73, 0.18),
        _make_metrics_with_diversity("ModelB", "temp_0.7", 0.70, 0.40),
    ]


def test_plot_pareto_scatter(tmp_path):
    out = tmp_path / "pareto.png"
    plot_pareto_scatter(_pareto_metrics_list(), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pareto_scatter_with_methods_filter(tmp_path):
    out = tmp_path / "pareto_filtered.png"
    plot_pareto_scatter(_pareto_metrics_list(), out, methods=["greedy", "p_less"])
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_method_heatmaps(tmp_path):
    out = tmp_path / "heatmaps.png"
    plot_method_heatmaps(_pareto_metrics_list(), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_method_heatmaps_with_missing_cells(tmp_path):
    """Heatmap handles missing (model, method) combinations gracefully."""
    metrics = [
        _make_metrics_with_diversity("ModelA", "greedy", 0.85, 0.0),
        _make_metrics_with_diversity("ModelA", "p_less", 0.82, 0.15),
        _make_metrics_with_diversity("ModelB", "p_less", 0.73, 0.18),
    ]
    out = tmp_path / "heatmaps_sparse.png"
    plot_method_heatmaps(metrics, out)
    assert out.exists()
    assert out.stat().st_size > 0
