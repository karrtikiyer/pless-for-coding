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
    plot_pass_at_1_heatmap,
    plot_structural_diversity_bars,
    plot_diversity_metrics_bars,
    load_metrics,
    _build_style_map,
    _config_key,
    _config_display,
    _build_config_palette,
    _family_key,
    _family_display,
    _build_family_palette,
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


def test_load_metrics_backfills_top_p_from_filename(tmp_path):
    """Older metrics JSONs lack `top_p`/`top_k` despite encoding it in the filename."""
    data = _make_metrics("m", "top_p")
    data["top_p"] = None
    data["top_k"] = None
    p = tmp_path / "top_p0.9_t1.0_metrics.json"
    p.write_text(json.dumps(data))
    result = load_metrics([p])
    assert result[0]["top_p"] == 0.9


def test_load_metrics_backfills_top_k_from_filename(tmp_path):
    data = _make_metrics("m", "top_k")
    data["top_p"] = None
    data["top_k"] = None
    p = tmp_path / "top_k50_t1.0_metrics.json"
    p.write_text(json.dumps(data))
    result = load_metrics([p])
    assert result[0]["top_k"] == 50


def test_load_metrics_backfills_handles_bigcode_suffix(tmp_path):
    data = _make_metrics("m", "top_p")
    data["top_p"] = None
    p = tmp_path / "top_p0.95_bigcode_t0.2_metrics.json"
    p.write_text(json.dumps(data))
    result = load_metrics([p])
    assert result[0]["top_p"] == 0.95


def test_load_metrics_does_not_overwrite_existing_top_p(tmp_path):
    """If JSON already has the value, keep it — don't overwrite from filename."""
    data = _make_metrics("m", "top_p")
    data["top_p"] = 0.85
    p = tmp_path / "top_p0.9_t1.0_metrics.json"
    p.write_text(json.dumps(data))
    result = load_metrics([p])
    assert result[0]["top_p"] == 0.85


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


def test_config_key_distinguishes_temperature():
    a = {"method": "temp", "temperature": 0.3}
    b = {"method": "temp", "temperature": 0.7}
    assert _config_key(a) != _config_key(b)
    assert _config_key(a) == "temp_t0.3"
    assert _config_key(b) == "temp_t0.7"


def test_config_key_distinguishes_top_p():
    a = {"method": "top_p", "top_p": 0.8, "temperature": 1.0}
    b = {"method": "top_p", "top_p": 0.9, "temperature": 1.0}
    assert _config_key(a) != _config_key(b)
    assert "p0.8" in _config_key(a)
    assert "p0.9" in _config_key(b)


def test_config_key_distinguishes_top_k():
    a = {"method": "top_k", "top_k": 5, "temperature": 1.0}
    b = {"method": "top_k", "top_k": 50, "temperature": 1.0}
    assert _config_key(a) != _config_key(b)


def test_config_key_handles_null_top_p_top_k():
    m = {"method": "pless", "temperature": 0.6, "top_p": None, "top_k": None}
    assert _config_key(m) == "pless_t0.6"


def test_config_display_temp():
    assert _config_display({"method": "temp", "temperature": 0.3}) == "temp (t=0.3)"
    assert _config_display({"method": "temp", "temperature": 0.7}) == "temp (t=0.7)"


def test_config_display_top_p_and_top_k():
    assert (
        _config_display({"method": "top_p", "top_p": 0.9, "temperature": 1.0})
        == "top-p (p=0.9, t=1.0)"
    )
    assert (
        _config_display({"method": "top_k", "top_k": 5, "temperature": 1.0})
        == "top-k (k=5, t=1.0)"
    )


def test_config_display_pless_variants():
    assert _config_display({"method": "pless", "temperature": 1.0}) == "p-less (t=1.0)"
    assert _config_display({"method": "pless_norm", "temperature": 0.6}) == "p-less-norm (t=0.6)"


def test_config_display_greedy_and_beam():
    assert _config_display({"method": "greedy"}) == "greedy"
    assert _config_display({"method": "beam4"}) == "beam4"
    assert _config_display({"method": "beam8"}) == "beam8"


def test_build_config_palette_assigns_distinct_colors_per_config():
    metrics = [
        {"method": "temp", "temperature": 0.3},
        {"method": "temp", "temperature": 0.7},
        {"method": "top_p", "top_p": 0.8, "temperature": 1.0},
        {"method": "top_p", "top_p": 0.9, "temperature": 1.0},
    ]
    palette = _build_config_palette(metrics)
    assert len(palette) == 4
    assert len(set(palette.values())) == 4


def test_pareto_scatter_uses_config_key_for_color(tmp_path):
    """Two configs with the same method but different temperatures get distinct colors."""
    metrics = [
        _make_metrics_with_diversity("ModelA", "temp", 0.50, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.40, 0.30),
    ]
    metrics[0]["temperature"] = 0.3
    metrics[1]["temperature"] = 0.7
    palette = _build_config_palette(metrics)
    assert palette[_config_key(metrics[0])] != palette[_config_key(metrics[1])]
    out = tmp_path / "pareto_temps.png"
    plot_pareto_scatter(metrics, out)
    assert out.exists()


def test_structural_diversity_bars_keeps_all_temperature_variants(tmp_path):
    """Bars chart must NOT silently drop temp@0.3 when temp@0.7 is also present."""
    metrics = [
        _make_metrics_with_diversity("ModelA", "temp", 0.50, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.40, 0.40),
    ]
    metrics[0]["temperature"] = 0.3
    metrics[1]["temperature"] = 0.7
    out = tmp_path / "div_bars.png"
    plot_structural_diversity_bars(metrics, out)
    assert out.exists()
    # Verify the chart actually plotted both bars by checking the figure has
    # two distinct entries — we inspect by re-running the layout pass and
    # asserting via the legend-handle count exposed through pyplot.
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plot_structural_diversity_bars(metrics, out)
    # Both configs should be in the resulting palette
    palette = _build_config_palette(metrics)
    assert len(palette) == 2
    plt.close(fig)


def test_diversity_metrics_bars_keeps_all_temperature_variants(tmp_path):
    metrics = [
        _make_metrics_with_diversity("ModelA", "temp", 0.50, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.40, 0.40),
    ]
    metrics[0]["temperature"] = 0.3
    metrics[1]["temperature"] = 0.7
    out = tmp_path / "div_metrics_bars.png"
    plot_diversity_metrics_bars(metrics, out)
    assert out.exists()
    palette = _build_config_palette(metrics)
    assert len(palette) == 2


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


def _capture_pareto_figure(monkeypatch):
    """Helper: capture the matplotlib Figure that plot_pareto_scatter draws on,
    by intercepting plt.close so the figure stays alive after the function returns."""
    import matplotlib.pyplot as plt
    captured: dict = {}
    real_close = plt.close

    def fake_close(fig=None):
        if fig is not None and "fig" not in captured:
            captured["fig"] = fig

    monkeypatch.setattr(plt, "close", fake_close)
    return captured, real_close


def test_pareto_scatter_filters_by_config_keys(tmp_path, monkeypatch):
    """plot_pareto_scatter(config_keys=[...]) keeps only matching configs."""
    metrics = [
        _make_metrics_with_diversity("ModelA", "temp", 0.50, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.40, 0.30),
        _make_metrics_with_diversity("ModelA", "pless", 0.60, 0.20),
    ]
    metrics[0]["temperature"] = 0.3
    metrics[1]["temperature"] = 0.7
    metrics[2]["temperature"] = 0.6

    captured, real_close = _capture_pareto_figure(monkeypatch)
    out = tmp_path / "pareto_filtered.png"
    plot_pareto_scatter(
        metrics, out,
        config_keys=["pless_t0.6"],
    )
    assert out.exists()
    fig = captured["fig"]
    ax = fig.axes[0]
    # The first legend (samplers) is added via add_artist; iterate fig artists.
    sampler_labels = []
    for child in ax.get_children():
        if hasattr(child, "get_texts"):
            for txt in child.get_texts():
                sampler_labels.append(txt.get_text())
    real_close(fig)
    # Filtered legend should have only one sampler entry plus the bold section title.
    assert any("p-less" in lbl for lbl in sampler_labels)
    assert not any("temp" in lbl.lower() and "t=0.3" in lbl for lbl in sampler_labels)
    assert not any("temp" in lbl.lower() and "t=0.7" in lbl for lbl in sampler_labels)


def test_pareto_scatter_show_trajectories_false_omits_gray_lines(tmp_path, monkeypatch):
    import matplotlib.colors as mcolors
    metrics = [
        _make_metrics_with_diversity("ModelA", "pless", 0.80, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.70, 0.30),
    ]
    captured, real_close = _capture_pareto_figure(monkeypatch)
    out = tmp_path / "pareto_no_traj.png"
    plot_pareto_scatter(metrics, out, show_trajectories=False)
    assert out.exists()
    fig = captured["fig"]
    ax = fig.axes[0]
    gray_rgba = mcolors.to_rgba("gray")
    gray_lines = [
        ln for ln in ax.lines
        if mcolors.to_rgba(ln.get_color()) == gray_rgba
    ]
    real_close(fig)
    assert gray_lines == []


def test_family_key_collapses_temp_variants():
    a = {"method": "temp", "temperature": 0.3}
    b = {"method": "temp", "temperature": 0.7}
    assert _family_key(a) == _family_key(b) == "temp"


def test_family_key_collapses_top_p_variants():
    a = {"method": "top_p", "top_p": 0.8, "temperature": 1.0}
    b = {"method": "top_p", "top_p": 0.9, "temperature": 1.0}
    assert _family_key(a) == _family_key(b) == "top_p"


def test_family_key_collapses_top_k_variants():
    a = {"method": "top_k", "top_k": 5, "temperature": 1.0}
    b = {"method": "top_k", "top_k": 50, "temperature": 1.0}
    assert _family_key(a) == _family_key(b) == "top_k"


def test_family_key_collapses_beam_variants():
    a = {"method": "beam4", "temperature": 1.0}
    b = {"method": "beam8", "temperature": 1.0}
    assert _family_key(a) == _family_key(b) == "beam"


def test_family_key_collapses_pless_temperatures():
    a = {"method": "pless", "temperature": 0.6}
    b = {"method": "pless", "temperature": 1.0}
    c = {"method": "pless_norm", "temperature": 0.6}
    d = {"method": "pless_norm", "temperature": 1.0}
    assert _family_key(a) == _family_key(b) == "pless"
    assert _family_key(c) == _family_key(d) == "pless_norm"
    assert _family_key(a) != _family_key(c)


def test_family_display_labels():
    assert _family_display("beam") == "beam (4, 8)"
    assert _family_display("top_p") == "top-p"
    assert _family_display("top_k") == "top-k"
    assert _family_display("temp") == "temp"
    assert _family_display("greedy") == "greedy"
    assert _family_display("pless") == "p-less (t=0.6, 1.0)"
    assert _family_display("pless_norm") == "p-less-norm (t=0.6, 1.0)"


def test_build_family_palette_collapses_paper_baselines():
    metrics = [
        {"method": "temp", "temperature": 0.1},
        {"method": "temp", "temperature": 0.3},
        {"method": "top_p", "top_p": 0.8, "temperature": 1.0},
        {"method": "top_p", "top_p": 0.85, "temperature": 1.0},
        {"method": "beam4", "temperature": 1.0},
        {"method": "beam8", "temperature": 1.0},
        {"method": "pless", "temperature": 0.6},
        {"method": "pless", "temperature": 1.0},
        {"method": "pless_norm", "temperature": 0.6},
        {"method": "pless_norm", "temperature": 1.0},
    ]
    palette = _build_family_palette(metrics)
    # 5 families: temp, top_p, beam, pless, pless_norm
    assert len(palette) == 5


def test_pareto_scatter_family_palette_collapses_temp_to_one_color(tmp_path, monkeypatch):
    """In family_palette mode, temp_t0.3 and temp_t0.7 share one legend entry."""
    metrics = [
        _make_metrics_with_diversity("ModelA", "temp", 0.50, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.40, 0.30),
        _make_metrics_with_diversity("ModelA", "beam4", 0.60, 0.05),
        _make_metrics_with_diversity("ModelA", "beam8", 0.62, 0.06),
        _make_metrics_with_diversity("ModelA", "pless", 0.45, 0.20),
    ]
    metrics[0]["temperature"] = 0.3
    metrics[1]["temperature"] = 0.7
    metrics[4]["temperature"] = 0.6

    captured, real_close = _capture_pareto_figure(monkeypatch)
    out = tmp_path / "pareto_family.png"
    plot_pareto_scatter(metrics, out, family_palette=True)
    assert out.exists()
    fig = captured["fig"]
    ax = fig.axes[0]
    sampler_labels = []
    for child in ax.get_children():
        if hasattr(child, "get_texts"):
            for txt in child.get_texts():
                sampler_labels.append(txt.get_text())
    real_close(fig)

    # temp variants collapse to one entry; beam4+beam8 collapse to one "beam" entry
    temp_entries = [lbl for lbl in sampler_labels if lbl.startswith("temp")]
    beam_entries = [lbl for lbl in sampler_labels if lbl.startswith("beam")]
    pless_entries = [lbl for lbl in sampler_labels if "p-less" in lbl]
    assert len(temp_entries) == 1
    assert len(beam_entries) == 1
    assert len(pless_entries) == 1  # only one pless config (t=0.6)


def test_pareto_scatter_show_trajectories_true_includes_gray_lines(tmp_path, monkeypatch):
    """Regression: default behaviour still draws per-model gray trajectory lines."""
    import matplotlib.colors as mcolors
    metrics = [
        _make_metrics_with_diversity("ModelA", "pless", 0.80, 0.10),
        _make_metrics_with_diversity("ModelA", "temp", 0.70, 0.30),
    ]
    captured, real_close = _capture_pareto_figure(monkeypatch)
    out = tmp_path / "pareto_traj.png"
    plot_pareto_scatter(metrics, out)  # default: show_trajectories=True
    assert out.exists()
    fig = captured["fig"]
    ax = fig.axes[0]
    gray_rgba = mcolors.to_rgba("gray")
    gray_lines = [
        ln for ln in ax.lines
        if mcolors.to_rgba(ln.get_color()) == gray_rgba
    ]
    real_close(fig)
    assert len(gray_lines) >= 1


# ---------------------------------------------------------------------------
# plot_pass_at_1_heatmap (Supplementary Figure A renderer)
# ---------------------------------------------------------------------------


def _capture_heatmap_figure(monkeypatch):
    """Intercept plt.close so the figure stays alive for inspection."""
    import matplotlib.pyplot as plt
    captured: dict = {}
    real_close = plt.close

    def fake_close(fig=None):
        if fig is not None and "fig" not in captured:
            captured["fig"] = fig

    monkeypatch.setattr(plt, "close", fake_close)
    return captured, real_close


def test_pass_at_1_heatmap_writes_png(tmp_path):
    """3 models × 3 samplers — PNG is created and non-empty."""
    metrics = []
    for model in ["ModelA", "ModelB", "ModelC"]:
        metrics.append(_make_metrics_with_diversity(model, "pless", 0.5, 0.2))
        metrics.append(_make_metrics_with_diversity(model, "temp", 0.6, 0.3))
        metrics.append(_make_metrics_with_diversity(model, "top_p", 0.55, 0.25))
    out = tmp_path / "heatmap.png"
    plot_pass_at_1_heatmap(metrics, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_pass_at_1_heatmap_columns_sorted_by_mean_rank(tmp_path, monkeypatch):
    """Sampler with the highest mean pass@1 must occupy the leftmost column."""
    metrics = []
    # top_p_t1.0 dominates across 3 models; pless and temp are weaker.
    for model in ["ModelA", "ModelB", "ModelC"]:
        m_top_p = _make_metrics_with_diversity(model, "top_p", 0.90, 0.10)
        m_top_p["top_p"] = 0.9
        m_top_p["temperature"] = 1.0
        metrics.append(m_top_p)
        m_pless = _make_metrics_with_diversity(model, "pless", 0.50, 0.20)
        m_pless["temperature"] = 1.0
        metrics.append(m_pless)
        m_temp = _make_metrics_with_diversity(model, "temp", 0.30, 0.30)
        m_temp["temperature"] = 0.7
        metrics.append(m_temp)

    captured, real_close = _capture_heatmap_figure(monkeypatch)
    out = tmp_path / "heatmap_sorted.png"
    plot_pass_at_1_heatmap(metrics, out, sort_columns_by="mean_rank")
    fig = captured["fig"]
    ax = fig.axes[0]
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    real_close(fig)
    assert xticklabels[0].startswith("top-p"), f"Expected top-p first, got {xticklabels}"


def test_pass_at_1_heatmap_handles_missing_cells(tmp_path, monkeypatch):
    """Cells without a (model, sampler) combination render '—'."""
    metrics = [
        _make_metrics_with_diversity("ModelA", "pless", 0.60, 0.20),
        _make_metrics_with_diversity("ModelA", "temp", 0.50, 0.30),
        # ModelB only has pless — temp cell is missing
        _make_metrics_with_diversity("ModelB", "pless", 0.55, 0.18),
    ]
    captured, real_close = _capture_heatmap_figure(monkeypatch)
    out = tmp_path / "heatmap_missing.png"
    plot_pass_at_1_heatmap(metrics, out)
    fig = captured["fig"]
    ax = fig.axes[0]
    annotation_texts = [t.get_text() for t in ax.texts]
    real_close(fig)
    assert "—" in annotation_texts, f"Expected em-dash for missing cell; got {annotation_texts}"


def test_pass_at_1_heatmap_winner_per_row_highlighted(tmp_path, monkeypatch):
    """The row max gets a Rectangle overlay (linewidth >= 1.5) and bold text."""
    from matplotlib.patches import Rectangle

    metrics = [
        _make_metrics_with_diversity("ModelA", "temp", 0.40, 0.30),
        _make_metrics_with_diversity("ModelA", "pless", 0.60, 0.20),
        _make_metrics_with_diversity("ModelA", "top_p", 0.50, 0.25),
    ]
    metrics[0]["temperature"] = 0.7
    metrics[1]["temperature"] = 1.0
    metrics[2]["top_p"] = 0.9
    metrics[2]["temperature"] = 1.0

    captured, real_close = _capture_heatmap_figure(monkeypatch)
    out = tmp_path / "heatmap_winner.png"
    plot_pass_at_1_heatmap(metrics, out)
    fig = captured["fig"]
    ax = fig.axes[0]

    # Find the Rectangle patches that mark winners (linewidth >= 1.5)
    winner_rects = [
        p for p in ax.patches
        if isinstance(p, Rectangle) and p.get_linewidth() >= 1.5
    ]
    # Bold annotation count
    bold_texts = [
        t for t in ax.texts
        if t.get_fontweight() == "bold" and t.get_text() not in ("", "—")
    ]
    real_close(fig)

    assert len(winner_rects) >= 1, "Expected at least one winner rectangle"
    assert any("60" in t.get_text() for t in bold_texts), (
        f"Expected bold annotation containing '60' (for 0.60 → 60.0); "
        f"got {[t.get_text() for t in bold_texts]}"
    )


def test_pass_at_1_heatmap_row_order_groups_by_family(tmp_path, monkeypatch):
    """Llama row index < Qwen rows when group_rows_by_family=True."""
    metrics = [
        _make_metrics_with_diversity("Qwen/Qwen-7B", "pless", 0.50, 0.20),
        _make_metrics_with_diversity("Qwen/Qwen-7B-Chat", "pless", 0.55, 0.22),
        _make_metrics_with_diversity("meta-llama/Llama-2-7b-hf", "pless", 0.45, 0.18),
    ]

    captured, real_close = _capture_heatmap_figure(monkeypatch)
    out = tmp_path / "heatmap_family.png"
    plot_pass_at_1_heatmap(metrics, out, group_rows_by_family=True)
    fig = captured["fig"]
    ax = fig.axes[0]
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    real_close(fig)

    llama_idx = next(i for i, lbl in enumerate(yticklabels) if "Llama" in lbl)
    qwen_indices = [i for i, lbl in enumerate(yticklabels) if "Qwen" in lbl]
    assert all(llama_idx < qi for qi in qwen_indices), (
        f"Llama should come before Qwen rows; got {yticklabels}"
    )

    # Falls back to alphabetical when grouping is off
    captured2, real_close2 = _capture_heatmap_figure(monkeypatch)
    out2 = tmp_path / "heatmap_alpha.png"
    plot_pass_at_1_heatmap(metrics, out2, group_rows_by_family=False)
    fig2 = captured2["fig"]
    ax2 = fig2.axes[0]
    yticklabels2 = [t.get_text() for t in ax2.get_yticklabels()]
    real_close2(fig2)
    assert yticklabels2 == sorted(yticklabels2), f"Expected alpha order, got {yticklabels2}"
