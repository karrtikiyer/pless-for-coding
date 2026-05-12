"""Generate CSV, markdown report, and visualizations for the temperature sweep.

Loads all metrics JSONs from temprature_results/*/metrics/ and produces:
  - temperature_sweep_summary.csv
  - temperature_sweep_report.md
  - figures/ directory with plots

Usage:
    python -m bench.eval.report_temperature_sweep
"""

import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path("results/pless_human_eval_results/temprature_results")
FIGURES_DIR = RESULTS_ROOT / "analysis" / "figures"

# ---------------------------------------------------------------------------
# Method colours and styles for temperature sweep plots
# ---------------------------------------------------------------------------
_METHOD_COLORS = {
    "temp": "#2F855A",       # green — standard temperature
    "pless": "#6B46C1",      # purple — p-less
    "pless_norm": "#B7791F",  # gold — p-less normalized
}

_METHOD_MARKERS = {
    "temp": "s",
    "pless": "o",
    "pless_norm": "X",
}

_METHOD_LINESTYLES = {
    "temp": "-",
    "pless": "-",
    "pless_norm": "-.",
}

_MODEL_SHORT_NAMES = {
    "Qwen/Qwen2.5-Coder-7B": "Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder-30B",
    "codellama/CodeLlama-7b-hf": "CodeLlama-7b",
    "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama-7b-Instruct",
    "mistralai/Codestral-22B-v0.1": "Codestral-22B",
}


_DIVERSITY_LABELS = {
    "codebleu_diversity": "CodeBLEU Diversity",
    "ngram_match_diversity": "N-gram Diversity",
    "weighted_ngram_match_diversity": "Weighted N-gram Diversity",
    "syntax_match_diversity": "Syntax Match Diversity",
    "dataflow_match_diversity": "Dataflow Diversity",
}

_SUBCOMPONENT_PARETOS = [
    ("ngram_match_diversity",          "pareto_ngram_diversity.png"),
    ("weighted_ngram_match_diversity", "pareto_weighted_ngram_diversity.png"),
    ("syntax_match_diversity",         "pareto_syntax_diversity.png"),
    ("dataflow_match_diversity",       "pareto_dataflow_diversity.png"),
]

_TEMP_MARKERS = {
    0.7: "o", 1.0: "s", 1.5: "^", 2.0: "v", 2.5: "D", 3.0: "P",
}


def _short_model(model: str) -> str:
    return _MODEL_SHORT_NAMES.get(model, model.split("/")[-1])


def _method_base(method: str) -> str:
    """Normalize method name: 'pless' / 'pless_norm' / 'temp'."""
    if method == "pless_norm":
        return "pless_norm"
    if method == "pless":
        return "pless"
    if method.startswith("temp"):
        return "temp"
    return method


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics(root: Path) -> list[dict]:
    """Load all metrics JSON files from root/*/metrics/."""
    metrics = []
    for p in sorted(root.glob("*/metrics/*_metrics.json")):
        with open(p) as f:
            metrics.append(json.load(f))
    return metrics


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(metrics_list: list[dict], output_path: Path) -> None:
    fieldnames = [
        "Model", "Method", "Temperature",
        "pass@1", "pass@3", "pass@5", "pass@10",
        "cover@0.5", "cover@0.7",
        "structural_diversity",
        "codebleu_diversity",
        "syntax_match_diversity",
        "dataflow_match_diversity",
        "ngram_match_diversity",
        "weighted_ngram_match_diversity",
    ]
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": _short_model(m["model"]),
            "Method": m["method"],
            "Temperature": m["temperature"],
            "pass@1": f"{m['pass_at_k'].get('1', 0) * 100:.1f}",
            "pass@3": f"{m['pass_at_k'].get('3', 0) * 100:.1f}",
            "pass@5": f"{m['pass_at_k'].get('5', 0) * 100:.1f}",
            "pass@10": f"{m['pass_at_k'].get('10', 0) * 100:.1f}",
            "cover@0.5": f"{m['cover_at_t'].get('0.5', 0):.1f}",
            "cover@0.7": f"{m['cover_at_t'].get('0.7', 0):.1f}",
            "structural_diversity": f"{m.get('structural_diversity', 0):.4f}",
            "codebleu_diversity": f"{m.get('codebleu_diversity', 0):.4f}",
            "syntax_match_diversity": f"{m.get('syntax_match_diversity', 0):.4f}",
            "dataflow_match_diversity": f"{m.get('dataflow_match_diversity', 0):.4f}",
            "ngram_match_diversity": f"{m.get('ngram_match_diversity', 0):.4f}",
            "weighted_ngram_match_diversity": f"{m.get('weighted_ngram_match_diversity', 0):.4f}",
        })

    # Sort by model, method, temperature
    rows.sort(key=lambda r: (r["Model"], r["Method"], float(r["Temperature"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(metrics_list: list[dict], output_path: Path) -> None:
    models = sorted(set(_short_model(m["model"]) for m in metrics_list))
    methods = sorted(set(m["method"] for m in metrics_list))

    # Index for quick lookup
    idx = {}
    for m in metrics_list:
        key = (_short_model(m["model"]), m["method"], m["temperature"])
        idx[key] = m

    lines = []
    lines.append("# HumanEval Temperature Sweep Results\n")
    lines.append(f"**Models:** {', '.join(models)}  ")
    lines.append(f"**Methods:** {', '.join(methods)}  ")
    lines.append(f"**Temperatures:** 0.7, 1.0, 1.5, 2.0, 2.5, 3.0  ")
    lines.append(f"**Samples per task:** 10\n")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Model | Method | Temp | pass@1 | pass@5 | pass@10 | cover@0.5 | diversity |")
    lines.append("|-------|--------|------|--------|--------|---------|-----------|-----------|")

    sorted_metrics = sorted(
        metrics_list,
        key=lambda m: (_short_model(m["model"]), _method_base(m["method"]), m["temperature"]),
    )
    for m in sorted_metrics:
        p1 = m["pass_at_k"].get("1", 0) * 100
        p5 = m["pass_at_k"].get("5", 0) * 100
        p10 = m["pass_at_k"].get("10", 0) * 100
        c05 = m["cover_at_t"].get("0.5", 0)
        sd = m.get("structural_diversity", 0)
        lines.append(
            f"| {_short_model(m['model'])} | {m['method']} | {m['temperature']} "
            f"| {p1:.1f}% | {p5:.1f}% | {p10:.1f}% | {c05:.1f}% | {sd:.4f} |"
        )

    # Per-model analysis
    lines.append("\n## Per-Model Analysis\n")
    for model in models:
        lines.append(f"### {model}\n")
        lines.append("| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |")
        lines.append("|--------|-------|-------|-------|-------|-------|-------|")

        for method in methods:
            row = f"| {method}"
            for temp in [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
                key = (model, method, temp)
                if key in idx:
                    p1 = idx[key]["pass_at_k"].get("1", 0) * 100
                    row += f" | {p1:.1f}%"
                else:
                    row += " | —"
            row += " |"
            lines.append(row)
        lines.append("")

    # Key findings
    lines.append("## Key Findings\n")

    # Find best method per model at T=2.0 (highest temp with non-trivial results)
    lines.append("### Best Method at T=2.0\n")
    for model in models:
        best_method, best_p1 = None, -1
        for method in methods:
            key = (model, method, 2.0)
            if key in idx:
                p1 = idx[key]["pass_at_k"].get("1", 0) * 100
                if p1 > best_p1:
                    best_p1 = p1
                    best_method = method
        if best_method:
            lines.append(f"- **{model}**: **{best_method}** (pass@1={best_p1:.1f}%)")

    # Temperature robustness for pless methods: drop from T=0.7 to T=2.0
    lines.append("\n### Temperature Robustness (pass@1 drop from T=0.7 to T=2.0)\n")
    pless_methods = [m for m in methods if m in ("pless", "pless_norm")]
    for model in models:
        for method in pless_methods:
            key07 = (model, method, 0.7)
            key20 = (model, method, 2.0)
            if key07 in idx and key20 in idx:
                p1_07 = idx[key07]["pass_at_k"].get("1", 0) * 100
                p1_20 = idx[key20]["pass_at_k"].get("1", 0) * 100
                drop = p1_07 - p1_20
                lines.append(f"- {model} / {method}: {p1_07:.1f}% → {p1_20:.1f}% (Δ={drop:+.1f}pp)")

    # Temp baseline comparison at T=0.7 and T=1.0
    lines.append("\n### Standard Temperature Baselines (T=0.7 and T=1.0)\n")
    for model in models:
        for temp in [0.7, 1.0]:
            key_temp = (model, "temp", temp)
            if key_temp not in idx:
                continue
            p1_temp = idx[key_temp]["pass_at_k"].get("1", 0) * 100
            best_pless = None
            for method in pless_methods:
                key = (model, method, temp)
                if key in idx:
                    p1 = idx[key]["pass_at_k"].get("1", 0) * 100
                    if best_pless is None or p1 > best_pless[1]:
                        best_pless = (method, p1)
            if best_pless:
                diff = best_pless[1] - p1_temp
                lines.append(
                    f"- {model} T={temp}: temp={p1_temp:.1f}%, "
                    f"best pless ({best_pless[0]})={best_pless[1]:.1f}% (Δ={diff:+.1f}pp)"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _group_by_model(metrics_list: list[dict]) -> dict[str, list[dict]]:
    """Group metrics by model, preserving order."""
    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    by_model = {model: [] for model in models_order}
    for m in metrics_list:
        by_model[m["model"]].append(m)
    return by_model


def plot_pass_at_k_vs_temperature(metrics_list: list[dict], output_dir: Path) -> None:
    """Line plot: pass@1 vs temperature, one subplot per model, one line per method."""
    by_model = _group_by_model(metrics_list)
    n_models = len(by_model)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for col, (model, model_metrics) in enumerate(by_model.items()):
        ax = axes[col]

        # Group by method base
        by_method: dict[str, list[dict]] = {}
        for m in model_metrics:
            mb = _method_base(m["method"])
            by_method.setdefault(mb, []).append(m)

        for method, mlist in sorted(by_method.items()):
            mlist.sort(key=lambda m: m["temperature"])
            temps = [m["temperature"] for m in mlist]
            p1s = [m["pass_at_k"].get("1", 0) * 100 for m in mlist]

            ax.plot(
                temps, p1s,
                label=method,
                color=_METHOD_COLORS.get(method, "#333"),
                marker=_METHOD_MARKERS.get(method, "x"),
                linestyle=_METHOD_LINESTYLES.get(method, "-"),
                linewidth=2, markersize=7,
            )

        ax.set_title(_short_model(model), fontsize=11)
        ax.set_xlabel("Temperature")
        ax.set_xticks([0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("pass@1 (%)")

    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle("HumanEval: pass@1 vs Temperature", fontsize=13)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pass_at_1_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_structural_diversity_vs_temperature(metrics_list: list[dict], output_dir: Path) -> None:
    """Line plot: structural diversity vs temperature, one subplot per model."""
    by_model = _group_by_model(metrics_list)
    n_models = len(by_model)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for col, (model, model_metrics) in enumerate(by_model.items()):
        ax = axes[col]

        by_method: dict[str, list[dict]] = {}
        for m in model_metrics:
            mb = _method_base(m["method"])
            by_method.setdefault(mb, []).append(m)

        for method, mlist in sorted(by_method.items()):
            mlist.sort(key=lambda m: m["temperature"])
            temps = [m["temperature"] for m in mlist]
            divs = [m.get("structural_diversity", 0) for m in mlist]

            ax.plot(
                temps, divs,
                label=method,
                color=_METHOD_COLORS.get(method, "#333"),
                marker=_METHOD_MARKERS.get(method, "x"),
                linestyle=_METHOD_LINESTYLES.get(method, "-"),
                linewidth=2, markersize=7,
            )

        ax.set_title(_short_model(model), fontsize=11)
        ax.set_xlabel("Temperature")
        ax.set_xticks([0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("Structural Diversity")

    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle("HumanEval: Structural Diversity vs Temperature", fontsize=13)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "structural_diversity_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_heatmaps(metrics_list: list[dict], output_dir: Path) -> None:
    """Heatmap: models × temperatures, one heatmap per method, cell color = pass@1."""
    methods = sorted(set(_method_base(m["method"]) for m in metrics_list))
    models = sorted(set(m["model"] for m in metrics_list))
    temps = sorted(set(m["temperature"] for m in metrics_list))

    # Index
    idx = {}
    for m in metrics_list:
        idx[(_method_base(m["method"]), m["model"], m["temperature"])] = m

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, max(3, len(models) * 0.8 + 1)),
                             squeeze=False)
    axes = axes[0]

    for col, method in enumerate(methods):
        ax = axes[col]
        data = np.zeros((len(models), len(temps)))

        for i, model in enumerate(models):
            for j, temp in enumerate(temps):
                key = (method, model, temp)
                if key in idx:
                    data[i, j] = idx[key]["pass_at_k"].get("1", 0) * 100

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=80)
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([str(t) for t in temps], fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_short_model(m) for m in models], fontsize=9)
        ax.set_xlabel("Temperature")
        ax.set_title(method, fontsize=11)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(temps)):
                val = data[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.suptitle("HumanEval: pass@1 (%) by Model × Temperature", fontsize=13)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="pass@1 (%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pass_at_1_heatmap.png", dpi=150)
    plt.close(fig)


def plot_pass_at_k_curves_by_temperature(metrics_list: list[dict], output_dir: Path) -> None:
    """Faceted pass@k curves: one row per model, one column per method.
    Lines colored by temperature."""
    by_model = _group_by_model(metrics_list)
    methods = sorted(set(_method_base(m["method"]) for m in metrics_list))
    n_models = len(by_model)
    n_methods = len(methods)

    # Temperature color map
    temps_all = sorted(set(m["temperature"] for m in metrics_list))
    cmap = plt.cm.coolwarm
    temp_colors = {t: cmap(i / max(len(temps_all) - 1, 1)) for i, t in enumerate(temps_all)}

    fig, axes = plt.subplots(n_models, n_methods,
                             figsize=(5 * n_methods, 4 * n_models),
                             sharey=True, squeeze=False)

    for row, (model, model_metrics) in enumerate(by_model.items()):
        by_method: dict[str, list[dict]] = {}
        for m in model_metrics:
            mb = _method_base(m["method"])
            by_method.setdefault(mb, []).append(m)

        for col, method in enumerate(methods):
            ax = axes[row, col]
            mlist = by_method.get(method, [])
            mlist.sort(key=lambda m: m["temperature"])

            for m in mlist:
                ks = sorted(m["pass_at_k"], key=lambda x: int(x))
                ax.plot(
                    [int(k) for k in ks],
                    [m["pass_at_k"][k] * 100 for k in ks],
                    label=f"T={m['temperature']}",
                    color=temp_colors[m["temperature"]],
                    marker="o", linewidth=1.5, markersize=5,
                )

            ax.set_xticks([1, 3, 5, 10])
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(method, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{_short_model(model)}\n\npass@k (%)", fontsize=9)
            if row == n_models - 1:
                ax.set_xlabel("k")

    # Legend from the last subplot
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(temps_all),
               fontsize=8, frameon=True)

    fig.suptitle("HumanEval: pass@k Curves by Temperature", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pass_at_k_by_temperature.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-model plots
# ---------------------------------------------------------------------------

def plot_pareto_scatter(
    model_metrics: list[dict], model_name: str,
    output_path: Path, diversity_key: str = "codebleu_diversity",
) -> None:
    """Pareto scatter: pass@1 vs diversity for a single model's configs."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect points
    points: list[tuple[float, float, str, str, str]] = []
    for m in model_metrics:
        p1 = m["pass_at_k"].get("1", 0) * 100
        div = m.get(diversity_key, 0)
        # Skip 0/0 garbage points (extreme temps producing nothing)
        if p1 < 0.05 and div < 0.001:
            continue
        method = _method_base(m["method"])
        temp = m["temperature"]
        label = f"{method} T={temp}"
        color = _METHOD_COLORS.get(method, "#888888")
        marker = _TEMP_MARKERS.get(temp, "x")
        points.append((p1, div, label, color, marker))

    if not points:
        plt.close(fig)
        return

    for p1, div, _lbl, color, marker in points:
        ax.scatter(p1, div, s=140, color=color, marker=marker,
                   edgecolors="black", linewidth=0.6, zorder=3)

    # Pareto frontier
    pts_sorted = sorted([(p1, div) for p1, div, *_ in points], key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    max_div = -1
    for x, y in pts_sorted:
        if y > max_div:
            frontier_x.append(x)
            frontier_y.append(y)
            max_div = y
    if len(frontier_x) > 1:
        ax.plot(frontier_x, frontier_y, "--", color="gray", alpha=0.4, linewidth=1)

    # De-overlap labels
    label_entries = sorted(
        [(p1, div, lbl) for p1, div, lbl, _c, _m in points],
        key=lambda t: t[1],
    )
    all_x = [p1 for p1, *_ in points]
    all_y = [div for _, div, *_ in points]
    y_min_data, y_max_data = min(all_y), max(all_y)
    x_min_data, x_max_data = min(all_x), max(all_x)
    y_range = y_max_data - y_min_data if len(all_y) > 1 else 1.0
    x_range = x_max_data - x_min_data if len(all_x) > 1 else 1.0
    min_gap = y_range * 0.045
    x_proximity = max(x_range * 0.15, 2.0)

    label_ys = [py for _px, py, _lbl in label_entries]
    label_xs = [px for px, _py, _lbl in label_entries]

    for _iteration in range(50):
        moved = False
        for i in range(len(label_ys)):
            for j in range(i + 1, len(label_ys)):
                if abs(label_xs[i] - label_xs[j]) > x_proximity:
                    continue
                dy = label_ys[j] - label_ys[i]
                if abs(dy) < min_gap:
                    push = (min_gap - abs(dy)) / 2 + 0.001
                    label_ys[i] -= push
                    label_ys[j] += push
                    moved = True
        if not moved:
            break

    # Clamp labels within data bounds (with padding) so they don't
    # drift far outside the visible area, then re-deoverlap briefly
    y_clamp_lo = y_min_data - y_range * 0.06
    y_clamp_hi = y_max_data + y_range * 0.08
    for i in range(len(label_ys)):
        label_ys[i] = max(y_clamp_lo, min(y_clamp_hi, label_ys[i]))
    for _iteration in range(20):
        moved = False
        for i in range(len(label_ys)):
            for j in range(i + 1, len(label_ys)):
                if abs(label_xs[i] - label_xs[j]) > x_proximity:
                    continue
                dy = label_ys[j] - label_ys[i]
                if abs(dy) < min_gap:
                    push = (min_gap - abs(dy)) / 2 + 0.001
                    if label_ys[i] - push >= y_clamp_lo:
                        label_ys[i] -= push
                    if label_ys[j] + push <= y_clamp_hi:
                        label_ys[j] += push
                    moved = True
        if not moved:
            break

    x_offset = max(x_range * 0.02, 1.0)
    for idx, (px, py, lbl) in enumerate(label_entries):
        ly = label_ys[idx]
        ax.annotate(lbl, (px, py), xytext=(px + x_offset, ly),
                    textcoords="data", fontsize=5.5, alpha=0.7,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                    alpha=0.3, linewidth=0.5)
                    if abs(ly - py) > min_gap * 0.4 else None)

    div_label = _DIVERSITY_LABELS.get(diversity_key, diversity_key)
    ax.set_xlabel("pass@1 (%)", fontsize=11)
    ax.set_ylabel(div_label, fontsize=11)
    ax.set_title(f"HumanEval: Correctness vs {div_label} — {model_name}", fontsize=12)
    ax.grid(alpha=0.3)

    # Legend: method colors + temperature markers
    legend_items = [
        Patch(color=_METHOD_COLORS["pless"], label="P-less"),
        Patch(color=_METHOD_COLORS["pless_norm"], label="P-less norm"),
        Patch(color=_METHOD_COLORS["temp"], label="Temperature"),
        Patch(color="none", label=""),
    ]
    for temp, mk in _TEMP_MARKERS.items():
        legend_items.append(
            Line2D([0], [0], marker=mk, color="gray", linestyle="None",
                   markersize=7, label=f"T={temp}")
        )
    # Auto-select legend corner with fewest data points
    mid_x = (x_min_data + x_max_data) / 2
    mid_y = (y_min_data + y_max_data) / 2
    corner_counts = {
        "upper left": sum(1 for p1, div, *_ in points if p1 < mid_x and div > mid_y),
        "upper right": sum(1 for p1, div, *_ in points if p1 >= mid_x and div > mid_y),
        "lower left": sum(1 for p1, div, *_ in points if p1 < mid_x and div <= mid_y),
        "lower right": sum(1 for p1, div, *_ in points if p1 >= mid_x and div <= mid_y),
    }
    best_loc = min(corner_counts, key=corner_counts.get)
    ax.legend(handles=legend_items, loc=best_loc, fontsize=8,
              frameon=True, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}")


def plot_pass_at_1_bars(
    model_metrics: list[dict], model_name: str, output_path: Path,
) -> None:
    """Horizontal bar chart: all configs for one model ranked by pass@1."""
    from matplotlib.patches import Patch

    sorted_metrics = sorted(model_metrics, key=lambda m: m["pass_at_k"].get("1", 0))
    labels = [f"{_method_base(m['method'])} T={m['temperature']}" for m in sorted_metrics]
    values = [m["pass_at_k"].get("1", 0) * 100 for m in sorted_metrics]
    colors = [_METHOD_COLORS.get(_method_base(m["method"]), "#888888") for m in sorted_metrics]

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.4)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("pass@1 (%)", fontsize=11)
    ax.set_title(f"HumanEval pass@1: {model_name}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7)

    legend_items = [
        Patch(color=_METHOD_COLORS["pless"], label="P-less"),
        Patch(color=_METHOD_COLORS["pless_norm"], label="P-less norm"),
        Patch(color=_METHOD_COLORS["temp"], label="Temperature"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics_list = load_all_metrics(RESULTS_ROOT)
    if not metrics_list:
        print(f"No metrics JSONs found under {RESULTS_ROOT}/*/metrics/")
        print("Run `python -m bench.eval.eval_temperature_sweep` first.")
        return

    print(f"Loaded {len(metrics_list)} metrics files")

    # CSV
    csv_path = RESULTS_ROOT / "analysis" / "temperature_sweep_summary.csv"
    write_csv(metrics_list, csv_path)
    print(f"Wrote {csv_path}")

    # Markdown report
    md_path = RESULTS_ROOT / "analysis" / "temperature_sweep_report.md"
    write_markdown(metrics_list, md_path)
    print(f"Wrote {md_path}")

    # Visualizations
    print("Generating figures...")
    plot_pass_at_k_vs_temperature(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'pass_at_1_vs_temperature.png'}")

    plot_structural_diversity_vs_temperature(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'structural_diversity_vs_temperature.png'}")

    plot_heatmaps(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'pass_at_1_heatmap.png'}")

    plot_pass_at_k_curves_by_temperature(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'pass_at_k_by_temperature.png'}")

    # Per-model analysis: pareto scatters + bar charts
    print("\nGenerating per-model plots...")
    by_model = _group_by_model(metrics_list)
    for model, model_metrics in by_model.items():
        model_dir = model.replace("/", "--")
        model_figures = RESULTS_ROOT / "analysis" / model_dir / "figures"
        short = _short_model(model)
        print(f"\n  {short}:")

        plot_pass_at_1_bars(model_metrics, short,
                            model_figures / "pass_at_1_comparison.png")

        plot_pareto_scatter(model_metrics, short,
                            model_figures / "pareto_correctness_diversity.png")
        for div_key, filename in _SUBCOMPONENT_PARETOS:
            plot_pareto_scatter(model_metrics, short,
                                model_figures / filename, diversity_key=div_key)

    print("\nDone!")


if __name__ == "__main__":
    main()
