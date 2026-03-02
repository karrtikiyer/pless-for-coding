"""Generate visualizations from metrics JSON files."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Qualitative model colours (up to 8 models)
# ---------------------------------------------------------------------------
_MODEL_COLORS = [
    "#2B6CB0",  # blue
    "#C05621",  # orange
    "#2F855A",  # green
    "#9B2C2C",  # red
    "#6B46C1",  # purple
    "#B7791F",  # gold
    "#2C7A7B",  # teal
    "#702459",  # pink
]

# ---------------------------------------------------------------------------
# Per-method linestyle + marker
# ---------------------------------------------------------------------------
_METHOD_STYLES: dict[str, dict] = {
    "greedy":     dict(linestyle="-",  marker="o"),
    "temp_0.2":   dict(linestyle="-",  marker="^"),
    "temp_0.7":   dict(linestyle="-",  marker="s"),
    "top_p_0.95": dict(linestyle="-",  marker="D"),
    "p_less":     dict(linestyle="-",  marker="o"),
    "p_less_norm": dict(linestyle="-.", marker="X"),
    # legacy aliases used in MBPP results
    "pless":      dict(linestyle="-",  marker="o"),
    "pless_norm": dict(linestyle="-.", marker="X"),
}


def load_metrics(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def _label_for(m: dict) -> str:
    model = m["model"].split("/")[-1] if "/" in m["model"] else m["model"]
    return f"{model} ({m['method']})"


def _build_style_map(metrics_list: list[dict]) -> dict[tuple[str, str], dict]:
    """Pre-compute a colour/style dict keyed by (model, method)."""
    models = list(dict.fromkeys(m["model"] for m in metrics_list))
    model_color = {name: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, name in enumerate(models)}

    style_map: dict[tuple[str, str], dict] = {}
    for m in metrics_list:
        key = (m["model"], m["method"])
        if key in style_map:
            continue
        method_style = _METHOD_STYLES.get(m["method"], dict(linestyle="-", marker="x"))
        style_map[key] = dict(
            color=model_color[m["model"]],
            linestyle=method_style["linestyle"],
            marker=method_style["marker"],
            linewidth=2,
            markersize=6,
        )
    return style_map


def _style_for(m: dict, style_map: dict[tuple[str, str], dict] | None = None) -> dict:
    """Return color, linestyle, and marker for a config.

    If *style_map* is provided (from ``_build_style_map``), look up the entry.
    Otherwise fall back to the legacy 2-model heuristic for backward compat.
    """
    if style_map is not None:
        key = (m["model"], m["method"])
        if key in style_map:
            return style_map[key]

    # Legacy fallback (2-model MBPP case)
    is_instruct = "instruct" in m["model"].lower() or "coder" in m["model"].lower()
    is_norm = m["method"] in ("pless_norm", "p_less_norm")
    color = "#C05621" if is_instruct else "#2B6CB0"
    linestyle = "--" if is_norm else "-"
    marker = "s" if is_norm else "o"
    if is_instruct and is_norm:
        color = "#ED8936"
    elif not is_instruct and is_norm:
        color = "#63B3ED"
    return dict(color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6)


def plot_aggregate_lines(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Line plot with 3 subplots: pass@k, cover@t, cover@t (distinct).

    Each subplot has one line per model+method configuration.
    """
    style_map = _build_style_map(metrics_list)
    n_lines = len(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for m in metrics_list:
        label = _label_for(m)
        style = _style_for(m, style_map)

        # pass@k
        ks = sorted(m["pass_at_k"], key=lambda x: int(x))
        axes[0].plot(
            [int(k) for k in ks],
            [m["pass_at_k"][k] * 100 for k in ks],
            label=label, **style,
        )

        # cover@t
        ts = sorted(m["cover_at_t"], key=lambda x: float(x))
        axes[1].plot(
            [float(t) for t in ts],
            [m["cover_at_t"][t] for t in ts],
            label=label, **style,
        )

        # cover@t (distinct)
        ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
        axes[2].plot(
            [float(t) for t in ts_d],
            [m["cover_at_t_distinct"][t] for t in ts_d],
            label=label, **style,
        )

    axes[0].set_xlabel("k")
    axes[0].set_title("pass@k")
    axes[0].set_xticks([int(k) for k in ks])

    axes[1].set_xlabel("t")
    axes[1].set_title("cover@t")
    axes[1].set_xticks([float(t) for t in ts])

    axes[2].set_xlabel("t")
    axes[2].set_title("cover@t (distinct)")
    axes[2].set_xticks([float(t) for t in ts_d])

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.grid(alpha=0.3)

    # Adjust legend font size based on number of lines
    legend_fontsize = 5 if n_lines > 10 else 7
    axes[2].legend(fontsize=legend_fontsize, loc="upper right")
    fig.suptitle(f"{dataset_name}: Metrics Overview", fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-method colours for faceted plots (model is implicit from the row)
# ---------------------------------------------------------------------------
_METHOD_COLORS: dict[str, str] = {
    "greedy":      "#2B6CB0",  # blue
    "temp_0.2":    "#C05621",  # orange
    "temp_0.7":    "#2F855A",  # green
    "top_p_0.95":  "#9B2C2C",  # red
    "p_less":      "#6B46C1",  # purple
    "p_less_norm": "#B7791F",  # gold
    # legacy aliases
    "pless":       "#6B46C1",
    "pless_norm":  "#B7791F",
}


def plot_aggregate_lines_faceted(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Faceted line plot: one row per model, columns for pass@k / cover@t / cover@t distinct.

    Designed for many configs (e.g. 4 models × 6 methods = 24) where a single
    row of 3 subplots would be unreadable.
    """
    # Group by model, preserving insertion order
    models_order: list[str] = list(dict.fromkeys(m["model"] for m in metrics_list))
    by_model: dict[str, list[dict]] = {model: [] for model in models_order}
    for m in metrics_list:
        by_model[m["model"]].append(m)

    n_models = len(models_order)
    col_titles = ["pass@k", "cover@t", "cover@t (distinct)"]

    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(15, 4 * n_models),
        sharey=True,
        squeeze=False,
    )

    # Track which methods we've added to the legend
    legend_handles: dict[str, matplotlib.lines.Line2D] = {}

    for row, model in enumerate(models_order):
        for m in by_model[model]:
            method = m["method"]
            method_style = _METHOD_STYLES.get(method, dict(linestyle="-", marker="x"))
            color = _METHOD_COLORS.get(method, "#333333")
            style = dict(
                color=color,
                linestyle=method_style["linestyle"],
                marker=method_style["marker"],
                linewidth=2,
                markersize=6,
            )

            # pass@k
            ks = sorted(m["pass_at_k"], key=lambda x: int(x))
            line, = axes[row, 0].plot(
                [int(k) for k in ks],
                [m["pass_at_k"][k] * 100 for k in ks],
                **style,
            )
            if method not in legend_handles:
                legend_handles[method] = line

            # cover@t
            ts = sorted(m["cover_at_t"], key=lambda x: float(x))
            axes[row, 1].plot(
                [float(t) for t in ts],
                [m["cover_at_t"][t] for t in ts],
                **style,
            )

            # cover@t (distinct)
            ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
            axes[row, 2].plot(
                [float(t) for t in ts_d],
                [m["cover_at_t_distinct"][t] for t in ts_d],
                **style,
            )

        # Row label (model name) on the left
        short_model = model.split("/")[-1] if "/" in model else model
        axes[row, 0].set_ylabel(f"{short_model}\n\nPercentage", fontsize=9)

        for col in range(3):
            axes[row, col].set_ylim(0, 100)
            axes[row, col].grid(alpha=0.3)

        # x-tick labels only on the bottom row
        if row < n_models - 1:
            for col in range(3):
                axes[row, col].tick_params(labelbottom=False)

    # Column headers and x-labels on the bottom row
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)
    axes[-1, 0].set_xlabel("k")
    axes[-1, 1].set_xlabel("t")
    axes[-1, 2].set_xlabel("t")

    # Set x-ticks from the last plotted data
    if metrics_list:
        last = metrics_list[-1]
        ks = sorted(last["pass_at_k"], key=lambda x: int(x))
        ts = sorted(last["cover_at_t"], key=lambda x: float(x))
        ts_d = sorted(last["cover_at_t_distinct"], key=lambda x: float(x))
        for row in range(n_models):
            axes[row, 0].set_xticks([int(k) for k in ks])
            axes[row, 1].set_xticks([float(t) for t in ts])
            axes[row, 2].set_xticks([float(t) for t in ts_d])

    # Shared legend at the bottom
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(f"{dataset_name}: Metrics Overview (by model)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correctness_vs_diversity(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Bubble chart of num_correct vs num_distinct_correct per task.

    Uses only pless method configs (not pless_norm) to reduce clutter.
    """
    style_map = _build_style_map(metrics_list)
    fig, ax = plt.subplots(figsize=(8, 8))

    pless_configs = [m for m in metrics_list if m["method"] in ("pless", "p_less")]
    if not pless_configs:
        pless_configs = metrics_list  # fallback

    for m in pless_configs:
        color = _style_for(m, style_map)["color"]
        label = _label_for(m)

        counts: Counter[tuple[int, int]] = Counter()
        for task in m["per_task"]:
            nc = task["num_correct"]
            ndc = task["num_distinct_correct"]
            counts[(nc, ndc)] += 1

        xs, ys, sizes = [], [], []
        for (nc, ndc), count in counts.items():
            xs.append(nc)
            ys.append(ndc)
            sizes.append(count)

        # Scale bubble sizes for visibility
        sizes_arr = np.array(sizes, dtype=float)
        ax.scatter(
            xs, ys,
            s=sizes_arr * 15,
            alpha=0.55,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            label=label,
        )

    # y=x reference line
    ax.plot([0, 10], [0, 10], "k--", alpha=0.3, label="y = x (max diversity)")

    ax.set_xlabel("num_correct (out of 10)")
    ax.set_ylabel("num_distinct_correct (out of 10)")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"{dataset_name}: Correctness vs Diversity per Task (pless)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_structural_diversity_bars(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Grouped bar chart of structural diversity per model×method."""
    # Group by model
    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    methods_order = list(dict.fromkeys(m["method"] for m in metrics_list))

    fig, ax = plt.subplots(figsize=(max(8, len(models_order) * 2.5), 5))

    bar_width = 0.8 / max(len(methods_order), 1)
    x = np.arange(len(models_order))

    for i, method in enumerate(methods_order):
        values = []
        for model in models_order:
            match = [m for m in metrics_list if m["model"] == model and m["method"] == method]
            if match and "structural_diversity" in match[0]:
                values.append(match[0]["structural_diversity"])
            else:
                values.append(0.0)

        color = _METHOD_COLORS.get(method, "#333333")
        offset = (i - len(methods_order) / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width * 0.9, label=method, color=color, alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Structural Diversity\n(mean pairwise AST edit distance)")
    ax.set_title(f"{dataset_name}: Structural Diversity by Model & Method")
    ax.set_xticks(x)
    short_models = [m.split("/")[-1] if "/" in m else m for m in models_order]
    ax.set_xticklabels(short_models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pairwise_distance_distributions(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Box plot of per-task mean_pairwise_distance distributions, one box per model×method."""
    fig, ax = plt.subplots(figsize=(max(8, len(metrics_list) * 1.5), 5))

    labels = []
    data = []
    colors = []

    for m in metrics_list:
        distances = [
            t["mean_pairwise_distance"]
            for t in m.get("per_task", [])
            if t.get("num_correct", 0) >= 2 and "mean_pairwise_distance" in t
        ]
        if not distances:
            continue
        short_model = m["model"].split("/")[-1] if "/" in m["model"] else m["model"]
        labels.append(f"{short_model}\n({m['method']})")
        data.append(distances)
        colors.append(_METHOD_COLORS.get(m["method"], "#333333"))

    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Per-Task Mean Pairwise AST Edit Distance")
    ax.set_title(f"{dataset_name}: Distribution of Structural Diversity Across Tasks")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate metric visualizations from JSON files"
    )
    parser.add_argument(
        "--metrics", nargs="+", type=Path, required=True,
        help="Paths to metrics JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures"),
        help="Output directory for figures (default: results/figures/)",
    )
    parser.add_argument(
        "--dataset", type=str, default="MBPP",
        help="Dataset name used in plot titles (default: MBPP)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_list = load_metrics(args.metrics)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Determine how many distinct models are present
    n_models = len(dict.fromkeys(m["model"] for m in metrics_list))

    if n_models > 2:
        # Many configs — use the faceted (one-row-per-model) layout
        plot_aggregate_lines_faceted(
            metrics_list, out / "aggregate_lines_all.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'aggregate_lines_all.png'}")

        # Also produce a pless-only flat plot (readable with ≤8 lines)
        pless_only = [
            m for m in metrics_list
            if m["method"] in ("pless", "pless_norm", "p_less", "p_less_norm")
        ]
        if pless_only:
            plot_aggregate_lines(
                pless_only, out / "aggregate_lines_pless.png", dataset_name=args.dataset,
            )
            print(f"Saved {out / 'aggregate_lines_pless.png'}")
    else:
        plot_aggregate_lines(
            metrics_list, out / "aggregate_lines.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'aggregate_lines.png'}")

    plot_correctness_vs_diversity(
        metrics_list, out / "correctness_vs_diversity.png", dataset_name=args.dataset,
    )
    print(f"Saved {out / 'correctness_vs_diversity.png'}")

    # Structural diversity plots (only if data includes the new metrics)
    has_diversity = any("structural_diversity" in m for m in metrics_list)
    if has_diversity:
        plot_structural_diversity_bars(
            metrics_list, out / "structural_diversity_bars.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'structural_diversity_bars.png'}")

        plot_pairwise_distance_distributions(
            metrics_list, out / "pairwise_distance_distributions.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'pairwise_distance_distributions.png'}")


if __name__ == "__main__":
    main()
