"""Compare p-less sampling results against decoding methods from the paper.

Reference paper: "A Thorough Examination of Decoding Methods in the Era of LLMs"
(https://arxiv.org/abs/2402.06925)

Produces a markdown report and bar chart comparing our MBPP results (pless,
pless_norm, temp_0.7) with the paper's 14 decoding methods on Llama-2-7B
(base and chat).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paper results — MBPP pass@1 (%) from Table 1 of arXiv:2402.06925v3
# ---------------------------------------------------------------------------

PAPER_RESULTS: dict[str, dict[str, float]] = {
    "meta-llama/Llama-2-7b-hf": {
        "Greedy": 17.80,
        "Beam Search": 19.40,
        "Diverse Beam Search": 18.40,
        "Contrastive Search": 17.40,
        "FSD": 19.20,
        "FSD-d": 21.20,
        "Contrastive Decoding": 18.20,
        "DoLa": 18.40,
        "Temperature": 17.20,
        "Top-p": 14.80,
        "Top-k": 10.20,
        "η-Sampling": 9.40,
        "Mirostat": 7.80,
        "Typical": 12.00,
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "Greedy": 17.20,
        "Beam Search": 21.60,
        "Diverse Beam Search": 21.20,
        "Contrastive Search": 17.40,
        "FSD": 17.80,
        "FSD-d": 17.80,
        "Contrastive Decoding": 17.40,
        "DoLa": 18.00,
        "Temperature": 20.00,
        "Top-p": 17.60,
        "Top-k": 16.00,
        "η-Sampling": 17.00,
        "Mirostat": 16.00,
        "Typical": 18.00,
    },
}

# Display names for our methods
_OUR_METHOD_NAMES = {
    "pless": "P-Less (t=1.0)",
    "pless_norm": "P-Less Norm (t=1.0)",
    "temp": "Temperature (t=0.7)",
}

# Model directory name → paper model key
_MODEL_KEY_MAP = {
    "meta-llama--Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "meta-llama--Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
}

# Short display names for models
_MODEL_SHORT = {
    "meta-llama/Llama-2-7b-hf": "Llama-2-7B (base)",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7B-Chat",
}


def load_our_metrics(results_dir: Path) -> dict[str, list[dict]]:
    """Load computed metrics JSON files, grouped by model key.

    Returns dict mapping paper model key → list of metrics dicts.
    """
    metrics_by_model: dict[str, list[dict]] = {}
    for dir_name, model_key in _MODEL_KEY_MAP.items():
        metrics_dir = results_dir / dir_name / "metrics"
        if not metrics_dir.exists():
            continue
        metrics_list = []
        for p in sorted(metrics_dir.glob("*_metrics.json")):
            with open(p) as f:
                metrics_list.append(json.load(f))
        if metrics_list:
            metrics_by_model[model_key] = metrics_list
    return metrics_by_model


def build_comparison_rows(
    paper_results: dict[str, float],
    our_metrics: list[dict],
) -> list[dict]:
    """Build a sorted list of rows for the comparison table.

    Each row: {method, source, pass_at_1}
    """
    rows = []

    # Paper methods
    for method, score in paper_results.items():
        rows.append({"method": method, "source": "Paper", "pass_at_1": score})

    # Our methods
    for m in our_metrics:
        method_name = _OUR_METHOD_NAMES.get(m["method"], m["method"])
        pass_at_1 = m["pass_at_k"].get("1")
        if pass_at_1 is not None:
            rows.append({
                "method": method_name,
                "source": "Ours",
                "pass_at_1": pass_at_1 * 100,  # stored as fraction
            })

    rows.sort(key=lambda r: r["pass_at_1"], reverse=True)
    return rows


def rank_of(rows: list[dict], method_name: str) -> int | None:
    """Return 1-based rank of a method in the comparison rows."""
    for i, row in enumerate(rows):
        if row["method"] == method_name:
            return i + 1
    return None


def format_comparison_table(rows: list[dict], model_display: str) -> str:
    """Render the comparison rows as a markdown table."""
    lines = [
        f"### {model_display}\n",
        "| Rank | Method | Source | pass@1 (%) |",
        "| ---: | ------ | ------ | ---------: |",
    ]
    for i, row in enumerate(rows, 1):
        marker = " **←**" if row["source"] == "Ours" else ""
        lines.append(
            f"| {i} | {row['method']}{marker} | {row['source']} | {row['pass_at_1']:.1f} |"
        )
    return "\n".join(lines)


def format_extended_metrics_table(our_metrics: list[dict], model_display: str) -> str:
    """Render our pass@k and cover@t metrics as a markdown table."""
    if not our_metrics:
        return ""

    # Collect all k and t values
    k_values = sorted({int(k) for m in our_metrics for k in m["pass_at_k"]})
    t_values = sorted({float(t) for m in our_metrics for t in m["cover_at_t"]})

    pass_cols = [f"pass@{k}" for k in k_values]
    cover_cols = []
    for t in t_values:
        cover_cols.append(f"cover@{t}")
        cover_cols.append(f"cover@{t} (dist)")

    header = ["Method"] + pass_cols + cover_cols
    sep = ["---"] + ["---------:"] * (len(pass_cols) + len(cover_cols))

    lines = [
        f"### {model_display} — Extended Metrics\n",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for m in our_metrics:
        method_name = _OUR_METHOD_NAMES.get(m["method"], m["method"])
        row = [method_name]
        for k in k_values:
            val = m["pass_at_k"].get(str(k))
            row.append(f"{val * 100:.1f}" if val is not None else "-")
        for t in t_values:
            val = m["cover_at_t"].get(str(t))
            row.append(f"{val:.1f}" if val is not None else "-")
            val_d = m["cover_at_t_distinct"].get(str(t))
            row.append(f"{val_d:.1f}" if val_d is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; "
                 "(dist) = distinct correct samples only.*")
    return "\n".join(lines)


def generate_analysis(
    rows_by_model: dict[str, list[dict]],
    our_metrics_by_model: dict[str, list[dict]],
) -> str:
    """Generate the analysis section of the report."""
    parts = ["## Analysis\n"]

    for model_key, rows in rows_by_model.items():
        display = _MODEL_SHORT.get(model_key, model_key)
        parts.append(f"### {display}\n")

        total_methods = len(rows)

        for name in _OUR_METHOD_NAMES.values():
            r = rank_of(rows, name)
            if r is not None:
                parts.append(f"- **{name}**: rank {r}/{total_methods}")

        # Compare pless vs paper's Temperature
        pless_row = next((r for r in rows if r["method"] == "P-Less (t=1.0)"), None)
        temp_paper = next((r for r in rows if r["method"] == "Temperature" and r["source"] == "Paper"), None)
        if pless_row and temp_paper:
            diff = pless_row["pass_at_1"] - temp_paper["pass_at_1"]
            direction = "above" if diff > 0 else "below"
            parts.append(
                f"- P-Less vs paper's Temperature sampling: "
                f"{abs(diff):.1f}pp {direction} ({pless_row['pass_at_1']:.1f}% vs {temp_paper['pass_at_1']:.1f}%)"
            )

        # Our temp vs paper's temp (sanity check)
        temp_ours = next((r for r in rows if r["method"] == "Temperature (t=0.7)" and r["source"] == "Ours"), None)
        if temp_ours and temp_paper:
            diff = temp_ours["pass_at_1"] - temp_paper["pass_at_1"]
            parts.append(
                f"- Our temp_0.7 vs paper's Temperature: "
                f"{temp_ours['pass_at_1']:.1f}% vs {temp_paper['pass_at_1']:.1f}% "
                f"(Δ={diff:+.1f}pp — sanity check for setup alignment)"
            )

        parts.append("")

    parts.append("### Limitations\n")
    parts.append(
        "- We ran only 3 methods (pless, pless_norm, temp_0.7) vs the paper's 14. "
        "The comparison is partial.\n"
        "- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; "
        "exact match is not expected due to differences in prompting, generation length, "
        "and MBPP subset.\n"
        "- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator "
        "over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy."
    )

    return "\n".join(parts)


def generate_report(
    our_metrics_by_model: dict[str, list[dict]],
    paper_results: dict[str, dict[str, float]] = PAPER_RESULTS,
) -> str:
    """Generate the full markdown comparison report."""
    sections = [
        "# MBPP: P-Less vs Paper Decoding Methods (Llama-2-7B)\n",
        "Comparison of p-less sampling against 14 decoding methods from "
        '"A Thorough Examination of Decoding Methods in the Era of LLMs" '
        "(arXiv:2402.06925).\n",
        "## pass@1 Comparison\n",
    ]

    rows_by_model: dict[str, list[dict]] = {}

    for model_key in paper_results:
        display = _MODEL_SHORT.get(model_key, model_key)
        our_metrics = our_metrics_by_model.get(model_key, [])
        rows = build_comparison_rows(paper_results[model_key], our_metrics)
        rows_by_model[model_key] = rows
        sections.append(format_comparison_table(rows, display))
        sections.append("")

    sections.append("\n## Extended Metrics (Our Methods Only)\n")
    for model_key in paper_results:
        display = _MODEL_SHORT.get(model_key, model_key)
        our_metrics = our_metrics_by_model.get(model_key, [])
        table = format_extended_metrics_table(our_metrics, display)
        if table:
            sections.append(table)
            sections.append("")

    sections.append(generate_analysis(rows_by_model, our_metrics_by_model))

    return "\n".join(sections)


def plot_comparison(
    our_metrics_by_model: dict[str, list[dict]],
    output_path: Path,
    paper_results: dict[str, dict[str, float]] = PAPER_RESULTS,
) -> None:
    """Bar chart: pass@1 for all methods, our methods highlighted."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    models = list(paper_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6), squeeze=False)

    for col, model_key in enumerate(models):
        ax = axes[0, col]
        our_metrics = our_metrics_by_model.get(model_key, [])
        rows = build_comparison_rows(paper_results[model_key], our_metrics)

        methods = [r["method"] for r in rows]
        scores = [r["pass_at_1"] for r in rows]
        sources = [r["source"] for r in rows]

        colors = ["#6B46C1" if s == "Ours" else "#A0AEC0" for s in sources]
        edgecolors = ["#4C1D95" if s == "Ours" else "#718096" for s in sources]

        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor=edgecolors, linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("pass@1 (%)")
        ax.set_title(_MODEL_SHORT.get(model_key, model_key), fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", fontsize=7,
            )

    # Legend
    legend_elements = [
        Patch(facecolor="#A0AEC0", edgecolor="#718096", label="Paper"),
        Patch(facecolor="#6B46C1", edgecolor="#4C1D95", label="Ours"),
    ]
    axes[0, -1].legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.suptitle("MBPP pass@1: P-Less vs Paper Decoding Methods", fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics_overview(
    our_metrics_by_model: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Faceted line plot: one row per model, columns for pass@k / cover@t / cover@t (distinct).

    Matches the style of plots.py:plot_aggregate_lines but faceted by model.
    """
    import matplotlib.pyplot as plt

    # Flatten and group
    models = list(our_metrics_by_model.keys())
    n_models = len(models)

    # Method colours (consistent with plots.py)
    method_colors = {
        "pless": "#6B46C1",
        "pless_norm": "#B7791F",
        "temp": "#2F855A",
    }
    method_styles = {
        "pless": dict(linestyle="-", marker="o"),
        "pless_norm": dict(linestyle="-.", marker="X"),
        "temp": dict(linestyle="-", marker="s"),
    }
    method_labels = {
        "pless": "pless",
        "pless_norm": "pless_norm",
        "temp": "temp_0.7",
    }

    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4 * n_models), sharey=True, squeeze=False,
    )

    legend_handles: dict[str, object] = {}

    for row, model_key in enumerate(models):
        display = _MODEL_SHORT.get(model_key, model_key)
        metrics_list = our_metrics_by_model[model_key]

        for m in metrics_list:
            method = m["method"]
            color = method_colors.get(method, "#333333")
            ms = method_styles.get(method, dict(linestyle="-", marker="x"))
            style = dict(color=color, linewidth=2, markersize=6, **ms)
            label = method_labels.get(method, method)

            # pass@k
            ks = sorted(m["pass_at_k"], key=lambda x: int(x))
            line, = axes[row, 0].plot(
                [int(k) for k in ks],
                [m["pass_at_k"][k] * 100 for k in ks],
                **style,
            )
            if label not in legend_handles:
                legend_handles[label] = line

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

        axes[row, 0].set_ylabel(f"{display}\n\nPercentage", fontsize=9)
        for col in range(3):
            axes[row, col].set_ylim(0, 100)
            axes[row, col].grid(alpha=0.3)
        if row < n_models - 1:
            for col in range(3):
                axes[row, col].tick_params(labelbottom=False)

    # Column titles and x-labels
    for col, title in enumerate(["pass@k", "cover@t", "cover@t (distinct)"]):
        axes[0, col].set_title(title, fontsize=11)
    axes[-1, 0].set_xlabel("k")
    axes[-1, 1].set_xlabel("t")
    axes[-1, 2].set_xlabel("t")

    # Set x-ticks from the last plotted data
    if models:
        last = our_metrics_by_model[models[-1]][-1]
        ks = sorted(last["pass_at_k"], key=lambda x: int(x))
        ts = sorted(last["cover_at_t"], key=lambda x: float(x))
        for r in range(n_models):
            axes[r, 0].set_xticks([int(k) for k in ks])
            axes[r, 1].set_xticks([float(t) for t in ts])
            axes[r, 2].set_xticks([float(t) for t in ts])

    fig.legend(
        legend_handles.values(), legend_handles.keys(),
        loc="lower center", ncol=len(legend_handles), fontsize=9, frameon=True,
    )
    fig.suptitle("MBPP: Metrics Overview (Llama-2-7B)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare p-less MBPP results against decoding methods from the paper"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Parent directory containing model result subdirectories",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/paper_comparison/comparison_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--figures-dir", type=Path,
        default=Path("results/paper_comparison/figures"),
        help="Directory for generated plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    our_metrics = load_our_metrics(args.results_dir)
    if not our_metrics:
        print("ERROR: No metrics files found. Run `python -m bench.eval` first.")
        raise SystemExit(1)

    print(f"Loaded metrics for {len(our_metrics)} models:")
    for model_key, mlist in our_metrics.items():
        methods = [m["method"] for m in mlist]
        print(f"  {model_key}: {methods}")

    # Generate report
    report = generate_report(our_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {args.output}")

    # Generate figures
    fig_path = args.figures_dir / "pass_at_1_comparison.png"
    plot_comparison(our_metrics, fig_path)
    print(f"Figure saved to {fig_path}")

    overview_path = args.figures_dir / "metrics_overview.png"
    plot_metrics_overview(our_metrics, overview_path)
    print(f"Figure saved to {overview_path}")


if __name__ == "__main__":
    main()
