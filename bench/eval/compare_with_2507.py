"""Compare p-less sampling results against arXiv 2507.03160.

Reference paper: "Assessing Small Language Models for Code Generation"
(https://arxiv.org/abs/2507.03160)

Produces a markdown report and bar chart comparing our MBPP bigcode results
(pless, pless_norm, temp, top_p) with the paper's top_p=0.95/t=0.2 baseline
on Qwen2.5-Coder-3B, Qwen2.5-Coder-1.5B, and OpenCodeInterpreter-DS-1.3B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.eval.compare_with_paper import (
    generate_report,
    load_our_metrics,
    plot_comparison,
)

# ---------------------------------------------------------------------------
# Paper results — MBPP full pass@1 (%) from arXiv:2507.03160
# Config: top_p=0.95, temp=0.2, n=10, unbiased estimator, BigCode docstring format
# ---------------------------------------------------------------------------

PAPER_RESULTS: dict[str, dict[str, float]] = {
    "Qwen/Qwen2.5-Coder-3B": {
        "Top-p (paper, t=0.2)": 57.0,
    },
    "Qwen/Qwen2.5-Coder-1.5B": {
        "Top-p (paper, t=0.2)": 51.0,
    },
    "m-a-p/OpenCodeInterpreter-DS-1.3B": {
        "Top-p (paper, t=0.2)": 44.0,
    },
}

# Model directory name → paper model key
_MODEL_KEY_MAP = {
    "Qwen--Qwen2.5-Coder-3B":  "Qwen/Qwen2.5-Coder-3B",
    "Qwen--Qwen2.5-Coder-1.5B": "Qwen/Qwen2.5-Coder-1.5B",
    "m-a-p--OpenCodeInterpreter-DS-1.3B": "m-a-p/OpenCodeInterpreter-DS-1.3B",
}

# Short display names for models
_MODEL_SHORT = {
    "Qwen/Qwen2.5-Coder-3B":  "Qwen2.5-Coder-3B",
    "Qwen/Qwen2.5-Coder-1.5B": "Qwen2.5-Coder-1.5B",
    "m-a-p/OpenCodeInterpreter-DS-1.3B": "OCI-DS-1.3B",
}

# Display names for our methods — used by generate_report for table rows.
# top_p entries get renamed in _enrich_metrics() below so the table shows
# "Top-p p=0.95 (t=0.2, paper)" / "Top-p p=0.9 (t=1.0)" instead of generic "Top-p (ours)".
_OUR_METHOD_NAMES = {
    "pless": "P-Less",
    "pless_norm": "P-Less Norm",
    "temp": "Temperature",
    "top_p": "Top-p (ours)",          # fallback; usually overridden by _enrich_metrics
    "top_p0.95": "Top-p p=0.95 (paper replication)",
    "top_p0.9": "Top-p p=0.9",
}


# ---------------------------------------------------------------------------
# Per-curve style config for the metrics overview plot
# ---------------------------------------------------------------------------

# (method, linestyle_key) → label shown in the legend
# linestyle_key: "solid" for lower-temperature runs, "dashed" for higher-temp runs
_CURVE_STYLE: dict[str, dict] = {
    # method key  →  {color, marker, linestyle_solid, linestyle_dashed}
    "pless":      dict(color="#6B46C1", marker="o"),
    "pless_norm": dict(color="#B7791F", marker="X"),
    "temp":       dict(color="#2F855A", marker="s"),
    "top_p":      dict(color="#D53F8C", marker="D"),
}


def _curve_label(m: dict) -> str:
    """Build a unique, descriptive legend label for each (method, temperature) curve."""
    method = m["method"]
    temp = m.get("temperature")
    top_p_val = m.get("top_p")

    if method == "pless":
        return f"P-Less (t={temp})"
    if method == "pless_norm":
        return f"P-Less Norm (t={temp})"
    if method == "temp":
        return f"Temp (t={temp})"
    if method == "top_p":
        if top_p_val == 0.95:
            return f"Top-p p=0.95 (t={temp}, paper)"
        if top_p_val is not None:
            return f"Top-p p={top_p_val} (t={temp})"
        return f"Top-p (t={temp})"
    return f"{method} (t={temp})"


def _is_high_temp(m: dict) -> bool:
    """Return True for the 'exploratory' higher-temperature variant of a method."""
    temp = m.get("temperature", 0.0)
    top_p_val = m.get("top_p")
    # top_p paper replication runs at t=0.2, treat as primary (solid)
    if m["method"] == "top_p" and top_p_val == 0.95:
        return False
    return temp >= 1.0


def _enrich_metrics(our_metrics: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Return a copy of our_metrics with top_p method keys renamed to top_p0.95/top_p0.9.

    This makes the report table show specific top_p values instead of the generic "top_p".
    Does NOT mutate the original dicts.
    """
    enriched: dict[str, list[dict]] = {}
    for model_key, mlist in our_metrics.items():
        new_list = []
        for m in mlist:
            if m["method"] == "top_p" and m.get("top_p") is not None:
                m = {**m, "method": f"top_p{m['top_p']}"}
            new_list.append(m)
        enriched[model_key] = new_list
    return enriched


def plot_metrics_overview_2507(
    our_metrics_by_model: dict[str, list[dict]],
    output_path: Path,
    model_short: dict[str, str],
    suptitle: str,
) -> None:
    """Faceted line plot with per-curve labels and solid/dashed temperature distinction.

    Solid lines = primary / lower-temperature configs.
    Dashed lines = higher-temperature (t=1.0) variants.
    Each curve gets its own legend entry with a descriptive label.
    """
    import matplotlib.pyplot as plt

    models = list(our_metrics_by_model.keys())
    n_models = len(models)

    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4 * n_models), sharey=False, squeeze=False,
    )

    legend_handles: dict[str, object] = {}
    # Collect plotted values per column to set tight y limits after plotting
    col_vals: list[list[float]] = [[], [], []]

    for row, model_key in enumerate(models):
        display = model_short.get(model_key, model_key)
        metrics_list = our_metrics_by_model[model_key]

        for m in metrics_list:
            method = m["method"]
            style_cfg = _CURVE_STYLE.get(method, dict(color="#333333", marker="x"))
            linestyle = "--" if _is_high_temp(m) else "-"
            label = _curve_label(m)

            plot_kwargs = dict(
                color=style_cfg["color"],
                marker=style_cfg["marker"],
                linestyle=linestyle,
                linewidth=2.5,
                markersize=8,
            )

            # pass@k
            ks = sorted(m["pass_at_k"], key=lambda x: int(x))
            vals0 = [m["pass_at_k"][k] * 100 for k in ks]
            col_vals[0].extend(vals0)
            line, = axes[row, 0].plot([int(k) for k in ks], vals0, **plot_kwargs)
            if label not in legend_handles:
                legend_handles[label] = line

            # cover@t
            ts = sorted(m["cover_at_t"], key=lambda x: float(x))
            vals1 = [m["cover_at_t"][t] for t in ts]
            col_vals[1].extend(vals1)
            axes[row, 1].plot([float(t) for t in ts], vals1, **plot_kwargs)

            # cover@t (distinct)
            ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
            vals2 = [m["cover_at_t_distinct"][t] for t in ts_d]
            col_vals[2].extend(vals2)
            axes[row, 2].plot([float(t) for t in ts_d], vals2, **plot_kwargs)

        axes[row, 0].set_ylabel(f"{display}\n\nPercentage", fontsize=9)
        for col in range(3):
            axes[row, col].grid(alpha=0.3)
        if row < n_models - 1:
            for col in range(3):
                axes[row, col].tick_params(labelbottom=False)

    # Set tight per-column y limits so differences are visible
    import math
    for col in range(3):
        if not col_vals[col]:
            continue
        lo = max(0.0, math.floor(min(col_vals[col])) - 3)
        hi = min(100.0, math.ceil(max(col_vals[col])) + 3)
        if hi - lo < 15:   # guarantee readable spread
            mid = (lo + hi) / 2
            lo, hi = max(0.0, mid - 8), min(100.0, mid + 8)
        for r in range(n_models):
            axes[r, col].set_ylim(lo, hi)

    # Column titles and x-labels
    for col, title in enumerate(["pass@k", "cover@t", "cover@t (distinct)"]):
        axes[0, col].set_title(title, fontsize=11)
    axes[-1, 0].set_xlabel("k")
    axes[-1, 1].set_xlabel("t")
    axes[-1, 2].set_xlabel("t")

    # X-tick labels
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
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        title="solid line = primary/low temp  |  dashed line = t=1.0",
        title_fontsize=8.5,
    )
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.11, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare p-less MBPP results against arXiv 2507.03160"
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results/pless_full_mbpp_results"),
        help="Parent directory containing model result subdirectories",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/pless_full_mbpp_results/analysis/small_models_2507/comparison_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--figures-dir", type=Path,
        default=Path("results/pless_full_mbpp_results/analysis/small_models_2507/figures"),
        help="Directory for generated plots",
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Filter to specific model(s) by key (e.g. 'm-a-p/OpenCodeInterpreter-DS-1.3B')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Filter dicts when --models is provided
    paper_results = PAPER_RESULTS
    model_key_map = _MODEL_KEY_MAP
    model_short = _MODEL_SHORT
    if args.models:
        requested = set(args.models)
        paper_results = {k: v for k, v in PAPER_RESULTS.items() if k in requested}
        model_key_map = {k: v for k, v in _MODEL_KEY_MAP.items() if v in requested}
        model_short = {k: v for k, v in _MODEL_SHORT.items() if k in requested}

    our_metrics = load_our_metrics(args.results_dir, model_key_map=model_key_map)
    if not our_metrics:
        print("ERROR: No metrics files found. Run `python -m bench.eval` first.")
        raise SystemExit(1)

    print(f"Loaded metrics for {len(our_metrics)} models:")
    for model_key, mlist in our_metrics.items():
        methods = [(m["method"], m.get("top_p"), m.get("temperature")) for m in mlist]
        print(f"  {model_key}: {methods}")

    # Build title from selected models
    short_names = [model_short.get(k, k) for k in our_metrics]
    models_label = ", ".join(short_names)

    # Enrich top_p method keys with their actual p value for the report table
    enriched_metrics = _enrich_metrics(our_metrics)

    report = generate_report(
        enriched_metrics,
        paper_results=paper_results,
        title=f"MBPP: P-Less vs arXiv 2507.03160 Baseline ({models_label})",
        description=(
            "Comparison of p-less sampling against the top_p=0.95/temp=0.2 baseline from "
            '"Assessing Small Language Models for Code Generation" '
            "(arXiv:2507.03160). BigCode zero-shot docstring format, MBPP full (500 tasks), "
            f"n=10, unbiased pass@k estimator. Models: {models_label}."
        ),
        model_short=model_short,
        our_method_names=_OUR_METHOD_NAMES,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {args.output}")

    fig_path = args.figures_dir / "pass_at_1_comparison.png"
    plot_comparison(
        enriched_metrics, fig_path,
        paper_results=paper_results,
        model_short=model_short,
        suptitle=f"MBPP pass@1: P-Less vs arXiv 2507.03160 ({models_label})",
    )
    print(f"Figure saved to {fig_path}")

    overview_path = args.figures_dir / "metrics_overview.png"
    # Use custom plot that gives each (method, temperature) curve its own labeled entry
    plot_metrics_overview_2507(
        our_metrics,   # use original (not enriched) so _curve_label can read top_p field
        overview_path,
        model_short=model_short,
        suptitle=f"MBPP: Metrics Overview ({models_label}, BigCode format)",
    )
    print(f"Figure saved to {overview_path}")


if __name__ == "__main__":
    main()
