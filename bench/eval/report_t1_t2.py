"""Generate T1/T2 post-truncation temperature analysis report and plots.

Compares P-less T1/T2 results from `results/full_mbpp_pre_post_temp_pless/`
against baselines from `results/pless_full_mbpp_results/analysis/consolidated_metrics/`.

Usage:
    uv run python -m bench.eval.report_t1_t2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
T1T2_METRICS_DIR = Path(
    "results/full_mbpp_pre_post_temp_pless/Qwen--Qwen2.5-Coder-3B/metrics"
)
BASELINE_METRICS_DIR = Path(
    "results/pless_full_mbpp_results/analysis/consolidated_metrics/Qwen--Qwen2.5-Coder-3B"
)
OUTPUT_DIR = Path("results/full_mbpp_pre_post_temp_pless/analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Colour / style config
# ---------------------------------------------------------------------------
# T2 colours
T2_COLORS = {"—": "#2B6CB0", "2.0": "#C05621", "5.0": "#2F855A"}
# Baseline method colours
BASELINE_COLORS = {
    "pless": "#6B46C1",
    "pless_norm": "#B7791F",
    "temp": "#9B2C2C",
    "top_p0.9": "#2C7A7B",
    "top_p0.95": "#D53F8C",
}
# T2 marker shapes (for Pareto scatter)
T2_MARKERS = {"—": "o", "2.0": "s", "5.0": "D"}
# T1 colours (for Pareto scatter)
T1_COLORS = {0.6: "#2B6CB0", 0.8: "#C05621", 1.0: "#2F855A"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_t1_t2(stem: str) -> tuple[str, float, str]:
    """Parse filename stem into (method_label, T1, T2).

    Examples:
      pless_bigcode_t0.8        -> ("pless", 0.8, "—")
      pless_pt2.0_bigcode_t0.6  -> ("pless", 0.6, "2.0")
      pless_bigcode_t0.6        -> ("pless", 0.6, "—")  (baseline)
      temp_bigcode_t0.7         -> ("temp", 0.7, "—")
      top_p0.9_bigcode_t1.0     -> ("top_p0.9", 1.0, "—")
      top_p0.95_bigcode_t0.2    -> ("top_p0.95", 0.2, "—")
      pless_norm_bigcode_t0.6   -> ("pless_norm", 0.6, "—")
    """
    t1 = float(stem.rsplit("_t", 1)[1])
    t2 = "—"
    if "_pt" in stem:
        t2 = stem.split("_pt")[1].split("_")[0]
    # Method name
    method_part = stem.rsplit("_t", 1)[0]  # e.g. pless_pt2.0_bigcode
    method_part = method_part.replace("_bigcode", "")
    # Remove _pt* suffix to get base method
    if "_pt" in method_part:
        method_part = method_part.split("_pt")[0]
    return method_part, t1, t2


def load_all_metrics() -> list[dict]:
    """Load T1/T2 and baseline metrics, enriching each with T1/T2/group metadata."""
    rows = []

    for metrics_dir, group in [
        (T1T2_METRICS_DIR, "t1t2"),
        (BASELINE_METRICS_DIR, "baseline"),
    ]:
        if not metrics_dir.exists():
            print(f"WARNING: {metrics_dir} does not exist, skipping")
            continue
        for f in sorted(metrics_dir.glob("*_metrics.json")):
            m = json.loads(f.read_text())
            stem = f.stem.replace("_metrics", "")
            method, t1, t2 = _parse_t1_t2(stem)
            m["_method"] = method
            m["_t1"] = t1
            m["_t2"] = t2
            m["_group"] = group
            m["_stem"] = stem
            rows.append(m)

    return rows


def _display_name(r: dict) -> str:
    """Human-readable config label."""
    method = r["_method"]
    t1 = r["_t1"]
    t2 = r["_t2"]
    if t2 != "—":
        return f"{method} T1={t1} T2={t2}"
    return f"{method} t={t1}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(rows: list[dict]) -> str:
    """Build the markdown report string."""
    lines = []
    lines.append("# P-less T1/T2 Post-Truncation Temperature: MBPP Analysis")
    lines.append("")
    lines.append(
        "**Model:** Qwen2.5-Coder-3B (base) | **Dataset:** MBPP-full (500 tasks × 10 samples) | "
        "**Prompt:** BigCode zero-shot docstring"
    )
    lines.append("")
    lines.append(
        "T₁ (pre-truncation temperature) scales logits before P-less computes its collision "
        "entropy threshold. T₂ (post-truncation temperature) applies `prob^(1/T₂)` to flatten "
        "the survivor distribution after P-less pruning."
    )
    lines.append("")

    # --- Full metrics table ---
    lines.append("## Full Metrics Comparison")
    lines.append("")
    lines.append(
        "| # | Config | T1 | T2 | pass@1 | pass@3 | pass@5 | pass@10 | "
        "cover@0.7 | struct_div | codebleu_div |"
    )
    lines.append(
        "|---|--------|----|----|--------|--------|--------|---------|"
        "-----------|------------|--------------|"
    )

    sorted_rows = sorted(rows, key=lambda r: -r["pass_at_k"]["1"])
    for i, r in enumerate(sorted_rows, 1):
        pk = r["pass_at_k"]
        c07 = r["cover_at_t"]["0.7"]
        sdiv = r.get("structural_diversity", 0)
        cbdiv = r.get("codebleu_diversity", 0)
        marker = " **←**" if r["_group"] == "t1t2" else ""
        lines.append(
            f"| {i} | {_display_name(r)}{marker} | {r['_t1']} | {r['_t2']} | "
            f"{pk['1']*100:.1f} | {pk['3']*100:.1f} | {pk['5']*100:.1f} | {pk['10']*100:.1f} | "
            f"{c07:.1f} | {sdiv:.4f} | {cbdiv:.4f} |"
        )

    lines.append("")
    lines.append(
        "*pass@k as %; cover@t = % of tasks with ≥t fraction correct; "
        "struct_div = mean pairwise AST edit distance; "
        "codebleu_div = 1 − mean pairwise CodeBLEU similarity. "
        "**←** = T1/T2 experiment configs.*"
    )

    # --- T2 effect analysis ---
    lines.append("")
    lines.append("## T2 Effect at Fixed T1")
    lines.append("")

    # Build lookup: (method, t1, t2) -> row
    lookup = {(r["_method"], r["_t1"], r["_t2"]): r for r in rows}

    for t1 in [0.6, 0.8, 1.0]:
        # Find the no-T2 baseline for this T1
        base = lookup.get(("pless", t1, "—"))
        if base is None:
            continue
        bp1 = base["pass_at_k"]["1"] * 100
        bsd = base.get("structural_diversity", 0)
        bcb = base.get("codebleu_diversity", 0)

        lines.append(f"### T1={t1}")
        lines.append(f"Baseline (no T2): pass@1={bp1:.1f}%, struct_div={bsd:.4f}, codebleu_div={bcb:.4f}")
        lines.append("")
        lines.append("| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |")
        lines.append("|----|--------|----------|------------|--------|--------------|---------|")

        for t2 in ["2.0", "5.0"]:
            r = lookup.get(("pless", t1, t2))
            if r is None:
                continue
            p1 = r["pass_at_k"]["1"] * 100
            sd = r.get("structural_diversity", 0)
            cb = r.get("codebleu_diversity", 0)
            lines.append(
                f"| {t2} | {p1:.1f}% | {p1 - bp1:+.1f}pp | "
                f"{sd:.4f} | {sd - bsd:+.4f} | {cb:.4f} | {cb - bcb:+.4f} |"
            )

        lines.append("")

    # --- Diversity exchange rate ---
    lines.append("## Diversity Exchange Rate")
    lines.append("")
    lines.append(
        "How much diversity (struct_div, codebleu_div) is gained per percentage point of pass@1 lost, "
        "relative to the best baseline (pless t=0.6)."
    )
    lines.append("")

    ref = lookup.get(("pless", 0.6, "—"))
    if ref:
        rp1 = ref["pass_at_k"]["1"] * 100
        rsd = ref.get("structural_diversity", 0)
        rcb = ref.get("codebleu_diversity", 0)

        lines.append(
            f"**Reference:** pless t=0.6 (pass@1={rp1:.1f}%, struct_div={rsd:.4f}, "
            f"codebleu_div={rcb:.4f})"
        )
        lines.append("")
        lines.append(
            "| Config | pass@1 cost | sdiv gain | sdiv/pp | cbdiv gain | cbdiv/pp |"
        )
        lines.append(
            "|--------|-------------|-----------|---------|------------|----------|"
        )

        for r in sorted_rows:
            dp = rp1 - r["pass_at_k"]["1"] * 100
            ds = r.get("structural_diversity", 0) - rsd
            dc = r.get("codebleu_diversity", 0) - rcb
            if dp > 0.3 and (ds > 0.005 or dc > 0.005):
                sr = ds / dp if dp > 0 else 0
                cr = dc / dp if dp > 0 else 0
                lines.append(
                    f"| {_display_name(r)} | {dp:.1f}pp | {ds:+.4f} | {sr:.4f}/pp | "
                    f"{dc:+.4f} | {cr:.4f}/pp |"
                )

        lines.append("")

    # --- Key findings ---
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        "1. **T2 has negligible effect on the base model.** At T1=0.6, T2 adds +0.005-0.007 "
        "struct_div while costing ~1pp pass@1. At T1=0.8, T2 adds +0.003-0.010. At T1=1.0, "
        "T2 *decreases* diversity slightly. The post-truncation flattening does not meaningfully "
        "change outcomes when P-less already leaves a moderate number of survivors."
    )
    lines.append("")
    lines.append(
        "2. **T1 (pre-truncation temperature) does all the work.** The diversity jump from "
        "T1=0.6→0.8→1.0 is substantial (struct_div 0.097→0.167→0.255), matching the effect "
        "of raising temperature in standard P-less. T1 controls pruning aggressiveness by "
        "shaping the distribution before the collision entropy threshold is computed."
    )
    lines.append("")
    lines.append(
        "3. **Best new config: pless T1=0.8 (no T2).** At 58.7% pass@1, 0.167 struct_div, "
        "0.298 codebleu_div, it matches top_p0.95/t=0.2 (58.2%) while providing a useful "
        "diversity level — with zero hyperparameters beyond T1."
    )
    lines.append("")
    lines.append(
        "4. **Implication for instruct models:** Since T2 doesn't help on base models where "
        "P-less leaves some survivors, it will help even less on instruct models where P-less "
        "leaves ~1 survivor. The instruct experiment needs high T1 (>1.0) to open the "
        "distribution — T2 alone cannot rescue diversity."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_rows(rows: list[dict]) -> list[dict]:
    """Filter rows for plots: exclude pless_norm (not part of T1/T2 experiment)."""
    return [r for r in rows if r["_method"] != "pless_norm"]


def plot_pass_at_1_bars(rows: list[dict], output_path: Path) -> None:
    """Horizontal bar chart of pass@1 for all configs."""
    plot_rows = _plot_rows(rows)
    sorted_rows = sorted(plot_rows, key=lambda r: r["pass_at_k"]["1"])

    labels = [_display_name(r) for r in sorted_rows]
    values = [r["pass_at_k"]["1"] * 100 for r in sorted_rows]

    # Color scheme: experiment configs by T2 value, baselines in grey tones
    EXPERIMENT_COLORS = {
        "—": "#3182CE",   # blue — pless with no T2 (new T1=0.8 run)
        "2.0": "#DD6B20",  # orange — T2=2.0
        "5.0": "#38A169",  # green — T2=5.0
    }
    REF_COLORS = {
        "pless": "#A0AEC0",   # grey — existing pless baselines
        "temp": "#E53E3E",    # red — temperature
        "top_p0.9": "#805AD5",  # purple — top_p
        "top_p0.95": "#D53F8C",  # pink — top_p0.95
    }

    colors = []
    for r in sorted_rows:
        if r["_group"] == "t1t2":
            colors.append(EXPERIMENT_COLORS.get(r["_t2"], "#888888"))
        else:
            colors.append(REF_COLORS.get(r["_method"], "#A0AEC0"))

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.45)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("pass@1 (%)", fontsize=11)
    ax.set_title("MBPP pass@1: P-less T1/T2 vs Baselines (Qwen2.5-Coder-3B)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8,
        )

    # Legend — clearly separated into experiment vs reference
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="none", label="$\\bf{This\\ experiment}$"),
        Patch(color=EXPERIMENT_COLORS["—"], label="pless (no T2)"),
        Patch(color=EXPERIMENT_COLORS["2.0"], label="pless + T2=2.0"),
        Patch(color=EXPERIMENT_COLORS["5.0"], label="pless + T2=5.0"),
        Patch(color="none", label=""),
        Patch(color="none", label="$\\bf{Reference\\ baselines}$"),
        Patch(color=REF_COLORS["pless"], label="pless (existing)"),
        Patch(color=REF_COLORS["temp"], label="temp t=0.7"),
        Patch(color=REF_COLORS["top_p0.95"], label="top_p0.95 t=0.2"),
        Patch(color=REF_COLORS["top_p0.9"], label="top_p0.9 t=1.0"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_overview(rows: list[dict], output_path: Path) -> None:
    """Faceted line plot: pass@k, cover@t, cover@t (distinct)."""
    plot_rows = _plot_rows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Select key configs to plot (avoid too many lines)
    key_configs = []
    for r in plot_rows:
        # All T1/T2 configs
        if r["_group"] == "t1t2":
            key_configs.append(r)
        # Key baselines (pless t=0.6/1.0, temp, top_p0.95)
        elif r["_method"] in ("pless", "temp", "top_p0.95") and r["_t1"] in (0.6, 0.7, 1.0, 0.2):
            key_configs.append(r)

    for r in key_configs:
        if r["_group"] == "t1t2":
            color = T2_COLORS.get(r["_t2"], "#888888")
            ls = {0.6: "-", 0.8: "--", 1.0: ":"}[r["_t1"]]
            lw = 2.0
        else:
            color = BASELINE_COLORS.get(r["_method"], "#888888")
            ls = "-"
            lw = 2.5

        label = _display_name(r)

        # pass@k
        ks = sorted(r["pass_at_k"].keys(), key=lambda x: int(x))
        pk_vals = [r["pass_at_k"][k] * 100 for k in ks]
        axes[0].plot([int(k) for k in ks], pk_vals, color=color, linestyle=ls,
                     linewidth=lw, marker="o", markersize=4, label=label)

        # cover@t
        ts = sorted(r["cover_at_t"].keys(), key=lambda x: float(x))
        ct_vals = [r["cover_at_t"][t] for t in ts]
        axes[1].plot([float(t) for t in ts], ct_vals, color=color, linestyle=ls,
                     linewidth=lw, marker="o", markersize=4)

        # cover@t distinct
        ts_d = sorted(r["cover_at_t_distinct"].keys(), key=lambda x: float(x))
        ctd_vals = [r["cover_at_t_distinct"][t] for t in ts_d]
        axes[2].plot([float(t) for t in ts_d], ctd_vals, color=color, linestyle=ls,
                     linewidth=lw, marker="o", markersize=4)

    axes[0].set_title("pass@k", fontsize=11)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Percentage (%)")
    axes[1].set_title("cover@t", fontsize=11)
    axes[1].set_xlabel("t")
    axes[2].set_title("cover@t (distinct)", fontsize=11)
    axes[2].set_xlabel("t")

    for ax in axes:
        ax.grid(alpha=0.3)

    # Unified legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "MBPP: Metrics Overview — P-less T1/T2 (Qwen2.5-Coder-3B, BigCode)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_scatter(rows: list[dict], output_path: Path) -> None:
    """Pareto scatter: pass@1 vs codebleu_diversity."""
    plot_rows = _plot_rows(rows)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 7))

    points = []
    for r in plot_rows:
        p1 = r["pass_at_k"]["1"] * 100
        div = r.get("codebleu_diversity", 0)
        if r["_group"] == "t1t2":
            color = T1_COLORS.get(r["_t1"], "#888888")
            marker = T2_MARKERS.get(r["_t2"], "x")
            size = 160
        else:
            color = BASELINE_COLORS.get(r["_method"], "#888888")
            marker = "^"
            size = 120

        ax.scatter(p1, div, s=size, color=color, marker=marker,
                   edgecolors="black", linewidth=0.8, zorder=3)

        # Label each point
        label = _display_name(r)
        ax.annotate(label, (p1, div), textcoords="offset points",
                    xytext=(5, 5), fontsize=6, alpha=0.8)
        points.append((p1, div))

    # Pareto frontier
    pts = sorted(points, key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    max_div = -1
    for x, y in pts:
        if y > max_div:
            frontier_x.append(x)
            frontier_y.append(y)
            max_div = y
    if len(frontier_x) > 1:
        ax.plot(frontier_x, frontier_y, "--", color="gray", alpha=0.4, linewidth=1)

    ax.set_xlabel("pass@1 (%)", fontsize=11)
    ax.set_ylabel("CodeBLEU Diversity", fontsize=11)
    ax.set_title(
        "MBPP: Correctness vs Diversity — P-less T1/T2 (Qwen2.5-Coder-3B)",
        fontsize=12,
    )
    ax.grid(alpha=0.3)

    # Legend
    legend_items = []
    # T1 colours
    legend_items.append(Line2D([0], [0], color="none", label="$\\bf{T1\\ (color)}$"))
    for t1, c in T1_COLORS.items():
        legend_items.append(Patch(color=c, label=f"T1={t1}"))
    # T2 markers
    legend_items.append(Line2D([0], [0], color="none", label="$\\bf{T2\\ (shape)}$"))
    for t2, mk in T2_MARKERS.items():
        legend_items.append(
            Line2D([0], [0], marker=mk, color="gray", linestyle="None",
                   markersize=8, label=f"T2={t2}")
        )
    # Baselines
    legend_items.append(Line2D([0], [0], color="none", label="$\\bf{Baselines\\ (▲)}$"))
    for method, c in BASELINE_COLORS.items():
        legend_items.append(Patch(color=c, label=method))

    ax.legend(handles=legend_items, loc="upper left", fontsize=7.5, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_t2_effect_heatmap(rows: list[dict], output_path: Path) -> None:
    """Heatmap: T1 × T2 grid showing pass@1 and struct_div."""
    t1_vals = [0.6, 0.8, 1.0]
    t2_vals = ["—", "2.0", "5.0"]

    # Build lookup
    lookup = {(r["_method"], r["_t1"], r["_t2"]): r for r in rows if r["_method"] == "pless"}

    pass1_grid = np.full((3, 3), np.nan)
    sdiv_grid = np.full((3, 3), np.nan)

    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            r = lookup.get(("pless", t1, t2))
            if r:
                pass1_grid[i, j] = r["pass_at_k"]["1"] * 100
                sdiv_grid[i, j] = r.get("structural_diversity", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, grid, title, fmt, cmap in [
        (axes[0], pass1_grid, "pass@1 (%)", ".1f", "YlOrRd"),
        (axes[1], sdiv_grid, "struct_div", ".4f", "YlGnBu"),
    ]:
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=np.nanmin(grid) * 0.95,
                       vmax=np.nanmax(grid) * 1.02)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"T2={t}" for t in t2_vals], fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"T1={t}" for t in t1_vals], fontsize=9)
        ax.set_title(title, fontsize=11)

        # Annotate cells
        for i in range(3):
            for j in range(3):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if val > np.nanmean(grid) else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "T1 × T2 Interaction: P-less on Qwen2.5-Coder-3B (MBPP-full)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading metrics...")
    rows = load_all_metrics()
    print(f"  Loaded {len(rows)} configs ({sum(1 for r in rows if r['_group'] == 't1t2')} T1/T2, "
          f"{sum(1 for r in rows if r['_group'] == 'baseline')} baselines)")

    # Generate report
    report = generate_report(rows)
    report_path = OUTPUT_DIR / "t1_t2_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"Report written to {report_path}")

    # Generate plots
    plot_pass_at_1_bars(rows, FIGURES_DIR / "pass_at_1_comparison.png")
    print(f"Plot: {FIGURES_DIR / 'pass_at_1_comparison.png'}")

    plot_metrics_overview(rows, FIGURES_DIR / "metrics_overview.png")
    print(f"Plot: {FIGURES_DIR / 'metrics_overview.png'}")

    plot_pareto_scatter(rows, FIGURES_DIR / "pareto_correctness_diversity.png")
    print(f"Plot: {FIGURES_DIR / 'pareto_correctness_diversity.png'}")

    plot_t2_effect_heatmap(rows, FIGURES_DIR / "t2_effect_heatmap.png")
    print(f"Plot: {FIGURES_DIR / 't2_effect_heatmap.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
