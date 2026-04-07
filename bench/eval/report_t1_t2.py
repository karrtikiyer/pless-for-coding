"""Generate T1/T2 post-truncation temperature analysis report and plots.

Compares P-less T1/T2 results from `results/full_mbpp_pre_post_temp_pless/`
against baselines from `results/pless_full_mbpp_results/analysis/consolidated_metrics/`.

Usage:
    uv run python -m bench.eval.report_t1_t2             # base model (default)
    uv run python -m bench.eval.report_t1_t2 --instruct  # instruct model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths — base model (default)
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
# Paths — instruct model
# ---------------------------------------------------------------------------
INSTRUCT_METRICS_DIR = Path(
    "results/full_mbpp_pre_post_temp_pless/Qwen--Qwen2.5-Coder-3B-Instruct/metrics"
)
# Use the base model T1/T2 metrics as reference for cross-model comparison
BASE_T1T2_METRICS_DIR = T1T2_METRICS_DIR
BASE_CONSOLIDATED_DIR = BASELINE_METRICS_DIR
INSTRUCT_OUTPUT_DIR = Path("results/full_mbpp_pre_post_temp_pless/analysis/instruct")
INSTRUCT_FIGURES_DIR = INSTRUCT_OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Colour / style config
# ---------------------------------------------------------------------------
# T2 colours
T2_COLORS = {"—": "#2B6CB0", "2.0": "#C05621", "3.0": "#D69E2E", "4.0": "#9B2C2C", "5.0": "#2F855A"}
# Baseline method colours
BASELINE_COLORS = {
    "pless": "#6B46C1",
    "pless_norm": "#B7791F",
    "temp": "#9B2C2C",
    "top_p0.9": "#2C7A7B",
    "top_p0.95": "#D53F8C",
}
# T2 marker shapes (for Pareto scatter)
T2_MARKERS = {"—": "o", "2.0": "s", "3.0": "^", "4.0": "v", "5.0": "D"}
# T1 colours (for Pareto scatter)
T1_COLORS = {0.6: "#2B6CB0", 0.8: "#C05621", 1.0: "#2F855A", 1.5: "#D69E2E", 2.0: "#9B2C2C", 3.0: "#6B46C1"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_t1_t2(stem: str) -> tuple[str, float, str]:
    """Parse filename stem into (method_label, T1, T2).

    Handles both base model (with ``_bigcode_``) and instruct (without) naming:

      pless_bigcode_t0.8        -> ("pless", 0.8, "—")
      pless_pt2.0_bigcode_t0.6  -> ("pless", 0.6, "2.0")
      pless_t0.6                -> ("pless", 0.6, "—")        (instruct)
      pless_pt2.0_t2.0          -> ("pless", 2.0, "2.0")      (instruct T2)
      temp_t0.2                 -> ("temp", 0.2, "—")         (instruct)
      greedy_t1.0               -> ("greedy", 1.0, "—")       (instruct)
      top_p0.95_t0.2            -> ("top_p0.95", 0.2, "—")    (instruct)
      top_p0.95_bigcode_t0.2    -> ("top_p0.95", 0.2, "—")    (base)
    """
    t1 = float(stem.rsplit("_t", 1)[1])
    t2 = "—"
    if "_pt" in stem:
        t2 = stem.split("_pt")[1].split("_")[0]
    # Method name
    method_part = stem.rsplit("_t", 1)[0]  # e.g. pless_pt2.0_bigcode or pless
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
        "1. **T2 effect is regime-dependent on the base model.** At T1=0.6, T2 adds only "
        "+0.005–0.007 struct_div while costing ~1pp pass@1 (poor exchange rate). At T1=0.8, "
        "T2=5.0 is a notable anomaly: +0.1pp pass@1 (within noise, but no cost) AND +0.032 "
        "codebleu_div — the one case where T2 appears genuinely beneficial. At T1=1.0, "
        "T2 *reverses direction* and decreases diversity, suggesting the flattening causes "
        "convergence on a few popular alternatives rather than spreading across many."
    )
    lines.append("")
    lines.append(
        "2. **T1 (pre-truncation temperature) does most of the work.** The diversity jump from "
        "T1=0.6→0.8→1.0 is substantial (struct_div 0.097→0.167→0.255), matching the effect "
        "of raising temperature in standard P-less. T1 controls pruning aggressiveness by "
        "shaping the distribution before the collision entropy threshold is computed. T1 is "
        "2–14× more efficient than T2 at converting pass@1 into diversity."
    )
    lines.append("")
    lines.append(
        "3. **Best new config: pless T1=0.8 (no T2).** At 58.7% pass@1, 0.167 struct_div, "
        "0.298 codebleu_div, it matches top_p0.95/t=0.2 (58.2%) while providing a useful "
        "diversity level — with zero hyperparameters beyond T1."
    )
    lines.append("")
    lines.append(
        "4. **Implication for instruct models:** T2 alone cannot rescue diversity on instruct "
        "models where P-less at low T1 leaves ~1 survivor. The instruct experiment needs high "
        "T1 (>1.0) to open the distribution first. However, the T1=0.8/T2=5.0 anomaly "
        "suggests T2 may have a narrow sweet spot when the survivor set is moderate — worth "
        "testing at high T1 on instruct where more survivors exist."
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
# Instruct-specific loading
# ---------------------------------------------------------------------------
def load_instruct_metrics(
    instruct_metrics_dir: Path | None = None,
    base_metrics_dir: Path | None = None,
) -> list[dict]:
    """Load instruct experiment metrics plus base model references."""
    metrics_dir = instruct_metrics_dir or INSTRUCT_METRICS_DIR
    base_dir = base_metrics_dir or BASE_CONSOLIDATED_DIR
    rows = []

    # Instruct experiment configs
    if metrics_dir.exists():
        for f in sorted(metrics_dir.glob("*_metrics.json")):
            m = json.loads(f.read_text())
            stem = f.stem.replace("_metrics", "")
            method, t1, t2 = _parse_t1_t2(stem)
            m["_method"] = method
            m["_t1"] = t1
            m["_t2"] = t2
            m["_group"] = "instruct"
            m["_stem"] = stem
            rows.append(m)
    else:
        print(f"WARNING: {metrics_dir} does not exist")

    # Base model consolidated baselines (pless t=0.6/1.0, temp t=0.7, top_p)
    if base_dir.exists():
        for f in sorted(base_dir.glob("*_metrics.json")):
            m = json.loads(f.read_text())
            stem = f.stem.replace("_metrics", "")
            method, t1, t2 = _parse_t1_t2(stem)
            m["_method"] = method
            m["_t1"] = t1
            m["_t2"] = t2
            m["_group"] = "base"
            m["_stem"] = stem
            rows.append(m)

    return rows


# ---------------------------------------------------------------------------
# Instruct report generation
# ---------------------------------------------------------------------------
def generate_instruct_report(rows: list[dict], model_name: str = "Qwen2.5-Coder-3B-Instruct") -> str:
    """Build the instruct model analysis report."""
    instruct_rows = [r for r in rows if r["_group"] == "instruct"]
    base_rows = [r for r in rows if r["_group"] == "base"]
    lines = []

    lines.append("# P-less High-T1 on Instruct Model: MBPP Analysis")
    lines.append("")
    lines.append(
        f"**Model:** {model_name} | **Dataset:** MBPP-full (500 tasks × 10 samples) | "
        "**Prompt:** Chat template (auto-detected)"
    )
    lines.append("")
    lines.append(
        "**Hypothesis:** At matched diversity levels, high-T1/P-less achieves higher pass@1 than "
        "plain temperature on instruct models — P-less acts as a quality filter."
    )
    lines.append("")
    lines.append(
        "This experiment tests whether high T1 (>1.0) can open "
        "the peaked instruct distribution enough for P-less to provide value."
    )
    lines.append("")

    # --- Full metrics table ---
    lines.append("## Full Metrics Comparison")
    lines.append("")
    lines.append(
        "| # | Config | Group | T1 | T2 | pass@1 | pass@3 | pass@5 | pass@10 | "
        "cover@0.7 | struct_div | codebleu_div |"
    )
    lines.append(
        "|---|--------|-------|----|----|--------|--------|--------|---------|"
        "-----------|------------|--------------|"
    )

    sorted_rows = sorted(rows, key=lambda r: -r["pass_at_k"]["1"])
    for i, r in enumerate(sorted_rows, 1):
        pk = r["pass_at_k"]
        c07 = r["cover_at_t"].get("0.7", 0) if r.get("cover_at_t") else 0
        sdiv = r.get("structural_diversity", 0)
        cbdiv = r.get("codebleu_diversity", 0)
        group_label = "**instruct**" if r["_group"] == "instruct" else "base"
        pk3 = f"{pk['3']*100:.1f}" if "3" in pk else "—"
        pk5 = f"{pk['5']*100:.1f}" if "5" in pk else "—"
        pk10 = f"{pk['10']*100:.1f}" if "10" in pk else "—"
        lines.append(
            f"| {i} | {_display_name(r)} | {group_label} | {r['_t1']} | {r['_t2']} | "
            f"{pk['1']*100:.1f} | {pk3} | {pk5} | {pk10} | "
            f"{c07:.1f} | {sdiv:.4f} | {cbdiv:.4f} |"
        )
    lines.append("")

    # --- T1 sweep analysis ---
    lines.append("## T1 Sweep: Does High T1 Restore Diversity on Instruct?")
    lines.append("")
    lines.append(
        "The core question: at what T1 does P-less produce non-zero diversity on the instruct model?"
    )
    lines.append("")
    lines.append("| T1 | pass@1 | pass@10 | struct_div | codebleu_div | cover@0.7 |")
    lines.append("|----|--------|---------|------------|--------------|-----------|")

    pless_instruct = sorted(
        [r for r in instruct_rows if r["_method"] == "pless" and r["_t2"] == "—"],
        key=lambda r: r["_t1"],
    )
    for r in pless_instruct:
        pk = r["pass_at_k"]
        c07 = r["cover_at_t"].get("0.7", 0) if r.get("cover_at_t") else 0
        sdiv = r.get("structural_diversity", 0)
        cbdiv = r.get("codebleu_diversity", 0)
        lines.append(
            f"| {r['_t1']} | {pk['1']*100:.1f}% | {pk['10']*100:.1f}% | "
            f"{sdiv:.4f} | {cbdiv:.4f} | {c07:.1f} |"
        )
    lines.append("")

    # --- P-less vs temperature at matched diversity ---
    lines.append("## P-less as Quality Filter: Matched-Diversity Comparison")
    lines.append("")
    lines.append(
        "Compare P-less (T1=X) against temperature (T=Y) at similar diversity levels. "
        "If P-less achieves higher pass@1, it is acting as a quality filter."
    )
    lines.append("")

    temp_instruct = {
        r["_t1"]: r for r in instruct_rows if r["_method"] == "temp"
    }
    pless_lookup = {
        r["_t1"]: r for r in instruct_rows if r["_method"] == "pless" and r["_t2"] == "—"
    }

    lines.append("| P-less config | P-less pass@1 | P-less sdiv | Nearest temp | Temp pass@1 | Temp sdiv | Δ pass@1 |")
    lines.append("|---------------|---------------|-------------|--------------|-------------|-----------|----------|")

    for t1 in [1.5, 2.0, 3.0]:
        pr = pless_lookup.get(t1)
        if pr is None:
            continue
        psdiv = pr.get("structural_diversity", 0)
        pp1 = pr["pass_at_k"]["1"] * 100

        # Find nearest temp config by diversity
        best_temp = None
        best_dist = float("inf")
        for tr in temp_instruct.values():
            tsdiv = tr.get("structural_diversity", 0)
            dist = abs(tsdiv - psdiv)
            if dist < best_dist:
                best_dist = dist
                best_temp = tr

        if best_temp:
            tp1 = best_temp["pass_at_k"]["1"] * 100
            tsdiv = best_temp.get("structural_diversity", 0)
            lines.append(
                f"| pless T1={t1} | {pp1:.1f}% | {psdiv:.4f} | "
                f"temp t={best_temp['_t1']} | {tp1:.1f}% | {tsdiv:.4f} | "
                f"{pp1 - tp1:+.1f}pp |"
            )

    lines.append("")

    # --- T2 effect sections — one per T1 that has T2 configs ---
    # Find all T1 values that have T2 configs
    t2_by_t1: dict[float, list[dict]] = {}
    for r in instruct_rows:
        if r["_method"] == "pless" and r["_t2"] != "—":
            t1_val = r["_t1"]
            t2_by_t1.setdefault(t1_val, []).append(r)

    for t1_val in sorted(t2_by_t1.keys()):
        t2_configs = sorted(t2_by_t1[t1_val], key=lambda r: float(r["_t2"]))
        t2_base = pless_lookup.get(t1_val)
        if not t2_base:
            continue

        lines.append(f"## T2 Effect at T1={t1_val}")
        lines.append("")

        bp1 = t2_base["pass_at_k"]["1"] * 100
        bsd = t2_base.get("structural_diversity", 0)
        bcb = t2_base.get("codebleu_diversity", 0)
        lines.append(f"**Baseline:** pless T1={t1_val} (no T2): pass@1={bp1:.1f}%, struct_div={bsd:.4f}")
        lines.append("")
        lines.append("| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |")
        lines.append("|----|--------|----------|------------|--------|--------------|---------|")

        for r in t2_configs:
            p1 = r["pass_at_k"]["1"] * 100
            sd = r.get("structural_diversity", 0)
            cb = r.get("codebleu_diversity", 0)
            lines.append(
                f"| {r['_t2']} | {p1:.1f}% | {p1 - bp1:+.1f}pp | "
                f"{sd:.4f} | {sd - bsd:+.4f} | {cb:.4f} | {cb - bcb:+.4f} |"
            )
        lines.append("")

    # --- Industry comparison ---
    lines.append("## Industry Comparison")
    lines.append("")

    industry_configs = {}
    for r in instruct_rows:
        name = _display_name(r)
        if r["_method"] == "top_p0.95":
            industry_configs["top_p0.95 t=0.2"] = r
        elif r["_method"] == "temp" and abs(r["_t1"] - 0.2) < 0.01:
            industry_configs["temp t=0.2"] = r
        elif r["_method"] == "greedy":
            industry_configs["greedy"] = r

    # Find best pless config (highest pass@1 among pless with some diversity)
    pless_with_div = [
        r for r in pless_instruct
        if r.get("structural_diversity", 0) > 0.01
    ]
    best_pless = max(pless_with_div, key=lambda r: r["pass_at_k"]["1"]) if pless_with_div else None
    if not best_pless:
        best_pless = max(pless_instruct, key=lambda r: r["pass_at_k"]["1"]) if pless_instruct else None

    if best_pless:
        lines.append(f"**Best P-less config (with diversity):** {_display_name(best_pless)} "
                      f"(pass@1={best_pless['pass_at_k']['1']*100:.1f}%, "
                      f"struct_div={best_pless.get('structural_diversity', 0):.4f})")
        lines.append("")
        lines.append("| Config | pass@1 | struct_div | codebleu_div | Δ pass@1 vs best P-less |")
        lines.append("|--------|--------|------------|--------------|------------------------|")
        bp1 = best_pless["pass_at_k"]["1"] * 100
        for name, r in industry_configs.items():
            p1 = r["pass_at_k"]["1"] * 100
            sd = r.get("structural_diversity", 0)
            cb = r.get("codebleu_diversity", 0)
            lines.append(f"| {name} | {p1:.1f}% | {sd:.4f} | {cb:.4f} | {p1 - bp1:+.1f}pp |")
        lines.append("")

    # --- Key findings ---
    lines.append("## Key Findings")
    lines.append("")

    # Auto-generate key findings from data
    findings = []

    # Helper: lookup instruct pless configs
    pless_t1_map = {r["_t1"]: r for r in pless_instruct}

    # 1. T1 sweep trajectory
    if pless_instruct:
        low_t1 = min(pless_instruct, key=lambda r: r["_t1"])
        low_sdiv = low_t1.get("structural_diversity", 0)
        t20_r = pless_t1_map.get(2.0)
        t20_sdiv = t20_r.get("structural_diversity", 0) if t20_r else 0
        t30_r = pless_t1_map.get(3.0)
        t30_p1 = t30_r["pass_at_k"]["1"] * 100 if t30_r else 0
        findings.append(
            f"1. **High T1 does restore diversity on instruct models — but at a cost.** "
            f"P-less struct_div rises from {low_sdiv:.4f} (T1={low_t1['_t1']}) "
            f"to {t20_sdiv:.4f} (T1=2.0), confirming that "
            f"high T1 opens the peaked instruct distribution. However, T1=3.0 "
            f"{'catastrophically collapses' if t30_p1 < 10 else 'significantly degrades'} "
            f"correctness ({t30_p1:.1f}% pass@1)."
        )

    # 2. Hypothesis result — compute matched-diversity deltas
    delta_vals = []
    for t1 in [1.5, 2.0, 3.0]:
        pr = pless_t1_map.get(t1)
        if pr is None:
            continue
        psdiv = pr.get("structural_diversity", 0)
        pp1 = pr["pass_at_k"]["1"] * 100
        best_temp = None
        best_dist = float("inf")
        for tr in temp_instruct.values():
            tsdiv = tr.get("structural_diversity", 0)
            dist = abs(tsdiv - psdiv)
            if dist < best_dist:
                best_dist = dist
                best_temp = tr
        if best_temp:
            tp1 = best_temp["pass_at_k"]["1"] * 100
            delta_vals.append(pp1 - tp1)

    if delta_vals and all(d <= 0.5 for d in delta_vals):
        findings.append(
            f"2. **The quality-filter hypothesis is NOT confirmed.** At matched diversity, P-less does NOT "
            f"beat temperature — the Δ pass@1 ranges from {min(delta_vals):+.1f}pp to {max(delta_vals):+.1f}pp. "
            f"On instruct models, P-less and temperature perform similarly at similar diversity levels."
        )
    elif delta_vals:
        findings.append(
            f"2. **Mixed quality-filter results.** At matched diversity, Δ pass@1 ranges from "
            f"{min(delta_vals):+.1f}pp to {max(delta_vals):+.1f}pp. "
            f"{'P-less shows a slight advantage at some diversity levels.' if max(delta_vals) > 0.5 else 'No clear advantage for P-less.'}"
        )

    # 3. Sweet spot
    # Find configs with best balance: high pass@1 and non-trivial diversity
    sweet_candidates = [
        r for r in pless_instruct
        if r.get("structural_diversity", 0) > 0.05 and r["pass_at_k"]["1"] * 100 > 40
    ]
    if sweet_candidates:
        best_sweet = max(sweet_candidates, key=lambda r: r["pass_at_k"]["1"])
        greedy_r = next((r for r in instruct_rows if r["_method"] == "greedy"), None)
        greedy_p1 = greedy_r["pass_at_k"]["1"] * 100 if greedy_r else 0
        sweet_p1 = best_sweet["pass_at_k"]["1"] * 100
        sweet_sdiv = best_sweet.get("structural_diversity", 0)
        findings.append(
            f"3. **P-less T1={best_sweet['_t1']} is the instruct sweet spot.** "
            f"It gives {sweet_p1:.1f}% pass@1 with {sweet_sdiv:.4f} struct_div — "
            f"{'nearly as good as' if abs(sweet_p1 - greedy_p1) < 2 else f'{greedy_p1 - sweet_p1:.1f}pp below'} "
            f"greedy ({greedy_p1:.1f}%) while providing meaningful variety."
        )

    # 4. T2 effect — data-driven
    t2_at_t2_0 = [
        r for r in instruct_rows
        if r["_method"] == "pless" and r["_t2"] != "—" and abs(r["_t1"] - 2.0) < 0.01
    ]
    if t2_at_t2_0 and pless_t1_map.get(2.0):
        base_p1 = pless_t1_map[2.0]["pass_at_k"]["1"] * 100
        base_sdiv = pless_t1_map[2.0].get("structural_diversity", 0)
        t2_sorted = sorted(t2_at_t2_0, key=lambda r: float(r["_t2"]))
        best_t2 = max(t2_sorted, key=lambda r: r.get("structural_diversity", 0) - abs(r["pass_at_k"]["1"] * 100 - base_p1) * 0.01)
        worst_t2 = min(t2_sorted, key=lambda r: r["pass_at_k"]["1"])
        findings.append(
            f"4. **T2 effect at T1=2.0:** "
            f"T2 values tested: {', '.join(r['_t2'] for r in t2_sorted)}. "
            f"Best trade-off: T2={best_t2['_t2']} ({best_t2['pass_at_k']['1']*100:.1f}% pass@1, "
            f"{best_t2.get('structural_diversity', 0):.4f} sdiv). "
            f"Highest T2={worst_t2['_t2']} costs {worst_t2['pass_at_k']['1']*100 - base_p1:+.1f}pp pass@1."
        )

    # 4b. T2 at T1=1.0 if available
    t2_at_t1_0 = [
        r for r in instruct_rows
        if r["_method"] == "pless" and r["_t2"] != "—" and abs(r["_t1"] - 1.0) < 0.01
    ]
    if t2_at_t1_0 and pless_t1_map.get(1.0):
        base_p1_10 = pless_t1_map[1.0]["pass_at_k"]["1"] * 100
        t2_sorted_10 = sorted(t2_at_t1_0, key=lambda r: float(r["_t2"]))
        deltas_10 = [(r["_t2"], r["pass_at_k"]["1"] * 100 - base_p1_10) for r in t2_sorted_10]
        findings.append(
            f"5. **T2 effect at T1=1.0:** "
            f"T2 values tested: {', '.join(r['_t2'] for r in t2_sorted_10)}. "
            f"Pass@1 deltas: {', '.join(f'T2={t2}: {d:+.1f}pp' for t2, d in deltas_10)}. "
            f"{'T2 is mostly harmless at T1=1.0.' if all(abs(d) < 2 for _, d in deltas_10) else 'T2 has notable impact at T1=1.0.'}"
        )

    # 5/6. Industry comparison — data-driven
    greedy_r = next((r for r in instruct_rows if r["_method"] == "greedy"), None)
    if greedy_r:
        greedy_p1 = greedy_r["pass_at_k"]["1"] * 100
        top_configs = sorted(instruct_rows, key=lambda r: -r["pass_at_k"]["1"])[:5]
        top_strs = [f"{_display_name(r)} ({r['pass_at_k']['1']*100:.1f}%)" for r in top_configs]
        finding_num = len(findings) + 1
        findings.append(
            f"{finding_num}. **Top instruct pass@1 ({greedy_p1:.1f}% greedy).** "
            f"Top configs: {', '.join(top_strs)}."
        )

    for f in findings:
        lines.append(f)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Instruct-specific plots
# ---------------------------------------------------------------------------
def plot_instruct_t1_sweep(rows: list[dict], output_path: Path) -> None:
    """Two-panel line plot: T1 vs pass@1 and T1 vs struct_div for P-less on instruct."""
    instruct_pless = sorted(
        [r for r in rows if r["_group"] == "instruct" and r["_method"] == "pless" and r["_t2"] == "—"],
        key=lambda r: r["_t1"],
    )
    if not instruct_pless:
        return

    t1_vals = [r["_t1"] for r in instruct_pless]
    pass1_vals = [r["pass_at_k"]["1"] * 100 for r in instruct_pless]
    sdiv_vals = [r.get("structural_diversity", 0) for r in instruct_pless]
    cbdiv_vals = [r.get("codebleu_diversity", 0) for r in instruct_pless]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: pass@1 vs T1
    ax1.plot(t1_vals, pass1_vals, "o-", color="#2B6CB0", linewidth=2, markersize=8, label="P-less (instruct)")
    # Add temperature baselines as horizontal lines
    for r in rows:
        if r["_group"] == "instruct" and r["_method"] == "temp":
            ax1.axhline(y=r["pass_at_k"]["1"] * 100, color="#E53E3E", linestyle="--",
                        alpha=0.5, linewidth=1)
            ax1.text(max(t1_vals) + 0.05, r["pass_at_k"]["1"] * 100,
                     f"temp t={r['_t1']}", fontsize=7, va="center", color="#E53E3E")
    ax1.set_xlabel("T1 (pre-truncation temperature)", fontsize=11)
    ax1.set_ylabel("pass@1 (%)", fontsize=11)
    ax1.set_title("Correctness: P-less T1 Sweep", fontsize=12)
    ax1.grid(alpha=0.3)

    # Panel 2: diversity vs T1
    ax2.plot(t1_vals, sdiv_vals, "s-", color="#38A169", linewidth=2, markersize=8, label="struct_div")
    ax2.plot(t1_vals, cbdiv_vals, "D-", color="#805AD5", linewidth=2, markersize=8, label="codebleu_div")
    # Add temp baselines
    for r in rows:
        if r["_group"] == "instruct" and r["_method"] == "temp":
            ax2.axhline(y=r.get("structural_diversity", 0), color="#E53E3E", linestyle=":",
                        alpha=0.3, linewidth=1)
    ax2.set_xlabel("T1 (pre-truncation temperature)", fontsize=11)
    ax2.set_ylabel("Diversity", fontsize=11)
    ax2.set_title("Diversity: P-less T1 Sweep", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "P-less T1 Sweep on Instruct Model (MBPP-full)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_instruct_pass_at_1_bars(rows: list[dict], output_path: Path) -> None:
    """Horizontal bar chart: all instruct configs + greyed base model refs."""
    instruct_rows = [r for r in rows if r["_group"] == "instruct"]
    base_rows = [r for r in rows if r["_group"] == "base"]

    # Exclude pless_norm from base (not part of this experiment)
    base_rows = [r for r in base_rows if r["_method"] != "pless_norm"]

    all_rows = sorted(instruct_rows + base_rows, key=lambda r: r["pass_at_k"]["1"])

    labels = [f"{_display_name(r)} {'(instruct)' if r['_group'] == 'instruct' else '(base)'}"
              for r in all_rows]
    values = [r["pass_at_k"]["1"] * 100 for r in all_rows]

    METHOD_COLORS = {
        "pless": "#2B6CB0",
        "temp": "#E53E3E",
        "top_p0.95": "#D53F8C",
        "top_p0.9": "#805AD5",
        "greedy": "#2F855A",
    }

    colors = []
    for r in all_rows:
        base_color = METHOD_COLORS.get(r["_method"], "#888888")
        if r["_group"] == "base":
            colors.append("#C0C0C0")  # grey for base model refs
        else:
            colors.append(base_color)

    fig, ax = plt.subplots(figsize=(11, max(7, len(labels) * 0.4)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("pass@1 (%)", fontsize=11)
    ax.set_title("MBPP pass@1: Instruct High-T1 Experiment", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=7,
        )

    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="none", label="$\\bf{Instruct\\ experiment}$"),
        Patch(color=METHOD_COLORS["pless"], label="P-less"),
        Patch(color=METHOD_COLORS["temp"], label="Temperature"),
        Patch(color=METHOD_COLORS["top_p0.95"], label="Top-p"),
        Patch(color=METHOD_COLORS["greedy"], label="Greedy"),
        Patch(color="none", label=""),
        Patch(color="#C0C0C0", label="Base model reference"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_instruct_pareto(rows: list[dict], output_path: Path) -> None:
    """Pareto scatter: pass@1 vs codebleu_diversity — instruct primary, base greyed."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 7))

    METHOD_COLORS = {
        "pless": "#2B6CB0", "temp": "#E53E3E", "top_p0.95": "#D53F8C",
        "top_p0.9": "#805AD5", "greedy": "#2F855A",
    }

    for r in rows:
        if r["_method"] == "pless_norm":
            continue
        p1 = r["pass_at_k"]["1"] * 100
        div = r.get("codebleu_diversity", 0)

        if r["_group"] == "instruct":
            color = METHOD_COLORS.get(r["_method"], "#888888")
            marker = "o" if r["_t2"] == "—" else "s"
            size = 140
            alpha = 1.0
        else:
            color = "#C0C0C0"
            marker = "^"
            size = 80
            alpha = 0.6

        ax.scatter(p1, div, s=size, color=color, marker=marker,
                   edgecolors="black", linewidth=0.6, zorder=3, alpha=alpha)

        label = _display_name(r)
        if r["_group"] == "base":
            label = f"base: {label}"
        ax.annotate(label, (p1, div), textcoords="offset points",
                    xytext=(5, 5), fontsize=5.5, alpha=0.7)

    ax.set_xlabel("pass@1 (%)", fontsize=11)
    ax.set_ylabel("CodeBLEU Diversity", fontsize=11)
    ax.set_title(
        "Correctness vs Diversity: Instruct High-T1",
        fontsize=12,
    )
    ax.grid(alpha=0.3)

    legend_items = [
        Patch(color="none", label="$\\bf{Instruct}$"),
        Patch(color=METHOD_COLORS["pless"], label="P-less"),
        Patch(color=METHOD_COLORS["temp"], label="Temperature"),
        Patch(color=METHOD_COLORS["top_p0.95"], label="Top-p"),
        Patch(color=METHOD_COLORS["greedy"], label="Greedy"),
        Patch(color="none", label=""),
        Patch(color="#C0C0C0", label="Base model ref"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=7, label="No T2"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=7, label="With T2"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_instruct_t2_at_high_t1(rows: list[dict], output_path: Path) -> None:
    """Bar chart showing T2 impact at T1=2.0 on the instruct model."""
    instruct_rows = [r for r in rows if r["_group"] == "instruct"]
    t2_configs = [
        r for r in instruct_rows
        if r["_method"] == "pless" and abs(r["_t1"] - 2.0) < 0.01
    ]
    if not t2_configs:
        return

    t2_configs = sorted(t2_configs, key=lambda r: r["_t2"])
    labels = [f"T2={r['_t2']}" for r in t2_configs]
    pass1 = [r["pass_at_k"]["1"] * 100 for r in t2_configs]
    sdiv = [r.get("structural_diversity", 0) for r in t2_configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bar_palette = ["#2B6CB0", "#DD6B20", "#D69E2E", "#9B2C2C", "#38A169", "#6B46C1", "#805AD5"]
    colors = [bar_palette[i % len(bar_palette)] for i in range(len(labels))]

    ax1.bar(range(len(labels)), pass1, color=colors, edgecolor="white")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("pass@1 (%)", fontsize=11)
    ax1.set_title("pass@1 at T1=2.0", fontsize=11)
    for i, v in enumerate(pass1):
        ax1.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(range(len(labels)), sdiv, color=colors, edgecolor="white")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("struct_div", fontsize=11)
    ax2.set_title("Structural Diversity at T1=2.0", fontsize=11)
    for i, v in enumerate(sdiv):
        ax2.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "T2 Effect at High T1 (T1=2.0)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_instruct_vs_base(rows: list[dict], output_path: Path) -> None:
    """Side-by-side bars for matched configs: instruct vs base."""
    instruct_lookup = {
        (r["_method"], r["_t1"]): r
        for r in rows if r["_group"] == "instruct" and r["_t2"] == "—"
    }
    base_lookup = {
        (r["_method"], r["_t1"]): r
        for r in rows if r["_group"] == "base" and r["_t2"] == "—"
    }

    # Find matched pairs
    pairs = []
    for key in [("pless", 0.6), ("pless", 1.0), ("temp", 0.7)]:
        ir = instruct_lookup.get(key)
        # For temp on instruct, try matching temperature
        if key[0] == "temp" and ir is None:
            # Instruct has temp at 0.6, not 0.7 — skip if no match
            continue
        br = base_lookup.get(key)
        if ir and br:
            pairs.append((key, ir, br))

    if not pairs:
        return

    labels = [f"{m} t={t}" for (m, t), _, _ in pairs]
    instruct_vals = [ir["pass_at_k"]["1"] * 100 for _, ir, _ in pairs]
    base_vals = [br["pass_at_k"]["1"] * 100 for _, _, br in pairs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, instruct_vals, width, label="Instruct", color="#2B6CB0")
    bars2 = ax.bar(x + width / 2, base_vals, width, label="Base", color="#C0C0C0")

    ax.set_ylabel("pass@1 (%)", fontsize=11)
    ax.set_title("Instruct vs Base: Matched Configs (MBPP-full)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, instruct_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, base_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Generate T1/T2 analysis report")
    parser.add_argument("--instruct", action="store_true",
                        help="Generate instruct model analysis instead of base model")
    parser.add_argument("--model", default=None,
                        help="Model directory name (e.g. Qwen--Qwen2.5-Coder-7B-Instruct). "
                             "Overrides default paths for metrics and output.")
    return parser.parse_args()


def main():
    args = parse_cli_args()

    if args.instruct:
        # Determine paths based on --model flag
        if args.model:
            model_dir = args.model
            instruct_metrics = Path(f"results/full_mbpp_pre_post_temp_pless/{model_dir}/metrics")
            output_dir = Path(f"results/full_mbpp_pre_post_temp_pless/analysis/{model_dir}")
            model_display = model_dir.replace("--", "/")
        else:
            instruct_metrics = INSTRUCT_METRICS_DIR
            output_dir = INSTRUCT_OUTPUT_DIR
            model_display = "Qwen2.5-Coder-3B-Instruct"
        figures_dir = output_dir / "figures"

        print(f"Loading instruct + base reference metrics for {model_display}...")
        rows = load_instruct_metrics(
            instruct_metrics_dir=instruct_metrics,
        )
        n_instruct = sum(1 for r in rows if r["_group"] == "instruct")
        n_base = sum(1 for r in rows if r["_group"] == "base")
        print(f"  Loaded {len(rows)} configs ({n_instruct} instruct, {n_base} base references)")

        # Generate report
        report = generate_instruct_report(rows, model_name=model_display)
        report_path = output_dir / "instruct_t1_comparison_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report + "\n")
        print(f"Report written to {report_path}")

        # Generate plots
        plot_instruct_pass_at_1_bars(rows, figures_dir / "pass_at_1_comparison.png")
        print(f"Plot: {figures_dir / 'pass_at_1_comparison.png'}")

        plot_instruct_t1_sweep(rows, figures_dir / "t1_sweep.png")
        print(f"Plot: {figures_dir / 't1_sweep.png'}")

        plot_instruct_pareto(rows, figures_dir / "pareto_correctness_diversity.png")
        print(f"Plot: {figures_dir / 'pareto_correctness_diversity.png'}")

        plot_instruct_t2_at_high_t1(rows, figures_dir / "t2_effect_at_high_t1.png")
        print(f"Plot: {figures_dir / 't2_effect_at_high_t1.png'}")

        plot_instruct_vs_base(rows, figures_dir / "instruct_vs_base.png")
        print(f"Plot: {figures_dir / 'instruct_vs_base.png'}")

    else:
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
