"""Truncation vs quality tradeoff scatter — Qwen3-8B, APPS ATCODER-interview (252).

Turns the 16-config comparison table into the picture the slide is making: pless defaults
sit in the bad corner (high truncation, low pass@1 / diversity); the fixes (alpha-up,
temp-up, loop-force) move out of it. The reference is Qwen3's OFFICIAL recommended sampler
(top_p0.95 + top_k20 @ T0.6) — singled out with reference lines so "who beats it" is clear.

Numbers transcribed from committed source-of-truth docs (NOT memory):
  - docs/decoder_comparison_cot_apps_qwen3.md  (14 temp / alpha / T2.0 / default configs)
  - docs/loopforce_w1200_comparison_apps_qwen3.md  (the two loop-force rows)

Run: MPLBACKEND=Agg PYTHONPATH=. uv run python scripts/plot_truncation_tradeoff.py
"""
import os
import matplotlib.pyplot as plt

G_QWEN = "Qwen recommended (p0.95+k20 @T0.6)"
G_TEMP = "other temp / top-p / top-k"
G_ALPHA = "pless α↑ / T↑ (fix)"
G_LOOP = "pless loop-force (fix)"
G_DEF = "pless default (problem)"

# (label, pass@1, pass@10, cb_div, trunc%, group)
DATA = [
    ("Qwen rec", 0.680, 0.829, 0.4681, 1.0, G_QWEN),       # top_p0.95+k20 @T0.6
    ("temp p0.95 @T1.0", 0.705, 0.821, 0.4965, 0.2, G_TEMP),
    ("temp k20 @T1.0",   0.700, 0.841, 0.5017, 0.0, G_TEMP),
    ("temp @T0.6",       0.699, 0.841, 0.4757, 0.4, G_TEMP),
    ("top_k @T0.6",      0.698, 0.821, 0.4789, 0.8, G_TEMP),
    ("top_p @T0.6",      0.695, 0.841, 0.4797, 1.2, G_TEMP),
    ("pless α=4",        0.696, 0.821, 0.4689, 1.4, G_ALPHA),
    ("pless T2.0",       0.694, 0.821, 0.4609, 0.2, G_ALPHA),
    ("pless α=5",        0.686, 0.833, 0.4746, 0.6, G_ALPHA),
    ("pless α=3",        0.676, 0.806, 0.4663, 2.7, G_ALPHA),
    ("pless loop-force", 0.653, 0.813, 0.4555, 1.5, G_LOOP),
    ("pless-norm loop-force", 0.651, 0.806, 0.4708, 1.6, G_LOOP),
    ("pless_norm @α2",   0.629, 0.829, 0.4573, 16.0, G_DEF),
    ("pless @α2",        0.625, 0.825, 0.4528, 14.5, G_DEF),
    ("pless_norm @T0.6", 0.619, 0.806, 0.4513, 18.4, G_DEF),
    ("pless @T0.6",      0.615, 0.825, 0.4550, 19.0, G_DEF),
]
COLORS = {G_QWEN: "black", G_TEMP: "#9e9e9e", G_ALPHA: "#1f77b4", G_LOOP: "#2ca02c", G_DEF: "#d62728"}
MARK = {G_QWEN: "*", G_TEMP: "o", G_ALPHA: "s", G_LOOP: "^", G_DEF: "X"}
SIZE = {G_QWEN: 320, G_TEMP: 90, G_ALPHA: 120, G_LOOP: 120, G_DEF: 120}
LABEL = {"pless @α2", "pless @T0.6", "pless α=5", "pless loop-force"}  # star named in legend + on the line
REC = {1: 0.680, 3: 0.4681}  # Qwen-recommended pass@1 / cb_div reference values


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, yidx, ylab, ytitle in (
        (ax1, 1, "pass@1", "Accuracy vs truncation"),
        (ax2, 3, "self-CodeBLEU diversity (cb_div)", "Diversity vs truncation"),
    ):
        # reference line at the Qwen-recommended value: points above it beat the rec
        ax.axhline(REC[yidx], ls="--", c="black", lw=1, alpha=0.6, zorder=1)
        ax.text(ax.get_xlim()[1], REC[yidx], " Qwen-rec", va="center", fontsize=8, color="black")
        for g in (G_DEF, G_TEMP, G_ALPHA, G_LOOP, G_QWEN):
            xs = [r[4] for r in DATA if r[5] == g]
            ys = [r[yidx] for r in DATA if r[5] == g]
            ax.scatter(xs, ys, s=SIZE[g], c=COLORS[g], marker=MARK[g], edgecolors="black",
                       linewidths=0.6, label=g, zorder=3, alpha=0.92)
        for r in DATA:
            if r[0] in LABEL:
                ax.annotate(r[0], (r[4], r[yidx]), fontsize=7.5,
                            xytext=(6, 4), textcoords="offset points", zorder=4)
        ax.set_xlabel("truncation %  (lower = better →)")
        ax.set_ylabel(ylab)
        ax.set_title(ytitle, fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.margins(x=0.13, y=0.13)

    ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    fig.suptitle("pless α=5 beats Qwen's recommended sampler (p0.95+k20 @T0.6) on pass@1, pass@10, diversity AND truncation\n"
                 "Qwen3-8B · APPS ATCODER-interview (252) · n=10 · thinking on  (points above the dashed line beat Qwen-rec)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    out = "docs/figures/truncation_tradeoff_qwen3.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
