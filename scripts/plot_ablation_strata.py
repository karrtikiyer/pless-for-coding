#!/usr/bin/env python
"""Ablation figure candidates: Δpass@1 by baseline-difficulty stratum, both models, at each
model's optimum k (Qwen k=0.1, DeepSeek k=0.4) vs the k=2 default. Renders TWO variants for
comparison: grouped bars + CI whiskers, and a dot/forest plot + CI. Per-stratum 95% CI is a
two-level bootstrap (resample problems in the stratum AND the 10 draws, base/arm independently),
matching the paper's headline CI method. No GPU.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"DeepSeek-R1-Distill": "#0072B2", "Qwen3-8B": "#D55E00"}   # validated, == Fig 2
CFG = {
    "DeepSeek-R1-Distill": (
        "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_think_t1.0_t1.0_metrics.json",
        "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.4_t1.0_metrics.json", "k=0.4"),
    "Qwen3-8B": (
        "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json",
        "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.1_t1.0_metrics.json", "k=0.1"),
}
STRATA = ["dead", "hard", "mid", "easy"]
SLABEL = {"dead": "dead\n(0)", "hard": "hard\n(0,.3]", "mid": "mid\n(.3,.7]", "easy": "easy\n(.7,1]"}


def stratum(p1):
    return "dead" if p1 == 0 else "hard" if p1 <= 0.3 else "mid" if p1 <= 0.7 else "easy"


def by_id(path):
    m = json.load(open(path))
    return {t["task_id"]: np.array(t["pass_results"], dtype=bool) for t in m["per_task"]}


def stratum_stats(model, boot=2000, seed=0):
    bpath, apath, _ = CFG[model]
    B, A = by_id(bpath), by_id(apath)
    ids = sorted(set(B) & set(A))
    rng = np.random.default_rng(seed)
    out = {}
    for s in STRATA:
        sel = [i for i in ids if stratum(B[i].mean()) == s]
        n = len(sel)
        Bs = np.array([B[i] for i in sel]); As = np.array([A[i] for i in sel])  # (n,10)
        mean = float((As.mean(1) - Bs.mean(1)).mean()) if n else 0.0
        # two-level bootstrap
        lo = hi = mean
        if n:
            nd = Bs.shape[1]
            pi = rng.integers(0, n, (boot, n))
            dib = rng.integers(0, nd, (boot, n, nd)); dia = rng.integers(0, nd, (boot, n, nd))
            br = np.take_along_axis(Bs[pi], dib, 2).mean(2)
            ar = np.take_along_axis(As[pi], dia, 2).mean(2)
            reps = (ar - br).mean(1)
            lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
        out[s] = (mean, lo, hi, n)
    return out


def main():
    data = {m: stratum_stats(m) for m in CFG}
    models = list(CFG)
    x = np.arange(len(STRATA))

    # ---------- Variant A: grouped bars + CI whiskers ----------
    figA, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.38
    for j, m in enumerate(models):
        vals = [data[m][s][0] for s in STRATA]
        err = np.array([[data[m][s][0] - data[m][s][1] for s in STRATA],
                        [data[m][s][2] - data[m][s][0] for s in STRATA]])
        hatch = "///" if j == 1 else None
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=err, capsize=3, color=COL[m],
               edgecolor="black", linewidth=0.5, hatch=hatch,
               label=f"{m} ({CFG[m][2]})", error_kw=dict(lw=1))
        for xi, s in zip(x, STRATA):
            ax.annotate(f"n={data[m][s][3]}", (xi + (j - 0.5) * w, 0), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=6, color="gray", rotation=0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SLABEL[s] for s in STRATA])
    ax.set_ylabel(r"$\Delta$pass@1 vs. $k{=}2$")
    ax.set_title("Where the gain lands: Δpass@1 by baseline difficulty")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.15)
    figA.tight_layout(); figA.savefig("paper/paperB/figures/fig_ablation_strata_bars.png", dpi=200, bbox_inches="tight")

    # ---------- Variant B: dot / forest plot + CI ----------
    figB, ax = plt.subplots(figsize=(6.2, 3.4))
    off = 0.12
    for j, m in enumerate(models):
        xs = x + (j - 0.5) * 2 * off
        vals = [data[m][s][0] for s in STRATA]
        lo = [data[m][s][1] for s in STRATA]; hi = [data[m][s][2] for s in STRATA]
        err = np.array([[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]])
        ax.errorbar(xs, vals, yerr=err, fmt="o", ms=7, color=COL[m], capsize=3, lw=1.5,
                    markeredgecolor="black", markeredgewidth=0.5, label=f"{m} ({CFG[m][2]})")
    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xticks(x); ax.set_xticklabels([SLABEL[s] for s in STRATA])
    ax.set_ylabel(r"$\Delta$pass@1 vs. $k{=}2$ (95% CI)")
    ax.set_title("Where the gain lands: Δpass@1 by baseline difficulty")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.15)
    figB.tight_layout(); figB.savefig("paper/paperB/figures/fig_ablation_strata_dots.png", dpi=200, bbox_inches="tight")

    for m in models:
        print(m, {s: f"{data[m][s][0]:+.3f}[{data[m][s][1]:+.3f},{data[m][s][2]:+.3f}] n={data[m][s][3]}" for s in STRATA})
    print("wrote fig_ablation_strata_bars.png and fig_ablation_strata_dots.png")


if __name__ == "__main__":
    main()
