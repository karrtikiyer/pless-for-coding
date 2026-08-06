#!/usr/bin/env python
"""Figure for Paper B §7: pass@1 and non-termination vs the Rényi order k, both models.

Reads the live comparison tables (docs/decoder_comparison_cot_apps_{qwen3,deepseek}.md) so the
figure never drifts from Table 1. Left panel: pass@1 vs k (with the k=2 default marked and each
model's best swept-temperature as a dashed line). Right panel: non-termination % vs k. Lower k =
looser filter. Writes paper/paperB/figures/fig_gk_sweep.png.
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = {
    "DeepSeek-R1-Distill": "docs/decoder_comparison_cot_apps_deepseek.md",
    "Qwen3-8B": "docs/decoder_comparison_cot_apps_qwen3.md",
}
COLORS = {"DeepSeek-R1-Distill": "#0072B2", "Qwen3-8B": "#D55E00"}  # colorblind-safe


def parse(path):
    """Return dict: gk=[(k,pass1,nonterm)], default=(pass1,nonterm), best_temp_pass1."""
    gk, temps, default = [], [], None
    for line in open(path):
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("Config",):
            continue
        name = cells[0]
        try:
            p1 = float(cells[2]); nonterm = float(cells[6])
        except ValueError:
            continue
        m = re.match(r"G_k=([0-9.]+)", name)
        if m:
            gk.append((float(m.group(1)), p1, nonterm))
        elif "temp" in name.lower():
            temps.append(p1)
        elif "pless" in name.lower() and ("α=2" in name or "@α2" in name) and "norm" not in name.lower():
            default = (p1, nonterm)
    gk.sort()
    return {"gk": gk, "default": default, "best_temp": max(temps) if temps else None}


def main():
    data = {m: parse(p) for m, p in DOCS.items()}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.4))

    for model, d in data.items():
        c = COLORS[model]
        ks = [k for k, _, _ in d["gk"]]
        p1 = [p for _, p, _ in d["gk"]]
        nt = [n for _, _, n in d["gk"]]
        axL.plot(ks, p1, "o-", color=c, label=model)
        axR.plot(ks, nt, "o-", color=c, label=model)
        if d["default"]:
            axL.scatter([2.0], [d["default"][0]], color=c, marker="X", s=90, zorder=5,
                        edgecolor="black", linewidth=0.6)
            axR.scatter([2.0], [d["default"][1]], color=c, marker="X", s=90, zorder=5,
                        edgecolor="black", linewidth=0.6)
        if d["best_temp"] is not None:
            axL.axhline(d["best_temp"], color=c, ls="--", lw=1, alpha=0.7)

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8", "1.6", "2"])
        ax.set_xlabel(r"Rényi order $k$  (lower = looser; $\times$ = default $k{=}2$)")
        ax.grid(True, alpha=0.25)
    axL.set_ylabel("pass@1"); axL.set_title("pass@1 vs. order (dashed = best swept temp)")
    axR.set_ylabel("non-termination %"); axR.set_title("looping vs. order")
    axL.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = "paper/paperB/figures/fig_gk_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    for m, d in data.items():
        print(f"  {m}: default={d['default']} best_temp={d['best_temp']} arms={len(d['gk'])}")


if __name__ == "__main__":
    main()
