"""Appendix figure: per-token L[P] = sum_i p_i^2 on Qwen3-8B thinking tokens (APPS-252, p-less T=1.0).

Answers the reviewer question "did we compute L[P] on the reasoning-model distributions, and what counts
as close to 1?": histogram of L[P] per thinking token, pooled over trace classes (the three classes overlap, so splitting them added nothing).
One bar colour, no threshold line, no in-chart text (user, 2026-11-06); the share above 1/2 is printed to stderr
for the paper text. Above 1/2 the p-less admission set is the top-1 token alone (p-less == greedy at that position):
  sum p_i^2 <= p_max  => the top token survives;
  a 2nd token would need p_2 >= L > 1/2, but p_1 >= p_2 and p_1 + p_2 <= 1  => impossible.

Data: results/signal_diagnostic/Qwen--Qwen3-8B/per_sample.jsonl (teacher-forced per-token `coll`, produced
by scripts/signal_diagnostic.py on stored vLLM p-less runs, i.e. on p-less-sampled histories).
Counting rule: clean traces use all tokens; loop classes use only tokens BEFORE loop onset (truncated
traces were clipped at onset+500 before scoring, so their tail is not comparable).

Usage:
    uv run python scripts/paper/make_llm4code_fig_peaked.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR_CODE = REPO_ROOT / "paper" / "llm4code" / "figures"
FIG_DIR_PAPER = REPO_ROOT.parent / "pless-for-coding-paper" / "figures"
DATA = REPO_ROOT / "results" / "signal_diagnostic" / "Qwen--Qwen3-8B" / "per_sample.jsonl"
OUT_NAME = "figA1_lp_peakedness_qwen3.png"
SAVE_DPI = 180  # same as the other paper figures

LOOP_CLASSES = ("looping_completed", "looping_truncated")
def class_stats(path: Path, bins: int = 40) -> dict[str, dict]:
    """Per trace class: histogram counts of L[P] over [0, 1], token count n, and n with L[P] > 1/2."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cls, coll, onset = r["cls"], r["coll"], r.get("onset_token")
            if cls in LOOP_CLASSES and onset is not None:
                coll = coll[:onset]
            a = np.asarray(coll, dtype=float)
            d = out.setdefault(cls, {"counts": np.zeros(bins, dtype=np.int64), "n": 0, "n_gt_half": 0})
            d["counts"] += np.histogram(a, bins=bins, range=(0.0, 1.0))[0]
            d["n"] += int(a.size)
            d["n_gt_half"] += int((a > 0.5).sum())
    return out


def pooled_stats(stats: dict[str, dict]) -> dict:
    """Sum the per-class histograms/counts into one pooled histogram."""
    return {
        "counts": sum(d["counts"] for d in stats.values()),
        "n": sum(d["n"] for d in stats.values()),
        "n_gt_half": sum(d["n_gt_half"] for d in stats.values()),
    }


def make_figure(pooled: dict, bins: int = 40) -> plt.Figure:
    """Plain histogram, one colour, no annotations; the L[P] > 1/2 share is stated in the paper text."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    pct = 100.0 * pooled["counts"] / pooled["n"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(edges[:-1], pct, width=1.0 / bins, align="edge", color="#2a78d6", edgecolor="#1a4f93", linewidth=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 50)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "\u00bd", "0.75", "1"])
    ax.set_xlabel("$p$-less Truncation Threshold, $L[P]$")
    ax.set_ylabel("Frequency of Token\nDistributions (%)")  # one line is taller than the 3.0 in axes and clips
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def main() -> None:
    bins = 40
    stats = class_stats(DATA, bins=bins)
    pooled = pooled_stats(stats)
    print(f"pooled tokens={pooled['n']}  L[P]>1/2: {100.0 * pooled['n_gt_half'] / pooled['n']:.2f}%", file=sys.stderr)
    fig = make_figure(pooled, bins=bins)
    for out in (FIG_DIR_CODE, FIG_DIR_PAPER):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / OUT_NAME, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"wrote {OUT_NAME}", file=sys.stderr)


if __name__ == "__main__":
    main()
