"""Phase 2 (CPU) — render the Circular-Reasoning Figs 3b & 4 replicas from the
per-trace npz produced by scripts/loop_collapse_extract.py.

  Fig 3b  — dual-axis (twinx): per-token top-1 PROBABILITY (blue) and ENTROPY
            (red) around the loop onset, dashed onset line + shaded loop region.
  Fig 4   — two panels (Cosine Similarity, L2 Distance) vs layer id; "Repeat k" =
            anchor activation at cycle k vs k-1 (reds darken with depth); "Normal
            1/2" = a normal recurring token across its occurrences (blue/teal).

Per-model small multiples + an aggregate Fig-4 (mean across traces). Raw per-layer
cosine/L2 (no standardization) — faithful to the paper; cosine is scale-invariant.

Usage:
  uv run python scripts/loop_collapse_plot.py \
      --vec-dir results/loop_collapse_replication/Qwen--Qwen3-8B/vectors \
      --out-dir results/loop_collapse_replication/Qwen--Qwen3-8B/figures \
      --model-label Qwen3-8B [--onset-window 600] [--max-repeat 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150


def pair_metrics(acts: np.ndarray):
    """acts: (m, L+1, hidden). Returns cos[(m-1, L+1)], l2[(m-1, L+1)] for each
    consecutive cycle pair (k vs k-1), per layer."""
    m = acts.shape[0]
    if m < 2:
        return np.zeros((0, acts.shape[1])), np.zeros((0, acts.shape[1]))
    a, b = acts[1:], acts[:-1]                       # (m-1, L+1, hidden)
    dot = (a * b).sum(-1)
    cos = dot / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    l2 = np.linalg.norm(a - b, axis=-1)
    return cos, l2


def load_traces(vec_dir: Path):
    out = []
    for f in sorted(vec_dir.glob("*.npz")):
        z = np.load(f)
        out.append({"key": f.stem, **{k: z[k] for k in z.files}})
    return out


# ---------------------------------------------------------------------------
# Figure 3b — per-token probability + entropy around onset
# ---------------------------------------------------------------------------

def plot_fig3b(traces, out_path: Path, model_label: str, win: int):
    nt = len(traces)
    fig, axes = plt.subplots(nt, 1, figsize=(11, 2.6 * nt), squeeze=False)
    for ax, t in zip(axes[:, 0], traces):
        onset = int(t["onset_token"])
        n_think = int(t["n_think"])
        lo, hi = max(0, onset - win), min(n_think, onset + win)
        x = np.arange(lo, hi)
        prob = t["prob"][lo:hi]
        ent = t["entropy"][lo:hi]
        ax.plot(x, prob, color="#2B6CB0", lw=0.7, label="Probability")
        ax.set_ylabel("Probability", color="#2B6CB0")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="y", labelcolor="#2B6CB0")
        ax2 = ax.twinx()
        ax2.plot(x, ent, color="#C0392B", lw=0.7, label="Entropy")
        ax2.set_ylabel("Entropy (nats)", color="#C0392B")
        ax2.tick_params(axis="y", labelcolor="#C0392B")
        ax.axvline(onset, color="purple", ls="--", lw=1.2, label="Loop Onset")
        ax.axvspan(onset, hi, color="purple", alpha=0.08)
        ax.set_title(f"{t['key']}  (onset @ think-tok {onset})", fontsize=9)
        ax.set_xlabel("Think-token index")
    fig.suptitle(f"Fig 3b replica — {model_label}: per-token determinism at loop onset", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — layer-wise cosine / L2 across cycles
# ---------------------------------------------------------------------------

def _plot_fig4_axes(ax_cos, ax_l2, loop_acts, normal_acts, max_repeat, layers_x):
    cos, l2 = pair_metrics(loop_acts[:max_repeat])
    reds = plt.cm.Reds(np.linspace(0.45, 1.0, max(len(cos), 1)))
    for k in range(len(cos)):
        ax_cos.plot(layers_x, cos[k], color=reds[k], lw=1.2, marker=".", ms=3,
                    label=f"Repeat {k+1}")
        ax_l2.plot(layers_x, l2[k], color=reds[k], lw=1.2, marker=".", ms=3,
                   label=f"Repeat {k+1}")
    if normal_acts.shape[0] >= 2:
        ncos, nl2 = pair_metrics(normal_acts[:3])
        blues = ["#2B6CB0", "#2C7A7B"]
        for j in range(min(len(ncos), 2)):
            ax_cos.plot(layers_x, ncos[j], color=blues[j], lw=1.2, ls="--",
                        marker="^", ms=3, label=f"Normal {j+1}")
            ax_l2.plot(layers_x, nl2[j], color=blues[j], lw=1.2, ls="--",
                       marker="^", ms=3, label=f"Normal {j+1}")
    ax_cos.set_ylabel("Cosine Similarity"); ax_cos.set_xlabel("Layer ID")
    ax_l2.set_ylabel("L2 Distance"); ax_l2.set_xlabel("Layer ID")
    ax_cos.set_ylim(top=1.02)


def plot_fig4_per_trace(traces, out_path: Path, model_label: str, max_repeat: int):
    nt = len(traces)
    fig, axes = plt.subplots(nt, 2, figsize=(11, 3.0 * nt), squeeze=False)
    for r, t in enumerate(traces):
        layers_x = np.arange(t["loop_acts"].shape[1])
        _plot_fig4_axes(axes[r, 0], axes[r, 1], t["loop_acts"], t["normal_acts"],
                        max_repeat, layers_x)
        axes[r, 0].set_title(f"{t['key']}  (cosine)", fontsize=9)
        axes[r, 1].set_title(f"{t['key']}  (L2)", fontsize=9)
        if r == 0:
            axes[r, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Fig 4 replica — {model_label}: collapse of internal states across cycles", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_fig4_aggregate(traces, out_path: Path, model_label: str, max_repeat: int):
    """Mean cosine/L2 per Repeat-k across traces (traces must share layer count)."""
    L = traces[0]["loop_acts"].shape[1]
    layers_x = np.arange(L)
    fig, (ax_cos, ax_l2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # stack per-trace metrics, then average over traces for each Repeat-k
    cos_stack = {k: [] for k in range(max_repeat - 1)}
    l2_stack = {k: [] for k in range(max_repeat - 1)}
    for t in traces:
        if t["loop_acts"].shape[1] != L:
            continue
        cos, l2 = pair_metrics(t["loop_acts"][:max_repeat])
        for k in range(len(cos)):
            cos_stack[k].append(cos[k]); l2_stack[k].append(l2[k])
    reds = plt.cm.Reds(np.linspace(0.45, 1.0, max_repeat - 1))
    for k in range(max_repeat - 1):
        if not cos_stack[k]:
            continue
        cm = np.mean(cos_stack[k], axis=0); lm = np.mean(l2_stack[k], axis=0)
        ax_cos.plot(layers_x, cm, color=reds[k], lw=1.5, marker=".", label=f"Repeat {k+1}")
        ax_l2.plot(layers_x, lm, color=reds[k], lw=1.5, marker=".", label=f"Repeat {k+1}")
    ax_cos.set_ylabel("Cosine Similarity"); ax_cos.set_xlabel("Layer ID"); ax_cos.set_ylim(top=1.02)
    ax_l2.set_ylabel("L2 Distance"); ax_l2.set_xlabel("Layer ID")
    ax_cos.legend(fontsize=8); ax_cos.set_title("Cosine (mean over traces)")
    ax_l2.set_title("L2 (mean over traces)")
    fig.suptitle(f"Fig 4 replica (aggregate) — {model_label}  (n={len(traces)} traces)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--onset-window", type=int, default=600)
    ap.add_argument("--max-repeat", type=int, default=6, help="cycles → Repeat 1..(max-1)")
    args = ap.parse_args()

    traces = load_traces(args.vec_dir)
    if not traces:
        print(f"no npz in {args.vec_dir}"); return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] {len(traces)} traces from {args.vec_dir}")

    plot_fig3b(traces, args.out_dir / "fig3b_prob_entropy.png", args.model_label, args.onset_window)
    plot_fig4_per_trace(traces, args.out_dir / "fig4_per_trace.png", args.model_label, args.max_repeat)
    plot_fig4_aggregate(traces, args.out_dir / "fig4_aggregate.png", args.model_label, args.max_repeat)


if __name__ == "__main__":
    main()
