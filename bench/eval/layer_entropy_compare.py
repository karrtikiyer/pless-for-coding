"""Overlay two ``layer_entropy_probe`` runs (e.g. instruct vs base).

Loads two ``layer_entropy_stats.json`` files and produces overlaid plots
plus a small comparison summary so the go/no-go for the penultimate-layer
diversity hypothesis can be read off in one place.

The headline number it reports is the **RLHF layer signature**:

    Δ_gap = (penult_minus_final_entropy_gap)_run_a − (...)_run_b

where ``run_a`` is typically the instruct model and ``run_b`` the base
model. A meaningfully positive Δ_gap means the instruct model's last layer
sharpens the distribution more than the base model's does — i.e., the
diversity collapse is concentrated in the final 1-2 layers, which is what
penultimate-layer sampling would target.

Usage::

    uv run python -m bench.eval.layer_entropy_compare \\
        --run-a results/layer_entropy_probe/Qwen2.5-Coder-7B-Instruct \\
        --run-b results/layer_entropy_probe/Qwen2.5-Coder-7B \\
        --label-a "Qwen2.5-Coder-7B-Instruct" \\
        --label-b "Qwen2.5-Coder-7B (base)" \\
        --output-dir results/layer_entropy_probe/compare_instruct_vs_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_RUN_COLORS = {"a": "#C62828", "b": "#1565C0"}  # red = run_a, blue = run_b


def _load(stats_path: Path) -> dict:
    return json.loads(stats_path.read_text())


def _entropy_curve(stats: dict) -> np.ndarray:
    n = stats["n_layers"]
    return np.array([
        stats["per_layer_all"][str(i)]["entropy"]["mean"]
        if str(i) in stats["per_layer_all"]
        else stats["per_layer_all"][i]["entropy"]["mean"]
        for i in range(n)
    ])


def _kl_curve(stats: dict) -> np.ndarray:
    n = stats["n_layers"]
    return np.array([
        stats["per_layer_all"][str(i)]["kl_to_final"]["mean"]
        if str(i) in stats["per_layer_all"]
        else stats["per_layer_all"][i]["kl_to_final"]["mean"]
        for i in range(n)
    ])


def _top1_curve(stats: dict) -> np.ndarray:
    n = stats["n_layers"]
    return np.array([
        stats["per_layer_all"][str(i)]["top1_agreement"]
        if str(i) in stats["per_layer_all"]
        else stats["per_layer_all"][i]["top1_agreement"]
        for i in range(n)
    ])


def _plot_overlay(curves: dict[str, tuple[np.ndarray, str]],
                  *, ylabel: str, title: str, output_path: Path,
                  ylim: tuple[float, float] | None = None,
                  symlog: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (curve, label) in curves.items():
        xs = np.arange(len(curve))
        ax.plot(xs, curve, "-o", markersize=4, linewidth=2,
                color=_RUN_COLORS[key], label=label)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if symlog:
        ax.set_yscale("symlog", linthresh=0.01)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


def compare(run_a_dir: Path, run_b_dir: Path,
            label_a: str, label_b: str, output_dir: Path) -> dict:
    stats_a = _load(run_a_dir / "layer_entropy_stats.json")
    stats_b = _load(run_b_dir / "layer_entropy_stats.json")
    if stats_a["n_layers"] != stats_b["n_layers"]:
        print(f"Warning: layer counts differ ({stats_a['n_layers']} vs "
              f"{stats_b['n_layers']}); plots will use each run's own x-axis")

    output_dir.mkdir(parents=True, exist_ok=True)

    ent_a, ent_b = _entropy_curve(stats_a), _entropy_curve(stats_b)
    kl_a, kl_b = _kl_curve(stats_a), _kl_curve(stats_b)
    t1_a, t1_b = _top1_curve(stats_a), _top1_curve(stats_b)

    _plot_overlay(
        {"a": (ent_a, label_a), "b": (ent_b, label_b)},
        ylabel="Mean entropy (nats)",
        title="Per-Layer Next-Token Entropy at Code Positions",
        output_path=output_dir / "compare_entropy.png",
    )
    _plot_overlay(
        {"a": (kl_a, label_a), "b": (kl_b, label_b)},
        ylabel="KL(layer || final), nats",
        title="Per-Layer KL to Final Layer",
        output_path=output_dir / "compare_kl.png",
        symlog=True,
    )
    _plot_overlay(
        {"a": (t1_a, label_a), "b": (t1_b, label_b)},
        ylabel="Top-1 agreement with final",
        title="Per-Layer Top-1 Agreement with Final Layer",
        output_path=output_dir / "compare_top1.png",
        ylim=(0, 1.02),
    )

    gap_a = stats_a["headline"]["penult_minus_final_entropy_gap"]
    gap_b = stats_b["headline"]["penult_minus_final_entropy_gap"]
    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "n_layers_a": stats_a["n_layers"],
        "n_layers_b": stats_b["n_layers"],
        "headline_a": stats_a["headline"],
        "headline_b": stats_b["headline"],
        "rlhf_layer_signature": {
            "delta_penult_final_gap": gap_a - gap_b,
            "interpretation": (
                "Δ > 0 means run_a's penultimate layer is markedly more diverse "
                "than its final layer relative to run_b — consistent with the "
                "hypothesis that RLHF sharpens the last 1-2 layers."
            ),
        },
    }
    summary_path = output_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 60)
    print(f"RLHF LAYER SIGNATURE: {label_a}  vs  {label_b}")
    print("=" * 60)
    print(f"  penult-final gap, {label_a:<35s}: {gap_a:+.3f} nats")
    print(f"  penult-final gap, {label_b:<35s}: {gap_b:+.3f} nats")
    print(f"  Δ (a − b):                                       {gap_a - gap_b:+.3f} nats")
    print(f"  top-1 agree at penult, {label_a:<30s}: {stats_a['headline']['penultimate_top1_agreement_with_final']:.3f}")
    print(f"  top-1 agree at penult, {label_b:<30s}: {stats_b['headline']['penultimate_top1_agreement_with_final']:.3f}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", required=True, type=Path,
                        help="Output dir of layer_entropy_probe run A (typically instruct)")
    parser.add_argument("--run-b", required=True, type=Path,
                        help="Output dir of run B (typically base)")
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    compare(args.run_a, args.run_b, args.label_a, args.label_b, args.output_dir)


if __name__ == "__main__":
    main()
