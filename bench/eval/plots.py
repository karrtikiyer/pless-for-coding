"""Generate visualizations from metrics JSON files."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def _label_for(m: dict) -> str:
    model = m["model"].split("/")[-1] if "/" in m["model"] else m["model"]
    return f"{model} ({m['method']})"


def _style_for(m: dict) -> dict:
    """Return color, linestyle, and marker for a config."""
    is_instruct = "instruct" in m["model"].lower() or "coder" in m["model"].lower()
    is_norm = m["method"] == "pless_norm"
    color = "#C05621" if is_instruct else "#2B6CB0"
    linestyle = "--" if is_norm else "-"
    marker = "s" if is_norm else "o"
    if is_instruct and is_norm:
        color = "#ED8936"
    elif not is_instruct and is_norm:
        color = "#63B3ED"
    return dict(color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6)


def plot_aggregate_lines(metrics_list: list[dict], output_path: Path) -> None:
    """Line plot with 3 subplots: pass@k, cover@t, cover@t (distinct).

    Each subplot has one line per model+method configuration (4 lines).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for m in metrics_list:
        label = _label_for(m)
        style = _style_for(m)

        # pass@k
        ks = sorted(m["pass_at_k"], key=lambda x: int(x))
        axes[0].plot(
            [int(k) for k in ks],
            [m["pass_at_k"][k] * 100 for k in ks],
            label=label, **style,
        )

        # cover@t
        ts = sorted(m["cover_at_t"], key=lambda x: float(x))
        axes[1].plot(
            [float(t) for t in ts],
            [m["cover_at_t"][t] for t in ts],
            label=label, **style,
        )

        # cover@t (distinct)
        ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
        axes[2].plot(
            [float(t) for t in ts_d],
            [m["cover_at_t_distinct"][t] for t in ts_d],
            label=label, **style,
        )

    axes[0].set_xlabel("k")
    axes[0].set_title("pass@k")
    axes[0].set_xticks([int(k) for k in ks])

    axes[1].set_xlabel("t")
    axes[1].set_title("cover@t")
    axes[1].set_xticks([float(t) for t in ts])

    axes[2].set_xlabel("t")
    axes[2].set_title("cover@t (distinct)")
    axes[2].set_xticks([float(t) for t in ts_d])

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.grid(alpha=0.3)

    axes[2].legend(fontsize=7, loc="upper right")
    fig.suptitle("MBPP: Metrics Overview", fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correctness_vs_diversity(metrics_list: list[dict], output_path: Path) -> None:
    """Bubble chart of num_correct vs num_distinct_correct per task.

    Uses only pless method configs (not pless_norm) to reduce clutter.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    pless_configs = [m for m in metrics_list if m["method"] == "pless"]
    if not pless_configs:
        pless_configs = metrics_list  # fallback

    for m in pless_configs:
        is_instruct = "instruct" in m["model"].lower() or "coder" in m["model"].lower()
        color = "#C05621" if is_instruct else "#2B6CB0"
        label = _label_for(m)

        counts: Counter[tuple[int, int]] = Counter()
        for task in m["per_task"]:
            nc = task["num_correct"]
            ndc = task["num_distinct_correct"]
            counts[(nc, ndc)] += 1

        xs, ys, sizes = [], [], []
        for (nc, ndc), count in counts.items():
            xs.append(nc)
            ys.append(ndc)
            sizes.append(count)

        # Scale bubble sizes for visibility
        sizes_arr = np.array(sizes, dtype=float)
        ax.scatter(
            xs, ys,
            s=sizes_arr * 15,
            alpha=0.55,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            label=label,
        )

    # y=x reference line
    ax.plot([0, 10], [0, 10], "k--", alpha=0.3, label="y = x (max diversity)")

    ax.set_xlabel("num_correct (out of 10)")
    ax.set_ylabel("num_distinct_correct (out of 10)")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("MBPP: Correctness vs Diversity per Task (pless)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate metric visualizations from JSON files"
    )
    parser.add_argument(
        "--metrics", nargs="+", type=Path, required=True,
        help="Paths to metrics JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures"),
        help="Output directory for figures (default: results/figures/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_list = load_metrics(args.metrics)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_aggregate_lines(metrics_list, out / "aggregate_lines.png")
    print(f"Saved {out / 'aggregate_lines.png'}")

    plot_correctness_vs_diversity(metrics_list, out / "correctness_vs_diversity.png")
    print(f"Saved {out / 'correctness_vs_diversity.png'}")


if __name__ == "__main__":
    main()
