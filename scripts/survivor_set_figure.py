"""Real-data replacement for the (removed) synthetic mechanism figure.

Reads the `--log-entropy` sidecar(s) produced by `bench.apps.runner` on the
reasoning models (see scripts/run_gpu_baselines_paperB.sh, step C) and measures
the p-less **survivor-set size per decode step** = number of tokens whose prob
meets the collision threshold, i.e. `count(top32_probs >= sigma_p2)`.

Mechanism claim being evidenced: on peaked steps only the single most-probable
token survives (survivor size == 1 -> p-less is forced to emit the argmax), and
this happens far more on looping traces than on healthy ones.

Join: each sidecar row carries (task_id, sample_id); a sample is "looped" if its
`samples_with_thinking[sample_id]` in the sibling generation JSONL never closes
`</think>` (non-terminating). If the generation JSONL is absent, only the pooled
distribution is reported (no looped/healthy split).

Output: `paper/paperB/figures/fig_survivor.png` + printed stats. Every number is
computed here from the sidecar; nothing hard-coded.

Run (after the GPU probe run):
  uv run python scripts/survivor_set_figure.py --root results/_paperB_baselines
"""
import argparse
import glob
import gzip
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THINK_END = "</think>"


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _find_pairs(root):
    """Yield (label, entropy_sidecar_path, generation_jsonl_or_None)."""
    sidecars = glob.glob(os.path.join(root, "**", "*.entropy.jsonl"), recursive=True)
    sidecars += glob.glob(os.path.join(root, "**", "*.entropy.jsonl.gz"), recursive=True)
    for sc in sorted(set(sidecars)):
        gen = sc.replace(".entropy.jsonl.gz", "").replace(".entropy.jsonl", "")
        gen = gen if os.path.exists(gen) else (gen + ".gz" if os.path.exists(gen + ".gz") else None)
        # label = model dir name (…/<model>/<SOURCE_DIFFICULTY>/<file>)
        label = os.path.basename(os.path.dirname(os.path.dirname(sc))) or os.path.basename(sc)
        yield label, sc, gen


def _looped_mask(gen_path):
    """Map (task_id, sample_id) -> looped(bool) from a generation JSONL."""
    if gen_path is None:
        return None
    looped = {}
    with _open(gen_path) as f:
        for line in f:
            r = json.loads(line)
            tid = r.get("task_id")
            sw = r.get("samples_with_thinking") or r.get("samples") or []
            for i, s in enumerate(sw):
                looped[(tid, i)] = (THINK_END not in s)
    return looped


def _survivor_counts(sidecar_path, looped):
    """Return dict: 'looped'/'healthy'/'all' -> list of survivor-set sizes."""
    out = {"looped": [], "healthy": [], "all": []}
    with _open(sidecar_path) as f:
        for line in f:
            r = json.loads(line)
            thr = r["sigma_p2"]
            probs = r["top32_probs"]
            n = int(sum(1 for p in probs if p >= thr))   # >=32 => censored (flat step)
            out["all"].append(n)
            if looped is not None:
                key = (r.get("task_id"), r.get("sample_id"))
                if key in looped:
                    out["looped" if looped[key] else "healthy"].append(n)
    return out


def _frac_single(xs):
    return 100.0 * sum(1 for x in xs if x == 1) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/_paperB_baselines")
    ap.add_argument("--out", default="paper/paperB/figures/fig_survivor.png")
    args = ap.parse_args()

    panels = []
    for label, sc, gen in _find_pairs(args.root):
        looped = _looped_mask(gen)
        counts = _survivor_counts(sc, looped)
        if not counts["all"]:
            print(f"[skip] {label}: empty sidecar {sc}")
            continue
        panels.append((label, counts))
        print(f"\n=== {label}  ({os.path.basename(sc)}) ===")
        print(f"  logged think-steps: {len(counts['all']):,}")
        print(f"  single-token (survivor==1) overall: {_frac_single(counts['all']):.1f}%")
        if counts["looped"] or counts["healthy"]:
            print(f"    looped traces:  {_frac_single(counts['looped']):.1f}%  "
                  f"(n={len(counts['looped']):,})")
            print(f"    healthy traces: {_frac_single(counts['healthy']):.1f}%  "
                  f"(n={len(counts['healthy']):,})")

    if not panels:
        print(f"\nNo sidecars found under {args.root}. Run scripts/run_gpu_baselines_paperB.sh (step C) first.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False)
    for ax, (label, counts) in zip(axes[0], panels):
        bins = np.arange(1, 12) - 0.5    # survivor size 1..10 (>=10 lumped)
        def clip(xs): return [min(x, 10) for x in xs]
        # only plot non-empty classes (avoids empty-hist divide-by-zero)
        series = [(counts["looped"], "looped", "#d62728"),
                  (counts["healthy"], "healthy", "#1f77b4")]
        series = [(xs, name, c) for xs, name, c in series if xs]
        if not series:                                   # no looped/healthy split available
            series = [(counts["all"], "all", "#d62728")]
        ax.hist([clip(xs) for xs, _, _ in series], bins=bins, density=True,
                color=[c for _, _, c in series],
                label=[f"{name} ({_frac_single(xs):.0f}% =1)" for xs, name, _ in series])
        ax.legend(fontsize=7)
        ax.set_xlabel("p-less survivor-set size per step (10 = $\\geq$10)")
        ax.set_ylabel("fraction of steps")
        ax.set_title(label, fontsize=9)
    fig.suptitle("Survivor-set collapses to a single token far more on looping traces", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
