"""Merge algosim NAUADC/EA/DA@K with our split-decoding summary and report.

Reads:
  - algosim_data/algosim_metrics.json
        (produced on the GPU box by algosim/compute_metrics.py;
         metrics["clustering"]["ATCODER"] holds DA@1..25, EA, NAUADC)
  - results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis/
        split_decoding_report_summary.json
        (our existing per-config pass_at_k / struct_div / codebleu_div)

But algosim's metrics.json buckets results by problem_id prefix (a single
"ATCODER" bucket pooled across all configs, because we encoded the config in
the suffix). So we re-read the per-config response parquets in
algosim_data/responses/<config>.parquet to recover per-config metrics.

Writes:
  - <analysis>/algosim_report.md
  - <analysis>/algosim_struct_vs_nauadc.png
  - <analysis>/algosim_pass_vs_nauadc.png

Usage:
    uv run python -m bench.eval.algosim_report
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy

ANALYSIS_DIR = Path("results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis")
SUMMARY_PATH = ANALYSIS_DIR / "split_decoding_report_summary.json"
RESPONSES_DIR = Path("algosim_data/responses")
ALGOSIM_METRICS_PATH = Path("algosim_data/algosim_metrics.json")

K_VALUES = list(range(1, 26))


def _comb(n: int, k: int) -> float:
    if n < k:
        return 0.0
    return float(np.prod(np.arange(n, n - k, -1) / np.arange(k, 0, -1)))


def _da_at_k(group_sizes: np.ndarray, k: int) -> float:
    n, m = int(np.sum(group_sizes)), len(group_sizes)
    if k > n:
        return float(m)
    return m - float(np.sum([_comb(n - s, k) / _comb(n, k) for s in group_sizes]))


def _per_config_metrics(parquet_path: Path) -> dict:
    """Recompute NAUADC / EA / DA@K for a single config's response parquet.

    Mirrors algosim/compute_metrics.py:compute_clustering_metrics, but scoped
    to one parquet file (= one config) instead of pooling all ATCODER rows.
    """
    df = pd.read_parquet(parquet_path)
    df["group_index"] = df["group_index"].apply(
        lambda x: [] if x is None or (isinstance(x, float) and np.isnan(x)) else list(x)
    )
    df["solution_group"] = df["group_index"].apply(
        lambda x: [count for _, count in Counter(x).most_common()]
    )
    mask = df["solution_group"].map(sum) >= 1
    df = df[mask]
    if len(df) == 0:
        return {"DA": [0.0] * len(K_VALUES), "NAUADC": 0.0, "EA": 0.0, "n_problems": 0}

    da_curve = []
    for k in K_VALUES:
        da_curve.append(df["solution_group"].apply(lambda g, k=k: _da_at_k(np.array(g), k)).mean())

    nauadc = float(np.trapezoid(da_curve, K_VALUES)) / (K_VALUES[-1] - K_VALUES[0])

    def _ea(counts: list[int]) -> float:
        prob = np.array(counts) / np.sum(counts)
        return float(np.exp(-np.sum(xlogy(prob, prob))))

    ea = df["solution_group"].apply(_ea).mean()

    return {
        "DA": [float(x) for x in da_curve],
        "NAUADC": float(nauadc),
        "EA": float(ea),
        "n_problems": int(len(df)),
    }


def _gather_algosim_metrics(responses_dir: Path) -> dict[str, dict]:
    """Walk responses_dir, one parquet per config, return {config_key: metrics}."""
    out: dict[str, dict] = {}
    for p in sorted(responses_dir.glob("*.parquet")):
        config_key = p.stem
        out[config_key] = _per_config_metrics(p)
    return out


def _load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def _load_metrics_from_per_config_files(metrics_dir: Path) -> dict[str, dict]:
    """Build a {config_key: summary-entry} dict from individual
    ``metrics/<config>_metrics.json`` files in the model's results
    directory. Used when the legacy ``split_decoding_report_summary.json``
    doesn't exist (i.e. for arbitrary models / α-sweep results).

    Maps the field names ``structural_diversity``/``codebleu_diversity``
    (full metric output) → ``struct_div``/``codebleu_div`` (legacy
    summary schema), and constructs a human label from method +
    temperature when available.
    """
    out: dict[str, dict] = {}
    for p in sorted(metrics_dir.glob("*_metrics.json")):
        cfg = p.stem.removesuffix("_metrics")
        m = json.loads(p.read_text())
        label = cfg
        if "method" in m and "temperature" in m:
            label = f"{m['method']} @ T={m['temperature']}"
        out[cfg] = {
            "label": label,
            "pass_at_k": {str(k): float(v) for k, v in m.get("pass_at_k", {}).items()},
            "struct_div": float(m.get("structural_diversity", 0.0)),
            "codebleu_div": float(m.get("codebleu_diversity", 0.0)),
        }
    return out


def _write_markdown(joined: list[dict], out_path: Path,
                    pending: list[str] | None = None,
                    title: str = "Qwen3-8B Split Decoding",
                    judge: str = "Llama-3.1-8B-Instruct") -> None:
    joined_sorted = sorted(joined, key=lambda r: -r["NAUADC"])
    lines = [
        f"# algosim Diversity Report — {title}",
        "",
        f"AlgoSim NAUADC / EA / DA@10 ({judge} judge, correct samples only) "
        "joined with our existing structural / CodeBLEU / pass@k metrics. "
        "Sorted by NAUADC descending.",
        "",
    ]
    if pending:
        lines += [
            f"> **Pending:** {len(pending)} config(s) listed in `algosim_data/manifest.json` "
            f"do not yet have a response parquet — algosim clustering has not been run "
            f"on them: **{', '.join(pending)}**. Until they land, the table below shows "
            f"{len(joined)} configs only.",
            "",
        ]
    lines += [
        "| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in joined_sorted:
        lines.append(
            f"| **{r['config']}** | {r['label']} | "
            f"{r['pass@1']:.3f} | {r['pass@10']:.3f} | "
            f"{r['struct_div']:.3f} | {r['codebleu_div']:.3f} | "
            f"**{r['NAUADC']:.3f}** | {r['EA']:.3f} | {r['DA@10']:.3f} | {r['n_problems']} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _scatter(joined: list[dict], x_key: str, y_key: str, xlabel: str, ylabel: str,
             out_path: Path, draw_yx: bool = False,
             title_suffix: str = "Qwen3-8B Split Decoding") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [r[x_key] for r in joined]
    ys = [r[y_key] for r in joined]
    ax.scatter(xs, ys, s=60, alpha=0.85)
    for r in joined:
        ax.annotate(r["config"], (r[x_key], r[y_key]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel} — {title_suffix}")
    ax.grid(True, alpha=0.3)
    if draw_yx:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, label="y = x")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, default=RESPONSES_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix appended to output filenames "
             "(e.g. '_claude' -> algosim_report_claude.md). "
             "Use when running a second judge alongside the default Llama outputs.",
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=None,
        help="Directory of per-config <config>_metrics.json files for the "
             "pass@k / struct_div / codebleu_div join. Defaults to a sibling "
             "'metrics/' under --analysis-dir's parent (i.e. results/X/metrics/ "
             "when --analysis-dir is results/X/analysis/). Falls back to the "
             "legacy split_decoding_report_summary.json path if neither exists.",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Title prefix used in the report markdown and scatter plots. "
             "Default: derived from --analysis-dir's parent directory name.",
    )
    parser.add_argument(
        "--judge", type=str, default="Llama-3.1-8B-Instruct",
        help="Judge model name used in the narrative line of the markdown.",
    )
    args = parser.parse_args()
    suf = args.output_suffix

    # Title: derive from analysis-dir if not supplied
    title = args.title
    if title is None:
        title = args.analysis_dir.parent.name.replace("--", "/") or "α-sweep"

    # Try the new path (per-config metrics files) first, fall back to legacy.
    metrics_dir = args.metrics_dir
    if metrics_dir is None:
        metrics_dir = args.analysis_dir.parent / "metrics"
    if metrics_dir.is_dir():
        summary = _load_metrics_from_per_config_files(metrics_dir)
        print(f"[algosim_report] loaded {len(summary)} configs from {metrics_dir}")
    else:
        print(f"[algosim_report] no metrics dir at {metrics_dir}; using legacy "
              f"{SUMMARY_PATH}")
        summary = _load_summary()
    algo_metrics = _gather_algosim_metrics(args.responses_dir)
    if not algo_metrics:
        raise SystemExit(f"No parquet files found in {args.responses_dir}")

    # Detect which configs from the manifest haven't been clustered yet.
    pending: list[str] = []
    manifest_path = args.responses_dir.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = [c["config"] for c in manifest.get("configs", [])]
        pending = [c for c in expected if c not in algo_metrics]
        if pending:
            print(f"[algosim_report] pending (no response parquet): {', '.join(pending)}")

    joined = []
    for cfg, metrics in algo_metrics.items():
        if cfg not in summary:
            print(f"[warn] config {cfg} not in summary.json; skipping")
            continue
        s = summary[cfg]
        joined.append({
            "config": cfg,
            "label": s["label"],
            "pass@1": s["pass_at_k"]["1"],
            "pass@10": s["pass_at_k"]["10"],
            "struct_div": s["struct_div"],
            "codebleu_div": s["codebleu_div"],
            "NAUADC": metrics["NAUADC"],
            "EA": metrics["EA"],
            "DA@10": metrics["DA"][K_VALUES.index(10)],
            "n_problems": metrics["n_problems"],
        })

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.analysis_dir / f"algosim_report{suf}.md"
    _write_markdown(joined, md_path, pending=pending, title=title, judge=args.judge)
    print(f"[algosim_report] wrote {md_path}")

    raw_path = args.analysis_dir / f"algosim_per_config{suf}.json"
    raw_path.write_text(json.dumps({"k_values": K_VALUES, "configs": joined,
                                    "raw": algo_metrics}, indent=2))
    print(f"[algosim_report] wrote {raw_path}")

    _scatter(joined, "struct_div", "NAUADC",
             "struct_div (ours)", "NAUADC (algosim)",
             args.analysis_dir / f"algosim_struct_vs_nauadc{suf}.png",
             title_suffix=title)
    print(f"[algosim_report] wrote {args.analysis_dir / f'algosim_struct_vs_nauadc{suf}.png'}")

    _scatter(joined, "pass@10", "NAUADC",
             "pass@10", "NAUADC (algosim)",
             args.analysis_dir / f"algosim_pass_vs_nauadc{suf}.png",
             title_suffix=title)
    print(f"[algosim_report] wrote {args.analysis_dir / f'algosim_pass_vs_nauadc{suf}.png'}")


if __name__ == "__main__":
    main()
