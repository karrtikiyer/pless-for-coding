"""Analyze the bimodal-entropy experiment sidecar JSONLs.

Reads <results-dir>/<model>/temp_t1.0.jsonl.entropy.jsonl (one row per
position with sigma_p2, sigma_p3, sigma_p5, max_p, top-32, token_str)
and produces:
  - per-model H_α histograms (α=2, 3, 5)
  - Hartigan dip test for bimodality (per α, per model)
  - per-token-class boxplot of H_2
  - summary.md with the verdict

Usage:
    uv run python -m bench.eval.analyze_entropy \\
        --entropy-dir results/pless_alpha_entropy \\
        --analysis-dir results/pless_alpha_entropy/analysis
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diptest import diptest


# Token-class heuristics — coarse, regex-based.
KEYWORDS = {
    "def", "class", "if", "elif", "else", "for", "while", "return",
    "import", "from", "as", "try", "except", "finally", "raise",
    "with", "lambda", "yield", "global", "nonlocal", "pass", "break",
    "continue", "in", "is", "not", "and", "or", "True", "False", "None",
}
OPERATORS = set("+-*/%=<>!&|^~")
PUNCT_TOKENS = set("()[]{},.:;")


def classify_token(tok: str) -> str:
    """Coarse classification of a decoded token string."""
    if not tok:
        return "empty"
    if tok.isspace():
        return "whitespace"
    if tok.strip() == "":
        return "whitespace"
    stripped = tok.strip()
    if stripped in KEYWORDS:
        return "keyword"
    if all(c in OPERATORS for c in stripped):
        return "operator"
    if all(c in PUNCT_TOKENS for c in stripped):
        return "punctuation"
    if stripped.isdigit() or (stripped.startswith(("0x", "-0x", "0X")) and len(stripped) > 2):
        return "numeric"
    if re.match(r'^[a-zA-Z_][a-zA-Z_0-9]*$', stripped):
        return "identifier"
    if stripped.startswith(('"', "'", '"""', "'''")):
        return "string"
    return "other"


def load_sidecar(path: Path) -> list[dict]:
    """Stream-load a JSONL sidecar."""
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def compute_entropies(records: list[dict]) -> dict[str, np.ndarray]:
    """Derive H_α from sigma_p^α columns."""
    p2 = np.array([r["sigma_p2"] for r in records])
    p3 = np.array([r["sigma_p3"] for r in records])
    p5 = np.array([r["sigma_p5"] for r in records])
    maxp = np.array([r["max_p"] for r in records])
    eps = 1e-30
    return {
        "H2": -np.log(np.clip(p2, eps, 1.0)),
        "H3": -0.5 * np.log(np.clip(p3, eps, 1.0)),  # H_α = log(Σpᵢ^α) / (1-α)
        "H5": -0.25 * np.log(np.clip(p5, eps, 1.0)),
        "Hinf": -np.log(np.clip(maxp, eps, 1.0)),
    }


def plot_hist(values: np.ndarray, model_label: str, metric_label: str,
              out_path: Path, dip_stat: float | None = None, dip_p: float | None = None,
              passing_only: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=80, density=True, alpha=0.7, color="steelblue")
    title = f"{metric_label} histogram — {model_label}"
    if passing_only:
        title += " (passing samples only)"
    if dip_stat is not None and dip_p is not None:
        title += f"\nHartigan's dip = {dip_stat:.4f}, p = {dip_p:.4g}"
    ax.set_title(title)
    ax.set_xlabel(f"{metric_label} (nats)")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    median = float(np.median(values))
    p05 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    ax.axvline(median, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label=f"median = {median:.2f}")
    ax.axvline(p05, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label=f"5th pct = {p05:.2f}")
    ax.axvline(p95, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label=f"95th pct = {p95:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_class_box(records: list[dict], H2: np.ndarray,
                       model_label: str, out_path: Path) -> dict[str, dict]:
    """Boxplot of H_2 grouped by token class."""
    classes = [classify_token(r["token_str"]) for r in records]
    grouped = {}
    for cl, h in zip(classes, H2):
        grouped.setdefault(cl, []).append(h)
    # Order by median ascending
    order = sorted(grouped.keys(), key=lambda k: np.median(grouped[k]))
    data = [grouped[k] for k in order]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, labels=[f"{k}\n(n={len(grouped[k])})" for k in order],
               showfliers=False)
    ax.set_ylabel("H₂ (nats)")
    ax.set_title(f"H₂ by token class — {model_label}\n(boxes sorted by median entropy)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {k: {"n": len(grouped[k]),
                "median": float(np.median(grouped[k])),
                "mean": float(np.mean(grouped[k])),
                "p25": float(np.percentile(grouped[k], 25)),
                "p75": float(np.percentile(grouped[k], 75))} for k in order}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy-dir", type=Path,
                        default=Path("results/pless_alpha_entropy"))
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("results/pless_alpha_entropy/analysis"))
    parser.add_argument("--max-records", type=int, default=None,
                        help="Subsample to this many records per model (for fast iteration)")
    args = parser.parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    overall_summary = {}
    for model_dir in sorted(args.entropy_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "analysis":
            continue
        slug = model_dir.name
        sidecars = list(model_dir.glob("*.entropy.jsonl"))
        if not sidecars:
            print(f"[skip] {slug}: no sidecar found")
            continue
        sidecar = sidecars[0]
        print(f"[{slug}] loading {sidecar}")
        records = load_sidecar(sidecar)
        if args.max_records is not None and len(records) > args.max_records:
            print(f"  subsampling: {len(records)} -> {args.max_records}")
            rng = np.random.default_rng(42)
            idx = rng.choice(len(records), args.max_records, replace=False)
            records = [records[i] for i in idx]
        print(f"  {len(records)} positions")
        H = compute_entropies(records)

        # Hartigan's dip test for each H_α
        dip_results = {}
        for key, values in H.items():
            dip_stat, dip_p = diptest(values)
            dip_results[key] = {"stat": float(dip_stat), "p": float(dip_p)}
            print(f"  {key} dip={dip_stat:.4f} p={dip_p:.4g}")

        # Histograms (one per α)
        plot_hist(H["H2"], slug, "H₂ (Rényi entropy of order 2)",
                  args.analysis_dir / f"hist_H2_{slug}.png",
                  dip_stat=dip_results["H2"]["stat"], dip_p=dip_results["H2"]["p"])
        plot_hist(H["H3"], slug, "H₃ (Rényi)",
                  args.analysis_dir / f"hist_H3_{slug}.png",
                  dip_stat=dip_results["H3"]["stat"], dip_p=dip_results["H3"]["p"])
        plot_hist(H["H5"], slug, "H₅ (Rényi)",
                  args.analysis_dir / f"hist_H5_{slug}.png",
                  dip_stat=dip_results["H5"]["stat"], dip_p=dip_results["H5"]["p"])
        plot_hist(H["Hinf"], slug, "H_∞ (−log max(p), min-entropy)",
                  args.analysis_dir / f"hist_Hinf_{slug}.png",
                  dip_stat=dip_results["Hinf"]["stat"], dip_p=dip_results["Hinf"]["p"])

        # Per-token-class boxplot
        per_class = plot_per_class_box(records, H["H2"], slug,
                                       args.analysis_dir / f"boxplot_per_class_{slug}.png")

        overall_summary[slug] = {
            "n_positions": len(records),
            "dip_test": dip_results,
            "per_token_class": per_class,
            "H2_stats": {
                "median": float(np.median(H["H2"])),
                "p05": float(np.percentile(H["H2"], 5)),
                "p25": float(np.percentile(H["H2"], 25)),
                "p75": float(np.percentile(H["H2"], 75)),
                "p95": float(np.percentile(H["H2"], 95)),
            },
        }

    summary_path = args.analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2))
    print(f"\n[done] wrote {summary_path}")
    print(f"[done] figures in {args.analysis_dir}/")


if __name__ == "__main__":
    main()
