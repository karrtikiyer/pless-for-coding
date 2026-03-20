"""Generate comparison reports and charts for full MBPP results.

Usage:
    python -m bench.eval.visualize [--model-family llama|codellama|qwen|all]

Reads metrics JSONs produced by the eval pipeline and generates per-family:
  - comparison_report.md  (ranked pass@1 table + extended metrics)
  - {family}_full_mbpp.csv
  - figures/pass_at_1_comparison.png  (horizontal bar chart)
  - figures/metrics_overview.png      (faceted line plots)

Output goes under results/pless_full_mbpp_results/analysis/{family}/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.eval.compare_with_paper import (
    build_comparison_rows,
    format_comparison_table,
    format_extended_metrics_table,
    generate_analysis,
    generate_report,
    load_our_metrics,
    plot_comparison,
    plot_metrics_overview,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MBPP_FULL_METRICS_ROOT = Path("results/pless_full_mbpp_results")
OUTPUT_ROOT = MBPP_FULL_METRICS_ROOT / "analysis"

# ---------------------------------------------------------------------------
# Display names for our methods (shared across families)
# ---------------------------------------------------------------------------
OUR_METHOD_NAMES = {
    "pless": "P-Less",
    "pless_norm": "P-Less Norm",
    "temp": "Temperature",
    "top_p": "Top-p (ours)",
}

# ---------------------------------------------------------------------------
# Family configurations
# ---------------------------------------------------------------------------

FAMILIES: dict[str, dict] = {
    "llama": {
        "paper_results": {
            "meta-llama/Llama-2-7b-hf": {
                "Greedy": 17.80,
                "Beam Search": 19.40,
                "Diverse Beam Search": 18.40,
                "Contrastive Search": 17.40,
                "FSD": 19.20,
                "FSD-d": 21.20,
                "Contrastive Decoding": 18.20,
                "DoLa": 18.40,
                "Temperature": 17.20,
                "Top-p": 14.80,
                "Top-k": 10.20,
                "η-Sampling": 9.40,
                "Mirostat": 7.80,
                "Typical": 12.00,
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                "Greedy": 17.20,
                "Beam Search": 21.60,
                "Diverse Beam Search": 21.20,
                "Contrastive Search": 17.40,
                "FSD": 17.80,
                "FSD-d": 17.80,
                "Contrastive Decoding": 17.40,
                "DoLa": 18.00,
                "Temperature": 20.00,
                "Top-p": 17.60,
                "Top-k": 16.00,
                "η-Sampling": 17.00,
                "Mirostat": 16.00,
                "Typical": 18.00,
            },
        },
        "model_key_map": {
            "meta-llama--Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
            "meta-llama--Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        },
        "model_short": {
            "meta-llama/Llama-2-7b-hf": "Llama-2-7B (base)",
            "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7B-Chat",
        },
        "title": "Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Llama-2-7B)",
        "description": (
            "Comparison of p-less sampling against decoding methods from "
            '"A Thorough Examination of Decoding Methods in the Era of LLMs" '
            "(arXiv:2402.06925), Table 1."
        ),
        "suptitle_bar": "MBPP pass@1: P-Less vs Paper Decoding Methods (Llama-2-7B)",
        "suptitle_overview": "MBPP: Metrics Overview (Llama-2-7B) — Full 500 Problems",
        "csv_name": "llama_full_mbpp.csv",
        "num_paper_methods": 14,
    },
    "codellama": {
        "paper_results": {
            "codellama/CodeLlama-7b-hf": {
                "Greedy": 35.40,
                "Beam Search": 34.20,
                "Diverse Beam Search": 35.00,
                "Contrastive Search": 36.00,
                "FSD": 37.00,
                "FSD-d": 39.60,
                "Temperature": 35.00,
                "Top-p": 32.80,
                "Top-k": 25.40,
                "η-Sampling": 23.60,
                "Mirostat": 21.20,
                "Typical": 31.80,
            },
            "codellama/CodeLlama-7b-Instruct-hf": {
                "Greedy": 36.80,
                "Beam Search": 40.80,
                "Diverse Beam Search": 41.60,
                "Contrastive Search": 37.00,
                "FSD": 37.20,
                "FSD-d": 36.60,
                "Temperature": 39.00,
                "Top-p": 37.60,
                "Top-k": 35.60,
                "η-Sampling": 35.40,
                "Mirostat": 34.40,
                "Typical": 38.20,
            },
        },
        "model_key_map": {
            "codellama--CodeLlama-7b-hf": "codellama/CodeLlama-7b-hf",
            "codellama--CodeLlama-7b-Instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",
        },
        "model_short": {
            "codellama/CodeLlama-7b-hf": "CodeLlama-7B (base)",
            "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama-7B-Instruct",
        },
        "title": "Full MBPP (500 problems): P-Less vs Paper Decoding Methods (CodeLlama-7B)",
        "description": (
            "Comparison of p-less sampling against decoding methods from "
            '"A Thorough Examination of Decoding Methods in the Era of LLMs" '
            "(arXiv:2402.06925), Table 26."
        ),
        "suptitle_bar": "MBPP pass@1: P-Less vs Paper Decoding Methods (CodeLlama-7B)",
        "suptitle_overview": "MBPP: Metrics Overview (CodeLlama-7B) — Full 500 Problems",
        "csv_name": "codellama_full_mbpp.csv",
        "num_paper_methods": 12,
    },
    "qwen": {
        "paper_results": {
            "Qwen/Qwen-7B": {
                "Greedy": 33.00,
                "Beam Search": 34.40,
                "Diverse Beam Search": 33.20,
                "Contrastive Search": 28.40,
                "FSD": 33.00,
                "FSD-d": 33.60,
                "Temperature": 33.80,
                "Top-p": 27.40,
                "Top-k": 19.80,
                "η-Sampling": 25.80,
                "Mirostat": 18.40,
                "Typical": 27.00,
            },
            "Qwen/Qwen-7B-Chat": {
                "Greedy": 30.40,
                "Beam Search": 30.80,
                "Diverse Beam Search": 33.60,
                "Contrastive Search": 25.80,
                "FSD": 30.80,
                "FSD-d": 29.80,
                "Temperature": 30.00,
                "Top-p": 28.80,
                "Top-k": 26.80,
                "η-Sampling": 24.20,
                "Mirostat": 25.00,
                "Typical": 27.20,
            },
        },
        "model_key_map": {
            "Qwen--Qwen-7B": "Qwen/Qwen-7B",
            "Qwen--Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
        },
        "model_short": {
            "Qwen/Qwen-7B": "Qwen-7B (base)",
            "Qwen/Qwen-7B-Chat": "Qwen-7B-Chat",
        },
        "title": "Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Qwen-7B)",
        "description": (
            "Comparison of p-less sampling against decoding methods from "
            '"A Thorough Examination of Decoding Methods in the Era of LLMs" '
            "(arXiv:2402.06925), Table 26."
        ),
        "suptitle_bar": "MBPP pass@1: P-Less vs Paper Decoding Methods (Qwen-7B)",
        "suptitle_overview": "MBPP: Metrics Overview (Qwen-7B) — Full 500 Problems",
        "csv_name": "qwen_full_mbpp.csv",
        "num_paper_methods": 12,
    },
}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(
    our_metrics_by_model: dict[str, list[dict]],
    paper_results: dict[str, dict[str, float]],
    model_short: dict[str, str],
    output_path: Path,
) -> None:
    """Write a CSV with ranked pass@1 for all models in the family."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_key in paper_results:
        our_metrics = our_metrics_by_model.get(model_key, [])
        comparison_rows = build_comparison_rows(
            paper_results[model_key], our_metrics, OUR_METHOD_NAMES,
        )
        display = model_short.get(model_key, model_key)
        for rank, row in enumerate(comparison_rows, 1):
            rows.append({
                "Model": display,
                "Rank": rank,
                "Method": row["method"],
                "Source": row["source"],
                "pass@1 (%)": round(row["pass_at_1"], 1),
            })

    fieldnames = ["Model", "Rank", "Method", "Source", "pass@1 (%)"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV written to {output_path}")


# ---------------------------------------------------------------------------
# Per-family pipeline
# ---------------------------------------------------------------------------

def run_family(family_name: str, metrics_root: Path, output_root: Path) -> None:
    """Generate report, CSV, and figures for one model family."""
    cfg = FAMILIES[family_name]
    out_dir = output_root / family_name
    fig_dir = out_dir / "figures"

    print(f"\n{'='*60}")
    print(f"  {family_name.upper()} family")
    print(f"{'='*60}")

    # Load metrics
    our_metrics = load_our_metrics(metrics_root, model_key_map=cfg["model_key_map"])
    if not our_metrics:
        print(f"  WARNING: No metrics found for {family_name}. Skipping.")
        return

    for model_key, mlist in our_metrics.items():
        methods = [f"{m['method']}(t={m.get('temperature', '?')})" for m in mlist]
        print(f"  {model_key}: {methods}")

    # Report
    report = generate_report(
        our_metrics,
        paper_results=cfg["paper_results"],
        title=cfg["title"],
        description=cfg["description"],
        model_short=cfg["model_short"],
        our_method_names=OUR_METHOD_NAMES,
    )
    report_path = out_dir / "comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"  Report: {report_path}")

    # CSV
    write_csv(our_metrics, cfg["paper_results"], cfg["model_short"],
              out_dir / cfg["csv_name"])

    # Figures
    bar_path = fig_dir / "pass_at_1_comparison.png"
    plot_comparison(
        our_metrics, bar_path,
        paper_results=cfg["paper_results"],
        model_short=cfg["model_short"],
        suptitle=cfg["suptitle_bar"],
    )
    print(f"  Bar chart: {bar_path}")

    overview_path = fig_dir / "metrics_overview.png"
    plot_metrics_overview(
        our_metrics, overview_path,
        model_short=cfg["model_short"],
        suptitle=cfg["suptitle_overview"],
    )
    print(f"  Overview: {overview_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate full MBPP comparison reports and charts",
    )
    parser.add_argument(
        "--model-family",
        choices=list(FAMILIES.keys()) + ["all"],
        default="all",
        help="Which model family to generate reports for (default: all)",
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=MBPP_FULL_METRICS_ROOT,
        help="Root directory containing model result subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT,
        help="Root output directory (family subdirs created automatically)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    families = list(FAMILIES.keys()) if args.model_family == "all" else [args.model_family]

    for family in families:
        run_family(family, args.metrics_dir, args.output_dir)

    print(f"\nDone. Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
