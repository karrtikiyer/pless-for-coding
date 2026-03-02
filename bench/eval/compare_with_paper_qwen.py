"""Compare p-less sampling results against decoding methods from the paper — Qwen-7B.

Reference paper: "A Thorough Examination of Decoding Methods in the Era of LLMs"
(https://arxiv.org/abs/2402.06925)

Produces a markdown report and bar chart comparing our MBPP results (pless,
pless_norm, temp_0.7) with the paper's 12 decoding methods on Qwen-7B
(Table 26).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.eval.compare_with_paper import (
    generate_report,
    load_our_metrics,
    plot_comparison,
    plot_metrics_overview,
)

# ---------------------------------------------------------------------------
# Paper results — MBPP pass@1 (%) from Table 26 of arXiv:2402.06925v3
# ---------------------------------------------------------------------------

PAPER_RESULTS: dict[str, dict[str, float]] = {
    "Qwen/Qwen-7B": {
        "Greedy": 33.00,
        "Beam Search": 34.40,
        "Diverse Beam Search": 33.20,
        "Contrastive Search": 28.40,
        "FSD": 33.00,
        "FSD-d": 33.80,
        "Temperature": 33.00,
        "Top-p": 27.40,
        "Top-k": 19.80,
        "η-Sampling": 25.80,
        "Mirostat": 18.40,
        "Typical": 27.00,
    },
}

# Model directory name → paper model key
_MODEL_KEY_MAP = {
    "Qwen--Qwen-7B": "Qwen/Qwen-7B",
}

# Short display names for models
_MODEL_SHORT = {
    "Qwen/Qwen-7B": "Qwen-7B",
}

# Display names for our methods
_OUR_METHOD_NAMES = {
    "pless": "P-Less (t=1.0)",
    "pless_norm": "P-Less Norm (t=1.0)",
    "temp": "Temperature (t=0.7)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare p-less MBPP results against decoding methods from the paper (Qwen-7B)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Parent directory containing model result subdirectories",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/qwen_paper_comparison/comparison_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--figures-dir", type=Path,
        default=Path("results/qwen_paper_comparison/figures"),
        help="Directory for generated plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    our_metrics = load_our_metrics(args.results_dir, model_key_map=_MODEL_KEY_MAP)
    if not our_metrics:
        print("ERROR: No metrics files found. Run `python -m bench.eval` first.")
        raise SystemExit(1)

    print(f"Loaded metrics for {len(our_metrics)} models:")
    for model_key, mlist in our_metrics.items():
        methods = [m["method"] for m in mlist]
        print(f"  {model_key}: {methods}")

    # Generate report
    report = generate_report(
        our_metrics,
        paper_results=PAPER_RESULTS,
        title="MBPP: P-Less vs Paper Decoding Methods (Qwen-7B)",
        description=(
            "Comparison of p-less sampling against 12 decoding methods from "
            '"A Thorough Examination of Decoding Methods in the Era of LLMs" '
            "(arXiv:2402.06925), Table 26."
        ),
        model_short=_MODEL_SHORT,
        our_method_names=_OUR_METHOD_NAMES,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {args.output}")

    # Generate figures
    fig_path = args.figures_dir / "pass_at_1_comparison.png"
    plot_comparison(
        our_metrics, fig_path,
        paper_results=PAPER_RESULTS,
        model_short=_MODEL_SHORT,
        suptitle="MBPP pass@1: P-Less vs Paper Decoding Methods (Qwen-7B)",
    )
    print(f"Figure saved to {fig_path}")

    overview_path = args.figures_dir / "metrics_overview.png"
    plot_metrics_overview(
        our_metrics, overview_path,
        model_short=_MODEL_SHORT,
        suptitle="MBPP: Metrics Overview (Qwen-7B)",
    )
    print(f"Figure saved to {overview_path}")


if __name__ == "__main__":
    main()
