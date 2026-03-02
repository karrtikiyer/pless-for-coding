"""Parse pre-evaluated HumanEval detailed JSON files and compute metrics.

Skips code execution — the detailed JSON already contains per-sample pass/fail
booleans. Converts to the format expected by metrics.py and computes pass@k,
cover@t, and cover@t (distinct) via AST fingerprinting.
"""

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from bench.eval.metrics import (
    add_distinct_counts,
    add_structural_diversity,
    compute_cover_at_t,
    compute_pass_at_k,
    compute_structural_diversity,
)


def parse_detailed(data: dict, method: str) -> tuple[list[dict], list[dict]]:
    """Extract one method from detailed JSON into metrics.py's expected format.

    Returns:
        task_results: [{"task_id", "num_correct", "pass_results"}, ...]
        records: [{"task_id", "samples"}, ...]
    """
    task_results = []
    records = []

    for task in data[method]:
        task_id = task["task_id"]
        samples = task["samples"]

        pass_results = [s["passed"] for s in samples]
        num_correct = sum(pass_results)

        task_results.append({
            "task_id": task_id,
            "num_correct": num_correct,
            "pass_results": pass_results,
        })

        # Dedent code bodies so AST fingerprinting can parse them
        dedented_codes = [textwrap.dedent(s["code"]) for s in samples]
        records.append({
            "task_id": task_id,
            "samples": dedented_codes,
        })

    return task_results, records


def compute_metrics_for_method(
    task_results: list[dict],
    records: list[dict],
    model: str,
    method: str,
    k_values: list[int],
    t_values: list[float],
) -> dict:
    """Compute all metrics for a single method and return a metrics dict."""
    # Add distinct counts via AST fingerprinting
    add_distinct_counts(task_results, records)

    # Add structural diversity via pairwise AST edit distance
    add_structural_diversity(task_results, records)

    num_samples_per_task = len(task_results[0]["pass_results"]) if task_results else 0

    pass_at_k = compute_pass_at_k(task_results, k_values)
    cover_at_t, cover_at_t_distinct = compute_cover_at_t(
        task_results, t_values, num_samples_per_task
    )

    structural_diversity = compute_structural_diversity(task_results)

    return {
        "model": model,
        "method": method,
        "dataset": "humaneval",
        "num_tasks": len(task_results),
        "num_samples_per_task": num_samples_per_task,
        "pass_at_k": pass_at_k,
        "cover_at_t": cover_at_t,
        "cover_at_t_distinct": cover_at_t_distinct,
        "structural_diversity": structural_diversity,
        "per_task": task_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pass@k and cover@t from pre-evaluated HumanEval detailed JSON"
    )
    parser.add_argument(
        "--detailed", required=True, type=Path,
        help="Path to *_detailed.json file",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name (e.g. Qwen2.5-Coder-7B)",
    )
    parser.add_argument(
        "--k", default="1,3,5,10",
        help="Comma-separated k values for pass@k (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--t", default="0.1,0.3,0.5,0.7",
        help="Comma-separated fractional t values for cover@t (default: 0.1,0.3,0.5,0.7)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for metrics JSONs (default: <detailed_dir>/metrics/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    t_values = [float(x) for x in args.t.split(",")]

    print(f"Loading detailed results from {args.detailed}")
    with open(args.detailed) as f:
        data = json.load(f)

    methods = list(data.keys())
    print(f"Found {len(methods)} methods: {methods}")

    output_dir = args.output_dir or (args.detailed.parent / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        print(f"\nProcessing method: {method}")
        task_results, records = parse_detailed(data, method)

        metrics = compute_metrics_for_method(
            task_results, records, args.model, method, k_values, t_values
        )

        output_path = output_dir / f"{method}_metrics.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Tasks: {metrics['num_tasks']}")
        print(f"  Samples per task: {metrics['num_samples_per_task']}")
        print(f"  pass@k: {metrics['pass_at_k']}")
        print(f"  cover@t: {metrics['cover_at_t']}")
        print(f"  cover@t (distinct): {metrics['cover_at_t_distinct']}")
        print(f"  Written to {output_path}")


if __name__ == "__main__":
    main()
