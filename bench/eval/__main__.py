import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bench.eval.executor import evaluate_all
from bench.eval.loader import load_results
from bench.eval.metrics import add_distinct_counts, compute_cover_at_t, compute_pass_at_k


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate code samples and compute pass@k / cover@t metrics"
    )
    parser.add_argument(
        "--results-file", required=True, type=Path,
        help="Path to JSONL results file",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["mbpp", "humaneval"],
        help="Dataset type (determines test program builder)",
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
        "--timeout", type=float, default=5.0,
        help="Per-sample execution timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: auto in metrics/ subdir)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    return parser.parse_args()


def infer_output_path(results_file: Path) -> Path:
    """Derive metrics output path from results file path.

    e.g. results/Qwen--Qwen2.5-7B/pless_t1.0.jsonl
      -> results/Qwen--Qwen2.5-7B/metrics/pless_t1.0_metrics.json
    """
    metrics_dir = results_file.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = results_file.stem  # e.g. pless_t1.0
    return metrics_dir / f"{stem}_metrics.json"


def infer_metadata(results_file: Path, first_record: dict) -> dict:
    """Extract model, method, temperature from filename or first record."""
    model = first_record.get("model", "unknown")
    method = first_record.get("method", "unknown")
    temperature = first_record.get("temperature", 0.0)
    return {"model": model, "method": method, "temperature": temperature}


def main():
    args = parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    t_values = [float(x) for x in args.t.split(",")]

    # Load
    print(f"Loading results from {args.results_file}")
    records = load_results(args.results_file)
    print(f"Loaded {len(records)} tasks")

    # Evaluate
    print(f"Evaluating samples (workers={args.workers}, timeout={args.timeout}s)...")
    task_results = evaluate_all(records, args.dataset, args.timeout, args.workers)

    # Fingerprint for distinct counts
    print("Computing AST fingerprints...")
    add_distinct_counts(task_results, records)

    # Metadata
    meta = infer_metadata(args.results_file, records[0])
    num_samples_per_task = len(records[0]["samples"]) if records else 0

    # Metrics
    pass_at_k = compute_pass_at_k(task_results, k_values)
    cover_at_t, cover_at_t_distinct = compute_cover_at_t(
        task_results, t_values, num_samples_per_task
    )

    output = {
        "model": meta["model"],
        "method": meta["method"],
        "temperature": meta["temperature"],
        "dataset": args.dataset,
        "num_tasks": len(records),
        "num_samples_per_task": num_samples_per_task,
        "pass_at_k": pass_at_k,
        "cover_at_t": cover_at_t,
        "cover_at_t_distinct": cover_at_t_distinct,
        "per_task": task_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    output_path = args.output or infer_output_path(args.results_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\nResults written to {output_path}")
    print(f"  Tasks: {len(records)}")
    print(f"  Samples per task: {num_samples_per_task}")
    print(f"  pass@k: {pass_at_k}")
    print(f"  cover@t: {cover_at_t}")
    print(f"  cover@t (distinct): {cover_at_t_distinct}")


if __name__ == "__main__":
    main()
