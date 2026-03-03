"""Batch-evaluate all HumanEval temperature sweep JSONL files.

Discovers JSONL files under temprature_results/*/humaneval/, runs the
standard evaluation pipeline (execute → fingerprint → metrics), and writes
a metrics JSON per file.  Skips files that already have a metrics JSON
(for resumability).

Usage:
    python -m bench.eval.eval_temperature_sweep [--workers 4] [--timeout 5]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bench.eval.executor import evaluate_all
from bench.eval.loader import load_results
from bench.eval.metrics import (
    add_distinct_counts,
    add_structural_diversity,
    compute_cover_at_t,
    compute_pass_at_k,
    compute_structural_diversity,
)

RESULTS_ROOT = Path("results/pless_human_eval_results/temprature_results")

K_VALUES = [1, 3, 5, 10]
T_VALUES = [0.1, 0.3, 0.5, 0.7]


def discover_jsonl_files(root: Path) -> list[Path]:
    """Find all JSONL files under root/*/humaneval/."""
    files = sorted(root.glob("*/humaneval/*.jsonl"))
    return files


def metrics_path_for(jsonl_path: Path) -> Path:
    """Derive metrics output path: .../metrics/{stem}_metrics.json."""
    metrics_dir = jsonl_path.parent.parent / "metrics"
    return metrics_dir / f"{jsonl_path.stem}_metrics.json"


def evaluate_file(jsonl_path: Path, workers: int, timeout: float) -> dict:
    """Run the full evaluation pipeline on a single JSONL file."""
    records = load_results(jsonl_path)

    # Evaluate
    task_results = evaluate_all(records, "humaneval", timeout, workers)

    # Fingerprint for distinct counts
    add_distinct_counts(task_results, records)

    # Structural diversity
    add_structural_diversity(task_results, records)

    # Metadata from first record
    first = records[0]
    num_samples_per_task = len(first["samples"]) if records else 0

    # Metrics
    pass_at_k = compute_pass_at_k(task_results, K_VALUES)
    cover_at_t, cover_at_t_distinct = compute_cover_at_t(
        task_results, T_VALUES, num_samples_per_task
    )
    structural_diversity = compute_structural_diversity(task_results)

    return {
        "model": first.get("model", "unknown"),
        "method": first.get("method", "unknown"),
        "temperature": first.get("temperature", 0.0),
        "dataset": "humaneval",
        "num_tasks": len(records),
        "num_samples_per_task": num_samples_per_task,
        "pass_at_k": pass_at_k,
        "cover_at_t": cover_at_t,
        "cover_at_t_distinct": cover_at_t_distinct,
        "structural_diversity": structural_diversity,
        "per_task": task_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate HumanEval temperature sweep results"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for code execution (default: 4)",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Per-sample execution timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if metrics JSON already exists",
    )
    args = parser.parse_args()

    jsonl_files = discover_jsonl_files(RESULTS_ROOT)
    print(f"Found {len(jsonl_files)} JSONL files to evaluate")

    done, skipped = 0, 0
    for i, jsonl_path in enumerate(jsonl_files, 1):
        out_path = metrics_path_for(jsonl_path)
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(jsonl_files)}] SKIP (exists): {out_path.name}")
            skipped += 1
            continue

        print(f"[{i}/{len(jsonl_files)}] Evaluating: {jsonl_path}")
        metrics = evaluate_file(jsonl_path, args.workers, args.timeout)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        p1 = metrics["pass_at_k"].get("1", 0)
        sd = metrics["structural_diversity"]
        print(f"  → {out_path.name}  pass@1={p1:.3f}  diversity={sd:.4f}")
        done += 1

    print(f"\nDone: {done} evaluated, {skipped} skipped (already existed)")


if __name__ == "__main__":
    main()
