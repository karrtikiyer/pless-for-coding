import argparse
import json
from pathlib import Path

from bench.eval.executor import evaluate_all
from bench.eval.loader import load_results
from bench.eval.metrics import build_metrics_output


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
    """Extract model, method, temperature (and top_p if present) from first record.

    For beam search files, the JSONL ``method`` field is just "beam" regardless
    of beam width.  Derive a more specific name (beam4, beam8, …) from the
    filename so that metrics for different beam widths stay distinguishable.
    """
    model = first_record.get("model", "unknown")
    method = first_record.get("method", "unknown")
    temperature = first_record.get("temperature", 0.0)
    top_p = first_record.get("top_p")

    # Derive method from filename for beam configs (beam4_t1.0.jsonl → beam4)
    stem = results_file.stem  # e.g. beam4_t1.0 or beam8_bigcode_t1.0
    if method == "beam" and stem.startswith("beam"):
        # Extract beam prefix: "beam4_t1.0" → "beam4", "beam8_bigcode_t1.0" → "beam8"
        parts = stem.split("_")
        if parts and parts[0].startswith("beam"):
            method = parts[0]

    return {"model": model, "method": method, "temperature": temperature, "top_p": top_p}


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

    # Compute all metrics
    print("Computing AST fingerprints and structural diversity...")
    meta = infer_metadata(args.results_file, records[0])

    output = build_metrics_output(
        task_results,
        records,
        model=meta["model"],
        method=meta["method"],
        temperature=meta["temperature"],
        top_p=meta["top_p"],
        dataset=args.dataset,
        k_values=k_values,
        t_values=t_values,
    )

    # Write output
    output_path = args.output or infer_output_path(args.results_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\nResults written to {output_path}")
    print(f"  Tasks: {output['num_tasks']}")
    print(f"  Samples per task: {output['num_samples_per_task']}")
    print(f"  pass@k: {output['pass_at_k']}")
    print(f"  cover@t: {output['cover_at_t']}")
    print(f"  cover@t (distinct): {output['cover_at_t_distinct']}")
    print(f"  structural_diversity: {output['structural_diversity']}")


if __name__ == "__main__":
    main()
