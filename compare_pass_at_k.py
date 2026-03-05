"""Compare pass@k between new temp-sweep pipeline and old full-precision pipeline.

Re-runs all tests (does NOT use precomputed `passed` values).

Usage:
  uv run python compare_pass_at_k.py \
    --new-results /path/to/new.jsonl \
    --old-method temp_0.7

  uv run python compare_pass_at_k.py \
    --new-results /path/to/new.jsonl \
    --old-results /path/to/old.json \
    --old-method p_less
"""

import argparse
import json
import math
from pathlib import Path

import importlib.util as _ilu

# Load executor directly to avoid bench.eval.__init__ pulling in zss
_spec = _ilu.spec_from_file_location("executor", "bench/eval/executor.py")
_executor = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_executor)
extract_python_code = _executor.extract_python_code
_build_program_humaneval = _executor._build_program_humaneval
check_sample = _executor.check_sample


DEFAULT_OLD_RESULTS = Path(
    "results/pless_human_eval_results/full_precision_results/"
    "Codestral-22B/humaneval_gpu_20251217_001844_detailed.json"
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_new_results(path: Path) -> dict:
    """Load new pipeline JSONL results, keyed by task_id.

    Each record includes 'test' and 'entry_point', so we use these as the
    source of truth for test cases (avoids downloading the HumanEval dataset).
    """
    results = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            results[record["task_id"]] = record
    return results


def load_old_results(path: Path, method: str) -> dict:
    """Load old full-precision JSON results for the given method key, keyed by task_id."""
    with open(path) as f:
        data = json.load(f)
    if method not in data:
        available = ", ".join(data.keys())
        raise KeyError(f"Method '{method}' not found in {path}. Available keys: {available}")
    results = {}
    for task in data[method]:
        results[task["task_id"]] = task
    return results


def evaluate_samples(samples: list[str], test: str, entry_point: str) -> list[bool]:
    """Run each sample through extract -> build -> check, return pass/fail list."""
    results = []
    for sample in samples:
        clean = extract_python_code(sample)
        program = _build_program_humaneval(clean, test, entry_point)
        passed = check_sample(clean, program, timeout=5.0)
        results.append(passed)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare pass@k between new and old Codestral pipelines"
    )
    parser.add_argument(
        "--new-results", type=Path, required=True,
        help="Path to new pipeline JSONL file",
    )
    parser.add_argument(
        "--old-results", type=Path, default=DEFAULT_OLD_RESULTS,
        help=f"Path to old pipeline JSON file (default: {DEFAULT_OLD_RESULTS})",
    )
    parser.add_argument(
        "--old-method", type=str, required=True,
        help="Key in the old JSON to compare against (e.g. temp_0.7, p_less)",
    )
    args = parser.parse_args()

    print(f"New results: {args.new_results}")
    print(f"Old results: {args.old_results} [method={args.old_method}]")

    new_data = load_new_results(args.new_results)
    old_data = load_old_results(args.old_results, args.old_method)

    all_task_ids = sorted(set(new_data.keys()) | set(old_data.keys()))
    print(f"\nNew pipeline: {len(new_data)} tasks | Old pipeline: {len(old_data)} tasks | Union: {len(all_task_ids)} tasks\n")

    # Per-task results storage
    task_results = []  # list of dicts

    new_pass1_sum = 0.0
    new_pass10_sum = 0.0
    old_pass1_sum = 0.0
    old_pass10_sum = 0.0
    new_task_count = 0
    old_task_count = 0
    differ_tasks = []

    for i, task_id in enumerate(all_task_ids):
        # Get test cases from the new pipeline results (avoids dataset download)
        new_record = new_data.get(task_id)
        if not new_record:
            print(f"  WARNING: {task_id} missing from new pipeline, skipping")
            continue
        test = new_record["test"]
        entry_point = new_record["entry_point"]
        prompt = new_record["prompt_text"]

        row = {"task_id": task_id}

        # --- New pipeline ---
        new_samples = new_record["samples"]
        new_pass = evaluate_samples(new_samples, test, entry_point)
        n = len(new_pass)
        c = sum(new_pass)
        row["new_n"] = n
        row["new_c"] = c
        row["new_pass1"] = pass_at_k(n, c, 1)
        row["new_pass10"] = pass_at_k(n, c, 10)
        new_pass1_sum += row["new_pass1"]
        new_pass10_sum += row["new_pass10"]
        new_task_count += 1

        # --- Old pipeline ---
        if task_id in old_data:
            old_record = old_data[task_id]
            old_samples = [prompt + s["code"] for s in old_record["samples"]]
            old_pass = evaluate_samples(old_samples, test, entry_point)
            n = len(old_pass)
            c = sum(old_pass)
            row["old_n"] = n
            row["old_c"] = c
            row["old_pass1"] = pass_at_k(n, c, 1)
            row["old_pass10"] = pass_at_k(n, c, 10)
            old_pass1_sum += row["old_pass1"]
            old_pass10_sum += row["old_pass10"]
            old_task_count += 1

        # Check for significant difference
        if "new_pass1" in row and "old_pass1" in row:
            diff1 = abs(row["new_pass1"] - row["old_pass1"])
            diff10 = abs(row["new_pass10"] - row["old_pass10"])
            if diff1 > 0.3 or diff10 > 0.3:
                row["flag"] = True
                differ_tasks.append(task_id)

        task_results.append(row)

        # Progress indicator every 20 tasks
        if (i + 1) % 20 == 0 or (i + 1) == len(all_task_ids):
            print(f"  Evaluated {i + 1}/{len(all_task_ids)} tasks...")

    # --- Print per-task table ---
    print(f"\n{'=' * 100}")
    print(f"{'Task':<18} {'New pass@1':>10} {'New pass@10':>11} {'New C/N':>8}"
          f"  {'Old pass@1':>10} {'Old pass@10':>11} {'Old C/N':>8} {'Flag':>5}")
    print(f"{'-' * 100}")

    for row in task_results:
        tid = row["task_id"]
        np1 = f"{row['new_pass1']:.4f}" if "new_pass1" in row else "---"
        np10 = f"{row['new_pass10']:.4f}" if "new_pass10" in row else "---"
        ncn = f"{row.get('new_c', '-')}/{row.get('new_n', '-')}" if "new_n" in row else "---"
        op1 = f"{row['old_pass1']:.4f}" if "old_pass1" in row else "---"
        op10 = f"{row['old_pass10']:.4f}" if "old_pass10" in row else "---"
        ocn = f"{row.get('old_c', '-')}/{row.get('old_n', '-')}" if "old_n" in row else "---"
        flag = "  ***" if row.get("flag") else ""
        print(f"{tid:<18} {np1:>10} {np10:>11} {ncn:>8}"
              f"  {op1:>10} {op10:>11} {ocn:>8} {flag}")

    # --- Aggregate ---
    print(f"\n{'=' * 100}")
    print("  AGGREGATE (mean pass@k across tasks)")
    print(f"{'=' * 100}")
    if new_task_count > 0:
        print(f"  NEW: pass@1 = {new_pass1_sum / new_task_count:.4f}  |  "
              f"pass@10 = {new_pass10_sum / new_task_count:.4f}  ({new_task_count} tasks)")
    if old_task_count > 0:
        print(f"  OLD: pass@1 = {old_pass1_sum / old_task_count:.4f}  |  "
              f"pass@10 = {old_pass10_sum / old_task_count:.4f}  ({old_task_count} tasks)")

    if differ_tasks:
        print(f"\n  *** {len(differ_tasks)} tasks with >0.3 difference in pass@1 or pass@10:")
        for t in differ_tasks:
            print(f"      {t}")
    print()


if __name__ == "__main__":
    main()
