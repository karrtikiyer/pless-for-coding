"""Merge multiple per-batch JSONL files (from run_pless_variants_qwen3_8b_cuda.py
launched with different --seed and --batch-suffix values) into a single JSONL
with concatenated ``samples`` per task.

The runner produces one file per (variant, mode, temperature, batch-suffix). To
get an effective N-per-task = sum-of-batch-N-per-task, run this merger:

    uv run python scripts/merge_pless_batches.py \\
        --out-file results/…/pless_effk_think_t1.0.n10.jsonl \\
        --in-files  results/…/pless_effk_think_t1.0.batch0.jsonl \\
                     results/…/pless_effk_think_t1.0.batch1.jsonl

Each input JSONL is a line-per-task record with a ``samples`` array (and
optionally ``samples_with_thinking``). We group by ``task_id``, concatenate
those arrays across inputs, and emit one line per task with the metadata
from the first input in which the task appears. If a task is missing from
one input (e.g. that batch didn't reach it before deadline), we still merge
whatever samples exist for it — the merged N for that task will be less
than the sum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_records(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--in-files", type=Path, nargs="+", required=True,
                    help="Batch JSONL files to merge, in order.")
    args = ap.parse_args()

    # task_id → merged record. First occurrence carries the metadata; subsequent
    # occurrences append to samples[] and samples_with_thinking[].
    merged: dict[int, dict] = {}
    for path in args.in_files:
        if not path.exists():
            print(f"WARN: missing input {path} — skipping")
            continue
        n_lines = 0
        for rec in _iter_records(path):
            n_lines += 1
            tid = int(rec["task_id"])
            if tid not in merged:
                merged[tid] = {**rec, "samples": list(rec.get("samples", []))}
                if "samples_with_thinking" in rec:
                    merged[tid]["samples_with_thinking"] = list(
                        rec["samples_with_thinking"]
                    )
                merged[tid]["_source_files"] = [str(path)]
            else:
                merged[tid]["samples"].extend(rec.get("samples", []))
                if "samples_with_thinking" in rec:
                    merged[tid].setdefault(
                        "samples_with_thinking", []
                    ).extend(rec["samples_with_thinking"])
                merged[tid]["_source_files"].append(str(path))
        print(f"  read {n_lines:>3} records from {path.name}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w") as fh:
        for tid in sorted(merged):
            fh.write(json.dumps(merged[tid]) + "\n")

    print(f"\nMerged {len(merged)} tasks → {args.out_file}")
    per_task_n = [len(r["samples"]) for r in merged.values()]
    if per_task_n:
        print(f"  samples per task: min={min(per_task_n)} "
              f"max={max(per_task_n)} avg={sum(per_task_n)/len(per_task_n):.1f}")


if __name__ == "__main__":
    main()
