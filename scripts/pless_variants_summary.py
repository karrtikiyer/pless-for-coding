"""Summarise per-task pass counts across pless-variant metric JSONs.

Reads every ``*_metrics.json`` under
results/pless_variants_mps/Qwen--Qwen3-8B/ATCODER_interview/metrics/
and prints:
  - per-variant per-task num_correct out of N samples,
  - solved-tasks (num_correct ≥ 1) count and %,
  - whether the 50% goal (≥8 of 16 CSV ids) is met per variant.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS_DIRS = [
    REPO / "results/pless_variants_mps/Qwen--Qwen3-8B/ATCODER_interview/metrics",
    REPO / "results/pless_variants_cuda/Qwen--Qwen3-8B/ATCODER_interview/metrics",
]
CSV_PATH = (
    REPO / "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/"
    "ATCODER_interview_all_252/analysis/pless_a2_failures_solved_by_others.csv"
)


def main() -> None:
    with CSV_PATH.open() as f:
        csv_ids = [int(r["task_id"]) for r in csv.DictReader(f)]
    csv_set = set(csv_ids)
    csv_ids_sorted = sorted(csv_ids)

    metric_files = []
    for d in METRICS_DIRS:
        if d.exists():
            metric_files.extend(sorted(d.glob("*_metrics.json")))
    if not metric_files:
        print(f"No metric JSONs found under {METRICS_DIRS}")
        return

    per_variant = {}
    variant_source = {}
    for mf in metric_files:
        d = json.loads(mf.read_text())
        variant = mf.stem.replace("_metrics", "")
        if variant in per_variant:
            raise SystemExit(
                f"Metric-file stem collision: '{variant}' appears in both "
                f"{variant_source[variant]} and {mf.parent}. Combining "
                "per_variant[variant] = {...} would silently overwrite the "
                "first backend's data with the second's. Rename one of the "
                "output files (or extend METRICS_DIRS handling to merge "
                "explicitly) before re-running this summary."
            )
        per_variant[variant] = {row["task_id"]: row["num_correct"]
                                for row in d.get("per_task", [])}
        variant_source[variant] = mf.parent

    variants = list(per_variant.keys())
    print(f"{'task_id':>8} | " + " | ".join(f"{v:>28}" for v in variants))
    header_sep = "-" * 9 + "+" + "+".join(["-" * 30] * len(variants))
    print(header_sep)
    for tid in csv_ids_sorted:
        cells = []
        for v in variants:
            nc = per_variant[v].get(tid)
            if nc is None:
                cells.append(f"{'—':>28}")
            else:
                mark = "✓" if nc >= 1 else "·"
                cells.append(f"{mark} nc={nc:>2}                    "[:28])
        print(f"{tid:>8} | " + " | ".join(cells))
    print(header_sep)

    print()
    print("Solved-task summary (num_correct ≥ 1):")
    for v in variants:
        solved = [tid for tid, nc in per_variant[v].items()
                  if tid in csv_set and nc >= 1]
        ran = [tid for tid in per_variant[v] if tid in csv_set]
        pct16 = len(solved) / 16 * 100
        pct_ran = (len(solved) / len(ran) * 100) if ran else 0.0
        goal = "✅ ≥50% of 16" if len(solved) >= 8 else "❌ < 50% of 16"
        print(f"  {v:<32}  solved {len(solved)}/16 CSV ids "
              f"({pct16:.0f}%)  |  {len(solved)}/{len(ran)} of tasks run "
              f"({pct_ran:.0f}%)  |  {goal}")
        if solved:
            print(f"    solved ids: {sorted(solved)}")


if __name__ == "__main__":
    main()
