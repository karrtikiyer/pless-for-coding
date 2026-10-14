"""Generate a stratified batch assignment for the full LCB v6 dataset
(1046 tasks: 602 AtCoder + 444 LeetCode, CodeForces's 9 excluded since the
runner scripts' --platforms only support atcoder/leetcode).

Each batch is itself a representative mini-sample of the full dataset (not a
sequential slice) -- within each (platform, difficulty) stratum, tasks are
sorted by task_id (deterministic) and round-robin-assigned across batches,
so every batch gets as close to a proportional share of every stratum as
integer division allows (off by at most 1 per stratum).

Motivation: a full-1046 run is a multi-week commitment on a single GPU at
reasoning-model token budgets (measured, not estimated -- see docs/theory/
todos.md and this session's discussion). Processing in stratified batches
means batch 1 alone is both a real throughput pilot AND useful, keepable
data -- not a throwaway smoke test -- and later batches can run on whichever
GPU frees up next (gpu2 or Laptop) independently, since resume/dedup is
already task_id-keyed in the runner scripts.

Usage::

    uv run python scripts/generate_lcb_full1046_batches.py --batch-size 250
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from bench.livecodebench.dataset import load_lcb

REPO = Path(__file__).resolve().parent.parent
OUT_CSV = REPO / "results/lcb_v6_full1046/batch_assignment.csv"

# NOTE: the original version of this script excluded the 60 already-
# generated stratified tasks from the batch assignment, to reuse their
# existing data and save compute. Reverted (2026-10-09): those 60 were
# generated at the OLD max_new_tokens=25000/max_model_len=27000 budget;
# the full-1046 run uses a raised 35000/37500 budget (per docs/theory/
# todos.md A67/C12), so reusing them would silently mix two different
# generation conditions into one "1046-task" result -- invalid apples-to-
# apples comparison. All 1046 tasks (including the 60) are now regenerated
# uniformly at the new budget, so they're batched together with everything
# else.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=250)
    args = ap.parse_args()

    problems = list(load_lcb(platforms=("atcoder", "leetcode"), with_tests=False))
    total = len(problems)
    n_batches = -(-total // args.batch_size)  # ceil
    print(f"Total tasks: {total}; batch_size={args.batch_size} -> {n_batches} batches")

    by_stratum: dict[tuple[str, str], list] = defaultdict(list)
    for p in problems:
        by_stratum[(p.platform, p.difficulty)].append(p)

    rows = []
    for (platform, difficulty), plist in sorted(by_stratum.items()):
        plist.sort(key=lambda p: p.task_id)
        for i, p in enumerate(plist):
            batch = i % n_batches  # round-robin -> proportional per stratum
            rows.append({
                "task_id": p.task_id, "platform": p.platform,
                "difficulty": p.difficulty, "batch": batch,
            })
        print(f"  {platform:10s} {difficulty:7s}: {len(plist):4d} tasks "
              f"-> ~{len(plist)/n_batches:.1f}/batch")

    rows.sort(key=lambda r: (r["batch"], r["platform"], r["task_id"]))

    batch_counts = defaultdict(int)
    for r in rows:
        batch_counts[r["batch"]] += 1
    print("Batch sizes:", dict(sorted(batch_counts.items())))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "platform", "difficulty", "batch"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
