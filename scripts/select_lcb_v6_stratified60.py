"""Select a deterministic, stratified 60-problem subset of LiveCodeBench v6 for
the pless-variants-vs-baselines decoding comparison on Qwen3-8B.

Design (confirmed with user 2026-09-25): AtCoder + LeetCode only (CodeForces has
just 9 problems total in v6 — too few to stratify), 10 problems per
(platform x difficulty) cell = 60 total. Deterministic seed=42 sample within
each cell, sorted by question_id for reproducibility.

Output: results/lcb_v6_stratified60/task_selection.csv with columns
task_id, platform, difficulty, contest_date — mirrors the shape of
results/pless_cot_efficiency_vllm/.../pless_a2_failures_solved_by_others.csv
so downstream tooling can load_lcb(task_ids=...) as an allowlist.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "results/lcb_v6_stratified60/task_selection.csv"

PLATFORMS = ("atcoder", "leetcode")
DIFFICULTIES = ("easy", "medium", "hard")
PER_CELL = 10
SEED = 42


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(
        "livecodebench/code_generation_lite", version_tag="release_v6",
        split="test", trust_remote_code=True,
    )

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in ds:
        plat, diff = row["platform"], row["difficulty"]
        if plat not in PLATFORMS:
            continue
        by_cell.setdefault((plat, diff), []).append({
            "task_id": str(row["question_id"]),
            "platform": plat,
            "difficulty": diff,
            "contest_date": row["contest_date"][:10],
        })

    rng = random.Random(SEED)
    selected: list[dict] = []
    for plat in PLATFORMS:
        for diff in DIFFICULTIES:
            pool = by_cell.get((plat, diff), [])
            if len(pool) < PER_CELL:
                raise RuntimeError(
                    f"cell ({plat}, {diff}) has only {len(pool)} problems, "
                    f"need {PER_CELL}"
                )
            pool_sorted = sorted(pool, key=lambda r: r["task_id"])
            picked = rng.sample(pool_sorted, PER_CELL)
            picked.sort(key=lambda r: r["task_id"])
            selected.extend(picked)
            print(f"  {plat:<10} {diff:<8} picked {len(picked)}/{len(pool)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "platform", "difficulty", "contest_date"])
        w.writeheader()
        for row in selected:
            w.writerow(row)

    print(f"\nWrote {len(selected)} task ids -> {OUT_PATH}")


if __name__ == "__main__":
    main()
