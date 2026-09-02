#!/usr/bin/env python
"""Select a reproducible seeded-random subset of APPS CodeForces-interview problem_ids.

Paper B's cross-source replication runs on a 748-of-2386 subset of CODEFORCES/interview.
This script fixes that subset forever: it loads the full CF-interview bucket, sorts by
problem_id for determinism (independent of HF iteration order), draws N without replacement
with a fixed seed, and writes the sorted ids (one per line) to a committed file. Re-running
with the same (seed, N, source, difficulty) yields a byte-identical file.

Run: uv run python scripts/select_cf_subset.py
     (offline once cached) HF_HUB_OFFLINE=1 uv run python scripts/select_cf_subset.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from bench.apps.dataset import load_apps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="CODEFORCES")
    ap.add_argument("--difficulty", default="interview")
    ap.add_argument("--n", type=int, default=748, help="subset size")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="data/cf_interview_748_ids.txt")
    args = ap.parse_args()

    # Full bucket, tests not needed for id selection (faster startup).
    ids = sorted(
        p.problem_id
        for p in load_apps(source=args.source, difficulty=args.difficulty, with_tests=False)
    )
    total = len(ids)
    if args.n > total:
        raise SystemExit(f"requested n={args.n} > bucket size {total}")

    rng = np.random.default_rng(args.seed)
    # Sample indices without replacement, then return the sorted problem_ids so the
    # output order is deterministic and independent of the draw order.
    picked = sorted(int(ids[i]) for i in rng.choice(total, size=args.n, replace=False))
    assert len(picked) == args.n == len(set(picked)), "selection not unique / wrong size"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(str(i) for i in picked) + "\n")

    print(f"{args.source}/{args.difficulty}: bucket={total}, selected={len(picked)} "
          f"(seed={args.seed})")
    print(f"id range: {picked[0]}..{picked[-1]}  ->  {args.out}")


if __name__ == "__main__":
    main()
