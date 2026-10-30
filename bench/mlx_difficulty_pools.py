"""Pure scheduler for the mlx-lm difficulty-pool design (todos E24,
docs/mlx_seq_groups_difficulty_pool_scheduling_proposal.md).

Replaces "groups of 3 tasks, medium+hard capped together" with three global
phases drained in order: easy, medium, hard. Every batch is capped and has
NO backfill: one `insert()` per batch, drained to completion, fresh
BatchGenerator for the next batch.

No backfill, even for easy: mlx-lm's BatchKVCache pads a backfilled row to the
longest resident row, so a 32-row easy batch with one looping row pins
32 x ~25k tokens of KV (~118GB for Qwen3-8B) -- easy rows DO hit the cap
(task 3626). Without backfill the batch shrinks as rows finish.

Deliberately a NEW module: `bench/mlx_seq_groups_resume.py` is checksum-gated
by the live v5 run. No mlx-lm/torch import on purpose.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

MLX_DEFAULT_COMPLETION_BATCH_SIZE = 32
EASY_CAP_DEFAULT = 25
MEDIUM_CAP_DEFAULT = 16
HARD_CAP_DEFAULT = 10

_PHASE_ORDER = ("easy", "medium", "hard")


@dataclass(frozen=True)
class PoolBatch:
    phase: str
    rows: list[tuple[str, int]]  # (task_id, n_rows)
    completion_batch_size: int  # pass to BatchGenerator(completion_batch_size=...)


def _pack(rows: list[tuple[str, int]], cap: int) -> list[list[tuple[str, int]]]:
    """Greedy-fill batches up to `cap` rows, splitting a task across batches."""
    batches: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    room = cap
    for tid, n in rows:
        while n > 0:
            take = min(n, room)
            current.append((tid, take))
            n -= take
            room -= take
            if room == 0:
                batches.append(current)
                current, room = [], cap
    if current:
        batches.append(current)
    return batches


def build_pool_batches(
    tasks: list[tuple[str, str, int]],
    *,
    easy_cap: int = EASY_CAP_DEFAULT,
    medium_cap: int = MEDIUM_CAP_DEFAULT,
    hard_cap: int = HARD_CAP_DEFAULT,
) -> list[PoolBatch]:
    """Split (task_id, difficulty, n_to_generate) tuples into ordered batches.

    `completion_batch_size >= rows` so mlx-lm never queues a row, and so can
    never backfill past the cap.
    """
    caps = {"easy": easy_cap, "medium": medium_cap, "hard": hard_cap}
    for phase, value in caps.items():
        if value <= 0:
            raise ValueError(f"{phase}_cap must be positive, got {value}")

    pools: dict[str, list[tuple[str, int]]] = {p: [] for p in _PHASE_ORDER}
    seen: set[str] = set()
    for tid, difficulty, n in tasks:
        if difficulty not in pools:
            raise ValueError(f"unknown difficulty {difficulty!r} for task {tid}")
        if type(n) is not int or n < 0:
            raise ValueError(f"row count for task {tid} must be a non-negative int, got {n!r}")
        if tid in seen:
            raise ValueError(f"duplicate task_id {tid!r}")
        seen.add(tid)
        if n > 0:
            pools[difficulty].append((tid, n))

    batches: list[PoolBatch] = []
    for phase in _PHASE_ORDER:
        for rows in _pack(pools[phase], caps[phase]):
            total = sum(n for _, n in rows)
            batches.append(PoolBatch(phase, rows, max(total, MLX_DEFAULT_COMPLETION_BATCH_SIZE)))
    return batches


def select_next_phase(
    batches: list[PoolBatch], only_phase: str | None = None,
) -> list[PoolBatch]:
    """Batches of the first phase (in easy -> medium -> hard order) that still
    has work. The runner restarts the process after each phase, so each
    process runs exactly one phase. `only_phase` restricts to one phase
    (empty list if it has no work).
    """
    if only_phase is not None:
        if only_phase not in _PHASE_ORDER:
            raise ValueError(f"unknown phase {only_phase!r}")
        return [b for b in batches if b.phase == only_phase]
    for phase in _PHASE_ORDER:
        selected = [b for b in batches if b.phase == phase]
        if selected:
            return selected
    return []


def assign_row_ranges(
    batches: list[PoolBatch], banked: dict[str, int],
) -> list[list[tuple[str, int, int]]]:
    """Give each (task_id, n) in each batch a row range (task_id, first_row, n).
    A task's first row starts at its banked count and advances in batch order.
    Banked counts only grow, so a range that banked anything never recurs (the
    seed guarantee). The numbers are slot labels, NOT unique row ids: after a
    mid-batch crash, rows that finished out of order can repeat a label."""
    cursor = dict(banked)
    out = []
    for b in batches:
        ranges = []
        for tid, n in b.rows:
            start = cursor.get(tid, 0)
            ranges.append((tid, start, n))
            cursor[tid] = start + n
        out.append(ranges)
    return out


def batch_seed(base: int, config: str, phase: str, row_ranges: list[tuple[str, int, int]]) -> int:
    """Deterministic per-batch seed from (base, config, phase, row ranges).

    A repeated seed can only recur if the earlier batch banked nothing for any
    of its tasks (banking moves the row ranges on), so it can only replay work
    that was lost -- never duplicate a saved sample. sha256, not hash(): hash()
    is salted per process.
    """
    payload = json.dumps([base, config, phase, [list(r) for r in row_ranges]])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
