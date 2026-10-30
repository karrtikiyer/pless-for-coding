"""Sample-level resume bookkeeping for the mlx-lm sequential-groups-of-3
production script (`/tmp/mlxlm_pless_default_seq_groups.py` on the 96GB Mac).
Pure, dependency-free functions, unit tested here and copied verbatim to the
Mac's checkout (no mlx-lm/torch import on purpose).
"""
from __future__ import annotations

import json
from pathlib import Path

N_SAMPLES_DEFAULT = 10


def truncate_to_last_newline(path: Path) -> int:
    """If `path` doesn't end in a newline, truncate it back to the last one.

    Returns the number of trailing bytes dropped (0 if the file already ends
    cleanly, or doesn't exist). This must run BEFORE any reader parses the
    file and before any writer re-opens it in append mode -- otherwise a
    torn trailing line fuses with the next appended record into one
    unparseable line, silently destroying a complete record.
    """
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    last_nl = data.rfind(b"\n")
    kept = last_nl + 1 if last_nl != -1 else 0
    dropped = len(data) - kept
    path.write_bytes(data[:kept])
    return dropped


def load_jsonl_tolerant(path: Path) -> list[dict]:
    """Load a jsonl file, skipping any line that fails to parse.

    Caller is expected to have already run `truncate_to_last_newline` on
    `path` so a torn *final* line is removed outright rather than silently
    skipped here (skipping would still lose that record, just without
    flagging it) -- this function's tolerance is a second line of defense
    for any other malformed line.
    """
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_banked_samples(partial_path: Path) -> dict[str, list[dict]]:
    """Group banked (finished-but-not-yet-task-complete) samples by task_id."""
    banked: dict[str, list[dict]] = {}
    for rec in load_jsonl_tolerant(partial_path):
        banked.setdefault(rec["task_id"], []).append(rec)
    return banked


def remaining_samples_needed(
    task_id: str,
    done_task_ids: set[str],
    banked_by_task: dict[str, list[dict]],
    n_samples: int = N_SAMPLES_DEFAULT,
) -> int:
    """How many NEW samples must be generated for `task_id` on resume.

    0 if the task already has a full record in the main output file
    (`done_task_ids`). Otherwise `n_samples - len(banked)`, floored at 0
    (banked count can meet or exceed n_samples if a prior run banked exactly
    enough but crashed before assembling/writing the full record).
    """
    if task_id in done_task_ids:
        return 0
    banked_count = len(banked_by_task.get(task_id, []))
    return max(0, n_samples - banked_count)


def assemble_record(
    task_id: str,
    banked_samples: list[dict],
    fresh_samples: list[dict],
    n_samples: int,
    base_fields: dict,
) -> dict:
    """Combine banked + freshly-generated samples into one task record.

    Takes exactly `n_samples` total, banked samples first (oldest first,
    insertion order), then fresh. Raises if there aren't enough combined --
    this should be unreachable by construction (caller submits exactly the
    shortfall) and failing loudly here catches a logic error rather than
    silently writing an under-sized record.

    `base_fields` carries everything in the output record that isn't
    sample-derived (model/backend/method/task metadata/etc.) -- same shape
    as the production script's existing `record` dict minus `samples` /
    `samples_with_thinking` / `attempt_ids`.
    """
    combined = banked_samples + fresh_samples
    if len(combined) < n_samples:
        raise ValueError(
            f"task {task_id}: only {len(combined)} samples available "
            f"({len(banked_samples)} banked + {len(fresh_samples)} fresh), "
            f"need {n_samples} -- caller must submit the full shortfall "
            f"before assembling."
        )
    combined = combined[:n_samples]
    record = dict(base_fields)
    record["task_id"] = task_id
    record["samples"] = [s["sample"] for s in combined]
    record["samples_with_thinking"] = [s["sample_with_thinking"] for s in combined]
    record["attempt_ids"] = [s["attempt_id"] for s in combined]
    record["seeds"] = [s.get("seed") for s in combined]
    return record


def build_waves(
    tasks: list[tuple[str, str, int]],
    deep_row_cap: int = 8,
    max_rows_per_wave: int = 32,
    deep_difficulties: tuple[str, ...] = ("medium", "hard"),
) -> list[list[tuple[str, int]]]:
    """Split a group's (task_id, difficulty, n_to_generate) tuples into waves,
    each submitted as one no-backfill BatchGenerator batch.

    Combined rows across ALL "deep" tasks (medium+hard by default -- a crash
    post-mortem on real log data confirmed medium rows can be just as
    resident-and-growing as hard ones, not just hard) are capped at
    `deep_row_cap` TOTAL per wave, not per-task -- a resident deep-looping
    row is the OOM risk regardless of which task it belongs to. Deep tasks
    are drawn down greedily in order, moving to the next task once the
    current one is exhausted, naturally interleaving multiple deep tasks
    within one wave when a task's remaining count doesn't fill the whole cap.
    Easy (non-deep) rows are never split and all ride in the first wave
    alongside the first deep chunk (so they amortize weight-read cost
    instead of running in their own tiny standalone batch). All waves must
    fit within `max_rows_per_wave` (the completion_batch_size ceiling that
    keeps this a no-backfill batch).
    """
    easy = [(tid, n) for tid, diff, n in tasks if diff not in deep_difficulties]
    deep_remaining = [[tid, n] for tid, diff, n in tasks if diff in deep_difficulties]

    waves: list[list[tuple[str, int]]] = []
    first = True
    while True:
        chunk: list[tuple[str, int]] = []
        budget = deep_row_cap
        for item in deep_remaining:
            if budget <= 0:
                break
            tid, rem = item
            if rem <= 0:
                continue
            take = min(rem, budget)
            chunk.append((tid, take))
            item[1] -= take
            budget -= take
        if not chunk:
            break
        wave = (easy + chunk) if first else chunk
        waves.append(wave)
        first = False

    if first and easy:
        # no deep rows at all -- easy-only wave
        waves.append(easy)

    waves = [w for w in waves if w]
    for w in waves:
        total = sum(n for _, n in w)
        if total > max_rows_per_wave:
            raise ValueError(
                f"wave {w} totals {total} rows, exceeds max_rows_per_wave={max_rows_per_wave}"
            )
    return waves


def banked_sample_record(
    task_id: str,
    sample_with_thinking: str,
    sample: str,
    finish_reason: str,
    n_tokens: int,
    attempt_id: str,
    seed: int | None = None,
) -> dict:
    """Shape of one line written to the sidecar partial file."""
    return {
        "task_id": task_id,
        "sample_with_thinking": sample_with_thinking,
        "sample": sample,
        "finish_reason": finish_reason,
        "n_tokens": n_tokens,
        "attempt_id": attempt_id,
        "seed": seed,
    }
