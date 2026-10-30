# Proposal: difficulty-pool scheduling for the mlx-lm production pipeline (PROPOSED, not implemented)

**Status**: PROPOSED — discussed 2026-10-29, parked until the current v5
60-task stratified run (`/tmp/mlxlm_pless_default_seq_groups.py`) finishes
all 20 groups. Do not implement without a fresh discussion first (standing
no-unilateral-decisions rule).

## Context: current (v5) design

`bench/mlx_seq_groups_resume.py`'s `build_waves()` processes the 60
LCB_v6_stratified60 tasks (20 easy + 20 medium + 20 hard, verified via
`results/pless_variants_cuda/Qwen--Qwen3-8B/LCB_v6_stratified60/pless_default_*.jsonl`)
in **groups of 3** (1 easy + 1 medium + 1 hard), sequentially. Within a
group, `DEEP_ROW_CAP=8` caps **medium+hard rows combined** (not hard
alone — see rationale below) per wave; easy rows ride along uncapped in
wave 1. A typical 10/10/10 group splits into 3 waves: easy(10)+medium(8),
medium(2)+hard(6), hard(4).

**Why medium is included in the cap, not just hard** (verified from real
data, not assumed): sample word-counts for Group 4's tasks showed easy
~1.4-1.9k words, but medium (6.7k-10.3k) and hard (7.5k-16.2k) **heavily
overlap** — medium's longest sample (10,258 words) exceeds 6 of hard's 10
samples. A human-assigned difficulty label doesn't reliably predict how
verbose the model's own reasoning trace will be. Capping hard alone would
have left the same batch-size x max-length OOM risk via a long medium row.

**Root OOM mechanism** (verified against mlx-lm's own source,
`mlx_lm/models/cache.py:880-933`, `BatchKVCache.update_and_fetch`):
the KV cache is one shared tensor per batch, shape
`(B, n_kv_heads, n_steps*256, head_dim)` — sized to whichever row in the
batch has generated the most tokens so far. Memory scales as
`B x max_active_length`, not the sum of each row's actual length. Before
the fix, up to 30 rows (10 easy+10 medium+10 hard) could be resident
together; easy rows finish fast but don't shrink the cache, so by the
time only the slow medium/hard rows remained, the tensor was sized for
30 rows x 20k+ tokens, hitting the wired-memory ceiling
(`RuntimeError: [METAL] Command buffer execution failed: Insufficient
Memory`, confirmed in `/tmp/mlxlm_pless_default_seq_groups_v3_crash.log`).

**Current run's observed safety margin** (verified from the live v5 log,
zero OOM/error/traceback across 8 process attempts so far): highest peak
recorded is 63.036GB against an 83.49GB wired-memory limit (device's
recommended max working set, auto-set by `BatchGenerator.__init__`) —
~20.5GB headroom, and ~33GB unused against the 96GB physical total.

## The observation motivating this proposal

The current design leaves real headroom on the table. Two independent
inefficiencies:

1. **No backfill, ever.** Every wave is a single `gen.insert(...)` call
   followed by draining to completion with no further `insert()` calls
   (verified: `run_wave()` in `/tmp/mlxlm_pless_default_seq_groups.py`
   calls `insert()` exactly once per wave). As rows finish, the batch
   only shrinks — nothing replaces a finished easy row to keep the GPU/ANE
   busy. mlx-lm's own `BatchGenerator` supports backfill via
   `completion_batch_size` (default 32, verified in
   `mlx_lm/generate.py:1614,1880-1888`) precisely for this situation, but
   our design never exercises it, because we always submit <= 32 rows per
   wave on purpose (to keep the deep-row cap airtight — see below).
2. **~20-33GB of unused memory margin** at the worst observed peak,
   while concurrency caps (18 max in wave 1, 8 in deep-only waves) are
   conservative relative to what's actually been proven safe so far.

## Proposed alternative: three global difficulty pools

Instead of groups-of-3 processed sequentially, restructure into three
sequential phases across **all 60 tasks at once**, each phase sized to
that difficulty's actual length risk:

1. **Easy phase**: capped batches of 20 rows, **no backfill** (same
   mechanics as medium). *Revised 2026-10-30 — the original version of
   this step (all 200 easy rows in one backfill batch at 32) was WRONG and
   was dropped after independent review; see "Correction" below.*

   **Correction (2026-10-30).** The original claim — easy rows are
   uniformly short, so backfill is low-risk — was false on both counts:
   - Easy rows are not uniformly short. Checked in the vLLM baseline
     (`results/pless_variants_cuda/Qwen--Qwen3-8B/LCB_v6_stratified60/pless_default_*_think_t1.0.jsonl`):
     easy task `3626` has 4 rows of 71-75k characters (likely at or near
     the 25000-token cap) and `3471` has one of 48k; the median easy row
     is ~8.7k characters.
   - Backfill keeps the batch at full size while a long row is resident.
     mlx-lm 0.32.0's `BatchKVCache.extend` left-pads a joining row to the
     current longest row, and `filter` only trims padding common to all
     rows (reviewer read `mlx_lm/models/cache.py`). So one looping easy row
     pins the whole 32-row tensor at its length.
   - Qwen3-8B KV cost (config on the Mac: 36 layers, 8 KV heads, head_dim
     128, bf16) is 147,456 bytes per token per row; 32 x 25000 tokens is
     ~118GB, more than the machine's 96GB.
   Without backfill the batch shrinks as rows finish, which is the safety
   property the current v5 design relies on.
2. **Medium phase**: cap at 20 concurrent, no backfill, clear-and-requeue
   per sub-batch of 20 (200 medium samples / 20 = 10 sub-batches) — same
   no-backfill mechanics as today, just a higher cap and applied to a
   medium-only pool (no hard rows mixed in).
3. **Hard phase**: cap at 10 concurrent — i.e. one hard task's full 10
   samples per sub-batch, cleared before the next task. This mirrors
   what GPU2's vLLM full-1046 sweep already does per the user's
   description (effectively one task at a time for the slowest content).
   This is actually a *simplification* relative to today's design, which
   can split a single hard task's rows awkwardly across waves interleaved
   with a different task's leftover medium rows (e.g. observed:
   `wave 2: medium(2)+hard(6)`, `wave 3: hard(4)` — one hard task's rows
   split non-contiguously because `build_waves()`'s deep budget draws
   down greedily across task boundaries).

## Assessment

| Phase | Verdict | Confidence |
|---|---|---|
| Easy (cap 20, no backfill) | Revised from "32 with backfill" after review showed that version could exceed physical memory. Same exposure class as medium (worst case 20 rows all running long together), but most easy rows finish early so the batch shrinks fast | Needs the same empirical check as medium |
| Medium (cap 20, no backfill) | Untested — 2.5x today's effective cap (8), and unlike today's mixed waves, nothing short drops out early to shrink the batch. No verified memory model to confirm 20 concurrent ~13-14k-token rows stays under the 83.49GB wired limit | Needs empirical validation before trusting |
| Hard (cap 10, no backfill, one task at a time) | Plausible, and simpler than today's cross-task-split waves; a modest increase over today's observed-safe isolated hard-only waves (4-6 rows) | Likely safe, still worth a smoke test |

## What implementing this actually requires (bigger than a config tweak)

Today's resume/restart bookkeeping (`GROUPS_PER_PROCESS`, done-task
counting, wave labeling in `run_wave()`/`main()`) is built around
"group of 3 tasks" as the unit of work. Restructuring into three global
difficulty-pools means:

- Task-level resume (keyed by `task_id` in `bench/mlx_seq_groups_resume.py`)
  still works unchanged underneath — this part needs no redesign.
- The group/wave loop in `main()`, the `GROUPS_PER_PROCESS` restart
  trigger, and the wrapper's stop-condition logic all assume groups-of-3
  as the atomic unit and would need rework to key off "difficulty phase"
  instead.
- `DEEP_ROW_CAP` is currently a hardcoded literal (`= 8` at
  `/tmp/mlxlm_pless_default_seq_groups.py:136`), not env-overridable like
  `MAX_NEW_TOKENS`/`MAX_GROUPS`/`GROUPS_PER_PROCESS` — a per-phase cap
  (32/20/10) would need its own wiring, not just a value change.

## Before implementing

1. Let the current v5 60-task run finish all 20 groups (explicit user
   instruction, 2026-10-29) — do not interrupt it for this.
2. Validate the medium=20 cap empirically with a scaled-down kill-test
   (same methodology used to validate `DEEP_ROW_CAP=8`: scratch
   `OUT_DIR`, small `MAX_NEW_TOKENS`, poll-based observation of peak
   memory) before trusting it on production data.
3. Discuss the restructuring scope with the user before touching the
   resume/wave bookkeeping — this is a real architecture change, not a
   one-line constant edit.
