# Chop + Restart-Thinking Nudge + α=5 — Findings

**Date**: 2026-06-23  
**Experiment**: Pre-registered fair test — chop real saved traces at loop onset, inject restart nudge, continue at α=5 vs α=2 vs forced-`</think>`.  
**Script**: `scripts/chop_restart_alpha_compare.py`  
**Results file**: `results/_chop_restart_probe/qwen3_restart_alpha_n2.json`  
**Log**: `results/_chop_restart_probe/full_n2.log`

---

## Pre-registered arms

| Arm | Action |
|-----|--------|
| A_force | inject `</think>` + python fence → extract existing solution |
| C_restart | chop + restart nudge + continue at pless_alpha(α=5) |
| B_restart | chop + restart nudge + continue at pless_alpha(α=2) (mechanism control) |

**Pre-registered win condition**: C recovers ≥1 task A_force misses AND C_total ≥ A_total.  
**Null hypothesis** (expected per 2506.10979): C ≈ B ≤ A.

---

## Executed results

**Run scope**: 40 tasks (pless_think_t1.0_t1.0, Qwen3-8B ATCODER_interview), n=2, MAX_CONT=2048.

**Run outcome: crashed at task 3 of 40 (task 326). Only 2 tasks completed.**

| Arm | recovered samples / total | tasks recovered |
|-----|--------------------------|-----------------|
| A_force | 0 / 4 | 0 / 2 |
| C_restart | 0 / 4 | 0 / 2 |
| B_restart | 0 / 4 | 0 / 2 |

Per-task breakdown (2 tasks):

| task_id | cut | cut% | A_force | C_restart | B_restart |
|---------|-----|------|---------|-----------|-----------|
| 117 | 14080 chars | 9% | Failed (both) | cap/no_</think> (both) | cap/no_</think> (both) |
| 280 | 3680 chars | 3% | Failed (both) | cap/no_</think> (both) | cap/no_</think> (both) |

**Pre-registered conclusion: null — C ≈ B = A = 0.**  
No arm recovered any task. The null matches the prediction from 2506.10979.

---

## Root-cause analysis

### Why the run crashed

Task 326 (cut@66080 chars ≈ 15K tokens) triggered an unrecoverable Metal assertion:  
`Failed to allocate private MTLBuffer for size 33439048832`

MPS does not support flash attention — attention matrices are O(n²) in memory. At ~15K tokens, each layer's attention matrix is ~32 heads × 15360² × 2 bytes ≈ 14 GB. Two layers of activations in flight → ~33 GB allocation request, exceeding the 64 GB MPS pool.

This is not catchable by the try/except block (Metal assertion, not a Python exception).

**3 of 40 tasks have cuts ≥40K chars** (risky for MPS):

| task_id | cut (chars) | cut% |
|---------|-------------|------|
| 326 | 66080 | 47% |
| 1175 | 59040 | 80% |
| 739 | 43680 | 45% |

### Why MAX_CONT=2048 was too small

**Cut distribution across all 40 tasks** (computed by running `find_loop` offline):

| percentile | cut (chars) | cut% through trace |
|-----------|-------------|-------------------|
| p10 | 6000 | 6% |
| p25 | 9200 | 7% |
| p50 | 16240 | 12% |
| p75 | 30640 | 26% |
| p90 | 36720 | 41% |
| max | 66080 | 47% |

The plan assumed "cap ~2048 tokens (we only generate the short continuation)". This assumed cuts would be late in reasoning. In practice, loop detection fires at **median 12%** through the reasoning trace. A restart from 12% through a hard ATCODER problem requires the model to redo ~88% of its reasoning — easily 5K–20K tokens. 2048 tokens is not a short continuation.

Both completed tasks confirm this: C/B restart hits the 2048-token cap with `end=cap, exec=no_</think>` on all 4 samples — the model starts thinking again but cannot converge within budget.

### Why A_force also failed

A_force injects `</think>` + code fence at the cut point and extracts code. Both tasks had cuts at 3-9% through reasoning — the model hadn't developed any solution yet. Forcing code from 3-9% of a complete reasoning chain produces incorrect solutions.

---

## Conclusions

**On the pre-registered question (C vs A)**: cannot conclude — the run did not produce meaningful data. The 2 completed tasks show a degenerate outcome (all arms = 0) driven by the design mismatch below.

**Design mismatch discovered**: The experiment assumed loop detection fires late (so 2048-token continuation is a "short" extension). Actual data shows it fires at **median 12%** of the reasoning trace. The restart arms need a continuation budget ≫ 2048 to have any chance of closing `</think>` and writing code.

**What would be needed for a valid experiment**:
- MAX_CONT ≥ 8192 (to cover the median case; ideally 16384)
- Filter out 3 tasks with cuts ≥40K chars (MPS OOM), leaving 37 tasks
- Or run on CUDA server (no quadratic attention limit)

**Standing conclusion (unaffected by this experiment)**: Prevention via α=5 from start (A31: 0.80 pass@1, 0.4% trunc) remains the dominant strategy. Among rescue approaches, forced-`</think>` (11/27 recovery, verified) is the only empirically validated lever.

---

## Relationship to prior A28 findings

This experiment was designed as the "fair test" to resolve the confound identified 2026-06-13: `chop_regen_probe.py` used freshly regenerated traces (confounded) while this script uses real saved traces. That design fix was correct. The new failure was in the continuation cap assumption and MPS context limits.

The 2506.10979 null prediction (C ≈ B ≤ A, restart nudge is weak) remains unrefuted — but also untested at the scale needed to be informative.
