# Chop-and-Continue Loop Rescue — Findings (Qwen3-8B, APPS ATCODER-interview)

**Date**: 2026-06-25
**Question**: while running pless @ α=2 (the confident-case default — best pass@1), can we
DETECT a thinking-phase loop and ESCAPE it — recovering a task that would otherwise
truncate-and-fail — and is that escape better than the current rescue (force `</think>` and
extract whatever solution already exists)?

**Method**: for each task, take its REAL saved pless ramble (the α=2 generation that looped to
the 32768-token cap, emitted no `</think>`, produced no code → auto-failed), chop it at the loop
onset (`find_loop`), and run four arms on the IDENTICAL chopped prefix (only the post-chop action
differs):

| Arm | Action after chop |
|-----|-------------------|
| A_force | force `</think>` + ```` ```python ````, extract the existing solution (baseline rescue) |
| chop_only | continue thinking, NO nudge (bare-chop control) |
| chop_pivot | nudge "step back, try a different approach" + continue thinking |
| chop_restart | nudge "discard it, reconsider from scratch" + continue thinking |

All three chop arms re-detect loops live (30-gram × 6 / window 1200) and re-chop up to 3×.

**Two runs**, identical except the chop-arm continuation sampler:
- **Phase 1 (α=5)**: `results/_chop_restart_probe/qwen3_chop_restart_phase1_n4.json`
- **α=2 control**: `results/_chop_restart_probe/qwen3_chop_restart_a2_n4.json`

**Scope**: the 14 "signal-before-loop" anchored tasks (A30 / `proxy_reasoning_depth.md`) — the
subset with a passing-config reference depth, so MAX_CONT=16384 is known-adequate. n=4 per arm.
HF token-by-token on CUDA. This is the **optimistic subset** (solvable, signal present); it does
not give a deployment-wide recovery rate (that is the deferred all-40 Phase 2).

---

## Results (pass-count per arm, α=5 ‖ α=2)

Totals:

| run | A_force | chop_only | chop_pivot | chop_restart |
|-----|---------|-----------|------------|--------------|
| **α=5** | 33/56 (9 tasks) | 41/56 (12 t) | 42/56 (12 t) | 37/56 (10 t) |
| **α=2** | 34/56 (9 t) | **20/56 (8 t)** | 24/56 (11 t) | 28/56 (10 t) |

Per task (pass/4, shown α5/α2; ★ = A_force scored 0):

| task | A_force | chop_only | chop_pivot | chop_restart |
|------|---------|-----------|------------|--------------|
| 417 ★ | 0/0 | 1/1 | 2/1 | 0/0 |
| 558 | 4/4 | 4/3 | 4/4 | 4/4 |
| 616 | 4/4 | 4/2 | 4/1 | 4/0 |
| 927 | 4/4 | 4/4 | 4/4 | 4/4 |
| 990 | 4/4 | 4/**0** | 4/1 | 4/4 |
| 1085 ★ | 0/0 | 4/1 | 4/3 | 4/2 |
| 1086 | 1/2 | 1/0 | 2/2 | 1/2 |
| 1125 ★ | 0/0 | 0/0 | 0/0 | 0/0 |
| 1126 | 4/4 | 4/4 | 4/2 | 4/4 |
| 1171 | 4/4 | 4/1 | 4/0 | 4/1 |
| 1178 ★ | 0/0 | 0/0 | 0/0 | 0/0 |
| 1224 | 4/4 | 4/4 | 4/1 | 4/2 |
| 1226 | 4/4 | 4/**0** | 4/4 | 4/4 |
| 1328 ★ | 0/0 | 3/0 | 2/1 | 0/1 |

(A_force re-ran in both passes as a consistency check: 33 vs 34 samples — matches within 1, the
small drift is executor flakiness, RuntimeError/Timeout on generated code.)

---

## Findings

### 1. The chop is a real rescue lever (recovers tasks force-extract cannot), at BOTH α
A_force fails completely (0/4) on 5 tasks: **417, 1085, 1125, 1178, 1328**. On **417, 1085, 1328**
some chop arm recovers — at **both** α=5 and α=2. These are tasks whose pre-loop trace contained no
extractable solution (A_force = 0), yet continuing to think after the chop reached one. Because it
holds at α=2 as well as α=5, the **chop itself** earns the recovery — not merely the α-switch.
(1125, 1178 never recover under any arm — capability ceiling for the chop, see §4.)

### 2. α=5's specific role is RE-ENTRY SUPPRESSION — and it is the reliability engine
Dropping α=5 → α=2, bare **chop_only collapses: 41→20 samples, 12→8 tasks — falling *below*
A_force** (9 tasks). The mechanism, quantified over the 168 chop-arm samples per run:

| | end=eos (closed think) | end=loop (re-entry death) | re-chopped (chops>0) | never closed `</think>` |
|---|---|---|---|---|
| α=5 | 159 | **0** | 4 | 6 |
| α=2 | 92 | **54** | 81 | 67 |

At α=2 the chopped context re-derives its loop: the detector re-fires 81× (vs 4×), and 54 samples
re-loop through all 3 chops and die with no code. At α=5 there are **zero** loop-deaths — the
flatter distribution prevents re-entry. Cleanest single proof: **task 990, chop_only 4/4 at α5 →
0/4 at α2** (same chop point, only the sampler changed; all four α2 samples re-looped to death).

### 3. The nudge flips from REDUNDANT (α=5) to LOAD-BEARING (α=2) — but its direction is unreliable
- At α=5, chop_only ≈ chop_pivot (both 12 tasks, 41/42 samples) — the nudge adds nothing, because
  α=5 already suppresses re-entry.
- At α=2, bare chop_only drops to 8 tasks, but the nudge recovers **990, 1086, 1226, 1328** that
  bare chop loses (chop_pivot 11 t, chop_restart 10 t). So a nudge **partially substitutes** for
  α=5's re-entry suppression.
- But which nudge wins is task-dependent and reverses: restart saves 990 (0→4/4) yet kills 417
  (1→0); pivot best on 1085; both hurt 616. **No deployable "best nudge."** Consistent with
  2506.10979 (nudge-to-reconsider effects are real but unreliable).

### 4. Prevention still numerically dominates rescue
α=5 run **from the start** (full-252 recovery sweep, vLLM, n=10) solves the A_force-fail tasks at
higher rates than any rescue here: **417 = 5/10, 1085 = 9/10, 1328 = 10/10, 1125 = 7/10,
1178 = 2/10**. Two implications:
- The looped-prefix seed can **handicap**: on **1125**, α=5-from-start gets 7/10 but chop (both α)
  gets 0/4 — seeding from the degenerate trace prevented a recovery a clean start achieves.
- Prevention (just run α=5) beats rescue on this subset. Rescue is only relevant if one is
  committed to α=2 for its confident-case pass@1 and willing to accept flaky recovery on the
  looping subset.

### 5. Cross-check — the chop does NOT exceed the capability ceiling
Of the 13 tasks unsolvable across all 6 original configs (`truncated_solvability.md`:
117, 280, 326, 370, 454, 455, 512, 661, 962, 1122, 1175, 1223, 1368), the α3/α4/α5 recovery arms
solved **0**; only 280 ever cracked, and only via **pless T2.0** (1/10) — temperature, not α. The
chop arms continue with α — i.e. a handicapped α-from-start — so a chop probe on those tasks is a
near-certain 0. The dedicated unsolvable-task probe was therefore **dropped** (the alpha sweep
already answered it). This confirms chop rescues *within* capability, it does not manufacture
solutions for tasks the model cannot solve.

---

## Deployable hierarchy (established by this experiment)

**α=5 from start (prevention) > chop + α=5 (rescue) > chop + nudge + α=2 (flaky rescue) >
A_force ≈ chop-only + α=2 > do nothing.**

Answer to "can we adaptively escape while staying on α=2?": **Yes, but only with a nudge, and
flakily.** Bare chop on α=2 fails (re-entry). chop+nudge on α=2 recovers more *distinct tasks*
than A_force (11 vs 9) but at *lower per-draw reliability* (24 vs 34 samples) — it trades precision
for coverage. The dependable rescue is **chop + α=5**; the most reliable option overall is
**prevention (α=5 from start)**.

---

## Caveats
- **n=4, 14 optimistic tasks.** Per-task rates are directional, not precise; the nudge
  direction-flip (restart 4/4 on 990 vs 0/4 on 417) is suggestive, not settled.
- **Subset bias.** These 14 are signal-before-loop + solvable; they overstate recovery vs the full
  40 truncated tasks (let alone all 252). A real recovery *rate* needs the deferred Phase 2.
- **n/backend not fully paired** for the prevention comparison: chop runs are n=4 HF; the α5-from-
  start rates are n=10 vLLM. The qualitative signals (seed handicap on 1125; α5 > α2 reliability)
  are too large to be artifacts, but the exact rates are not directly comparable.

## Artifacts
- `results/_chop_restart_probe/qwen3_chop_restart_phase1_n4.json` (α=5) + `.log`
- `results/_chop_restart_probe/qwen3_chop_restart_a2_n4.json` (α=2 control) + `.log`
- Prevention/unsolvable cross-check: `results/pless_recovery_full252/.../metrics/*.json`
- Script: `scripts/chop_restart_alpha_compare.py`; launcher: `run_chop_restart_apps_qwen3.sh`
