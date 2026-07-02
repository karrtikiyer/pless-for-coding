# Degenerate reasoning loops in hyperparameter-free (pless) decoding: prevention vs. rescue

**Audience**: AI researchers. **Model/data**: Qwen3-8B, APPS ATCODER-interview.
**Window**: 2026-06-13 → 2026-06-25. **All numbers pulled live from committed result JSONs**
(paths in §Artifacts); the "no extractable code" figure is computed as the fraction of samples
whose `per_task.extraction_success == False` — a truncation proxy (a looped sample emits no
`</think>` → no code → no extraction).

## Background
**pless** is a hyperparameter-free sampler: keep tokens above threshold `p = Σpᵢ²` (collision
entropy), renormalize, sample. On hard reasoning tasks Qwen3-8B under pless enters a
**verbatim thinking-phase loop**, never emits `</think>`, hits the 32768-token cap, and produces
no code → auto-fail. This report covers two questions: (A) can we *prevent* the loop, and (B) if
we insist on the low-temperature operating point, can we *rescue* a loop once detected?

---

## A. Prevention: the loop is a peakedness effect — hot decoding removes it

pless truncation is **monotonic in temperature** (pless, N=252, n=10):

| sampler | pass@1 | no extractable code (≈truncation) |
|---|---|---|
| pless @T0.6 | 0.615 | **16.2%** |
| pless @T1.0 | 0.625 | 10.5% |
| pless @T2.0 | 0.694 | **0.2%** |

Lowering temperature *sharpens* the distribution before the Σpᵢ² threshold is applied → more mass
concentrated → more looping; raising it flattens the distribution → looping vanishes. The loop is
a **peakedness phenomenon**, not a property of the task.

This is pless-specific. At the same low temperature, standard truncation samplers are healthy
(T0.6, N=252, n=10):

| @T0.6 | pass@1 | no extractable code |
|---|---|---|
| pless | 0.615 | 16.2% |
| pless_norm | 0.619 | 15.3% |
| top-p 0.95 | 0.695 | 1.0% |
| top-k 20 | 0.698 | 0.6% |

**Takeaway A**: never run pless at low temperature on long CoT — it is the worst operating point.
Decode hot (T≥1.5, or the α-generalization α≥5, which flattens the survivor set similarly) and the
truncation-driven pass@1 gap to top-p/top-k largely closes. Prevention is cheap and needs no
detector.

---

## B. Rescue: if you must sit at the peaked operating point, can you escape a detected loop?

**Design (controlled "fair test").** Take each task's *real saved looped trace*, chop it at the
loop onset, and run 4 arms on the **identical chopped prefix** — only the post-chop action differs:

- **A_force** — force `</think>` + code fence, extract the existing solution (prior best rescue).
- **chop_only** — continue thinking, no nudge.
- **chop_pivot / chop_restart** — continue thinking after a steering nudge (*redirect* vs
  *discard & restart*).

Run twice — continuation at **α=5** vs an **α=2 control** (identical otherwise) — to attribute the
effect to the chop vs. the sampler switch. 14 anchored tasks (the subset with a known-adequate
continuation budget), n=4, HF token-by-token, GPU. Live loop re-detection (30-gram×6) + re-chop ≤3.

**Results (passing samples / 56; distinct tasks recovered):**

| | A_force | chop_only | chop_pivot | chop_restart |
|---|---|---|---|---|
| **α=5** | 33 (9t) | 41 (12t) | 42 (12t) | 37 (10t) |
| **α=2** | 34 (9t) | **20 (8t)** | 24 (11t) | 28 (10t) |

### Finding 1 — the chop is a genuine rescue lever (not just extraction)
A_force scores 0/4 on 5 tasks. On **417, 1085, 1328**, chop-continue recovers — at **both** α=5 and
α=2. These solutions were absent from the pre-loop trace (nothing for A_force to extract), so
continuing to think *produced* them. Holding at α=2 shows the **chop itself** earns the recovery,
not the sampler switch.

### Finding 2 — α=5's mechanistic role is re-entry suppression, and it is the reliability engine
Over 168 chop-arm samples per run:

| | closed `</think>` | re-entry deaths (`end=loop`) | re-chops fired |
|---|---|---|---|
| α=5 | 162 | **0** | 4 |
| α=2 | 101 | **54** | 81 |

Bare chop collapses α5→α2: **41→20 passing samples, 12→8 tasks** (below A_force's 9). At α=2 the
chopped context re-derives its loop; α=5's flatter distribution prevents it. Clean proof —
**task 990: chop_only 4/4 @α5 → 0/4 @α2**, same chop point, only the sampler changed (all four α=2
samples re-looped to death).

### Finding 3 — a nudge is redundant under α=5, load-bearing under α=2, and unreliable in direction
At α=5, chop_only ≈ chop_pivot (nudge invisible — re-entry already suppressed). At α=2 the nudge
recovers 4 tasks bare chop loses (chop_pivot 11t, chop_restart 10t vs chop_only 8t). But *which*
nudge helps reverses per task: "restart" saves 990 (0→4/4) yet kills 417; "pivot" wins on 1085.
**No deployable best-nudge** — consistent with Yang et al. 2025 (arXiv:2506.10979): reasoning models
identify but fail to *reliably* recover from bad thoughts (nudge-to-reconsider is weak / inverse
scaling). Note their setup leaves the bad thought in context; our chop removes it, which is why a
nudge helps at all here.

### Finding 4 — prevention still numerically dominates rescue
α=5 run *from the start* (full-252 sweep, n=10) solves the A_force-fail tasks at higher rates than
any rescue: **417 = 5/10, 1085 = 9/10, 1328 = 10/10**. And the looped seed can *handicap*: on
**1125**, α=5-from-start gets 7/10 but chop (both α) gets 0/4 — a clean start beats rescuing a
degenerate trace.

### Finding 5 — chop does not exceed the capability ceiling
Of 13 tasks unsolvable across all original configs, the α3/α4/α5 arms solve **0**; only task 280
ever cracks, via temperature-2.0 (1/10), not α. Chop rescues *within* capability; it does not
invent solutions the model cannot reach.

---

## Deployable hierarchy (established by these experiments)

**α=5-from-start (prevention) > chop + α=5 (reliable rescue) > chop + nudge + α=2 (flaky rescue) >
A_force ≈ bare-chop + α=2 > do nothing.**

If forced to stay at the peaked operating point, chop+nudge recovers more *distinct tasks* than
force-extract (11 vs 9) but at lower *per-draw* reliability (24 vs 34 passing samples) — precision
traded for coverage. The dependable move is to switch to α=5 on detection; the cheapest is to
prevent by decoding hot from the start.

## One-line summary
For pless loop failures, **the chop removes the loop but α=5 is what stops it re-forming**
(0 vs 54 re-entry deaths), and **prevention (decode hot from the start) still beats every rescue**.

## Caveats (stated up front)
- Rescue results: **n=4, 14 tasks**, and those 14 are an *optimistic* subset (signal-before-loop +
  solvable) — directional, **not** a deployment-wide recovery rate. Honest rate needs an all-40
  Phase 2 (not yet run).
- Rescue runs are n=4 (HF); prevention/α-sweep rates are n=10 (vLLM) — qualitative signals (α=5
  re-entry suppression, seed handicap) are far too large to be artifacts, but exact rates aren't
  paired.
- One model, one dataset (Qwen3-8B, ATCODER-interview). The temperature-monotonicity result is
  N=252; the rescue result is the 14-task subset.

## Artifacts (reproducibility)
- Temperature sweep: `results/decoders_t0.6/.../metrics/*_metrics.json`,
  `results/pless_cot_efficiency_vllm/.../ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json`,
  `results/pless_recovery_full252/.../metrics/pless_think_t2.0_t2.0_metrics.json`
- Rescue: `results/_chop_restart_probe/qwen3_chop_restart_phase1_n4.json` (α=5),
  `qwen3_chop_restart_a2_n4.json` (α=2); full writeup `docs/chop_restart_alpha_findings_qwen3.md`
- Prevention rates & unsolvable cross-check: `results/pless_recovery_full252/.../metrics/*.json`
- Code: `scripts/chop_restart_alpha_compare.py`, `run_chop_restart_apps_qwen3.sh`
