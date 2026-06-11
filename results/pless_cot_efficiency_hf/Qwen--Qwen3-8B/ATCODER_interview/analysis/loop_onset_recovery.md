# Loop-Onset Recovery Analysis: Can Pless Escape the Loop on Solvable Tasks?

## Method

For each of the 27 solvable pless-truncated tasks (tasks where at least one other config
got ≥1 correct sample), we:

1. Find the earliest loop_pos across all truncated samples for that task
2. Find the per-task best config (highest n_correct) and its avg thinking chars at `</think>`
   (reference depth = how deep passing configs reasoned before committing)
3. Classify the pless thinking at loop_pos:
   - Is the loop starting before or after the reference commit depth?
   - Does the snippet at loop_pos show concrete code/algorithm or still-searching?

---

## Classification (27 solvable tasks)

### R1 — loop>ref: loop started AFTER reference commit depth (4 tasks)
Pless reasoned past the depth where passing configs committed, then looped.
Forced `</think>` at loop_pos most likely to recover correct code.

| task | best_cfg | ref_depth | loop_pos | loop% | note |
|------|----------|-----------|----------|-------|------|
| 558 | pless_norm_think_t1.0_t1.0 | 30,412 | 31,800 | 30% | Concrete Python code in snippet |
| 739 | temp_k20_think_t1.0_t1.0 | 47,786 | 61,000 | 63% | Code being written at loop onset |
| 1125 | temp_k20_think_t1.0_t1.0 | 46,287 | 48,800 | 54% | Past ref depth, reasoning still active |
| 1328 | temp_p0.95_think_t1.0_t1.0 | 38,442 | 52,700 | 37% | DP reasoning past ref depth |

### R2 — REF>LOOP but concrete solution at loop onset (5 tasks)
Loop started before the reference depth, BUT the snippet at loop_pos shows concrete
code or a structured algorithm — the model may have enough to generate code if escaped.

| task | best_cfg | ref_depth | loop_pos | loop% | note |
|------|----------|-----------|----------|-------|------|
| 1087 | temp_k20_think_t1.0_t1.0 | 38,829 | 30,700 | 26% | "code should be correct. Now, the code:" |
| 1126 | pless_norm_think_t1.0_t1.0 | 36,242 | 25,300 | 24% | Actual Python cycle-detection code visible |
| 1224 | temp_k20_think_t1.0_t1.0 | 22,734 | 10,900 | 12% | Concrete 3^a+5^b algorithm structured |
| 1226 | pless_norm_think_t1.0_t1.0 | 18,189 | 8,700 | 7% | Actual fact/inv_fact Python code at loop onset |
| 1369 | temp_k20_think_t1.0_t1.0 | 36,510 | 31,900 | 24% | "code is correct. Now, code:" — solution referenced |

### S — REF>LOOP, still searching at loop onset (12 tasks)
Loop started before reference depth AND no concrete solution/code at loop onset.
Injection at loop_pos would produce incomplete/wrong code.

Tasks: 417, 616, 793, 930, 1037, 1085, 1171, 1178, 1277, 1329, 1374, 1426

### L — Pure nonsense loop from very early (6 tasks)
Loop started at 10–27% of total thinking with no reasoning content — pure repetition
of fragments. Nothing to recover.

Tasks: 711, 927, 990, 1086, 1090, 1373

---

## Summary

| Category | N | Recovery outlook |
|----------|---|-----------------|
| R1: loop>ref, solution at onset | 4 | Strong — forced `</think>` at loop_pos likely recovers |
| R2: REF>LOOP, code at onset | 5 | Possible — reasoning may be sufficient despite being pre-ref-depth |
| S: still searching at onset | 12 | Unlikely — reasoning incomplete, injection gives wrong code |
| L: pure loop from early | 6 | None — no reasoning to recover |
| **Total** | **27** | **9/27 (33%) are recovery candidates** |

---

## Implications

**For A27 (budget forcing):** A hard token cap forces `</think>` at a fixed depth, not at
loop onset. For the 6 L-tasks (loop at 10–27%), a budget cap at 32k tokens is already
too late — they looped thousands of tokens ago. A cap at say 16k would catch them but
would also cut off the 12 S-tasks mid-search. Budget forcing is a blunt instrument.

**For A28 (loop-onset detector):** The 9 R1+R2 tasks have code/solution visible at
loop_pos. A detector that fires at loop onset — detecting verbatim repetition or entropy
degeneration — and injects `</think>` there would target exactly these cases. It would
NOT help the 12 S or 6 L tasks (no solution to recover), but it avoids wasting tokens.

**For the 18 S+L tasks:** The problem is upstream — pless never formed a solution before
looping. This is a harder failure mode: the sampling itself is failing to explore the
right reasoning paths, not just failing to terminate after finding one.

---

## EMPIRICAL RESULTS — all-27 forced-</think> screen (2026-06-11, n=1, MPS)

Ran the forced-</think> intervention on all 27 solvable tasks (`scripts/forced_think_all27.py`):
cut each pless truncated trace at loop onset, force `</think>` + ```python, generate
with pless (n=1, near-deterministic), execute against the real APPS tests.

**RECOVERED: 11/27** (or ~11/25 ≈ 44% excluding 2 generation exceptions).

Per category (vs the a-priori prediction above):

| category | recovered | note |
|----------|-----------|------|
| R2 (concrete code at loop onset) | **5/5** | prediction HOLDS — reliable positive signal |
| R1 (loop>ref depth)              | **0/4** | prediction WRONG (had ranked these "strongest") |
| S (still searching)              | 2/12    | ≈ right (mostly no recovery) |
| L (pure loop early)              | 4/6     | prediction WRONG (predicted no recovery) |

Recovered: 616, 927, 990, 1087, 1090, 1126, 1171, 1224, 1226, 1369, 1373.
Failure modes (16): 11 Failed (wrong code), 1 RuntimeError, 2 ParsingError (739, 1125 —
possible prose-after-fence undercount), 2 EXC:RuntimeError (711, 1277 — MPS errors mid
generation, INCONCLUSIVE, re-run on GPU).

**Corrected takeaway**: the only reliable predictor of recovery is whether CONCRETE CODE
exists in the pre-loop reasoning (R2: 5/5). The loop-timing (R1) and loop-onset-snippet
(S/L) heuristics do NOT track recovery — R1 (loop after ref depth) recovered nothing, and
L (degenerate snippet) recovered 4/6 because real reasoning preceded the degenerate tail.
A loop-onset detector (A28) should fire on the loop, but a "is there a solution to recover"
gate should look for concrete code in the full pre-loop trace, not the onset snippet.

**Caveats**: n=1 (pless near-deterministic, and NOT bit-stable across MPS process runs —
558 gave ParsingError in one run, RuntimeError here). The GPU n=10-with-sampling pass
(A31-adjacent) is needed for real per-task recovery rates.
