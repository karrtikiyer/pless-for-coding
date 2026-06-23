# DeepSeek-R1-Distill-Llama-8B loop-force — findings

**Status:** FULL 252 DONE (n=30, **k=8**, window=3000), `results/loop_forcethink_deepseek_w3000_k8/`.
All numbers reproduced by `scripts/deepseek_full252_analysis.py` (matched comparison + the
over-truncation token check) and `scripts/completion_breakdown.py` / `scripts/detector_deepseek_nk_grid.py`.

---

## 0. FULL-252 VERDICT (supersedes the 25-smoke verdict in §2–3 below)

On the full 252 — where solvable tasks dominate, unlike the dead-end-heavy 25-slice — loop-force
is the method's **best showcase**, not a failure-relabeler. Matched vs the no-force baseline:

| config | trunc% | compl%(+code) | closed-no-code% | cond | pass@1 | pass@5 | pass@10 |
|---|---|---|---|---|---|---|---|
| pless baseline | 64.9 | 35.1 | 0.0 | 0.495 | 0.174 | 0.368 | 0.464 |
| **pless loop-force k8** | **4.1** | 68.3 | 27.6 | 0.327 | **0.223** | 0.451 | **0.528** |
| pnorm baseline | 63.8 | 36.2 | 0.0 | 0.450 | 0.163 | 0.357 | 0.444 |
| **pnorm loop-force k8** | **4.0** | 67.8 | 28.2 | 0.347 | **0.235** | 0.473 | **0.556** |

- **Truncation 64.9→4.1% / 63.8→4.0%**; **pass@1 +4.9 / +7.2pp**; **pass@10 (coverage) +6.4 / +11.2pp**.
- **Not over-truncating** (the token-length check): forced/cut traces median **6,094** / mean 6,527
  sit squarely in the *productive* reasoning band (temp passed+completed: median 6,746 / mean 7,215,
  p25–p75 4,413–9,564) and are **longer** than the model's own natural closes (median 5,026). The
  ~5–6K overall mean is DeepSeek's intrinsic short-close behaviour, not the detector clipping good
  reasoning. Decisive corroboration: pass@1 **and** pass@10 both rose — over-truncation would drop
  coverage. (k=8 was tuned for ~0% false-positive on productive reasoning; see §4.)
- **Cross-model law:** loop-force *improves* DeepSeek coverage (pass@10 ↑) but *hurt* Qwen3's
  (−1–2pp). The payoff **scales with the baseline truncation rate** — DeepSeek wastes 64.9% to
  truncation (huge headroom; gains swamp the false-positive cost), Qwen3 only 14.5% (FP cost
  dominates). DeepSeek is therefore the stronger showcase for the method.
- **The residual cost** is conditional-correctness (0.495→0.327) + ~28% closed-no-code — the
  *dead-end loops* (forcing a stuck loop with no solution yields no/wrong code), not over-cutting.
  Net is strongly positive.

The §2–3 25-task smoke below was the **worst-case subset** (8/25 globally unsolvable, baseline
pass@1 0.10) and is kept as the conservative bound; the full-252 numbers above are the verdict.

---

## 1. The model-aware `</think>` fix (E8) works end-to-end

The smoke ran to completion with no crash — confirming `resolve_think_end_id` correctly
resolves DeepSeek's `</think>`=128014 (vs Qwen3's 151668; DeepSeek vocab=128000, so the old
hardcoded id would have `IndexError`-ed at `row[151668]`). Truncation collapsed: pless
74.8→1.2%, pless_norm 75.2→0.4% (matched on the same 25 tasks). The detector fired on ~46–50%
of samples. **The mechanism is validated; that was the smoke's job.**

## 2. But on these 25 hard tasks, loop-force bought no accuracy — it relabeled the failure

Matched on the same 25 task_ids (DeepSeek baseline vs w3000 loop-force):

| arm | trunc% | cond-corr | pass@1 | pass@10 |
|---|---|---|---|---|
| pless baseline (no-force) | 74.8 | 0.397 | 0.100 | 0.200 |
| pless loop-force w3000 | 1.2 | **0.081** | 0.080 | 0.200 |
| pnorm baseline (no-force) | 75.2 | 0.339 | 0.084 | 0.160 |
| pnorm loop-force w3000 | 0.4 | **0.116** | 0.116 | 0.200 |

pass@1 flat (noise), pass@10 flat; conditional-correctness **collapsed** (0.40→0.08). Forcing
`</think>` on a stuck loop converts a truncated-fail into a completed-but-wrong (or no-code) fail.

## 3. The clean diagnostic — `closed-but-no-code%` (`scripts/completion_breakdown.py`)

`cot_efficiency` splits samples three ways (they do NOT sum as compl+trunc): truncated (no
`</think>`) + closed-no-code (`</think>` but no extractable code) + completed (`</think>` + code).
The **closed-no-code** rate is the diagnostic for whether the model had a solution when it stopped:

| config (force-`</think>`) | trunc% | closed-no-code% | completed% |
|---|---|---|---|
| Qwen3 pless w1200 (full 252) | 1.5 | **0.4** | 98.1 |
| Qwen3 pless w1200 (same 25) | 1.6 | **0.4** | 98.0 |
| Qwen3 pless BASE (full 252) | 14.5 | 0.0 | 85.5 |
| DeepSeek pless w3000 (25 smoke) | 1.2 | **40.8** | 58.0 |
| DeepSeek pnorm w3000 (25 smoke) | 0.4 | **33.6** | 66.0 |
| DeepSeek pless BASE (full 252) | 64.9 | 0.0 | 35.1 |

Same tasks, same intervention, only the model differs: **Qwen3 produces extractable code 98% of
the time when forced; DeepSeek only 58–66%.** This is the project's "solution-existence is the
disease, not the loop" finding made quantitative — Qwen3's loops are recoverable overthinking;
DeepSeek's (on these hard tasks) are dead-ends with nothing to write.

## 4. Detector re-tune for DeepSeek: k=6 → k=8 (`scripts/detector_deepseek_nk_grid.py`)

n=30/k=6 was carried over from Qwen3 without a DeepSeek-specific check. The grid (window=3000):

| k | FP% (good reasoning fired on) | catch% (loops) | median fire pos |
|---|---|---|---|
| 6 (carried over) | **1.5** | 98.5 | 5,430 |
| **8** | **0.0** | 93.5 | 5,830 |
| 10 | 0.0 | 87.0 | 5,830 |
| 12 | 0.0 | 82.0 | 6,030 |

**Use k=8 for the DeepSeek full run** — it zeroes the false-positive on productive reasoning at
only ~5pp catch cost. Note the fire position barely moves with k (5.4K→6.4K even at k=16):
DeepSeek's loops onset early (~5K) and are dense, so the ~6K think length is intrinsic, not an
over-aggression artifact (forced and natural-close traces were the same ~5.4K length; productive
passing reasoning on these tasks is ~7K median, p25–p75 4.4K–9.4K).

## 5. What's pending / caveats

- **The 25-slice is the worst case** (8/25 unsolvable, baseline pass@1 0.10). On the **full 252**,
  DeepSeek pless cond-correctness is 0.4955 — the easier ~227 tasks have real solutions, where
  forcing should produce passing code (like Qwen3). The full run at **k=8** is the real verdict.
- Run: `MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B TOKENIZER=… LOOP_N=30 LOOP_K=8
  LOOP_WINDOW=3000 RESULTS_DIR=results/loop_forcethink_deepseek_w3000_k8 …
  ./run_loop_forcethink_apps_qwen3.sh` (fresh dir — k=8 ≠ the k=6 smoke, so no resume).
- Even on the full set, expect loop-force to keep relabeling more than rescuing where loops are
  dead-ends; the `closed-no-code%` column is the metric to watch.
