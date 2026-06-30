# Findings — p-less on a reasoning model (DeepSeek-R1-Distill): α-tuning fixes its looping but does not beat standard decoding

**Date:** 2026-06-30 · **Status:** DeepSeek complete; **Qwen cross-method comparison PENDING (the decider)**
· **Data:** `results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/`
(α=2 + standard decoders) and `results/pless_recovery_full252/.../ATCODER_interview/` (α=3/4/5)
· **Diversity:** `scripts/compute_diversity_deepseek.py` → `.../diversity_all9.json`

## TL;DR

On **DeepSeek-R1-Distill-Llama-8B**, ATCODER-interview (252 tasks × 10 samples, 32768-token
budget), comparing all 9 decoding configs on identical data:

- The hyperparameter-free **pless (α=2) and pless_norm loop catastrophically** — ~64–65 %
  of samples truncate (never close `</think>`), pass@1 craters to 0.17.
- **Raising the Rényi-α exponent fixes the looping** (truncation 64.9 % → 4.5 % at α=4 →
  1.2 % at α=5) and recovers pass@1 to ~0.30–0.32 — but this only reaches **parity-minus**
  with plain temperature decoding.
- **`temp t1.0 top_p0.95` is ≥ the best pless-α arm on every axis at once** — pass@1,
  pass@5, pass@10, cov@0.3, cov@0.5, **and both diversity metrics (struct_div, cb_div)** —
  while never looping (0.1 % truncation).

So on this reasoning model, **p-less offers no advantage over standard decoding, including
on diversity (its claimed strength), and its hyperparameter-free default is the worst of
the nine.** The earlier "+14pp pass@1" framing was pless *recovering from its own
pathology*, not beating good decoding.

## The full table (same 252×10, budget 32768)

| method | trunc% | pass@1 | pass@5 | pass@10 | cov@0.3 | cov@0.5 | struct_div | cb_div |
|--------|-------|--------|--------|---------|---------|---------|-----------|--------|
| **temp t1.0 top_p0.95** | 0.1 | **0.328** | **0.559** | **0.635** | **46.4** | **37.3** | **0.576** | **0.619** |
| temp t1.0 topk20 | 0.0 | 0.306 | 0.539 | 0.619 | 42.9 | 32.1 | 0.566 | 0.619 |
| temp t0.6 | 3.2 | 0.318 | 0.549 | 0.615 | 45.6 | 32.5 | 0.538 | 0.582 |
| temp t0.6 p0.95 k20 | 11.4 | 0.315 | 0.544 | 0.623 | 44.4 | 33.7 | 0.525 | 0.570 |
| pless_alpha a=4 | 4.5 | 0.315 | 0.536 | 0.591 | 44.8 | 33.3 | 0.527 | 0.583 |
| pless_alpha a=5 | 1.2 | 0.295 | 0.500 | 0.560 | 41.3 | 29.8 | 0.533 | 0.578 |
| pless_alpha a=3 | 19.4 | 0.299 | 0.527 | 0.599 | 43.3 | 31.7 | 0.508 | 0.554 |
| pless (a=2) | 64.9 | 0.174 | 0.368 | 0.464 | 24.2 | 15.5 | 0.483 | 0.546 |
| pless_norm | 63.8 | 0.163 | 0.357 | 0.444 | 23.8 | 13.1 | 0.479 | 0.534 |

Best pless-α is **a=4** (pass@1 0.315, truncation 4.5 %); **a=5 slightly over-loosens**
(pass@1 0.295, pass@10 0.560) — the mild precision-cost-at-very-high-α also seen on Qwen.

## Reading it honestly (margin calibration)

- The pass@1 gap top_p vs pless-a4 (0.328 vs 0.315) is ~1.4 SE — **marginal in isolation**.
  Same for pass@10.
- What makes the verdict robust is **consistency**: top_p0.95 is ≥ the best pless-α on
  **all six quality+diversity axes simultaneously**, which is very unlikely if they were
  truly equal. So the defensible claim is *"pless shows no advantage,"* not *"top_p
  crushes pless."*
- **Undeniable** (large, not marginal): pless α=2 / pless_norm are catastrophically worse
  (65 % truncation, pass@1 0.17, lowest diversity) — the hyperparameter-free configs are
  the worst here.

## Why the diversity result matters most

p-less's core value proposition is **diversity without a tuned temperature**. On DeepSeek
that proposition fails twice over: (1) the hyperparameter-free default is the *least*
diverse (it barely completes, so few correct samples to vary over); (2) even the tuned
α arms produce *less* diverse correct solutions than vanilla `top_p0.95`
(struct_div 0.53 vs 0.58; cb_div 0.58 vs 0.62). Standard sampling is both more accurate
*and* more diverse here.

## Relationship to the detection thread

This complements the detection negative (`docs/pilot1_findings_hidden_state_detection.md`):
neither **detecting** loops early (hidden-state precursor doesn't generalize) nor
**preventing** them with p-less buys anything over just decoding well on reasoning-model
code. The loop pathology is largely **p-less-specific** — standard decoders barely loop
on this model (top_p0.95: 0.1 % truncation) — so "fix p-less's looping" is solving a
problem that good decoding doesn't have.

## Caveats / scope

1. **Single model, single dataset** (DeepSeek-R1-Distill, ATCODER-interview). Does NOT yet
   generalize to instruct/coder models or other benchmarks.
2. **The Qwen cross-method comparison is the decider and is PENDING.** The original p-less
   value (MBPP/HumanEval on instruct/coder models) was never compared head-to-head against
   standard decoders *with diversity* on the α arms. If Qwen3-8B shows the same parity-minus
   pattern → the "p-less on reasoning models" verdict is general; if pless-α *wins* on Qwen
   → this DeepSeek result is model-specific. **Do not generalize beyond DeepSeek until that
   table exists.**
3. Diversity computed without re-execution, reusing the existing pass_results + the
   canonical `add_structural_diversity` / `add_self_codebleu`; validated against α=5's
   pipeline-computed struct_div (0.524 vs 0.533, Δ 0.009).

## Reproduce

```
# cross-method pass@k/cov: per-method metrics JSONs (already evaluated)
# truncation%: analysis/cot_efficiency_apps.csv (both dirs)
# diversity (struct_div + cb_div), no re-execution:
HF_HUB_OFFLINE=1 uv run python scripts/compute_diversity_deepseek.py
```
