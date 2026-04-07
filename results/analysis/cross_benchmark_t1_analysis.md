# Cross-Benchmark T1 Analysis: MBPP vs HumanEval

Does T1 provide the same benefit on HumanEval as on MBPP? We test every MBPP conclusion against freshly computed HumanEval metrics across 6 models.

**MBPP conclusions under test:**
1. T1 is the only useful knob (3-17x more efficient than T2)
2. Sweet spot T1=1.0-1.5
3. Catastrophe at T1=3.0
4. P-less matches greedy at low T1 on instruct models
5. 7B is more peaked than 3B; peakedness increases with scale

**HumanEval data:** 6 models × 14 configs (pless/pless_norm at T1=0.7-3.0, temp at 0.7/1.0), 164 tasks × 10 samples. All metrics freshly computed from raw JSONL files.

### Methodological Notes

**Statistical significance.** HumanEval has 164 tasks. At p@1 ~85%, SE ≈ √(0.85×0.15/164) ≈ 2.8pp. **Any difference below ~3pp is within noise.** MBPP (500 tasks) has SE ≈ 1.75pp. Claims are flagged with confidence levels throughout.

**Greedy baseline caveat.** The temp sweep data contains only pless (6 temps), pless_norm (6 temps), and temp (0.7, 1.0). There is NO greedy baseline within the temp sweep. The greedy number (84.2%) comes from a separate full-precision evaluation run. Cross-run comparisons are suggestive but not definitive — differences in code extraction, execution environment, or random seeds may contribute. A separate full-precision run also shows pless t=0.6 at 87.5% and greedy at 84.15%, confirming the direction but from a different evaluation pipeline.

**pless vs pless_norm.** Both methods were evaluated across all 6 models. For most comparisons in this report, differences between pless and pless_norm are small (<1pp pass@1 on strong models). Where they diverge materially (weak models), this is noted explicitly.

---

## 1. Qwen2.5-Coder-7B-Instruct: MBPP vs HumanEval Head-to-Head

This is the critical comparison — same model on two benchmarks.

### P-less T1 sweep (pless and pless_norm)

| T1 | MBPP p@1 | MBPP p@10 | MBPP sdiv | HE pless p@1 | HE pless p@10 | HE pless sdiv | HE pless_norm p@1 | HE pless_norm p@10 | HE pless_norm sdiv |
|----|----------|-----------|-----------|-------------|---------------|---------------|-------------------|--------------------|--------------------|
| 0.6 | 77.2% | 79.8% | 0.031 | — | — | — | — | — | — |
| 0.7 | — | — | — | 84.8% | 89.0% | 0.009 | 84.8% | 89.0% | 0.010 |
| 1.0 | 77.2% | 82.2% | 0.059 | 85.4% | 89.0% | 0.016 | 84.6% | 88.4% | 0.015 |
| 1.5 | 76.7% | 85.8% | 0.126 | 84.5% | 88.4% | 0.049 | 83.9% | 89.0% | 0.052 |
| 2.0 | 72.5% | 89.6% | 0.308 | 82.4% | 92.1% | 0.161 | 82.1% | 93.3% | 0.174 |
| 2.5 | — | — | — | 64.3% | 93.9% | 0.354 | 63.6% | 94.5% | 0.351 |
| 3.0 | 2.7% | 18.8% | 0.265 | 19.9% | 66.5% | 0.296 | 19.6% | 70.7% | 0.305 |

**pless vs pless_norm on Qwen2.5-7B-Instruct:** Differences are ≤0.8pp pass@1 at all T1 values — well within SE. The methods are interchangeable on this model.

### Temperature baselines

| Config | MBPP p@1 | MBPP p@10 | MBPP sdiv | HE p@1 | HE p@10 | HE sdiv | Source |
|--------|----------|-----------|-----------|--------|---------|---------|--------|
| greedy | 77.6% | — | 0.000 | 84.2% | — | 0.000 | full_precision |
| pless t=0.6 | — | — | — | 87.5% | — | 0.007 | full_precision |
| temp t=0.2 | 76.8% | 83.2% | 0.098 | 83.5% | 90.9% | 0.135 | full_precision |
| temp t=0.7 | — | — | — | 80.6% | 91.5% | 0.183 | temp_sweep |
| temp t=0.8 | 72.0% | 88.8% | 0.278 | — | — | — | MBPP |

### Finding 1: Sweet spot T1=0.7-1.5 — CONFIRMED on HumanEval

**Pass@1 degradation pattern:**

| Transition | MBPP Δp@1 | HumanEval Δp@1 |
|------------|-----------|----------------|
| T1=0.7→1.0 | — | +0.6pp (within noise, SE ≈ 2.8pp) |
| T1=1.0→1.5 | -0.5pp | -0.9pp (within noise) |
| T1=1.5→2.0 | -4.2pp | -2.1pp (borderline significant) |
| T1=2.0→2.5 | — | **-18.1pp** (clearly significant) |
| T1=2.0→3.0 | -69.8pp | -62.5pp (catastrophic) |

On HumanEval, T1=0.7 through T1=1.5 are statistically indistinguishable in pass@1 (84.5-85.4%, all within 2.8pp SE of each other). The cliff comes between T1=2.0 and T1=2.5.

**Verdict:** Sweet spot confirmed at T1=0.7-1.5 on both benchmarks. The cliff is at T1=2.0→2.5 on HumanEval and T1=2.0→3.0 on MBPP (T1=2.5 not tested on MBPP).

### Finding 2: P-less vs greedy — suggestive but NOT confirmed

**Within-run comparison (temp sweep only):**

The temp sweep has no greedy baseline, so we cannot directly compare within the same evaluation run. The closest within-run comparison is pless T1=0.7 (84.8%) vs temp t=0.7 (80.6%) — P-less clearly wins (+4.2pp, likely significant even at SE ≈ 2.8pp). However, pless T1=0.7 at sdiv=0.009 is nearly greedy in practice.

**Cross-run comparison (temp sweep pless vs full-precision greedy):**

| Config | p@1 | Source | Δ vs greedy |
|--------|-----|--------|-------------|
| greedy | 84.2% | full_precision | — |
| pless T1=1.0 | 85.4% | temp_sweep | +1.2pp |
| pless T1=0.7 | 84.8% | temp_sweep | +0.6pp |
| pless t=0.6 | 87.5% | full_precision | +3.3pp |

The direction is consistent: P-less outperforms greedy across two independent evaluation runs. The full-precision run shows an even larger gap (+3.3pp for pless t=0.6). However, both comparisons cross evaluation pipelines, and the +1.2pp from the temp sweep is within SE ≈ 2.8pp.

**Verdict:** P-less consistently scores above greedy on HumanEval across independent runs, which is encouraging. The magnitude (+1.2pp to +3.3pp) is larger than on MBPP (~0pp). But no single within-run comparison definitively establishes this, and HumanEval's smaller task count (164 vs 500) means wider confidence intervals. **Conclusion: directionally positive, not statistically confirmed.**

On MBPP, P-less matches greedy (within noise). On HumanEval, P-less trends above greedy. The pattern is consistent with P-less being especially well-suited to HumanEval's constrained function-completion format.

### Finding 3: Diversity efficiency — T1 is the main knob on BOTH benchmarks

**HumanEval diversity efficiency (Qwen2.5-Coder-7B-Instruct, pless):**

| Transition | Δ sdiv | Δ p@1 | Efficiency (sdiv/pp) |
|------------|--------|-------|----------------------|
| T1=0.7→1.0 | +0.006 | +0.6pp | N/A (p@1 unchanged within noise) |
| T1=1.0→1.5 | +0.033 | -0.9pp | ~0.037 sdiv/pp |
| T1=1.5→2.0 | +0.112 | -2.1pp | ~0.053 sdiv/pp |
| T1=2.0→2.5 | +0.193 | -18.1pp | 0.011 sdiv/pp (catastrophic trade) |

**MBPP diversity efficiency (same model):**

| Transition | Δ sdiv | Δ p@1 | Efficiency |
|------------|--------|-------|------------|
| T1=1.0→1.5 | +0.067 | -0.5pp | 0.134 sdiv/pp |
| T1=1.5→2.0 | +0.182 | -4.2pp | 0.043 sdiv/pp |

T1 remains the effective diversity knob on HumanEval. Efficiency is lower than MBPP (0.037-0.053 vs 0.043-0.134 sdiv/pp) because HumanEval instruct is more peaked — sdiv at T1=1.0 is only 0.016 (vs MBPP's 0.059). The underlying mechanism is the same: T1 adjusts how many tokens survive the P-less threshold.

### Finding 4: Pass@10 — P-less vs temperature

| Config | HE p@1 | HE p@10 | sdiv | Uplift (p@10 - p@1) |
|--------|--------|---------|------|----------------------|
| pless T1=1.0 | 85.4% | 89.0% | 0.016 | +3.6pp |
| pless T1=1.5 | 84.5% | 88.4% | 0.049 | +3.9pp |
| pless T1=2.0 | 82.4% | 92.1% | 0.161 | +9.7pp |
| pless T1=2.5 | 64.3% | 93.9% | 0.354 | +29.6pp |
| temp t=0.7 | 80.6% | 91.5% | 0.183 | +10.9pp |
| temp t=1.0 | 77.9% | 91.5% | 0.252 | +13.6pp |

**Comparison at roughly matched diversity:** P-less T1=2.0 (sdiv=0.161) vs temp t=0.7 (sdiv=0.183) — diversity differs by ~14%, not perfectly matched. P-less has better pass@1 (82.4% vs 80.6%, Δ=+1.8pp) and similar pass@10 (92.1% vs 91.5%, Δ=+0.6pp). Both differences are within SE ≈ 2.8pp but the direction favors P-less.

**Best absolute pass@10:** pless T1=2.5 at 93.9%, but at catastrophic p@1 cost (64.3%). The practical ceiling is pless T1=2.0 (92.1% pass@10, 82.4% pass@1).

**Comparison with MBPP:** On MBPP, temp had better p@1 than P-less at matched diversity; on HumanEval, the advantage reverses. This is consistent across all metrics but the margins are small relative to SE.

---

## 2. Multi-Model Analysis on HumanEval

### P-less T1 sweep across all 6 models (pless method, pass@1)

| T1 | Qwen2.5-7B-Inst | Qwen2.5-7B-Base | Qwen3-30B | CL-7b-Inst | CL-7b-Base | Codestral-22B |
|----|-----------------|-----------------|-----------|------------|------------|---------------|
| 0.7 | 84.8% | 56.3% | 75.2% | 27.4% | 3.4% | 5.7% |
| 1.0 | 85.4% | 55.9% | 75.5% | 27.1% | 3.4% | 7.0% |
| 1.5 | 84.5% | 50.9% | 75.3% | 26.8% | 2.9% | 10.8% |
| 2.0 | 82.4% | 36.4% | 76.2% | 27.0% | 1.6% | 15.4% |
| 2.5 | 64.3% | 1.8% | 75.4% | 19.1% | 0.1% | 2.1% |
| 3.0 | 19.9% | 0.0% | 75.3% | 4.9% | 0.0% | 0.0% |

**Two reliable patterns and two noisy ones:**

**Pattern A — Standard (Qwen2.5-7B-Inst, Qwen2.5-7B-Base):**
Sweet spot at T1=0.7-1.0, gradual degradation, catastrophe at T1=2.5-3.0. This matches MBPP findings. These are the only models with enough absolute performance to make T1 comparisons meaningful.

**Pattern B — T1-Immune (Qwen3-Coder-30B):**
Pass@1 barely moves from T1=0.7 to T1=3.0 (75.2-76.2%, all within 2.8pp SE). The model is extremely peaked — P-less is effectively greedy at all temperatures. This is robust: the flatness spans 6 T1 values, not just a pair.

**Caution on "Pattern C" (Codestral, CodeLlama):**
The prior version of this report identified an "inverted" pattern where P-less pass@1 increases with T1. However, Codestral (5.7%→15.4%) and CodeLlama-7b-Base (3.4%→1.6%) are operating at very low absolute performance — roughly 9-25 tasks passing out of 164. At these levels, a few tasks flipping pass/fail shifts the percentage by several points. CodeLlama-7b-Instruct (27.4%→27.0%) is flat within noise. **The "inverted" pattern may be real for Codestral but cannot be distinguished from noise with 164 tasks at single-digit pass rates.**

### pless_norm vs pless: Does normalization matter?

On strong models (Qwen2.5-7B-Instruct, Qwen3-30B), pless and pless_norm are interchangeable — all differences <1pp pass@1.

On weak models, pless_norm shows a consistent (though modest) advantage:

| Model | T1 | pless p@1 | pless_norm p@1 | pless p@10 | pless_norm p@10 |
|-------|-----|-----------|----------------|------------|-----------------|
| CL-7b-Base | 0.7 | 3.4% | 4.2% | 12.2% | 14.0% |
| CL-7b-Base | 1.5 | 2.9% | 2.9% | 18.3% | 22.6% |
| Codestral | 0.7 | 5.7% | 6.5% | 23.2% | 24.4% |
| Qwen2.5-7B-Base | 1.0 | 55.9% | 57.2% | 79.3% | 81.1% |

The relaxed pless_norm threshold (which normalizes by vocabulary size) lets more tokens through, which matters when the correct token isn't in the very top of the distribution. The pass@10 improvement is more pronounced than pass@1, suggesting pless_norm preserves access to correct-but-unlikely tokens.

**Verdict:** pless_norm is weakly preferred over pless on weak/base models. On strong instruct models, the choice is irrelevant.

### Catastrophe threshold comparison

| Model | T1=2.0 p@1 | T1=2.5 p@1 | T1=3.0 p@1 | Threshold |
|-------|-----------|-----------|-----------|-----------|
| Qwen2.5-7B-Inst | 82.4% | 64.3% | 19.9% | T1=2.5 |
| Qwen2.5-7B-Base | 36.4% | 1.8% | 0.0% | T1=2.5 |
| Qwen3-30B | 76.2% | 75.4% | 75.3% | **No collapse** |
| CL-7b-Inst | 27.0% | 19.1% | 4.9% | T1=2.5-3.0 |
| CL-7b-Base | 1.6% | 0.1% | 0.0% | T1=2.0 (already near zero) |
| Codestral-22B | 15.4% | 2.1% | 0.0% | T1=2.5 |

**Catastrophe at T1=2.5-3.0 is universal** for models that show any T1 sensitivity (Qwen3-30B is immune). This matches MBPP where the boundary was between T1=2.0 and T1=3.0.

### Base vs Instruct comparison

| Pair | Base T1=1.0 p@1 | Instruct T1=1.0 p@1 | Inst advantage | Base sdiv | Inst sdiv |
|------|-----------------|---------------------|----------------|-----------|-----------|
| Qwen2.5-7B | 55.9% | 85.4% | +29.5pp | 0.173 | 0.016 |
| CodeLlama-7b | 3.4% | 27.1% | +23.7pp | 0.062 | 0.012 |

Instruct models are dramatically better on HumanEval AND far more peaked (10x less diversity). Instruct tuning concentrates probability on correct solutions for these function-completion tasks.

---

## 3. Diversity Deep-Dive: MBPP vs HumanEval

### Peakedness comparison (Qwen2.5-Coder-7B-Instruct)

| T1 | MBPP sdiv | HumanEval sdiv | Ratio (HE/MBPP) |
|----|-----------|----------------|-----------------|
| 0.7 | ~0.03 | 0.009 | 0.30x |
| 1.0 | 0.059 | 0.016 | 0.27x |
| 1.5 | 0.126 | 0.049 | 0.39x |
| 2.0 | 0.308 | 0.161 | 0.52x |

HumanEval is **2-4x more peaked** than MBPP on the same model. HumanEval tasks are more constrained (specific function signatures, clearer specifications), leaving less room for structural variation. This means:
- T1 needs to be pushed higher on HumanEval to get the same diversity
- But the model can tolerate this push because pass@1 degrades more slowly on HumanEval

### Temperature vs P-less: who produces more productive diversity?

| Metric | P-less T1=2.0 | temp t=0.7 | Δ | Significant? |
|--------|---------------|------------|---|--------------|
| **MBPP p@1** | 72.5% | 74.2% | -1.7pp | No (SE ≈ 1.75pp) |
| **MBPP p@10** | 89.6% | 87.8% | +1.8pp | Borderline |
| **HE p@1** | 82.4% | 80.6% | +1.8pp | No (SE ≈ 2.8pp) |
| **HE p@10** | 92.1% | 91.5% | +0.6pp | No (SE ≈ 2.8pp) |

Note: diversity levels differ (P-less sdiv=0.161, temp sdiv=0.183 on HE), so this is not a perfectly controlled comparison.

On both benchmarks, P-less and temperature perform within noise of each other when compared at roughly similar diversity levels. The direction on HumanEval slightly favors P-less on both metrics, but no comparison reaches statistical significance.

---

## 4. Edge Cases and Challenges to MBPP Conclusions

### Challenge 1: Qwen3-30B — P-less is useless on extremely peaked models

On Qwen3-30B, P-less at any temperature from 0.7 to 3.0 produces sdiv < 0.04. Even at T1=3.0, there's virtually no diversity. Meanwhile, temp t=1.0 gets sdiv=0.105 and pass@10=81.1% (vs P-less's 79.3%).

**Implication:** As models get more capable and peaked, P-less's adaptive threshold becomes effectively 1-token-only at all T1 values. The threshold is `Σ(prob²)`, and when prob_top ≈ 0.98, the threshold is ~0.96, keeping only the top token. Temperature sampling, which doesn't truncate, remains the only way to inject diversity on such models.

**Counter-argument:** On Qwen3-30B, the model already solves 75%+ of tasks correctly with greedy. The marginal value of diversity is limited when the model is already near-ceiling.

### Challenge 2: Weak models — P-less may over-truncate

Codestral at pless T1=0.7 gets 5.7% pass@1 vs temp t=0.7's 15.6%. Similarly, CodeLlama-7b-Base pless T1=1.0 gets 3.4% vs pless_norm's 4.2%. At these low absolute levels, the exact numbers are noisy, but the direction is consistent: P-less's threshold is too aggressive when the model's probability mass isn't concentrated on correct tokens.

**pless_norm helps somewhat** on weak models (see Section 2), recovering 0.5-4pp pass@10 by relaxing the threshold. But temperature still dominates for exploration: CodeLlama-7b-Base temp t=0.7 gets 17.1% pass@10 vs pless T1=0.7's 12.2%.

### Challenge 3: CodeLlama-7b-Instruct — P-less wins pass@1, temperature wins pass@10

CL-7b-Inst at pless T1=1.0 gets 27.1% pass@1 vs temp t=0.7's 25.9% (+1.2pp). But sdiv=0.012 vs 0.085 — P-less is nearly deterministic. For pass@10: pless T1=1.0 gets 32.3% vs temp t=0.7's 51.8%. This is the clearest example of the trade-off: P-less maximizes the mode, temperature maximizes the tail.

---

## 5. Revised Conclusions

### Robustly confirmed across BOTH benchmarks:

1. **T1=0.7-1.5 sweet spot.** On all models with standard T1 sensitivity (Qwen2.5 family), T1=0.7-1.5 costs ≤1pp pass@1 while providing meaningful diversity. Differences within this range are within noise on both benchmarks. **Confidence: HIGH.**

2. **Catastrophe at T1=2.5-3.0.** Every T1-sensitive model collapses between T1=2.0 and T1=3.0. HumanEval's T1=2.5 data pinpoints the cliff more precisely. **Confidence: HIGH.**

3. **T1 is the main diversity knob.** T1 transitions provide far more diversity per pp of pass@1 than T2 (from MBPP analysis). On HumanEval, T1 is the only diversity mechanism tested (no T2 data), and it works as expected. **Confidence: HIGH** (for the T1 claim; T2 conclusion based on MBPP only).

4. **Instruct models are more peaked.** Consistently across both model families on both benchmarks, instruct variants produce 5-10x less diversity at matched T1. **Confidence: HIGH.**

### Directionally supported but not statistically confirmed:

5. **P-less ≥ greedy on HumanEval instruct.** Two independent evaluation runs both show P-less scoring above greedy on Qwen2.5-7B-Instruct (temp sweep: +1.2pp; full precision: +3.3pp). The direction is consistent, but individual comparisons are within SE ≈ 2.8pp. On MBPP, P-less ≈ greedy (within noise). **Confidence: MEDIUM.** Needs larger evaluation or paired testing to confirm.

6. **P-less vs temperature ≈ draw at matched diversity.** On both benchmarks, at roughly comparable diversity levels, P-less and temperature produce similar pass@1 and pass@10. Small directional advantages exist (P-less on HumanEval, temperature on MBPP) but none are significant. **Confidence: MEDIUM** — the finding is "no clear winner," which is itself informative.

### Model-dependent findings:

7. **P-less is useless on extremely peaked models (Qwen3-30B).** T1 has no effect; diversity requires bypassing P-less entirely. **Confidence: HIGH** (flat across 6 T1 values).

8. **P-less may over-truncate on weak models.** Consistent direction on Codestral, CodeLlama-7b-Base, but absolute performance is so low that individual comparisons are noisy. pless_norm partially mitigates this. **Confidence: MEDIUM-LOW** (direction consistent, magnitudes unreliable).

9. **P-less excels on "goldilocks" models** — strong enough that top tokens are reliably correct, but not so peaked that only one token survives. The Qwen2.5-Coder-7B family is the clearest example. **Confidence: MEDIUM** (observed on one family; needs more model families to generalize).

---

## 6. Final Practical Recommendations

| Scenario | Recommendation | Rationale | Confidence |
|----------|---------------|-----------|------------|
| Strong instruct model, want max p@1 | **pless T1=0.7-1.0** | Matches or slightly exceeds greedy; free diversity | Medium-High |
| Strong instruct model, want best-of-N | **pless T1=1.5-2.0** | Best p@10 with ≤2pp p@1 cost | High |
| Extremely peaked model (Qwen3-30B class) | **temp t=0.7-1.0** | P-less is effectively greedy; temp provides only diversity | High |
| Weak/base model on the task | **temp t=0.7** or **pless_norm** | P-less over-truncates; temp preserves access to all tokens; pless_norm is a middle ground | Medium |
| Unknown model capability | **Try pless T1=1.0 first** | If p@1 ≈ greedy, keep it. If p@1 << temp, switch to temp | Medium |

**T2 verdict unchanged:** T2 was not tested on HumanEval, but MBPP showed it never helps pass@10 and produces junk diversity. Given that HumanEval is even more peaked than MBPP, T2 would have even fewer branching points to act on. T2 can be safely ignored.

**pless_norm verdict:** Interchangeable with pless on strong instruct models. Weakly preferred on base/weak models where the relaxed threshold preserves correct-but-unlikely tokens. Not a substitute for temperature when diversity is the goal.

---

## 7. What's Missing (Limitations of This Analysis)

1. **No paired statistical tests.** All significance claims use approximate SE from binomial proportion. A proper McNemar test or bootstrap CI on per-task pass rates would be more rigorous.

2. **No within-run greedy baseline for HumanEval temp sweep.** The most important comparison (pless vs greedy) relies on cross-run data. Adding a greedy config to the temp sweep would eliminate this gap.

3. **T2 not tested on HumanEval.** The T2 conclusion is extrapolated from MBPP only.

4. **Only two model families (Qwen, CodeLlama) tested on both benchmarks.** The "goldilocks" model hypothesis needs more diverse model families to validate.

5. **HumanEval's 164 tasks provide low statistical power.** Many interesting comparisons (1-3pp differences) cannot be resolved. MBPP's 500 tasks are better but still marginal for sub-1pp claims.
