# Cross-Model Comparison: P-less T1/T2 on Instruct Models

**3B:** Qwen2.5-Coder-3B-Instruct | **7B:** Qwen2.5-Coder-7B-Instruct | **Dataset:** MBPP-full (500 tasks x 10 samples)

This report systematically validates each conclusion from the 3B experiment against 7B results. The 7B experiment includes additional T2 configurations (T2=3.0, T2=4.0) and T2 sweeps at T1=1.0 (not tested on 3B).

### Methodological Notes

1. **Statistical significance.** With 500 tasks and per-task pass rate std ~0.39, the standard error of mean pass@1 is ~1.75pp. Differences below ~3.5pp (2 SE) should be treated as **directional signals, not confirmed effects**. Many key comparisons in this report fall in the 0.3–0.7pp range.

2. **Greedy baseline measurement.** Greedy uses n=1 sample per task (deterministic). Pass@1 for stochastic methods uses n=10 samples. Both estimate the same quantity (probability a single sample is correct), but greedy has no sampling variance while stochastic estimates have SE ~1.75pp.

3. **Base model reference.** Both the 3B and 7B reports reference the same base model: Qwen2.5-Coder-3B (base). No 7B base model results exist. The "base" rows in the 7B report are the 3B base — this does not affect instruct-vs-instruct comparisons but means we cannot assess the 7B instruct-vs-base gap.

4. **Token survivor mechanism.** Interpretations citing "97% deterministic steps" or "3% branching" are extrapolated from the 3B token survivor analysis. We do not have token-level survivor data for 7B. Given 7B's lower output diversity, the actual 7B deterministic fraction is likely even higher, making the extrapolation directionally valid but not quantitatively precise.

---

## 1. Scale Effect: Absolute Performance

| Config | 3B pass@1 | 7B pass@1 | Δ (7B - 3B) |
|--------|-----------|-----------|-------------|
| greedy | 62.6% | 77.6% | **+15.0pp** |
| pless T1=0.6 | 62.5% | 77.2% | +14.7pp |
| pless T1=1.0 | 62.3% | 77.2% | +14.9pp |
| pless T1=1.5 | 61.2% | 76.7% | +15.5pp |
| pless T1=2.0 | 56.0% | 72.5% | +16.5pp |
| pless T1=3.0 | 0.2% | 2.7% | +2.5pp |
| temp t=0.2 | 62.2% | 76.8% | +14.6pp |
| temp t=0.6 | 60.0% | 74.2% | +14.2pp |
| temp t=0.8 | 56.4% | 72.0% | +15.6pp |
| temp t=1.5 | 4.5% | 12.3% | +7.8pp |

**Observation:** The 7B model provides a consistent ~15pp advantage across configs in the usable range (T1 <= 2.0). The gap widens slightly at higher T1 (T1=2.0: +16.5pp) and narrows at catastrophic settings (T1=3.0, temp t=1.5), where both models collapse but 7B retains marginally more signal. These scale effects are well beyond noise (15pp >> 3.5pp threshold).

---

## 2. Finding Validation: T1=3.0 Catastrophic Collapse

| Metric | 3B | 7B | Verdict |
|--------|----|----|---------|
| pass@1 | 0.2% | 2.7% | Both collapse |
| pass@10 | 2.2% | 18.8% | 7B retains some signal |
| cover@0.7 | 0.0% | 0.0% | Both at zero |
| struct_div | 0.9024 | 0.2645 | 3B near-random; 7B retains structure |

**Verdict: CONFIRMED.** T1=3.0 is catastrophic on both models. The effect size is massive (>70pp drop from T1=2.0), far beyond any statistical uncertainty. 7B degrades more gracefully — 2.7% vs 0.2% pass@1, 0.26 vs 0.90 struct_div (suggesting 7B still produces somewhat coherent code rather than random tokens), and meaningful pass@10 (18.8%) — but pass@1 is unusable on both. The catastrophe threshold remains between T1=2.0 and T1=3.0 regardless of model size.

---

## 3. Finding Validation: Quality-Filter Hypothesis

**3B conclusion:** REJECTED — P-less never beats temperature at matched diversity (Δ = -0.4pp to -4.3pp).

| Comparison | 3B Δ pass@1 | 7B Δ pass@1 |
|------------|-------------|-------------|
| pless T1=1.5 vs nearest temp | -1.0pp | -0.1pp |
| pless T1=2.0 vs nearest temp | -0.4pp | +0.4pp |
| pless T1=3.0 vs nearest temp | -4.3pp | -9.5pp |

**Detail at T1=2.0 (the interesting case):**

| | 3B | 7B |
|-|----|----|
| P-less T1=2.0 pass@1 | 56.0% | 72.5% |
| P-less T1=2.0 sdiv | 0.4156 | 0.3082 |
| Nearest temp pass@1 | 56.4% (t=0.8) | 72.0% (t=0.8) |
| Nearest temp sdiv | 0.4294 | 0.2779 |
| Δ pass@1 | -0.4pp | +0.4pp |

**Self-critique:** The matched-diversity comparison has methodological weaknesses:
- **Diversity matching is imprecise.** On 7B, P-less sdiv=0.3082 vs temp sdiv=0.2779 — a 10% gap. P-less has MORE diversity than its "matched" temperature config, so the +0.4pp advantage is slightly inflated (more diversity usually correlates with lower pass@1). On 3B the gap is smaller (0.4156 vs 0.4294, ~3%).
- **Both deltas are within noise.** The 0.4pp swings in both directions (3B: -0.4pp, 7B: +0.4pp) are well below the ~3.5pp significance threshold. We cannot distinguish these from zero.
- **We only have two comparable temperature anchors** (t=0.2 and t=0.8) to match against continuous P-less diversity levels. A denser temperature grid would improve matching fidelity.

**Verdict: STILL UNCONFIRMED.** The 7B data does not reject the quality-filter hypothesis as strongly as 3B did, but the +0.4pp signal is indistinguishable from noise. The honest conclusion: **P-less and temperature produce statistically equivalent pass@1 at similar diversity levels on instruct models**, regardless of scale. The quality-filter hypothesis is neither confirmed nor rejected — it would require substantially more tasks or models to resolve a sub-1pp effect.

---

## 4. Finding Validation: Sweet Spot T1=1.0–1.5

### T1 sweep comparison (P-less only, no T2)

| T1 | 3B pass@1 | 3B sdiv | 7B pass@1 | 7B sdiv |
|----|-----------|---------|-----------|---------|
| 0.6 | 62.5% | 0.0345 | 77.2% | 0.0305 |
| 1.0 | 62.3% | 0.1078 | 77.2% | 0.0586 |
| 1.5 | 61.2% | 0.2331 | 76.7% | 0.1262 |
| 2.0 | 56.0% | 0.4156 | 72.5% | 0.3082 |

**Pass@1 degradation pattern:**

| Transition | 3B Δ | 7B Δ |
|------------|------|------|
| T1=0.6 → 1.0 | -0.2pp | 0.0pp |
| T1=1.0 → 1.5 | -1.1pp | -0.5pp |
| T1=1.5 → 2.0 | **-5.2pp** | **-4.2pp** |
| T1=2.0 → 3.0 | **-55.8pp** | **-69.8pp** |

**Verdict: CONFIRMED.** The sweet spot is robust:
- T1=0.6 to 1.0: negligible cost on both models (within noise)
- T1=1.0 to 1.5: small cost (0.5–1.1pp), still within or near noise, but consistent direction
- T1=1.5 to 2.0: clear penalty (~5pp), well beyond noise
- T1=2.0 to 3.0: catastrophic collapse

The T1=1.0–1.5 sweet spot holds for both models. The T1=1.5→2.0 cliff and T1=2.0→3.0 catastrophe are the only transitions with unambiguous statistical significance.

**Key difference — 7B is more peaked:** At the same T1, 7B produces roughly HALF the diversity of 3B:

| T1 | 3B sdiv | 7B sdiv | Ratio (7B/3B) |
|----|---------|---------|---------------|
| 0.6 | 0.0345 | 0.0305 | 0.88x |
| 1.0 | 0.1078 | 0.0586 | 0.54x |
| 1.5 | 0.2331 | 0.1262 | 0.54x |
| 2.0 | 0.4156 | 0.3082 | 0.74x |

The ratio is most dramatic at T1=1.0–1.5 (0.54x). This is consistent with larger instruct models having sharper probability distributions. Consequence: to achieve 3B-equivalent diversity on 7B, T1 must be pushed higher — but 7B tolerates this better (gentler degradation curve).

**Caveat:** We infer "more peaked distribution" from lower output diversity, but sdiv measures the diversity of *complete program outputs*, not the token-level distribution. Lower sdiv could also reflect 7B producing more canonical/convergent solutions even from a non-trivially-spread token distribution. Without 7B token survivor data, the mechanism is inferred, not measured.

---

## 5. Finding Validation: T2 at T1=2.0

### 3B (only T2=2.0 and T2=5.0 tested)

| T2 | pass@1 | Δ pass@1 | sdiv | Δ sdiv |
|----|--------|----------|------|--------|
| — (baseline) | 56.0% | — | 0.4156 | — |
| 2.0 | 54.1% | -1.9pp | 0.4473 | +0.032 |
| 5.0 | 51.1% | -4.9pp | 0.4558 | +0.040 |

### 7B (T2=2.0, 3.0, 4.0, 5.0 tested)

| T2 | pass@1 | Δ pass@1 | sdiv | Δ sdiv |
|----|--------|----------|------|--------|
| — (baseline) | 72.5% | — | 0.3082 | — |
| 2.0 | 70.1% | -2.4pp | 0.3278 | +0.020 |
| 3.0 | 70.0% | -2.5pp | 0.3422 | +0.034 |
| 4.0 | 70.5% | -2.0pp | 0.3384 | +0.030 |
| 5.0 | 68.5% | -4.0pp | 0.3388 | +0.031 |

**Self-critique on "T2=4.0 is best":** The original report claimed T2=4.0 > T2=2.0 based on 70.5% vs 70.1% — a 0.4pp difference, which is deep within noise. A more honest reading:

- **T2=2.0/3.0/4.0 are statistically indistinguishable** at T1=2.0 (70.0–70.5%, spread < 0.5pp)
- **T2=5.0 is clearly worse** (-4.0pp from baseline, vs -2.0 to -2.5pp for the others)
- **All T2 values hurt pass@1** at this T1 level on both models

**Verdict: CONFIRMED with correction.**
- T2=5.0 overshooting: confirmed (3B: -4.9pp, 7B: -4.0pp). Effect size is above noise.
- T2=2.0 Pareto-optimal: **not meaningfully updated.** T2=2.0/3.0/4.0 perform equivalently. The original 3B finding (T2=2.0 is best) and the finer 7B grid (T2=4.0 is best) are both over-reading noise. The real finding: **any T2 in [2.0, 4.0] costs ~2pp pass@1 for ~0.03 sdiv gain at T1=2.0.**
- The practical implication holds: if you want T2 at T1=2.0, pick any value in [2.0, 4.0]; avoid T2=5.0.

---

## 6. NEW Finding: T2 at T1=1.0 (7B only)

This was NOT tested on 3B. Results:

| T2 | pass@1 | Δ pass@1 | sdiv | Δ sdiv |
|----|--------|----------|------|--------|
| — (baseline) | 77.2% | — | 0.0586 | — |
| 2.0 | 77.9% | +0.7pp | 0.0572 | -0.001 |
| 3.0 | 77.8% | +0.6pp | 0.0581 | -0.001 |
| 4.0 | 77.9% | +0.7pp | 0.0555 | -0.003 |
| 5.0 | 77.6% | +0.4pp | 0.0562 | -0.002 |

### Critical analysis

**What the data shows:** All four T2 values at T1=1.0 yield equal or higher pass@1 than the T2-free baseline, with the top configs (T2=2.0 and T2=4.0) at 77.9% — 0.3pp above greedy's 77.6%.

**What we can and cannot conclude:**

1. **Directional consistency is real.** All 4 T2 values improve pass@1. Under the null (T2 has no effect), the probability of 4/4 positive is 1/16 = 6.25%. However, since these configs share the same T1=1.0 survivor set and differ only in redistribution strength, they are highly correlated — effectively one treatment at four doses rather than four independent tests. So we have roughly p ≈ 0.5 for a single directional observation. **Not significant.**

2. **"Beats greedy" is NOT established.** The 77.9% vs 77.6% gap (0.3pp) is well within the SE of the pless estimate (~1.75pp). Per-task analysis shows:
   - 364/500 tasks: pless passes all 10/10 samples (trivially correct)
   - 86/500 tasks: pless fails all 10/10 samples (trivially wrong)
   - 50/500 tasks: pless partially correct (1–9 of 10)
   - 350/500 tasks: both greedy and pless-10/10 agree (both pass)
   - 75/500 tasks: both greedy and pless-0/10 agree (both fail)
   - Only ~75 tasks drive the entire difference between configs

   A paired test on these ~75 discriminating tasks would be needed to assess significance. With 0.3pp overall = ~1.5 net correct samples difference (out of 5000), this is almost certainly not significant.

3. **The mechanism story is plausible but unverified for 7B.** The interpretation (T2 only touches ~3% of steps, gentle nudges at rare branching points) is based on 3B token survivor data. At T1=1.0 on 3B, branching=0.4%, constrained=2.8%. On 7B, these fractions are likely even lower. A +0.7pp improvement from touching <3% of steps would require T2 to improve ~23pp of outcomes at affected steps — implausibly large. This further suggests the +0.7pp is noise.

4. **The diversity DECREASE is the one robust signal.** All four T2 values slightly reduce sdiv (by 0.001–0.003). This is consistent with T2 flattening the few multi-survivor steps: more uniform sampling at branching points → less structural variety. This is a small but directionally consistent effect (same direction across all 4 T2 values at this T1, AND directionally opposite from T2 at T1=2.0 where diversity increases).

**Verdict: INTERESTING SIGNAL, NOT CONFIRMED.** T2 at T1=1.0 shows a consistent positive direction for pass@1, but the magnitudes (0.3–0.7pp) are within noise. The claim "beats greedy" requires a paired statistical test that we have not performed. Until validated on additional models/datasets, or with a proper significance test, this should be reported as a **hypothesis to investigate**, not a confirmed finding.

---

## 7. Summary Table: 3B Findings vs 7B Validation

| # | 3B Finding | 7B Result | Status | Confidence |
|---|-----------|-----------|--------|------------|
| 1 | T1=3.0 catastrophically collapses | 3B: 0.2%, 7B: 2.7% — both collapse | **CONFIRMED** | High (>70pp effect) |
| 2 | Quality-filter hypothesis rejected | 7B: +0.4pp (3B: -0.4pp). Both within noise | **UNRESOLVED** | Low (<1pp signal) |
| 3 | Sweet spot: T1=1.0–1.5 | Both models peak at T1=1.0–1.5 | **CONFIRMED** | High (5pp cliff at T1=2.0) |
| 4 | T2=2.0 Pareto-optimal at T1=2.0 | T2=2.0/3.0/4.0 are equivalent (~70%) | **CONFIRMED** (any T2 in [2,4] works) | Medium (0.5pp spread = noise) |
| 5 | T2=5.0 overshoots | 3B: -4.9pp, 7B: -4.0pp at T1=2.0 | **CONFIRMED** | Medium-High (~4pp effect) |
| 6 | — (not tested on 3B) | T2 at T1=1.0: +0.4–0.7pp pass@1 | **SUGGESTIVE** | Low (within noise, needs validation) |

---

## 8. Recommended Configurations

### For maximum pass@1 (instruct model):

| | 3B | 7B |
|-|----|----|
| Best config | greedy (62.6%) | pless T1=1.0 T2=2.0 (77.9%) |
| Greedy | 62.6% | 77.6% |
| Gap | 0.0pp | +0.3pp |

**Caveat:** On 7B, the "best" P-less config edges greedy by 0.3pp — within noise. A defensible statement is: **P-less T1=1.0 (with or without T2) matches greedy on 7B instruct**, which is itself notable since P-less provides non-zero diversity while greedy provides none.

### For best pass@1/diversity trade-off:

| | 3B | 7B |
|-|----|----|
| **Recommended** | pless T1=1.0 (62.3%, sdiv=0.108) | pless T1=1.0 (77.2%, sdiv=0.059) |
| High-diversity | pless T1=1.5 (61.2%, sdiv=0.233) | pless T1=1.5 (76.7%, sdiv=0.126) |

This is the strongest practical recommendation: P-less T1=1.0 provides greedy-equivalent pass@1 with meaningful (though small) output diversity on both model sizes. T1=1.5 doubles diversity for <1pp cost.

### For pass@10 maximization (best-of-N):

| | 3B | 7B |
|-|----|----|
| Best pass@10 | pless T1=2.0 (82.2%) | pless T1=2.0 (89.6%) |
| Notes | -6pp pass@1 cost | -5pp pass@1 cost |

T2 does NOT help pass@10 at T1=2.0 — it slightly reduces it (89.6% → 89.0% on 7B). This makes sense: T2 adds marginal diversity (~0.03 sdiv) but the per-sample quality loss from redistribution outweighs the exploration benefit within just 10 samples.

---

## 9. Robustly Supported Takeaways

These conclusions are supported by effect sizes well above noise on both models:

1. **Scale helps uniformly.** 7B provides ~15pp pass@1 advantage across all sampling methods. The relative ordering is preserved.

2. **T1=1.0–1.5 is the sweet spot.** The cliff at T1=2.0 (~5pp) and catastrophe at T1=3.0 (>70pp) are clear on both models.

3. **T2 hurts at T1=2.0.** Any T2 value costs 2–5pp pass@1 for small diversity gains. T2=5.0 is strictly worse than T2 in [2.0, 4.0].

4. **7B is more peaked than 3B.** Output diversity is roughly halved at matched T1 settings. Larger instruct models need higher T1 to achieve the same diversity level.

5. **P-less matches greedy.** On both models, P-less T1=1.0 is within 0.4pp of greedy while providing non-zero diversity. This is the clearest practical value proposition.

## 10. Open Questions Requiring More Data

1. **Does T2 at T1=1.0 genuinely help?** The +0.7pp on 7B is suggestive but within noise. Testing on 3B instruct with T2 at T1=1.0, or on other model families, would resolve this.

2. **Quality-filter hypothesis.** The 3B→7B swing from -0.4pp to +0.4pp hints at scale dependence, but both are within noise. Testing on 13B+ models or with >500 tasks would help.

3. **Is 7B genuinely more peaked, or does it just produce more canonical code?** Token survivor analysis on 7B would distinguish distribution peakedness from output convergence.

4. **Where is the catastrophe boundary?** Testing T1=2.5 would narrow the gap between "usable T1=2.0" and "catastrophic T1=3.0."
