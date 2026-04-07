# P-less T1/T2 Post-Truncation Temperature: MBPP Analysis

**Model:** Qwen2.5-Coder-3B (base) | **Dataset:** MBPP-full (500 tasks × 10 samples) | **Prompt:** BigCode zero-shot docstring

T₁ (pre-truncation temperature) scales logits before P-less computes its collision entropy threshold. T₂ (post-truncation temperature) applies `prob^(1/T₂)` to flatten the survivor distribution after P-less pruning.

## Full Metrics Comparison

| # | Config | T1 | T2 | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.7 | struct_div | codebleu_div |
|---|--------|----|----|--------|--------|--------|---------|-----------|------------|--------------|
| 1 | beam4 t=1.0 | 1.0 | — | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 0.0000 | 0.0000 |
| 2 | greedy t=1.0 | 1.0 | — | 60.2 | 60.2 | 60.2 | 60.2 | 60.2 | 0.0000 | 0.0000 |
| 3 | pless_norm t=0.6 | 0.6 | — | 59.4 | 63.7 | 65.2 | 66.6 | 57.2 | 0.0957 | 0.2040 |
| 4 | pless t=0.6 | 0.6 | — | 59.3 | 63.4 | 64.8 | 66.2 | 57.2 | 0.0968 | 0.2033 |
| 5 | beam8 t=1.0 | 1.0 | — | 59.2 | 59.2 | 59.2 | 59.2 | 59.2 | 0.0000 | 0.0000 |
| 6 | pless_norm t=0.7 | 0.7 | — | 59.0 | 65.0 | 66.7 | 68.0 | 55.2 | 0.1365 | 0.2666 |
| 7 | pless T1=0.8 T2=5.0 **←** | 0.8 | 5.0 | 58.8 | 65.8 | 67.5 | 68.6 | 55.0 | 0.1701 | 0.3299 |
| 8 | pless t=0.8 **←** | 0.8 | — | 58.7 | 65.7 | 67.7 | 69.2 | 54.4 | 0.1673 | 0.2980 |
| 9 | pless t=0.7 | 0.7 | — | 58.6 | 64.5 | 66.2 | 67.4 | 55.4 | 0.1318 | 0.2585 |
| 10 | pless T1=0.6 T2=5.0 **←** | 0.6 | 5.0 | 58.5 | 63.2 | 64.7 | 66.2 | 55.6 | 0.1031 | 0.2070 |
| 11 | pless T1=0.6 T2=2.0 **←** | 0.6 | 2.0 | 58.3 | 63.1 | 64.5 | 65.6 | 56.0 | 0.1013 | 0.2150 |
| 12 | top_p0.95 t=0.2 | 0.2 | — | 58.2 | 66.6 | 69.2 | 71.6 | 54.4 | 0.1832 | 0.3871 |
| 13 | pless T1=0.8 T2=2.0 **←** | 0.8 | 2.0 | 58.0 | 65.3 | 67.3 | 68.8 | 54.2 | 0.1776 | 0.3223 |
| 14 | pless_norm t=1.0 | 1.0 | — | 57.6 | 66.9 | 69.8 | 72.4 | 53.4 | 0.2497 | 0.4335 |
| 15 | pless t=1.0 | 1.0 | — | 56.5 | 66.4 | 69.4 | 72.2 | 51.6 | 0.2578 | 0.4498 |
| 16 | pless T1=1.0 T2=5.0 **←** | 1.0 | 5.0 | 56.0 | 66.1 | 69.1 | 72.0 | 50.2 | 0.2504 | 0.4449 |
| 17 | pless T1=1.0 T2=2.0 **←** | 1.0 | 2.0 | 55.9 | 65.4 | 68.5 | 71.2 | 50.6 | 0.2551 | 0.4375 |
| 18 | temp t=0.7 | 0.7 | — | 42.6 | 63.9 | 70.6 | 77.6 | 34.2 | 0.5365 | 0.6903 |
| 19 | top_p0.9 t=1.0 | 1.0 | — | 34.7 | 58.4 | 67.3 | 77.0 | 20.6 | 0.5886 | 0.7167 |

*pass@k as %; cover@t = % of tasks with ≥t fraction correct; struct_div = mean pairwise AST edit distance; codebleu_div = 1 − mean pairwise CodeBLEU similarity. **←** = T1/T2 experiment configs.*

## T2 Effect at Fixed T1

### T1=0.6
Baseline (no T2): pass@1=59.3%, struct_div=0.0968, codebleu_div=0.2033

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 58.3% | -1.0pp | 0.1013 | +0.0045 | 0.2150 | +0.0117 |
| 5.0 | 58.5% | -0.9pp | 0.1031 | +0.0063 | 0.2070 | +0.0037 |

### T1=0.8
Baseline (no T2): pass@1=58.7%, struct_div=0.1673, codebleu_div=0.2980

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 58.0% | -0.7pp | 0.1776 | +0.0103 | 0.3223 | +0.0243 |
| 5.0 | 58.8% | +0.1pp | 0.1701 | +0.0028 | 0.3299 | +0.0319 |

### T1=1.0
Baseline (no T2): pass@1=56.5%, struct_div=0.2578, codebleu_div=0.4498

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 55.9% | -0.7pp | 0.2551 | -0.0027 | 0.4375 | -0.0123 |
| 5.0 | 56.0% | -0.5pp | 0.2504 | -0.0074 | 0.4449 | -0.0049 |

## Diversity Exchange Rate

How much diversity (struct_div, codebleu_div) is gained per percentage point of pass@1 lost, relative to the best baseline (pless t=0.6).

**Reference:** pless t=0.6 (pass@1=59.3%, struct_div=0.0968, codebleu_div=0.2033)

| Config | pass@1 cost | sdiv gain | sdiv/pp | cbdiv gain | cbdiv/pp |
|--------|-------------|-----------|---------|------------|----------|
| pless T1=0.8 T2=5.0 | 0.6pp | +0.0733 | 0.1309/pp | +0.1266 | 0.2261/pp |
| pless t=0.8 | 0.6pp | +0.0705 | 0.1137/pp | +0.0947 | 0.1527/pp |
| pless t=0.7 | 0.7pp | +0.0350 | 0.0473/pp | +0.0552 | 0.0746/pp |
| pless T1=0.6 T2=5.0 | 0.9pp | +0.0063 | 0.0073/pp | +0.0037 | 0.0043/pp |
| pless T1=0.6 T2=2.0 | 1.0pp | +0.0045 | 0.0044/pp | +0.0117 | 0.0115/pp |
| top_p0.95 t=0.2 | 1.1pp | +0.0864 | 0.0771/pp | +0.1838 | 0.1641/pp |
| pless T1=0.8 T2=2.0 | 1.3pp | +0.0808 | 0.0622/pp | +0.1190 | 0.0915/pp |
| pless_norm t=1.0 | 1.7pp | +0.1529 | 0.0899/pp | +0.2302 | 0.1354/pp |
| pless t=1.0 | 2.8pp | +0.1610 | 0.0579/pp | +0.2465 | 0.0887/pp |
| pless T1=1.0 T2=5.0 | 3.3pp | +0.1536 | 0.0465/pp | +0.2416 | 0.0732/pp |
| pless T1=1.0 T2=2.0 | 3.4pp | +0.1583 | 0.0460/pp | +0.2342 | 0.0681/pp |
| temp t=0.7 | 16.7pp | +0.4397 | 0.0263/pp | +0.4870 | 0.0292/pp |
| top_p0.9 t=1.0 | 24.6pp | +0.4918 | 0.0200/pp | +0.5134 | 0.0209/pp |

## Key Findings

1. **T2 effect is regime-dependent on the base model.** At T1=0.6, T2 adds only +0.005–0.007 struct_div while costing ~1pp pass@1 (poor exchange rate). At T1=0.8, T2=5.0 is a notable anomaly: +0.1pp pass@1 (within noise, but no cost) AND +0.032 codebleu_div — the one case where T2 appears genuinely beneficial. At T1=1.0, T2 *reverses direction* and decreases diversity, suggesting the flattening causes convergence on a few popular alternatives rather than spreading across many.

2. **T1 (pre-truncation temperature) does most of the work.** The diversity jump from T1=0.6→0.8→1.0 is substantial (struct_div 0.097→0.167→0.255), matching the effect of raising temperature in standard P-less. T1 controls pruning aggressiveness by shaping the distribution before the collision entropy threshold is computed. T1 is 2–14× more efficient than T2 at converting pass@1 into diversity.

3. **Best new config: pless T1=0.8 (no T2).** At 58.7% pass@1, 0.167 struct_div, 0.298 codebleu_div, it matches top_p0.95/t=0.2 (58.2%) while providing a useful diversity level — with zero hyperparameters beyond T1.

4. **Implication for instruct models:** T2 alone cannot rescue diversity on instruct models where P-less at low T1 leaves ~1 survivor. The instruct experiment needs high T1 (>1.0) to open the distribution first. However, the T1=0.8/T2=5.0 anomaly suggests T2 may have a narrow sweet spot when the survivor set is moderate — worth testing at high T1 on instruct where more survivors exist.

