# P-less T1/T2 Post-Truncation Temperature: MBPP Analysis

**Model:** Qwen2.5-Coder-3B (base) | **Dataset:** MBPP-full (500 tasks × 10 samples) | **Prompt:** BigCode zero-shot docstring

T₁ (pre-truncation temperature) scales logits before P-less computes its collision entropy threshold. T₂ (post-truncation temperature) applies `prob^(1/T₂)` to flatten the survivor distribution after P-less pruning.

## Full Metrics Comparison

| # | Config | T1 | T2 | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.7 | struct_div | codebleu_div |
|---|--------|----|----|--------|--------|--------|---------|-----------|------------|--------------|
| 1 | pless_norm t=0.6 | 0.6 | — | 59.5 | 63.9 | 65.4 | 66.8 | 57.2 | 0.0954 | 0.2035 |
| 2 | pless t=0.6 | 0.6 | — | 59.4 | 63.6 | 65.0 | 66.4 | 57.2 | 0.0965 | 0.2028 |
| 3 | pless T1=0.8 T2=5.0 **←** | 0.8 | 5.0 | 58.8 | 65.8 | 67.5 | 68.6 | 55.0 | 0.1701 | 0.3302 |
| 4 | pless t=0.8 **←** | 0.8 | — | 58.7 | 65.7 | 67.7 | 69.2 | 54.4 | 0.1673 | 0.2982 |
| 5 | pless T1=0.6 T2=5.0 **←** | 0.6 | 5.0 | 58.5 | 63.2 | 64.7 | 66.2 | 55.6 | 0.1031 | 0.2071 |
| 6 | pless T1=0.6 T2=2.0 **←** | 0.6 | 2.0 | 58.3 | 63.1 | 64.5 | 65.6 | 56.0 | 0.1013 | 0.2152 |
| 7 | top_p0.95 t=0.2 | 0.2 | — | 58.2 | 66.6 | 69.2 | 71.6 | 54.4 | 0.1832 | 0.3872 |
| 8 | pless T1=0.8 T2=2.0 **←** | 0.8 | 2.0 | 58.0 | 65.3 | 67.3 | 68.8 | 54.2 | 0.1776 | 0.3227 |
| 9 | pless_norm t=1.0 | 1.0 | — | 57.7 | 67.0 | 69.9 | 72.4 | 53.4 | 0.2512 | 0.4345 |
| 10 | pless t=1.0 | 1.0 | — | 56.6 | 66.5 | 69.4 | 72.2 | 51.6 | 0.2588 | 0.4503 |
| 11 | pless T1=1.0 T2=5.0 **←** | 1.0 | 5.0 | 56.0 | 66.1 | 69.1 | 72.0 | 50.2 | 0.2504 | 0.4451 |
| 12 | pless T1=1.0 T2=2.0 **←** | 1.0 | 2.0 | 55.9 | 65.4 | 68.5 | 71.2 | 50.6 | 0.2551 | 0.4376 |
| 13 | temp t=0.7 | 0.7 | — | 42.6 | 63.9 | 70.7 | 77.8 | 34.2 | 0.5365 | 0.6903 |
| 14 | top_p0.9 t=1.0 | 1.0 | — | 34.7 | 58.4 | 67.3 | 77.0 | 20.6 | 0.5886 | 0.7169 |

*pass@k as %; cover@t = % of tasks with ≥t fraction correct; struct_div = mean pairwise AST edit distance; codebleu_div = 1 − mean pairwise CodeBLEU similarity. **←** = T1/T2 experiment configs.*

## T2 Effect at Fixed T1

### T1=0.6
Baseline (no T2): pass@1=59.4%, struct_div=0.0965, codebleu_div=0.2028

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 58.3% | -1.1pp | 0.1013 | +0.0048 | 0.2152 | +0.0124 |
| 5.0 | 58.5% | -1.0pp | 0.1031 | +0.0066 | 0.2071 | +0.0043 |

### T1=0.8
Baseline (no T2): pass@1=58.7%, struct_div=0.1673, codebleu_div=0.2982

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 58.0% | -0.7pp | 0.1776 | +0.0103 | 0.3227 | +0.0245 |
| 5.0 | 58.8% | +0.1pp | 0.1701 | +0.0028 | 0.3302 | +0.0320 |

### T1=1.0
Baseline (no T2): pass@1=56.6%, struct_div=0.2588, codebleu_div=0.4503

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 55.9% | -0.7pp | 0.2551 | -0.0037 | 0.4376 | -0.0127 |
| 5.0 | 56.0% | -0.6pp | 0.2504 | -0.0084 | 0.4451 | -0.0052 |

## Diversity Exchange Rate

How much diversity (struct_div, codebleu_div) is gained per percentage point of pass@1 lost, relative to the best baseline (pless t=0.6).

**Reference:** pless t=0.6 (pass@1=59.4%, struct_div=0.0965, codebleu_div=0.2028)

| Config | pass@1 cost | sdiv gain | sdiv/pp | cbdiv gain | cbdiv/pp |
|--------|-------------|-----------|---------|------------|----------|
| pless T1=0.8 T2=5.0 | 0.7pp | +0.0736 | 0.1082/pp | +0.1274 | 0.1874/pp |
| pless t=0.8 | 0.7pp | +0.0708 | 0.0957/pp | +0.0954 | 0.1289/pp |
| pless T1=0.6 T2=5.0 | 1.0pp | +0.0066 | 0.0067/pp | +0.0043 | 0.0044/pp |
| pless T1=0.6 T2=2.0 | 1.1pp | +0.0048 | 0.0042/pp | +0.0124 | 0.0109/pp |
| top_p0.95 t=0.2 | 1.2pp | +0.0867 | 0.0699/pp | +0.1844 | 0.1487/pp |
| pless T1=0.8 T2=2.0 | 1.4pp | +0.0811 | 0.0571/pp | +0.1199 | 0.0844/pp |
| pless_norm t=1.0 | 1.8pp | +0.1547 | 0.0879/pp | +0.2317 | 0.1316/pp |
| pless t=1.0 | 2.9pp | +0.1623 | 0.0567/pp | +0.2475 | 0.0865/pp |
| pless T1=1.0 T2=5.0 | 3.4pp | +0.1539 | 0.0450/pp | +0.2423 | 0.0708/pp |
| pless T1=1.0 T2=2.0 | 3.6pp | +0.1586 | 0.0446/pp | +0.2348 | 0.0660/pp |
| temp t=0.7 | 16.8pp | +0.4400 | 0.0262/pp | +0.4875 | 0.0290/pp |
| top_p0.9 t=1.0 | 24.7pp | +0.4921 | 0.0199/pp | +0.5141 | 0.0208/pp |

## Key Findings

1. **T2 effect is regime-dependent on the base model.** At T1=0.6, T2 adds only +0.005–0.007 struct_div while costing ~1pp pass@1 (poor exchange rate). At T1=0.8, T2=5.0 is a notable anomaly: +0.1pp pass@1 (within noise, but no cost) AND +0.032 codebleu_div — the one case where T2 appears genuinely beneficial. At T1=1.0, T2 *reverses direction* and decreases diversity, suggesting the flattening causes convergence on a few popular alternatives rather than spreading across many.

2. **T1 (pre-truncation temperature) does most of the work.** The diversity jump from T1=0.6→0.8→1.0 is substantial (struct_div 0.097→0.167→0.255), matching the effect of raising temperature in standard P-less. T1 controls pruning aggressiveness by shaping the distribution before the collision entropy threshold is computed. T1 is 2–14× more efficient than T2 at converting pass@1 into diversity.

3. **Best new config: pless T1=0.8 (no T2).** At 58.7% pass@1, 0.167 struct_div, 0.298 codebleu_div, it matches top_p0.95/t=0.2 (58.2%) while providing a useful diversity level — with zero hyperparameters beyond T1.

4. **Implication for instruct models:** T2 alone cannot rescue diversity on instruct models where P-less at low T1 leaves ~1 survivor. The instruct experiment needs high T1 (>1.0) to open the distribution first. However, the T1=0.8/T2=5.0 anomaly suggests T2 may have a narrow sweet spot when the survivor set is moderate — worth testing at high T1 on instruct where more survivors exist.

