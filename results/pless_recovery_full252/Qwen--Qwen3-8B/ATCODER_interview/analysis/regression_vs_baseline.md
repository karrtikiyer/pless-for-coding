# Recovery variants vs baseline — paired per-problem regression

Baseline: `pless_think_t1.0_t1.0` over 252 problems (n=10).  
Per-problem pass@1 = num_correct/n. Partitions by BASELINE strength: **strong** (≥8/n), **mid** (3–7), **weak** (≤2, the truncation-prone set).

**Regression** = variant num_correct < baseline. **HARD regression** = baseline solved (≥1) → variant 0 (lost the problem entirely).


## pless_alpha_think_t1.0_a3.0_t1.0

Overall pass@1: baseline **0.625** → variant **0.676** (Δ +0.050) over 252 problems.

| baseline partition | #probs | baseline pass@1 | variant pass@1 | Δ |
|---|---|---|---|---|
| strong (≥8) | 133 | 0.947 | 0.955 | +0.008 |
| mid (3–7) | 57 | 0.514 | 0.686 | +0.172 |
| weak (≤2) | 62 | 0.039 | 0.068 | +0.029 |

- improved: **81**  | regressed: **38**  | unchanged: 133
- **HARD regressions** (solved→lost): **8** [1086, 1178, 1370, 2367, 2385, 2389, 2490, 2661]
- gained (lost→solved): 3 [559, 2476, 2523]
- on STRONG problems: 21 regressed, 0 lost entirely (CLEAN — no regression on easy problems)

## pless_alpha_think_t1.0_a4.0_t1.0

Overall pass@1: baseline **0.625** → variant **0.696** (Δ +0.071) over 252 problems.

| baseline partition | #probs | baseline pass@1 | variant pass@1 | Δ |
|---|---|---|---|---|
| strong (≥8) | 133 | 0.947 | 0.963 | +0.017 |
| mid (3–7) | 57 | 0.514 | 0.728 | +0.214 |
| weak (≤2) | 62 | 0.039 | 0.094 | +0.055 |

- improved: **92**  | regressed: **27**  | unchanged: 133
- **HARD regressions** (solved→lost): **6** [2367, 2378, 2388, 2389, 2490, 2644]
- gained (lost→solved): 5 [559, 2476, 2522, 2523, 2656]
- on STRONG problems: 15 regressed, 0 lost entirely (CLEAN — no regression on easy problems)

## pless_alpha_think_t1.0_a5.0_t1.0

Overall pass@1: baseline **0.625** → variant **0.686** (Δ +0.061) over 252 problems.

| baseline partition | #probs | baseline pass@1 | variant pass@1 | Δ |
|---|---|---|---|---|
| strong (≥8) | 133 | 0.947 | 0.962 | +0.016 |
| mid (3–7) | 57 | 0.514 | 0.711 | +0.196 |
| weak (≤2) | 62 | 0.039 | 0.071 | +0.032 |

- improved: **90**  | regressed: **26**  | unchanged: 136
- **HARD regressions** (solved→lost): **6** [2367, 2381, 2388, 2389, 2644, 2661]
- gained (lost→solved): 8 [270, 559, 579, 615, 1718, 2476, 2478, 2523]
- on STRONG problems: 16 regressed, 0 lost entirely (CLEAN — no regression on easy problems)

## pless_think_t2.0_t2.0

Overall pass@1: baseline **0.625** → variant **0.694** (Δ +0.069) over 252 problems.

| baseline partition | #probs | baseline pass@1 | variant pass@1 | Δ |
|---|---|---|---|---|
| strong (≥8) | 133 | 0.947 | 0.967 | +0.020 |
| mid (3–7) | 57 | 0.514 | 0.721 | +0.207 |
| weak (≤2) | 62 | 0.039 | 0.084 | +0.045 |

- improved: **96**  | regressed: **25**  | unchanged: 131
- **HARD regressions** (solved→lost): **6** [1086, 2367, 2382, 2385, 2389, 2644]
- gained (lost→solved): 5 [280, 369, 559, 2476, 2523]
- on STRONG problems: 13 regressed, 0 lost entirely (CLEAN — no regression on easy problems)
