# AlgoSim APPS Report — Qwen2.5-Coder-7B-Instruct α-sweep vs paper reference baselines

Per (source, difficulty) bucket: NAUADC / EA / DA@10 for each entity (a Qwen2.5-Coder-7B-Instruct α-sweep config OR a paper-baseline model re-clustered with our pipeline). Paper-published Table 2 numbers are interleaved as `paper:` rows where known.

**Comparability caveats — read before quoting any number across blocks:**

1. **Sample-budget asymmetry.** Paper-baseline NAUADC is computed over 100 samples/problem; our Qwen2.5-Coder-7B-Instruct α-sweep configs use 10/problem. DA@10 stays directly comparable; NAUADC integrals span k=1..25 on different sample budgets and should be read accordingly.

2. **Sample-filter asymmetry.** Paper baselines were clustered after filtering to functionally-correct samples (`status == "Passed"`). Our Qwen2.5-Coder-7B-Instruct α-sweep configs are clustered without a correctness filter (we don't run APPS execution at algosim-export time). On easy problems with high pass rates this matters little; on competition difficulty, where most Qwen2.5-Coder-7B-Instruct α-sweep samples are broken-in-different-ways, the unfiltered NAUADC inflates because the judge sees those broken samples as distinct "algorithms". The **relative ordering across our 4 configs** remains informative; the **absolute comparison to paper baselines on the same bucket** is only meaningful where pass rates are high enough that filter vs no-filter would converge.

## ATCODER / interview

| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |
|---|---:|---:|---:|---:|
| pless_alpha_a3.0_t1.0 | **1.125** | 1.109 | 1.132 | 53 |
| pless_alpha_a2.5_t1.0 | **1.077** | 1.072 | 1.080 | 50 |
| pless_alpha_a5.0_t1.0 | **1.036** | 1.035 | 1.038 | 52 |
| pless_alpha_a2.0_t1.0 | **1.000** | 1.000 | 1.000 | 41 |

## ATCODER / introductory

| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |
|---|---:|---:|---:|---:|
| pless_alpha_a5.0_t1.0 | **1.116** | 1.092 | 1.129 | 318 |
| pless_alpha_a3.0_t1.0 | **1.095** | 1.068 | 1.105 | 305 |
| pless_alpha_a2.5_t1.0 | **1.066** | 1.054 | 1.072 | 293 |
| pless_alpha_a2.0_t1.0 | **1.031** | 1.025 | 1.034 | 268 |

| Paper Table 2 (their numbers, not re-clustered) | NAUADC |
|---|---:|
| paper: deepseek-coder-33b | 1.780 |
| paper: gpt-4o-2024-08-06 | 1.302 |

## CODEFORCES / interview

| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |
|---|---:|---:|---:|---:|
| pless_alpha_a5.0_t1.0 | **1.083** | 1.068 | 1.090 | 619 |
| pless_alpha_a2.5_t1.0 | **1.062** | 1.051 | 1.068 | 528 |
| pless_alpha_a3.0_t1.0 | **1.052** | 1.040 | 1.057 | 579 |
| pless_alpha_a2.0_t1.0 | **1.025** | 1.020 | 1.027 | 405 |

## CODEFORCES / introductory

| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |
|---|---:|---:|---:|---:|
| pless_alpha_a5.0_t1.0 | **1.082** | 1.070 | 1.087 | 80 |
| pless_alpha_a3.0_t1.0 | **1.068** | 1.060 | 1.072 | 83 |
| pless_alpha_a2.0_t1.0 | **1.052** | 1.044 | 1.058 | 52 |
| pless_alpha_a2.5_t1.0 | **1.035** | 1.021 | 1.040 | 75 |

| Paper Table 2 (their numbers, not re-clustered) | NAUADC |
|---|---:|
| paper: deepseek-coder-33b | 1.952 |
| paper: gpt-4o-2024-08-06 | 1.507 |

