# Sampler comparison — best config per metric

For each (model, dataset) cell, the config that maxes the named metric is shown, along with its other two metrics for context. Non-stochastic samplers (greedy, beam) excluded — those collapse the diversity metric to 0 by construction.

**Group A** = α-arm samplers (`pless_alpha_a{α}_t1.0`); these are the reference. **Group B** = all other stochastic samplers (pless, pless_norm, temp, top_p, top_k, split, pless_pt, etc.) — the comparison set.

**Scope gaps** to be aware of (limit the comparison):
- Several non-α HumanEval configs come from the older `full_precision_results` format which lacks `codebleu_diversity`. Those cells show `—` in the cb_div column and are excluded from Pareto-dominance checks on the cb_div axis.
- m-a-p OCI-DS-1.3B has no non-α HumanEval results in the included directories. Cell shows only the α-arm rows.
- The `temprature_results` HumanEval directory is OUT of scope per user request; HumanEval T-sweep data for any model is not included.

## CodeLlama-7B-Instruct — HumanEval

Config count: **4 α-arm** + **8 other stochastic**. (Skipped 1 non-stochastic.)

| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |
|---|---|---|---:|---:|---:|---:|
| A (α-arm) | best pass@1 | `pless_alpha_a2.0_t1.0` | 27.74% | 32.32% | 0.0566 | 164 |
| A (α-arm) | best pass@10 | `pless_alpha_a5.0_t1.0` | 24.82% | 46.95% | 0.2804 | 164 |
| A (α-arm) | best codebleu_div | `pless_alpha_a5.0_t1.0` | 24.82% | 46.95% | 0.2804 | 164 |
| B (other stochastic) | best pass@1 | `temp_0.2` | 36.89% | 46.95% | — | 164 |
| B (other stochastic) | best pass@10 | `temp_0.7` | 36.22% | 62.80% | — | 164 |
| B (other stochastic) | best codebleu_div | `top_p0.9_t1.0` | 25.73% | 51.22% | 0.3902 | 164 |

**Pareto-dominance on (pass@10, cb_div):** the following Group-B configs strictly dominate the best Group-A configs (≥ on both, > on at least one). cb_div=`None` entries are excluded from this check.

| Group-B dominator | Group-A dominated | B pass@10 | B cb_div | A pass@10 | A cb_div |
|---|---|---:|---:|---:|---:|
| `top_p0.9_t1.0` | `pless_alpha_a5.0_t1.0` | 51.22% | 0.3902 | 46.95% | 0.2804 |

## CodeLlama-7B-Instruct — MBPP

Config count: **4 α-arm** + **13 other stochastic**. (Skipped 3 non-stochastic.)

| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |
|---|---|---|---:|---:|---:|---:|
| A (α-arm) | best pass@1 | `pless_alpha_a2.0_t1.0` | 41.78% | 44.20% | 0.0677 | 500 |
| A (α-arm) | best pass@10 | `pless_alpha_a5.0_t1.0` | 40.32% | 53.20% | 0.3042 | 500 |
| A (α-arm) | best codebleu_div | `pless_alpha_a5.0_t1.0` | 40.32% | 53.20% | 0.3042 | 500 |
| B (other stochastic) | best pass@1 | `pless_t0.7` | 42.16% | 43.00% | 0.0326 | 500 |
| B (other stochastic) | best pass@10 | `top_k5_t1.0` | 36.56% | 59.60% | 0.4040 | 500 |
| B (other stochastic) | best codebleu_div | `top_k5_t1.0` | 36.56% | 59.60% | 0.4040 | 500 |

**Pareto-dominance on (pass@10, cb_div):** the following Group-B configs strictly dominate the best Group-A configs (≥ on both, > on at least one). cb_div=`None` entries are excluded from this check.

| Group-B dominator | Group-A dominated | B pass@10 | B cb_div | A pass@10 | A cb_div |
|---|---|---:|---:|---:|---:|
| `top_p0.9_t1.0` | `pless_alpha_a5.0_t1.0` | 59.00% | 0.3660 | 53.20% | 0.3042 |
| `top_p0.8_t1.0` | `pless_alpha_a5.0_t1.0` | 56.00% | 0.3436 | 53.20% | 0.3042 |
| `temp_t0.7` | `pless_alpha_a5.0_t1.0` | 55.20% | 0.3619 | 53.20% | 0.3042 |
| `top_k5_t1.0` | `pless_alpha_a5.0_t1.0` | 59.60% | 0.4040 | 53.20% | 0.3042 |

## OpenCodeInterpreter-DS-1.3B — HumanEval

Config count: **4 α-arm** + **0 other stochastic**. (Skipped 0 non-stochastic.)

| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |
|---|---|---|---:|---:|---:|---:|
| A (α-arm) | best pass@1 | `pless_alpha_a2.0_t1.0` | 58.60% | 75.61% | 0.1360 | 164 |
| A (α-arm) | best pass@10 | `pless_alpha_a5.0_t1.0` | 55.61% | 83.54% | 0.2665 | 164 |
| A (α-arm) | best codebleu_div | `pless_alpha_a5.0_t1.0` | 55.61% | 83.54% | 0.2665 | 164 |
| B (other stochastic) | — | _no configs_ | | | | |

**Pareto-dominance on (pass@10, cb_div):** no Group-B config strictly dominates the best α-arm on both axes. (α-arm is on the Pareto frontier.)

## OpenCodeInterpreter-DS-1.3B — MBPP

Config count: **4 α-arm** + **12 other stochastic**. (Skipped 3 non-stochastic.)

| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |
|---|---|---|---:|---:|---:|---:|
| A (α-arm) | best pass@1 | `pless_alpha_a3.0_t1.0` | 48.00% | 65.00% | 0.4371 | 500 |
| A (α-arm) | best pass@10 | `pless_alpha_a5.0_t1.0` | 46.26% | 66.40% | 0.5414 | 500 |
| A (α-arm) | best codebleu_div | `pless_alpha_a5.0_t1.0` | 46.26% | 66.40% | 0.5414 | 500 |
| B (other stochastic) | best pass@1 | `pless_t1.0` | 47.74% | 55.60% | 0.1687 | 500 |
| B (other stochastic) | best pass@10 | `pless_t2.0` | 45.82% | 67.20% | 0.6463 | 500 |
| B (other stochastic) | best codebleu_div | `pless_t2.0` | 45.82% | 67.20% | 0.6463 | 500 |

**Pareto-dominance on (pass@10, cb_div):** the following Group-B configs strictly dominate the best Group-A configs (≥ on both, > on at least one). cb_div=`None` entries are excluded from this check.

| Group-B dominator | Group-A dominated | B pass@10 | B cb_div | A pass@10 | A cb_div |
|---|---|---:|---:|---:|---:|
| `pless_t2.0` | `pless_alpha_a5.0_t1.0` | 67.20% | 0.6463 | 66.40% | 0.5414 |

## Qwen2.5-Coder-7B-Instruct — HumanEval

Config count: **4 α-arm** + **8 other stochastic**. (Skipped 1 non-stochastic.)

| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |
|---|---|---|---:|---:|---:|---:|
| A (α-arm) | best pass@1 | `pless_alpha_a2.0_t1.0` | 87.38% | 89.63% | 0.0396 | 164 |
| A (α-arm) | best pass@10 | `pless_alpha_a5.0_t1.0` | 84.57% | 91.46% | 0.2578 | 164 |
| A (α-arm) | best codebleu_div | `pless_alpha_a5.0_t1.0` | 84.57% | 91.46% | 0.2578 | 164 |
| B (other stochastic) | best pass@1 | `pless_t0.6` | 87.50% | 87.80% | 0.0172 | 164 |
| B (other stochastic) | best pass@10 | `p_less_norm` | 75.18% | 95.12% | — | 164 |
| B (other stochastic) | best codebleu_div | `top_p0.9_t1.0` | 82.44% | 92.68% | 0.3568 | 164 |

**Pareto-dominance on (pass@10, cb_div):** the following Group-B configs strictly dominate the best Group-A configs (≥ on both, > on at least one). cb_div=`None` entries are excluded from this check.

| Group-B dominator | Group-A dominated | B pass@10 | B cb_div | A pass@10 | A cb_div |
|---|---|---:|---:|---:|---:|
| `top_p0.9_t1.0` | `pless_alpha_a5.0_t1.0` | 92.68% | 0.3568 | 91.46% | 0.2578 |

## Qwen2.5-Coder-7B-Instruct — MBPP

Config count: **4 α-arm** + **18 other stochastic**. (Skipped 1 non-stochastic.)

| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |
|---|---|---|---:|---:|---:|---:|
| A (α-arm) | best pass@1 | `pless_alpha_a2.0_t1.0` | 77.08% | 82.00% | 0.1328 | 500 |
| A (α-arm) | best pass@10 | `pless_alpha_a5.0_t1.0` | 75.32% | 88.00% | 0.4257 | 500 |
| A (α-arm) | best codebleu_div | `pless_alpha_a5.0_t1.0` | 75.32% | 88.00% | 0.4257 | 500 |
| B (other stochastic) | best pass@1 | `pless_pt4.0_t1.0` | 77.90% | 82.80% | 0.1295 | 500 |
| B (other stochastic) | best pass@10 | `pless_t2.0` | 72.48% | 89.60% | 0.5587 | 500 |
| B (other stochastic) | best codebleu_div | `pless_pt3.0_t2.0` | 69.96% | 89.00% | 0.5969 | 500 |

**Pareto-dominance on (pass@10, cb_div):** the following Group-B configs strictly dominate the best Group-A configs (≥ on both, > on at least one). cb_div=`None` entries are excluded from this check.

| Group-B dominator | Group-A dominated | B pass@10 | B cb_div | A pass@10 | A cb_div |
|---|---|---:|---:|---:|---:|
| `pless_t2.0` | `pless_alpha_a5.0_t1.0` | 89.60% | 0.5587 | 88.00% | 0.4257 |
| `temp_t0.8` | `pless_alpha_a5.0_t1.0` | 88.80% | 0.5353 | 88.00% | 0.4257 |
| `pless_pt4.0_t2.0` | `pless_alpha_a5.0_t1.0` | 88.80% | 0.5931 | 88.00% | 0.4257 |
| `pless_pt5.0_t2.0` | `pless_alpha_a5.0_t1.0` | 89.00% | 0.5818 | 88.00% | 0.4257 |
| `pless_pt3.0_t2.0` | `pless_alpha_a5.0_t1.0` | 89.00% | 0.5969 | 88.00% | 0.4257 |
| `pless_pt2.0_t2.0` | `pless_alpha_a5.0_t1.0` | 89.00% | 0.5713 | 88.00% | 0.4257 |

