# AlgoSim APPS Report — Qwen3-8B vs paper reference baselines

Per (source, difficulty) bucket: NAUADC / EA / DA@10 for each entity (our Qwen3-8B config OR a paper-baseline model re-clustered with our pipeline). Paper-published Table 2 numbers are interleaved as `paper:` rows where known.

**Comparability caveats — read before quoting any number across blocks:**

1. **Sample-budget asymmetry.** Paper-baseline NAUADC is computed over 100 samples/problem; our Qwen3-8B configs use 10/problem. DA@10 stays directly comparable; NAUADC integrals span k=1..25 on different sample budgets and should be read accordingly.

2. **Sample-filter asymmetry.** Paper baselines were clustered after filtering to functionally-correct samples (`status == "Passed"`). Our Qwen3-8B configs are clustered without a correctness filter (we don't run APPS execution at algosim-export time). On easy problems with high pass rates this matters little; on competition difficulty, where most Qwen3-8B samples are broken-in-different-ways, the unfiltered NAUADC inflates because the judge sees those broken samples as distinct "algorithms". The **relative ordering across our 6 configs** remains informative; the **absolute comparison to paper baselines on the same bucket** is only meaningful where pass rates are high enough that filter vs no-filter would converge.

## ATCODER / competition

| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |
|---|---:|---:|---:|---:|
| H8P | **4.666** | 4.640 | 5.317 | 41 |
| H7P | **4.555** | 4.481 | 5.195 | 41 |
| H9P | **4.482** | 4.452 | 5.098 | 41 |
| T15N | **4.427** | 4.368 | 5.024 | 41 |
| T15P | **4.352** | 4.288 | 4.927 | 41 |
| P15 | **4.029** | 3.921 | 4.537 | 41 |
| deepseek-coder-6.7b-instruct | **1.630** | 1.609 | 1.667 | 3 |
| deepseek-coder-33b-instruct (AWQ) | **1.000** | 1.000 | 1.000 | 4 |
| deepseek-coder-6.7b-base | **1.000** | 1.000 | 1.000 | 3 |

## CODEFORCES / competition

| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |
|---|---:|---:|---:|---:|
| deepseek-coder-33b-instruct (AWQ) | **1.984** | 1.788 | 1.992 | 102 |
| deepseek-coder-6.7b-instruct | **1.578** | 1.487 | 1.602 | 71 |
| deepseek-coder-6.7b-base | **1.399** | 1.373 | 1.428 | 53 |

