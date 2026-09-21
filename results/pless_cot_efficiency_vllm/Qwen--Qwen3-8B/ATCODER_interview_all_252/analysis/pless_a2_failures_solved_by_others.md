# APPS-252 tasks where pless α=2 (T=1.0) failed but another decoder solved

Model: **Qwen3-8B** (think). Dataset: **ATCODER_interview** (full 252 subset).

- **pless α=2 failed** = 0/10 samples pass at `pless@T=1.0`.
- **Non-pless solved** = ≥1/10 samples pass at any of: temp@t1.0, temp@t0.6, temp_k20@t1.0, temp_k20@t0.6, temp_p0.95@t1.0, temp_p0.95@t0.6, temp_p0.95_k20@t0.6, top_p0.7@t1.0, top_p0.75@t1.0, top_p0.8@t1.0, top_p0.85@t1.0, top_p0.9@t1.0.
- Variant flag columns show which pless-family variants also solved (≥1/10). 16 variants tracked.

**Count: 16 tasks** meet the criterion. Of these, 15 are recovered by ≥1 pless-family variant.

## Summary table

| task_id | # non-pless solvers | any pless variant? | pless_norm@t1.0 | pless_norm@t0.6 | pless@t0.6 | pless@t2.0 | tau_alpha_a3 | tau_alpha_a4 | tau_alpha_a5 | G_k_0.05 | G_k_0.1 | G_k_0.2 | G_k_0.25 | G_k_0.3 | G_k_0.35 | G_k_0.4 | G_k_0.8 | G_k_1.6 |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 270 | 6 | yes | 3 | 1 | 1 | · | · | · | 1 | 1 | 1 | 2 | · | · | 3 | · | 1 | · |
| 280 | 1 | yes | · | · | · | 1 | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 369 | 4 | yes | 1 | · | · | 1 | · | · | · | · | 1 | · | 2 | · | · | · | · | · |
| 559 | 11 | yes | 2 | 4 | 4 | 3 | 4 | 4 | 4 | 9 | 10 | 2 | 1 | 3 | 3 | 5 | 5 | · |
| 579 | 2 | yes | · | · | · | · | · | · | 1 | 1 | · | · | · | · | 1 | · | · | · |
| 615 | 2 | yes | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · |
| 1583 | 2 | yes | · | · | 1 | · | · | · | · | · | · | · | 1 | 2 | · | · | · | · |
| 1718 | 3 | yes | 2 | · | · | · | · | · | 1 | · | 1 | 1 | · | · | · | 2 | 2 | · |
| 2306 | 1 | yes | 1 | 1 | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · |
| 2476 | 11 | yes | 2 | 3 | 2 | 4 | 3 | 3 | 1 | 5 | 2 | 3 | 4 | 4 | 1 | 2 | 2 | 2 |
| 2478 | 8 | yes | · | · | · | · | · | · | 2 | 1 | 2 | 1 | 3 | 4 | 1 | 1 | 1 | · |
| 2479 | 1 | — | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 2502 | 3 | yes | · | · | · | · | · | · | · | · | · | · | 1 | · | · | 1 | · | 1 |
| 2522 | 5 | yes | · | · | · | · | · | 1 | · | · | · | 1 | · | 1 | 1 | 1 | · | · |
| 2523 | 9 | yes | 2 | · | · | 2 | 3 | 2 | 3 | 1 | 1 | · | 1 | · | 1 | 2 | 1 | · |
| 2643 | 3 | yes | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 |

Cell values in variant columns = num_correct out of 10 samples (`·` = 0). See `pless_a2_failures_solved_by_others.csv` for the full per-task table incl. the specific non-pless solver names.
