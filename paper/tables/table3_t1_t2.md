# Table 3 — T1/T2 grid on Qwen2.5-Coder-7B-Instruct (MBPP-500)

P-less with two-stage temperature: T₁ scales logits before the collision-entropy
threshold; T₂ flattens the survivor distribution after pruning.

## T1 sweep (T2=∅)

| T1 | pass@1 (%) | pass@10 (%) | struct_div | codebleu_div | cover@0.7 (%) |
|:---|----------:|-----------:|-----------:|-------------:|--------------:|
| 0.6 | 77.2 | 79.8 | 0.0305 | 0.0808 | 76.0 |
| 1.0 | 77.2 | 82.2 | 0.0586 | 0.1359 | 75.0 |
| 1.5 | 76.7 | 85.8 | 0.1262 | 0.2792 | 74.0 |
| 2.0 | 72.5 | 89.6 | 0.3082 | 0.5587 | 70.4 |
| 3.0 |  2.7 | 18.8 | 0.2645 | 0.4128 |  0.0 |

_Source: `results/full_mbpp_pre_post_temp_pless/analysis/Qwen--Qwen2.5-Coder-7B-Instruct/instruct_t1_comparison_report.md` lines 37–43._

## T2 sweep at T1=1.0 (instruct)

Baseline (T2=∅): pass@1 = 77.2%, struct_div = 0.0586.

| T2 | pass@1 (%) | Δ pass@1 | struct_div | Δ struct_div |
|:---|----------:|---------:|-----------:|-------------:|
| 2.0 | 77.9 | +0.7 | 0.0572 | -0.0014 |
| 3.0 | 77.8 | +0.6 | 0.0581 | -0.0005 |
| 4.0 | 77.9 | +0.7 | 0.0555 | -0.0031 |
| 5.0 | 77.6 | +0.4 | 0.0562 | -0.0024 |

_Source: `instruct_t1_comparison_report.md:60-64`._

## T2 sweep at T1=2.0 (instruct)

Baseline (T2=∅): pass@1 = 72.5%, struct_div = 0.3082.

| T2 | pass@1 (%) | Δ pass@1 | struct_div | Δ struct_div |
|:---|----------:|---------:|-----------:|-------------:|
| 2.0 | 70.1 | -2.4 | 0.3278 | +0.0196 |
| 3.0 | 70.0 | -2.5 | 0.3422 | +0.0340 |
| 4.0 | 70.5 | -2.0 | 0.3384 | +0.0302 |
| 5.0 | 68.5 | -4.0 | 0.3388 | +0.0306 |

_Source: `instruct_t1_comparison_report.md:69-75`._

## Matched-diversity comparison (P-less vs temperature)

| P-less config | pless pass@1 | pless struct_div | nearest temp | temp pass@1 | temp struct_div | Δ pass@1 |
|:--------------|:------------:|:----------------:|:-------------|:-----------:|:---------------:|:--------:|
| pless T1=1.5  | 76.7% | 0.1262 | temp t=0.2 | 76.8% | 0.0982 | -0.1pp |
| pless T1=2.0  | 72.5% | 0.3082 | temp t=0.8 | 72.0% | 0.2779 | +0.4pp |
| pless T1=3.0  |  2.7% | 0.2645 | temp t=1.5 | 12.3% | 0.2741 | -9.5pp |

_Source: `instruct_t1_comparison_report.md:49-53,91`._

**Interpretation tag (for §4.4):** at matched diversity the Δ pass@1 ranges from
-9.5pp to +0.4pp; the quality-filter hypothesis is not confirmed (claim **C11**).
T2 with T1=2.0 buys 0.020–0.034 struct_div for 2.0–4.0pp pass@1 (claim **C7**),
which is dominated by simply increasing T1 instead.
