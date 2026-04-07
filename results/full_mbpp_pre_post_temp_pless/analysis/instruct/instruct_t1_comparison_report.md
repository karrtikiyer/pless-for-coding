# P-less High-T1 on Instruct Model: MBPP Analysis

**Model:** Qwen2.5-Coder-3B-Instruct | **Dataset:** MBPP-full (500 tasks × 10 samples) | **Prompt:** Chat template (auto-detected)

**Hypothesis:** At matched diversity levels, high-T1/P-less achieves higher pass@1 than plain temperature on instruct models — P-less acts as a quality filter.

This experiment tests whether high T1 (>1.0) can open the peaked instruct distribution enough for P-less to provide value.

## Full Metrics Comparison

| # | Config | Group | T1 | T2 | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.7 | struct_div | codebleu_div |
|---|--------|-------|----|----|--------|--------|--------|---------|-----------|------------|--------------|
| 1 | greedy t=1.0 | **instruct** | 1.0 | — | 62.6 | — | — | — | 62.6 | 0.0000 | 0.0000 |
| 2 | pless t=0.6 | **instruct** | 0.6 | — | 62.5 | 64.2 | 64.8 | 65.4 | 61.8 | 0.0345 | 0.0844 |
| 3 | top_p0.95 t=0.2 | **instruct** | 0.2 | — | 62.5 | 67.7 | 69.5 | 71.6 | 60.0 | 0.1380 | 0.3047 |
| 4 | pless t=1.0 | **instruct** | 1.0 | — | 62.3 | 66.4 | 67.9 | 69.6 | 59.4 | 0.1078 | 0.2301 |
| 5 | temp t=0.2 | **instruct** | 0.2 | — | 62.2 | 68.3 | 70.8 | 74.2 | 60.2 | 0.1411 | 0.3117 |
| 6 | pless t=1.5 | **instruct** | 1.5 | — | 61.2 | 70.4 | 73.6 | 77.2 | 56.6 | 0.2331 | 0.4440 |
| 7 | beam4 t=1.0 | base | 1.0 | — | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 0.0000 | 0.0000 |
| 8 | greedy t=1.0 | base | 1.0 | — | 60.2 | 60.2 | 60.2 | 60.2 | 60.2 | 0.0000 | 0.0000 |
| 9 | temp t=0.6 | **instruct** | 0.6 | — | 60.0 | 71.8 | 75.4 | 79.6 | 56.6 | 0.3588 | 0.5543 |
| 10 | pless_norm t=0.6 | base | 0.6 | — | 59.4 | 63.7 | 65.2 | 66.6 | 57.2 | 0.0957 | 0.2040 |
| 11 | pless t=0.6 | base | 0.6 | — | 59.3 | 63.4 | 64.8 | 66.2 | 57.2 | 0.0968 | 0.2033 |
| 12 | beam8 t=1.0 | base | 1.0 | — | 59.2 | 59.2 | 59.2 | 59.2 | 59.2 | 0.0000 | 0.0000 |
| 13 | pless_norm t=0.7 | base | 0.7 | — | 59.0 | 65.0 | 66.7 | 68.0 | 55.2 | 0.1365 | 0.2666 |
| 14 | pless t=0.7 | base | 0.7 | — | 58.6 | 64.5 | 66.2 | 67.4 | 55.4 | 0.1318 | 0.2585 |
| 15 | top_p0.95 t=0.2 | base | 0.2 | — | 58.2 | 66.6 | 69.2 | 71.6 | 54.4 | 0.1832 | 0.3871 |
| 16 | pless_norm t=1.0 | base | 1.0 | — | 57.6 | 66.9 | 69.8 | 72.4 | 53.4 | 0.2497 | 0.4335 |
| 17 | pless t=1.0 | base | 1.0 | — | 56.5 | 66.4 | 69.4 | 72.2 | 51.6 | 0.2578 | 0.4498 |
| 18 | temp t=0.8 | **instruct** | 0.8 | — | 56.4 | 71.8 | 76.4 | 81.4 | 52.6 | 0.4294 | 0.6176 |
| 19 | pless t=2.0 | **instruct** | 2.0 | — | 56.0 | 72.1 | 77.0 | 82.2 | 49.6 | 0.4156 | 0.6531 |
| 20 | pless T1=2.0 T2=2.0 | **instruct** | 2.0 | 2.0 | 54.1 | 70.6 | 75.7 | 80.8 | 46.8 | 0.4473 | 0.6636 |
| 21 | pless T1=2.0 T2=5.0 | **instruct** | 2.0 | 5.0 | 51.1 | 68.0 | 73.6 | 79.2 | 45.0 | 0.4558 | 0.6664 |
| 22 | temp t=0.7 | base | 0.7 | — | 42.6 | 63.9 | 70.6 | 77.6 | 34.2 | 0.5365 | 0.6903 |
| 23 | top_p0.9 t=1.0 | base | 1.0 | — | 34.7 | 58.4 | 67.3 | 77.0 | 20.6 | 0.5886 | 0.7167 |
| 24 | temp t=1.5 | **instruct** | 1.5 | — | 4.5 | 11.6 | 16.9 | 25.6 | 0.0 | 0.4472 | 0.6242 |
| 25 | pless t=3.0 | **instruct** | 3.0 | — | 0.2 | 0.7 | 1.2 | 2.2 | 0.0 | 0.9024 | 0.6985 |

## T1 Sweep: Does High T1 Restore Diversity on Instruct?

The core question: at what T1 does P-less produce non-zero diversity on the instruct model?

| T1 | pass@1 | pass@10 | struct_div | codebleu_div | cover@0.7 |
|----|--------|---------|------------|--------------|-----------|
| 0.6 | 62.5% | 65.4% | 0.0345 | 0.0844 | 61.8 |
| 1.0 | 62.3% | 69.6% | 0.1078 | 0.2301 | 59.4 |
| 1.5 | 61.2% | 77.2% | 0.2331 | 0.4440 | 56.6 |
| 2.0 | 56.0% | 82.2% | 0.4156 | 0.6531 | 49.6 |
| 3.0 | 0.2% | 2.2% | 0.9024 | 0.6985 | 0.0 |

## P-less as Quality Filter: Matched-Diversity Comparison

Compare P-less (T1=X) against temperature (T=Y) at similar diversity levels. If P-less achieves higher pass@1, it is acting as a quality filter.

| P-less config | P-less pass@1 | P-less sdiv | Nearest temp | Temp pass@1 | Temp sdiv | Δ pass@1 |
|---------------|---------------|-------------|--------------|-------------|-----------|----------|
| pless T1=1.5 | 61.2% | 0.2331 | temp t=0.2 | 62.2% | 0.1411 | -1.0pp |
| pless T1=2.0 | 56.0% | 0.4156 | temp t=0.8 | 56.4% | 0.4294 | -0.4pp |
| pless T1=3.0 | 0.2% | 0.9024 | temp t=1.5 | 4.5% | 0.4472 | -4.3pp |

## T2 Effect at T1=2.0

**Baseline:** pless T1=2.0 (no T2): pass@1=56.0%, struct_div=0.4156

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 54.1% | -1.9pp | 0.4473 | +0.0317 | 0.6636 | +0.0105 |
| 5.0 | 51.1% | -4.9pp | 0.4558 | +0.0402 | 0.6664 | +0.0133 |

## Industry Comparison

**Best P-less config (with diversity):** pless t=0.6 (pass@1=62.5%, struct_div=0.0345)

| Config | pass@1 | struct_div | codebleu_div | Δ pass@1 vs best P-less |
|--------|--------|------------|--------------|------------------------|
| greedy | 62.6% | 0.0000 | 0.0000 | +0.1pp |
| temp t=0.2 | 62.2% | 0.1411 | 0.3117 | -0.3pp |
| top_p0.95 t=0.2 | 62.5% | 0.1380 | 0.3047 | -0.0pp |

## Key Findings

1. **High T1 does restore diversity on instruct models — but at a cost.** P-less struct_div rises from 0.0345 (T1=0.6) to 0.4156 (T1=2.0), confirming that high T1 opens the peaked instruct distribution. However, T1=3.0 catastrophically collapses correctness (0.2% pass@1).

2. **The quality-filter hypothesis is NOT confirmed.** At matched diversity, P-less does NOT beat temperature — the Δ pass@1 ranges from -4.3pp to -0.4pp. On instruct models, P-less and temperature perform similarly at similar diversity levels.

3. **P-less T1=1.0 is the instruct sweet spot.** It gives 62.3% pass@1 with 0.1078 struct_div — nearly as good as greedy (62.6%) while providing meaningful variety.

4. **T2 effect at T1=2.0:** T2 values tested: 2.0, 5.0. Best trade-off: T2=2.0 (54.1% pass@1, 0.4473 sdiv). Highest T2=5.0 costs -4.9pp pass@1.

5. **Top instruct pass@1 (62.6% greedy).** Top configs: greedy t=1.0 (62.6%), pless t=0.6 (62.5%), top_p0.95 t=0.2 (62.5%), pless t=1.0 (62.3%), temp t=0.2 (62.2%).

