# P-less High-T1 on Instruct Model: MBPP Analysis

**Model:** Qwen/Qwen2.5-Coder-7B-Instruct | **Dataset:** MBPP-full (500 tasks × 10 samples) | **Prompt:** Chat template (auto-detected)

**Hypothesis:** At matched diversity levels, high-T1/P-less achieves higher pass@1 than plain temperature on instruct models — P-less acts as a quality filter.

This experiment tests whether high T1 (>1.0) can open the peaked instruct distribution enough for P-less to provide value.

## Full Metrics Comparison

| # | Config | Group | T1 | T2 | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.7 | struct_div | codebleu_div |
|---|--------|-------|----|----|--------|--------|--------|---------|-----------|------------|--------------|
| 1 | pless T1=1.0 T2=4.0 | **instruct** | 1.0 | 4.0 | 77.9 | 81.0 | 81.9 | 82.8 | 76.8 | 0.0555 | 0.1295 |
| 2 | pless T1=1.0 T2=2.0 | **instruct** | 1.0 | 2.0 | 77.9 | 80.8 | 81.6 | 82.2 | 76.4 | 0.0572 | 0.1345 |
| 3 | pless T1=1.0 T2=3.0 | **instruct** | 1.0 | 3.0 | 77.8 | 80.8 | 81.7 | 82.4 | 76.8 | 0.0581 | 0.1357 |
| 4 | greedy t=1.0 | **instruct** | 1.0 | — | 77.6 | — | — | — | 77.6 | 0.0000 | 0.0000 |
| 5 | pless T1=1.0 T2=5.0 | **instruct** | 1.0 | 5.0 | 77.6 | 80.7 | 81.4 | 82.0 | 75.8 | 0.0562 | 0.1351 |
| 6 | pless t=0.6 | **instruct** | 0.6 | — | 77.2 | 78.7 | 79.3 | 79.8 | 76.0 | 0.0305 | 0.0808 |
| 7 | pless t=1.0 | **instruct** | 1.0 | — | 77.2 | 80.4 | 81.3 | 82.2 | 75.0 | 0.0586 | 0.1359 |
| 8 | top_p0.95 t=0.2 | **instruct** | 0.2 | — | 76.8 | 80.4 | 81.4 | 82.6 | 75.6 | 0.0990 | 0.2424 |
| 9 | temp t=0.2 | **instruct** | 0.2 | — | 76.8 | 81.0 | 82.1 | 83.2 | 75.4 | 0.0982 | 0.2548 |
| 10 | pless t=1.5 | **instruct** | 1.5 | — | 76.7 | 82.5 | 84.3 | 85.8 | 74.0 | 0.1262 | 0.2792 |
| 11 | temp t=0.6 | **instruct** | 0.6 | — | 74.2 | 82.9 | 85.3 | 87.8 | 72.0 | 0.2199 | 0.4628 |
| 12 | pless t=2.0 | **instruct** | 2.0 | — | 72.5 | 83.4 | 86.5 | 89.6 | 70.4 | 0.3082 | 0.5587 |
| 13 | temp t=0.8 | **instruct** | 0.8 | — | 72.0 | 82.8 | 85.8 | 88.8 | 68.0 | 0.2779 | 0.5353 |
| 14 | pless T1=2.0 T2=4.0 | **instruct** | 2.0 | 4.0 | 70.5 | 82.4 | 85.8 | 88.8 | 67.2 | 0.3384 | 0.5931 |
| 15 | pless T1=2.0 T2=2.0 | **instruct** | 2.0 | 2.0 | 70.1 | 82.4 | 85.8 | 89.0 | 66.2 | 0.3278 | 0.5713 |
| 16 | pless T1=2.0 T2=3.0 | **instruct** | 2.0 | 3.0 | 70.0 | 82.0 | 85.5 | 89.0 | 65.8 | 0.3422 | 0.5969 |
| 17 | pless T1=2.0 T2=5.0 | **instruct** | 2.0 | 5.0 | 68.5 | 81.5 | 85.4 | 89.0 | 63.8 | 0.3388 | 0.5818 |
| 18 | beam4 t=1.0 | base | 1.0 | — | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 0.0000 | 0.0000 |
| 19 | greedy t=1.0 | base | 1.0 | — | 60.2 | 60.2 | 60.2 | 60.2 | 60.2 | 0.0000 | 0.0000 |
| 20 | pless_norm t=0.6 | base | 0.6 | — | 59.4 | 63.7 | 65.2 | 66.6 | 57.2 | 0.0957 | 0.2040 |
| 21 | pless t=0.6 | base | 0.6 | — | 59.3 | 63.4 | 64.8 | 66.2 | 57.2 | 0.0968 | 0.2033 |
| 22 | beam8 t=1.0 | base | 1.0 | — | 59.2 | 59.2 | 59.2 | 59.2 | 59.2 | 0.0000 | 0.0000 |
| 23 | pless_norm t=0.7 | base | 0.7 | — | 59.0 | 65.0 | 66.7 | 68.0 | 55.2 | 0.1365 | 0.2666 |
| 24 | pless t=0.7 | base | 0.7 | — | 58.6 | 64.5 | 66.2 | 67.4 | 55.4 | 0.1318 | 0.2585 |
| 25 | top_p0.95 t=0.2 | base | 0.2 | — | 58.2 | 66.6 | 69.2 | 71.6 | 54.4 | 0.1832 | 0.3871 |
| 26 | pless_norm t=1.0 | base | 1.0 | — | 57.6 | 66.9 | 69.8 | 72.4 | 53.4 | 0.2497 | 0.4335 |
| 27 | pless t=1.0 | base | 1.0 | — | 56.5 | 66.4 | 69.4 | 72.2 | 51.6 | 0.2578 | 0.4498 |
| 28 | temp t=0.7 | base | 0.7 | — | 42.6 | 63.9 | 70.6 | 77.6 | 34.2 | 0.5365 | 0.6903 |
| 29 | top_p0.9 t=1.0 | base | 1.0 | — | 34.7 | 58.4 | 67.3 | 77.0 | 20.6 | 0.5886 | 0.7167 |
| 30 | temp t=1.5 | **instruct** | 1.5 | — | 12.3 | 27.8 | 36.9 | 48.2 | 1.2 | 0.2741 | 0.5029 |
| 31 | pless t=3.0 | **instruct** | 3.0 | — | 2.7 | 7.4 | 11.3 | 18.8 | 0.0 | 0.2645 | 0.4128 |

## T1 Sweep: Does High T1 Restore Diversity on Instruct?

The core question: at what T1 does P-less produce non-zero diversity on the instruct model?

| T1 | pass@1 | pass@10 | struct_div | codebleu_div | cover@0.7 |
|----|--------|---------|------------|--------------|-----------|
| 0.6 | 77.2% | 79.8% | 0.0305 | 0.0808 | 76.0 |
| 1.0 | 77.2% | 82.2% | 0.0586 | 0.1359 | 75.0 |
| 1.5 | 76.7% | 85.8% | 0.1262 | 0.2792 | 74.0 |
| 2.0 | 72.5% | 89.6% | 0.3082 | 0.5587 | 70.4 |
| 3.0 | 2.7% | 18.8% | 0.2645 | 0.4128 | 0.0 |

## P-less as Quality Filter: Matched-Diversity Comparison

Compare P-less (T1=X) against temperature (T=Y) at similar diversity levels. If P-less achieves higher pass@1, it is acting as a quality filter.

| P-less config | P-less pass@1 | P-less sdiv | Nearest temp | Temp pass@1 | Temp sdiv | Δ pass@1 |
|---------------|---------------|-------------|--------------|-------------|-----------|----------|
| pless T1=1.5 | 76.7% | 0.1262 | temp t=0.2 | 76.8% | 0.0982 | -0.1pp |
| pless T1=2.0 | 72.5% | 0.3082 | temp t=0.8 | 72.0% | 0.2779 | +0.4pp |
| pless T1=3.0 | 2.7% | 0.2645 | temp t=1.5 | 12.3% | 0.2741 | -9.5pp |

## T2 Effect at T1=1.0

**Baseline:** pless T1=1.0 (no T2): pass@1=77.2%, struct_div=0.0586

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 77.9% | +0.7pp | 0.0572 | -0.0014 | 0.1345 | -0.0014 |
| 3.0 | 77.8% | +0.6pp | 0.0581 | -0.0005 | 0.1357 | -0.0002 |
| 4.0 | 77.9% | +0.7pp | 0.0555 | -0.0031 | 0.1295 | -0.0064 |
| 5.0 | 77.6% | +0.4pp | 0.0562 | -0.0024 | 0.1351 | -0.0008 |

## T2 Effect at T1=2.0

**Baseline:** pless T1=2.0 (no T2): pass@1=72.5%, struct_div=0.3082

| T2 | pass@1 | Δ pass@1 | struct_div | Δ sdiv | codebleu_div | Δ cbdiv |
|----|--------|----------|------------|--------|--------------|---------|
| 2.0 | 70.1% | -2.4pp | 0.3278 | +0.0196 | 0.5713 | +0.0126 |
| 3.0 | 70.0% | -2.5pp | 0.3422 | +0.0340 | 0.5969 | +0.0382 |
| 4.0 | 70.5% | -2.0pp | 0.3384 | +0.0302 | 0.5931 | +0.0344 |
| 5.0 | 68.5% | -4.0pp | 0.3388 | +0.0306 | 0.5818 | +0.0231 |

## Industry Comparison

**Best P-less config (with diversity):** pless t=0.6 (pass@1=77.2%, struct_div=0.0305)

| Config | pass@1 | struct_div | codebleu_div | Δ pass@1 vs best P-less |
|--------|--------|------------|--------------|------------------------|
| greedy | 77.6% | 0.0000 | 0.0000 | +0.4pp |
| temp t=0.2 | 76.8% | 0.0982 | 0.2548 | -0.5pp |
| top_p0.95 t=0.2 | 76.8% | 0.0990 | 0.2424 | -0.5pp |

## Key Findings

1. **High T1 does restore diversity on instruct models — but at a cost.** P-less struct_div rises from 0.0305 (T1=0.6) to 0.3082 (T1=2.0), confirming that high T1 opens the peaked instruct distribution. However, T1=3.0 catastrophically collapses correctness (2.7% pass@1).

2. **The quality-filter hypothesis is NOT confirmed.** At matched diversity, P-less does NOT beat temperature — the Δ pass@1 ranges from -9.5pp to +0.4pp. On instruct models, P-less and temperature perform similarly at similar diversity levels.

3. **P-less T1=1.0 is the instruct sweet spot.** It gives 77.2% pass@1 with 0.0586 struct_div — nearly as good as greedy (77.6%) while providing meaningful variety.

4. **T2 effect at T1=2.0:** T2 values tested: 2.0, 3.0, 4.0, 5.0. Best trade-off: T2=4.0 (70.5% pass@1, 0.3384 sdiv). Highest T2=5.0 costs -4.0pp pass@1.

5. **T2 effect at T1=1.0:** T2 values tested: 2.0, 3.0, 4.0, 5.0. Pass@1 deltas: T2=2.0: +0.7pp, T2=3.0: +0.6pp, T2=4.0: +0.7pp, T2=5.0: +0.4pp. T2 is mostly harmless at T1=1.0.

6. **Top instruct pass@1 (77.6% greedy).** Top configs: pless T1=1.0 T2=4.0 (77.9%), pless T1=1.0 T2=2.0 (77.9%), pless T1=1.0 T2=3.0 (77.8%), greedy t=1.0 (77.6%), pless T1=1.0 T2=5.0 (77.6%).

