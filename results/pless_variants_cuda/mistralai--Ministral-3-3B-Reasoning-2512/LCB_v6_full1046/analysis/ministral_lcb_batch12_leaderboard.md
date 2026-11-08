# Ministral-3-3B-Reasoning — LiveCodeBench v6, GPU2 batches 1+2: pooled leaderboard

Ranked by pooled pass@1 over all 420 tasks (AtCoder 242 + LeetCode 178); pooled pass@1 = mean over tasks of per-task c/n. Ties broken by pooled pass@10 (desc), then config name (asc). All values in %.

| Rank | Config | Family | T | pass@1 (pooled, N=420) | 95% CI | AtCoder pass@1 | LeetCode pass@1 | pass@10 (pooled) | term% (pooled, sample-weighted) | mean_tok (pooled, sample-weighted) | timeout% (pooled, sample-weighted) |
|---:|:---|:---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | top_p_0_9 | top-p | 1.0 | 67.64 | 63.64–71.45 | 64.59 | 71.80 | 81.90 | 85.2 | 15785 | 4.2 |
| 2 | top_p_0_8 | top-p | 1.0 | 67.57 | 63.62–71.33 | 64.55 | 71.69 | 82.38 | 85.4 | 15841 | 4.1 |
| 3 | pless_topmass | p-less variants | 1.0 | 67.57 | 63.59–71.43 | 64.75 | 71.40 | 82.14 | 84.6 | 16063 | 4.3 |
| 4 | pless_effk_nofloor | p-less variants | 1.0 | 67.52 | 63.45–71.33 | 65.04 | 70.90 | 83.33 | 84.6 | 15892 | 3.8 |
| 5 | pless_gini | p-less variants | 1.0 | 67.48 | 63.45–71.31 | 64.17 | 71.97 | 80.95 | 84.7 | 16050 | 3.9 |
| 6 | top_nsigma_1_5 | top-nsigma | 1.0 | 67.43 | 63.43–71.29 | 64.34 | 71.63 | 80.95 | 85.1 | 15943 | 4.4 |
| 7 | pless_effk | p-less variants | 1.0 | 67.10 | 63.02–70.93 | 64.26 | 70.96 | 80.95 | 83.8 | 16143 | 4.2 |
| 8 | top_p_0_7† | top-p | 1.0 | 67.07 | 63.07–70.93 | 64.30 | 70.84 | 82.14 | 84.0 | 15961 | 4.0 |
| 9 | top_p_0_85 | top-p | 1.0 | 66.93 | 62.93–70.79 | 64.63 | 70.06 | 80.48 | 84.9 | 15763 | 4.3 |
| 10 | ministral_recipe | Ministral recipe (T=0.7, top-p 0.95) | 0.7‡ | 66.83 | 62.90–70.69 | 63.88 | 70.84 | 82.38 | 84.4 | 15893 | 4.1 |
| 11 | gk_0_4 | G_k (Renyi) | 1.0 | 66.40 | 62.40–70.21 | 62.52 | 71.69 | 80.24 | 82.0 | 15954 | 4.1 |
| 12 | top_p_0_95 | top-p | 1.0 | 66.38 | 62.40–70.17 | 63.47 | 70.34 | 82.38 | 83.7 | 15969 | 4.3 |
| 13 | top_p_0_75 | top-p | 1.0 | 66.19 | 62.19–70.00 | 64.01 | 69.16 | 80.95 | 85.0 | 15903 | 4.3 |
| 14 | gk_0_2 | G_k (Renyi) | 1.0 | 66.14 | 62.14–70.02 | 62.85 | 70.62 | 81.67 | 81.3 | 16325 | 4.2 |
| 15 | gk_0_05 | G_k (Renyi) | 1.0 | 65.76 | 61.81–69.67 | 62.81 | 69.78 | 79.29 | 81.4 | 16423 | 4.4 |
| 16 | gk_0_1 | G_k (Renyi) | 1.0 | 65.67 | 61.62–69.50 | 62.11 | 70.51 | 79.76 | 81.1 | 16328 | 4.4 |
| 17 | gk_0_8 | G_k (Renyi) | 1.0 | 63.10 | 59.19–66.88 | 61.07 | 65.84 | 80.48 | 78.6 | 16760 | 3.6 |
| 18 | gk_1_6 | G_k (Renyi) | 1.0 | 60.57 | 56.76–64.19 | 59.34 | 62.25 | 79.52 | 72.8 | 17484 | 3.0 |
| 19 | pless_default | p-less variants | 1.0 | 59.64 | 55.64–63.55 | 58.18 | 61.63 | 78.10 | 71.9 | 17588 | 3.0 |

Caveat: single generation seed per task; the per-config 95% CIs describe each config alone; differences between configs must be judged with the PAIRED task-bootstrap below.

- 95% CI: percentile bootstrap over tasks (2000 resamples with replacement, numpy `default_rng(seed=42)`) of the pooled per-task pass@1 vector; the same seed is used for every config, so all configs share the same task resamples.
- † top_p_0_7: AtCoder metrics file provenance is UNVERIFIED (could not confirm from files that it was generated with the post-fix nucleus sampler; user accepted this risk on 2026-11-08).
- ‡ ministral_recipe uses T=0.7; all other configs use T=1.0.
- Sampler parameters such as the top-p cutoff are NOT stored in the run records (the records' `top_p` field reads 1.0 for all top-p configs); the cutoff values in the config names come from the runner configuration (`scripts/run_pless_variants_ministral_3b_lcb_cuda.py`); only T is recorded.
- Tie on displayed pooled pass@1: top_p_0_8, pless_topmass (ordered by the pass@10 / name tie-break, not by pass@1).

## Paired difference vs rank 1

delta = pooled pass@1 of rank 1 (top_p_0_9) minus this config, in percentage points; paired 95% percentile bootstrap over the 420 tasks (2000 resamples, seed 42, the same resample index matrix as the per-config CIs).

| Rank | Config | delta pass@1 vs rank 1 (pp) | paired 95% CI (pp) | CI excludes 0? |
|---:|:---|---:|:---:|:---:|
| 2 | top_p_0_8 | 0.07 | -1.02–1.12 | no |
| 3 | pless_topmass | 0.07 | -1.00–1.10 | no |
| 4 | pless_effk_nofloor | 0.12 | -0.93–1.12 | no |
| 5 | pless_gini | 0.17 | -0.88–1.17 | no |
| 6 | top_nsigma_1_5 | 0.21 | -0.98–1.36 | no |
| 7 | pless_effk | 0.55 | -0.55–1.62 | no |
| 8 | top_p_0_7† | 0.57 | -0.60–1.71 | no |
| 9 | top_p_0_85 | 0.71 | -0.33–1.74 | no |
| 10 | ministral_recipe | 0.81 | -0.43–1.90 | no |
| 11 | gk_0_4 | 1.24 | 0.09–2.36 | yes |
| 12 | top_p_0_95 | 1.26 | 0.26–2.29 | yes |
| 13 | top_p_0_75 | 1.45 | 0.21–2.64 | yes |
| 14 | gk_0_2 | 1.50 | 0.24–2.67 | yes |
| 15 | gk_0_05 | 1.88 | 0.79–3.00 | yes |
| 16 | gk_0_1 | 1.98 | 0.93–2.98 | yes |
| 17 | gk_0_8 | 4.55 | 3.19–5.98 | yes |
| 18 | gk_1_6 | 7.07 | 5.50–8.71 | yes |
| 19 | pless_default | 8.00 | 6.36–9.79 | yes |

18 comparisons, no multiplicity correction; borderline cases (a CI bound close to 0) are not solid.
Ranks not distinguishable from rank 1 (paired 95% CI includes 0): 2 (top_p_0_8), 3 (pless_topmass), 4 (pless_effk_nofloor), 5 (pless_gini), 6 (top_nsigma_1_5), 7 (pless_effk), 8 (top_p_0_7), 9 (top_p_0_85), 10 (ministral_recipe).

## Canonical tables (pooled, then per platform)

Generated with `bench.eval.headline_table.headline_table(..., columns=CANONICAL_COLUMNS)`; rows in the pooled-ranking order. struct_div shows — because it was intentionally skipped (`--skip-structural-diversity`). Pooled table: pass@k, cov@t, cov@t distinct are task-weighted over all 420 tasks (each task judged against its own platform's n); cb_div is the mean per-task self-CodeBLEU diversity over the pooled tasks with >=2 correct samples (the same task set `bench.eval` averages over per platform), so it is not a task-count blend of the two platform values; term%, mean_tok, timeout% are sample-weighted; struct_div is not pooled (skipped).

### Pooled (AtCoder + LeetCode, N=420)

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top_p_0_9 | 420 | 67.64% | 76.31% | 78.91% | 80.98% | 81.90% | 74.8% | 70.5% | 65.5% | 72.6% | 62.1% | 48.3% | 85.2% | 15785 | 4.2% | — | 0.5000 |
| top_p_0_8 | 420 | 67.57% | 76.78% | 79.59% | 81.63% | 82.38% | 75.5% | 70.0% | 65.0% | 72.4% | 59.5% | 46.0% | 85.4% | 15841 | 4.1% | — | 0.4847 |
| pless_topmass | 420 | 67.57% | 76.28% | 79.08% | 81.26% | 82.14% | 74.8% | 70.0% | 65.0% | 73.3% | 60.7% | 47.6% | 84.6% | 16063 | 4.3% | — | 0.5017 |
| pless_effk_nofloor | 420 | 67.52% | 76.88% | 79.98% | 82.35% | 83.33% | 75.5% | 69.3% | 65.0% | 73.1% | 58.6% | 46.0% | 84.6% | 15892 | 3.8% | — | 0.4858 |
| pless_gini | 420 | 67.48% | 75.89% | 78.32% | 80.21% | 80.95% | 74.3% | 70.7% | 65.7% | 71.9% | 61.4% | 47.6% | 84.7% | 16050 | 3.9% | — | 0.4890 |
| top_nsigma_1_5 | 420 | 67.43% | 76.07% | 78.53% | 80.28% | 80.95% | 75.2% | 70.0% | 65.7% | 72.6% | 60.5% | 46.9% | 85.1% | 15943 | 4.4% | — | 0.4914 |
| pless_effk | 420 | 67.10% | 75.59% | 78.27% | 80.24% | 80.95% | 73.8% | 69.3% | 64.8% | 71.7% | 60.2% | 49.3% | 83.8% | 16143 | 4.2% | — | 0.5000 |
| top_p_0_7† | 420 | 67.07% | 76.62% | 79.52% | 81.52% | 82.14% | 75.0% | 69.3% | 64.3% | 71.4% | 58.1% | 44.5% | 84.0% | 15961 | 4.0% | — | 0.4746 |
| top_p_0_85 | 420 | 66.93% | 75.72% | 78.23% | 79.86% | 80.48% | 75.7% | 69.0% | 64.8% | 72.1% | 58.6% | 45.7% | 84.9% | 15763 | 4.3% | — | 0.4910 |
| ministral_recipe | 420 | 66.83% | 76.29% | 79.00% | 81.28% | 82.38% | 74.8% | 70.7% | 64.3% | 71.4% | 59.0% | 45.0% | 84.4% | 15893 | 4.1% | — | 0.4877 |
| gk_0_4 | 420 | 66.40% | 74.95% | 77.53% | 79.47% | 80.24% | 74.0% | 68.3% | 64.8% | 72.6% | 59.8% | 49.3% | 82.0% | 15954 | 4.1% | — | 0.5071 |
| top_p_0_95 | 420 | 66.38% | 76.06% | 79.09% | 81.41% | 82.38% | 74.8% | 69.3% | 63.3% | 73.6% | 61.2% | 46.0% | 83.7% | 15969 | 4.3% | — | 0.5059 |
| top_p_0_75 | 420 | 66.19% | 75.80% | 78.46% | 80.27% | 80.95% | 75.0% | 69.8% | 63.6% | 71.7% | 58.3% | 42.9% | 85.0% | 15903 | 4.3% | — | 0.4782 |
| gk_0_2 | 420 | 66.14% | 75.05% | 78.02% | 80.54% | 81.67% | 72.9% | 68.8% | 63.3% | 71.0% | 61.9% | 48.1% | 81.3% | 16325 | 4.2% | — | 0.5135 |
| gk_0_05 | 420 | 65.76% | 74.66% | 77.03% | 78.63% | 79.29% | 74.5% | 68.6% | 64.0% | 73.1% | 61.7% | 46.4% | 81.4% | 16423 | 4.4% | — | 0.5175 |
| gk_0_1 | 420 | 65.67% | 75.10% | 77.64% | 79.26% | 79.76% | 74.3% | 69.0% | 62.9% | 73.1% | 62.9% | 46.9% | 81.1% | 16328 | 4.4% | — | 0.5168 |
| gk_0_8 | 420 | 63.10% | 74.46% | 77.43% | 79.60% | 80.48% | 73.3% | 67.6% | 61.2% | 66.0% | 52.4% | 37.1% | 78.6% | 16760 | 3.6% | — | 0.4553 |
| gk_1_6 | 420 | 60.57% | 73.08% | 76.60% | 78.87% | 79.52% | 71.2% | 65.0% | 56.7% | 64.0% | 48.8% | 30.2% | 72.8% | 17484 | 3.0% | — | 0.4438 |
| pless_default | 420 | 59.64% | 71.88% | 74.98% | 77.14% | 78.10% | 71.4% | 64.5% | 57.4% | 64.0% | 47.9% | 30.5% | 71.9% | 17588 | 3.0% | — | 0.4447 |

### AtCoder

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top_p_0_9 | 242 | 64.59% | 71.80% | 74.25% | 76.31% | 77.27% | 70.2% | 66.1% | 62.0% | 68.6% | 59.9% | 47.5% | 82.5% | 16538 | 3.4% | — | 0.5203 |
| top_p_0_8 | 242 | 64.55% | 72.67% | 75.41% | 77.48% | 78.10% | 70.2% | 66.1% | 62.8% | 66.9% | 57.4% | 46.7% | 83.2% | 16463 | 3.8% | — | 0.5036 |
| pless_topmass | 242 | 64.75% | 72.19% | 74.65% | 76.72% | 77.69% | 70.7% | 67.4% | 61.6% | 69.4% | 57.9% | 46.7% | 81.9% | 16634 | 3.8% | — | 0.5237 |
| pless_effk_nofloor | 242 | 65.04% | 73.33% | 75.90% | 77.79% | 78.51% | 72.3% | 66.5% | 63.6% | 71.1% | 57.0% | 46.3% | 82.8% | 16363 | 3.4% | — | 0.5072 |
| pless_gini | 242 | 64.17% | 71.46% | 73.65% | 75.33% | 76.03% | 70.7% | 66.9% | 61.6% | 69.0% | 59.5% | 46.3% | 82.1% | 16723 | 3.6% | — | 0.5131 |
| top_nsigma_1_5 | 242 | 64.34% | 71.93% | 74.20% | 75.81% | 76.45% | 71.1% | 66.1% | 62.4% | 69.8% | 58.7% | 46.3% | 82.1% | 16625 | 3.8% | — | 0.5199 |
| pless_effk | 242 | 64.26% | 71.63% | 73.98% | 75.73% | 76.45% | 70.7% | 66.5% | 61.6% | 69.0% | 59.5% | 48.8% | 80.7% | 16812 | 3.7% | — | 0.5300 |
| top_p_0_7† | 242 | 64.30% | 72.94% | 75.61% | 77.26% | 77.69% | 71.9% | 65.7% | 60.7% | 69.4% | 56.6% | 42.6% | 82.5% | 16467 | 3.8% | — | 0.4931 |
| top_p_0_85 | 242 | 64.63% | 72.64% | 74.88% | 76.09% | 76.45% | 73.6% | 65.7% | 62.8% | 70.7% | 57.4% | 46.3% | 83.1% | 16352 | 3.8% | — | 0.5146 |
| ministral_recipe | 242 | 63.88% | 71.57% | 73.89% | 75.90% | 76.86% | 70.2% | 66.9% | 61.6% | 67.4% | 57.4% | 44.6% | 81.9% | 16479 | 3.6% | — | 0.5084 |
| gk_0_4 | 242 | 62.52% | 70.88% | 73.47% | 75.32% | 76.03% | 70.2% | 64.5% | 59.5% | 69.4% | 58.3% | 46.3% | 78.3% | 16491 | 3.9% | — | 0.5336 |
| top_p_0_95 | 242 | 63.47% | 71.95% | 74.95% | 77.44% | 78.51% | 69.8% | 65.7% | 60.3% | 68.6% | 59.1% | 47.5% | 80.2% | 16702 | 3.3% | — | 0.5298 |
| top_p_0_75 | 242 | 64.01% | 72.37% | 74.71% | 76.01% | 76.45% | 73.1% | 66.1% | 60.7% | 70.7% | 56.2% | 43.8% | 83.4% | 16416 | 4.1% | — | 0.4991 |
| gk_0_2 | 242 | 62.85% | 70.50% | 72.98% | 75.07% | 76.03% | 69.0% | 64.9% | 60.7% | 67.8% | 59.1% | 48.3% | 77.5% | 16898 | 3.5% | — | 0.5404 |
| gk_0_05 | 242 | 62.81% | 70.52% | 72.44% | 73.77% | 74.38% | 70.2% | 65.7% | 61.2% | 69.4% | 58.7% | 46.7% | 77.4% | 16978 | 3.4% | — | 0.5461 |
| gk_0_1 | 242 | 62.11% | 70.31% | 72.46% | 73.90% | 74.38% | 69.4% | 65.7% | 60.3% | 68.6% | 59.5% | 46.7% | 76.3% | 16897 | 3.9% | — | 0.5419 |
| gk_0_8 | 242 | 61.07% | 71.22% | 73.83% | 75.71% | 76.45% | 69.8% | 64.9% | 59.9% | 63.6% | 52.1% | 38.8% | 78.7% | 17035 | 3.6% | — | 0.4751 |
| gk_1_6 | 242 | 59.34% | 69.39% | 72.00% | 73.79% | 74.38% | 67.8% | 64.0% | 58.3% | 61.2% | 48.8% | 33.9% | 73.3% | 17681 | 2.9% | — | 0.4516 |
| pless_default | 242 | 58.18% | 69.18% | 71.93% | 73.67% | 74.38% | 69.0% | 62.4% | 55.0% | 61.2% | 46.7% | 29.8% | 72.4% | 17756 | 3.1% | — | 0.4636 |

### LeetCode

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top_p_0_9 | 178 | 71.80% | 82.43% | 85.26% | 87.34% | 88.20% | 80.9% | 76.4% | 70.2% | 78.1% | 65.2% | 49.4% | 88.9% | 14762 | 5.3% | — | 0.4762 |
| top_p_0_8 | 178 | 71.69% | 82.37% | 85.26% | 87.28% | 88.20% | 82.6% | 75.3% | 68.0% | 79.8% | 62.4% | 44.9% | 88.3% | 14995 | 4.5% | — | 0.4615 |
| pless_topmass | 178 | 71.40% | 81.84% | 85.11% | 87.43% | 88.20% | 80.3% | 73.6% | 69.7% | 78.7% | 64.6% | 48.9% | 88.4% | 15286 | 5.0% | — | 0.4758 |
| pless_effk_nofloor | 178 | 70.90% | 81.72% | 85.52% | 88.56% | 89.89% | 79.8% | 73.0% | 66.9% | 75.8% | 60.7% | 45.5% | 87.0% | 15251 | 4.3% | — | 0.4596 |
| pless_gini | 178 | 71.97% | 81.93% | 84.68% | 86.85% | 87.64% | 79.2% | 75.8% | 71.3% | 75.8% | 64.0% | 49.4% | 88.3% | 15134 | 4.3% | — | 0.4606 |
| top_nsigma_1_5 | 178 | 71.63% | 81.69% | 84.42% | 86.34% | 87.08% | 80.9% | 75.3% | 70.2% | 76.4% | 62.9% | 47.8% | 89.2% | 15016 | 5.1% | — | 0.4573 |
| pless_effk | 178 | 70.96% | 80.97% | 84.10% | 86.38% | 87.08% | 78.1% | 73.0% | 69.1% | 75.3% | 61.2% | 50.0% | 88.1% | 15234 | 4.8% | — | 0.4645 |
| top_p_0_7 | 178 | 70.84% | 81.61% | 84.82% | 87.30% | 88.20% | 79.2% | 74.2% | 69.1% | 74.2% | 60.1% | 47.2% | 86.0% | 15274 | 4.3% | — | 0.4520 |
| top_p_0_85 | 178 | 70.06% | 79.91% | 82.78% | 84.99% | 85.96% | 78.7% | 73.6% | 67.4% | 74.2% | 60.1% | 44.9% | 87.2% | 14962 | 4.9% | — | 0.4615 |
| ministral_recipe | 178 | 70.84% | 82.72% | 85.94% | 88.59% | 89.89% | 80.9% | 75.8% | 68.0% | 77.0% | 61.2% | 45.5% | 87.9% | 15097 | 4.7% | — | 0.4635 |
| gk_0_4 | 178 | 71.69% | 80.49% | 83.06% | 85.11% | 85.96% | 79.2% | 73.6% | 71.9% | 77.0% | 61.8% | 53.4% | 87.1% | 15225 | 4.4% | — | 0.4751 |
| top_p_0_95 | 178 | 70.34% | 81.65% | 84.72% | 86.80% | 87.64% | 81.5% | 74.2% | 67.4% | 80.3% | 64.0% | 43.8% | 88.4% | 14973 | 5.7% | — | 0.4773 |
| top_p_0_75 | 178 | 69.16% | 80.46% | 83.56% | 86.07% | 87.08% | 77.5% | 74.7% | 67.4% | 73.0% | 61.2% | 41.6% | 87.2% | 15206 | 4.6% | — | 0.4526 |
| gk_0_2 | 178 | 70.62% | 81.25% | 84.87% | 87.98% | 89.33% | 78.1% | 74.2% | 66.9% | 75.3% | 65.7% | 47.8% | 86.6% | 15545 | 5.1% | — | 0.4820 |
| gk_0_05 | 178 | 69.78% | 80.29% | 83.27% | 85.23% | 85.96% | 80.3% | 72.5% | 68.0% | 78.1% | 65.7% | 46.1% | 86.9% | 15669 | 5.7% | — | 0.4839 |
| gk_0_1 | 178 | 70.51% | 81.62% | 84.70% | 86.54% | 87.08% | 80.9% | 73.6% | 66.3% | 79.2% | 67.4% | 47.2% | 87.7% | 15553 | 5.1% | — | 0.4876 |
| gk_0_8 | 178 | 65.84% | 78.88% | 82.33% | 84.88% | 85.96% | 78.1% | 71.3% | 62.9% | 69.1% | 52.8% | 34.8% | 78.4% | 16387 | 3.7% | — | 0.4311 |
| gk_1_6 | 178 | 62.25% | 78.08% | 82.85% | 85.78% | 86.52% | 75.8% | 66.3% | 54.5% | 68.0% | 48.9% | 25.3% | 72.2% | 17215 | 3.1% | — | 0.4346 |
| pless_default | 178 | 61.63% | 75.55% | 79.13% | 81.86% | 83.15% | 74.7% | 67.4% | 60.7% | 68.0% | 49.4% | 31.5% | 71.1% | 17361 | 2.9% | — | 0.4210 |

(† in the pooled and AtCoder tables marks top_p_0_7, whose AtCoder metrics file provenance is unverified as noted above; its LeetCode file is not affected and the LeetCode table carries no †.)

## Provenance

1. Data = GPU2 batches 1+2 only, 242 AtCoder + 178 LeetCode tasks = 420/1046; batches 0/3/4 not included.
2. 38 metrics files generated by `python -m bench.eval --dataset livecodebench --workers 4 --skip-structural-diversity` (6.0s per-test timeout), re-evaluated locally 2026-11-07/08.
3. results-validator verdict: PASS with one flag.
4. FLAG: top_p_0_7 AtCoder file provenance is UNVERIFIED (could not confirm from files that it was generated with the post-fix nucleus sampler; user accepted this risk on 2026-11-08); the top_p_0_7 row is marked †.
5. Excluded: legacy partial top_nsigma (n=1.0) AtCoder file (111/242 tasks), no LeetCode counterpart.
6. Not included in this table: Mac mlx gk_0_025 run and MSI batch-0 runs.
7. ministral_recipe uses T=0.7, all others T=1.0.

Generated by `scripts/ministral_lcb_leaderboard.py` (script git status at generation: untracked; repo HEAD at generation: ce09f99) from `results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/metrics`.
