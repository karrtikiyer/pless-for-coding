# gk_0_45 local eval + 420-task pooling (Ministral-3-3B-Reasoning-2512, LCB v6, Renyi G_k k=0.45, T=1.0, n=10)

Generated 2026-11-13 IST on the local Mac. Every number below is copied verbatim from `/tmp/eval_gk045/pool_output.txt` (the stdout of `pool_and_bootstrap.py`) or from the eval logs.

## Inputs / copies
- /tmp/gk045_validate/gk_0_45_{atcoder,leetcode}_think_t1.0.jsonl were copied into results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/. Neither target existed beforehand. sha256 of both copies matches (atcoder 012e8af9...7934c2, leetcode f1c4d03e...5fd9).

## Commands (repo root, run sequentially by /tmp/eval_gk045/run_evals.sh under nohup)
    R=results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046
    uv run python -m bench.eval --results-file $R/gk_0_45_atcoder_think_t1.0.jsonl  --dataset livecodebench --workers 4 --skip-structural-diversity
    uv run python -m bench.eval --results-file $R/gk_0_45_leetcode_think_t1.0.jsonl --dataset livecodebench --workers 4 --skip-structural-diversity
    uv run python /tmp/eval_gk045/pool_and_bootstrap.py   # pooling + bootstrap (imports scripts/ministral_lcb_leaderboard.py)

## Timings (from the START/END stamps in the logs)
- AtCoder: 2026-11-13 14:56:53 to 15:16:08 IST = 1155 s (19 min 15 s), rc=0. Log: /tmp/eval_gk045/logs/gk_0_45_atcoder_think_t1.0.log
- LeetCode: 2026-11-13 15:16:08 to 15:22:38 IST = 390 s (6 min 30 s), rc=0. Log: /tmp/eval_gk045/logs/gk_0_45_leetcode_think_t1.0.log
- Metrics: $R/metrics/gk_0_45_{atcoder,leetcode}_think_t1.0_metrics.json (num_tasks 242/178, 10 samples per task; load_platform validation passed)
- Log warnings: only SyntaxWarnings coming from executed model samples plus an HF unauthenticated-request notice. No errors or tracebacks.

## Pooling / bootstrap output (verbatim)
gk_0_45: atcoder N=242 leetcode N=178 pooled N=420 n=10
gk_0_4: atcoder N=242 leetcode N=178 pooled N=420 n=10
top_p_0_9: atcoder N=242 leetcode N=178 pooled N=420 n=10
top_p_0_95: atcoder N=242 leetcode N=178 pooled N=420 n=10
ministral_recipe: atcoder N=242 leetcode N=178 pooled N=420 n=10

### Pooled canonical table (N=420)

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gk_0_45 pooled | 420 | 66.86% | 75.67% | 78.25% | 80.28% | 81.19% | 74.5% | 69.8% | 64.0% | 72.9% | 61.0% | 46.9% | 83.7% | 15617 | 4.1% | — | 0.5032 |
| gk_0_4 pooled | 420 | 66.40% | 74.95% | 77.53% | 79.47% | 80.24% | 74.0% | 68.3% | 64.8% | 72.6% | 59.8% | 49.3% | 82.0% | 15954 | 4.1% | — | 0.5071 |
| top_p_0_9 pooled | 420 | 67.64% | 76.31% | 78.91% | 80.98% | 81.90% | 74.8% | 70.5% | 65.5% | 72.6% | 62.1% | 48.3% | 85.2% | 15785 | 4.2% | — | 0.5000 |
| top_p_0_95 pooled | 420 | 66.38% | 76.06% | 79.09% | 81.41% | 82.38% | 74.8% | 69.3% | 63.3% | 73.6% | 61.2% | 46.0% | 83.7% | 15969 | 4.3% | — | 0.5059 |
| ministral_recipe pooled | 420 | 66.83% | 76.29% | 79.00% | 81.28% | 82.38% | 74.8% | 70.7% | 64.3% | 71.4% | 59.0% | 45.0% | 84.4% | 15893 | 4.1% | — | 0.4877 |

### gk_0_45 per platform

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gk_0_45 atcoder | 242 | 63.35% | 71.47% | 74.00% | 75.98% | 76.86% | 70.2% | 65.7% | 60.3% | 69.0% | 58.3% | 45.0% | 80.0% | 16266 | 3.7% | — | 0.5283 |
| gk_0_45 leetcode | 178 | 71.63% | 81.37% | 84.04% | 86.13% | 87.08% | 80.3% | 75.3% | 69.1% | 78.1% | 64.6% | 49.4% | 88.8% | 14734 | 4.7% | — | 0.4731 |

### Paired task bootstrap (tasks = resampling unit; seed 42; percentile 95% CI)

| comparison | metric | resamples | observed diff (pp) | 95% CI (pp) | CI excludes 0 | share diff<=0 |
|:--|:--|--:|--:|:--:|:--:|--:|
| gk_0_45 - gk_0_4 | pass@1 | 2000 | 0.45 | -0.71 to 1.62 | no | 0.2130 |
| gk_0_45 - gk_0_4 | pass@1 | 5000 | 0.45 | -0.67 to 1.60 | no | 0.2132 |
<!-- id check gk_0_45 vs gk_0_4: identical 420-id set=True, same concatenation order=True -->
| gk_0_45 - gk_0_4 | pass@10 | 2000 | 0.95 | -0.95 to 3.10 | no | 0.1940 |
| gk_0_45 - gk_0_4 | pass@10 | 5000 | 0.95 | -0.95 to 3.10 | no | 0.1964 |
| gk_0_45 - top_p_0_9 | pass@1 | 2000 | -0.79 | -1.86 to 0.36 | no | 0.9190 |
| gk_0_45 - top_p_0_9 | pass@1 | 5000 | -0.79 | -1.81 to 0.33 | no | 0.9212 |
<!-- id check gk_0_45 vs top_p_0_9: identical 420-id set=True, same concatenation order=True -->
| gk_0_45 - top_p_0_9 | pass@10 | 2000 | -0.71 | -2.38 to 1.19 | no | 0.8020 |
| gk_0_45 - top_p_0_9 | pass@10 | 5000 | -0.71 | -2.38 to 1.19 | no | 0.8138 |

## Notes
- Pooling uses `pool_platform_metrics` from scripts/ministral_lcb_leaderboard.py: pass@k and cov are task-weighted, term%/mean_tok/timeout% are sample-weighted, cb_div is the mean over pooled tasks with at least 2 correct samples, and struct_div was skipped. Context rows come from the existing local metrics in $R/metrics. No mlx-lm results are included.
- Bootstrap: the resampling unit is the task (420 tasks). It uses lb.bootstrap_indices / lb.bootstrap_ci with seed 42 and a percentile 95% CI. Per-task differences are paired by the `platform::task_id` key. The 420-id sets are identical, and so is the concatenation order. "share diff<=0" is descriptive only: it is the fraction of resample means <= 0. pass@10 is the per-task unbiased estimate at n=10, so it is 0 or 1 for each task.
- Each config is one run (one generation pass, n=10 samples per task). None of the four CIs exclude 0, and no significance claim beyond that is made.
