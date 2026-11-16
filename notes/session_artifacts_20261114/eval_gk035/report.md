# gk_0_35 local eval + 420-task pooling (Ministral-3-3B-Reasoning-2512, LCB v6, Renyi G_k k=0.35, T=1.0, n=10, vLLM 0.21.0 on GPU2)

Generated 2026-11-14 IST on the local Mac (not gpu2). Every number below is copied verbatim from `/tmp/eval_gk035/pool_output.txt` (the stdout of `pool_and_bootstrap.py`), from the eval logs, or from the check commands listed. Scripts are adapted from /tmp/eval_gk045/, and the originals were not edited.

## Inputs / copies
- Sources: /tmp/gk035_validate/gk_0_35_{atcoder,leetcode}_think_t1.0.jsonl (242 / 178 lines).
- Both files were copied into results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/. Neither target existed beforehand. The sha256 of each copy was checked after `cp` and matches: atcoder 7459e5217690e4c5bd9098cac86663bc1fb2ce9323818f517a99547960352b97, leetcode 069d63bb7984aefe64eb42d2cee8141345be42645e3fd11f1bd9a2caa7d5b9d0.
- The new jsonl and metrics files are untracked (`git ls-files` finds 0 gk_0_35 files in that metrics dir, and the metrics dir itself is `??`). Nothing was git-added, the branch was not switched (still feat/humaneval-regen), and no tracked file was edited.

## Load gate (before the evals)
- /tmp/eval_gk035/wait_idle.sh polled `pgrep -fl "pytest|uv run python -m bench.eval"` and the 1-minute load average every 20 s, with a 900 s maximum. Log: /tmp/eval_gk035/logs/wait_idle.log
- Started 12:46:22 IST. Another agent's pytest runs (in /tmp/wt_topk) were active for the first 60 s. The gate passed at 12:47:42 IST (elapsed 80 s, load1=2.69, no pytest/bench.eval process). The maximum wait was NOT hit.

## Commands (repo root, run sequentially by /tmp/eval_gk035/run_evals.sh under nohup after the gate)
    R=results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046
    uv run python -m bench.eval --results-file $R/gk_0_35_atcoder_think_t1.0.jsonl  --dataset livecodebench --workers 4 --skip-structural-diversity
    uv run python -m bench.eval --results-file $R/gk_0_35_leetcode_think_t1.0.jsonl --dataset livecodebench --workers 4 --skip-structural-diversity
    uv run python /tmp/eval_gk035/pool_and_bootstrap.py > /tmp/eval_gk035/pool_output.txt   # imports scripts/ministral_lcb_leaderboard.py

## Timings (measured from the START/END epoch stamps in the logs)
- AtCoder: 2026-11-14 12:47:42 to 13:06:32 IST, epoch 1794640662 to 1794641792 = 1130 s, rc=0. Log: /tmp/eval_gk035/logs/gk_0_35_atcoder_think_t1.0.log
- LeetCode: 2026-11-14 13:06:32 to 13:13:01 IST, epoch 1794641792 to 1794642181 = 389 s, rc=0. Log: /tmp/eval_gk035/logs/gk_0_35_leetcode_think_t1.0.log
- Metrics: $R/metrics/gk_0_35_{atcoder,leetcode}_think_t1.0_metrics.json have num_tasks 242/178, num_samples_per_task 10, and the per-task pass_results lengths are all {10}. load_platform and pool_platform_metrics validation (re-derived aggregates) passed.
- Log warnings: SyntaxWarnings from executed model samples (16 in the atcoder log, 24 in the leetcode log). A grep for error/traceback/exception, excluding SyntaxWarning, returned nothing in either log.

## Pooling / bootstrap output (verbatim from pool_output.txt)
gk_0_35: atcoder N=242 leetcode N=178 pooled N=420 n=10
gk_0_45: atcoder N=242 leetcode N=178 pooled N=420 n=10
gk_0_4: atcoder N=242 leetcode N=178 pooled N=420 n=10
top_p_0_9: atcoder N=242 leetcode N=178 pooled N=420 n=10
top_p_0_95: atcoder N=242 leetcode N=178 pooled N=420 n=10
ministral_recipe: atcoder N=242 leetcode N=178 pooled N=420 n=10

### Pooled canonical table (N=420)

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gk_0_35 pooled | 420 | 66.40% | 75.84% | 78.52% | 80.66% | 81.67% | 74.5% | 69.8% | 65.0% | 71.9% | 61.2% | 49.0% | 82.2% | 16144 | 4.3% | — | 0.5115 |
| gk_0_45 pooled | 420 | 66.86% | 75.67% | 78.25% | 80.28% | 81.19% | 74.5% | 69.8% | 64.0% | 72.9% | 61.0% | 46.9% | 83.7% | 15617 | 4.1% | — | 0.5032 |
| gk_0_4 pooled | 420 | 66.40% | 74.95% | 77.53% | 79.47% | 80.24% | 74.0% | 68.3% | 64.8% | 72.6% | 59.8% | 49.3% | 82.0% | 15954 | 4.1% | — | 0.5071 |
| top_p_0_9 pooled | 420 | 67.64% | 76.31% | 78.91% | 80.98% | 81.90% | 74.8% | 70.5% | 65.5% | 72.6% | 62.1% | 48.3% | 85.2% | 15785 | 4.2% | — | 0.5000 |
| top_p_0_95 pooled | 420 | 66.38% | 76.06% | 79.09% | 81.41% | 82.38% | 74.8% | 69.3% | 63.3% | 73.6% | 61.2% | 46.0% | 83.7% | 15969 | 4.3% | — | 0.5059 |
| ministral_recipe pooled | 420 | 66.83% | 76.29% | 79.00% | 81.28% | 82.38% | 74.8% | 70.7% | 64.3% | 71.4% | 59.0% | 45.0% | 84.4% | 15893 | 4.1% | — | 0.4877 |

### gk_0_35 per platform

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gk_0_35 atcoder | 242 | 63.22% | 71.65% | 74.22% | 76.31% | 77.27% | 70.2% | 65.3% | 62.0% | 68.6% | 57.9% | 49.2% | 78.8% | 16799 | 3.6% | — | 0.5380 |
| gk_0_35 leetcode | 178 | 70.73% | 81.54% | 84.35% | 86.58% | 87.64% | 80.3% | 75.8% | 69.1% | 76.4% | 65.7% | 48.9% | 86.8% | 15255 | 5.1% | — | 0.4798 |

### Paired task bootstrap (tasks = resampling unit; seed 42; percentile 95% CI)

| comparison | metric | resamples | observed diff (pp) | 95% CI (pp) | CI excludes 0 | share diff<=0 |
|:--|:--|--:|--:|:--:|:--:|--:|
| gk_0_35 - gk_0_4 | pass@1 | 2000 | -0.00 | -1.07 to 1.05 | no | 0.4810 |
| gk_0_35 - gk_0_4 | pass@1 | 5000 | -0.00 | -1.07 to 1.07 | no | 0.4872 |
<!-- id check gk_0_35 vs gk_0_4: identical 420-id set=True, same concatenation order=True -->
| gk_0_35 - gk_0_4 | pass@10 | 2000 | 1.43 | -0.48 to 3.33 | no | 0.0705 |
| gk_0_35 - gk_0_4 | pass@10 | 5000 | 1.43 | -0.48 to 3.33 | no | 0.0722 |
| gk_0_35 - gk_0_45 | pass@1 | 2000 | -0.45 | -1.57 to 0.67 | no | 0.7955 |
| gk_0_35 - gk_0_45 | pass@1 | 5000 | -0.45 | -1.57 to 0.64 | no | 0.7976 |
<!-- id check gk_0_35 vs gk_0_45: identical 420-id set=True, same concatenation order=True -->
| gk_0_35 - gk_0_45 | pass@10 | 2000 | 0.48 | -1.19 to 2.14 | no | 0.3460 |
| gk_0_35 - gk_0_45 | pass@10 | 5000 | 0.48 | -1.19 to 2.14 | no | 0.3440 |
| gk_0_35 - top_p_0_9 | pass@1 | 2000 | -1.24 | -2.21 to -0.26 | yes | 0.9935 |
| gk_0_35 - top_p_0_9 | pass@1 | 5000 | -1.24 | -2.21 to -0.26 | yes | 0.9932 |
<!-- id check gk_0_35 vs top_p_0_9: identical 420-id set=True, same concatenation order=True -->
| gk_0_35 - top_p_0_9 | pass@10 | 2000 | -0.24 | -1.90 to 1.43 | no | 0.6540 |
| gk_0_35 - top_p_0_9 | pass@10 | 5000 | -0.24 | -1.90 to 1.43 | no | 0.6536 |

### Descriptive: GPU2 k values (one run per config; pooled N=420)

| k | config | pass@1 | pass@10 | term% | mean_tok |
|--:|:--|--:|--:|--:|--:|
| 0.35 | gk_0_35 | 66.40% | 81.67% | 82.2% | 16144 |
| 0.4 | gk_0_4 | 66.40% | 80.24% | 82.0% | 15954 |
| 0.45 | gk_0_45 | 66.86% | 81.19% | 83.7% | 15617 |

Adjacent-k paired differences (same bootstrap):

| comparison | metric | resamples | observed diff (pp) | 95% CI (pp) | CI excludes 0 | share diff<=0 |
|:--|:--|--:|--:|:--:|:--:|--:|
| gk_0_4 - gk_0_35 | pass@1 | 2000 | 0.00 | -1.05 to 1.07 | no | 0.5220 |
| gk_0_4 - gk_0_35 | pass@1 | 5000 | 0.00 | -1.07 to 1.07 | no | 0.5150 |
<!-- id check gk_0_4 vs gk_0_35: identical 420-id set=True, same concatenation order=True -->
| gk_0_4 - gk_0_35 | pass@10 | 2000 | -1.43 | -3.33 to 0.48 | no | 0.9565 |
| gk_0_4 - gk_0_35 | pass@10 | 5000 | -1.43 | -3.33 to 0.48 | no | 0.9548 |
| gk_0_45 - gk_0_4 | pass@1 | 2000 | 0.45 | -0.71 to 1.62 | no | 0.2130 |
| gk_0_45 - gk_0_4 | pass@1 | 5000 | 0.45 | -0.67 to 1.60 | no | 0.2132 |
<!-- id check gk_0_45 vs gk_0_4: identical 420-id set=True, same concatenation order=True -->
| gk_0_45 - gk_0_4 | pass@10 | 2000 | 0.95 | -0.95 to 3.10 | no | 0.1940 |
| gk_0_45 - gk_0_4 | pass@10 | 5000 | 0.95 | -0.95 to 3.10 | no | 0.1964 |

## Notes
- The gk_0_35 - gk_0_4 pooled pass@1 is an exact tie. Both configs have 2789 correct out of 4200 samples (checked by summing num_correct over both platform metrics files), so the printed "-0.00" is floating-point noise.
- Pooling uses `pool_platform_metrics` from scripts/ministral_lcb_leaderboard.py: pass@k and cov are task-weighted, term%/mean_tok/timeout% are sample-weighted, cb_div is the mean over pooled tasks with at least 2 correct samples, and struct_div was skipped. Context rows come from the existing local metrics in $R/metrics. No mlx-lm results are included. The context rows are identical to the gk_0_45 report of 2026-11-13.
- Bootstrap: the resampling unit is the task (420 tasks). It uses lb.bootstrap_indices / lb.bootstrap_ci with seed 42, a percentile 95% CI, and 2000 and 5000 resamples. Differences are paired by the `platform::task_id` key. The 420-id sets and the concatenation order are identical for every comparison (see the id-check comments). "share diff<=0" is descriptive only: it is the fraction of resample means <= 0.
- Each config is one run (one generation pass, n=10 samples per task). The only CI that excludes 0 is gk_0_35 - top_p_0_9 on pass@1, at both resample counts. No multiple-comparison correction was applied across the comparisons shown.
- k = 0.35 / 0.4 / 0.45 table: this is descriptive only. None of the adjacent-k paired differences (0.4-0.35, 0.45-0.4) has a CI that excludes 0 for pass@1 or pass@10, so no trend claim is made.
- Unrelated pre-existing working-tree changes (not made by this run): two `M` Qwen3-8B stratified60 gk_0_35 metrics files (mtime 2026-11-10) and a `D` on LCB_v6_full1046/analysis/atcoder_gk_sweep_leaderboard.md.
