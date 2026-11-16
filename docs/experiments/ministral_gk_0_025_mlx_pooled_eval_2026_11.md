# Ministral-3-3B-Reasoning-2512 — G_k (k = 0.025) on LCB v6 full1046, mlx-lm (96GB Mac): pooled eval

Status: generation DONE (2026-11-09 ~17:27 IST), local eval DONE (2026-11-09 18:27–18:50 IST), numbers independently validated (results-validator agent, 2026-11-09). One sampling run per task set; no confidence interval has been computed.

## What this is

`--method pless_renyi`, `renyi_k = 0.025`, T = 1.0, top_p = 1.0, top_k = 0, thinking on, 10 samples per task, `max_new_tokens = 35000`, backend **mlx-lm** (not vLLM). Slice: the 420 tasks of full1046 batches 1+2 (AtCoder 242, LeetCode 178). The task ids are identical to the `gk_0_4` / `top_p_0_9` vLLM runs on GPU2 (validated: 242/242 and 178/178 overlap, no differences).

Part of the Mac G_k sweep (`scripts/run_mlxlm_ministral_gk_pools.py`, configs run in order with `gk_0_1` last as the same-backend anchor). This note covers `gk_0_025` only; the later k values are still running.

## Files

Raw generations, local copy (copied from the Mac with rsync `--checksum`; sha256 verified on both ends):

| File (dir `results/pless_variants_mlxlm_pools/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/`) | Lines | sha256 |
|---|---:|---|
| `gk_0_025_atcoder_think_t1.0.jsonl` | 242 | `ea989fc13883fdaafb61f7aa208bd2a0b47915e76265d14cc3083d1cccfd31e9` |
| `gk_0_025_leetcode_think_t1.0.jsonl` | 178 | `45ea290b8ab29964995694a8081287072c97d77a726cec714405a745e6a6e296` |

Metrics: same dir, `metrics/gk_0_025_{atcoder,leetcode}_think_t1.0_metrics.json`. Results files are not committed in this repo (move with scp/rsync).

Eval command (run locally, per CLAUDE.md gotcha #9; same settings as the Ministral vLLM re-eval): `uv run python -m bench.eval --results-file <f> --dataset livecodebench --workers 4 --skip-structural-diversity` (6.0 s per-test timeout). Wall time: AtCoder 18:27–18:44 IST, LeetCode 18:44–18:50 IST, both exit code 0.

## Pooled table (420 tasks) and per-platform

Canonical column set (`CANONICAL_COLUMNS`), pooled with `scripts/ministral_lcb_leaderboard.py:pool_platform_metrics`.

| Config | N | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@0.3 | cov@0.5 | cov@0.7 | cov@0.3d | cov@0.5d | cov@0.7d | term% | mean_tok | timeout% | struct_div | cb_div |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gk_0_025 pooled (420) | 420 | 65.90% | 75.29% | 78.37% | 80.61% | 81.43% | 73.3% | 67.9% | 62.4% | 72.1% | 60.2% | 45.2% | 81.1% | 16569 | 4.2% | — | 0.5045 |
| gk_0_025 atcoder | 242 | 63.06% | 71.02% | 73.86% | 75.85% | 76.45% | 69.4% | 64.0% | 60.3% | 68.6% | 57.4% | 45.5% | 77.4% | 17258 | 3.7% | — | 0.5224 |
| gk_0_025 leetcode | 178 | 69.78% | 81.08% | 84.49% | 87.09% | 88.20% | 78.7% | 73.0% | 65.2% | 77.0% | 64.0% | 44.9% | 86.2% | 15634 | 4.8% | — | 0.4828 |

Pooling rules: pass@k = mean over all 420 tasks of the unbiased per-task estimator (n = 10), not a blend of the platform percentages; cov@t = share of tasks with count >= t·n; term%, mean_tok, timeout% are sample-weighted across platforms; cb_div = mean per-task self_codebleu over tasks with >= 2 correct samples and a value (327 tasks: 179 AtCoder + 148 LeetCode); struct_div not computed (`--skip-structural-diversity`).

## Validation (independent recompute, own code, raw data)

All 39 claimed values match at the displayed precision. Checks: sha256/line counts; no duplicate task_ids; `num_correct == sum(pass_results)`; pass@k and cov recomputed from `per_task`; `term%` recomputed from the raw jsonl using the Ministral marker `[/THINK]` (77.438 / 86.180, pooled 81.143); `mean_tok` retokenized locally on transformers 5.3.0 (17257.79 / 15633.55, pooled 16569.4, exact match); `timeout%` from execution statuses (89/2420 and 86/1780, pooled 175/4200 = 4.17%); sample-weighted pooling arithmetic. The distinct-cov numbers reuse the stored `num_distinct_correct` (AST fingerprints not recomputed). No repeated sample text within or across tasks.

## Provenance (from the jsonl rows)

Every row: backend mlx-lm, method `gk_0_025`, sampler `pless_renyi`, `renyi_k` 0.025, T 1.0, top_p 1.0, top_k 0, thinking on, `max_new_tokens` 35000. No precision/quantization field is recorded. Row timestamps: AtCoder 2026-10-30T16:04Z to 2026-11-08T03:13Z; LeetCode 2026-10-31T04:24Z to 2026-11-09T11:57Z. Rows were written by three versions of the batching code (4 / 124 / 292 rows, each with different batch caps). 15 tasks (10 AtCoder, 5 LeetCode) combine samples from more than one attempt/seed after crash/stop resumes (5 AtCoder easy tasks show repeated `row_index`, explained by resume renumbering).

## Caveats — read before using these numbers in any comparison

1. **Different backend.** mlx-lm and vLLM differ in kernels and RNG, so a gap to a vLLM row (`top_p_0_9` 67.64%, `gk_0_4` 66.40%) mixes the decoder effect with the backend effect. 65.90% must not be placed in the vLLM leaderboard without a note. The like-for-like comparison is against the mlx `gk_0_1` anchor (still to run), or a vLLM rerun of `gk_0_025`.
2. **Single run, no CI.** Seeds are per batch, not independent reruns. Before any ranking statement, run a paired per-task bootstrap on the 420 tasks (the repo's leaderboard uses 2000 resamples, seed 42).
3. **Generation cap of the vLLM references not confirmed.** vLLM records have no `max_new_tokens` field. The mlx run used 35000; the CUDA runner's argparse default is 25000 (`scripts/run_pless_variants_ministral_3b_lcb_cuda.py:349`). Confirm the references' cap from the GPU2 run logs before treating the termination/pass numbers as cap-matched.
4. **unverified hypothesis: possible bias toward short samples in the 15 multi-attempt tasks.** When a batch was killed, unfinished rows were discarded and redrawn on resume, which could favour shorter samples. To confirm, compare `_killed/reconstructed_crashes.jsonl` against the banked samples. Expected impact small (15 of 420 tasks).

## Related

TODO row A72 in `docs/theory/todos.md`. Validator working files (volatile): `/tmp/validate_gk025/report.md`, scripts and outputs alongside.
