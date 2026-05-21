# Full per-config table — Qwen2.5-Coder-7B-Instruct / HumanEval

All 13 configs found in the searched directories. Non-stochastic (greedy, beam) included here but flagged so you can see what was excluded from the 'best' tables.

Sorted by pass@10 descending (NaNs last).

| Config | Group | pass@1 | pass@10 | codebleu_div | n_tasks | stochastic? | metrics_path |
|---|---|---:|---:|---:|---:|---:|---|
| `p_less_norm` | other | 75.18% | 95.12% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/p_less_norm_metrics.json` |
| `top_p_0.95` | other | 79.76% | 94.51% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/top_p_0.95_metrics.json` |
| `temp_0.7` | other | 79.02% | 94.51% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/temp_0.7_metrics.json` |
| `top_p0.9_t1.0` | other | 82.44% | 92.68% | 0.3568 | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/top_p0.9_t1.0_metrics.json` |
| `pless_alpha_a5.0_t1.0` | α | 84.57% | 91.46% | 0.2578 | 164 | yes | `results/pless_alpha_full_humaneval/Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/metrics/pless_alpha_a5.0_t1.0_metrics.json` |
| `pless_alpha_a2.5_t1.0` | α | 87.13% | 91.46% | 0.1423 | 164 | yes | `results/pless_alpha_full_humaneval/Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/metrics/pless_alpha_a2.5_t1.0_metrics.json` |
| `pless_alpha_a3.0_t1.0` | α | 85.98% | 91.46% | 0.1693 | 164 | yes | `results/pless_alpha_full_humaneval/Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/metrics/pless_alpha_a3.0_t1.0_metrics.json` |
| `temp_0.2` | other | 83.48% | 90.85% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/temp_0.2_metrics.json` |
| `p_less` | other | 83.29% | 90.24% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/p_less_metrics.json` |
| `pless_alpha_a2.0_t1.0` | α | 87.38% | 89.63% | 0.0396 | 164 | yes | `results/pless_alpha_full_humaneval/Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/metrics/pless_alpha_a2.0_t1.0_metrics.json` |
| `pless_norm_t0.6` | other | 87.50% | 88.41% | 0.0173 | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/pless_norm_t0.6_metrics.json` |
| `pless_t0.6` | other | 87.50% | 87.80% | 0.0172 | 164 | yes | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/pless_t0.6_metrics.json` |
| `greedy` | other | 84.15% | 84.15% | — | 164 | **no** | `results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/greedy_metrics.json` |
