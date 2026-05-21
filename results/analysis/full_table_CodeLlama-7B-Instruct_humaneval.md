# Full per-config table — CodeLlama-7B-Instruct / HumanEval

All 13 configs found in the searched directories. Non-stochastic (greedy, beam) included here but flagged so you can see what was excluded from the 'best' tables.

Sorted by pass@10 descending (NaNs last).

| Config | Group | pass@1 | pass@10 | codebleu_div | n_tasks | stochastic? | metrics_path |
|---|---|---:|---:|---:|---:|---:|---|
| `temp_0.7` | other | 36.22% | 62.80% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_0.7_metrics.json` |
| `top_p_0.95` | other | 36.16% | 62.20% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p_0.95_metrics.json` |
| `top_p0.9_t1.0` | other | 25.73% | 51.22% | 0.3902 | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p0.9_t1.0_metrics.json` |
| `temp_0.2` | other | 36.89% | 46.95% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_0.2_metrics.json` |
| `pless_alpha_a5.0_t1.0` | α | 24.82% | 46.95% | 0.2804 | 164 | yes | `results/pless_alpha_full_humaneval/codellama--CodeLlama-7b-Instruct-hf/humaneval/metrics/pless_alpha_a5.0_t1.0_metrics.json` |
| `pless_alpha_a3.0_t1.0` | α | 25.24% | 44.51% | 0.2381 | 164 | yes | `results/pless_alpha_full_humaneval/codellama--CodeLlama-7b-Instruct-hf/humaneval/metrics/pless_alpha_a3.0_t1.0_metrics.json` |
| `pless_alpha_a2.5_t1.0` | α | 25.85% | 40.85% | 0.1606 | 164 | yes | `results/pless_alpha_full_humaneval/codellama--CodeLlama-7b-Instruct-hf/humaneval/metrics/pless_alpha_a2.5_t1.0_metrics.json` |
| `p_less_norm` | other | 35.85% | 39.02% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/p_less_norm_metrics.json` |
| `p_less` | other | 36.10% | 38.41% | — | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/p_less_metrics.json` |
| `greedy` | other | 35.98% | 35.98% | — | 164 | **no** | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/greedy_metrics.json` |
| `pless_alpha_a2.0_t1.0` | α | 27.74% | 32.32% | 0.0566 | 164 | yes | `results/pless_alpha_full_humaneval/codellama--CodeLlama-7b-Instruct-hf/humaneval/metrics/pless_alpha_a2.0_t1.0_metrics.json` |
| `pless_t0.6` | other | 28.11% | 31.71% | 0.0132 | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t0.6_metrics.json` |
| `pless_norm_t0.6` | other | 28.05% | 31.71% | 0.0113 | 164 | yes | `results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t0.6_metrics.json` |
