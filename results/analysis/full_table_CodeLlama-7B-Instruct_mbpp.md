# Full per-config table — CodeLlama-7B-Instruct / MBPP

All 20 configs found in the searched directories. Non-stochastic (greedy, beam) included here but flagged so you can see what was excluded from the 'best' tables.

Sorted by pass@10 descending (NaNs last).

| Config | Group | pass@1 | pass@10 | codebleu_div | n_tasks | stochastic? | metrics_path |
|---|---|---:|---:|---:|---:|---:|---|
| `top_k5_t1.0` | other | 36.56% | 59.60% | 0.4040 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_k5_t1.0_metrics.json` |
| `top_p0.9_t1.0` | other | 38.26% | 59.00% | 0.3660 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p0.9_t1.0_metrics.json` |
| `top_p0.8_t1.0` | other | 39.40% | 56.00% | 0.3436 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p0.8_t1.0_metrics.json` |
| `temp_t0.7` | other | 38.30% | 55.20% | 0.3619 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_t0.7_metrics.json` |
| `pless_t2.0` | other | 37.10% | 53.80% | 0.2989 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t2.0_metrics.json` |
| `pless_alpha_a5.0_t1.0` | α | 40.32% | 53.20% | 0.3042 | 500 | yes | `results/pless_alpha_full_mbpp/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_alpha_a5.0_t1.0_metrics.json` |
| `pless_alpha_a3.0_t1.0` | α | 40.66% | 50.80% | 0.2354 | 500 | yes | `results/pless_alpha_full_mbpp/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_alpha_a3.0_t1.0_metrics.json` |
| `temp_t0.3` | other | 41.36% | 50.20% | 0.2454 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_t0.3_metrics.json` |
| `pless_alpha_a2.5_t1.0` | α | 41.24% | 49.20% | 0.1920 | 500 | yes | `results/pless_alpha_full_mbpp/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_alpha_a2.5_t1.0_metrics.json` |
| `pless_t1.5` | other | 41.18% | 47.20% | 0.1649 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t1.5_metrics.json` |
| `pless_t1.0` | other | 41.64% | 44.20% | 0.0641 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t1.0_metrics.json` |
| `pless_alpha_a2.0_t1.0` | α | 41.78% | 44.20% | 0.0677 | 500 | yes | `results/pless_alpha_full_mbpp/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_alpha_a2.0_t1.0_metrics.json` |
| `beam4_t1.0` | other | 44.00% | 44.00% | 0.0000 | 500 | **no** | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/beam4_t1.0_metrics.json` |
| `pless_norm_t1.0` | other | 41.44% | 43.80% | 0.0694 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t1.0_metrics.json` |
| `beam8_t1.0` | other | 43.40% | 43.40% | 0.0000 | 500 | **no** | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/beam8_t1.0_metrics.json` |
| `pless_t0.7` | other | 42.16% | 43.00% | 0.0326 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t0.7_metrics.json` |
| `pless_norm_t0.7` | other | 42.10% | 43.00% | 0.0321 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t0.7_metrics.json` |
| `pless_t0.6` | other | 41.22% | 42.20% | 0.0514 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t0.6_metrics.json` |
| `greedy_t1.0` | other | 42.20% | 42.20% | 0.0000 | 500 | **no** | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/greedy_t1.0_metrics.json` |
| `pless_norm_t0.6` | other | 41.06% | 42.20% | 0.0533 | 500 | yes | `results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t0.6_metrics.json` |
