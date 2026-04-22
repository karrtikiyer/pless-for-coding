# Consolidated Metrics Report

## Summary

Total configurations evaluated: **36**

## HUMANEVAL

| Model | Method | Temp | pass@1 | pass@5 | pass@10 | Diversity | Distinct-3 | LOC σ | CodeBLEU |
|-------|--------|------|--------|--------|---------|-----------|------------|-------|----------|
| Qwen--Qwen2.5-Coder-7B-Instruct | greedy | 0.0 | 0.8415 | 0.8415 | 0.8415 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen2.5-Coder-7B-Instruct | p_less | 1.0 | 0.8335 | 0.8934 | 0.9024 | 0.1488 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen2.5-Coder-7B-Instruct | p_less_norm | 1.0 | 0.7567 | 0.9346 | 0.9512 | 0.4000 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen2.5-Coder-7B-Instruct | temp_0.2 | 0.2 | 0.8366 | 0.8993 | 0.9085 | 0.1595 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen2.5-Coder-7B-Instruct | temp_0.7 | 0.7 | 0.7921 | 0.9293 | 0.9512 | 0.3862 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen2.5-Coder-7B-Instruct | top_p_0.95 | 1.0 | 0.8012 | 0.9334 | 0.9573 | 0.4023 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | greedy | 0.0 | 0.7561 | 0.7561 | 0.7561 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | p_less | 1.0 | 0.7604 | 0.7790 | 0.7805 | 0.0145 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | p_less_norm | 1.0 | 0.7573 | 0.7760 | 0.7805 | 0.0132 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | temp_0.2 | 0.2 | 0.7652 | 0.8152 | 0.8354 | 0.0450 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | temp_0.7 | 0.7 | 0.7750 | 0.8511 | 0.8720 | 0.1156 | 0.0000 | 0.00 | 0.0000 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | top_p_0.95 | 1.0 | 0.7659 | 0.8145 | 0.8232 | 0.0330 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-7B-Instruct | pless | 0.6 | 0.8750 | 0.8780 | 0.8780 | 0.0074 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-7B-Instruct | pless_norm | 0.6 | 0.8750 | 0.8810 | 0.8841 | 0.0079 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-7B-Instruct | top_p0.9 | 1.0 | 0.8244 | 0.9075 | 0.9268 | 0.2156 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | pless | 0.6 | 0.7890 | 0.7982 | 0.7988 | 0.0088 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | pless_norm | 0.6 | 0.7854 | 0.7951 | 0.7988 | 0.0096 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | top_p0.9 | 1.0 | 0.7817 | 0.8060 | 0.8110 | 0.0863 | 0.0000 | 0.00 | 0.0000 |
| codellama--CodeLlama-7b-Instruct-hf | greedy | 0.0 | 0.3598 | 0.3598 | 0.3598 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama--CodeLlama-7b-Instruct-hf | p_less | 1.0 | 0.3549 | 0.3749 | 0.3780 | 0.0635 | 0.0000 | 0.00 | 0.0000 |
| codellama--CodeLlama-7b-Instruct-hf | p_less_norm | 1.0 | 0.3512 | 0.3799 | 0.3841 | 0.0735 | 0.0000 | 0.00 | 0.0000 |
| codellama--CodeLlama-7b-Instruct-hf | temp_0.2 | 0.2 | 0.3659 | 0.4369 | 0.4695 | 0.1053 | 0.0000 | 0.00 | 0.0000 |
| codellama--CodeLlama-7b-Instruct-hf | temp_0.7 | 0.7 | 0.3634 | 0.5566 | 0.6341 | 0.2493 | 0.0000 | 0.00 | 0.0000 |
| codellama--CodeLlama-7b-Instruct-hf | top_p_0.95 | 1.0 | 0.3579 | 0.5401 | 0.6220 | 0.1918 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | pless | 0.6 | 0.2811 | 0.3135 | 0.3171 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | pless_norm | 0.6 | 0.2805 | 0.3150 | 0.3171 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | top_p0.9 | 1.0 | 0.2573 | 0.4474 | 0.5122 | 0.1021 | 0.0000 | 0.00 | 0.0000 |
| mistralai--Codestral-22B-v0.1 | greedy | 0.0 | 0.7561 | 0.7561 | 0.7561 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| mistralai--Codestral-22B-v0.1 | p_less | 1.0 | 0.7799 | 0.8382 | 0.8476 | 0.1634 | 0.0000 | 0.00 | 0.0000 |
| mistralai--Codestral-22B-v0.1 | p_less_norm | 1.0 | 0.7768 | 0.8368 | 0.8476 | 0.1703 | 0.0000 | 0.00 | 0.0000 |
| mistralai--Codestral-22B-v0.1 | temp_0.2 | 0.2 | 0.7707 | 0.8498 | 0.8659 | 0.1949 | 0.0000 | 0.00 | 0.0000 |
| mistralai--Codestral-22B-v0.1 | temp_0.7 | 0.7 | 0.7299 | 0.8799 | 0.9085 | 0.4018 | 0.0000 | 0.00 | 0.0000 |
| mistralai--Codestral-22B-v0.1 | top_p_0.95 | 1.0 | 0.7323 | 0.8868 | 0.9146 | 0.3990 | 0.0000 | 0.00 | 0.0000 |
| mistralai/Codestral-22B-v0.1 | pless | 0.6 | 0.7512 | 0.7858 | 0.7866 | 0.0631 | 0.0000 | 0.00 | 0.0000 |
| mistralai/Codestral-22B-v0.1 | pless_norm | 0.6 | 0.7494 | 0.7899 | 0.7927 | 0.0678 | 0.0000 | 0.00 | 0.0000 |
| mistralai/Codestral-22B-v0.1 | top_p0.9 | 1.0 | 0.6689 | 0.8862 | 0.9207 | 0.4632 | 0.0000 | 0.00 | 0.0000 |

## Key Research Questions

### Does p-less sampling produce more diverse correct solutions?

**HUMANEVAL**: p-less avg diversity = 0.0198 (pass@1=0.6741) vs temp avg diversity = 0.2072 (pass@1=0.6749)

### Does normalization improve p-less sampling?

**HUMANEVAL** pass@1: pless=0.6741, pless_norm=0.6726

**HUMANEVAL** structural_diversity: pless=0.0198, pless_norm=0.0213

**HUMANEVAL** mean_distinct_3: pless=0.0000, pless_norm=0.0000

