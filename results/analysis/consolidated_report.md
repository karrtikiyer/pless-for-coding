# Consolidated Metrics Report

## Summary

Total configurations evaluated: **79**

## MBPP

| Model | Method | Temp | pass@1 | pass@5 | pass@10 | Diversity | Distinct-3 | LOC σ | CodeBLEU |
|-------|--------|------|--------|--------|---------|-----------|------------|-------|----------|
| Qwen/Qwen-7B | pless | 0.6 | 0.3536 | 0.3962 | 0.4040 | 0.0656 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless | 1.0 | 0.4051 | 0.5751 | 0.6148 | 0.1997 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless | 1.0 | 0.3502 | 0.4581 | 0.4860 | 0.1666 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_bs | 0.6 | 0.1279 | 0.2817 | 0.3256 | 0.0396 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_hybrid | 0.6 | 0.3146 | 0.3727 | 0.3820 | 0.0545 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_norm | 0.6 | 0.3554 | 0.3981 | 0.4040 | 0.0654 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_norm | 1.0 | 0.4109 | 0.5776 | 0.6187 | 0.2436 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_norm | 1.0 | 0.3568 | 0.4699 | 0.5020 | 0.1656 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_norm_hybrid | 0.6 | 0.3138 | 0.3721 | 0.3820 | 0.0547 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | pless_ns0 | 0.6 | 0.1396 | 0.2467 | 0.2820 | 0.2282 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | temp | 0.7 | 0.3276 | 0.5798 | 0.6498 | 0.6226 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | temp | 0.7 | 0.2980 | 0.5087 | 0.5812 | 0.3975 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | temp_ns0 | 0.7 | 0.0972 | 0.3007 | 0.3980 | 0.4062 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B | top_p0.9 | 1.0 | 0.2747 | 0.4859 | 0.5591 | 0.4289 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B-Chat | pless | 0.6 | 0.3402 | 0.3636 | 0.3700 | 0.0184 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B-Chat | pless | 1.0 | 0.3438 | 0.3961 | 0.4060 | 0.0461 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B-Chat | pless_norm | 0.6 | 0.3440 | 0.3726 | 0.3760 | 0.0187 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B-Chat | pless_norm | 1.0 | 0.3446 | 0.3993 | 0.4080 | 0.0424 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B-Chat | temp | 0.7 | 0.2870 | 0.4471 | 0.5040 | 0.2319 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen-7B-Chat | top_p0.9 | 1.0 | 0.2708 | 0.4367 | 0.5060 | 0.2474 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-7B | pless | 1.0 | 0.6564 | 0.7810 | 0.8093 | 0.2545 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-7B | pless_norm | 1.0 | 0.6529 | 0.7811 | 0.8093 | 0.2451 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | pless_bigcode | 0.6 | 0.5314 | 0.5932 | 0.6080 | 0.1030 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | pless_bigcode | 1.0 | 0.5136 | 0.6489 | 0.6900 | 0.2626 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | pless_norm_bigcode | 0.6 | 0.5288 | 0.5990 | 0.6140 | 0.1081 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | pless_norm_bigcode | 1.0 | 0.5202 | 0.6462 | 0.6780 | 0.2645 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | temp_bigcode | 0.7 | 0.3814 | 0.6488 | 0.7120 | 0.5282 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | top_p0.95_bigcode | 0.2 | 0.5248 | 0.6445 | 0.6860 | 0.1998 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-1.5B | top_p0.9_bigcode | 1.0 | 0.3272 | 0.6147 | 0.7020 | 0.6206 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | pless_bigcode | 0.6 | 0.5944 | 0.6495 | 0.6640 | 0.0965 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | pless_bigcode | 1.0 | 0.5658 | 0.6945 | 0.7220 | 0.2588 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | pless_norm_bigcode | 0.6 | 0.5948 | 0.6536 | 0.6680 | 0.0954 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | pless_norm_bigcode | 1.0 | 0.5768 | 0.6988 | 0.7240 | 0.2512 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | temp_bigcode | 0.7 | 0.4264 | 0.7071 | 0.7780 | 0.5365 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | top_p0.95_bigcode | 0.2 | 0.5820 | 0.6916 | 0.7160 | 0.1832 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-3B | top_p0.9_bigcode | 1.0 | 0.3472 | 0.6732 | 0.7700 | 0.5886 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-7B-Instruct | pless | 1.0 | 0.8389 | 0.8557 | 0.8599 | 0.0637 | 0.0000 | 0.00 | 0.0000 |
| Qwen/Qwen2.5-Coder-7B-Instruct | pless_norm | 1.0 | 0.8358 | 0.8594 | 0.8638 | 0.0596 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | pless | 0.6 | 0.4122 | 0.4198 | 0.4220 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | pless | 1.0 | 0.4164 | 0.4380 | 0.4420 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | pless_norm | 0.6 | 0.4106 | 0.4205 | 0.4220 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | pless_norm | 1.0 | 0.4144 | 0.4354 | 0.4380 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | temp | 0.7 | 0.3830 | 0.5158 | 0.5520 | 0.0106 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-Instruct-hf | top_p0.9 | 1.0 | 0.3826 | 0.5372 | 0.5900 | 0.0391 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-hf | pless | 0.6 | 0.4176 | 0.4817 | 0.4940 | 0.0814 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-hf | pless | 1.0 | 0.4146 | 0.5356 | 0.5720 | 0.2305 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-hf | pless_norm | 0.6 | 0.4172 | 0.4793 | 0.4900 | 0.0890 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-hf | pless_norm | 1.0 | 0.4152 | 0.5375 | 0.5740 | 0.2245 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-hf | temp | 0.7 | 0.3684 | 0.5886 | 0.6520 | 0.4360 | 0.0000 | 0.00 | 0.0000 |
| codellama/CodeLlama-7b-hf | top_p0.9 | 1.0 | 0.3460 | 0.5949 | 0.6820 | 0.4986 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | pless_bigcode | 0.6 | 0.4388 | 0.4823 | 0.4900 | 0.0482 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | pless_bigcode | 1.0 | 0.4412 | 0.4973 | 0.5120 | 0.0720 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | pless_norm_bigcode | 0.6 | 0.4392 | 0.4843 | 0.4940 | 0.0519 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | pless_norm_bigcode | 1.0 | 0.4464 | 0.4981 | 0.5100 | 0.0722 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | temp_bigcode | 0.7 | 0.4306 | 0.5798 | 0.6240 | 0.2751 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | top_p0.95_bigcode | 0.2 | 0.4452 | 0.5014 | 0.5180 | 0.0849 | 0.0000 | 0.00 | 0.0000 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | top_p0.9_bigcode | 1.0 | 0.4278 | 0.5875 | 0.6400 | 0.3115 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | pless | 0.6 | 0.2050 | 0.2134 | 0.2140 | 0.0018 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | pless | 1.0 | 0.2856 | 0.3181 | 0.3307 | 0.0112 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | pless | 1.0 | 0.2010 | 0.2189 | 0.2240 | 0.0194 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | pless_norm | 0.6 | 0.2042 | 0.2129 | 0.2140 | 0.0041 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | pless_norm | 1.0 | 0.2903 | 0.3180 | 0.3230 | 0.0166 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | pless_norm | 1.0 | 0.2020 | 0.2196 | 0.2220 | 0.0247 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | temp | 0.7 | 0.2732 | 0.3832 | 0.4280 | 0.1567 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | temp | 0.7 | 0.1780 | 0.2711 | 0.3020 | 0.1418 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-chat-hf | top_p0.9 | 1.0 | 0.1802 | 0.2902 | 0.3360 | 0.1719 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless | 0.6 | 0.2254 | 0.2777 | 0.2920 | 0.0671 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless | 1.0 | 0.3152 | 0.4875 | 0.5331 | 0.4855 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless | 1.0 | 0.2236 | 0.3360 | 0.3700 | 0.1677 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless_hybrid | 0.6 | 0.2194 | 0.2625 | 0.2740 | 0.0770 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless_norm | 0.6 | 0.2258 | 0.2811 | 0.3000 | 0.0725 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless_norm | 1.0 | 0.3175 | 0.4815 | 0.5292 | 0.4658 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless_norm | 1.0 | 0.2178 | 0.3306 | 0.3720 | 0.1592 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless_norm_hybrid | 0.6 | 0.2220 | 0.2630 | 0.2720 | 0.0847 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | pless_ns0 | 0.6 | 0.1376 | 0.2399 | 0.2780 | 0.1658 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | temp | 0.7 | 0.2257 | 0.4452 | 0.5292 | 0.6044 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | temp | 0.7 | 0.1714 | 0.3597 | 0.4460 | 0.3968 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | temp_ns0 | 0.7 | 0.0404 | 0.1560 | 0.2420 | 0.2837 | 0.0000 | 0.00 | 0.0000 |
| meta-llama/Llama-2-7b-hf | top_p0.9 | 1.0 | 0.1404 | 0.3196 | 0.3940 | 0.4287 | 0.0000 | 0.00 | 0.0000 |

## Key Research Questions

### Does p-less sampling produce more diverse correct solutions?

**MBPP**: p-less avg diversity = 0.1105 (pass@1=0.3768) vs temp avg diversity = 0.3591 (pass@1=0.2777)

### Does normalization improve p-less sampling?

**MBPP** pass@1: pless=0.3768, pless_norm=0.3774

**MBPP** structural_diversity: pless=0.1105, pless_norm=0.1116

**MBPP** mean_distinct_3: pless=0.0000, pless_norm=0.0000

