# algosim Diversity Report — Qwen2.5-Coder-7B-Instruct α-sweep (MBPP-500)

AlgoSim NAUADC / EA / DA@10 (Claude-Sonnet-4.6 judge, correct samples only) joined with our existing structural / CodeBLEU / pass@k metrics. Sorted by NAUADC descending.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **pless_alpha_a5.0_t1.0** | pless_alpha @ T=1.0 | 0.753 | 0.880 | 0.210 | 0.426 | **1.167** | 1.127 | 1.186 | 440 |
| **pless_alpha_a3.0_t1.0** | pless_alpha @ T=1.0 | 0.766 | 0.864 | 0.160 | 0.340 | **1.110** | 1.094 | 1.120 | 432 |
| **pless_alpha_a2.5_t1.0** | pless_alpha @ T=1.0 | 0.768 | 0.864 | 0.131 | 0.283 | **1.101** | 1.080 | 1.111 | 432 |
| **pless_alpha_a2.0_t1.0** | pless_alpha @ T=1.0 | 0.771 | 0.820 | 0.058 | 0.133 | **1.041** | 1.036 | 1.044 | 410 |
