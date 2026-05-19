# algosim Diversity Report — CodeLlama-7B-Instruct α-sweep (MBPP-500)

AlgoSim NAUADC / EA / DA@10 (Claude-Sonnet-4.6 judge, correct samples only) joined with our existing structural / CodeBLEU / pass@k metrics. Sorted by NAUADC descending.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **pless_alpha_a5.0_t1.0** | pless_alpha @ T=1.0 | 0.403 | 0.532 | 0.008 | 0.304 | **1.119** | 1.104 | 1.128 | 266 |
| **pless_alpha_a3.0_t1.0** | pless_alpha @ T=1.0 | 0.407 | 0.508 | 0.002 | 0.235 | **1.077** | 1.069 | 1.083 | 254 |
| **pless_alpha_a2.5_t1.0** | pless_alpha @ T=1.0 | 0.412 | 0.492 | 0.004 | 0.192 | **1.045** | 1.037 | 1.049 | 246 |
| **pless_alpha_a2.0_t1.0** | pless_alpha @ T=1.0 | 0.418 | 0.442 | 0.000 | 0.068 | **1.009** | 1.008 | 1.009 | 221 |
