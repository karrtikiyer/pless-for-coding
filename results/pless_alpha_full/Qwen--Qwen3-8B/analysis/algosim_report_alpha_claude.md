# algosim Diversity Report — Qwen3-8B (thinking) Rényi-α sweep

AlgoSim NAUADC / EA / DA@10 (claude-sonnet-4-6 judge, correct samples only) joined with our existing structural / CodeBLEU / pass@k metrics. Sorted by NAUADC descending.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **pless_alpha_think_a5.0_t1.0** | pless_alpha @ T=1.0 | 0.737 | 0.828 | 0.168 | 0.332 | **1.120** | 1.087 | 1.135 | 414 |
| **pless_alpha_think_a3.0_t1.0** | pless_alpha @ T=1.0 | 0.728 | 0.826 | 0.161 | 0.321 | **1.102** | 1.077 | 1.114 | 413 |
| **pless_alpha_think_a2.5_t1.0** | pless_alpha @ T=1.0 | 0.732 | 0.834 | 0.151 | 0.307 | **1.097** | 1.073 | 1.108 | 417 |
| **pless_alpha_think_a2.0_t1.0** | pless_alpha @ T=1.0 | 0.717 | 0.820 | 0.127 | 0.265 | **1.075** | 1.059 | 1.083 | 410 |
