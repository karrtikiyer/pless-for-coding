# algosim Diversity Report — OpenCodeInterpreter-DS-1.3B α-sweep (MBPP-500)

AlgoSim NAUADC / EA / DA@10 (Claude-Sonnet-4.6 judge, correct samples only) joined with our existing structural / CodeBLEU / pass@k metrics. Sorted by NAUADC descending.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **pless_alpha_a5.0_t1.0** | pless_alpha @ T=1.0 | 0.463 | 0.664 | 0.342 | 0.541 | **1.209** | 1.169 | 1.229 | 332 |
| **pless_alpha_a3.0_t1.0** | pless_alpha @ T=1.0 | 0.480 | 0.650 | 0.254 | 0.437 | **1.165** | 1.131 | 1.182 | 325 |
| **pless_alpha_a2.5_t1.0** | pless_alpha @ T=1.0 | 0.473 | 0.614 | 0.202 | 0.351 | **1.132** | 1.113 | 1.143 | 307 |
| **pless_alpha_a2.0_t1.0** | pless_alpha @ T=1.0 | 0.477 | 0.554 | 0.090 | 0.175 | **1.073** | 1.064 | 1.079 | 277 |
