# algosim Diversity Report — Qwen3-8B Split Decoding

AlgoSim NAUADC / EA / DA@10 (Llama-3.1-8B-Instruct judge, correct samples only) joined with our existing structural / CodeBLEU / pass@k metrics. Sorted by NAUADC descending.

Scope: baseline configs (no thinking, thinking, uniform pless) plus the **pure-temp** split-decoding series (`temp_pure` on the `<think>` phase). `temp_standard` (top_p=0.95, top_k=20) configs are deliberately excluded to keep the comparison clean.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **H8P** | split: temp(pure) 1.5 → pless 1.5 | 0.811 | 0.898 | 0.206 | 0.384 | **1.322** | 1.229 | 1.365 | 449 |
| **H9P** | split: temp(pure) 1.5 → pless 2.0 | 0.807 | 0.908 | 0.217 | 0.401 | **1.319** | 1.235 | 1.359 | 454 |
| **H7P** | split: temp(pure) 1.5 → pless 1.0 | 0.805 | 0.910 | 0.214 | 0.398 | **1.312** | 1.218 | 1.354 | 455 |
| **T15P** | split: temp(pure) 1.5 → temp(pure) 1.5 | 0.801 | 0.882 | 0.208 | 0.390 | **1.303** | 1.215 | 1.342 | 441 |
| **H10P** | split: temp(pure) 1.5 → pless 3.0 | 0.803 | 0.906 | 0.202 | 0.389 | **1.283** | 1.205 | 1.318 | 453 |
| **T15N** | uniform temp 1.5 (native, thinking) | 0.799 | 0.888 | 0.200 | 0.384 | **1.277** | 1.194 | 1.315 | 444 |
| **C** | temp_think 0.6 | 0.738 | 0.834 | 0.167 | 0.354 | **1.234** | 1.166 | 1.264 | 417 |
| **P15** | uniform pless 1.5 (thinking) | 0.824 | 0.898 | 0.159 | 0.296 | **1.222** | 1.159 | 1.252 | 449 |
| **H11P** | split: temp(pure) 2.0 → pless 3.0 | 0.458 | 0.802 | 0.201 | 0.370 | **1.198** | 1.157 | 1.214 | 401 |
| **D** | pless_think 0.6 | 0.718 | 0.816 | 0.131 | 0.256 | **1.156** | 1.114 | 1.174 | 408 |
| **E** | pless_norm_think 0.6 | 0.719 | 0.822 | 0.124 | 0.245 | **1.151** | 1.104 | 1.170 | 411 |
| **H12P** | split: temp(pure) 2.5 → pless 3.0 | 0.242 | 0.736 | 0.201 | 0.327 | **1.145** | 1.128 | 1.152 | 368 |
| **A** | temp 0.7 | 0.662 | 0.734 | 0.057 | 0.137 | **1.096** | 1.067 | 1.109 | 367 |
| **B** | pless 0.7 | 0.669 | 0.674 | 0.007 | 0.015 | **1.015** | 1.008 | 1.018 | 337 |
