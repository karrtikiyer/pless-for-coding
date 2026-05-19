# algosim Diversity Report — Qwen3-8B Split Decoding

AlgoSim NAUADC / EA / DA@10 (Llama-3.1-8B-Instruct judge, correct samples only) joined with our existing structural / CodeBLEU / pass@k metrics. Sorted by NAUADC descending.

Scope: baseline configs (no thinking, thinking, uniform pless) plus the **pure-temp** split-decoding series (`temp_pure` on the `<think>` phase). `temp_standard` (top_p=0.95, top_k=20) configs are deliberately excluded to keep the comparison clean.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC | EA | DA@10 | n_problems |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
