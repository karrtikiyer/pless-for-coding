# algosim Full Comparison — Qwen3-8B Configs (excl. temp_standard)

Single view across our existing diversity metrics (`struct_div`, `codebleu_div`), the new algosim NAUADC, and pass@k. All configs that use `temp_standard` (top_p=0.95, top_k=20) on any phase are excluded; the remaining 14 configs span the pure-temp split-decoding family plus every non-split baseline.

NAUADC values are shown only where algosim has been run; remaining configs are marked `—`.

| Config | Label | pass@1 | pass@3 | pass@5 | pass@10 | struct_div | codebleu_div | NAUADC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **No-thinking baselines** |||||||||
| A | temp 0.7 | 0.662 | 0.704 | 0.719 | 0.734 | 0.057 | 0.137 | **1.096** |
| B | pless 0.7 | 0.669 | 0.673 | 0.674 | 0.674 | 0.007 | 0.015 | **1.015** |
| **Thinking-only baselines (no split)** |||||||||
| C | temp_think 0.6 | 0.738 | 0.799 | 0.817 | 0.834 | 0.167 | 0.354 | **1.234** |
| D | pless_think 0.6 | 0.718 | 0.777 | 0.795 | 0.816 | 0.131 | 0.256 | **1.156** |
| E | pless_norm_think 0.6 | 0.719 | 0.784 | 0.802 | 0.822 | 0.124 | 0.245 | **1.151** |
| **Uniform high-temp thinking (no split)** |||||||||
| T15N | uniform temp 1.5 (native, thinking) | 0.799 | 0.857 | 0.872 | 0.888 | 0.200 | 0.384 | **1.277** |
| P15 | uniform pless 1.5 (thinking) | 0.824 | 0.872 | 0.885 | 0.898 | 0.159 | 0.296 | **1.222** |
| **Pure-temp split baseline (no pless)** |||||||||
| T15P | split: temp(pure) 1.5 → temp(pure) 1.5 | 0.801 | 0.855 | 0.868 | 0.882 | 0.208 | 0.390 | **1.303** |
| **Pure-temp split + pless on code (think@1.5, code∈{1.0,1.5,2.0,3.0})** |||||||||
| H7P | split: temp(pure) 1.5 → pless 1.0 | 0.805 | 0.869 | 0.890 | 0.910 | 0.214 | 0.398 | **1.312** |
| H8P | split: temp(pure) 1.5 → pless 1.5 | 0.811 | 0.869 | 0.883 | 0.898 | 0.206 | 0.384 | **1.322** |
| H9P | split: temp(pure) 1.5 → pless 2.0 | 0.807 | 0.871 | 0.889 | 0.908 | 0.217 | 0.401 | **1.319** |
| H10P | split: temp(pure) 1.5 → pless 3.0 | 0.803 | 0.865 | 0.885 | 0.906 | 0.202 | 0.389 | **1.283** |
| **Pure-temp split stress tests (think>1.5)** |||||||||
| H11P | split: temp(pure) 2.0 → pless 3.0 | 0.458 | 0.655 | 0.732 | 0.802 | 0.201 | 0.370 | **1.198** |
| H12P | split: temp(pure) 2.5 → pless 3.0 | 0.242 | 0.498 | 0.621 | 0.736 | 0.201 | 0.327 | **1.145** |
