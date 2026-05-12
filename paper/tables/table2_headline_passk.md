# Table 2 — Headline pass@k

Per-row format: `pass@1 / pass@10`. `—` = configuration not run for that (model, benchmark) cell. `n=1` cells (greedy, beam) report the single-sample success rate in both columns.

| Model | Benchmark | greedy | temp@0.7 | pless@0.6 | pless_norm@0.6 | pless@1.0 | pless_norm@1.0 |
|:-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen/Qwen-7B | mbpp | 0.368 / 0.368 | 0.097 / 0.398 | 0.314 / 0.382 | 0.313 / 0.382 | 0.350 / 0.486 | 0.357 / 0.502 |
| Qwen/Qwen-7B-Chat | mbpp | 0.314 / 0.314 | 0.287 / 0.504 | 0.340 / 0.370 | 0.344 / 0.376 | 0.344 / 0.406 | 0.345 / 0.408 |
| Qwen/Qwen2.5-Coder-1.5B | mbpp | 0.544 / 0.544 | 0.381 / 0.710 | 0.531 / 0.608 | 0.528 / 0.614 | 0.513 / 0.688 | 0.519 / 0.678 |
| Qwen/Qwen2.5-Coder-3B | mbpp | 0.602 / 0.602 | 0.426 / 0.776 | 0.593 / 0.662 | 0.594 / 0.666 | 0.565 / 0.722 | 0.576 / 0.724 |
| Qwen/Qwen2.5-Coder-7B | humaneval | — | 0.497 / 0.890 | — | — | 0.559 / 0.762 | 0.563 / 0.762 |
| Qwen/Qwen2.5-Coder-7B-Instruct | humaneval | 0.842 / 0.842 | 0.792 / 0.951 | 0.875 / 0.878 | 0.875 / 0.884 | 0.834 / 0.902 | 0.757 / 0.951 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | humaneval | 0.756 / 0.756 | 0.775 / 0.872 | 0.789 / 0.799 | 0.785 / 0.799 | 0.760 / 0.780 | 0.757 / 0.780 |
| codellama/CodeLlama-7b-Instruct-hf | humaneval | 0.360 / 0.360 | 0.363 / 0.634 | 0.281 / 0.317 | 0.281 / 0.317 | 0.355 / 0.378 | 0.351 / 0.384 |
| codellama/CodeLlama-7b-Instruct-hf | mbpp | 0.422 / 0.422 | 0.383 / 0.552 | 0.412 / 0.422 | 0.411 / 0.422 | 0.416 / 0.442 | 0.414 / 0.438 |
| codellama/CodeLlama-7b-hf | mbpp | 0.410 / 0.410 | 0.368 / 0.652 | 0.417 / 0.494 | 0.417 / 0.490 | 0.414 / 0.572 | 0.415 / 0.574 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | mbpp | 0.442 / 0.442 | 0.431 / 0.624 | 0.439 / 0.490 | 0.439 / 0.494 | 0.441 / 0.512 | 0.446 / 0.510 |
| meta-llama/Llama-2-7b-chat-hf | mbpp | 0.206 / 0.206 | 0.178 / 0.302 | 0.205 / 0.214 | 0.204 / 0.214 | 0.201 / 0.224 | 0.202 / 0.222 |
| meta-llama/Llama-2-7b-hf | mbpp | 0.230 / 0.230 | 0.040 / 0.242 | 0.219 / 0.274 | 0.222 / 0.272 | 0.224 / 0.370 | 0.218 / 0.372 |
| mistralai/Codestral-22B-v0.1 | humaneval | 0.756 / 0.756 | 0.730 / 0.908 | 0.751 / 0.787 | 0.749 / 0.793 | 0.780 / 0.848 | 0.777 / 0.848 |

_Source: `results/analysis/consolidated_summary.csv`. Method aliases (`pless`/`p_less`, `temp`/`temp_0.7`) collapsed by `canon_method()` in this script._
