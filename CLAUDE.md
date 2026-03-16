# Project: Pless Samplers — Code Generation Benchmarks

Benchmarks the **pless** and **pless-norm** hyperparameter-free sampling methods on MBPP and HumanEval, comparing against vanilla temperature sampling across multiple model families. Based on ["A Thorough Examination of Decoding Methods in the Era of LLMs"](https://arxiv.org/abs/2402.06925).

## Setup Essentials

- **Python >=3.12** required (`requires-python` in pyproject.toml)
- **CUDA 12.4** — torch is pulled from `https://download.pytorch.org/whl/cu124` (non-macOS)
- **`uv sync`** installs everything. Never use pip.
- **`git submodule update --init`** to clone `p-less/` sampler
- **`models/`** is gitignored — download weights locally or use HF model IDs
- **`uv.lock`** is committed for reproducibility

## P-Less Sampler (`p-less/` submodule)

Git remote: `https://github.com/ryttry/p-less-sampling.git` — do not modify directly.

Two functions from `p_less_samplers.py`:
- **`p_less_decode(probs)`** — threshold `p = sum(probs²)` (collision entropy). Tokens with `prob < p` are zeroed, rest renormalized and sampled.
- **`p_less_norm_decode(probs)`** — threshold `p = (v·sum(probs²) - 1) / (v - 1)` where `v` is vocab size. Relaxed variant favoring diversity.
- Both take `(batch_size, vocab_size)` prob tensors, return `(batch_size, 1)` token indices, and modify input in-place.

## Key Architecture

### Generation pipeline (`bench/`)
- `runner.py` — MBPP CLI entry point (`python -m bench`)
- `generator.py` — Token-by-token generation (pless/pless_norm) + `model.generate()` (temp). Heavy monkey-patching at module top for old Qwen compatibility with transformers 5.x (stubs for removed classes, DynamicCache subscript access, etc.)
- `sampler_bridge.py` — Imports pless samplers from the `p-less/` submodule via `sys.path`
- `prompts.py` — MBPP prompt formatting; auto-detects chat/instruct models by name and uses `apply_chat_template()`
- `checkpointing.py` — JSONL streaming writes + resume logic (skips completed task_ids)

### Evaluation pipeline (`bench/eval/`)
- `__main__.py` — CLI: `python -m bench.eval --results-file <file> --dataset mbpp`
- `executor.py` — Code extraction (`extract_python_code()` with `_strip_after_function`, `_strip_check_and_main`, `_strip_code_fences`, `_trim_to_compilable`) + sandboxed execution via subprocess
- `metrics.py` — `compute_pass_at_k()`, `compute_cover_at_t()`
- `fingerprint.py` — AST fingerprinting for structural diversity (uses `zss` Zhang-Shasha tree edit distance)
- `loader.py` — JSONL results loader
- `visualize.py` — CLI: `python -m bench.eval.visualize --model-family [llama|codellama|qwen|all]`. Generates comparison reports, CSVs, and bar charts under `results/pless_full_mbpp_results/analysis/`
- `plots.py`, `report.py` — Helpers for chart and markdown report generation
- `compare_with_paper.py` — Compares our metrics against published paper numbers (Table 26 of arXiv:2402.06925)
- `compare_with_paper_qwen.py` — Same but focused on Qwen-7B specifically. Output: `results/pless_mbpp_results/analysis/qwen_paper_comparison/`
- `consolidated_eval.py` — Cross-dataset pipeline: auto-discovers all MBPP + HumanEval result files, re-executes code, computes uniform metrics (pass@k + diversity). CLI: `python -m bench.eval.consolidated_eval [--verify-only] [--report-only]`. Writes `results/analysis/consolidated_summary.csv` and `consolidated_report.md`. **Note:** hardcoded paths include `temprature_results` (typo preserved in directory name and code).
- `eval_temperature_sweep.py` — Batch-evaluates all HumanEval JSONL files under `results/pless_human_eval_results/temprature_results/`. Run before `report_temperature_sweep`.
- `report_temperature_sweep.py` — Generates temperature sweep reports/charts (pass@1 vs temperature, heatmaps, structural diversity plots)
- `parse_humaneval.py` — Parses pre-evaluated HumanEval `*_detailed.json` files (already contain pass/fail booleans, no re-execution). CLI: `python -m bench.eval.parse_humaneval --detailed <path> --model <name>`
- `curate_examples.py` — Analyzes 164 HumanEval tasks across all models/methods, selects top p-less wins/losses, writes `curated_examples.md`

Public API exports from `bench/eval/__init__.py`: `check_sample`, `evaluate_task`, `ast_fingerprint`, `load_results`, `compute_pass_at_k`, `compute_cover_at_t`.

### HumanEval (`bench/humaneval/`)
- `runner.py` — CLI: `python -m bench.humaneval`
- `prompts.py` — HumanEval prompt formatting
- `run_humaneval.py` (repo root) — Orchestration: loads model once, runs all 14 configs:
  - `temp` at 0.7, 1.0
  - `pless` at 0.7, 1.0, 1.5, 2.0, 2.5, 3.0
  - `pless_norm` at 0.7, 1.0, 1.5, 2.0, 2.5, 3.0
  - Extra flags: `--task-ids` (specific HumanEval tasks), `--no-stop` (disable stop sequences)

### Root-level scripts
- `run_bench.sh` — Runs all 5 MBPP configs for a model: `temp@0.7, pless@0.6, pless_norm@0.6, pless@1.0, pless_norm@1.0`. Auto-detects legacy Qwen models and switches transformers version.
- `compare_pass_at_k.py` — Pipeline validation: re-runs tests from scratch comparing new JSONL results against old full-precision JSON. Flags tasks with >0.3 absolute pass rate difference. Uses dynamic import of `executor.py` to avoid pulling in `zss`.
- `debug_generation.py` — Compares float32 (CPU) vs bfloat16 (CUDA/MPS) generation for `Qwen2.5-Coder-7B-Instruct`. **Note:** designed for a separate `.venv-debug` environment, not `uv run`.

## Models

11 models supported. Legacy Qwen models (Qwen-7B, Qwen-7B-Chat) require `transformers<5` due to incompatible remote code. All others use `transformers>=5`.

Chat/instruct detection: model name contains "chat", "instruct", or "coder" → uses `tokenizer.apply_chat_template()`.

## Results Directory Structure

Three top-level buckets (directory names use `--` as HF separator, e.g. `Qwen--Qwen-7B`):

| Path | Contents |
|------|----------|
| `results/pless_full_mbpp_results/` | Full MBPP (500 problems × 10 samples × 5 configs) for 6 models. Per-model JSONL + `metrics/` + `analysis/` |
| `results/pless_mbpp_results/` | Earlier/partial MBPP results + paper comparison analysis |
| `results/pless_human_eval_results/full_precision_results/` | Full-precision HumanEval JSON (`*_detailed.json`) for 4 models |
| `results/pless_human_eval_results/temprature_results/` | Temperature sweep HumanEval JSONL for 6 models (**typo "temprature" is baked into paths**) |

Also: `metrics_before_fix/` and `analysis_before_fix/` directories are snapshots from before the base model eval fix (commit `839de92`).

## Known Gotchas & Hard-Won Fixes

1. **Old Qwen compatibility** — Qwen-7B and Qwen-7B-Chat need `transformers<5,>=4.37`, eager attention (no SDPA), `transformers-stream-generator` package (PyPI name), and extensive monkey-patching in `generator.py` (stubs for `DisjunctiveConstraint`, `BeamSearchScorer`, `GenerateOutput`, `SampleOutput`, `DynamicCache.__getitem__`, `PreTrainedModel.get_head_mask`).

2. **Instruct prompt tokenization** — Chat templates must be tokenized directly (not text → tokenize) to preserve special tokens like `<|im_start|>`.

3. **Code extraction pipeline** — Models often generate extra content after the target function (test harnesses, "next problem" continuations). The extraction pipeline strips these: `_strip_after_function` (for base models generating continuations), `_strip_check_and_main` (for models generating `def check()`/`if __name__`), `_trim_to_compilable` (progressive truncation to find compilable prefix).

4. **`eos_token_id=None`** — Old Qwen tokenizers can return `None` for `eos_token_id`. Manual generation must handle this gracefully.

5. **DynamicCache API change** — transformers 5.x removed subscript access (`past_key_values[i]`). Patched in `generator.py`.

6. **`torch.compile` disabled** — Incompatible with transformers 5.x KV cache implementation.

7. **"temprature" typo** — Directory name `temprature_results` and all code referencing it preserve this typo. Do not "fix" it without renaming the directory and updating all scripts.

## Key Dependencies

- `torch` (CUDA 12.4), `transformers`, `datasets`, `accelerate`
- `tiktoken`, `einops`, `sentencepiece` — tokenizer support for various models
- `transformers-stream-generator` — required by old Qwen remote code
- `zss` — Zhang-Shasha tree edit distance, used by `fingerprint.py` for structural diversity
- `hf-transfer` — fast HF downloads (activate with `HF_HUB_ENABLE_HF_TRANSFER=1`)
- `matplotlib`, `numpy` — plotting

## Development

- Use `uv` for all package management (never pip)
- Run scripts: `uv run python -m bench ...`
- GPU: tested on RTX 4090, ~15GB VRAM per model
- The `p-less/` directory is a git submodule — do not modify directly
- No `[project.scripts]` entry points — everything uses `python -m` pattern
