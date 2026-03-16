# Project: Pless Samplers — Code Generation Benchmarks

Benchmarks the **pless** and **pless-norm** hyperparameter-free sampling methods on MBPP and HumanEval, comparing against vanilla temperature sampling across multiple model families.

## Key Architecture

### Generation pipeline (`bench/`)
- `runner.py` — MBPP CLI entry point (`python -m bench`)
- `generator.py` — Token-by-token generation (pless/pless_norm) + `model.generate()` (temp). Heavy monkey-patching at module top for old Qwen compatibility with transformers 5.x (stubs for removed classes, DynamicCache subscript access, etc.)
- `sampler_bridge.py` — Imports pless samplers from the `p-less/` git submodule via `sys.path`
- `prompts.py` — MBPP prompt formatting; auto-detects chat/instruct models by name and uses `apply_chat_template()`
- `checkpointing.py` — JSONL streaming writes + resume logic (skips completed task_ids)

### Evaluation pipeline (`bench/eval/`)
- `__main__.py` — CLI: `python -m bench.eval --results-file <file> --dataset mbpp`
- `executor.py` — Code extraction (`extract_python_code()` with `_strip_after_function`, `_strip_check_and_main`, `_strip_code_fences`, `_trim_to_compilable`) + sandboxed execution via subprocess
- `metrics.py` — `compute_pass_at_k()`, `compute_cover_at_t()`
- `fingerprint.py` — AST fingerprinting for structural diversity measurement
- `loader.py` — JSONL results loader
- `visualize.py` — CLI: `python -m bench.eval.visualize --model-family [llama|codellama|qwen|all]`. Generates comparison reports, CSVs, and bar charts under `results/pless_full_mbpp_results/analysis/`
- `plots.py`, `report.py` — Helpers for chart and markdown report generation
- `compare_with_paper.py` — Compares our metrics against published paper numbers

### HumanEval (`bench/humaneval/`)
- `runner.py` — CLI: `python -m bench.humaneval`
- `prompts.py` — HumanEval prompt formatting
- `run_humaneval.py` (repo root) — Orchestration: runs all 14 (method, temperature) configs, loading model once

### Batch runner
- `run_bench.sh` — Runs all 5 MBPP configs for a model: `temp@0.7, pless@0.6, pless_norm@0.6, pless@1.0, pless_norm@1.0`. Auto-detects legacy Qwen models and switches transformers version.

## Models

11 models supported. Legacy Qwen models (Qwen-7B, Qwen-7B-Chat) require `transformers<5` due to incompatible remote code. All others use `transformers>=5`.

Chat/instruct detection: model name contains "chat", "instruct", or "coder" → uses `tokenizer.apply_chat_template()`.

## Known Gotchas & Hard-Won Fixes

1. **Old Qwen compatibility** — Qwen-7B and Qwen-7B-Chat need `transformers<5,>=4.37`, eager attention (no SDPA), `stream-generator` package, and extensive monkey-patching in `generator.py` (stubs for `DisjunctiveConstraint`, `BeamSearchScorer`, `GenerateOutput`, `SampleOutput`, `DynamicCache.__getitem__`, `PreTrainedModel.get_head_mask`).

2. **Instruct prompt tokenization** — Chat templates must be tokenized directly (not text → tokenize) to preserve special tokens like `<|im_start|>`.

3. **Code extraction pipeline** — Models often generate extra content after the target function (test harnesses, "next problem" continuations). The extraction pipeline strips these: `_strip_after_function` (for base models generating continuations), `_strip_check_and_main` (for models generating `def check()`/`if __name__`), `_trim_to_compilable` (progressive truncation to find compilable prefix).

4. **`eos_token_id=None`** — Old Qwen tokenizers can return `None` for `eos_token_id`. Manual generation must handle this gracefully.

5. **DynamicCache API change** — transformers 5.x removed subscript access (`past_key_values[i]`). Patched in `generator.py`.

6. **`torch.compile` disabled** — Incompatible with transformers 5.x KV cache implementation.

## Results

- Full MBPP results (500 problems × 10 samples × 5 configs) for 6 models in `results/pless_full_mbpp_results/`
- HumanEval results under `results/<model-name>/humaneval/`
- Metrics JSONs in `metrics/` subdirectories next to result JSONL files
- Analysis reports and charts under `results/pless_full_mbpp_results/analysis/`

## Development

- Use `uv` for all package management (never pip)
- Run scripts: `uv run python -m bench ...`
- GPU: tested on RTX 4090, ~15GB VRAM per model
- The `p-less/` directory is a git submodule — do not modify directly
