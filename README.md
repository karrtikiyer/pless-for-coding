# Pless Samplers — Code Generation Benchmarks

Benchmark the **pless** and **pless-norm** hyperparameter-free sampling methods on MBPP and HumanEval code generation benchmarks. Reproduces comparisons from ["A Thorough Examination of Decoding Methods in the Era of LLMs"](https://arxiv.org/abs/2402.06925) across multiple model families.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.4 (tested on RTX 4090, ~15GB VRAM per model)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/karrtikiyer/pless-for-coding.git
cd pless-for-coding
git submodule update --init   # clones p-less/ sampler (github.com/ryttry/p-less-sampling)
uv sync                       # installs all dependencies with CUDA torch
```

## Supported Models

| Model | Type | HuggingFace ID | Benchmarks |
|-------|------|----------------|------------|
| Codestral-22B | Base | `mistralai/Codestral-22B-v0.1` | HumanEval, MBPP |
| Qwen2.5-Coder-7B | Base | `Qwen/Qwen2.5-Coder-7B` | HumanEval |
| Qwen2.5-Coder-7B-Instruct | Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct` | HumanEval, MBPP |
| Qwen3-Coder-30B-A3B-Instruct | Instruct | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | HumanEval |
| CodeLlama-7B | Base | `codellama/CodeLlama-7b-hf` | HumanEval |
| CodeLlama-7B-Instruct | Instruct | `codellama/CodeLlama-7b-Instruct-hf` | HumanEval |
| Qwen2.5-7B | Base | `Qwen/Qwen2.5-7B` | MBPP |
| Llama-2-7B | Base | `meta-llama/Llama-2-7b-hf` | MBPP |
| Llama-2-7B-Chat | Chat | `meta-llama/Llama-2-7b-chat-hf` | MBPP |
| Qwen-7B | Base | `Qwen/Qwen-7B` | MBPP |
| Qwen-7B-Chat | Chat | `Qwen/Qwen-7B-Chat` | MBPP |

Chat/instruct models are auto-detected by model name (contains "chat", "instruct", or "coder") and use `tokenizer.apply_chat_template()` for prompt formatting.

## Download Models

Download models into `models/` or use HuggingFace model IDs directly (downloads to cache on first run):

```bash
# Local download
huggingface-cli download Qwen/Qwen2.5-7B --local-dir models/qwen257b

# Or use HF IDs directly
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless
```

## Benchmarks

### MBPP

257 problems (sanitized split) or 500 problems (full split). 10 samples per problem. Three sampling methods:

| Method | Description |
|--------|-------------|
| `temp` | Vanilla temperature sampling via `model.generate()` |
| `pless` | Hyperparameter-free pless sampler |
| `pless_norm` | Normalized pless sampler |

```bash
# Single run
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method temp --temperature 0.7

# All 5 configs for a model
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method temp --temperature 0.7
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless --temperature 0.6
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless_norm --temperature 0.6
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless --temperature 1.0
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless_norm --temperature 1.0

# Full MBPP (500 problems)
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless --mbpp-config full
```

**Batch runner:** Run all 5 configs for a model with a single command:

```bash
bash run_bench.sh <model_id> [gpu_id]
```

This auto-detects legacy Qwen models and switches the `transformers` version accordingly. Each config runs full MBPP (500 problems) with 10 samples.

### HumanEval

164 problems from OpenAI's HumanEval dataset. Three sampling methods across multiple temperatures (14 configs total):

| Method | Temperatures |
|--------|-------------|
| `temp` (vanilla temperature) | 0.7, 1.0 |
| `pless` | 0.7, 1.0, 1.5, 2.0, 2.5, 3.0 |
| `pless_norm` | 0.7, 1.0, 1.5, 2.0, 2.5, 3.0 |

**Run a single config:**

```bash
uv run python -m bench.humaneval --model models/qwen257b --method pless --temperature 1.5
```

**Run all 14 configs for a model (loads model once):**

```bash
uv run python run_humaneval.py --model models/qwen257b
```

The orchestration script loads the model once and runs all 14 (method, temperature) combinations sequentially, avoiding repeated model loading.

## CLI Options

### MBPP (`python -m bench`)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model ID or local path |
| `--method` | (required) | `pless`, `pless_norm`, or `temp` |
| `--n-samples` | 10 | Number of samples per problem |
| `--max-new-tokens` | 512 | Max tokens per sample |
| `--temperature` | 1.0 | Temperature applied to logits before softmax |
| `--max-problems` | all | Limit number of problems (for testing) |
| `--no-resume` | false | Start fresh, delete existing results |
| `--no-stop` | false | Disable stop sequences (for debugging) |
| `--mbpp-config` | `sanitized` | `sanitized` (257 problems) or `full` (500 problems) |
| `--results-dir` | `results/` | Output directory |

### HumanEval (`python -m bench.humaneval`)

Same flags as MBPP. All three methods (`pless`, `pless_norm`, `temp`) are supported.

### Orchestration (`run_humaneval.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model ID or local path |
| `--n-samples` | 10 | Number of samples per problem |
| `--max-new-tokens` | 512 | Max tokens per sample |
| `--max-problems` | all | Limit number of problems (for testing) |
| `--no-resume` | false | Start fresh for all configs |
| `--results-dir` | `results/` | Output directory |

## Quick Smoke Tests

```bash
# MBPP — pless sampler
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method pless --n-samples 2 --max-new-tokens 50 --max-problems 3

# MBPP — temp baseline
uv run python -m bench --model meta-llama/Llama-2-7b-hf --method temp --n-samples 2 --max-new-tokens 50 --max-problems 3

# MBPP — Qwen-7B (verifies trust_remote_code)
uv run python -m bench --model Qwen/Qwen-7B --method pless --n-samples 2 --max-new-tokens 50 --max-problems 3

# HumanEval (single config)
uv run python -m bench.humaneval --model models/qwen257b --method pless --temperature 1.0 --n-samples 2 --max-new-tokens 50 --max-problems 3

# HumanEval (all 14 configs, quick)
uv run python run_humaneval.py --model models/qwen257b --n-samples 1 --max-new-tokens 50 --max-problems 1
```

## Output

Results are saved as JSONL files:

- **MBPP**: `results/<model-name>/pless_t1.0.jsonl`
- **HumanEval**: `results/<model-name>/humaneval/<method>_t<temperature>.jsonl`

### MBPP record format

```json
{
  "model": "models/qwen257b",
  "method": "pless",
  "temperature": 1.0,
  "task_id": 11,
  "prompt_text": "Write a python function to ...",
  "samples": ["def func(...): ...", "..."],
  "test_list": ["assert func(...) == ..."],
  "timestamp": "2026-02-25T12:00:00+00:00"
}
```

### HumanEval record format

```json
{
  "model": "models/qwen257b",
  "method": "pless",
  "temperature": 1.0,
  "task_id": "HumanEval/0",
  "prompt_text": "from typing import List\ndef has_close_elements...",
  "samples": ["def has_close_elements(...): ...", "..."],
  "test": "def check(candidate):\n    assert ...",
  "entry_point": "has_close_elements",
  "timestamp": "2026-02-26T12:00:00+00:00"
}
```

## Evaluation

After generating samples, evaluate them to compute pass@k metrics and generate comparison reports.

### Compute metrics for a single results file

```bash
uv run python -m bench.eval --results-file results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/pless_t0.6.jsonl --dataset mbpp
```

This extracts Python code from each sample, runs it against test cases in a sandbox, and computes pass@k (k=1,3,5,10) and cover@t metrics. Results are saved as a JSON file in a `metrics/` subdirectory next to the input file.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--results-file` | (required) | Path to JSONL results file |
| `--dataset` | (required) | `mbpp` or `humaneval` |
| `--k` | `1,3,5,10` | k values for pass@k |
| `--t` | `0.1,0.3,0.5,0.7` | Fractional t values for cover@t |
| `--timeout` | `5.0` | Per-sample execution timeout (seconds) |
| `--workers` | `4` | Number of parallel workers |

### Generate comparison reports and charts

```bash
uv run python -m bench.eval.visualize --model-family all
```

Reads metrics JSONs and generates per model family (`llama`, `codellama`, `qwen`, or `all`):
- `comparison_report.md` — ranked pass@1 table with extended metrics
- `{family}_full_mbpp.csv` — tabular export
- `figures/pass_at_1_comparison.png` — horizontal bar chart
- `figures/metrics_overview.png` — faceted line plots

Output goes to `results/pless_full_mbpp_results/analysis/{family}/`.

### Results

Full MBPP results (500 problems, 10 samples each) for 6 models are in `results/pless_full_mbpp_results/`, with per-model JSONL files, metrics, and analysis reports.

## Checkpoint / Resume

Runs save after each problem. If interrupted, re-run the same command and it skips completed problems automatically. Use `--no-resume` to start fresh.

## Running on Multiple Machines

To parallelize across two GPUs/machines:

- **Machine A**: Run the base model combinations
- **Machine B**: Run the instruct model combinations

Then merge the `results/` directories.

## Project Structure

```
pless-for-coding/
├── pyproject.toml              # Dependencies (torch, transformers, datasets, etc.)
├── run_humaneval.py            # Orchestration: all 14 HumanEval configs, model loaded once
├── run_bench.sh                # Batch MBPP runner: all 5 configs per model
├── debug_generation.py         # Debugging script for model generation
├── p-less/                     # Original pless sampler code (git submodule, do not modify)
│   └── p_less_samplers.py
├── bench/                      # Benchmarking package
│   ├── sampler_bridge.py       # Imports pless samplers via sys.path
│   ├── generator.py            # Token-by-token generation + standard model.generate()
│   ├── checkpointing.py        # JSONL streaming writes + resume logic
│   ├── prompts.py              # MBPP prompt formatting (base vs instruct)
│   ├── runner.py               # MBPP CLI entry point (temp, pless, pless_norm methods)
│   ├── eval/                   # Evaluation pipeline
│   │   ├── __main__.py         # CLI: compute pass@k metrics for a results file
│   │   ├── executor.py         # Code extraction, sandboxed execution
│   │   ├── metrics.py          # pass@k and cover@t computation
│   │   ├── loader.py           # JSONL results loader
│   │   ├── fingerprint.py      # AST fingerprinting for structural diversity
│   │   ├── plots.py            # Chart generation helpers
│   │   ├── report.py           # Markdown report generation
│   │   └── visualize.py        # Comparison reports and charts across model families
│   └── humaneval/              # HumanEval benchmark
│       ├── prompts.py          # HumanEval prompt formatting (base vs instruct)
│       └── runner.py           # HumanEval CLI entry point + run_benchmark()
├── compare_pass_at_k.py        # Compare pass@k between pipelines
├── models/                     # Local model weights (gitignored)
└── results/                    # Benchmark results (JSONL + analysis)
    └── pless_full_mbpp_results/  # Full MBPP results for 6 models × 5 methods
```
