# Pless Samplers — Code Generation Benchmarks

Benchmark the **pless** and **pless-norm** hyperparameter-free sampling methods on MBPP and HumanEval code generation benchmarks using Qwen2.5 7B models.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.4 (tested on RTX 4090, ~15GB VRAM per model)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/karrtikiyer/pless-for-coding.git
cd pless-for-coding
git submodule update --init   # pulls p-less/ sampler code
uv sync                       # installs all dependencies with CUDA torch
```

## Download Models

Download the models into the `models/` directory:

```bash
# Base model
huggingface-cli download Qwen/Qwen2.5-7B --local-dir models/qwen257b

# Instruct model
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/Qwen2.5-Coder-7B-Instruct
```

Or use HuggingFace model IDs directly (downloads to cache on first run):

```bash
uv run python -m bench.runner --model Qwen/Qwen2.5-7B --method pless
```

## Benchmarks

### MBPP (Sanitized)

257 problems from the MBPP sanitized test split. 10 samples per problem.

```bash
# Single run
uv run python -m bench --model models/qwen257b --method pless

# All 4 combinations (2 models x 2 methods)
uv run python -m bench --model models/qwen257b --method pless
uv run python -m bench --model models/qwen257b --method pless_norm
uv run python -m bench --model models/Qwen2.5-Coder-7B-Instruct --method pless
uv run python -m bench --model models/Qwen2.5-Coder-7B-Instruct --method pless_norm
```

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
| `--method` | (required) | `pless` or `pless_norm` |
| `--n-samples` | 10 | Number of samples per problem |
| `--max-new-tokens` | 512 | Max tokens per sample |
| `--temperature` | 1.0 | Temperature applied to logits before softmax |
| `--max-problems` | all | Limit number of problems (for testing) |
| `--no-resume` | false | Start fresh, delete existing results |
| `--results-dir` | `results/` | Output directory |

### HumanEval (`python -m bench.humaneval`)

Same flags as MBPP, plus `--method` also accepts `temp` for vanilla temperature sampling via `model.generate()`.

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
# MBPP
uv run python -m bench --model models/qwen257b --method pless --n-samples 2 --max-new-tokens 50 --max-problems 3

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
├── p-less/                     # Original pless sampler code (git submodule, do not modify)
│   └── p_less_samplers.py
├── bench/                      # Benchmarking package
│   ├── sampler_bridge.py       # Imports pless samplers via sys.path
│   ├── generator.py            # Token-by-token generation + standard model.generate()
│   ├── checkpointing.py        # JSONL streaming writes + resume logic
│   ├── prompts.py              # MBPP prompt formatting (base vs instruct)
│   ├── runner.py               # MBPP CLI entry point
│   └── humaneval/              # HumanEval benchmark
│       ├── prompts.py          # HumanEval prompt formatting (base vs instruct)
│       └── runner.py           # HumanEval CLI entry point + run_benchmark()
├── models/                     # Local model weights (gitignored)
└── results/                    # Output JSONL files (gitignored)
```
