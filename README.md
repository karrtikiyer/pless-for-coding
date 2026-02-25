# Pless Samplers — MBPP Benchmark

Benchmark the **pless** and **pless-norm** hyperparameter-free sampling methods on the MBPP code generation benchmark using Qwen2.5 7B models.

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

## Running the Benchmark

Each run generates 10 samples per problem on the MBPP sanitized test split (257 problems).

### All 4 combinations

```bash
# Base model (Qwen2.5-7B)
uv run python -m bench.runner --model models/qwen257b --method pless
uv run python -m bench.runner --model models/qwen257b --method pless_norm

# Instruct model (Qwen2.5-Coder-7B-Instruct)
uv run python -m bench.runner --model models/Qwen2.5-Coder-7B-Instruct --method pless
uv run python -m bench.runner --model models/Qwen2.5-Coder-7B-Instruct --method pless_norm
```

### CLI Options

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

### Quick Smoke Test

```bash
uv run python -m bench.runner --model models/qwen257b --method pless --n-samples 2 --max-new-tokens 50 --max-problems 3
```

## Output

Results are saved as JSONL files under `results/<model-name>/<method>_t<temperature>.jsonl`.

Each line is a JSON object:

```json
{
  "model": "models/qwen257b",
  "method": "pless",
  "temperature": 1.0,
  "task_id": 11,
  "prompt_text": "Write a python function to ...",
  "samples": ["def func(...): ...", "def func(...): ..."],
  "test_list": ["assert func(...) == ..."],
  "timestamp": "2026-02-25T12:00:00+00:00"
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
├── p-less/                     # Original pless sampler code (git submodule, do not modify)
│   └── p_less_samplers.py
├── bench/                      # Benchmarking package
│   ├── sampler_bridge.py       # Imports pless samplers via sys.path
│   ├── prompts.py              # Prompt formatting (base vs instruct)
│   ├── generator.py            # Token-by-token generation with KV cache reuse
│   ├── checkpointing.py        # JSONL streaming writes + resume logic
│   └── runner.py               # CLI entry point
├── models/                     # Local model weights (gitignored)
└── results/                    # Output JSONL files (gitignored)
```
