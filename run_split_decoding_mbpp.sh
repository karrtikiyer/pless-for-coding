#!/usr/bin/env bash
# Split decoding experiment on Qwen3-8B: different samplers for <think> vs code phases.
#
# 7 configs:
#   A) temp 0.7, no thinking       — baseline
#   B) pless 0.7, no thinking      — baseline
#   C) temp 0.6, thinking          — unified temp + thinking
#   D) pless 0.6, thinking         — unified pless + thinking
#   E) pless_norm 0.6, thinking    — unified pless_norm + thinking
#   F) split: temp_std think + pless code, thinking    — core experiment
#   G) split: temp_std think + pless_norm code, thinking — variant
#
# Temperature rationale (Qwen3-8B model card):
#   Non-thinking mode: T=0.7 (configs A, B)
#   Thinking mode:     T=0.6 (configs C-G)
#
# Usage:
#   bash run_split_decoding_mbpp.sh [--gpu 0] [--dry-run]
set -e

DRY_RUN=false
GPU_ID="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true; shift ;;
    --gpu)      GPU_ID="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"

MODEL="Qwen/Qwen3-8B"
RESULTS_DIR="results/pless_full_mbpp_results"
N_SAMPLES=10
MAX_TOKENS_NO_THINK=512   # standard budget for code-only generation
MAX_TOKENS_THINK=4096     # generous ceiling: don't truncate long reasoning chains
MBPP_CONFIG="full"

echo "=== Split decoding experiment on Qwen3-8B ==="
echo "    GPU: $GPU_ID | n_samples: $N_SAMPLES"
echo "    max_tokens: $MAX_TOKENS_NO_THINK (no-think) / $MAX_TOKENS_THINK (think)"
$DRY_RUN && echo "    (DRY RUN)"

# ---------------------------------------------------------------------------
# Validate CLI contract before any GPU work
# ---------------------------------------------------------------------------
echo ""
echo "Validating CLI contract..."
if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'split',
            '--sampler-think', 'temp_standard', '--sampler-code', 'pless',
            '--temp-think', '0.6', '--temp-code', '0.6',
            '--enable-thinking', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
"; then
  echo "ERROR: CLI contract validation failed — check bench/runner.py"
  exit 1
fi
echo "  CLI OK"

# ---------------------------------------------------------------------------
run_cmd() {
  local cmd=("$@")
  echo "  CMD: ${cmd[*]}"
  if ! $DRY_RUN; then
    echo "  Started: $(date)"
    "${cmd[@]}"
    echo "  Finished: $(date)"
  fi
}

# ---------------------------------------------------------------------------
# Config A: temp 0.7, no thinking (baseline)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config A: temp 0.7, no thinking"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method temp \
  --temperature 0.7 \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_NO_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Config B: pless 0.7, no thinking (baseline)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config B: pless 0.7, no thinking"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method pless \
  --temperature 0.7 \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_NO_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Config C: temp 0.6, thinking (unified)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config C: temp 0.6, thinking"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method temp \
  --temperature 0.6 \
  --enable-thinking \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Config D: pless 0.6, thinking (unified)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config D: pless 0.6, thinking"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method pless \
  --temperature 0.6 \
  --enable-thinking \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Config E: pless_norm 0.6, thinking (unified)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config E: pless_norm 0.6, thinking"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method pless_norm \
  --temperature 0.6 \
  --enable-thinking \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Config F: split — temp_standard think + pless code (core experiment)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config F: split — temp_standard think + pless code"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method split \
  --sampler-think temp_standard \
  --sampler-code pless \
  --temp-think 0.6 \
  --temp-code 0.6 \
  --enable-thinking \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Config G: split — temp_standard think + pless_norm code (variant)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Config G: split — temp_standard think + pless_norm code"
echo "========================================"
run_cmd uv run python -m bench \
  --model "$MODEL" \
  --method split \
  --sampler-think temp_standard \
  --sampler-code pless_norm \
  --temp-think 0.6 \
  --temp-code 0.6 \
  --enable-thinking \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_TOKENS_THINK" \
  --mbpp-config "$MBPP_CONFIG" \
  --results-dir "$RESULTS_DIR"

echo ""
echo "=== All 7 configs complete ==="
