#!/usr/bin/env bash
set -e

DRY_RUN=false
GPU_ID="0"

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *)         GPU_ID="$arg" ;;
  esac
done

RESULTS_DIR="results/pless_human_eval_results/temprature_results"
LEGACY_MODELS=("Qwen/Qwen-7B" "Qwen/Qwen-7B-Chat")

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Collect model IDs from result directories
MODELS=()
for dir in "$RESULTS_DIR"/*/; do
  dirname=$(basename "$dir")
  # Skip non-model dirs
  [[ "$dirname" == "analysis"* ]] && continue
  model_id="${dirname/--//}"
  MODELS+=("$model_id")
done

echo "=== HumanEval top_p=0.9 sweep across ${#MODELS[@]} models on GPU $GPU_ID$($DRY_RUN && echo ' (DRY RUN)') ==="

# Validate CLI contract before any uv add or GPU work
echo "Validating CLI contract with bench.humaneval runner..."
if ! uv run python -c "
import sys
sys.argv = ['bench.humaneval', '--model', 'x', '--method', 'top_p', '--top-p', '0.9']
from bench.humaneval.runner import parse_args
parse_args()
"; then
  echo "ERROR: bench.humaneval runner does not accept --method top_p — check bench/humaneval/runner.py"
  exit 1
fi
echo "  CLI OK"

for MODEL_ID in "${MODELS[@]}"; do
  echo ""
  echo ">>> Model: $MODEL_ID"

  # Legacy detection
  is_legacy=false
  for m in "${LEGACY_MODELS[@]}"; do
    [[ "$MODEL_ID" == "$m" ]] && is_legacy=true
  done

  if $is_legacy; then
    echo ">>> Legacy model — transformers <5"
    $DRY_RUN && echo "  (would run: uv add 'transformers<5,>=4.37')"
    $DRY_RUN || uv add 'transformers<5,>=4.37'
  else
    echo ">>> Modern model — transformers >=5"
    $DRY_RUN && echo "  (would run: uv add 'transformers>=5')"
    $DRY_RUN || uv add 'transformers>=5'
  fi

  CMD=("uv" "run" "python" "-m" "bench.humaneval" "--model" "$MODEL_ID" "--method" "top_p" "--top-p" "0.9" "--temperature" "1.0")
  echo "  CMD: ${CMD[*]}"
  if ! $DRY_RUN; then
    echo "Started: $(date)"
    "${CMD[@]}"
    echo "Finished: $(date)"
  fi
done

echo ""
echo "=== HumanEval top_p sweep complete ==="
