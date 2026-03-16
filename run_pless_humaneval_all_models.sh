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

CONFIGS=(
  "pless 0.6"
  "pless_norm 0.6"
)

# Collect model IDs from result directories
MODELS=()
for dir in "$RESULTS_DIR"/*/; do
  dirname=$(basename "$dir")
  [[ "$dirname" == "analysis"* ]] && continue
  model_id="${dirname/--//}"
  MODELS+=("$model_id")
done

echo "=== HumanEval pless/pless_norm temp=0.6 sweep across ${#MODELS[@]} models on GPU $GPU_ID$($DRY_RUN && echo ' (DRY RUN)') ==="

# Validate CLI contract before any uv add or GPU work
echo "Validating CLI contract with bench.humaneval runner..."
if ! uv run python -c "
import sys
sys.argv = ['bench.humaneval', '--model', 'x', '--method', 'pless', '--temperature', '0.6']
from bench.humaneval.runner import parse_args
parse_args()
"; then
  echo "ERROR: bench.humaneval runner does not accept --method pless — check bench/humaneval/runner.py"
  exit 1
fi
if ! uv run python -c "
import sys
sys.argv = ['bench.humaneval', '--model', 'x', '--method', 'pless_norm', '--temperature', '0.6']
from bench.humaneval.runner import parse_args
parse_args()
"; then
  echo "ERROR: bench.humaneval runner does not accept --method pless_norm — check bench/humaneval/runner.py"
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

  for cfg in "${CONFIGS[@]}"; do
    read -r method temp_val <<< "$cfg"
    CMD=("uv" "run" "python" "-m" "bench.humaneval" "--model" "$MODEL_ID" "--method" "$method" "--temperature" "$temp_val")
    echo "  CMD: ${CMD[*]}"
    if ! $DRY_RUN; then
      echo "Started: $(date)"
      "${CMD[@]}"
      echo "Finished: $(date)"
    fi
  done
done

echo ""
echo "=== HumanEval pless sweep complete ==="
