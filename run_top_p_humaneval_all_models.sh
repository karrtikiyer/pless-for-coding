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

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Read model configs from bench.models registry (no folder scanning)
# Each line: model_id|prompt_style|legacy
MODEL_CONFIGS=$(uv run python -c "
from bench.models import HUMANEVAL_MODELS
for m in HUMANEVAL_MODELS:
    print(f'{m.model_id}|{m.prompt_style}|{m.legacy}')
")

echo "=== HumanEval top_p=0.9 sweep on GPU $GPU_ID$($DRY_RUN && echo ' (DRY RUN)') ==="

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

while IFS='|' read -r MODEL_ID PROMPT_STYLE IS_LEGACY; do
  echo ""
  echo ">>> Model: $MODEL_ID"

  if [ "$IS_LEGACY" = "True" ]; then
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
done <<< "$MODEL_CONFIGS"

echo ""
echo "=== HumanEval top_p sweep complete ==="
