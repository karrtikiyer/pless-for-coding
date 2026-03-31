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
from bench.models import MBPP_MODELS
for m in MBPP_MODELS:
    print(f'{m.model_id}|{m.prompt_style}|{m.legacy}')
")

echo "=== New samplers (pless@0.7, pless_norm@0.7, greedy, beam4, beam8) on GPU $GPU_ID$($DRY_RUN && echo ' (DRY RUN)') ==="

# Validate CLI contract before any uv add or GPU work
echo "Validating CLI contract..."
if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'greedy', '--temperature', '1.0', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
sys.argv = ['bench', '--model', 'x', '--method', 'beam', '--num-beams', '4', '--temperature', '1.0', '--mbpp-config', 'full']
parse_args()
"; then
  echo "ERROR: bench runner does not accept --method greedy/beam — check bench/runner.py"
  exit 1
fi
echo "  CLI OK"

# Configs: "method temperature extra_args"
CONFIGS=(
  "pless 0.7"
  "pless_norm 0.7"
  "greedy 1.0"
  "beam 1.0 --num-beams 4"
  "beam 1.0 --num-beams 8"
)

while IFS='|' read -r MODEL_ID PROMPT_STYLE IS_LEGACY; do
  echo ""
  echo ">>> Model: $MODEL_ID (prompt_style=$PROMPT_STYLE, legacy=$IS_LEGACY)"

  # Legacy transformers version handling
  if [ "$IS_LEGACY" = "True" ]; then
    echo ">>> Legacy model — transformers <5"
    $DRY_RUN && echo "  (would run: uv add 'transformers<5,>=4.37')"
    $DRY_RUN || uv add 'transformers<5,>=4.37'
  else
    echo ">>> Modern model — transformers >=5"
    $DRY_RUN && echo "  (would run: uv add 'transformers>=5')"
    $DRY_RUN || uv add 'transformers>=5'
  fi

  # Build prompt-style args
  PROMPT_ARGS=()
  if [ "$PROMPT_STYLE" = "bigcode" ]; then
    PROMPT_ARGS=("--prompt-style" "bigcode")
  fi

  for cfg in "${CONFIGS[@]}"; do
    read -r method temp extra <<< "$cfg"
    echo ""
    echo "--- method=$method temperature=$temp ${extra:+$extra} ---"

    CMD=("uv" "run" "python" "-m" "bench"
         "--model" "$MODEL_ID"
         "--method" "$method"
         "--temperature" "$temp"
         "--mbpp-config" "full")
    [[ ${#PROMPT_ARGS[@]} -gt 0 ]] && CMD+=("${PROMPT_ARGS[@]}")

    # Greedy: override n-samples to 1
    if [ "$method" = "greedy" ]; then
      CMD+=("--n-samples" "1")
    fi

    # Append extra args (e.g. --num-beams 4)
    if [ -n "$extra" ]; then
      read -ra EXTRA_ARGS <<< "$extra"
      CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "  CMD: ${CMD[*]}"
    if ! $DRY_RUN; then
      echo "Started: $(date)"
      "${CMD[@]}"
      echo "Finished: $(date)"
    fi
  done
done <<< "$MODEL_CONFIGS"

echo ""
echo "=== All new sampler configs complete ==="
