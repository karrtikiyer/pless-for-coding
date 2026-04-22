#!/usr/bin/env bash
# Reproduce the top-3 stochastic decoding methods (temperature, top-p, top-k)
# from "A Thorough Examination of Decoding Methods in the Era of LLMs"
# (arXiv:2402.06925) on MBPP-full (500 tasks) with per-model hyperparameters
# from Tables 26, 27, and 30.
#
# Usage:
#   bash run_paper_stochastic_mbpp.sh --family llama2    [--gpu 0] [--dry-run]
#   bash run_paper_stochastic_mbpp.sh --family codellama [--gpu 0] [--dry-run]
#   bash run_paper_stochastic_mbpp.sh --family qwen      [--gpu 0] [--dry-run]
#   bash run_paper_stochastic_mbpp.sh --family all       [--gpu 0] [--dry-run]
#
# Run one family per machine (each has its own GPU + venv).
set -e

DRY_RUN=false
GPU_ID="0"
FAMILY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true; shift ;;
    --gpu)      GPU_ID="$2"; shift 2 ;;
    --family)   FAMILY="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$FAMILY" ]; then
  echo "ERROR: --family is required (llama2 | codellama | qwen | all)"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# ---------------------------------------------------------------------------
# Per-model configs from paper Tables 26 (CodeLlama), 27 (Llama-2), 30 (Qwen)
# Format: MODEL_ID|TEMP|TOP_P|TOP_K|LEGACY
# ---------------------------------------------------------------------------
LLAMA2_MODELS=(
  "meta-llama/Llama-2-7b-hf|0.3|0.8|5|false"
  "meta-llama/Llama-2-7b-chat-hf|0.3|0.8|5|false"
)

CODELLAMA_MODELS=(
  "codellama/CodeLlama-7b-hf|0.6|0.8|5|false"
  "codellama/CodeLlama-7b-Instruct-hf|0.3|0.8|5|false"
)

QWEN_MODELS=(
  "Qwen/Qwen-7B|0.1|0.85|5|true"
  "Qwen/Qwen-7B-Chat|0.2|0.85|50|true"
)

# Select models based on family
case "$FAMILY" in
  llama2)    MODELS=("${LLAMA2_MODELS[@]}") ;;
  codellama) MODELS=("${CODELLAMA_MODELS[@]}") ;;
  qwen)      MODELS=("${QWEN_MODELS[@]}") ;;
  all)       MODELS=("${LLAMA2_MODELS[@]}" "${CODELLAMA_MODELS[@]}" "${QWEN_MODELS[@]}") ;;
  *)         echo "ERROR: unknown family '$FAMILY' (llama2 | codellama | qwen | all)"; exit 1 ;;
esac

echo "=== Paper stochastic methods (temp/top-p/top-k) on MBPP-full ==="
echo "    Family: $FAMILY | GPU: $GPU_ID$($DRY_RUN && echo ' (DRY RUN)')"

# ---------------------------------------------------------------------------
# Validate CLI contract before any GPU work
# ---------------------------------------------------------------------------
echo "Validating CLI contract..."
if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'top_k', '--top-k', '5', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
"; then
  echo "ERROR: bench runner does not accept --method top_k — check bench/runner.py"
  exit 1
fi
echo "  CLI OK"

# ---------------------------------------------------------------------------
# Install correct transformers version for this family
# ---------------------------------------------------------------------------
NEEDS_LEGACY=false
for entry in "${MODELS[@]}"; do
  IFS='|' read -r _ _ _ _ IS_LEGACY <<< "$entry"
  [ "$IS_LEGACY" = "true" ] && NEEDS_LEGACY=true
done

if [ "$FAMILY" = "all" ]; then
  echo ">>> Family 'all': will switch transformers version as needed"
elif $NEEDS_LEGACY; then
  echo ">>> Installing transformers <5 for legacy Qwen models"
  $DRY_RUN || uv add 'transformers<5,>=4.37'
else
  echo ">>> Installing transformers >=5"
  $DRY_RUN || uv add 'transformers>=5'
fi

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
CURRENT_TF=""

run_cmd() {
  local cmd=("$@")
  echo "  CMD: ${cmd[*]}"
  if ! $DRY_RUN; then
    echo "  Started: $(date)"
    "${cmd[@]}"
    echo "  Finished: $(date)"
  fi
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_ID TEMP TOP_P TOP_K IS_LEGACY <<< "$entry"
  SHORT_NAME="${MODEL_ID##*/}"

  echo ""
  echo "========================================"
  echo ">>> Model: $SHORT_NAME ($MODEL_ID)"
  echo ">>>   temp=$TEMP  top_p=$TOP_P  top_k=$TOP_K"
  echo "========================================"

  # Handle transformers version switch for --family all
  if [ "$FAMILY" = "all" ]; then
    if [ "$IS_LEGACY" = "true" ] && [ "$CURRENT_TF" != "legacy" ]; then
      echo ">>> Switching to transformers <5 for legacy model"
      $DRY_RUN || uv add 'transformers<5,>=4.37'
      CURRENT_TF="legacy"
    elif [ "$IS_LEGACY" = "false" ] && [ "$CURRENT_TF" != "modern" ]; then
      echo ">>> Switching to transformers >=5"
      $DRY_RUN || uv add 'transformers>=5'
      CURRENT_TF="modern"
    fi
  fi

  # --- Temperature ---
  echo ""
  echo "  --- temp=$TEMP ---"
  run_cmd uv run python -m bench \
    --model "$MODEL_ID" \
    --method temp \
    --temperature "$TEMP" \
    --mbpp-config full

  # --- Top-p (temperature=1.0, matching paper's generate.py) ---
  echo ""
  echo "  --- top_p=$TOP_P (temperature=1.0) ---"
  run_cmd uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_p \
    --top-p "$TOP_P" \
    --temperature 1.0 \
    --mbpp-config full

  # --- Top-k (temperature=1.0, matching paper's generate.py) ---
  echo ""
  echo "  --- top_k=$TOP_K (temperature=1.0) ---"
  run_cmd uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_k \
    --top-k "$TOP_K" \
    --temperature 1.0 \
    --mbpp-config full

done

echo ""
echo "=== Family '$FAMILY' complete ==="
