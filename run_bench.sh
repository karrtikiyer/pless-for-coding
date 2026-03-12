#!/usr/bin/env bash
set -e

MODEL_ID="${1:?Usage: bash run_bench.sh <model_id> [gpu_id]}"
GPU_ID="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Models that require transformers 4.x (incompatible with 5.x remote code)
LEGACY_MODELS=("Qwen/Qwen-7B" "Qwen/Qwen-7B-Chat")

is_legacy=false
for m in "${LEGACY_MODELS[@]}"; do
  [[ "$MODEL_ID" == "$m" ]] && is_legacy=true
done

if $is_legacy; then
  echo ">>> Legacy model detected — installing transformers <5"
  uv add 'transformers<5,>=4.37'
else
  echo ">>> Modern model — ensuring transformers >=5"
  uv add 'transformers>=5'
fi

CONFIGS=(
  "temp 0.7"
  "pless 0.6"
  "pless_norm 0.6"
  "pless 1.0"
  "pless_norm 1.0"
)

echo "=== Benchmark: $MODEL_ID on GPU $GPU_ID ==="

for cfg in "${CONFIGS[@]}"; do
  read -r method temp <<< "$cfg"
  echo ""
  echo "--- method=$method temperature=$temp ---"
  echo "Started: $(date)"
  uv run python -m bench --model "$MODEL_ID" --method "$method" --temperature "$temp" --mbpp-config full
  echo "Finished: $(date)"
done

echo ""
echo "=== All configurations complete ==="
