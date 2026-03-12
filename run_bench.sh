#!/usr/bin/env bash
set -e

MODEL_ID="${1:?Usage: bash run_bench.sh <model_id> [gpu_id]}"
GPU_ID="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

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
