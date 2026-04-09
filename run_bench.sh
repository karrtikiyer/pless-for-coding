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
  "pless 0.7"
  "pless_norm 0.7"
  "pless 1.0"
  "pless_norm 1.0"
  "top_p 0.9"
  "top_h 1.0"
  "top_nsigma 1.0"
  "greedy 1.0"
  "beam4 1.0"
  "beam8 1.0"
)

echo "=== Benchmark: $MODEL_ID on GPU $GPU_ID ==="

for cfg in "${CONFIGS[@]}"; do
  read -r method val <<< "$cfg"
  echo ""
  if [ "$method" = "top_p" ]; then
    echo "--- method=$method top_p=$val ---"
    echo "Started: $(date)"
    uv run python -m bench --model "$MODEL_ID" --method top_p --top-p "$val" --temperature 1.0 --mbpp-config full
  elif [ "$method" = "greedy" ]; then
    echo "--- method=$method ---"
    echo "Started: $(date)"
    uv run python -m bench --model "$MODEL_ID" --method greedy --temperature "$val" --n-samples 1 --mbpp-config full
  elif [[ "$method" == beam* ]]; then
    num_beams="${method#beam}"
    echo "--- method=beam num_beams=$num_beams ---"
    echo "Started: $(date)"
    uv run python -m bench --model "$MODEL_ID" --method beam --num-beams "$num_beams" --temperature "$val" --n-samples 1 --mbpp-config full
  else
    echo "--- method=$method temperature=$val ---"
    echo "Started: $(date)"
    uv run python -m bench --model "$MODEL_ID" --method "$method" --temperature "$val" --mbpp-config full
  fi
  echo "Finished: $(date)"
done

echo ""
echo "=== All configurations complete ==="
