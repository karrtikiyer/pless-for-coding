#!/usr/bin/env bash
# Experiment: P-less with post-truncation temperature on MBPP-full.
#
# Tests decoupled temperature: T₁ (pre-truncation, controls pruning aggressiveness)
# + T₂ (post-truncation, controls sampling uniformity among survivors).
#
# Model: Qwen2.5-Coder-3B (BigCode prompt format)
# Variant: pless only (pless_norm skipped — statistical tie)
#
# Baselines already computed (no post-temp):
#   pless_bigcode_t0.6.jsonl, pless_bigcode_t1.0.jsonl
#
# Usage: bash run_pless_posttemp_mbpp.sh

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-Coder-3B"
PROMPT_STYLE="bigcode"
METHOD="pless"
MBPP_CONFIG="full"

# Format: "T1 T2" where T2=none means standard pless (no post-temp)
CONFIGS=(
  "0.8 none"     # new baseline: pless at T₁=0.8 without post-temp
  "0.6 2.0"      # aggressive pruning + moderate flattening
  "0.6 5.0"      # aggressive pruning + strong flattening
  "0.8 2.0"      # moderate pruning + moderate flattening
  "0.8 5.0"      # moderate pruning + strong flattening
  "1.0 2.0"      # natural pruning + moderate flattening
  "1.0 5.0"      # natural pruning + strong flattening
)

echo "=== P-less Post-Temperature Experiment ==="
echo "Model: ${MODEL_ID}"
echo "Configs: ${#CONFIGS[@]}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r t1 t2 <<< "$cfg"

  POST_ARG=""
  LABEL="T₁=${t1}"
  if [ "$t2" != "none" ]; then
    POST_ARG="--post-temperature $t2"
    LABEL="${LABEL}, T₂=${t2}"
  fi

  echo "--- Running: ${LABEL} ---"
  uv run python -m bench \
    --model "$MODEL_ID" \
    --method "$METHOD" \
    --temperature "$t1" \
    --prompt-style "$PROMPT_STYLE" \
    --mbpp-config "$MBPP_CONFIG" \
    $POST_ARG

  echo ""
done

echo "=== All configs complete ==="
echo "Run evaluation with:"
echo "  uv run python -m bench.eval.consolidated_eval --dataset mbpp --force"
