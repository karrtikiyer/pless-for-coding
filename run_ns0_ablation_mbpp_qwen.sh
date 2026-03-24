#!/usr/bin/env bash
# run_ns0_ablation_mbpp_qwen.sh
# Zero-shot (--n-shots 0) ablation for Qwen-7B on MBPP full.
# Isolates the example vs format-change effect for Qwen-7B.
# Parallel to the Llama-2-7b ns0 ablation (temp_ns0_t0.7, pless_ns0_t0.6).
#
# Comparison points (existing H100 runs):
#   pless_ns0_t0.6  vs  pless_t0.6  (35.4%)
#   temp_ns0_t0.7   vs  temp_t0.7   (29.8%)

set -euo pipefail

MODEL_ID="Qwen/Qwen-7B"
RESULTS_DIR="results/pless_full_mbpp_results"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "Pinning transformers<5 for legacy Qwen..."
uv add 'transformers<5,>=4.37'

for METHOD_TEMP in "pless:0.6" "temp:0.7"; do
    METHOD="${METHOD_TEMP%%:*}"
    TEMP="${METHOD_TEMP##*:}"
    LOG_FILE="${LOG_DIR}/qwen7b_ns0_${METHOD}_t${TEMP}.log"

    echo "Running ${METHOD} @ t=${TEMP} --n-shots 0 → ${LOG_FILE}"
    uv run python -m bench \
        --model "$MODEL_ID" \
        --method "$METHOD" \
        --temperature "$TEMP" \
        --n-shots 0 \
        --mbpp-config full \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$LOG_FILE"
    echo "Done: ${METHOD}_ns0_t${TEMP}"
done

echo "Restoring transformers>=5..."
uv add 'transformers>=5'

echo ""
echo "All done. Results in ${RESULTS_DIR}/Qwen--Qwen-7B/"
echo "Next: evaluate both files:"
echo "  uv run python -m bench.eval --results-file ${RESULTS_DIR}/Qwen--Qwen-7B/pless_ns0_t0.6.jsonl --dataset mbpp"
echo "  uv run python -m bench.eval --results-file ${RESULTS_DIR}/Qwen--Qwen-7B/temp_ns0_t0.7.jsonl --dataset mbpp"
