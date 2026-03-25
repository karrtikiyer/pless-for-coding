#!/usr/bin/env bash
# run_bs_ablation_mbpp_qwen.sh
# begin_scaffold ablation on the 43 paper-only tasks for Qwen-7B on MBPP full.
# Tests whether [BEGIN]+scaffold suppresses the one-liner bias seen in hybrid format.
#
# Baseline:
#   hybrid (pless_hybrid_t0.6):  0/43 tasks pass (0%)
#   paper  (pless_t0.6):        43/43 tasks pass (100% by construction)
# Decision threshold: ≥30% (13+ tasks) → proceed to full 500-task run

set -euo pipefail

MODEL_ID="Qwen/Qwen-7B"
RESULTS_DIR="results/pless_full_mbpp_results"
LOG_DIR="logs"
PAPER_ONLY_IDS="27 49 50 57 68 70 82 85 108 130 131 154 178 186 197 208 210 212 228 247 261 270 330 338 350 353 357 362 393 403 416 449 450 451 459 463 466 478 482 486 496 504 509"

mkdir -p "$LOG_DIR"

echo "Pinning transformers<5 for legacy Qwen..."
uv add 'transformers<5,>=4.37'

LOG_FILE="${LOG_DIR}/qwen7b_bs_t0.6_targeted.log"
echo "Running pless begin_scaffold @ t=0.6 on 43 paper-only tasks → ${LOG_FILE}"
uv run python -m bench \
    --model "$MODEL_ID" \
    --method pless \
    --temperature 0.6 \
    --prompt-style begin_scaffold \
    --task-ids $PAPER_ONLY_IDS \
    --mbpp-config full \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "Done. Evaluating..."
uv run python -m bench.eval \
    --results-file "${RESULTS_DIR}/Qwen--Qwen-7B/pless_bs_t0.6.jsonl" \
    --dataset mbpp

echo "Restoring transformers>=5..."
uv add 'transformers>=5'

echo ""
echo "All done."
echo "Compare:"
echo "  hybrid (0/43): results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/pless_hybrid_t0.6_metrics.json"
echo "  begin_scaffold: results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/pless_bs_t0.6_metrics.json"
