#!/usr/bin/env bash
# run_bigcode_mbpp_opencodeinterpreter13b.sh
# Benchmark OpenCodeInterpreter-DS-1.3B on MBPP full (500 tasks, n=10) with two prompt styles.
#
# Model: m-a-p/OpenCodeInterpreter-DS-1.3B (based on DeepSeek-Coder-1.3B base)
#
# Phase A — BigCode zero-shot docstring format (matches arXiv 2507.03160):
#   Direct comparison target: OpenCodeInterpreter-1.3B pass@1 = 0.44
#   (their config: temp=0.2/top-p=0.95, n=10, unbiased estimator — same methodology)
#   First config replicates paper exactly to validate pipeline before pless runs.
#
# Phase B — Paper 3-shot [BEGIN]/[DONE] format:
#   Internal cross-model comparison.
#
# Phase A: 7 configs (replication + pless@0.6, pless@1.0, pless_norm@0.6, pless_norm@1.0, temp@0.7, top_p@0.9)
# Phase B: 6 configs (pless@0.6, pless@1.0, pless_norm@0.6, pless_norm@1.0, temp@0.7, top_p@0.9)

set -euo pipefail

MODEL_ID="m-a-p/OpenCodeInterpreter-DS-1.3B"
RESULTS_DIR="results/pless_full_mbpp_results"
MODEL_DIR="m-a-p--OpenCodeInterpreter-DS-1.3B"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "Ensuring transformers>=5..."
uv add 'transformers>=5'

# ── Phase A: BigCode zero-shot docstring format ────────────────────────────────

echo ""
echo "=== Phase A: BigCode prompt (arXiv 2507.03160 comparison) ==="

# Paper-replication config — run first to validate pipeline reproduces baseline (~0.44)
LOG_FILE="${LOG_DIR}/oci13b_bigcode_replication.log"
echo ""
echo "--- top_p @ 0.95 t=0.2 bigcode [PAPER REPLICATION — expected pass@1 ~0.44] ---"
echo "Started: $(date)"
uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_p \
    --top-p 0.95 \
    --temperature 0.2 \
    --prompt-style bigcode \
    --mbpp-config full \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_FILE"
echo "Finished: $(date)"

for METHOD_TEMP in "pless:0.6" "pless:1.0" "pless_norm:0.6" "pless_norm:1.0" "temp:0.7"; do
    METHOD="${METHOD_TEMP%%:*}"
    TEMP="${METHOD_TEMP##*:}"
    LOG_FILE="${LOG_DIR}/oci13b_bigcode_${METHOD}_t${TEMP}.log"

    echo ""
    echo "--- ${METHOD} @ t=${TEMP} bigcode ---"
    echo "Started: $(date)"
    uv run python -m bench \
        --model "$MODEL_ID" \
        --method "$METHOD" \
        --temperature "$TEMP" \
        --prompt-style bigcode \
        --mbpp-config full \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$LOG_FILE"
    echo "Finished: $(date)"
done

LOG_FILE="${LOG_DIR}/oci13b_bigcode_top_p0.9.log"
echo ""
echo "--- top_p @ 0.9 bigcode ---"
echo "Started: $(date)"
uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_p \
    --top-p 0.9 \
    --temperature 1.0 \
    --prompt-style bigcode \
    --mbpp-config full \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_FILE"
echo "Finished: $(date)"

echo ""
echo "=== Phase A complete. Evaluating bigcode results... ==="

for f in \
    top_p0.95_bigcode_t0.2.jsonl \
    pless_bigcode_t0.6.jsonl \
    pless_bigcode_t1.0.jsonl \
    pless_norm_bigcode_t0.6.jsonl \
    pless_norm_bigcode_t1.0.jsonl \
    temp_bigcode_t0.7.jsonl \
    top_p0.9_bigcode_t1.0.jsonl; do
    echo "Evaluating $f ..."
    uv run python -m bench.eval \
        --results-file "${RESULTS_DIR}/${MODEL_DIR}/${f}" \
        --dataset mbpp
done

# ── Phase B: Paper 3-shot [BEGIN]/[DONE] format ───────────────────────────────

echo ""
echo "=== Phase B: Paper 3-shot format (cross-model comparison) ==="

for METHOD_TEMP in "pless:0.6" "pless:1.0" "pless_norm:0.6" "pless_norm:1.0" "temp:0.7"; do
    METHOD="${METHOD_TEMP%%:*}"
    TEMP="${METHOD_TEMP##*:}"
    LOG_FILE="${LOG_DIR}/oci13b_paper_${METHOD}_t${TEMP}.log"

    echo ""
    echo "--- ${METHOD} @ t=${TEMP} paper ---"
    echo "Started: $(date)"
    uv run python -m bench \
        --model "$MODEL_ID" \
        --method "$METHOD" \
        --temperature "$TEMP" \
        --mbpp-config full \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$LOG_FILE"
    echo "Finished: $(date)"
done

LOG_FILE="${LOG_DIR}/oci13b_paper_top_p0.9.log"
echo ""
echo "--- top_p @ 0.9 paper ---"
echo "Started: $(date)"
uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_p \
    --top-p 0.9 \
    --temperature 1.0 \
    --mbpp-config full \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_FILE"
echo "Finished: $(date)"

echo ""
echo "=== Phase B complete. Evaluating paper results... ==="

for f in \
    pless_t0.6.jsonl \
    pless_t1.0.jsonl \
    pless_norm_t0.6.jsonl \
    pless_norm_t1.0.jsonl \
    temp_t0.7.jsonl \
    top_p0.9_t1.0.jsonl; do
    echo "Evaluating $f ..."
    uv run python -m bench.eval \
        --results-file "${RESULTS_DIR}/${MODEL_DIR}/${f}" \
        --dataset mbpp
done

echo ""
echo "=== All done. Results in ${RESULTS_DIR}/${MODEL_DIR}/ ==="
echo ""
echo "Pipeline check (should reproduce arXiv 2507.03160 baseline pass@1 = 0.44):"
echo "  ${RESULTS_DIR}/${MODEL_DIR}/metrics/top_p0.95_bigcode_t0.2_metrics.json"
echo ""
echo "Key pless comparisons:"
echo "  ${RESULTS_DIR}/${MODEL_DIR}/metrics/pless_bigcode_t0.6_metrics.json"
echo "  ${RESULTS_DIR}/${MODEL_DIR}/metrics/pless_norm_bigcode_t0.6_metrics.json"
