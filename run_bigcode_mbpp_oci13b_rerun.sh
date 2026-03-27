#!/usr/bin/env bash
# run_bigcode_mbpp_oci13b_rerun.sh
#
# Re-runs OpenCodeInterpreter-DS-1.3B MBPP benchmarks in BigCode format,
# matching the methodology of arXiv 2507.03160.
#
# Root cause of previous bad results: transformers 5.x loads LlamaTokenizer
# which overrides the ByteLevel decoder from tokenizer.json, destroying
# whitespace. Fixed in bench/generator.py: auto-detects broken round-trip
# and reloads as PreTrainedTokenizerFast.
#
# is_instruct_model() no longer matches OCI, so --prompt-style bigcode
# produces *_bigcode_*.jsonl output files (same as all other base models).
#
# Configs: 7 (replication + pless@0.6, pless@1.0, pless_norm@0.6,
#              pless_norm@1.0, temp@0.7, top_p@0.9)
#
# Expected: top_p0.95_bigcode_t0.2 pass@1 should be close to paper's 44%.

set -euo pipefail

MODEL_ID="m-a-p/OpenCodeInterpreter-DS-1.3B"
RESULTS_DIR="results/pless_full_mbpp_results"
MODEL_DIR="${RESULTS_DIR}/m-a-p--OpenCodeInterpreter-DS-1.3B"
LOG_DIR="logs"
BACKUP_DIR="${MODEL_DIR}/archive_broken_tokenizer_$(date +%Y%m%d)"

mkdir -p "$LOG_DIR"

echo "=== OCI-DS-1.3B BigCode re-run (fixed tokenizer) ==="
echo ""
echo "Backing up old results to: ${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"
for f in "${MODEL_DIR}"/*.jsonl; do
    [ -f "$f" ] && mv "$f" "${BACKUP_DIR}/"
done
if [ -d "${MODEL_DIR}/metrics" ]; then
    mkdir -p "${BACKUP_DIR}/metrics"
    for f in "${MODEL_DIR}/metrics"/*_metrics.json; do
        [ -f "$f" ] && mv "$f" "${BACKUP_DIR}/metrics/"
    done
fi
echo "Backup complete."

echo ""
echo "Ensuring transformers>=5..."
uv add 'transformers>=5'

# ── 7 configs — all use BigCode format ──────────────────────────────────────

echo ""
echo "--- top_p @ 0.95 t=0.2 [PAPER REPLICATION — expected pass@1 ~0.44] ---"
echo "Started: $(date)"
uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_p \
    --top-p 0.95 \
    --temperature 0.2 \
    --prompt-style bigcode \
    --mbpp-config full \
    --results-dir "$RESULTS_DIR" \
    --no-resume \
    2>&1 | tee "${LOG_DIR}/oci13b_replication.log"
echo "Finished: $(date)"

for METHOD_TEMP in "pless:0.6" "pless:1.0" "pless_norm:0.6" "pless_norm:1.0" "temp:0.7"; do
    METHOD="${METHOD_TEMP%%:*}"
    TEMP="${METHOD_TEMP##*:}"
    echo ""
    echo "--- ${METHOD} @ t=${TEMP} ---"
    echo "Started: $(date)"
    uv run python -m bench \
        --model "$MODEL_ID" \
        --method "$METHOD" \
        --temperature "$TEMP" \
        --prompt-style bigcode \
        --mbpp-config full \
        --results-dir "$RESULTS_DIR" \
        --no-resume \
        2>&1 | tee "${LOG_DIR}/oci13b_${METHOD}_t${TEMP}.log"
    echo "Finished: $(date)"
done

echo ""
echo "--- top_p @ 0.9 t=1.0 ---"
echo "Started: $(date)"
uv run python -m bench \
    --model "$MODEL_ID" \
    --method top_p \
    --top-p 0.9 \
    --temperature 1.0 \
    --prompt-style bigcode \
    --mbpp-config full \
    --results-dir "$RESULTS_DIR" \
    --no-resume \
    2>&1 | tee "${LOG_DIR}/oci13b_top_p0.9.log"
echo "Finished: $(date)"

echo ""
echo "=== Generation complete. Evaluating... ==="

# With is_instruct_model()=False + --prompt-style bigcode, filenames get _bigcode_ infix
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
        --results-file "${MODEL_DIR}/${f}" \
        --dataset mbpp
done

echo ""
echo "=== Regenerating comparison report ==="
uv run python -m bench.eval.compare_with_2507

echo ""
echo "=== Done. ==="
echo ""
echo "Pipeline check (should be close to paper's 44%):"
echo "  ${MODEL_DIR}/metrics/top_p0.95_bigcode_t0.2_metrics.json"
