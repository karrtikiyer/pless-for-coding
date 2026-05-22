#!/bin/bash
# Cross-domain entropy probe — pilot.
#
# Tests the bimodal-entropy mechanism hypothesis on non-code domains.
# For each (model, dataset) pair, greedy-generates one completion per
# problem and teacher-forces the (prompt + completion) tokens, logging
# per-token entropy. Output: results/entropy_probe/<model>/<dataset>/.
#
# Cost: ~2.5 GPU-hours for the default 2 × 3 grid × 50 problems.
#
# Usage:
#   ./run_entropy_probe_cross_domain.sh                     # full grid
#   MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" ./run_...       # one model
#   DATASETS="gsm8k math" ./run_...                          # skip MBPP baseline
#   MAX_PROBLEMS=10 ./run_...                                # smoke
#
# Env overrides:
#   MODELS         space-separated HF ids (default 2 models below)
#   DATASETS       space-separated dataset keys (default "mbpp gsm8k math")
#   MAX_PROBLEMS   per (model, dataset) cell (default 50)
#   OUTPUT_DIR     (default results/entropy_probe)
#   DTYPE          bfloat16 | float16  (default bfloat16)
#   MAX_NEW_TOKENS (default 512 — enough for chain-of-thought without
#                  burning budget on degenerate repetition)
#   LOG_DIR        (default /tmp/entropy_probe_logs)

set -euo pipefail

MODELS_DEFAULT="Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen3-8B"
DATASETS_DEFAULT="mbpp gsm8k math"

MODELS="${MODELS:-$MODELS_DEFAULT}"
DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-results/entropy_probe}"
DTYPE="${DTYPE:-bfloat16}"
LOG_DIR="${LOG_DIR:-/tmp/entropy_probe_logs}"

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "Cross-domain entropy probe"
echo "  Models:        $MODELS"
echo "  Datasets:      $DATASETS"
echo "  Max problems:  $MAX_PROBLEMS per cell"
echo "  Max tokens:    $MAX_NEW_TOKENS"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Log dir:       $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

for MODEL in $MODELS; do
  MODEL_SLUG=$(echo "$MODEL" | tr '/' '-')
  for DS in $DATASETS; do
    LOG="$LOG_DIR/${MODEL_SLUG}_${DS}.log"
    echo
    echo "───── $MODEL / $DS  (log: $LOG) ─────"
    uv run python -m bench.entropy_probe \
      --model "$MODEL" \
      --dataset "$DS" \
      --max-problems "$MAX_PROBLEMS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output-dir "$OUTPUT_DIR" \
      --dtype "$DTYPE" \
      2>&1 | tee "$LOG"
  done
done

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "All cells done. Inspect:"
for MODEL in $MODELS; do
  MODEL_SLUG=$(echo "$MODEL" | tr '/' '-')
  for DS in $DATASETS; do
    echo "  $OUTPUT_DIR/$MODEL_SLUG/$DS/dip_test.json"
  done
done
echo "═══════════════════════════════════════════════════════════════════════"
