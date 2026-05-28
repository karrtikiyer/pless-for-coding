#!/bin/bash
# APPS entropy probe — generates the per-position softmax sidecar for the
# central survival-vs-entropy figure, extended to APPS.
#
# Mirrors run_gsm8k_entropy_probe.sh's protocol exactly, adapted for
# APPS's task-native conventions (paper-replica chat-template prompts).
# Once this lands, regenerate the central figure with:
#
#   uv run python -m bench.eval.entropy_survival_curves \
#     --models deepseek-ai--deepseek-coder-6.7b-instruct \
#     --datasets apps \
#     --output-dir results/entropy_probe/_central_figure_apps
#
# Locked parameters (decided with user 2026-05-28):
# - Model:           deepseek-ai/deepseek-coder-6.7b-instruct
# - Bucket:          CODEFORCES / introductory (matches Phase A scope)
# - Problems:        299 (full bucket; matches Phase A nucleus & α arms)
# - Samples/problem: 10 (matches MBPP/GSM8K entropy protocol)
# - max_new_tokens:  1024 (matches paper, larger than the 512 used in
#                          MBPP/GSM8K — APPS solutions are longer)
# - Method:          pless (= α=2 baseline; matches existing MBPP/GSM8K
#                            entropy data — survival curves at α=2 and α=5
#                            are computed post-hoc from logged top-32 probs)
# - Backend:         hf (vLLM doesn't support entropy_log)
# - dtype:           bfloat16 (matches MBPP/GSM8K entropy)
# - Prompt:          paper-replica (chat-template-wrapped Deepseek format
#                                   from sh0416/outputs-apps — matches
#                                   Phase A nucleus and α-arm runs)
# - HF batch size:   10 (single-chunk at N=10 — chunking is a no-op)
#
# Outputs:
#   results/pless_alpha_entropy/apps/deepseek-ai--deepseek-coder-6.7b-instruct/
#     pless_t1.0.jsonl                  (regular samples, ~30-100 MB)
#     pless_t1.0.jsonl.entropy.jsonl    (sidecar, ~1.5-3.4 GB — see below)
#
# Sidecar size estimate:
#   299 problems × 10 samples × ~300-800 tokens generated each =
#     ~1-2.4M positions × ~1.1 KB/position (top-32 probs + indices) =
#     ~1.1-2.7 GB on disk
#
# Estimated GPU cost (H100):
#   ~4-12 hr wallclock for full 299 × 10 × 1024 generation
#
# Usage:
#   ./run_apps_entropy_probe.sh                # full run
#   SMOKE=1 ./run_apps_entropy_probe.sh        # 5 problems × 1 sample, fast pipeline check
#   MAX_PROBLEMS=20 ./run_apps_entropy_probe.sh  # override scope
#
# Pre-flight (~5 min):
#   SMOKE=1 ./run_apps_entropy_probe.sh
#   Then inspect:
#     - results/pless_alpha_entropy/apps/.../pless_t1.0.jsonl                (5 records)
#     - results/pless_alpha_entropy/apps/.../pless_t1.0.jsonl.entropy.jsonl  (~hundreds of rows)
#     - Confirm sidecar schema: task_id, sample_id, position, sigma_p2/p3/p5, max_p, top32_probs, top32_indices

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-coder-6.7b-instruct}"
SOURCE="${SOURCE:-CODEFORCES}"
DIFFICULTY="${DIFFICULTY:-introductory}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
HF_BATCH_SIZE="${HF_BATCH_SIZE:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_entropy/apps}"
LOG_DIR="${LOG_DIR:-/tmp/apps_entropy_logs}"
mkdir -p "$LOG_DIR"

# Smoke override — fast pipeline check (~5 min)
if [ "${SMOKE:-0}" = "1" ]; then
  N_SAMPLES=1
  MAX_NEW_TOKENS=512
  MAX_PROBLEMS_FLAG="--max-problems 5"
  RESULTS_DIR="${RESULTS_DIR}_smoke"
  echo "[SMOKE MODE] 5 problems × 1 sample × max_new_tokens=$MAX_NEW_TOKENS"
else
  MAX_PROBLEMS_FLAG="${MAX_PROBLEMS:+--max-problems $MAX_PROBLEMS}"
fi

# Env hygiene — matches other HF-backend APPS drivers.
export MPLBACKEND="${MPLBACKEND:-Agg}"  # avoid GUI popups from hallucinated matplotlib imports

echo "═══════════════════════════════════════════════════════════════════════"
echo "APPS entropy probe — $MODEL on $SOURCE/$DIFFICULTY"
echo "  Problems:       $([ "${SMOKE:-0}" = "1" ] && echo 5 || echo "299 (or MAX_PROBLEMS override)")"
echo "  Samples/task:   $N_SAMPLES"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Method:         pless (α=2 baseline; survival at α=5 computed post-hoc)"
echo "  Backend:        hf"
echo "  HF batch size:  $HF_BATCH_SIZE"
echo "  Results dir:    $RESULTS_DIR/<slug>/${SOURCE}_${DIFFICULTY}/"
echo "  Log dir:        $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

LOG="$LOG_DIR/$(echo "$MODEL" | tr / -)_${SOURCE}_${DIFFICULTY}.log"

uv run python -m bench.apps \
  --model "$MODEL" \
  --source "$SOURCE" --difficulty "$DIFFICULTY" \
  --method pless \
  --temperature "$TEMPERATURE" \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --backend hf --dtype bfloat16 \
  --hf-batch-size "$HF_BATCH_SIZE" \
  --paper-replica-model "$MODEL" \
  --results-dir "$RESULTS_DIR" \
  --log-entropy \
  $MAX_PROBLEMS_FLAG \
  2>&1 | tee "$LOG"

echo
echo "═══════════════════════════════════════════════════════════════════════"
SLUG="$(echo "$MODEL" | tr / -)"
OUT="$RESULTS_DIR/$SLUG/${SOURCE}_${DIFFICULTY}"
echo "Done. Outputs:"
echo "  $OUT/pless_t1.0.jsonl                ($(ls -la "$OUT/pless_t1.0.jsonl" 2>/dev/null | awk '{print $5}') bytes)"
echo "  $OUT/pless_t1.0.jsonl.entropy.jsonl  ($(ls -la "$OUT/pless_t1.0.jsonl.entropy.jsonl" 2>/dev/null | awk '{print $5}') bytes)"
echo
echo "Next: generate the central figure"
echo "  uv run python -m bench.eval.entropy_survival_curves \\"
echo "    --models $SLUG \\"
echo "    --datasets apps \\"
echo "    --output-dir results/entropy_probe/_central_figure_apps"
echo "═══════════════════════════════════════════════════════════════════════"
