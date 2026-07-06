#!/bin/bash
# Live adaptive loop-rescue (alpha=2 -> live n-gram detect -> chop -> alpha=5), HF token-by-token.
#
# The deployable pipeline, measured FRESH on ATCODER-interview (per-sample looping is
# stochastic, so we cannot reuse saved traces). Each sample yields its plain-alpha=2
# counterfactual for free (baseline_recovered = recovered AND NOT fired), so one run gives
# both the adaptive pass@1 and the no-rescue baseline.
#
# Detector configs are the tuned per-model winners (see scripts/detector_config_choose.py +
# A35/A37): choice favors CATCH over low-FP because chop->alpha=5 forgives false positives
# (a wrongly-chopped good trace keeps thinking at alpha=5), unlike force-</think>.
#   Qwen3-8B            : n=30 k=6 window=1600
#   DeepSeek-R1-Distill : n=30 k=8 window=3000   (candidate: n=40 window=4000)
#
# RUN ON A CUDA GPU (HF token-by-token; vLLM cannot chop mid-stream). Default uv venv.
# Weights: defaults to HF_HUB_OFFLINE=1 (both models were run before → cached). On a FRESH
# pod with no cache, run with HF_HUB_OFFLINE=0 and export HF_TOKEN (e.g. from .env) to allow
# the download. Unlisted driver knobs (TASK_IDS, SOURCE, DIFFICULTY, BASE_ALPHA, ESC_ALPHA,
# MAX_CTX) have defaults in the driver and still pass through via env inheritance.
# SMOKE FIRST (validate the method + token-level detector before the full run):
#   MODEL_KEY=qwen   MAX_PROBLEMS=4 N=4 ./run_live_adaptive_apps.sh
#   MODEL_KEY=deepseek MAX_PROBLEMS=4 N=4 ./run_live_adaptive_apps.sh
# FULL: drop MAX_PROBLEMS and set N=10.
# Env: MODEL_KEY (qwen|deepseek), N, MAX_PROBLEMS, MAX_NEW, plus NGRAM_* to override.

set -euo pipefail

# Cache/env hygiene (parity with the pod vLLM launch scripts).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_KEY="${MODEL_KEY:-qwen}"
case "$MODEL_KEY" in
  qwen)
    MODEL="${MODEL:-Qwen/Qwen3-8B}"
    NGRAM_N="${NGRAM_N:-30}"; NGRAM_K="${NGRAM_K:-6}"; NGRAM_WINDOW="${NGRAM_WINDOW:-1600}" ;;
  deepseek)
    MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
    NGRAM_N="${NGRAM_N:-30}"; NGRAM_K="${NGRAM_K:-8}"; NGRAM_WINDOW="${NGRAM_WINDOW:-3000}" ;;
  *) echo "MODEL_KEY must be qwen|deepseek"; exit 1 ;;
esac

N="${N:-10}"
MAX_PROBLEMS="${MAX_PROBLEMS:-0}"
MAX_NEW="${MAX_NEW:-32768}"
MAX_CHOPS="${MAX_CHOPS:-3}"
TAG="${TAG:-$( [ "$MAX_PROBLEMS" = 0 ] && echo full || echo smoke${MAX_PROBLEMS} )}"
OUT="${OUT:-results/_live_adaptive/${MODEL_KEY}_${TAG}_n${N}.json}"

mkdir -p "$(dirname "$OUT")"
echo "live-adaptive | $MODEL | n=$N max_problems=$MAX_PROBLEMS | detect ${NGRAM_N}/${NGRAM_K}/${NGRAM_WINDOW} | out=$OUT"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONPATH=. \
MODEL="$MODEL" N="$N" MAX_PROBLEMS="$MAX_PROBLEMS" MAX_NEW="$MAX_NEW" MAX_CHOPS="$MAX_CHOPS" \
NGRAM_N="$NGRAM_N" NGRAM_K="$NGRAM_K" NGRAM_WINDOW="$NGRAM_WINDOW" OUT="$OUT" \
  uv run python scripts/live_adaptive_decode.py 2>&1 | tee "${OUT%.json}.log"
