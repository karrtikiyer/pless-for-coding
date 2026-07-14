#!/bin/bash
# Score (bench.eval) the corrected DeepSeek jsonls produced by the generation-only
# run_crossmethod_fixed_apps_deepseek.sh (and the fixed α=2/α=5 runs). GPU-FREE — pure
# CPU code-execution — so run it on any box (pod or the Mac) once generations finish.
#
# Scans the whole OUT_DIR and scores every *.jsonl, so it also covers the fixed α=2/α=5
# files if they share the tree. IDEMPOTENT: skips an arm whose metrics/ already exists
# (set FORCE=1 to re-score), so you can invoke it repeatedly as arms complete.
#
# Usage:
#   VLLM_VENV=/workspace/vllm_env/.venv ./run_crossmethod_eval_apps_deepseek.sh      # on the pod
#   PYTHON="uv run python" ./run_crossmethod_eval_apps_deepseek.sh                    # on the Mac
#   FORCE=1 ./run_crossmethod_eval_apps_deepseek.sh                                   # re-score all
# Env: MODEL, RESULTS_DIR, PYTHON | VLLM_VENV, WORKERS(12), FORCE, HF_HUB_OFFLINE(0).

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="ATCODER"
DIFFICULTY="interview"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
PYTHON="${PYTHON:-$VLLM_VENV/bin/python}"      # override with PYTHON="uv run python" off-pod
MODEL_DIR="${MODEL//\//--}"
RESULTS_DIR="${RESULTS_DIR:-results/_deepseek_fixed_full252}"
OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
WORKERS="${WORKERS:-12}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"     # bench.eval needs the APPS test data
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

[ -d "$OUT_DIR" ] || { echo "Error: OUT_DIR not found: $OUT_DIR" >&2; exit 1; }

shopt -s nullglob
jsonls=("$OUT_DIR"/*.jsonl)
[ ${#jsonls[@]} -gt 0 ] || { echo "No *.jsonl in $OUT_DIR yet." >&2; exit 1; }

echo "=================================================================="
echo " Scoring corrected DeepSeek runs in $OUT_DIR  (workers=$WORKERS, force=${FORCE:-0})"
echo "=================================================================="

for f in "${jsonls[@]}"; do
  stem="$(basename "$f" .jsonl)"
  m="$OUT_DIR/metrics/${stem}_metrics.json"
  if [ -f "$m" ] && [ -z "${FORCE:-}" ]; then
    echo ">>> skip (already scored): $stem"
    continue
  fi
  echo ">>> scoring: $stem"
  "$PYTHON" -m bench.eval --results-file "$f" --dataset apps --workers "$WORKERS"
done

echo ">>> Done. Metrics under $OUT_DIR/metrics/."
