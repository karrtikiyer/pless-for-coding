#!/bin/bash
# Verify the vLLM prompt-tokenization fix for DeepSeek-R1-Distill-Llama-8B
# (transformers-v5 #45488: LlamaTokenizer's Metaspace override ate whitespace, so
# vLLM fed the model `deff(a,b):` instead of `def f(a, b):`). The HF backend was
# unaffected; the vLLM backend now pre-encodes with the safe tokenizer
# (bench/generator_vllm.py:encode_prompt_for_vllm) so it feeds byte-identical ids
# to HF. This script re-runs vLLM pless α=2 WITH the fix and compares.
#
# APPLE-TO-APPLE: matches the broken vLLM baseline's config + env EXACTLY, so the
# ONLY difference is the tokenizer fix.
#   baseline: results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/
#             pless_think_t1.0_t1.0.jsonl (pless α=2, 32768, trunc 61.9% / pass@1 0.190 on the 52-task subset)
#   config:   --method pless --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 32768 --enable-thinking
#   env:      VLLM_USE_FLASHINFER_SAMPLER=0, MPLBACKEND=Agg, HF_HOME  (mirrors
#             run_pless_recovery_full252_apps_deepseek.sh:50-52 — the documented
#             apples-to-apples env for this baseline).
#
# MODE=smoke     (default) 2 tasks, n=1, 1024 tok — mechanics only: confirms no
#                crash + the "[vllm] Installed safe PreTrainedTokenizerFast" line
#                prints + output is coherent. Does NOT measure truncation (needs
#                the full 32768 budget). ~a few minutes on 1 GPU.
# MODE=validate  10 tasks, n=10, 32768 tok — the real test: truncation should drop
#                from ~62% toward HF's 35%, pass@1 rise from 0.19 toward 0.41.
#                Runs bench.eval + the paired compare vs broken-vLLM and vs HF.
#                ~1.5 hr on 1 GPU.
#
# RUN ON A CUDA POD (vLLM). Requires .venv-vllm.
# Usage:
#   GPUS=0 ./run_backend_delta_vllm_apps_deepseek.sh                 # smoke
#   MODE=validate GPUS=0 ./run_backend_delta_vllm_apps_deepseek.sh   # full validation
# Env: MODE, MODEL, GPUS, VLLM_VENV, MAX_TOKENS(32768), N_SAMPLES, TASK_IDS.

set -euo pipefail

MODE="${MODE:-smoke}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="ATCODER"
DIFFICULTY="interview"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
PYTHON="$VLLM_VENV/bin/python"
MODEL_DIR="${MODEL//\//--}"
BASELINE_DIR="results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
HF_DIR="results/_backend_delta_deepseek/hf/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"

# --- vLLM env hygiene: IDENTICAL to run_pless_recovery_full252_apps_deepseek.sh:50-52
#     so fixed-vs-broken differs ONLY in the tokenizer fix (not the sampler path). ---
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

[ -x "$PYTHON" ] || { echo "Error: $PYTHON missing. Bootstrap:" >&2
  echo "  uv venv $VLLM_VENV --python 3.12" >&2
  echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2; exit 2; }
"$PYTHON" -c "import vllm" 2>/dev/null || { echo "Error: 'import vllm' failed in $VLLM_VENV." >&2; exit 3; }

if [ "$MODE" = "smoke" ]; then
  N_SAMPLES="${N_SAMPLES:-1}"; MAX_TOKENS="${MAX_TOKENS:-1024}"
  TASK_IDS="${TASK_IDS:-117 370}"
  RESULTS_DIR="results/_backend_delta_deepseek/vllm_fixed_smoke"
elif [ "$MODE" = "validate" ]; then
  N_SAMPLES="${N_SAMPLES:-10}"; MAX_TOKENS="${MAX_TOKENS:-32768}"
  TASK_IDS="${TASK_IDS:-117 370 587 827 962 1038 1123 1177 1274 1370}"
  RESULTS_DIR="results/_backend_delta_deepseek/vllm_fixed"
else
  echo "unknown MODE '$MODE' (smoke|validate)" >&2; exit 2
fi

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
JSONL="$OUT_DIR/pless_think_t1.0_t1.0.jsonl"
mkdir -p "$OUT_DIR"

echo "=================================================================="
echo " vLLM tokenizer-fix $MODE — DeepSeek pless α=2"
echo "   tasks=[$TASK_IDS]  n=$N_SAMPLES  cap=$MAX_TOKENS"
echo "   env: VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER  gpu=$CUDA_VISIBLE_DEVICES"
echo "   watch for: '[vllm] Installed safe PreTrainedTokenizerFast ...' (fix path active)"
echo "=================================================================="

"$PYTHON" -m bench.apps \
  --model "$MODEL" --backend vllm --method pless \
  --source "$SOURCE" --difficulty "$DIFFICULTY" \
  --enable-thinking \
  --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --task-ids $TASK_IDS \
  --results-dir "$RESULTS_DIR"

if [ "$MODE" = "smoke" ]; then
  echo
  echo ">>> smoke check: first sample should be coherent reasoning WITH whitespace/structure"
  "$PYTHON" - "$JSONL" <<'PY'
import json, sys
recs=[json.loads(l) for l in open(sys.argv[1])]
s=recs[0]["samples_with_thinking"][0]
print("task", recs[0]["task_id"], "| gen chars:", len(s))
print("has newlines/indent?", ("\n" in s), "| looks mangled (no spaces)?", (" " not in s[:400]))
print("--- first 400 chars ---")
print(s[:400])
PY
  echo ">>> If the text reads coherently with spaces/newlines, the prompt reached the model intact."
  exit 0
fi

echo ">>> scoring (bench.eval) + paired compare vs broken-vLLM and vs HF"
"$PYTHON" -m bench.eval --results-file "$JSONL" --dataset apps

# Compare fixed-vLLM vs the BROKEN baseline (isolates the fix effect) on the same task_ids.
"$PYTHON" scripts/compare_backend_delta.py \
  --hf-metrics   "$OUT_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
  --hf-jsonl     "$JSONL" \
  --vllm-metrics "$BASELINE_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
  --vllm-jsonl   "$BASELINE_DIR/pless_think_t1.0_t1.0.jsonl" \
  --task-ids $TASK_IDS \
  --out "$OUT_DIR/vs_broken_vllm.md" || echo "(compare vs broken skipped)"

# Compare fixed-vLLM vs HF (should now MATCH — same ids fed to the model).
if [ -f "$HF_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" ]; then
  "$PYTHON" scripts/compare_backend_delta.py \
    --hf-metrics   "$HF_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
    --hf-jsonl     "$HF_DIR/pless_think_t1.0_t1.0.jsonl" \
    --vllm-metrics "$OUT_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
    --vllm-jsonl   "$JSONL" \
    --task-ids $TASK_IDS \
    --out "$OUT_DIR/vs_hf.md" || echo "(compare vs HF skipped)"
fi
echo ">>> Prediction: vs broken-vLLM truncation drops (~62%→~35%), pass@1 up (~0.19→~0.41);"
echo "    vs HF the gap should be ~0 (CI includes 0) — fixed vLLM ≡ HF."
