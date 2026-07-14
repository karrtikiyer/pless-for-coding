#!/bin/bash
# CORRECTED DeepSeek-R1-Distill full-252 baseline: pless α=2 and α=5, ATCODER-interview,
# POST the vLLM prompt-tokenizer fix (bench/generator_vllm.py:encode_prompt_for_vllm).
#
# WHY: the committed DeepSeek α=2 (0.174) and α=5 (0.295) numbers were generated on vLLM
# with WHITESPACE-MANGLED prompts (transformers-v5 #45488) — the model saw `deff(a,b):`.
# The fix (validated: fixed-vLLM ≡ HF, α=2 pass@1 gap CI includes 0, N=28) is in the shared
# bench.apps path, so ALL vLLM methods (pless, pless_alpha) now feed correct prompts. This
# regenerates BOTH arms on the full 252 so the α=2 / α=5 / adaptive comparison is honest.
#
# APPLE-TO-APPLE with the HF adaptive live run (results/_live_adaptive/deepseek_full_n10.jsonl):
#   fixed-vLLM ≈ HF (validated), and the adaptive run's own α=2 baseline (non-fired samples,
#   HF) is the cross-check — fixed-vLLM α=2 here should reproduce it on 252.
#   Matched config: --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 32768
#   --enable-thinking; env VLLM_USE_FLASHINFER_SAMPLER=0 (baseline parity).
#
# FRESH output dir (results/_deepseek_fixed_full252/) — do NOT point at an old dir with
# pre-fix rows, or resume would silently mix mangled + fixed samples.
#
# RUN ON A CUDA POD. Requires .venv-vllm.
# Usage:
#   GPUS=0 ./run_pless_fixed_full252_apps_deepseek.sh                 # both arms, sequential
#   ARMS="pless_a5" GPUS=1 ./run_pless_fixed_full252_apps_deepseek.sh # one arm (e.g. on a 2nd GPU)
# Env: ARMS(default "pless pless_a5"), MODEL, GPUS, VLLM_VENV, N_SAMPLES(10), MAX_TOKENS(32768),
#      RESULTS_DIR.

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="ATCODER"
DIFFICULTY="interview"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
PYTHON="$VLLM_VENV/bin/python"
MODEL_DIR="${MODEL//\//--}"
RESULTS_DIR="${RESULTS_DIR:-results/_deepseek_fixed_full252}"
read -ra ARMS_ARR <<< "${ARMS:-pless pless_a5}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

[ -x "$PYTHON" ] || { echo "Error: $PYTHON missing (set VLLM_VENV)." >&2; exit 2; }
"$PYTHON" -c "import vllm" 2>/dev/null || { echo "Error: 'import vllm' failed in $VLLM_VENV." >&2; exit 3; }

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"

# arm -> (bench.apps method args, JSONL basename the runner writes)
arm_method() {
  case "$1" in
    pless)    echo "--method pless" ;;                       # α=2 (default pless)
    pless_a5) echo "--method pless_alpha --alpha 5" ;;       # α=5 prevention
    pless_a3) echo "--method pless_alpha --alpha 3" ;;
    pless_a4) echo "--method pless_alpha --alpha 4" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}
arm_jsonl() {
  case "$1" in
    pless)    echo "pless_think_t1.0_t1.0.jsonl" ;;
    pless_a5) echo "pless_alpha_think_t1.0_a5.0_t1.0.jsonl" ;;
    pless_a3) echo "pless_alpha_think_t1.0_a3.0_t1.0.jsonl" ;;
    pless_a4) echo "pless_alpha_think_t1.0_a4.0_t1.0.jsonl" ;;
  esac
}

echo "=================================================================="
echo " CORRECTED DeepSeek full-252 — arms: ${ARMS_ARR[*]}  (fix ON)"
echo "   cap=$MAX_TOKENS n=$N_SAMPLES gpu=$CUDA_VISIBLE_DEVICES flashinfer=$VLLM_USE_FLASHINFER_SAMPLER"
echo "   out=$OUT_DIR   (watch for '[vllm] Installed safe PreTrainedTokenizerFast')"
echo "=================================================================="

for arm in "${ARMS_ARR[@]}"; do
  JSONL="$OUT_DIR/$(arm_jsonl "$arm")"
  echo ">>> arm=$arm  ($(arm_method "$arm"))  [FULL 252, no --task-ids]"
  "$PYTHON" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    --temperature 1.0 --top-p 1.0 --top-k 0 \
    $(arm_method "$arm") \
    --results-dir "$RESULTS_DIR"
  echo ">>> scoring arm=$arm via bench.eval"
  "$PYTHON" -m bench.eval --results-file "$JSONL" --dataset apps
  echo ">>> arm=$arm done -> $JSONL"
done

echo ">>> ALL arms done. Metrics under $OUT_DIR/metrics/. Next: assemble the corrected"
echo "    α=2 / α=5 table and compare against adaptive (results/_live_adaptive/)."
