#!/bin/bash
# CORRECTED DeepSeek-R1-Distill cross-method configs (pless_norm + 4 temp variants) on the
# full 252 ATCODER-interview, POST the vLLM prompt-tokenizer fix (#45488 whitespace-mangling).
#
# WHY: EVERY config in results/pless_cot_efficiency_vllm/deepseek-ai--…/ATCODER_interview was
# generated on vLLM with mangled prompts — not just pless. The mangling is a prompt-ENCODING
# bug (broken engine.get_tokenizer()), upstream of and independent of the sampler, so temp /
# top_p / top_k / pless_norm were all fed `deff(a,b):`. This regenerates the non-pless-α arms
# with the fix (pless α=2 + α=5 are covered separately by run_pless_fixed_full252_apps_deepseek.sh).
#
# Together with that run, this replaces the confounded 2026-06-30 DeepSeek cross-method table.
# Writes into the SAME fresh tree (results/_deepseek_fixed_full252/) so the corrected α=2/α=5 +
# these land in one clean directory. Filenames match the originals (verified via _method_key)
# so downstream tooling is drop-in.
#
# Matched to the originals — only the tokenizer fix differs. Config per arm below; env
# VLLM_USE_FLASHINFER_SAMPLER=0 (baseline parity). Fix applies automatically (encode_prompt_for_vllm).
#
# RUN ON A CUDA POD. Requires .venv-vllm (set VLLM_VENV to your path).
# Usage:
#   VLLM_VENV=/workspace/vllm_env/.venv GPUS=0 ./run_crossmethod_fixed_apps_deepseek.sh
#   ARMS="temp_k20 temp_t0.6" GPUS=1 ./run_crossmethod_fixed_apps_deepseek.sh   # subset
# Env: ARMS, MODEL, GPUS, VLLM_VENV, N_SAMPLES(10), MAX_TOKENS(32768), RESULTS_DIR.

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="ATCODER"
DIFFICULTY="interview"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
PYTHON="$VLLM_VENV/bin/python"
MODEL_DIR="${MODEL//\//--}"
RESULTS_DIR="${RESULTS_DIR:-results/_deepseek_fixed_full252}"   # SAME tree as α=2/α=5
read -ra ARMS_ARR <<< "${ARMS:-pless_norm temp_k20 temp_p0.95_k20_t0.6 temp_p0.95_t1.0 temp_t0.6 temp_rec_t0.6}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

[ -x "$PYTHON" ] || { echo "Error: $PYTHON missing (set VLLM_VENV)." >&2; exit 2; }
"$PYTHON" -c "import vllm" 2>/dev/null || { echo "Error: 'import vllm' failed in $VLLM_VENV." >&2; exit 3; }

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"

# arm -> full bench.apps method+sampling flags (matched to the confounded originals).
arm_flags() {
  case "$1" in
    pless_norm)          echo "--method pless_norm --temperature 1.0 --top-p 1.0 --top-k 0" ;;
    temp_k20)            echo "--method temp --temperature 1.0 --top-p 1.0 --top-k 20" ;;
    temp_p0.95_k20_t0.6) echo "--method temp --temperature 0.6 --top-p 0.95 --top-k 20" ;;
    temp_p0.95_t1.0)     echo "--method temp --temperature 1.0 --top-p 0.95 --top-k 0" ;;
    temp_t0.6)           echo "--method temp --temperature 0.6 --top-p 1.0 --top-k 0" ;;
    # DeepSeek-R1-Distill AUTHOR-RECOMMENDED config (model card + generation_config.json:
    # temp 0.6, top_p 0.95, no top_k). The fair "good decoding" baseline for DeepSeek.
    temp_rec_t0.6)       echo "--method temp --temperature 0.6 --top-p 0.95 --top-k 0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}
# arm -> JSONL basename the runner writes (verified against _method_key + the original files).
arm_jsonl() {
  case "$1" in
    pless_norm)          echo "pless_norm_think_t1.0_t1.0.jsonl" ;;
    temp_k20)            echo "temp_k20_think_t1.0_t1.0.jsonl" ;;
    temp_p0.95_k20_t0.6) echo "temp_p0.95_k20_think_t0.6_t0.6.jsonl" ;;
    temp_p0.95_t1.0)     echo "temp_p0.95_think_t1.0_t1.0.jsonl" ;;
    temp_t0.6)           echo "temp_think_t0.6_t0.6.jsonl" ;;
    temp_rec_t0.6)       echo "temp_p0.95_think_t0.6_t0.6.jsonl" ;;
  esac
}

echo "=================================================================="
echo " CORRECTED DeepSeek cross-method (fix ON) — arms: ${ARMS_ARR[*]}"
echo "   cap=$MAX_TOKENS n=$N_SAMPLES gpu=$CUDA_VISIBLE_DEVICES flashinfer=$VLLM_USE_FLASHINFER_SAMPLER"
echo "   out=$OUT_DIR   (watch for '[vllm] Installed safe PreTrainedTokenizerFast')"
echo "=================================================================="

for arm in "${ARMS_ARR[@]}"; do
  JSONL="$OUT_DIR/$(arm_jsonl "$arm")"
  echo ">>> arm=$arm  ($(arm_flags "$arm"))  [FULL 252]"
  "$PYTHON" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    $(arm_flags "$arm") \
    --results-dir "$RESULTS_DIR"
  echo ">>> scoring arm=$arm via bench.eval"
  "$PYTHON" -m bench.eval --results-file "$JSONL" --dataset apps
  echo ">>> arm=$arm done -> $JSONL"
done

echo ">>> Done. Corrected cross-method configs in $OUT_DIR (+ metrics/)."
echo "    Combine with fixed α=2/α=5 (same dir) for the corrected DeepSeek cross-method table."
