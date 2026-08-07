#!/bin/bash
# top-p (nucleus) sweep for DeepSeek-R1-Distill-Llama-8B (ATCODER-interview, vLLM).
#
# Comparison baseline for the Rényi G_k sweep: can plain nucleus sampling at temperature 1.0 match
# what lowering the p-less/G_k order achieves? Arms are `--method temp --top-p <p> --top-k 0
# --temperature 1.0` for p in {0.8, 0.85, 0.9, 1.0} (p=1.0 = pure temperature, no truncation).
#
# APPLES-TO-APPLES with the G_k sweep + the α=2 baseline: SAME backend (vLLM), cap (MAX_TOKENS=32768),
# n=10, --enable-thinking, VLLM_USE_FLASHINFER_SAMPLER=0. Results land in results/_top_p_sweep_full252
# (fresh dir; fold into scripts/build_decoder_comparison_table.py alongside G_k / τ_α).
#
# RUN ON A CUDA POD. Requires the vLLM venv (pyproject-vllm.toml).
#
# Usage:
#   # smoke: MAX_PROBLEMS=5 N_SAMPLES=2 ARMS="topp0.9" GPUS=0 ./run_top_p_sweep_full252_apps_deepseek.sh
#   # full : GPUS=0,1 ./run_top_p_sweep_full252_apps_deepseek.sh
# Env: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES(10), MAX_TOKENS(32768),
#   MAX_PROBLEMS(unset=full 252), VLLM_VENV, WORKERS(32), TOKENIZER, GPUS, ARMS.

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/_top_p_sweep_full252}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"        # matches the G_k sweep + α=2 baseline cap
MAX_PROBLEMS="${MAX_PROBLEMS:-}"          # set (e.g. 25) for a pilot; unset = full 252
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
MODEL_DIR="${MODEL//\//--}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"

check_vllm() {
  [ -x "$VLLM_VENV/bin/python" ] || { echo "Error: vLLM venv $VLLM_VENV missing." >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2; exit 2; }
  "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null || { echo "Error: import vllm failed." >&2; exit 3; }
}

# Any arm "topp<P>" runs temperature sampling at T=1.0 with nucleus cutoff <P> (top-k off).
# P=1.0 is pure temperature (no truncation). Matches the G_k arms' footing except the sampler.
arm_args() {
  case "$1" in
    topp*) echo "--method temp --top-p ${1#topp} --top-k 0 --temperature 1.0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}
# arm -> JSONL basename the runner writes (reference only; the eval loop globs *.jsonl).
# P<1.0 -> temp_p<P>_think_t1.0_t1.0.jsonl ; P=1.0 -> temp_think_t1.0_t1.0.jsonl (no _p suffix).
arm_jsonl() {
  case "$1" in
    topp1.0|topp1) echo "temp_think_t1.0_t1.0.jsonl" ;;
    topp*)         echo "temp_p${1#topp}_think_t1.0_t1.0.jsonl" ;;
  esac
}

ARMS_DEFAULT="topp0.8 topp0.85 topp0.9 topp1.0"
read -ra ARMS_ARR <<< "${ARMS:-$ARMS_DEFAULT}"

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm  ($(arm_args "$arm"))  cap=$MAX_TOKENS  n=$N_SAMPLES  max_problems=${MAX_PROBLEMS:-full}"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    ${MAX_PROBLEMS:+--max-problems "$MAX_PROBLEMS"} \
    --results-dir "$RESULTS_DIR" \
    $(arm_args "$arm")
}

check_vllm

echo "=================================================================="
echo " DeepSeek top-p sweep  model=$MODEL  cap=$MAX_TOKENS  arms=${ARMS_ARR[*]}"
echo "   ${MAX_PROBLEMS:+PILOT: first $MAX_PROBLEMS problems, n=$N_SAMPLES}"
echo "=================================================================="

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPULIST <<< "$GPUS"; ngpu=${#GPULIST[@]}
  echo "Parallel: ${#ARMS_ARR[@]} arms across $ngpu GPU(s) [$GPUS]"
  pids=()
  for ((g=0; g<ngpu; g++)); do
    group=(); for ((i=0; i<${#ARMS_ARR[@]}; i++)); do (( i % ngpu == g )) && group+=("${ARMS_ARR[$i]}"); done
    [ ${#group[@]} -eq 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES="${GPULIST[$g]}"
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/topp_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/topp_gpu*.log" >&2; exit 4; }
else
  for a in "${ARMS_ARR[@]}"; do gen_arm "$a"; done
fi

echo "---- eval (re-execute against APPS tests) ----"
for f in "$OUT_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  case "$f" in *.entropy.*) continue;; esac
  echo "  eval $(basename "$f")"
  uv run python -m bench.eval --results-file "$f" --dataset apps --workers "$WORKERS" --skip-diversity
done

echo "---- cot-efficiency report (pass@k + trunc% per arm) ----"
uv run python -m bench.eval.cot_efficiency \
  --results-dir "$OUT_DIR" --dataset apps --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"

echo
echo "Done. Compare top-p vs G_k / τ_α via scripts/build_decoder_comparison_table.py."
