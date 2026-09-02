#!/bin/bash
# Rényi G_k sweep for Qwen3-8B (ATCODER-interview, vLLM).
#
# MIRRORS run_renyi_sweep_full252_apps_deepseek.sh exactly, retargeted to Qwen3-8B.
# Swaps the α (power-sum τ_α=Σpᵢ^α) arms for the origin paper's *rooted* Rényi form
# G_k=(Σpᵢ^k)^{1/(k-1)}=exp(-H_k) (`--method pless_renyi --renyi-k <k>`). Purpose:
# empirically compare the two families on the same footing (the author's request);
# G_k loosens as k *decreases* below 2 (k=2 ≡ plain pless = G_2, NOT regenerated).
# See docs/research/paperA_renyi_nonequivalence.md.
#
# k grid: {1.6, 0.8, 0.4, 0.2, 0.1, 0.05}. Qwen3 uses Qwen2Tokenizer (immune to #45488).
#
# APPLES-TO-APPLES with the Qwen α-sweep (results/pless_recovery_full252/Qwen--Qwen3-8B) +
# the α=2 baseline: SAME vLLM backend, cap 32768, n=10, --enable-thinking,
# VLLM_USE_FLASHINFER_SAMPLER=0.
#
# RUN ON A CUDA POD. Requires .venv-vllm (pyproject-vllm.toml).
#
# Usage:
#   MAX_PROBLEMS=25 N_SAMPLES=3 ARMS="renyi_k1.6 renyi_k0.2" GPUS=0,1 ./run_renyi_sweep_full252_apps_qwen3.sh
#   GPUS=0,1 ./run_renyi_sweep_full252_apps_qwen3.sh
# Env: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES(10), MAX_TOKENS(32768),
#   MAX_PROBLEMS(unset=full 252), VLLM_VENV, WORKERS(32), TOKENIZER, GPUS, ARMS.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/_renyi_sweep_full252}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
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

arm_args() {
  case "$1" in
    renyi_k*) echo "--method pless_renyi --renyi-k ${1#renyi_k} --temperature 1.0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}

arm_jsonl() {
  case "$1" in
    renyi_k*) echo "pless_renyi_think_t1.0_k${1#renyi_k}_t1.0.jsonl" ;;
  esac
}

ARMS_DEFAULT="renyi_k1.6 renyi_k0.8 renyi_k0.4 renyi_k0.2 renyi_k0.1 renyi_k0.05"
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
echo " Qwen3-8B Rényi G_k sweep  model=$MODEL  cap=$MAX_TOKENS  arms=${ARMS_ARR[*]}"
echo "   k=2 (=G_2=pless α=2) is the existing baseline, NOT regenerated."
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
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/renyi_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/renyi_gpu*.log" >&2; exit 4; }
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

echo "---- cot-efficiency report (pass@k + trunc% per k arm) ----"
uv run python -m bench.eval.cot_efficiency \
  --results-dir "$OUT_DIR" --dataset apps --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"

echo
echo "Done. Compare G_k vs τ_α: add a 'renyi' SET to scripts/build_decoder_comparison_table.py"
echo "  (τ_α baseline: results/pless_recovery_full252/Qwen--Qwen3-8B/${SOURCE}_${DIFFICULTY})."
