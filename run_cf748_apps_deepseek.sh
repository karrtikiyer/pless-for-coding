#!/bin/bash
# Paper B cross-source replication: DeepSeek-R1-Distill-Llama-8B on APPS CODEFORCES-interview,
# 748-problem subset.
#
# MIRRORS run_cf748_apps_qwen3.sh, retargeted to DeepSeek. Runs the full Paper-B Table-1
# config set + top-p sweep (22 arms) on the seeded-random 748-of-2386 CodeForces-interview
# subset (scripts/select_cf_subset.py -> data/cf_interview_748_ids.txt, seed 20260902).
# BYTE-IDENTICAL generation settings to the 252 AtCoder runs; only source, subset, and results
# dir differ. DeepSeek's #45488 safe-tokenizer fix is auto-applied on the vLLM path.
#
# Arms (default 16): pless(k=2) | pless_norm | G_k x6 {1.6,0.8,0.4,0.2,0.1,0.05} (Paper B Table 1;
#   densify with the in-between k's later only if a hump appears) | rec (T0.6/p0.95, NO top-k)
#   | top-p x6 {0.7,0.75,0.8,0.85,0.9,0.95} @T1.0 | pure-temp T1.0.
#
# RUN ON A CUDA POD. Requires .venv-vllm (pyproject-vllm.toml).
#
# Usage:
#   MAX_PROBLEMS=5 N_SAMPLES=3 ARMS="pless renyi_k0.4" GPUS=0,1 ./run_cf748_apps_deepseek.sh   # smoke
#   GPUS=0,1 ./run_cf748_apps_deepseek.sh                                                        # full

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="${SOURCE:-CODEFORCES}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/_cf_interview_748}"
TASK_IDS_FILE="${TASK_IDS_FILE:-data/cf_interview_748_ids.txt}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
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
[ -f "$TASK_IDS_FILE" ] || { echo "Error: TASK_IDS_FILE $TASK_IDS_FILE missing (run scripts/select_cf_subset.py)." >&2; exit 5; }

# DeepSeek recommended config is T0.6 / top-p 0.95 (NO top-k), per the model card.
arm_args() {
  case "$1" in
    pless)      echo "--method pless --temperature 1.0" ;;
    pless_norm) echo "--method pless_norm --temperature 1.0" ;;
    renyi_k*)   echo "--method pless_renyi --renyi-k ${1#renyi_k} --temperature 1.0" ;;
    rec)        echo "--method temp --temperature 0.6 --top-p 0.95" ;;
    topp*)      echo "--method temp --top-p ${1#topp} --top-k 0 --temperature 1.0" ;;
    temp)       echo "--method temp --temperature 1.0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}

# Default = Paper B Table-1 6-point G_k grid (staged: run these first, then densify with the
# in-between k's only if the coarse curve shows a hump). Add later via, e.g.:
#   ARMS="renyi_k0.45 renyi_k0.35 renyi_k0.3 renyi_k0.25 renyi_k0.15 renyi_k0.075" ./run_cf748_apps_deepseek.sh
ARMS_DEFAULT="pless pless_norm \
renyi_k1.6 renyi_k0.8 renyi_k0.4 renyi_k0.2 renyi_k0.1 renyi_k0.05 \
rec topp0.7 topp0.75 topp0.8 topp0.85 topp0.9 topp0.95 temp"
read -ra ARMS_ARR <<< "${ARMS:-$ARMS_DEFAULT}"

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm  ($(arm_args "$arm"))  cap=$MAX_TOKENS  n=$N_SAMPLES  subset=$TASK_IDS_FILE  max_problems=${MAX_PROBLEMS:-full748}"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --task-ids-file "$TASK_IDS_FILE" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    ${MAX_PROBLEMS:+--max-problems "$MAX_PROBLEMS"} \
    --results-dir "$RESULTS_DIR" \
    $(arm_args "$arm")
}

check_vllm

echo "=================================================================="
echo " DeepSeek CF-interview 748  model=$MODEL  cap=$MAX_TOKENS  arms=${#ARMS_ARR[@]}"
echo "   subset=$TASK_IDS_FILE  ${MAX_PROBLEMS:+PILOT: first $MAX_PROBLEMS of subset, n=$N_SAMPLES}"
echo "=================================================================="

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPULIST <<< "$GPUS"; ngpu=${#GPULIST[@]}
  echo "Parallel: ${#ARMS_ARR[@]} arms across $ngpu GPU(s) [$GPUS]"
  pids=()
  for ((g=0; g<ngpu; g++)); do
    group=(); for ((i=0; i<${#ARMS_ARR[@]}; i++)); do (( i % ngpu == g )) && group+=("${ARMS_ARR[$i]}"); done
    [ ${#group[@]} -eq 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES="${GPULIST[$g]}"
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/cf748_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/cf748_gpu*.log" >&2; exit 4; }
else
  for a in "${ARMS_ARR[@]}"; do gen_arm "$a"; done
fi

echo "---- eval (re-execute against APPS/CodeForces tests) ----"
for f in "$OUT_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  case "$f" in *.entropy.*) continue;; esac
  echo "  eval $(basename "$f")"
  uv run python -m bench.eval --results-file "$f" --dataset apps --workers "$WORKERS" --skip-diversity
done

echo "---- cot-efficiency report ----"
uv run python -m bench.eval.cot_efficiency \
  --results-dir "$OUT_DIR" --dataset apps --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"

echo
echo "Done. Build the CF table: SET=deepseek_cf PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py"
