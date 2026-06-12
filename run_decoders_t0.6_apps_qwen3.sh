#!/bin/bash
# Four decoders at temperature 0.6 on APPS (Qwen3-8B, ATCODER-interview, vLLM, thinking on).
#
# Compares pless / pless_norm / top-p / top-k all at T=0.6 on the same problem set:
#   pless      @ T0.6   (--method pless)
#   pless_norm @ T0.6   (--method pless_norm)
#   top_p      @ T0.6   (--method temp --top-p 0.95)
#   top_k      @ T0.6   (--method temp --top-k 20)
#
# NOTE on pless @ T0.6: T<1 SHARPENS the distribution before the pless Σpᵢ² threshold
# (Σpᵢ² rises → fewer tokens survive → near-greedy). This is the OPPOSITE direction
# from the recovery sweep's T1.5/2.0 (which flattened to escape the loop). Expect
# pless/pless_norm @ T0.6 to be low-diversity and possibly truncate MORE on hard tasks.
#
# RUN ON A CUDA POD. Requires .venv-vllm (pyproject-vllm.toml).
# Usage:
#   GPUS=0,1 ./run_decoders_t0.6_apps_qwen3.sh
#   ONLY=pless ./run_decoders_t0.6_apps_qwen3.sh
# Env: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES(10), MAX_TOKENS(32768),
#   TEMPERATURE(0.6), TOP_P(0.95), TOP_K(20), VLLM_VENV, WORKERS(32), TOKENIZER,
#   GPUS, ONLY, MAX_PROBLEMS (cap problem count; unset = full 252).

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/decoders_t0.6}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL//\//--}"

# vLLM env hygiene (see other vLLM drivers). VLLM_USE_FLASHINFER_SAMPLER=0 is REQUIRED:
# top_p/top_k use FlashInfer's sampler, which JIT-compiles via `ninja` (often absent
# on pods → FileNotFoundError at engine start). =0 forces the PyTorch-native path.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Do NOT set VLLM_WORKER_MULTIPROC_METHOD=spawn (factory-local processor class is
# unpicklable; vLLM uses Linux fork by default).

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
    pless)      echo "--method pless      --temperature $TEMPERATURE" ;;
    pless_norm) echo "--method pless_norm --temperature $TEMPERATURE" ;;
    top_p)      echo "--method temp       --temperature $TEMPERATURE --top-p $TOP_P" ;;
    top_k)      echo "--method temp       --temperature $TEMPERATURE --top-k $TOP_K" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}

# Slow (token-by-token) pless/pless_norm first so GPU round-robin spreads them.
ARMS=(pless pless_norm top_p top_k)

want() { [ -z "${ONLY:-}" ] || [ "$1" = "$ONLY" ]; }

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm  ($(arm_args "$arm"))"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    ${MAX_PROBLEMS:+--max-problems $MAX_PROBLEMS} \
    --results-dir "$RESULTS_DIR" \
    $(arm_args "$arm")
}

check_vllm
selected=()
for a in "${ARMS[@]}"; do want "$a" && selected+=("$a"); done

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPULIST <<< "$GPUS"; ngpu=${#GPULIST[@]}
  echo "Parallel: ${#selected[@]} arms across $ngpu GPU(s) [$GPUS]"
  pids=()
  for ((g=0; g<ngpu; g++)); do
    group=(); for ((i=0; i<${#selected[@]}; i++)); do (( i % ngpu == g )) && group+=("${selected[$i]}"); done
    [ ${#group[@]} -eq 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES="${GPULIST[$g]}"
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/decoders_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/decoders_gpu*.log" >&2; exit 4; }
else
  for a in "${selected[@]}"; do gen_arm "$a"; done
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
echo "Done. Report: $OUT_DIR/analysis/cot_efficiency_apps_report.md"
echo "4-way @ T${TEMPERATURE}: compare pless / pless_norm / top_p${TOP_P} / top_k${TOP_K}"
echo "on pass@1, pass@10, and trunc% (does sharpening at T0.6 raise pless truncation?)."
