#!/usr/bin/env bash
# HumanEval regeneration (2026-11) — plan: docs/experiments/humaneval_regen_2026_11_plan.md
#
# Runs bench/humaneval/regen.py on the GPU box with the envs/vllm-033 environment:
# per model, the vLLM configs in one process, then HF beam in a second process.
#
#   ./run_regen_humaneval_8models.sh smoke          # 5 tasks, models 1-3, writes results/humaneval_regen_2026_11_smoke
#   ./run_regen_humaneval_8models.sh smoke-rerun    # same, second copy for the reproducibility check
#                                                   (SMOKE_NAME=smoke2 ... writes ..._smoke2 / ..._smoke2_rerun)
#   ./run_regen_humaneval_8models.sh fit MODEL_ID   # 7B fit test: engine start + 5 tasks (vLLM greedy, HF beam)
#   ./run_regen_humaneval_8models.sh small          # full run, models 1-3 (resumable)
#   ./run_regen_humaneval_8models.sh large [IDS..]  # full run, 7B models (resumable); default all five
#
# Full runs need the user's approved time budget first (standing rule). Nothing here deletes files:
# the runner refuses to touch an existing output unless resuming it.
set -euo pipefail

cd "$(dirname "$0")"
PY="${PY:-.venv-vllm033/bin/python}"
export VLLM_USE_V2_MODEL_RUNNER=0          # our p-less processor uses the V1 logits-processor interface
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0    # deterministic scheduling (vLLM reproducibility docs)
export HF_HUB_ENABLE_HF_TRANSFER=1
# The gated Llama-2 repos need a Hugging Face login on this machine (`hf auth login`, done by the user).
# This script never reads token files; it only asks huggingface_hub whether a login exists.
require_hf_login() {
  "$PY" -c "import huggingface_hub as h; h.whoami()" >/dev/null 2>&1 \
    || { echo "No Hugging Face login on this machine; run: $PY -m huggingface_hub.commands.huggingface_cli login  (or: hf auth login)"; exit 3; }
}

SMALL=(m-a-p/OpenCodeInterpreter-DS-1.3B Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-Coder-3B)
LARGE=(codellama/CodeLlama-7b-hf codellama/CodeLlama-7b-Instruct-hf meta-llama/Llama-2-7b-hf
       meta-llama/Llama-2-7b-chat-hf Qwen/Qwen2.5-Coder-7B-Instruct)
mkdir -p logs

vllm_configs() {   # all vLLM configs in the plan for a model (single source of truth: bench/humaneval/regen.py)
  "$PY" -c "import sys; from bench.humaneval.regen import CONFIGS, configs_for; \
print(','.join(c for c in configs_for(sys.argv[1]) if CONFIGS[c].backend == 'vllm'))" "$1"
}

run() {  # run MODEL CONFIGS OUT_ROOT [extra args...]
  local model="$1" configs="$2" root="$3"; shift 3
  local log="logs/regen_humaneval_$(basename "$root")_${model//\//--}_${configs//,/+}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(TZ=Asia/Kolkata date '+%F %T IST')] $model | $configs -> $root (log $log)"
  "$PY" -m bench.humaneval.regen --model "$model" --configs "$configs" --out-root "$root" "$@" 2>&1 | tee "$log"
}

case "${1:-}" in
  smoke|smoke-rerun)
    root="results/humaneval_regen_2026_11_${SMOKE_NAME:-smoke}"; [[ "$1" == smoke-rerun ]] && root="${root}_rerun"
    for m in "${SMALL[@]}"; do
      run "$m" "greedy,temp_t0.7,top_p0.9_t1.0,pless_t0.7,pless_norm_t0.7" "$root" --max-problems 5
      run "$m" "beam4" "$root" --max-problems 5
    done ;;
  fit)
    m="${2:?model id}"
    [[ "$m" == meta-llama/* ]] && require_hf_login
    run "$m" "greedy" "results/humaneval_regen_2026_11_fit" --max-problems 5
    run "$m" "beam4" "results/humaneval_regen_2026_11_fit" --max-problems 5 ;;
  small)
    for m in "${SMALL[@]}"; do
      run "$m" "$(vllm_configs "$m")" "results/humaneval_regen_2026_11" --resume
      run "$m" "beam4" "results/humaneval_regen_2026_11" --resume
    done ;;
  large)
    shift; models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=("${LARGE[@]}")
    for m in "${models[@]}"; do [[ "$m" == meta-llama/* ]] && require_hf_login; done
    for m in "${models[@]}"; do
      run "$m" "$(vllm_configs "$m")" "results/humaneval_regen_2026_11" --resume
      run "$m" "beam4" "results/humaneval_regen_2026_11" --resume
    done ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
