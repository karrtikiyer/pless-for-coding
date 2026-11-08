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
#   ./run_regen_humaneval_8models.sh diag           # greedy diagnostics, models 1-3, 164 tasks, writes
#                                                   results/humaneval_regen_2026_11_diag: HF greedy (greedy_hf.jsonl)
#                                                   and vLLM greedy on prompt.strip() (greedy__prompt_stripped.jsonl)
#   PROMPT_VARIANT=stripped ./run_regen_humaneval_8models.sh extra [IDS..]
#                                                   extra comparison configs (NOT for the paper; user request
#                                                   2026-11-08): p-less / p-less-norm at T=1.0 and T=0.6
#                                                   (regen.EXTRA_CONFIGS), one vLLM process per model, resumable;
#                                                   default all eight models in order; writes
#                                                   results/humaneval_regen_2026_11_stripped_extra
#   ./run_regen_humaneval_8models.sh instruct7b     # analysis only (NOT for the paper; user request 2026-11-08):
#                                                   Qwen/Qwen2.5-Coder-7B-Instruct, our chat protocol, the six common
#                                                   configs (vLLM, then HF beam4), resumable; writes
#                                                   results/humaneval_regen_2026_11_instruct7b (any PROMPT_VARIANT:
#                                                   chat prompts take no variant flag)
#   ./run_regen_humaneval_8models.sh instruct7b-smoke   # same, 5 tasks -> results/humaneval_regen_2026_11_instruct7b_smoke
#
# PROMPT_VARIANT=stripped (smoke / smoke-rerun / fit / small / large; user decision 2026-11-07, run log):
#   prompt.strip() (BigCode task `humaneval`) for the raw-prompted models; chat models keep their chat template
#   (regen.prompt_variant_args decides per model). Writes NEW roots, chat models included, so the final table
#   reads one folder: results/humaneval_regen_2026_11_stripped (small, large), ..._stripped_smoke[_rerun]
#   (SMOKE_NAME=smoke2 -> ..._stripped_smoke2[_rerun]), ..._stripped_fit. Default (unset): unstripped, as before.
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
       meta-llama/Llama-2-7b-chat-hf Qwen/Qwen2.5-Coder-7B)
mkdir -p logs

PROMPT_VARIANT="${PROMPT_VARIANT:-unstripped}"
case "$PROMPT_VARIANT" in
  unstripped) ROOT="results/humaneval_regen_2026_11" ;;
  stripped)   ROOT="results/humaneval_regen_2026_11_stripped" ;;
  *) echo "PROMPT_VARIANT must be unstripped or stripped; got '$PROMPT_VARIANT'"; exit 2 ;;
esac
SMOKE_NAME="${SMOKE_NAME:-smoke}"   # must start with "smoke": e.g. SMOKE_NAME=stripped would hit the stripped full root
[[ "$SMOKE_NAME" == smoke* ]] || { echo "SMOKE_NAME must start with 'smoke'; got '$SMOKE_NAME'"; exit 2; }

vllm_configs() {   # all vLLM configs in the plan for a model (single source of truth: bench/humaneval/regen.py)
  "$PY" -c "import sys; from bench.humaneval.regen import CONFIGS, configs_for; \
print(','.join(c for c in configs_for(sys.argv[1]) if CONFIGS[c].backend == 'vllm'))" "$1"
}

variant_args() {   # runner flags for PROMPT_VARIANT and a model (single source of truth: regen.prompt_variant_args)
  [[ "$PROMPT_VARIANT" == unstripped ]] && return 0
  "$PY" -c "import sys; from bench.humaneval.regen import prompt_variant_args; \
print(' '.join(prompt_variant_args(sys.argv[1], sys.argv[2])))" "$1" "$PROMPT_VARIANT"
}

check_model_root() {   # MODEL OUT_ROOT: refuse the instruct7b model outside its root (regen.check_model_for_root)
  "$PY" -c "import sys; from pathlib import Path; from bench.humaneval.regen import check_model_for_root; \
check_model_for_root(sys.argv[1], Path(sys.argv[2]))" "$1" "$2"
}

I7_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
I7_ROOT="results/humaneval_regen_2026_11_instruct7b"

run() {  # run MODEL CONFIGS OUT_ROOT [extra args...]
  local model="$1" configs="$2" root="$3"; shift 3
  local tag="" pv="unstripped"; [[ " $* " == *" --prompt-variant stripped "* ]] && tag="__prompt_stripped" && pv="stripped"
  local log="logs/regen_humaneval_$(basename "$root")_${model//\//--}_${configs//,/+}${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(TZ=Asia/Kolkata date '+%F %T IST')] $model | $configs | prompt-variant $pv -> $root (log $log)"
  "$PY" -m bench.humaneval.regen --model "$model" --configs "$configs" --out-root "$root" "$@" 2>&1 | tee "$log"
}

case "${1:-}" in
  smoke|smoke-rerun)
    root="${ROOT}_${SMOKE_NAME}"; [[ "$1" == smoke-rerun ]] && root="${root}_rerun"
    for m in "${SMALL[@]}"; do
      va=$(variant_args "$m")   # assigned first so a failure stops the script (set -e); empty by default
      run "$m" "greedy,temp_t0.7,top_p0.9_t1.0,pless_t0.7,pless_norm_t0.7" "$root" --max-problems 5 $va
      run "$m" "beam4" "$root" --max-problems 5 $va
    done ;;
  fit)
    m="${2:?model id}"
    [[ "$m" == meta-llama/* ]] && require_hf_login
    check_model_root "$m" "${ROOT}_fit"
    va=$(variant_args "$m")
    run "$m" "greedy" "${ROOT}_fit" --max-problems 5 $va
    run "$m" "beam4" "${ROOT}_fit" --max-problems 5 $va ;;
  small)
    for m in "${SMALL[@]}"; do
      va=$(variant_args "$m")
      run "$m" "$(vllm_configs "$m")" "$ROOT" --resume $va
      run "$m" "beam4" "$ROOT" --resume $va
    done ;;
  large)
    shift; models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=("${LARGE[@]}")
    for m in "${models[@]}"; do [[ "$m" == meta-llama/* ]] && require_hf_login; check_model_root "$m" "$ROOT"; done
    for m in "${models[@]}"; do
      va=$(variant_args "$m")
      run "$m" "$(vllm_configs "$m")" "$ROOT" --resume $va
      run "$m" "beam4" "$ROOT" --resume $va
    done ;;
  extra)   # companion of the stripped main run: same settings, only the temperature of p-less / p-less-norm differs
    [[ "$PROMPT_VARIANT" == stripped ]] || { echo "extra runs with the stripped protocol only; set PROMPT_VARIANT=stripped"; exit 2; }
    shift; models=("$@"); [[ ${#models[@]} -eq 0 ]] && models=("${SMALL[@]}" "${LARGE[@]}")
    for m in "${models[@]}"; do [[ "$m" == meta-llama/* ]] && require_hf_login; check_model_root "$m" "${ROOT}_extra"; done
    cfgs=$("$PY" -c "from bench.humaneval.regen import EXTRA_CONFIGS; print(','.join(EXTRA_CONFIGS))")
    for m in "${models[@]}"; do
      va=$(variant_args "$m")
      run "$m" "$cfgs" "${ROOT}_extra" --resume $va
    done ;;
  instruct7b|instruct7b-smoke)   # analysis only: Qwen2.5-Coder-7B-Instruct, chat protocol, six common configs
    root="$I7_ROOT"; lim=(--resume)
    [[ "$1" == instruct7b-smoke ]] && root="${I7_ROOT}_smoke" && lim=(--max-problems 5)
    check_model_root "$I7_MODEL" "$root"
    va=$(variant_args "$I7_MODEL")   # empty for a chat model, whatever PROMPT_VARIANT says
    [[ -z "$va" ]] || { echo "unexpected prompt-variant args for $I7_MODEL: $va"; exit 2; }
    run "$I7_MODEL" "$(vllm_configs "$I7_MODEL")" "$root" "${lim[@]}"
    run "$I7_MODEL" "beam4" "$root" "${lim[@]}" ;;
  diag)   # (a) backend effect: HF greedy, unstripped prompt; (b) prompt effect: vLLM greedy, stripped prompt
    [[ "$PROMPT_VARIANT" == unstripped ]] || { echo "diag fixes its own prompt variants; unset PROMPT_VARIANT"; exit 2; }
    for m in "${SMALL[@]}"; do
      run "$m" "greedy_hf" "results/humaneval_regen_2026_11_diag" --resume
      run "$m" "greedy" "results/humaneval_regen_2026_11_diag" --prompt-variant stripped --resume
    done ;;
  *) sed -n '2,37p' "$0"; exit 2 ;;
esac
