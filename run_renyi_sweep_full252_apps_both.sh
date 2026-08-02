#!/bin/bash
# Launch the DeepSeek + Qwen3 Rényi G_k sweeps IN PARALLEL, one model per GPU.
#
# Thin orchestrator over run_renyi_sweep_full252_apps_{deepseek,qwen3}.sh. It only
# assigns each model a GPU and waits; every other knob (venv, caps, arms, pilot size,
# vLLM env) is inherited by the child scripts from the environment, so pass them the
# usual way and they flow through to BOTH runs.
#
# GPU assignment: DS_GPU (default 0) → DeepSeek, QW_GPU (default 1) → Qwen3.
#
# Env passed through to both children (set via `VAR=val ./...` prefix or `export`):
#   VLLM_VENV           vLLM virtualenv path            (child default .venv-vllm)
#   VLLM_USE_FLASHINFER_SAMPLER, VLLM_* , HF_HOME, HF_TOKEN, MPLBACKEND, ...  (any exported var inherits)
#   MAX_TOKENS(32768) N_SAMPLES(10) MAX_PROBLEMS(unset=full) ARMS RESULTS_DIR WORKERS
#
# Examples:
#   # pilot, custom venv, GPUs 0 & 1:
#   VLLM_VENV=/workspace/vllm_env/.venv MAX_PROBLEMS=25 N_SAMPLES=3 \
#     ARMS="renyi_k1.6 renyi_k0.2" ./run_renyi_sweep_full252_apps_both.sh
#   # full sweep on GPUs 2 & 3, extra vLLM env:
#   VLLM_VENV=/workspace/vllm_env/.venv DS_GPU=2 QW_GPU=3 \
#     VLLM_ATTENTION_BACKEND=FLASHINFER ./run_renyi_sweep_full252_apps_both.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DS_GPU="${DS_GPU:-0}"
QW_GPU="${QW_GPU:-1}"
LOGDIR="${LOGDIR:-$HERE}"
DS_SH="$HERE/run_renyi_sweep_full252_apps_deepseek.sh"
QW_SH="$HERE/run_renyi_sweep_full252_apps_qwen3.sh"

[ -x "$DS_SH" ] || { echo "Error: $DS_SH not executable" >&2; exit 2; }
[ -x "$QW_SH" ] || { echo "Error: $QW_SH not executable" >&2; exit 2; }
[ "$DS_GPU" = "$QW_GPU" ] && { echo "Error: DS_GPU and QW_GPU must differ (got $DS_GPU/$QW_GPU)" >&2; exit 2; }

# Force-export the pass-through knobs so the children definitely inherit them even if a
# caller set them without exporting. (Unset vars are a harmless no-op here.) Arbitrary
# VLLM_* / HF_* the caller exported are already in the environment and inherit automatically.
export VLLM_VENV MAX_TOKENS N_SAMPLES MAX_PROBLEMS ARMS RESULTS_DIR WORKERS \
       VLLM_USE_FLASHINFER_SAMPLER MPLBACKEND HF_HOME HF_TOKEN 2>/dev/null || true

echo "=================================================================="
echo " Rényi sweep — BOTH models in parallel"
echo "   DeepSeek → GPU $DS_GPU   |   Qwen3 → GPU $QW_GPU"
echo "   venv=${VLLM_VENV:-.venv-vllm}  cap=${MAX_TOKENS:-32768}  n=${N_SAMPLES:-10}"
echo "   arms=${ARMS:-<default 6>}  ${MAX_PROBLEMS:+PILOT max_problems=$MAX_PROBLEMS}"
echo "   logs: $LOGDIR/renyi_{deepseek,qwen3}.log"
echo "=================================================================="

GPUS="$DS_GPU" "$DS_SH" > "$LOGDIR/renyi_deepseek.log" 2>&1 &
ds_pid=$!
GPUS="$QW_GPU" "$QW_SH" > "$LOGDIR/renyi_qwen3.log"   2>&1 &
qw_pid=$!
echo "launched: deepseek pid=$ds_pid, qwen3 pid=$qw_pid  (tail -f the logs to watch)"

fail=0
wait "$ds_pid" || { echo "!! DeepSeek run FAILED — see $LOGDIR/renyi_deepseek.log" >&2; fail=1; }
wait "$qw_pid" || { echo "!! Qwen3 run FAILED — see $LOGDIR/renyi_qwen3.log" >&2; fail=1; }
[ $fail -eq 0 ] && echo "Both runs completed." || { echo "One or more runs failed." >&2; exit 1; }
