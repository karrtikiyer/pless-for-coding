#!/bin/bash
# Pless loop-recovery sweep on the 27 solvable pless-truncated ATCODER-interview
# tasks (Qwen3-8B, thinking ON, vLLM backend).
#
# HYPOTHESIS: pless at temp=1.0 rambles/loops in the think phase on hard tasks and
# truncates (never closes </think> → no code → auto-fail). Do sampler variants that
# inject diversity BEFORE/INSIDE the pless threshold prevent the loop and recover
# pass@k on these specific tasks?
#
#   - temp-before-pless (T=1.5, 2.0): flattens the peaked loop distribution.
#     Mechanistically the most direct fix (the loop is a LOW-entropy attractor).
#   - alpha arms (a=3,4,5): tau_a = sum(p_i^a) threshold. Per this project, alpha
#     loosens at HIGH-entropy positions — the OPPOSITE of where the loop lives, so
#     PREDICTION: alpha helps LESS than temperature for this failure. Test it.
#   - pless T=1.0: the baseline that exhibits the failure.
#
# 6 arms x 27 tasks x 10 samples = 1,620 generations, 32768-token cap (matches the
# original CoT-efficiency run so truncation is comparable).
#
# RUN ON A CUDA POD (not macOS). Requires the vLLM venv (see pyproject-vllm.toml):
#   uv venv .venv-vllm --python 3.12
#   UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --project pyproject-vllm.toml
#
# Usage:
#   ./run_pless_recovery_sweep_apps_qwen3.sh                 # all 6 arms, then eval+report
#   ONLY=pless_t1.5 ./run_pless_recovery_sweep_apps_qwen3.sh # one arm
#   GPUS=0,1,2 ./run_pless_recovery_sweep_apps_qwen3.sh      # spread arms across GPUs
#
# Env overrides: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES, MAX_TOKENS,
#   VLLM_VENV, WORKERS (eval parallelism), TOKENIZER, GPUS, ONLY.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_recovery_sweep}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL//\//--}"

# ── vLLM env hygiene (matches all working vLLM drivers in this repo) ──────────
# REQUIRED: disable FlashInfer's top-k/top-p sampler. It JIT-compiles a CUDA
# kernel via `ninja` on first use; pods often lack ninja in PATH → engine
# startup crashes with FileNotFoundError. =0 forces the PyTorch-native (Triton
# for bs>=8) sampler — negligible perf hit for an 8B model. (commit c347527)
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Headless matplotlib in eval subprocesses (models sometimes emit
# `import matplotlib.pyplot`; Agg avoids a GUI popup / crash).
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Do NOT set VLLM_WORKER_MULTIPROC_METHOD=spawn: bench/generator_vllm.py attaches
# a PlessSplitLogitsProcessor defined inside a factory function (kept Mac-
# importable), which spawn-mode workers cannot pickle. Leave unset → vLLM uses
# Linux `fork` (child inherits parent memory, no pickling). Tensor-parallel is
# NOT wired (repo todo E1); each arm uses one GPU, pinned via CUDA_VISIBLE_DEVICES.

# The 27 solvable pless-truncated ATCODER-interview tasks (pless truncated 40,
# minus the 13 that NO config solves). Recovery is only meaningful where a
# solution exists. Edit TASK_IDS to change scope.
TASK_IDS="417 558 616 711 739 793 927 930 990 1037 1085 1086 1087 1090 1125 1126 1171 1178 1224 1226 1277 1328 1329 1369 1373 1374 1426"

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"

check_vllm() {
  [ -d "$VLLM_VENV" ] || { echo "Error: vLLM venv $VLLM_VENV missing. Bootstrap:" >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2; exit 2; }
  "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null || {
    echo "Error: 'import vllm' failed in $VLLM_VENV." >&2; exit 3; }
}

# arm key -> sampler args. temp-before-pless via --temperature; alpha via pless_alpha.
arm_args() {
  case "$1" in
    pless_t1.0) echo "--method pless       --temperature 1.0" ;;
    pless_t1.5) echo "--method pless       --temperature 1.5" ;;
    pless_t2.0) echo "--method pless       --temperature 2.0" ;;
    pless_a3)   echo "--method pless_alpha --alpha 3 --temperature 1.0" ;;
    pless_a4)   echo "--method pless_alpha --alpha 4 --temperature 1.0" ;;
    pless_a5)   echo "--method pless_alpha --alpha 5 --temperature 1.0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}

ARMS=(pless_t1.0 pless_t1.5 pless_t2.0 pless_a3 pless_a4 pless_a5)

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm  ($(arm_args "$arm"))"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    --task-ids $TASK_IDS \
    --results-dir "$RESULTS_DIR" \
    $(arm_args "$arm")
}

want() { [ -z "${ONLY:-}" ] || [ "$1" = "$ONLY" ]; }

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
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/sweep_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/sweep_gpu*.log" >&2; exit 4; }
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

echo "---- CoT-efficiency report (pass@k + truncation% + think-tokens per arm) ----"
uv run python -m bench.eval.cot_efficiency \
  --results-dir "$OUT_DIR" --dataset apps --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"

echo
echo "Done. Per-arm metrics: $OUT_DIR/metrics/"
echo "Report: $OUT_DIR/analysis/cot_efficiency_apps_report.md"
echo
echo "Compare vs the original pless-truncation baseline: did temp-before-pless"
echo "(T1.5/T2.0) reduce truncation% and raise pass@k? Did the alpha arms? The"
echo "prediction is temperature helps more than alpha for the loop failure."
