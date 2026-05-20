#!/bin/bash
# Rényi-α p-less sweep on HumanEval-164 for Qwen3-8B with thinking enabled.
#
# Sibling of run_pless_alpha_qwen3_mbpp.sh — same Model × α × T grid,
# different benchmark. Adds Qwen3-8B as the 4th model in the HumanEval
# α-sweep alongside Qwen2.5-Coder, CodeLlama, and m-a-p OCI-1.3B.
#
# Configuration:
#   * Model:       Qwen/Qwen3-8B
#   * α-arms:      {2.0, 2.5, 3.0, 5.0}
#   * Temperature: 1.0
#   * Samples:     10 per task
#   * HE scope:    full 164 tasks
#   * Backend:     HF (default)
#   * Thinking:    --enable-thinking
#
# Usage:
#   ./run_pless_alpha_qwen3_humaneval.sh
#
# Env overrides:
#   ALPHAS          space-separated α values (default "2.0 2.5 3.0 5.0")
#   N_SAMPLES                              (default 10)
#   TEMPERATURE                            (default 1.0)
#   MAX_NEW_TOKENS                         (default 4096 — covers thinking + code on
#                                          HumanEval-164's short prompts. Bump only if
#                                          you see truncated samples; 8192 was overkill
#                                          and slowed HF inference via larger KV cache.)
#   RESULTS_DIR                            (default results/pless_alpha_full_humaneval)
#   BACKEND          hf | vllm             (default hf)
#   GPUS                                    (default: auto-detect)
#   MAX_PROBLEMS     cap for smoke         (default: unset)
#   ONLY_ALPHA                             (default: unset)
#   LOG_DIR                                (default /tmp/alpha_qwen3_humaneval_logs)

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
ALPHAS_DEFAULT="2.0 2.5 3.0 5.0"

ALPHAS="${ALPHAS:-$ALPHAS_DEFAULT}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_full_humaneval}"
BACKEND="${BACKEND:-hf}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
LOG_DIR="${LOG_DIR:-/tmp/alpha_qwen3_humaneval_logs}"

MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

if [ "$BACKEND" = "vllm" ]; then
  if [ ! -x "$VLLM_VENV/bin/python" ]; then
    cat >&2 <<EOF
Error: vLLM venv not found at '$VLLM_VENV/bin/python'.
Override with VLLM_VENV=/path/to/.venv (e.g. /workspace/.venv on a
consolidated single-venv pod).
EOF
    exit 4
  fi
fi

if [ -n "${GPUS:-}" ]; then
  GPU_LIST=($GPUS)
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found and GPUS not set." >&2; exit 2
  fi
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
  [ "$GPU_COUNT" -eq 0 ] && { echo "Error: 0 GPUs." >&2; exit 2; }
  GPU_LIST=()
  for ((i=0; i<GPU_COUNT; i++)); do GPU_LIST+=($i); done
fi
N_GPUS=${#GPU_LIST[@]}

mkdir -p "$LOG_DIR"

if [ -n "${ONLY_ALPHA:-}" ]; then ALPHAS="$ONLY_ALPHA"; fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Rényi-α p-less HumanEval-164 sweep — Qwen3-8B (thinking enabled)"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Model:        $MODEL"
echo "  α grid:       $ALPHAS"
echo "  N samples:    $N_SAMPLES per task"
echo "  Temperature:  $TEMPERATURE"
echo "  Max tokens:   $MAX_NEW_TOKENS"
echo "  Backend:      $BACKEND"
echo "  Results dir:  $RESULTS_DIR"
echo "  GPUs to use:  ${GPU_LIST[*]} (count: $N_GPUS)"
echo "  Logs:         $LOG_DIR"
[ -n "${MAX_PROBLEMS:-}" ] && echo "  Max problems: $MAX_PROBLEMS (smoke)"
echo "═══════════════════════════════════════════════════════════════════════"
echo

run_arm() {
  local gpu="$1"
  local alpha="$2"
  local log="$LOG_DIR/qwen3_a${alpha}.log"
  echo "[GPU $gpu] start α=$alpha at $(date +%H:%M:%S) → $log"
  if [ "$BACKEND" = "vllm" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    "$VLLM_VENV/bin/python" -m bench.humaneval \
      --model "$MODEL" \
      --method pless_alpha --alpha "$alpha" \
      --temperature "$TEMPERATURE" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend vllm \
      --enable-thinking \
      --results-dir "$RESULTS_DIR" \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" uv run python -m bench.humaneval \
      --model "$MODEL" \
      --method pless_alpha --alpha "$alpha" \
      --temperature "$TEMPERATURE" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend hf \
      --enable-thinking \
      --results-dir "$RESULTS_DIR" \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  fi
  echo "[GPU $gpu] done α=$alpha at $(date +%H:%M:%S)"
}

ALPHA_ARR=($ALPHAS)
N_ALPHAS=${#ALPHA_ARR[@]}

QUEUE=()
for ((lane=0; lane<N_GPUS; lane++)); do QUEUE[$lane]=""; done
for ((i=0; i<N_ALPHAS; i++)); do
  lane=$((i % N_GPUS))
  QUEUE[$lane]="${QUEUE[$lane]} ${ALPHA_ARR[$i]}"
done

for ((lane=0; lane<N_GPUS; lane++)); do
  echo "  GPU ${GPU_LIST[$lane]}: α =${QUEUE[$lane]}"
done
echo

PIDS=()
for ((lane=0; lane<N_GPUS; lane++)); do
  gpu="${GPU_LIST[$lane]}"
  alphas_for_lane="${QUEUE[$lane]}"
  (
    for alpha in $alphas_for_lane; do
      run_arm "$gpu" "$alpha"
    done
  ) &
  PIDS+=($!)
done

echo "  Lane PIDs: ${PIDS[*]}"
echo "  Monitor: tail -f $LOG_DIR/*.log"
echo
wait "${PIDS[@]}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Qwen3-8B HumanEval α-sweep complete at $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

out_dir="$RESULTS_DIR/Qwen--Qwen3-8B/humaneval"
if [ -d "$out_dir" ]; then
  echo "  Outputs in $out_dir:"
  ls -lh "$out_dir"/*.jsonl 2>/dev/null || echo "    (no JSONLs found)"
  echo
fi

echo "Next steps:"
echo "  for f in $out_dir/pless_alpha_think_a*_t1.0.jsonl; do"
echo "    uv run python -m bench.eval --dataset humaneval --results-file \"\$f\""
echo "  done"
