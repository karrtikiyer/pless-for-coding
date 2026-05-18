#!/bin/bash
# Rényi-α p-less sweep on HumanEval across two model families.
#
# Replicates the MBPP-500 sweep recipe on HumanEval-164:
#   * 4 α-arms: {2.0, 2.5, 3.0, 5.0}
#   * 10 samples per task
#   * T=1.0, HF backend
#   * α=2.0 is the sanity gate (must match existing pless@T=1.0 baseline)
#
# Default models:
#   - Qwen/Qwen2.5-Coder-7B-Instruct
#   - codellama/CodeLlama-7b-Instruct-hf
#
# Multi-GPU parallelism: auto-detects the number of visible CUDA GPUs and
# distributes α-arms across them, one arm per GPU running sequentially when
# arms > GPUs. Models are run sequentially (one model at a time) so a single
# model occupies all available GPUs.
#
# Usage:
#   ./run_pless_alpha_humaneval.sh
#
# Env var overrides:
#   MODELS         space-separated HF model IDs (default: the two above)
#   ALPHAS         space-separated α values     (default: "2.0 2.5 3.0 5.0")
#   N_SAMPLES      samples per problem          (default: 10)
#   TEMPERATURE    T parameter                  (default: 1.0)
#   MAX_NEW_TOKENS                              (default: 512)
#   RESULTS_DIR                                 (default: results/pless_alpha_full_humaneval)
#   GPUS           explicit GPU index list,
#                  e.g. "0 1 2 3"               (default: auto-detected via nvidia-smi)
#   MAX_PROBLEMS   cap (for smoke tests)        (default: unset, full 164)
#   ONLY_ALPHA     run only this α              (default: unset)
#   LOG_DIR        per-arm log destination      (default: /tmp/alpha_humaneval_logs)
#
# Resume-friendly: the underlying bench.humaneval runner appends to JSONL
# and skips task_ids already present. Re-running this script will resume
# any interrupted arms cleanly. Wrap the whole thing in tmux for a 5–20h run.

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MODELS_DEFAULT="Qwen/Qwen2.5-Coder-7B-Instruct codellama/CodeLlama-7b-Instruct-hf"
ALPHAS_DEFAULT="2.0 2.5 3.0 5.0"

MODELS="${MODELS:-$MODELS_DEFAULT}"
ALPHAS="${ALPHAS:-$ALPHAS_DEFAULT}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_full_humaneval}"
LOG_DIR="${LOG_DIR:-/tmp/alpha_humaneval_logs}"
MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

# ── GPU auto-detection ──────────────────────────────────────────────────────
if [ -n "${GPUS:-}" ]; then
  GPU_LIST=($GPUS)
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found and GPUS not set. Cannot determine GPU count." >&2
    exit 2
  fi
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
  if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: nvidia-smi reports 0 GPUs." >&2
    exit 2
  fi
  GPU_LIST=()
  for ((i=0; i<GPU_COUNT; i++)); do GPU_LIST+=($i); done
fi
N_GPUS=${#GPU_LIST[@]}

mkdir -p "$LOG_DIR"

# ── α filter (ONLY_ALPHA) ──────────────────────────────────────────────────
if [ -n "${ONLY_ALPHA:-}" ]; then
  ALPHAS="$ONLY_ALPHA"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Rényi-α p-less HumanEval sweep"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Models:       $MODELS"
echo "  α grid:       $ALPHAS"
echo "  N samples:    $N_SAMPLES per task"
echo "  Temperature:  $TEMPERATURE"
echo "  Results dir:  $RESULTS_DIR"
echo "  GPUs to use:  ${GPU_LIST[*]} (count: $N_GPUS)"
echo "  Logs:         $LOG_DIR"
[ -n "${MAX_PROBLEMS:-}" ] && echo "  Max problems: $MAX_PROBLEMS (smoke mode)"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# ── Per-arm runner ─────────────────────────────────────────────────────────
run_arm() {
  local gpu="$1"
  local model="$2"
  local alpha="$3"
  local model_slug; model_slug=$(echo "$model" | tr '/' '-' | tr '/' '-')
  local log="$LOG_DIR/${model_slug}_a${alpha}.log"

  echo "[GPU $gpu] starting model=$model α=$alpha at $(date +%H:%M:%S) → $log"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python -m bench.humaneval \
    --model "$model" \
    --method pless_alpha --alpha "$alpha" \
    --temperature "$TEMPERATURE" \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --backend hf \
    --results-dir "$RESULTS_DIR" \
    $MAX_PROBLEMS_FLAG \
    > "$log" 2>&1
  echo "[GPU $gpu] finished model=$model α=$alpha at $(date +%H:%M:%S)"
}

# ── Run each model: distribute α-arms across GPUs ──────────────────────────
ALPHA_ARR=($ALPHAS)
N_ALPHAS=${#ALPHA_ARR[@]}

for MODEL in $MODELS; do
  echo "═══ Model: $MODEL ═══════════════════════════════════════════════════"
  echo "  $N_ALPHAS α-arms over $N_GPUS GPU(s)"
  echo

  # Build per-GPU work queue using indexed array (bash 3.2 compatible).
  # QUEUE[lane] stores space-separated α-values assigned to that lane.
  QUEUE=()
  for ((lane=0; lane<N_GPUS; lane++)); do QUEUE[$lane]=""; done
  for ((i=0; i<N_ALPHAS; i++)); do
    lane=$((i % N_GPUS))
    QUEUE[$lane]="${QUEUE[$lane]} ${ALPHA_ARR[$i]}"
  done

  # Print assignment
  for ((lane=0; lane<N_GPUS; lane++)); do
    echo "  GPU ${GPU_LIST[$lane]}: α =${QUEUE[$lane]}"
  done
  echo

  # Launch each lane in the background; each lane processes its assigned α-arms sequentially.
  PIDS=()
  for ((lane=0; lane<N_GPUS; lane++)); do
    gpu="${GPU_LIST[$lane]}"
    alphas_for_lane="${QUEUE[$lane]}"
    (
      for alpha in $alphas_for_lane; do
        run_arm "$gpu" "$MODEL" "$alpha"
      done
    ) &
    PIDS+=($!)
  done

  # Wait for all lanes to finish for this model
  echo "  Lane PIDs: ${PIDS[*]}"
  echo "  Monitor: tail -f $LOG_DIR/*.log"
  echo
  wait "${PIDS[@]}"
  echo "═══ Model done: $MODEL at $(date) ═══"
  echo

  # Print resulting JSONLs
  model_slug=$(echo "$MODEL" | tr '/' '-' | tr '/' '-')
  out_dir="$RESULTS_DIR/${model_slug}/humaneval"
  if [ -d "$out_dir" ]; then
    echo "  Outputs in $out_dir:"
    ls -lh "$out_dir"/*.jsonl 2>/dev/null || echo "    (no JSONLs found)"
    echo
  fi
done

echo "═══════════════════════════════════════════════════════════════════════"
echo "  All models and α-arms complete at $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo
echo "Next steps (CPU-only, run on Mac after rsync, or here):"
echo "  for f in $RESULTS_DIR/*/humaneval/*.jsonl; do"
echo "    uv run python -m bench.eval --dataset humaneval --results-file \"\$f\""
echo "  done"
