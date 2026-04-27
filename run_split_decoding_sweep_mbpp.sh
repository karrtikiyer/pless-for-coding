#!/usr/bin/env bash
# Extended split decoding sweep on Qwen3-8B: temp × pless grid in both directions.
#
# 18 configs across two directions:
#   Direction 1 (H1-H9): temp think → pless code
#     temp-think ∈ {0.7, 0.8, 0.9}, temp-code ∈ {1.0, 1.5, 2.0}
#   Direction 2 (R1-R9): pless think → temp code
#     temp-think ∈ {1.0, 1.5, 2.0}, temp-code ∈ {0.7, 0.8, 0.9}
#
# Multi-GPU: configs are assigned round-robin across GPUs.
# Each GPU loads the model and runs its share sequentially.
#
# Usage:
#   bash run_split_decoding_sweep_mbpp.sh --gpus 0,1,2 [--dry-run]
#   bash run_split_decoding_sweep_mbpp.sh --gpus 0,1   [--dry-run]
#   bash run_split_decoding_sweep_mbpp.sh --gpus 0      # single GPU
set -e

DRY_RUN=false
GPU_LIST="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true; shift ;;
    --gpus)     GPU_LIST="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

MODEL="Qwen/Qwen3-8B"
RESULTS_DIR="results/pless_full_mbpp_results"
N_SAMPLES=10
MAX_TOKENS=8192
MBPP_CONFIG="full"

echo "=== Split Decoding Sweep — Qwen3-8B ==="
echo "    GPUs: ${GPUS[*]} (${NUM_GPUS} total)"
echo "    n_samples: $N_SAMPLES | max_tokens: $MAX_TOKENS"
$DRY_RUN && echo "    (DRY RUN)"

# ---------------------------------------------------------------------------
# Validate CLI contract before any GPU work
# ---------------------------------------------------------------------------
echo ""
echo "Validating CLI contract..."
if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'split',
            '--sampler-think', 'temp_standard', '--sampler-code', 'pless',
            '--temp-think', '0.7', '--temp-code', '1.0',
            '--enable-thinking', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
"; then
  echo "ERROR: CLI contract validation failed — check bench/runner.py"
  exit 1
fi

# Also validate reverse direction
if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'split',
            '--sampler-think', 'pless', '--sampler-code', 'temp_standard',
            '--temp-think', '1.0', '--temp-code', '0.7',
            '--enable-thinking', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
"; then
  echo "ERROR: CLI contract validation failed for reverse direction"
  exit 1
fi
echo "  CLI OK (both directions)"

# ---------------------------------------------------------------------------
# Build config array: each entry is "sampler_think temp_think sampler_code temp_code label"
# ---------------------------------------------------------------------------
CONFIGS=()

# Direction 1: temp think → pless code (H1–H9)
for t_think in 0.7 0.8 0.9; do
  for t_code in 1.0 1.5 2.0; do
    CONFIGS+=("temp_standard ${t_think} pless ${t_code} H:temp${t_think}→pless@${t_code}")
  done
done

# Direction 2: pless think → temp code (R1–R9)
for t_think in 1.0 1.5 2.0; do
  for t_code in 0.7 0.8 0.9; do
    CONFIGS+=("pless ${t_think} temp_standard ${t_code} R:pless@${t_think}→temp${t_code}")
  done
done

TOTAL=${#CONFIGS[@]}
echo ""
echo "Total configs: $TOTAL"
echo "Configs per GPU: $(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))"
echo ""

# ---------------------------------------------------------------------------
# Create log directory
# ---------------------------------------------------------------------------
LOG_DIR="logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

# ---------------------------------------------------------------------------
# Assign configs round-robin to GPUs and launch
# ---------------------------------------------------------------------------
run_config() {
  local gpu_id="$1"
  local sampler_think="$2"
  local temp_think="$3"
  local sampler_code="$4"
  local temp_code="$5"
  local label="$6"

  echo "[GPU $gpu_id] >>> $label"
  echo "[GPU $gpu_id]   Started: $(date)"

  if ! $DRY_RUN; then
    CUDA_VISIBLE_DEVICES="$gpu_id" uv run python -m bench \
      --model "$MODEL" \
      --method split \
      --sampler-think "$sampler_think" \
      --sampler-code "$sampler_code" \
      --temp-think "$temp_think" \
      --temp-code "$temp_code" \
      --enable-thinking \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_TOKENS" \
      --mbpp-config "$MBPP_CONFIG" \
      --results-dir "$RESULTS_DIR"
  fi

  echo "[GPU $gpu_id]   Finished: $(date)"
}

# Build per-GPU config lists and launch each GPU as a background job
for gpu_idx in $(seq 0 $(( NUM_GPUS - 1 ))); do
  gpu="${GPUS[$gpu_idx]}"
  log_file="$LOG_DIR/gpu_${gpu}.log"

  (
    echo "=== GPU $gpu — started $(date) ==="

    for i in $(seq 0 $(( TOTAL - 1 ))); do
      # Round-robin assignment
      if (( i % NUM_GPUS != gpu_idx )); then
        continue
      fi

      # Parse config fields
      read -r sampler_think temp_think sampler_code temp_code label <<< "${CONFIGS[$i]}"
      run_config "$gpu" "$sampler_think" "$temp_think" "$sampler_code" "$temp_code" "$label"
      echo ""
    done

    echo "=== GPU $gpu — all done $(date) ==="
  ) > "$log_file" 2>&1 &

  echo "Launched GPU $gpu (PID $!) → $log_file"
done

echo ""
echo "All GPUs launched. Waiting for completion..."
echo "  Monitor: tail -f $LOG_DIR/gpu_*.log"
wait
echo ""
echo "=== All GPUs finished $(date) ==="
