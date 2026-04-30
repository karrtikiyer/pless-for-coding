#!/usr/bin/env bash
# Pure-temperature re-run of 5 high-temp Qwen3-8B split-decoding configs.
#
# Why: SPLIT_SAMPLERS["temp_standard"] bakes in top_p=0.95 + top_k=20, which
# meaningfully prunes the distribution at temp_think=1.5. The new
# SPLIT_SAMPLERS["temp_pure"] sampler removes that filter
# (top_p=1.0, top_k=0). This script re-runs only the 5 configs where the
# confound is most material — split decoding with temp_think > 1.0:
#
#   H7P : split temp_pure 1.5 → pless 1.0
#   H8P : split temp_pure 1.5 → pless 1.5
#   H9P : split temp_pure 1.5 → pless 2.0
#   H10P: split temp_pure 1.5 → pless 3.0
#   T15P: split temp_pure 1.5 → temp_pure 1.5  (both phases unfiltered)
#
# Existing JSONLs (split_temp_standard_*.jsonl) are NOT touched — new files
# auto-derive to split_temp_pure_*.jsonl from runner.py:99.
#
# Multi-GPU: configs are assigned round-robin across GPUs.
# Each GPU loads the model once and runs its share sequentially.
#
# Usage:
#   bash run_pure_temp_rerun_mbpp.sh --gpus 0,1,2,3,4 [--dry-run]
#   bash run_pure_temp_rerun_mbpp.sh --gpus 0,1,2     [--dry-run]
#   bash run_pure_temp_rerun_mbpp.sh --gpus 0,1
#   bash run_pure_temp_rerun_mbpp.sh --gpus 0          # single GPU (sequential)
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

echo "=== Pure-Temp Re-run — Qwen3-8B (5 configs) ==="
echo "    GPUs: ${GPUS[*]} (${NUM_GPUS} total)"
echo "    n_samples: $N_SAMPLES | max_tokens: $MAX_TOKENS"
$DRY_RUN && echo "    (DRY RUN)"

# ---------------------------------------------------------------------------
# Validate CLI contract — confirm temp_pure is registered in SPLIT_SAMPLERS
# ---------------------------------------------------------------------------
echo ""
echo "Validating CLI contract..."
if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'split',
            '--sampler-think', 'temp_pure', '--sampler-code', 'pless',
            '--temp-think', '1.5', '--temp-code', '1.5',
            '--enable-thinking', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
"; then
  echo "ERROR: --sampler-think temp_pure not accepted — check bench/sampler_bridge.py SPLIT_SAMPLERS dict"
  exit 1
fi

if ! uv run python -c "
import sys
sys.argv = ['bench', '--model', 'x', '--method', 'split',
            '--sampler-think', 'temp_pure', '--sampler-code', 'temp_pure',
            '--temp-think', '1.5', '--temp-code', '1.5',
            '--enable-thinking', '--mbpp-config', 'full']
from bench.runner import parse_args
parse_args()
"; then
  echo "ERROR: temp_pure → temp_pure (T15P) CLI validation failed"
  exit 1
fi
echo "  CLI OK (temp_pure registered for both think and code phases)"

# ---------------------------------------------------------------------------
# Build config array: each entry is "sampler_think temp_think sampler_code temp_code label"
# Order: H7P → H8P → H9P → H10P → T15P
# ---------------------------------------------------------------------------
CONFIGS=(
  "temp_pure 1.5 pless     1.0 H7P:temp_pure_1.5→pless_1.0"
  "temp_pure 1.5 pless     1.5 H8P:temp_pure_1.5→pless_1.5"
  "temp_pure 1.5 pless     2.0 H9P:temp_pure_1.5→pless_2.0"
  "temp_pure 1.5 pless     3.0 H10P:temp_pure_1.5→pless_3.0"
  "temp_pure 1.5 temp_pure 1.5 T15P:temp_pure_1.5→temp_pure_1.5"
)

TOTAL=${#CONFIGS[@]}
echo ""
echo "Total configs: $TOTAL"
echo "Configs per GPU: $(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))"
echo ""

# ---------------------------------------------------------------------------
# Create log directory
# ---------------------------------------------------------------------------
LOG_DIR="logs/pure_temp_rerun_$(date +%Y%m%d_%H%M%S)"
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
echo ""
echo "Next steps:"
echo "  1. Eval the 5 new JSONLs:"
echo "     for f in split_temp_pure_t1.5_pless_t1.0_think_t1.0 \\"
echo "              split_temp_pure_t1.5_pless_t1.5_think_t1.0 \\"
echo "              split_temp_pure_t1.5_pless_t2.0_think_t1.0 \\"
echo "              split_temp_pure_t1.5_pless_t3.0_think_t1.0 \\"
echo "              split_temp_pure_t1.5_temp_pure_t1.5_think_t1.0; do"
echo "       uv run python -m bench.eval \\"
echo "         --results-file $RESULTS_DIR/Qwen--Qwen3-8B/\${f}.jsonl --dataset mbpp"
echo "     done"
echo "  2. Refresh the analysis report:"
echo "     uv run python -m bench.eval.split_decoding_analysis \\"
echo "       --results-dir $RESULTS_DIR/Qwen--Qwen3-8B \\"
echo "       --output-dir  $RESULTS_DIR/Qwen--Qwen3-8B/analysis"
