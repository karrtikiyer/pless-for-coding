#!/bin/bash
# GSM8K α-sweep — pass@k + diversity on math reasoning to test the
# bimodality→α-effect mechanism story we've established on code.
#
# Default scope (matches the existing 4-model × MBPP α-sweep scale):
#   * 1 model: Qwen/Qwen2.5-Coder-7B-Instruct (HF baseline; smoke 2026-05-22
#              showed 78% pass@1, 0% code-format slippage, clear α-driven
#              reasoning variation)
#   * 4 α arms: 2.0, 2.5, 3.0, 5.0
#   * 500 problems (random subset of GSM8K test, seed=0)
#   * 10 samples per problem
#   * Wei 2022 8-shot CoT prompt (verbatim from arXiv:2201.11903 Table 20)
#
# Total: 500 × 10 × 4 = 20,000 generations
# Est. runtime: ~25-30 GPU-hours on a single H100 with HF backend;
#               4× faster if 4 α arms parallel on 4 GPUs.
#
# Backend: HF only (no vLLM — we want numerical equivalence with our
# code-side data, and vLLM does not guarantee that per its own docs;
# see commit messages on b540978/1ef6128 for the discussion).
#
# Outputs:
#   results/pless_alpha_full_gsm8k/<model>/pless_alpha_a{α}_t1.0.jsonl
#   results/pless_alpha_full_gsm8k/<model>/metrics/pless_alpha_a{α}_t1.0_metrics.json
#
# Usage:
#   ./run_pless_alpha_gsm8k.sh
#   ALPHAS="2.0 5.0" ./run_pless_alpha_gsm8k.sh         # only the extremes
#   N_PROBLEMS=50 N_SAMPLES=3 ./run_pless_alpha_gsm8k.sh # smoke
#
# Env overrides:
#   MODELS         space-separated HF ids (default Qwen2.5-Coder-7B-Instruct)
#   ALPHAS         space-separated α grid (default "2.0 2.5 3.0 5.0")
#   N_PROBLEMS     random subset size (default 500)
#   N_SAMPLES      samples per problem (default 10)
#   SEED           subset random seed (default 0)
#   TEMPERATURE    sampler temperature (default 1.0)
#   MAX_NEW_TOKENS (default 400)
#   RESULTS_DIR    (default results/pless_alpha_full_gsm8k)
#   LOG_DIR        (default /tmp/alpha_gsm8k_logs)
#   GPUS           comma-separated CUDA_VISIBLE_DEVICES values (default auto)

set -euo pipefail

MODELS_DEFAULT="Qwen/Qwen2.5-Coder-7B-Instruct"
ALPHAS_DEFAULT="2.0 2.5 3.0 5.0"

MODELS="${MODELS:-$MODELS_DEFAULT}"
ALPHAS="${ALPHAS:-$ALPHAS_DEFAULT}"
N_PROBLEMS="${N_PROBLEMS:-500}"
N_SAMPLES="${N_SAMPLES:-10}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_full_gsm8k}"
LOG_DIR="${LOG_DIR:-/tmp/alpha_gsm8k_logs}"

# Detect GPUs if not specified
if [ -z "${GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_GPUS" -ge 1 ]; then
      GPUS=$(seq -s, 0 $((N_GPUS - 1)))
    fi
  fi
fi

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "GSM8K α-sweep"
echo "  Models:        $MODELS"
echo "  α arms:        $ALPHAS"
echo "  Problems:      $N_PROBLEMS (random subset, seed=$SEED)"
echo "  Samples/task:  $N_SAMPLES"
echo "  Temperature:   $TEMPERATURE"
echo "  Max tokens:    $MAX_NEW_TOKENS"
echo "  Results dir:   $RESULTS_DIR"
echo "  Log dir:       $LOG_DIR"
echo "  GPUs:          ${GPUS:-cpu/mps}"
echo "═══════════════════════════════════════════════════════════════════════"

run_arm() {
  local gpu="$1"
  local model="$2"
  local alpha="$3"
  local model_slug; model_slug=$(echo "$model" | tr '/' '-')
  local log="$LOG_DIR/${model_slug}_a${alpha}.log"
  local prefix=""
  if [ -n "$gpu" ]; then
    prefix="CUDA_VISIBLE_DEVICES=$gpu "
  fi
  echo "[gpu=$gpu] start $model_slug / α=$alpha → $log"
  bash -c "$prefix uv run python -m bench.gsm8k \
    --model '$model' \
    --method pless_alpha --alpha $alpha \
    --temperature $TEMPERATURE \
    --n-samples $N_SAMPLES \
    --max-new-tokens $MAX_NEW_TOKENS \
    --n-problems $N_PROBLEMS \
    --seed $SEED \
    --results-dir '$RESULTS_DIR' \
    > '$log' 2>&1"
  echo "[gpu=$gpu] done $model_slug / α=$alpha at $(date +%H:%M:%S)"
}

# Distribute α arms across GPUs using a per-GPU lane queue.
# - With N_GPUS = 1: one lane with all 4 α arms, runs SEQUENTIALLY (only one
#   ~14GB bf16 7B model loaded at a time → fits in any 24GB+ GPU).
# - With N_GPUS = 4: four lanes each with 1 α arm, runs IN PARALLEL.
# - With N_GPUS = 2: two lanes (α=2.0,3.0 on GPU 0; α=2.5,5.0 on GPU 1).
# Matches the lane pattern in run_pless_alpha_apps_all_models.sh.
GPU_ARR=()
if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPU_ARR <<<"$GPUS"
fi
N_GPUS=${#GPU_ARR[@]}
ALPHA_ARR=($ALPHAS)
N_ALPHAS=${#ALPHA_ARR[@]}

for MODEL in $MODELS; do
  echo "═══ Model: $MODEL ═══════════════════════════════════════════════════"

  if [ "$N_GPUS" -eq 0 ]; then
    # No GPU detected (MPS / CPU): one lane, fully sequential.
    for ALPHA in $ALPHAS; do
      run_arm "" "$MODEL" "$ALPHA"
    done
  else
    # Build per-GPU α queues via modulo assignment.
    QUEUE=()
    for ((lane=0; lane<N_GPUS; lane++)); do QUEUE[$lane]=""; done
    for ((i=0; i<N_ALPHAS; i++)); do
      lane=$((i % N_GPUS))
      QUEUE[$lane]="${QUEUE[$lane]} ${ALPHA_ARR[$i]}"
    done
    for ((lane=0; lane<N_GPUS; lane++)); do
      echo "  GPU ${GPU_ARR[$lane]}: α =${QUEUE[$lane]}"
    done

    PIDS=()
    for ((lane=0; lane<N_GPUS; lane++)); do
      gpu="${GPU_ARR[$lane]}"
      alphas_for_lane="${QUEUE[$lane]}"
      (
        for alpha in $alphas_for_lane; do
          run_arm "$gpu" "$MODEL" "$alpha"
        done
      ) &
      PIDS+=($!)
    done
    wait "${PIDS[@]}"
  fi

  echo "═══ Model $MODEL done at $(date) ═══"
done

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "Generation done. Run the evaluator on each JSONL to compute pass@k + diversity:"
for MODEL in $MODELS; do
  model_slug=$(echo "$MODEL" | tr '/' '-')
  for ALPHA in $ALPHAS; do
    jsonl="$RESULTS_DIR/$model_slug/pless_alpha_a${ALPHA}_t${TEMPERATURE}.jsonl"
    echo "  uv run python -m bench.gsm8k.eval_runner --results-file $jsonl"
  done
done
echo "═══════════════════════════════════════════════════════════════════════"
