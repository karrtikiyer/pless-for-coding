#!/usr/bin/env bash
# Experiment: P-less with high T1 on instruct model (MBPP-full).
#
# Tests whether high T1 (>1.0) opens the peaked instruct distribution enough
# for P-less to act as a quality filter — improving pass@1 at matched diversity
# vs plain temperature.
#
# Model: Qwen2.5-Coder-3B-Instruct (chat template, no --prompt-style needed)
# Dataset: MBPP-full (500 tasks × 10 samples)
#
# 13 configs:
#   A: Temperature baselines (t=0.2, 0.6, 0.8, 1.5) + top_p=0.95/t=0.2
#   B: P-less T1 only (0.6, 1.0, 1.5, 2.0, 3.0)
#   C: T2 targeted test at T1=2.0 (T2=2.0, T2=5.0)
#   D: Greedy reference
#
# Usage: bash run_pless_instruct_t1_mbpp.sh

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-Coder-3B-Instruct"
RESULTS_DIR="results/pless_full_mbpp_results"
MBPP_CONFIG="full"

COMMON_ARGS=(
  --model "$MODEL_ID"
  --mbpp-config "$MBPP_CONFIG"
  --results-dir "$RESULTS_DIR"
)

echo "=== P-less Instruct T1 Experiment ==="
echo "Model: ${MODEL_ID}"
echo "Results: ${RESULTS_DIR}"
echo ""

# --- Group A: Temperature baselines ---
echo "=== Group A: Temperature baselines ==="

for temp in 0.2 0.6 0.8 1.5; do
  echo "--- temp t=${temp} ---"
  uv run python -m bench \
    "${COMMON_ARGS[@]}" \
    --method temp \
    --temperature "$temp"
  echo ""
done

echo "--- top_p=0.95 t=0.2 ---"
uv run python -m bench \
  "${COMMON_ARGS[@]}" \
  --method top_p \
  --top-p 0.95 \
  --temperature 0.2
echo ""

# --- Group B: P-less T1 only ---
echo "=== Group B: P-less T1 sweep ==="

for t1 in 0.6 1.0 1.5 2.0 3.0; do
  echo "--- pless T1=${t1} ---"
  uv run python -m bench \
    "${COMMON_ARGS[@]}" \
    --method pless \
    --temperature "$t1"
  echo ""
done

# --- Group C: T2 targeted test at T1=2.0 ---
echo "=== Group C: T2 targeted test (T1=2.0) ==="

for t2 in 2.0 5.0; do
  echo "--- pless T1=2.0 T2=${t2} ---"
  uv run python -m bench \
    "${COMMON_ARGS[@]}" \
    --method pless \
    --temperature 2.0 \
    --post-temperature "$t2"
  echo ""
done

# --- Group D: Greedy reference ---
echo "=== Group D: Greedy reference ==="
echo "--- greedy ---"
uv run python -m bench \
  "${COMMON_ARGS[@]}" \
  --method greedy \
  --n-samples 1
echo ""

echo "=== All 13 configs complete ==="
echo "Run evaluation with:"
echo "  uv run python -m bench.eval.consolidated_eval --dataset mbpp --force"
