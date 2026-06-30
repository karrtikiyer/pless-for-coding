#!/usr/bin/env bash
# Replicate Circular-Reasoning Figs 3b & 4 (arXiv:2601.05693) on our p-less loop
# traces, for Qwen3-8B and DeepSeek-R1-Distill-Llama-8B.
#
# Phase 0 (screen) and Phase 2 (plot) are CPU; Phase 1 (extract) needs a CUDA pod
# (~80 GB recommended — teacher-forces up to ~38k tokens with all-layer hidden
# states + logits). Run phases 0 and 2 locally; run phase 1 on the pod.
#
# Usage:
#   ./run_loop_collapse_replication.sh screen   # CPU, local
#   ./run_loop_collapse_replication.sh extract  # GPU pod
#   ./run_loop_collapse_replication.sh plot      # CPU
set -euo pipefail

ROOT="results/loop_collapse_replication"
declare -A JSONL=(
  ["Qwen/Qwen3-8B"]="results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl"
  ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl"
)
declare -A TAG=(
  ["Qwen/Qwen3-8B"]="Qwen--Qwen3-8B"
  ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-ai--DeepSeek-R1-Distill-Llama-8B"
)
declare -A LABEL=(
  ["Qwen/Qwen3-8B"]="Qwen3-8B"
  ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="DeepSeek-R1-Distill-Llama-8B"
)

phase="${1:-screen}"
for model in "Qwen/Qwen3-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"; do
  tag="${TAG[$model]}"; dir="$ROOT/$tag"
  case "$phase" in
    screen)
      HF_HUB_OFFLINE=1 uv run python scripts/loop_collapse_screen.py \
        --model "$model" --jsonl "${JSONL[$model]}" --out "$dir/manifest.jsonl"
      ;;
    extract)
      [ -f "$dir/manifest.jsonl" ] || { echo "skip $tag: no manifest (no verbatim loops?)"; continue; }
      HF_HUB_OFFLINE=1 uv run python scripts/loop_collapse_extract.py \
        --manifest "$dir/manifest.jsonl" --model "$model" --out-dir "$dir"
      ;;
    plot)
      [ -d "$dir/vectors" ] || { echo "skip $tag: no vectors"; continue; }
      uv run python scripts/loop_collapse_plot.py \
        --vec-dir "$dir/vectors" --out-dir "$dir/figures" --model-label "${LABEL[$model]}"
      ;;
    *) echo "unknown phase: $phase (screen|extract|plot)"; exit 1;;
  esac
done
