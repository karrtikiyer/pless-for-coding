#!/usr/bin/env bash
# run_bigcode_mbpp_qwen25coder7b.sh — MBPP-full (500 tasks) for Qwen/Qwen2.5-Coder-7B (base), BigCode zero-shot prompt.
# Analysis only (2026-11), NOT for the paper. Staging root: results/mbpp_qwen25coder7b_2026_11/Qwen--Qwen2.5-Coder-7B/
#
# Same `python -m bench` flags as the Qwen2.5-Coder-1.5B / -3B MBPP Table I rows (verified against their JSONL
# records): HF backend, --prompt-style bigcode, --mbpp-config full, runner defaults n-samples 10 /
# max-new-tokens 512; greedy and beam with --n-samples 1 (run_new_samplers_mbpp.sh).
#
#   CUDA_VISIBLE_DEVICES=0 ./run_bigcode_mbpp_qwen25coder7b.sh               # full run (resumable)
#   MAX_PROBLEMS=3 CUDA_VISIBLE_DEVICES=0 ./run_bigcode_mbpp_qwen25coder7b.sh # smoke (resumes into the same files)
#   DRY_RUN=1 ./run_bigcode_mbpp_qwen25coder7b.sh                             # print commands only
#   EXTRA_PLESS_TEMPS=1 CUDA_VISIBLE_DEVICES=0 ./run_bigcode_mbpp_qwen25coder7b.sh   # ONLY p-less / p-less-norm at T=1.0 and T=0.6
#     (user request 2026-11-08; same flags as the T=0.7 configs; see scripts/mbpp_staging_common.sh)
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="Qwen/Qwen2.5-Coder-7B"
RESULTS_DIR="results/mbpp_qwen25coder7b_2026_11"
LOG_TAG="mbpp_qwen25coder7b"
PROMPT_ARGS=(--prompt-style bigcode)
CONFIGS=(   # "<output file stem>|<bench args>"
  "greedy_bigcode_t1.0|--method greedy --temperature 1.0 --n-samples 1"
  "beam4_bigcode_t1.0|--method beam --num-beams 4 --temperature 1.0 --n-samples 1"
  "temp_bigcode_t0.7|--method temp --temperature 0.7"
  "top_p0.9_bigcode_t1.0|--method top_p --top-p 0.9 --temperature 1.0"
  "pless_bigcode_t0.7|--method pless --temperature 0.7"
  "pless_norm_bigcode_t0.7|--method pless_norm --temperature 0.7"
  "top_p0.95_bigcode_t0.2|--method top_p --top-p 0.95 --temperature 0.2"   # Hasan et al. config
)

source scripts/mbpp_staging_common.sh
run_configs
