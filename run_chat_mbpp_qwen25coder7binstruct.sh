#!/usr/bin/env bash
# run_chat_mbpp_qwen25coder7binstruct.sh — MBPP-full (500 tasks) for Qwen/Qwen2.5-Coder-7B-Instruct, chat template.
# Analysis only (2026-11), NOT for the paper. Staging root:
#   results/mbpp_qwen25coder7binstruct_2026_11/Qwen--Qwen2.5-Coder-7B-Instruct/
#
# Same `python -m bench` flags as the CodeLlama-7b-Instruct-hf / Llama-2-7b-chat-hf MBPP Table I rows
# (run_bench.sh, run_new_samplers_mbpp.sh, run_top_p_mbpp_all_models.sh; no --prompt-style flag: the name
# contains "instruct", so bench/prompts.py:is_instruct_model routes it to format_prompt_instruct, no stop
# strings, no file-name suffix). Runner defaults n-samples 10 / max-new-tokens 512; greedy and beam --n-samples 1.
# No Hasan config (Hasan et al. evaluated the base model).
#
#   CUDA_VISIBLE_DEVICES=0 ./run_chat_mbpp_qwen25coder7binstruct.sh
#   MAX_PROBLEMS=3 CUDA_VISIBLE_DEVICES=0 ./run_chat_mbpp_qwen25coder7binstruct.sh   # smoke
#   DRY_RUN=1 ./run_chat_mbpp_qwen25coder7binstruct.sh
#   EXTRA_PLESS_TEMPS=1 CUDA_VISIBLE_DEVICES=0 ./run_chat_mbpp_qwen25coder7binstruct.sh   # ONLY p-less / p-less-norm at T=1.0 and T=0.6
#     (user request 2026-11-08; same flags as the T=0.7 configs; see scripts/mbpp_staging_common.sh)
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="Qwen/Qwen2.5-Coder-7B-Instruct"
RESULTS_DIR="results/mbpp_qwen25coder7binstruct_2026_11"
LOG_TAG="mbpp_qwen25coder7binstruct"
PROMPT_ARGS=()
CONFIGS=(   # "<output file stem>|<bench args>"
  "greedy_t1.0|--method greedy --temperature 1.0 --n-samples 1"
  "beam4_t1.0|--method beam --num-beams 4 --temperature 1.0 --n-samples 1"
  "temp_t0.7|--method temp --temperature 0.7"
  "top_p0.9_t1.0|--method top_p --top-p 0.9 --temperature 1.0"
  "pless_t0.7|--method pless --temperature 0.7"
  "pless_norm_t0.7|--method pless_norm --temperature 0.7"
)

source scripts/mbpp_staging_common.sh
run_configs
