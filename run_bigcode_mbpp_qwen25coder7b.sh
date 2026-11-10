#!/usr/bin/env bash
# run_bigcode_mbpp_qwen25coder7b.sh — MBPP-full (500 tasks) for Qwen/Qwen2.5-Coder-7B (base), BigCode zero-shot prompt.
# Analysis only (2026-11), NOT for the paper. Default root (2026-11-10): results/mbpp_qwen25coder7b_rerun/Qwen--Qwen2.5-Coder-7B/
#   The original unseeded run (results/mbpp_qwen25coder7b_2026_11/) now lives in the frozen archive
#   results/regen_2026_11/mbpp/unseeded_qwen25coder7b/ (old -> new: results/regen_2026_11/path_mapping.tsv); the shared
#   helper refuses any RESULTS_DIR resolving inside results/regen_2026_11/.
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
#   SEED=1234 / SINGLE_BOS=1 (opt-in, 2026-11-08): append --seed 1234 / --single-bos to every config; same
#     file names; put seeded reruns in a NEW root, e.g. RESULTS_DIR=results/mbpp_seeded_1234 SEED=1234 ./<this script>
#     (never inside the frozen archive results/regen_2026_11/: refused, exit 8, see scripts/mbpp_staging_common.sh)
#     (the runner refuses to resume a file generated with other settings)
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="Qwen/Qwen2.5-Coder-7B"
RESULTS_DIR="${RESULTS_DIR:-results/mbpp_qwen25coder7b_rerun}"   # env override: see scripts/mbpp_staging_common.sh
LOG_TAG="mbpp_qwen25coder7b"
PROMPT_ARGS=(--prompt-style bigcode)

source scripts/mbpp_staging_common.sh
# "<output file stem>|<bench args>": the 6 MBPP Table I configs + the Hasan et al. config (tables in the helper)
CONFIGS=("${MBPP_TABLE1_BIGCODE[@]}" "$MBPP_HASAN_BIGCODE")
run_configs
