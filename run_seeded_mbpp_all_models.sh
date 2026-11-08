#!/usr/bin/env bash
# run_seeded_mbpp_all_models.sh MODEL_ID [MODEL_ID ...] — SEEDED MBPP-full (500 tasks) regeneration, models run in the
# given order. Analysis only (2026-11-08), NOT for the paper. Output root: results/mbpp_seeded_2026_11/<model dir>/
#
# Per model, every config with --seed 42 (per-task reseeding, bench/seeding.py):
#   base (bench/prompts.py:is_instruct_model False), each with the prompt of its MBPP Table I rows (user, 2026-11-08):
#     bigcode: OpenCodeInterpreter-DS-1.3B, Qwen2.5-Coder-1.5B / -3B / -7B — --prompt-style bigcode, flags of
#       run_bigcode_mbpp_qwen25coder7b.sh ("_bigcode" stems)        6 Table I + top-p 0.95 T0.2 (Hasan) + 4 = 11
#     3-shot "paper" prompt: CodeLlama-7b-hf, Llama-2-7b-hf — runner default (no --prompt-style, --n-shots 3), the
#       flags / stems of their April rows (run_bench.sh, run_new_samplers_mbpp.sh, run_top_p_mbpp_all_models.sh)
#                                                                    6 Table I + 4 extra p-less          = 10
#   chat — flags of run_chat_mbpp_qwen25coder7binstruct.sh + --single-bos (one BOS; no-op for Qwen, which has no BOS):
#     CodeLlama-7b-Instruct-hf, Llama-2-7b-chat-hf, Qwen2.5-Coder-7B-Instruct  6 Table I + 4 extra p-less  = 10
#   Table I: greedy n=1, beam4 n=1, temp T0.7, top_p 0.9 T1.0, pless T0.7, pless_norm T0.7.
#   Extra p-less: pless / pless_norm at T1.0 and T0.6 (same flags as the T0.7 configs).
# Config tables, builders and checks live in scripts/mbpp_staging_common.sh (shared with the staging scripts): stop on
# the first failing config, FAIL on any "Error on task_id", 500 unique task_ids per file, transformers>=5, runner
# resume (+ its seed / single_bos resume guard), logs/mbpp_seeded_<model dir>_<config>_<ts>.log.
#
#   CUDA_VISIBLE_DEVICES=0 ./run_seeded_mbpp_all_models.sh Qwen/Qwen2.5-Coder-7B m-a-p/OpenCodeInterpreter-DS-1.3B ...
#   MAX_PROBLEMS=3 CUDA_VISIBLE_DEVICES=0 ./run_seeded_mbpp_all_models.sh <models>   # smoke (resumes into the same files)
#   DRY_RUN=1 ./run_seeded_mbpp_all_models.sh <models>                              # print commands only
# Environment: RESULTS_DIR (default results/mbpp_seeded_2026_11); PYTORCH_CUDA_ALLOC_CONF (default
#   expandable_segments:True, an allocator setting only; set it, even to empty, to override); CUDA_VISIBLE_DEVICES
#   is the caller's. SEED / SINGLE_BOS / EXTRA_PLESS_TEMPS are fixed by this script and refused if set.
# Evaluation is NOT run here: run `python -m bench.eval` locally (project CLAUDE.md gotcha 9).
set -euo pipefail
cd "$(dirname "$0")"

# model id|type  (hasan = bigcode base + Hasan config; paper = 3-shot default-prompt base; chat). Only these 9.
KNOWN_MODELS=(
  "m-a-p/OpenCodeInterpreter-DS-1.3B|hasan"
  "Qwen/Qwen2.5-Coder-1.5B|hasan"
  "Qwen/Qwen2.5-Coder-3B|hasan"
  "Qwen/Qwen2.5-Coder-7B|hasan"
  "codellama/CodeLlama-7b-hf|paper"
  "meta-llama/Llama-2-7b-hf|paper"
  "codellama/CodeLlama-7b-Instruct-hf|chat"
  "meta-llama/Llama-2-7b-chat-hf|chat"
  "Qwen/Qwen2.5-Coder-7B-Instruct|chat"
)

model_type() {   # print the type of $1, or fail
  local e
  for e in "${KNOWN_MODELS[@]}"; do
    [[ "${e%%|*}" == "$1" ]] && { echo "${e#*|}"; return 0; }
  done
  return 1
}

usage() {
  echo "usage: $0 MODEL_ID [MODEL_ID ...]   (known models:" >&2
  local e; for e in "${KNOWN_MODELS[@]}"; do echo "    ${e%%|*}" >&2; done
  echo "  )" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
for v in SEED SINGLE_BOS EXTRA_PLESS_TEMPS; do   # fixed here (seed 42; single BOS for chat; all configs)
  if [[ -n "${!v:-}" ]]; then echo "$v must be unset for $0 (got '${!v}')" >&2; exit 2; fi
done
seen=" "
for m in "$@"; do   # validate the whole queue before anything runs
  model_type "$m" > /dev/null || { echo "unknown model '$m'" >&2; usage; }
  [[ "$seen" != *" $m "* ]] || { echo "model '$m' given twice" >&2; exit 2; }
  seen+="$m "
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
RESULTS_DIR="${RESULTS_DIR:-results/mbpp_seeded_2026_11}"
SEED=42
SINGLE_BOS=""
MODEL_ID="$1"; LOG_TAG="mbpp_seeded"   # placeholders for the helper's source-time checks; set per model below
source scripts/mbpp_staging_common.sh

echo "[$(ist)] seeded MBPP queue ($#): $*"
echo "    RESULTS_DIR=$RESULTS_DIR  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" \
     " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
i=0
for m in "$@"; do
  i=$((i + 1))
  type=$(model_type "$m")
  MODEL_ID="$m"
  LOG_TAG="mbpp_seeded_${m//\//--}"
  case "$type" in
    chat)  PROMPT_ARGS=(); SINGLE_BOS=1;  CONFIGS=("${MBPP_TABLE1_DEFAULT_PROMPT[@]}") ;;
    paper) PROMPT_ARGS=(); SINGLE_BOS=""; CONFIGS=("${MBPP_TABLE1_DEFAULT_PROMPT[@]}") ;;
    hasan) PROMPT_ARGS=(--prompt-style bigcode); SINGLE_BOS=""
           CONFIGS=("${MBPP_TABLE1_BIGCODE[@]}" "$MBPP_HASAN_BIGCODE") ;;
    *) echo "internal error: type '$type'" >&2; exit 2 ;;
  esac
  append_extra_pless_configs
  echo ""
  echo "=== model $i/$#: $m ($type, ${#CONFIGS[@]} configs)"
  run_configs
done
echo ""
echo "[$(ist)] seeded MBPP queue done ($# models)$([[ "$DRY_RUN" == 1 ]] && echo ' (DRY RUN)')"
