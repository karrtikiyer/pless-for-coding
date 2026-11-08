#!/usr/bin/env bash
# run_bos_check_mbpp_chat.sh MODEL OUT_ROOT — does the double BOS of the MBPP chat prompts matter? (E28, 2026-11-08)
#
# MBPP-full (500 tasks) greedy (n=1; the same `python -m bench` flags as the existing chat-model MBPP greedy rows, e.g.
# run_chat_mbpp_qwen25coder7binstruct.sh's greedy_t1.0) twice:
#   OUT_ROOT/double_bos/<model dir>/greedy_t1.0.jsonl   default encoding (CodeLlama-Instruct / Llama-2-chat: [1, 1, ...])
#   OUT_ROOT/single_bos/<model dir>/greedy_t1.0.jsonl   --single-bos ([1, ...]; records carry "single_bos": true)
# MODEL must be a chat/instruct model (name contains "instruct" or "chat"; bench/prompts.py:is_instruct_model).
# Analysis only, not for the paper. Resumable (same rules as scripts/mbpp_staging_common.sh).
#
#   CUDA_VISIBLE_DEVICES=0 ./run_bos_check_mbpp_chat.sh codellama/CodeLlama-7b-Instruct-hf results/bos_check_2026_11
#   MAX_PROBLEMS=3 ... (smoke) ;  DRY_RUN=1 ... (print commands only)
# Then grade locally (project CLAUDE.md gotcha 9) and compare:
#   uv run python -m bench.eval --results-file OUT_ROOT/{double_bos,single_bos}/<model dir>/greedy_t1.0.jsonl --dataset mbpp
#   uv run python scripts/compare_bos_check.py --out-root OUT_ROOT --model MODEL
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL OUT_ROOT" >&2
  exit 2
fi
lower=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
if [[ "$lower" != *instruct* && "$lower" != *chat* ]]; then
  echo "MODEL must be a chat/instruct model (name contains 'instruct' or 'chat'); got '$1'" >&2
  exit 2
fi
for v in SEED SINGLE_BOS EXTRA_PLESS_TEMPS RESULTS_DIR; do   # this script sets the BOS arm and roots itself (OUT_ROOT)
  if [[ -n "${!v:-}" ]]; then echo "$v must be unset for $0 (got '${!v}')" >&2; exit 2; fi
done

MODEL_ID="$1"
OUT_ROOT="$2"
LOG_TAG="bos_check_${MODEL_ID//\//--}"
PROMPT_ARGS=()
CONFIGS=("greedy_t1.0|--method greedy --temperature 1.0 --n-samples 1")
RESULTS_DIR="${OUT_ROOT}/double_bos"
SINGLE_BOS=""
source scripts/mbpp_staging_common.sh

echo "=== arm 1/2: double_bos (default encoding)"
run_configs
RESULTS_DIR="${OUT_ROOT}/single_bos"
SINGLE_BOS=1
echo ""
echo "=== arm 2/2: single_bos (--single-bos)"
run_configs
