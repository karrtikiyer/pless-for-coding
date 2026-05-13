#!/bin/bash
# Generate Qwen3-8B samples on a single APPS (source, difficulty) bucket for
# our 6 top-performing configs, in the order they appeared on MBPP.
#
# Usage:
#   ./run_apps_qwen3_top_configs.sh ATCODER competition
#   ./run_apps_qwen3_top_configs.sh CODEFORCES competition
#   ./run_apps_qwen3_top_configs.sh ATCODER introductory
#   …
#
# Override env vars:
#   MODEL              HuggingFace model id (default: Qwen/Qwen3-8B)
#   N_SAMPLES          samples per problem (default: 10)
#   MAX_NEW_TOKENS     token budget incl. thinking (default: 8192)
#   RESULTS_DIR        output root (default: results/pless_apps_results)
#   CONFIGS            comma-separated config keys (default: H7P,H8P,H9P,T15P,T15N,P15)
#   MAX_PROBLEMS       cap (for smoke testing); empty = full bucket
#   ONLY               run a single config; e.g. ONLY=T15P
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <ATCODER|CODEFORCES> <introductory|interview|competition>" >&2
  exit 1
fi
SOURCE="$1"
DIFFICULTY="$2"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_apps_results}"
CONFIGS="${CONFIGS:-H7P,H8P,H9P,T15P,T15N,P15}"
MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

run_one() {
  local cfg="$1"; shift
  echo
  echo "──── $cfg on $SOURCE/$DIFFICULTY ────"
  uv run python -m bench.apps \
    --model "$MODEL" \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --results-dir "$RESULTS_DIR" \
    $MAX_PROBLEMS_FLAG \
    "$@"
}

# Tokenize the comma-separated list once
declare -A WANT
IFS=',' read -ra CFGLIST <<< "$CONFIGS"
for c in "${CFGLIST[@]}"; do WANT[$c]=1; done
filter() { [ -n "${ONLY:-}" ] && [ "$1" != "$ONLY" ] && return 1; [ -n "${WANT[$1]:-}" ]; }

# ─── H7P: temp_pure 1.5 (think) → pless 1.0 (code) ─────────────────────────
if filter H7P; then
  run_one H7P --method split --enable-thinking --temperature 1.0 \
    --sampler-think temp_pure --temp-think 1.5 \
    --sampler-code  pless     --temp-code  1.0
fi

# ─── H8P: temp_pure 1.5 → pless 1.5 ─────────────────────────────────────────
if filter H8P; then
  run_one H8P --method split --enable-thinking --temperature 1.0 \
    --sampler-think temp_pure --temp-think 1.5 \
    --sampler-code  pless     --temp-code  1.5
fi

# ─── H9P: temp_pure 1.5 → pless 2.0 ─────────────────────────────────────────
if filter H9P; then
  run_one H9P --method split --enable-thinking --temperature 1.0 \
    --sampler-think temp_pure --temp-think 1.5 \
    --sampler-code  pless     --temp-code  2.0
fi

# ─── T15P: temp_pure 1.5 → temp_pure 1.5 (pure split baseline) ──────────────
if filter T15P; then
  run_one T15P --method split --enable-thinking --temperature 1.0 \
    --sampler-think temp_pure --temp-think 1.5 \
    --sampler-code  temp_pure --temp-code  1.5
fi

# ─── T15N: native HF temp 1.5 throughout, no split ──────────────────────────
if filter T15N; then
  run_one T15N --method temp --enable-thinking --temperature 1.5
fi

# ─── P15: uniform pless 1.5 throughout, no split ───────────────────────────
if filter P15; then
  run_one P15 --method pless --enable-thinking --temperature 1.5
fi

echo
echo "All requested configs done for $SOURCE/$DIFFICULTY."
