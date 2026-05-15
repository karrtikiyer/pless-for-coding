#!/bin/bash
# vLLM-backend variant of run_apps_qwen3_top_configs.sh.
#
# Same 6-config sweep, same APPS bucket, same model — but generation
# is routed through bench/generator_vllm.py via --backend vllm. Uses
# the parallel .venv-vllm environment (not the main .venv).
#
# Usage:
#   ./run_apps_qwen3_top_configs_vllm.sh ATCODER competition
#   ./run_apps_qwen3_top_configs_vllm.sh CODEFORCES competition
#
# Override env vars (same as HF version, plus VLLM_VENV):
#   VLLM_VENV          path to vLLM venv root (default: .venv-vllm)
#   MODEL              HF model id (default: Qwen/Qwen3-8B)
#   N_SAMPLES          samples per problem (default: 10)
#   MAX_NEW_TOKENS     token budget incl. thinking (default: 8192)
#   RESULTS_DIR        output root (default: results/pless_apps_results_vllm)
#   CONFIGS            comma-separated config keys (default: H7P,H8P,H9P,T15P,T15N,P15)
#   MAX_PROBLEMS       cap for smoke testing
#   ONLY               run a single config
#
# Why a separate RESULTS_DIR default: each JSONL record already carries
# a "backend" field, but putting vLLM outputs under a parallel directory
# tree makes A/B comparisons more obvious and keeps the existing HF
# results untouched. Set RESULTS_DIR=results/pless_apps_results if you
# want them merged (they will not collide — different "backend" field).

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <ATCODER|CODEFORCES> <introductory|interview|competition>" >&2
  exit 1
fi
SOURCE="$1"
DIFFICULTY="$2"

VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_apps_results_vllm}"
CONFIGS="${CONFIGS:-H7P,H8P,H9P,T15P,T15N,P15}"
MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────
if [ ! -d "$VLLM_VENV" ]; then
  echo "Error: vLLM venv not found at $VLLM_VENV." >&2
  echo "Bootstrap it once with:" >&2
  echo "  uv venv $VLLM_VENV --python 3.12" >&2
  echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2
  exit 2
fi

if ! "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null; then
  echo "Error: vllm import failed inside $VLLM_VENV." >&2
  echo "Re-sync the venv from pyproject-vllm.toml and try again." >&2
  exit 3
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not found — vLLM requires CUDA." >&2
fi

run_one() {
  local cfg="$1"; shift
  echo
  echo "──── $cfg on $SOURCE/$DIFFICULTY (backend=vllm) ────"
  # PYTHONPATH="$PWD" so the source tree wins over any installed copy.
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" \
    --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --results-dir "$RESULTS_DIR" \
    $MAX_PROBLEMS_FLAG \
    "$@"
}

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

# ─── T15N: native temp 1.5 throughout, no split ─────────────────────────────
if filter T15N; then
  run_one T15N --method temp --enable-thinking --temperature 1.5
fi

# ─── P15: uniform pless 1.5 throughout, no split ───────────────────────────
if filter P15; then
  run_one P15 --method pless --enable-thinking --temperature 1.5
fi

echo
echo "All requested configs done for $SOURCE/$DIFFICULTY (backend=vllm)."
