#!/bin/bash
# Claude-Sonnet judge for Qwen2.5-Coder APPS α-sweep NAUADC.
#
# Scope (decided 2026-05-25): 4 buckets × 4 α arms = 16 configs.
# - ATCODER_introductory (5.3× struct_div growth)
# - ATCODER_interview    (4.2×)
# - CODEFORCES_introductory (4.7×)
# - CODEFORCES_interview (4.4×)
# Excludes: ATCODER_competition (0 correct samples), CODEFORCES_competition (2.3× growth, tiny n).
#
# Inputs:  algosim_data/apps_alpha/requests/Qwen--Qwen2.5-Coder-7B-Instruct/<bucket>/<config>.parquet
# Outputs: algosim_data/apps_alpha/responses/Qwen--Qwen2.5-Coder-7B-Instruct/<bucket>/<config>.parquet
#
# Idempotent: skips any (bucket, config) whose response parquet already exists.
#
# Cost estimate (uses Anthropic prompt caching via ephemeral cache_control):
#   ~$230-370 across all 16 configs at Sonnet 4.6 pricing.
#
# Usage:
#   ANTHROPIC_API_KEY=... ./run_algosim_judge_apps_alpha_qwen_coder.sh
#
# Env overrides:
#   WORKERS  parallel problems per config (default 8)
#   MODEL    judge model (default claude-sonnet-4-6)
#   SEED     RNG seed for cluster representative picks (default 42)

set -euo pipefail

MODEL_SLUG="Qwen--Qwen2.5-Coder-7B-Instruct"
REQUESTS_ROOT="algosim_data/apps_alpha/requests/$MODEL_SLUG"
RESPONSES_ROOT="algosim_data/apps_alpha/responses/$MODEL_SLUG"

BUCKETS=(
  "ATCODER_introductory"
  "ATCODER_interview"
  "CODEFORCES_introductory"
  "CODEFORCES_interview"
)
ALPHAS=("2.0" "2.5" "3.0" "5.0")
WORKERS="${WORKERS:-8}"
JUDGE_MODEL="${MODEL:-claude-sonnet-4-6}"
SEED="${SEED:-42}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set" >&2
  exit 1
fi

# Inventory + skip already-done
N_TOTAL=0
N_DONE=0
N_PENDING=0
TODO=()
for B in "${BUCKETS[@]}"; do
  for A in "${ALPHAS[@]}"; do
    CONFIG="pless_alpha_a${A}_t1.0"
    REQ="$REQUESTS_ROOT/$B/$CONFIG.parquet"
    RESP="$RESPONSES_ROOT/$B/$CONFIG.parquet"
    N_TOTAL=$((N_TOTAL + 1))
    if [ ! -f "$REQ" ]; then
      echo "[warn] missing request parquet: $REQ — skip"
      continue
    fi
    if [ -f "$RESP" ]; then
      N_DONE=$((N_DONE + 1))
      continue
    fi
    N_PENDING=$((N_PENDING + 1))
    TODO+=("$B|$CONFIG")
  done
done

echo "═══════════════════════════════════════════════════════════════════════"
echo "Algosim NAUADC judge — Qwen2.5-Coder APPS α-sweep"
echo "  Total configs:    $N_TOTAL"
echo "  Already done:     $N_DONE"
echo "  To run now:       $N_PENDING"
echo "  Judge model:      $JUDGE_MODEL"
echo "  Workers/config:   $WORKERS"
echo "  Seed:             $SEED"
echo "═══════════════════════════════════════════════════════════════════════"

if [ "$N_PENDING" -eq 0 ]; then
  echo "Nothing to do."
  exit 0
fi

I=0
T_START=$(date +%s)
for entry in "${TODO[@]}"; do
  I=$((I + 1))
  BUCKET="${entry%%|*}"
  CONFIG="${entry##*|}"
  REQ_DIR="$REQUESTS_ROOT/$BUCKET"
  RESP_DIR="$RESPONSES_ROOT/$BUCKET"
  mkdir -p "$RESP_DIR"

  T_NOW=$(date +%s); ELAPSED=$((T_NOW - T_START))
  echo
  echo "[$I/$N_PENDING] (elapsed ${ELAPSED}s) $BUCKET / $CONFIG"
  uv run python -m bench.eval.algosim_claude_judge \
    --configs "$CONFIG" \
    --requests-dir "$REQ_DIR" \
    --responses-dir "$RESP_DIR" \
    --model "$JUDGE_MODEL" \
    --workers "$WORKERS" \
    --seed "$SEED" 2>&1 | tail -20
done

T_END=$(date +%s)
echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "Done. Wallclock: $((T_END - T_START))s"
echo "Response parquets:"
find "$RESPONSES_ROOT" -name "*.parquet" | wc -l | xargs -I {} echo "  total: {}"
echo
echo "Next: generate NAUADC report"
echo "  uv run python -m bench.eval.algosim_report_apps \\"
echo "    --responses-dir $RESPONSES_ROOT \\"
echo "    --output-dir results/pless_alpha_apps/Qwen--Qwen2.5-Coder-7B-Instruct/analysis \\"
echo "    --label \"Qwen2.5-Coder-7B-Instruct α-sweep\""
echo "═══════════════════════════════════════════════════════════════════════"
