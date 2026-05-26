#!/bin/bash
# Phase A NAUADC — Deepseek-6.7B-Instruct CODEFORCES_introductory
#
# Scope (chosen 2026-05-26 to avoid statistical floor at N=10):
#   cell1_nucleus_n100  (config key: temp_t1.0)
#   cell4_alpha5_n100   (config key: pless_alpha_a5.0_t1.0)
# Both N=100 — enough correct samples per problem for clustering to
# actually find clusters (~3-4 correct/problem at our pass rate).
#
# Pipeline per cell:
#   1. Export: bench.eval.algosim_export_apps (uses extract_python_code_apps
#      with prefix/window-strip — gets the recovered samples too).
#   2. Judge: bench.eval.algosim_claude_judge (Claude-Sonnet on parquets).
#   3. (After both cells judged) Combined report via algosim_report_apps.
#
# Parquets land in algosim_data/apps_deepseek_phase_a/.../CODEFORCES_introductory/
# so algosim_report_apps's rglob discovers both cells in one shot.
#
# Cost estimate (Anthropic prompt caching active):
#   2 cells × ~3-8 USD each = ~6-16 USD total
#   Wallclock: ~1-3 hr (judges run in parallel via background &)
#
# Usage:
#   ANTHROPIC_API_KEY=... ./run_naudc_deepseek_phase_a.sh
#
# Env overrides:
#   WORKERS  parallel problems per judge (default 8)
#   MODEL    judge model (default claude-sonnet-4-6)
#   SEED     RNG seed for cluster representative picks (default 42)
#   SCOPE    "n100" (default — cells 1+4) or "all" (cells 1-4)

set -euo pipefail

PHASE_A_DIR="${PHASE_A_DIR:-results/pless_alpha_apps_deepseek_phase_a}"
MODEL_HF_SLUG="deepseek-ai--deepseek-coder-6.7b-instruct"
SOURCE="CODEFORCES"
DIFFICULTY="introductory"
REQ_ROOT="${REQ_ROOT:-algosim_data/apps_deepseek_phase_a/requests}"
RESP_ROOT="${RESP_ROOT:-algosim_data/apps_deepseek_phase_a/responses}"
REPORT_DIR="${REPORT_DIR:-$PHASE_A_DIR/_analysis}"
WORKERS="${WORKERS:-8}"
JUDGE_MODEL="${MODEL:-claude-sonnet-4-6}"
SEED="${SEED:-42}"
SCOPE="${SCOPE:-n100}"

# Cell → (subdir name in $PHASE_A_DIR, JSONL stem / config key)
if [ "$SCOPE" = "all" ]; then
  CELLS=(
    "cell1_nucleus_n100:temp_t1.0"
    "cell2_nucleus_n10:temp_t1.0"
    "cell3_alpha5_n10:pless_alpha_a5.0_t1.0"
    "cell4_alpha5_n100:pless_alpha_a5.0_t1.0"
  )
else
  CELLS=(
    "cell1_nucleus_n100:temp_t1.0"
    "cell4_alpha5_n100:pless_alpha_a5.0_t1.0"
  )
fi

mkdir -p "$REQ_ROOT" "$RESP_ROOT" "$REPORT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "Phase A NAUADC — Deepseek-6.7B-Instruct CODEFORCES_introductory"
echo "  Scope:           $SCOPE  (${#CELLS[@]} cells)"
echo "  Judge model:     $JUDGE_MODEL"
echo "  Workers/judge:   $WORKERS"
echo "  Request parquet: $REQ_ROOT/<cell>/$SOURCE_$DIFFICULTY/"
echo "  Response parquet: $RESP_ROOT/$SOURCE_$DIFFICULTY/<config>.parquet"
echo "  Report:          $REPORT_DIR/"
echo "═══════════════════════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────────
# 1. Export each cell's JSONL → parquet
# ────────────────────────────────────────────────────────────────────────
for entry in "${CELLS[@]}"; do
  CELL="${entry%%:*}"
  CONFIG="${entry##*:}"
  JSONL_DIR="$PHASE_A_DIR/$CELL/$MODEL_HF_SLUG"
  CELL_REQ_DIR="$REQ_ROOT/$CELL"

  echo
  echo "── EXPORT $CELL (config=$CONFIG) ──"
  if [ ! -f "$JSONL_DIR/${SOURCE}_${DIFFICULTY}/${CONFIG}.jsonl" ]; then
    echo "  [warn] expected $JSONL_DIR/${SOURCE}_${DIFFICULTY}/${CONFIG}.jsonl — not found, skip"
    continue
  fi
  uv run python -m bench.eval.algosim_export_apps \
    --results-dir "$JSONL_DIR" \
    --source "$SOURCE" \
    --difficulty "$DIFFICULTY" \
    --configs "$CONFIG" \
    --output-dir "$CELL_REQ_DIR"
done

# Move each cell's parquet into a unified responses-input layout
# so algosim_report_apps's rglob picks them up together.
# Source: $CELL_REQ_DIR/$SOURCE_$DIFFICULTY/$CONFIG.parquet
# (algosim_export_apps writes to $output_dir/$source_$difficulty/)

# ────────────────────────────────────────────────────────────────────────
# 2. Run judges in parallel (one bg process per cell)
# ────────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-/tmp/deepseek_phase_a_naudc_logs}"
mkdir -p "$LOG_DIR"

declare -a PIDS=()
declare -a CELL_NAMES=()
RESP_SUBDIR="$RESP_ROOT/${SOURCE}_${DIFFICULTY}"
mkdir -p "$RESP_SUBDIR"

for entry in "${CELLS[@]}"; do
  CELL="${entry%%:*}"
  CONFIG="${entry##*:}"
  REQ="$REQ_ROOT/$CELL/${SOURCE}_${DIFFICULTY}/${CONFIG}.parquet"
  RESP="$RESP_SUBDIR/${CONFIG}.parquet"
  LOG="$LOG_DIR/${CELL}.log"

  if [ ! -f "$REQ" ]; then
    echo "  [warn] $CELL: no request parquet at $REQ — skip"
    continue
  fi
  if [ -f "$RESP" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  [skip] $CELL: response parquet exists at $RESP (FORCE=1 to re-run)"
    continue
  fi

  echo
  echo "── JUDGE $CELL  (req=$REQ → resp=$RESP, log=$LOG)"
  uv run python -m bench.eval.algosim_claude_judge \
    --configs "$CONFIG" \
    --requests-dir "$REQ_ROOT/$CELL/${SOURCE}_${DIFFICULTY}" \
    --responses-dir "$RESP_SUBDIR" \
    --model "$JUDGE_MODEL" \
    --workers "$WORKERS" \
    --seed "$SEED" > "$LOG" 2>&1 &
  PIDS+=($!)
  CELL_NAMES+=("$CELL")
done

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "Nothing to judge (all cells already done). Skipping to report."
else
  echo
  echo "── Launched ${#PIDS[@]} judges (PIDs: ${PIDS[*]}). Waiting... ──"
  declare -a FAILED=()
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    cell="${CELL_NAMES[$i]}"
    if wait "$pid"; then
      echo "  ✓ $cell (PID $pid) exited 0"
    else
      rc=$?
      echo "  ✗ $cell (PID $pid) FAILED with exit $rc — see $LOG_DIR/${cell}.log"
      FAILED+=("$cell")
    fi
  done
  if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Some judges failed: ${FAILED[*]}"
    exit 1
  fi
fi

# ────────────────────────────────────────────────────────────────────────
# 3. Combined NAUADC report (both cells in one table)
# ────────────────────────────────────────────────────────────────────────
echo
echo "── REPORT (combined Phase A NAUADC) ──"
uv run python -m bench.eval.algosim_report_apps \
  --responses-dir "$RESP_ROOT" \
  --output-dir "$REPORT_DIR" \
  --label "Deepseek-6.7B-Instruct Phase A ($SCOPE)"

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "DONE. Outputs:"
echo "  Per-cell response parquets: $RESP_SUBDIR/"
echo "  NAUADC report markdown:     $REPORT_DIR/algosim_apps_report.md"
echo "  NAUADC bar chart:           $REPORT_DIR/algosim_apps_bar.png"
echo "  Per-cell metrics JSON:      $REPORT_DIR/algosim_apps_per_cell.json"
echo
echo "Paper reference for CODEFORCES_intro Deepseek-6.7B-Instruct (Llama judge): 1.643"
echo "Judge-asymmetry adjustment (Claude / Llama ratio ≈ 0.898 from prior Qwen3 study):"
echo "  → expected paper-equivalent Claude NAUADC ≈ 1.476"
echo "═══════════════════════════════════════════════════════════════════════"
