#!/bin/bash
# Phase A — pass@k evaluation in parallel for all 4 Deepseek cells.
#
# Uses bench.eval's new --skip-diversity flag so we only compute
# pass@k / cover@t (no AST fingerprinting, no self-CodeBLEU). Diversity
# is the slow part of the eval pipeline; skipping it cuts ~10-15% off
# the small cells and ~5-10% off the big cells (the executor dominates
# for big cells either way).
#
# Concurrency model: 4 evals run in background simultaneously, each with
# --workers 4. Total = 16 worker subprocesses. This shares the box with
# the existing APPS eval (PID 71522 on CODEFORCES_interview); subprocess
# work is mostly I/O-wait on the executed test programs, not CPU-bound,
# so 16+8 = 24 logical workers oversubscribes the cores but doesn't
# kill them. If you want to be gentler, override WORKERS_PER_CELL=2.
#
# Usage:
#   ./run_eval_deepseek_phase_a_parallel.sh             # 4-cell parallel eval
#   WORKERS_PER_CELL=2 ./run_eval_deepseek_phase_a_parallel.sh  # be gentler
#   TIMEOUT=10.0 ./run_eval_deepseek_phase_a_parallel.sh        # stricter timeout
#   PHASE_A_DIR=/path/to/dir ./run_eval_deepseek_phase_a_parallel.sh

set -euo pipefail

PHASE_A_DIR="${PHASE_A_DIR:-results/pless_alpha_apps_deepseek_phase_a}"
WORKERS_PER_CELL="${WORKERS_PER_CELL:-4}"
TIMEOUT="${TIMEOUT:-5.0}"
LOG_DIR="${LOG_DIR:-/tmp/deepseek_phase_a_eval_logs}"
mkdir -p "$LOG_DIR"

# Force matplotlib's headless backend in every subprocess we spawn (and every
# subprocess THEY spawn to execute model-generated code). Deepseek-Coder
# routinely hallucinates `import matplotlib.pyplot` in competitive-programming
# samples (~328/65,780 ≈ 0.5% in our Phase A data); on macOS the default
# `macosx` backend opens a GUI window on import, which we observed as flashing
# windows during eval (2026-05-26). Agg backend is in-memory only.
export MPLBACKEND="${MPLBACKEND:-Agg}"

if [ ! -d "$PHASE_A_DIR" ]; then
  echo "ERROR: $PHASE_A_DIR not found." >&2
  exit 2
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "Phase A eval (4 cells in parallel, pass@k only)"
echo "  Source dir:        $PHASE_A_DIR"
echo "  Workers per cell:  $WORKERS_PER_CELL  (total = 4 × $WORKERS_PER_CELL)"
echo "  Timeout per task:  ${TIMEOUT}s"
echo "  Diversity:         SKIPPED (--skip-diversity)"
echo "  Log dir:           $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

declare -a PIDS=()
declare -a CELLS=()

for cell in "$PHASE_A_DIR"/cell*; do
  [ -d "$cell" ] || continue
  cellname=$(basename "$cell")
  jsonl=$(find "$cell" -name "*.jsonl" -type f | head -1)
  if [ -z "$jsonl" ]; then
    echo "  [skip] $cellname: no JSONL found"
    continue
  fi
  log="$LOG_DIR/${cellname}.log"
  echo
  echo "── Launching $cellname (jsonl: $(basename $jsonl)) → $log"
  uv run python -m bench.eval \
    --results-file "$jsonl" \
    --dataset apps \
    --workers "$WORKERS_PER_CELL" \
    --timeout "$TIMEOUT" \
    --skip-diversity \
    > "$log" 2>&1 &
  PIDS+=($!)
  CELLS+=("$cellname")
done

echo
echo "── Launched ${#PIDS[@]} eval jobs (PIDs: ${PIDS[*]}) ──"
echo "── Waiting for all to complete (tail $LOG_DIR/<cell>.log to follow) ──"
echo

# Wait for each, collect status
declare -a FAILED=()
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  cellname="${CELLS[$i]}"
  if wait "$pid"; then
    echo "  ✓ $cellname (PID $pid) exited 0"
  else
    rc=$?
    echo "  ✗ $cellname (PID $pid) FAILED with exit $rc — see $LOG_DIR/${cellname}.log"
    FAILED+=("$cellname")
  fi
done

echo
echo "═══════════════════════════════════════════════════════════════════════"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "DONE — but ${#FAILED[@]} cell(s) failed: ${FAILED[*]}"
  echo "Inspect logs: ls $LOG_DIR"
  exit 1
fi
echo "DONE — all 4 cells evaluated. Headline pass@10:"
for cell in "$PHASE_A_DIR"/cell*; do
  [ -d "$cell" ] || continue
  cellname=$(basename "$cell")
  metrics=$(find "$cell" -name "*_metrics.json" -type f | head -1)
  if [ -n "$metrics" ]; then
    p10=$(uv run python -c "import json; m=json.load(open('$metrics')); print(f\"{m['pass_at_k'].get('10', 'N/A')}\")")
    echo "  $cellname:  pass@10 = $p10"
  fi
done
echo
echo "Paper reference (Deepseek-6.7B-Instruct, CODEFORCES intro, nucleus N=100, Llama judge): 0.1993"
echo "═══════════════════════════════════════════════════════════════════════"
