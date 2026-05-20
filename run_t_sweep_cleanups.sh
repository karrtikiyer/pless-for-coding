#!/bin/bash
# T-envelope cleanup runs to close two gaps before paper drafting.
#
# Closes:
#   C1. m-a-p OCI-1.3B HumanEval T-sweep — mirrors the 16-config set under
#       results/pless_human_eval_results/temprature_results/codellama--CodeLlama-7b-Instruct-hf/humaneval/
#       (pless × 7 T's, pless_norm × 7 T's, temp × 2 T's).
#   C2. pless@T=2.5 and pless@T=3.0 on CodeLlama-7B-Instruct MBPP.
#   C3. pless@T=2.5 and pless@T=3.0 on m-a-p OCI-1.3B MBPP.
#
# These are temperature-only baselines (NOT Rényi-α). They close the high-T
# cliff documented in results/pless_alpha_full/t_envelope_analysis.md.
#
# Sequential execution on a single GPU — small models + low generation
# volume — total wall-clock ~3–4 GPU-hours.
#
# Usage:
#   ./run_t_sweep_cleanups.sh                # run everything
#   SKIP_C1=1 ./run_t_sweep_cleanups.sh      # skip m-a-p HE T-sweep
#   SKIP_C2=1 SKIP_C3=1 ./run_t_sweep_cleanups.sh   # only m-a-p HE
#
# Env overrides:
#   GPU              CUDA_VISIBLE_DEVICES (default: 0)
#   N_SAMPLES        (default 10)
#   MAX_NEW_TOKENS   (default 512)
#   MAX_PROBLEMS     smoke cap (default: unset)
#   LOG_DIR          (default /tmp/t_sweep_cleanups_logs)

set -euo pipefail

GPU="${GPU:-0}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOG_DIR="${LOG_DIR:-/tmp/t_sweep_cleanups_logs}"

MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

mkdir -p "$LOG_DIR"

MAP_MODEL="m-a-p/OpenCodeInterpreter-DS-1.3B"
CL_MODEL="codellama/CodeLlama-7b-Instruct-hf"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  T-envelope cleanup runs"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  GPU:           $GPU"
echo "  N samples:     $N_SAMPLES"
echo "  Max tokens:    $MAX_NEW_TOKENS"
echo "  Logs:          $LOG_DIR"
[ -n "${MAX_PROBLEMS:-}" ] && echo "  Max problems:  $MAX_PROBLEMS (smoke)"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# ── C1: m-a-p HumanEval T-sweep ─────────────────────────────────────────────
if [ -z "${SKIP_C1:-}" ]; then
  echo "─── C1: m-a-p OCI-1.3B HumanEval T-sweep ──────────────────────────"
  HE_RESULTS_DIR="results/pless_human_eval_results/temprature_results"
  PLESS_TEMPS="0.6 0.7 1.0 1.5 2.0 2.5 3.0"
  TEMP_TEMPS="0.7 1.0"

  for method in pless pless_norm; do
    for T in $PLESS_TEMPS; do
      log="$LOG_DIR/map_he_${method}_t${T}.log"
      echo "[GPU $GPU] m-a-p HE $method @ T=$T → $log"
      CUDA_VISIBLE_DEVICES="$GPU" uv run python -m bench.humaneval \
        --model "$MAP_MODEL" \
        --method "$method" \
        --temperature "$T" \
        --n-samples "$N_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --backend hf \
        --results-dir "$HE_RESULTS_DIR" \
        $MAX_PROBLEMS_FLAG \
        > "$log" 2>&1
    done
  done

  for T in $TEMP_TEMPS; do
    log="$LOG_DIR/map_he_temp_t${T}.log"
    echo "[GPU $GPU] m-a-p HE temp @ T=$T → $log"
    CUDA_VISIBLE_DEVICES="$GPU" uv run python -m bench.humaneval \
      --model "$MAP_MODEL" \
      --method temp \
      --temperature "$T" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend hf \
      --results-dir "$HE_RESULTS_DIR" \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  done
  echo "─── C1 done at $(date) ───"
  echo
fi

# ── C2: CodeLlama MBPP pless@T=2.5/T=3.0 ────────────────────────────────────
if [ -z "${SKIP_C2:-}" ]; then
  echo "─── C2: CodeLlama MBPP pless@T=2.5/3.0 ────────────────────────────"
  MBPP_RESULTS_DIR="results/pless_full_mbpp_results"
  for T in 2.5 3.0; do
    log="$LOG_DIR/codellama_mbpp_pless_t${T}.log"
    echo "[GPU $GPU] CodeLlama MBPP pless @ T=$T → $log"
    CUDA_VISIBLE_DEVICES="$GPU" uv run python -m bench \
      --model "$CL_MODEL" \
      --method pless \
      --temperature "$T" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend hf \
      --mbpp-config full \
      --results-dir "$MBPP_RESULTS_DIR" \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  done
  echo "─── C2 done at $(date) ───"
  echo
fi

# ── C3: m-a-p MBPP pless@T=2.5/T=3.0 ────────────────────────────────────────
if [ -z "${SKIP_C3:-}" ]; then
  echo "─── C3: m-a-p MBPP pless@T=2.5/3.0 ────────────────────────────────"
  MBPP_RESULTS_DIR="results/pless_full_mbpp_results"
  for T in 2.5 3.0; do
    log="$LOG_DIR/map_mbpp_pless_t${T}.log"
    echo "[GPU $GPU] m-a-p MBPP pless @ T=$T → $log"
    CUDA_VISIBLE_DEVICES="$GPU" uv run python -m bench \
      --model "$MAP_MODEL" \
      --method pless \
      --temperature "$T" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend hf \
      --mbpp-config full \
      --results-dir "$MBPP_RESULTS_DIR" \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  done
  echo "─── C3 done at $(date) ───"
  echo
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "  T-envelope cleanups COMPLETE at $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo
echo "Next steps (eval the new JSONLs):"
echo "  # C1 m-a-p HE:"
echo "  for f in results/pless_human_eval_results/temprature_results/$MAP_MODEL/humaneval/*.jsonl; do"
echo "    uv run python -m bench.eval --dataset humaneval --results-file \"\$f\""
echo "  done"
echo "  # C2+C3 MBPP:"
echo "  for f in results/pless_full_mbpp_results/{codellama--CodeLlama-7b-Instruct-hf,m-a-p--OpenCodeInterpreter-DS-1.3B}/pless_t{2.5,3.0}.jsonl; do"
echo "    uv run python -m bench.eval --dataset mbpp --results-file \"\$f\""
echo "  done"
