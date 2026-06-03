#!/bin/bash
# CoT token-efficiency experiment on APPS (Qwen3-8B, HF backend, thinking on).
#
# Tests whether the per-token sampler moves the (CoT-length, pass@1) frontier on
# a HARD reasoning benchmark — the thing MBPP was too easy to show. See the plan
# (docs / .claude plan) for the full rationale. Two stages:
#
#   1) calibrate  — size difficulty + token budget cheaply (don't guess).
#                   Runs the standard sampler (temp 0.6 + top_p 0.95) on AtCoder
#                   {introductory, interview}, 12 problems, n=4, 16384 tokens,
#                   then reports completion-rate / median-completed-CoT / pass@1
#                   per difficulty so you can pick the Goldilocks cell + budget.
#
#   2) stageb <difficulty> <max_tokens>
#                 — the 6-config sampler comparison at the chosen difficulty and
#                   budget (k=10, ~30 problems). All UNIFIED (no split), thinking
#                   on, standard HF generation only — nothing handcrafted.
#
# Backend is HF on purpose: pless's Σpᵢ² threshold is numerically sensitive and
# vLLM kernels diverge from HF, which would confound the comparison.
#
# Runs on a CUDA GPU pod (this is not runnable on macOS). Usage:
#   ./run_cot_efficiency_apps_qwen3.sh calibrate
#   ./run_cot_efficiency_apps_qwen3.sh stageb interview 16384
#
# Override via env: MODEL, SOURCE, RESULTS_DIR, CALIB_PROBLEMS, STAGEB_PROBLEMS,
#   CALIB_BUDGET, N_SAMPLES, TOKENIZER, ONLY (run a single Stage B config key).

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_cot_efficiency}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
CALIB_PROBLEMS="${CALIB_PROBLEMS:-12}"
STAGEB_PROBLEMS="${STAGEB_PROBLEMS:-30}"
CALIB_BUDGET="${CALIB_BUDGET:-16384}"
N_SAMPLES="${N_SAMPLES:-10}"

MODEL_DIR="${MODEL//\//--}"   # Qwen/Qwen3-8B -> Qwen--Qwen3-8B

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not found — generation needs CUDA (HF backend)." >&2
fi

# Generate one config (unified, thinking on, HF). Extra args define the sampler.
gen() {
  local difficulty="$1" nsamp="$2" budget="$3" maxprob="$4"; shift 4
  uv run python -m bench.apps \
    --model "$MODEL" --backend hf \
    --source "$SOURCE" --difficulty "$difficulty" \
    --enable-thinking \
    --n-samples "$nsamp" --max-new-tokens "$budget" \
    --max-problems "$maxprob" \
    --results-dir "$RESULTS_DIR" \
    "$@"
}

# Evaluate every JSONL in a result dir (APPS), writing metrics/<stem>_metrics.json.
eval_dir() {
  local dir="$1"
  for f in "$dir"/*.jsonl; do
    [ -e "$f" ] || continue
    case "$f" in *.entropy.*) continue;; esac
    echo "  eval $(basename "$f")"
    uv run python -m bench.eval --results-file "$f" --dataset apps
  done
}

analyze() {
  local dir="$1" budget="$2"
  uv run python -m bench.eval.cot_efficiency \
    --results-dir "$dir" --dataset apps --max-tokens "$budget" \
    --tokenizer "$TOKENIZER"
}

want() { [ -z "${ONLY:-}" ] || [ "$1" = "$ONLY" ]; }

MODE="${1:-}"
case "$MODE" in
  calibrate)
    echo "=== Stage A calibration: $SOURCE {introductory, interview}, "\
"n=4, $CALIB_PROBLEMS problems, budget $CALIB_BUDGET ==="
    for diff in introductory interview; do
      echo "---- calibrate $SOURCE/$diff ----"
      # Standard probe sampler: temp 0.6 + nucleus 0.95 (Qwen default minus top_k).
      gen "$diff" 4 "$CALIB_BUDGET" "$CALIB_PROBLEMS" \
        --method temp --temperature 0.6 --top-p 0.95
      dir="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${diff}"
      eval_dir "$dir"
      echo "---- analysis $SOURCE/$diff ----"
      analyze "$dir" "$CALIB_BUDGET"
    done
    echo
    echo "Pick the difficulty with completion ≳80%, pass@1 in (0,1), largest"
    echo "length spread; then: ./run_cot_efficiency_apps_qwen3.sh stageb <difficulty> <budget>"
    ;;

  stageb)
    DIFFICULTY="${2:?usage: stageb <introductory|interview> <max_tokens>}"
    BUDGET="${3:?usage: stageb <difficulty> <max_tokens>}"
    echo "=== Stage B: $SOURCE/$DIFFICULTY, 6 configs, k=$N_SAMPLES, "\
"$STAGEB_PROBLEMS problems, budget $BUDGET (HF) ==="

    # 1) temp only (0.6, unfiltered)
    want temp        && gen "$DIFFICULTY" "$N_SAMPLES" "$BUDGET" "$STAGEB_PROBLEMS" \
      --method temp --temperature 0.6
    # 2) top_k 20 only (T=1.0)
    want topk        && gen "$DIFFICULTY" "$N_SAMPLES" "$BUDGET" "$STAGEB_PROBLEMS" \
      --method temp --temperature 1.0 --top-k 20
    # 3) top_p 0.95 only (T=1.0)
    want topp        && gen "$DIFFICULTY" "$N_SAMPLES" "$BUDGET" "$STAGEB_PROBLEMS" \
      --method temp --temperature 1.0 --top-p 0.95
    # 4) combined = Qwen recommended (temp 0.6 + top_p 0.95 + top_k 20), unified
    want combined    && gen "$DIFFICULTY" "$N_SAMPLES" "$BUDGET" "$STAGEB_PROBLEMS" \
      --method temp --temperature 0.6 --top-p 0.95 --top-k 20
    # 5) pless (T=1.0, hyperparameter-free as designed)
    want pless       && gen "$DIFFICULTY" "$N_SAMPLES" "$BUDGET" "$STAGEB_PROBLEMS" \
      --method pless --temperature 1.0
    # 6) pless_norm (T=1.0)
    want pless_norm  && gen "$DIFFICULTY" "$N_SAMPLES" "$BUDGET" "$STAGEB_PROBLEMS" \
      --method pless_norm --temperature 1.0

    dir="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
    echo "---- eval ----";    eval_dir "$dir"
    echo "---- analysis ----"; analyze "$dir" "$BUDGET"
    echo
    echo "Results: $dir/analysis/cot_efficiency_apps_report.md"
    ;;

  *)
    echo "usage: $0 calibrate" >&2
    echo "       $0 stageb <introductory|interview> <max_tokens>" >&2
    exit 1
    ;;
esac
