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
#   GPUS=0,1,2 ./run_cot_efficiency_apps_qwen3.sh stageb interview 16384
#
# GPUS (Stage B only): comma-separated GPU ids. When set, the 6 configs are
#   distributed across those GPUs (one pinned process per GPU, configs
#   round-robin'd in slow-first order so pless/pless_norm land on separate
#   GPUs), run in parallel, then eval+analyze once all finish. Unset => serial
#   on the default device. Best for the ~1,800-gen Stage B; Stage A is too
#   small to bother. Each process loads its own model copy (~16 GB) on its GPU.
#
# Override via env: MODEL, SOURCE, RESULTS_DIR, CALIB_PROBLEMS, STAGEB_PROBLEMS,
#   CALIB_BUDGET, N_SAMPLES, TOKENIZER, GPUS, ONLY (single config key:
#   temp|topk|topp|combined|pless|pless_norm).

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

# Stage B config keys. Order matters for GPU assignment: the two SLOW
# (token-by-token) configs pless/pless_norm come first, so round-robin
# placement (config i -> GPU[i % nGPU]) puts them on different GPUs and
# spreads the load (one slow config per GPU for nGPU>=2).
STAGEB_ORDER=(pless pless_norm temp topk topp combined)

run_config() {
  local name="$1" difficulty="$2" budget="$3"
  case "$name" in
    temp)       gen "$difficulty" "$N_SAMPLES" "$budget" "$STAGEB_PROBLEMS" \
                  --method temp --temperature 0.6 ;;
    topk)       gen "$difficulty" "$N_SAMPLES" "$budget" "$STAGEB_PROBLEMS" \
                  --method temp --temperature 1.0 --top-k 20 ;;
    topp)       gen "$difficulty" "$N_SAMPLES" "$budget" "$STAGEB_PROBLEMS" \
                  --method temp --temperature 1.0 --top-p 0.95 ;;
    combined)   gen "$difficulty" "$N_SAMPLES" "$budget" "$STAGEB_PROBLEMS" \
                  --method temp --temperature 0.6 --top-p 0.95 --top-k 20 ;;
    pless)      gen "$difficulty" "$N_SAMPLES" "$budget" "$STAGEB_PROBLEMS" \
                  --method pless --temperature 1.0 ;;
    pless_norm) gen "$difficulty" "$N_SAMPLES" "$budget" "$STAGEB_PROBLEMS" \
                  --method pless_norm --temperature 1.0 ;;
    *) echo "unknown config '$name'" >&2; return 1 ;;
  esac
}

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

    dir="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
    mkdir -p "$dir"

    # Configs to run (honor ONLY=<key>), in slow-first order.
    selected=()
    for name in "${STAGEB_ORDER[@]}"; do want "$name" && selected+=("$name"); done

    if [ -n "${GPUS:-}" ]; then
      # ── Parallel: one process per GPU, configs round-robin'd across GPUS,
      #    each GPU runs its group sequentially. Different configs => different
      #    output files, so no append collision. Per-GPU logs avoid garbled
      #    interleaving; we wait for all and fail if any worker fails.
      IFS=',' read -ra GPULIST <<< "$GPUS"
      ngpu=${#GPULIST[@]}
      echo "Parallel Stage B: ${#selected[@]} configs across $ngpu GPU(s) [$GPUS]"
      pids=(); gpus_used=()
      for ((g=0; g<ngpu; g++)); do
        group=()
        for ((i=0; i<${#selected[@]}; i++)); do
          (( i % ngpu == g )) && group+=("${selected[$i]}")
        done
        [ ${#group[@]} -eq 0 ] && continue
        gpu="${GPULIST[$g]}"
        log="$dir/stageb_gpu${gpu}.log"
        echo "  GPU $gpu -> ${group[*]}  (log: $log)"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          for name in "${group[@]}"; do
            echo ">>> [GPU $gpu] config $name"
            run_config "$name" "$DIFFICULTY" "$BUDGET"
          done
        ) >"$log" 2>&1 &
        pids+=($!); gpus_used+=("$gpu")
      done
      fail=0
      for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
          echo "ERROR: GPU ${gpus_used[$idx]} worker failed — see "\
"$dir/stageb_gpu${gpus_used[$idx]}.log" >&2
          fail=1
        fi
      done
      [ $fail -ne 0 ] && exit 4
      echo "All GPU workers finished."
    else
      # Sequential (single GPU / default device).
      for name in "${selected[@]}"; do
        echo "---- config $name ----"
        run_config "$name" "$DIFFICULTY" "$BUDGET"
      done
    fi

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
