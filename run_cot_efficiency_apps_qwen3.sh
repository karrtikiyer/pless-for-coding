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
# Backend defaults to HF; set BACKEND=vllm for ~10-20x throughput. vLLM's logits
# diverge slightly from HF (matmul accumulation, per the vLLM forum thread), but
# our vLLM pless threshold is computed in fp32 and the per-token prob error is
# ~1e-4 median — so run the `delta` mode once to confirm pless is backend-
# invariant before trusting vLLM for the full study.
#
# Runs on a CUDA GPU pod (this is not runnable on macOS). Usage:
#   ./run_cot_efficiency_apps_qwen3.sh calibrate
#   ./run_cot_efficiency_apps_qwen3.sh stageb interview 16384
#   ./run_cot_efficiency_apps_qwen3.sh delta  interview 16384     # HF vs vLLM check
#   BACKEND=vllm ./run_cot_efficiency_apps_qwen3.sh calibrate
#   GPUS=0,1,2 BACKEND=vllm ./run_cot_efficiency_apps_qwen3.sh stageb interview 16384
#
# GPUS (Stage B only): comma-separated GPU ids. When set, the 6 configs are
#   distributed across those GPUs (one pinned process per GPU, configs
#   round-robin'd in slow-first order so pless/pless_norm land on separate
#   GPUs), run in parallel, then eval+analyze once all finish. Unset => serial
#   on the default device. Best for the ~1,800-gen Stage B; Stage A is too
#   small to bother. Each process loads its own model copy (~16 GB) on its GPU.
#
# Override via env: MODEL, SOURCE, RESULTS_DIR, CALIB_PROBLEMS, STAGEB_PROBLEMS,
#   CALIB_BUDGET, N_SAMPLES, TOKENIZER, GPUS, BACKEND (hf|vllm), VLLM_VENV,
#   DELTA_PROBLEMS, DELTA_SAMPLES, ONLY (single config key:
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
BACKEND="${BACKEND:-hf}"          # hf | vllm
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"

MODEL_DIR="${MODEL//\//--}"   # Qwen/Qwen3-8B -> Qwen--Qwen3-8B

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not found — generation needs CUDA." >&2
fi

# vLLM preflight (only when generating with --backend vllm).
check_vllm() {
  if [ ! -d "$VLLM_VENV" ]; then
    echo "Error: BACKEND=vllm but venv not found at $VLLM_VENV. Bootstrap once:" >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2
    exit 2
  fi
  "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null || {
    echo "Error: 'import vllm' failed inside $VLLM_VENV — re-sync pyproject-vllm.toml." >&2
    exit 3; }
}

# Generate one config (unified, thinking on). Extra args define the sampler.
# Routes through HF (uv env) or vLLM (.venv-vllm) per $BACKEND. Eval/analysis
# always stay in the main uv env.
gen() {
  local difficulty="$1" nsamp="$2" budget="$3" maxprob="$4"; shift 4
  if [ "$BACKEND" = "vllm" ]; then
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    "$VLLM_VENV/bin/python" -m bench.apps \
      --model "$MODEL" --backend vllm \
      --source "$SOURCE" --difficulty "$difficulty" \
      --enable-thinking \
      --n-samples "$nsamp" --max-new-tokens "$budget" \
      --max-problems "$maxprob" \
      --results-dir "$RESULTS_DIR" \
      "$@"
  else
    uv run python -m bench.apps \
      --model "$MODEL" --backend hf \
      --source "$SOURCE" --difficulty "$difficulty" \
      --enable-thinking \
      --n-samples "$nsamp" --max-new-tokens "$budget" \
      --max-problems "$maxprob" \
      --results-dir "$RESULTS_DIR" \
      "$@"
  fi
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
[ "$BACKEND" = "vllm" ] && [ "$MODE" != "delta" ] && check_vllm
case "$MODE" in
  calibrate)
    echo "=== Stage A calibration: $SOURCE {introductory, interview}, "\
"n=4, $CALIB_PROBLEMS problems, budget $CALIB_BUDGET, backend=$BACKEND ==="
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
"$STAGEB_PROBLEMS problems, budget $BUDGET, backend=$BACKEND ==="

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

  delta)
    # One-command HF-vs-vLLM backend-delta check: run pless (the worry) + top_p
    # (standard ref) on the SAME problems on BOTH backends, then print a
    # side-by-side of pass@1 / median completed think-tokens / completion-rate.
    # If pless's HF↔vLLM deltas are within noise, vLLM is safe for the study.
    DIFFICULTY="${2:?usage: delta <introductory|interview> <max_tokens>}"
    BUDGET="${3:?usage: delta <difficulty> <max_tokens>}"
    DELTA_PROBLEMS="${DELTA_PROBLEMS:-8}"
    DELTA_SAMPLES="${DELTA_SAMPLES:-4}"
    check_vllm
    echo "=== Backend delta: $SOURCE/$DIFFICULTY, pless + top_p, "\
"$DELTA_PROBLEMS problems x $DELTA_SAMPLES, budget $BUDGET, HF vs vLLM ==="

    declare -A DELTA_DIR
    for bk in hf vllm; do
      BACKEND="$bk"
      RESULTS_DIR="${RESULTS_DIR%/}/delta_$bk"   # separate trees (filenames omit backend)
      d="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
      DELTA_DIR[$bk]="$d"
      echo "---- generate ($bk) ----"
      gen "$DIFFICULTY" "$DELTA_SAMPLES" "$BUDGET" "$DELTA_PROBLEMS" \
        --method pless --temperature 1.0
      gen "$DIFFICULTY" "$DELTA_SAMPLES" "$BUDGET" "$DELTA_PROBLEMS" \
        --method temp --temperature 1.0 --top-p 0.95
      echo "---- eval ($bk) ----";    eval_dir "$d"
      echo "---- analysis ($bk) ----"; analyze "$d" "$BUDGET"
      RESULTS_DIR="${RESULTS_DIR%/delta_$bk}"     # restore for next iter
    done

    echo
    echo "=== HF vs vLLM comparison ==="
    uv run python - "${DELTA_DIR[hf]}/analysis/cot_efficiency_apps.csv" \
                    "${DELTA_DIR[vllm]}/analysis/cot_efficiency_apps.csv" <<'PY'
import csv, sys
def load(p):
    out={}
    for r in csv.DictReader(open(p)):
        key=(r["method"], r.get("top_p",""), r.get("top_k",""))
        out[key]=r
    return out
hf, vl = load(sys.argv[1]), load(sys.argv[2])
def f(r,k):
    v=r.get(k,"")
    try: return float(v)
    except: return None
cols=[("pass@1","pass@1"),("median completed think tok","median_think_tokens_completed"),
      ("completion_rate","completion_rate")]
def cell(x):
    if x is None: return "—"
    return f"{x:.4f}" if isinstance(x, float) else str(x)
print(f"{'config':32} {'metric':28} {'HF':>10} {'vLLM':>10} {'Δ':>10}")
print("-"*94)
for key in sorted(set(hf)&set(vl)):
    label=hf[key].get("label") or hf[key]["method"]
    for name,col in cols:
        a,b=f(hf[key],col),f(vl[key],col)
        d=(b-a) if (a is not None and b is not None) else None
        print(f"{label[:32]:32} {name:28} {cell(a):>10} {cell(b):>10} {cell(d):>10}")
    cr_hf, cr_vl = f(hf[key],"completion_rate"), f(vl[key],"completion_rate")
    if (cr_hf is not None and cr_hf < 0.5) or (cr_vl is not None and cr_vl < 0.5):
        print(f"  ⚠ low completion (HF {cell(cr_hf)} / vLLM {cell(cr_vl)}) — "
              f"truncation-dominated; raise --max-tokens before trusting this row.")
    print()
print("Rule of thumb: pless pass@1 |Δ| within ~1-2 sampling SEs and median-token")
print("Δ within ~a few % => backend-invariant => vLLM safe for the full study.")
PY
    ;;

  *)
    echo "usage: $0 calibrate" >&2
    echo "       $0 stageb <introductory|interview> <max_tokens>" >&2
    echo "       $0 delta  <introductory|interview> <max_tokens>   (HF-vs-vLLM check)" >&2
    echo "  env: BACKEND=hf|vllm  GPUS=0,1,2  ONLY=<config>  MODEL=...  SOURCE=..." >&2
    exit 1
    ;;
esac
