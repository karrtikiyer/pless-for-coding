#!/bin/bash
# Full-252 validation of the recovery-sweep winners (Qwen3-8B, ATCODER-interview, vLLM).
#
# The 27-task recovery sweep showed pless T1.5/T2.0 + pless_alpha a3/a4/a5 cut the
# think-phase truncation from ~17% to ~0 and lift pass@1 on the tasks pless TRUNCATED.
# But those 27 were SELECTED as pless failures — this run checks the variants don't
# REGRESS the other ~225 problems baseline pless already handled, by running the full
# 252-problem set and comparing per-problem against the existing baseline.
#
# Baselines ALREADY EXIST on the full 252 (no regen needed):
#   results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/
#   (pless T1.0, pless_norm, + 4 temp configs). pless T1.0 pass@1=0.625, pass@10=0.825.
#
# This script generates ONLY the 4 winner arms on all 252, evals, then runs the paired
# regression analysis vs the baseline.
#
# Cost (grounded in the 27-run: ~1.8 hr/arm for the 27 hardest): ~5-10 hr/arm on the full
# 252; 4 arms across 2 GPUs (2/GPU) ≈ 10-20 hr wall-clock at the 32768 cap. Levers:
#   ARMS="pless_a5 pless_t2.0"  → just the 2 winners (~half).
#   MAX_TOKENS=16384            → ~half (clips almost nothing for these low-trunc arms,
#                                 but not bit-comparable to the 32k baseline).
#
# RUN ON A CUDA POD. Requires .venv-vllm (pyproject-vllm.toml).
# Usage:
#   GPUS=0,1 ./run_pless_recovery_full252_apps_qwen3.sh
#   ARMS="pless_a5 pless_t2.0" GPUS=0,1 ./run_pless_recovery_full252_apps_qwen3.sh
# Env: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES(10), MAX_TOKENS(32768),
#   VLLM_VENV, WORKERS(32), TOKENIZER, GPUS, ARMS, BASELINE_DIR.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_recovery_full252}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL//\//--}"
BASELINE_DIR="${BASELINE_DIR:-results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252}"

# vLLM env hygiene (see run_pless_recovery_sweep_apps_qwen3.sh).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"

check_vllm() {
  [ -x "$VLLM_VENV/bin/python" ] || { echo "Error: vLLM venv $VLLM_VENV missing." >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2; exit 2; }
  "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null || { echo "Error: import vllm failed." >&2; exit 3; }
}

arm_args() {
  case "$1" in
    pless_t1.5) echo "--method pless       --temperature 1.5" ;;
    pless_t2.0) echo "--method pless       --temperature 2.0" ;;
    pless_a3)   echo "--method pless_alpha --alpha 3 --temperature 1.0" ;;
    pless_a4)   echo "--method pless_alpha --alpha 4 --temperature 1.0" ;;
    pless_a5)   echo "--method pless_alpha --alpha 5 --temperature 1.0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}

# arm -> the JSONL basename the runner writes (must match the 27-sweep names so the
# pre-seed + resume-skip merge is valid).
arm_jsonl() {
  case "$1" in
    pless_t1.5) echo "pless_think_t1.5_t1.5.jsonl" ;;
    pless_t2.0) echo "pless_think_t2.0_t2.0.jsonl" ;;
    pless_a3)   echo "pless_alpha_think_t1.0_a3.0_t1.0.jsonl" ;;
    pless_a4)   echo "pless_alpha_think_t1.0_a4.0_t1.0.jsonl" ;;
    pless_a5)   echo "pless_alpha_think_t1.0_a5.0_t1.0.jsonl" ;;
  esac
}

# Default = the 4 winners (top-4 of the 27-sweep). NO --task-ids → full 252-problem set.
ARMS_DEFAULT="pless_a3 pless_a4 pless_a5 pless_t2.0"
read -ra ARMS_ARR <<< "${ARMS:-$ARMS_DEFAULT}"

# ── Skip-27 optimization: pre-seed the output dir with the existing 27-sweep JSONLs.
# The runner's resume (load_completed_ids) then skips those 27 task_ids and generates
# only the remaining 225 — saving the SLOWEST (truncation-prone) problems' time. The
# merged file ends up with all 252. ONLY valid when caps match (27-sweep was 32768);
# at a different cap the 27 results aren't comparable, so we refuse to pre-seed.
SEED_27="${SEED_27:-1}"
SWEEP27_DIR="${SWEEP27_DIR:-results/pless_recovery_sweep/$MODEL_DIR/${SOURCE}_${DIFFICULTY}}"
if [ "$SEED_27" = "1" ]; then
  if [ "$MAX_TOKENS" != "32768" ]; then
    echo "NOTE: SEED_27=1 but MAX_TOKENS=$MAX_TOKENS != 32768 (the 27-sweep cap). The 27-sweep" >&2
    echo "      results aren't cap-comparable; NOT pre-seeding — all 252 will be generated." >&2
  else
    for a in "${ARMS_ARR[@]}"; do
      src="$SWEEP27_DIR/$(arm_jsonl "$a")"; dst="$OUT_DIR/$(arm_jsonl "$a")"
      if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "pre-seeded $(arm_jsonl "$a") with $(wc -l < "$dst") completed tasks (27-sweep) — runner will skip them"
      fi
    done
  fi
fi

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm  ($(arm_args "$arm"))  [FULL 252 — no --task-ids]"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    --results-dir "$RESULTS_DIR" \
    $(arm_args "$arm")
}

check_vllm

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPULIST <<< "$GPUS"; ngpu=${#GPULIST[@]}
  echo "Parallel: ${#ARMS_ARR[@]} arms across $ngpu GPU(s) [$GPUS]"
  pids=()
  for ((g=0; g<ngpu; g++)); do
    group=(); for ((i=0; i<${#ARMS_ARR[@]}; i++)); do (( i % ngpu == g )) && group+=("${ARMS_ARR[$i]}"); done
    [ ${#group[@]} -eq 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES="${GPULIST[$g]}"
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/full252_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/full252_gpu*.log" >&2; exit 4; }
else
  for a in "${ARMS_ARR[@]}"; do gen_arm "$a"; done
fi

echo "---- eval (re-execute against APPS tests) ----"
for f in "$OUT_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  case "$f" in *.entropy.*) continue;; esac
  echo "  eval $(basename "$f")"
  uv run python -m bench.eval --results-file "$f" --dataset apps --workers "$WORKERS" --skip-diversity
done

echo "---- cot-efficiency report (pass@k + trunc% per arm, alpha-labeled) ----"
uv run python -m bench.eval.cot_efficiency \
  --results-dir "$OUT_DIR" --dataset apps --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"

echo "---- REGRESSION analysis vs baseline (the actual question) ----"
uv run python scripts/recovery_regression_analysis.py \
  --variant-dir "$OUT_DIR/metrics" \
  --baseline-metrics "$BASELINE_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
  --out "$OUT_DIR/analysis/regression_vs_baseline.md"

echo
echo "Done. Report: $OUT_DIR/analysis/cot_efficiency_apps_report.md"
echo "Regression vs baseline: $OUT_DIR/analysis/regression_vs_baseline.md"
