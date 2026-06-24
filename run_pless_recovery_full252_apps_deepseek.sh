#!/bin/bash
# Full-252 α-recovery sweep for DeepSeek-R1-Distill-Llama-8B (ATCODER-interview, vLLM).
#
# MIRRORS run_pless_recovery_full252_apps_qwen3.sh exactly, retargeted to DeepSeek.
# Tests the prevention hypothesis on the HARDER, more loop-prone model: does running
# pless_alpha at α=3/4/5 from the start cut DeepSeek's catastrophic looping (baseline
# pless α=2: ~52% runaway, pass@1 0.174) the way it did for Qwen3-8B
# (14.5%→0.6% truncation, pass@1 0.625→0.696)?
#
# APPLES-TO-APPLES with the existing DeepSeek baseline:
#   results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/
#   (pless α=2 T1.0 = pless_think_t1.0_t1.0.jsonl, pass@1=0.174). This run generates ONLY
#   the new α arms; the baseline is NOT regenerated. The comparison is valid ONLY if this
#   run uses the SAME generation cap as that baseline — see MAX_TOKENS below.
#
# MAX_TOKENS = 32768 — matches the DeepSeek α=2 baseline's --max-new-tokens.
#   Confirmed from docs/theory/todos.md (A37, verified prior-session finding): the baseline
#   "runs to the 32768 wall" (median think ~29.6K all / ~5.3K done; trunc 64.9% by </think>).
#   The cot_efficiency CSV's budget=32768 agrees. A re-tokenization of the stored text shows
#   samples up to ~47K "tokens", but that is an artifact of DeepSeek's byte-level BPE decoder
#   breaking under transformers 5.x (re-encoding inflates the count) — vLLM actually capped
#   generation at 32768. Keep MAX_TOKENS=32768 for the apples-to-apples comparison.
#
# RUN ON A CUDA POD. Requires .venv-vllm (pyproject-vllm.toml). DeepSeek-R1-Distill needs
# the larger 131072 model context (baseline used max_seq_len=131072 — vLLM default for
# this model; do NOT override --max-model-len or KV-cache behaviour changes vs baseline).
#
# Usage:
#   MAX_TOKENS=<baseline_cap> GPUS=0,1 ./run_pless_recovery_full252_apps_deepseek.sh
#   ARMS="pless_a5" MAX_TOKENS=<baseline_cap> GPUS=0 ./run_pless_recovery_full252_apps_deepseek.sh
# Env: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES(10), MAX_TOKENS(49152 placeholder),
#   VLLM_VENV, WORKERS(32), TOKENIZER, GPUS, ARMS, BASELINE_DIR.

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_recovery_full252_deepseek}"
N_SAMPLES="${N_SAMPLES:-10}"
# Matches the DeepSeek α=2 baseline cap (the "32768 wall" — see note above).
MAX_TOKENS="${MAX_TOKENS:-32768}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
MODEL_DIR="${MODEL//\//--}"
BASELINE_DIR="${BASELINE_DIR:-results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview}"

# vLLM env hygiene (identical to the Qwen recovery run).
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

# Same arm definitions as the Qwen recovery run — pless_alpha at α∈{3,4,5}, plus pless T2.0
# as a temperature-loosening comparator. α=2 is the existing baseline (not regenerated).
arm_args() {
  case "$1" in
    pless_t2.0) echo "--method pless       --temperature 2.0" ;;
    pless_a3)   echo "--method pless_alpha --alpha 3 --temperature 1.0" ;;
    pless_a4)   echo "--method pless_alpha --alpha 4 --temperature 1.0" ;;
    pless_a5)   echo "--method pless_alpha --alpha 5 --temperature 1.0" ;;
    *) echo "unknown arm '$1'" >&2; return 1 ;;
  esac
}

# arm -> JSONL basename the runner writes (matches the Qwen recovery naming for tooling reuse).
arm_jsonl() {
  case "$1" in
    pless_t2.0) echo "pless_think_t2.0_t2.0.jsonl" ;;
    pless_a3)   echo "pless_alpha_think_t1.0_a3.0_t1.0.jsonl" ;;
    pless_a4)   echo "pless_alpha_think_t1.0_a4.0_t1.0.jsonl" ;;
    pless_a5)   echo "pless_alpha_think_t1.0_a5.0_t1.0.jsonl" ;;
  esac
}

# Default = the 3 α arms (the prevention sweep). NO --task-ids → full 252-problem set.
# No 27-task pre-seed (unlike Qwen): DeepSeek has no prior recovery-sweep to reuse, so
# every arm generates all 252 from scratch (or resumes from a partial JSONL in OUT_DIR).
ARMS_DEFAULT="pless_a3 pless_a4 pless_a5"
read -ra ARMS_ARR <<< "${ARMS:-$ARMS_DEFAULT}"

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm  ($(arm_args "$arm"))  [FULL 252 — no --task-ids]  cap=$MAX_TOKENS"
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

echo "=================================================================="
echo " DeepSeek α-recovery sweep — full 252"
echo "   model=$MODEL  cap=$MAX_TOKENS  arms=${ARMS_ARR[*]}"
echo "   baseline (α=2, NOT regenerated): $BASELINE_DIR/pless_think_t1.0_t1.0.jsonl"
echo "   ⚠ verify cap=$MAX_TOKENS matches the baseline's --max-new-tokens (see header)"
echo "=================================================================="

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

echo "---- REGRESSION analysis vs α=2 baseline (the actual question) ----"
uv run python scripts/recovery_regression_analysis.py \
  --variant-dir "$OUT_DIR/metrics" \
  --baseline-metrics "$BASELINE_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
  --out "$OUT_DIR/analysis/regression_vs_baseline.md"

echo
echo "Done. Report: $OUT_DIR/analysis/cot_efficiency_apps_report.md"
echo "Regression vs baseline: $OUT_DIR/analysis/regression_vs_baseline.md"
