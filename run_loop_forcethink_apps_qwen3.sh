#!/bin/bash
# LIVE n-gram loop detection -> force </think>, on all 252 ATCODER-interview tasks.
# pless and pless_norm at DEFAULT config (temperature 1.0, no temp/alpha knob).
#
# The deployable "detect-rambling -> end-thinking" mechanism: during generation, an
# n-gram detector watches the think phase; on a detected loop it forces </think> and
# switches to code — online, at the loop onset, WITHOUT burning the rambling tokens.
# (Distinct from the post-hoc forced-</think> on saved traces; this is live in vLLM.)
#
# Compare against the baseline pless/pless_norm @α2 on the SAME 252 (no loop-force):
#   results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/
#   (baseline pless@α2: pass@1 0.625, trunc 14.5%; pless_norm: 0.629, trunc 16.0%)
# Same filenames here (different RESULTS_DIR), so the comparison is file-to-file.
#
# RUN ON A CUDA POD. Requires .venv-vllm. ⚠ The live-loop-force vLLM path is NEW and
# untested on GPU (no local CUDA) — do a smoke first (MAX_PROBLEMS=4) and confirm
# trunc% drops vs baseline before the full run.
#
# Usage:
#   GPUS=0,1 ./run_loop_forcethink_apps_qwen3.sh                                                          # Qwen3, w1200 (default)
#   MAX_PROBLEMS=4 ./run_loop_forcethink_apps_qwen3.sh                                                     # smoke
#   LOOP_WINDOW=800 RESULTS_DIR=results/loop_forcethink_w800 GPUS=0,1 ./run_loop_forcethink_apps_qwen3.sh  # coverage-protecting (half the FP)
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B TOKENIZER=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     LOOP_WINDOW=3000 RESULTS_DIR=results/loop_forcethink_deepseek_w3000 GPUS=0,1 ./run_loop_forcethink_apps_qwen3.sh  # DeepSeek
# Env: MODEL, SOURCE, DIFFICULTY, RESULTS_DIR, N_SAMPLES(10), MAX_TOKENS(32768),
#   LOOP_N(30), LOOP_K(6), LOOP_WINDOW(1200), VLLM_VENV, WORKERS(32), TOKENIZER,
#   GPUS, ONLY (pless|pless_norm), MAX_PROBLEMS.
#
# DETECTOR DEFAULTS n=30/k=6 are VALIDATED (scripts/detector_falsepos_check.py, no GPU):
# false-positive on productive reasoning 0.3% (vs 90.3% for the old n=8/k=4 that broke the
# first run — fired at median ~1.5K, cratering cond-correctness 0.73->0.53), catches 70%
# of genuine loops at median ~7K tokens. NO min-think floor needed — the detector's
# strictness self-separates loops from reasoning. Do NOT use n=8/k=4.
#
# WINDOW is the truncation<->coverage dial (validated full-252, n=30/k=6, Qwen3):
#   w400  cut truncation only ~5pp (too conservative — periods >400 tok escape it).
#   w1200 (DEFAULT): trunc 14.5->1.5% (pless) / 16.0->1.6% (pnorm), pass@1 +2-3pp, mean think
#         ~25% fewer — BUT pass@10 -1.2/-2.3pp (the FP cuts genuine long-reasoners).
#   w800  (offline 91% catch / 1.1% FP — HALF the FP): protects pass@10 at ~3-4% residual trunc.
# DeepSeek-R1-Distill needs a WIDER window (~3000): its loops have longer periods (offline
# 98.5% catch / 1.2% FP at w3000; w1200 only 80.5%). See docs/loopforce_w1200_comparison_apps_qwen3.md.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
RESULTS_DIR="${RESULTS_DIR:-results/loop_forcethink}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-6}"
LOOP_WINDOW="${LOOP_WINDOW:-1200}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
WORKERS="${WORKERS:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL//\//--}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Do NOT set VLLM_WORKER_MULTIPROC_METHOD=spawn (unpicklable factory-local processor).

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"

check_vllm() {
  [ -x "$VLLM_VENV/bin/python" ] || { echo "Error: vLLM venv $VLLM_VENV missing." >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2; exit 2; }
  "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null || { echo "Error: import vllm failed." >&2; exit 3; }
}

ARMS=(pless pless_norm)
want() { [ -z "${ONLY:-}" ] || [ "$1" = "$ONLY" ]; }

gen_arm() {
  local arm="$1"
  echo ">>> arm $arm (T1.0, --force-think-on-loop n=$LOOP_N k=$LOOP_K window=$LOOP_WINDOW)"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" -m bench.apps \
    --model "$MODEL" --backend vllm \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --enable-thinking \
    --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    ${MAX_PROBLEMS:+--max-problems $MAX_PROBLEMS} \
    --results-dir "$RESULTS_DIR" \
    --method "$arm" --temperature 1.0 \
    --force-think-on-loop --loop-ngram-n "$LOOP_N" --loop-ngram-k "$LOOP_K" --loop-window "$LOOP_WINDOW"
}

check_vllm
selected=()
for a in "${ARMS[@]}"; do want "$a" && selected+=("$a"); done

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPULIST <<< "$GPUS"; ngpu=${#GPULIST[@]}
  echo "Parallel: ${#selected[@]} arms across $ngpu GPU(s) [$GPUS]"
  pids=()
  for ((g=0; g<ngpu; g++)); do
    group=(); for ((i=0; i<${#selected[@]}; i++)); do (( i % ngpu == g )) && group+=("${selected[$i]}"); done
    [ ${#group[@]} -eq 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES="${GPULIST[$g]}"
      for a in "${group[@]}"; do gen_arm "$a"; done ) > "$OUT_DIR/loopforce_gpu${GPULIST[$g]}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -ne 0 ] && { echo "a GPU worker failed — see $OUT_DIR/loopforce_gpu*.log" >&2; exit 4; }
else
  for a in "${selected[@]}"; do gen_arm "$a"; done
fi

echo "---- eval (re-execute against APPS tests) ----"
for f in "$OUT_DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  case "$f" in *.entropy.*) continue;; esac
  echo "  eval $(basename "$f")"
  uv run python -m bench.eval --results-file "$f" --dataset apps --workers "$WORKERS" --skip-diversity
done

echo "---- cot-efficiency report (pass@k + trunc% per arm) ----"
uv run python -m bench.eval.cot_efficiency \
  --results-dir "$OUT_DIR" --dataset apps --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"

echo
echo "Done. Report: $OUT_DIR/analysis/cot_efficiency_apps_report.md"
echo "Compare vs baseline (no loop-force) at the canonical 252 dir: did live force-</think>"
echo "drop trunc% (baseline pless 14.5%, pless_norm 16.0%) and raise pass@1 (baseline 0.625/0.629)?"