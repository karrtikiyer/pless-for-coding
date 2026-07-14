#!/bin/bash
# Controlled HF-vs-vLLM backend-equivalence check for DeepSeek-R1-Distill-Llama-8B,
# pless α=2, APPS ATCODER-interview. This is the DeepSeek analogue of the Qwen3-8B check
# that justified "go with vLLM" (Qwen HF↔vLLM agreed: truncation 13.0% vs 12.3%,
# pass@1 0.631 vs 0.625 on the shared 100). That check was NEVER run for DeepSeek; the
# reconstructed numbers diverge hard (truncation ~40% vs 64.9%, pass@1 0.386 vs 0.174),
# so this run settles which backend to trust for DeepSeek.
#
# APPLE-TO-APPLE — only the backend differs from the existing vLLM baseline
#   results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/
#   pless_think_t1.0_t1.0.jsonl (pless α=2, cap 32768, pass@1 0.174, truncation 64.9%).
# Matched: --method pless (α=2), --temperature 1.0 --top-p 1.0 --top-k 0,
#   --max-new-tokens 32768, --enable-thinking, same prompt formatter, SAME task_ids.
# NO loop detector (--force-think-on-loop is OFF) → pure α=2, exactly like the baseline.
#
# STAGED (statistically: pass@1 gap ~21pp ⇒ N=25 is ~0.99-powered; expand only if the
# 25-tier CI overlaps 0). TIER1=25 → TIER2 adds 25 (=50) → TIER3 adds 50 (=100, matches
# the Qwen check's N). task_ids are a systematic every-Nth sample across all 252 (unbiased;
# not outcome-selected).
#
# RUN ON A CUDA POD. HF token-by-token DeepSeek to a 32768 cap is impractical on MPS.
# Requires a CUDA-enabled python env with torch+transformers (the .venv-vllm has both).
#
# Usage:
#   GPUS=0 ./run_backend_delta_apps_deepseek.sh                 # TIER1 (25 tasks)
#   TIER=tier2 GPUS=0 ./run_backend_delta_apps_deepseek.sh      # the +25 expansion
#   TIER=tier3 GPUS=0 ./run_backend_delta_apps_deepseek.sh      # the +50 expansion
# Env: MODEL, N_SAMPLES(10), MAX_TOKENS(32768), PYTHON, GPUS, TIER, RESULTS_DIR, VLLM_VENV.

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SOURCE="ATCODER"
DIFFICULTY="interview"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"          # MUST match the vLLM baseline cap.
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
PYTHON="${PYTHON:-$VLLM_VENV/bin/python}"  # any CUDA torch+transformers env works for --backend hf
RESULTS_DIR="${RESULTS_DIR:-results/_backend_delta_deepseek/hf}"
MODEL_DIR="${MODEL//\//--}"
TIER="${TIER:-tier1}"

BASELINE_DIR="results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"

# Systematic every-Nth subsets of the sorted 252 ATCODER-interview task_ids (disjoint;
# cumulative 25 → 50 → 100, the last matching the Qwen backend check's N).
TIER1="117 370 587 827 962 1038 1123 1177 1274 1370 1431 1528 1587 1719 2242 2372 2382 2471 2481 2491 2501 2511 2521 2644 2654"
TIER2="160 417 588 865 963 1039 1124 1178 1275 1371 1432 1578 1648 1788 2304 2373 2383 2472 2482 2492 2502 2512 2522 2645 2655"
TIER3="230 325 454 541 615 739 866 927 990 1034 1040 1088 1125 1173 1222 1226 1276 1328 1372 1427 1469 1524 1579 1583 1649 1715 1789 1925 2305 2368 2374 2378 2384 2388 2473 2477 2483 2487 2493 2497 2503 2507 2513 2517 2523 2640 2646 2650 2656 2660"

case "$TIER" in
  tier1) IDS="$TIER1" ;;
  tier2) IDS="$TIER1 $TIER2" ;;                 # cumulative: 50
  tier3) IDS="$TIER1 $TIER2 $TIER3" ;;          # cumulative: 100 (matches Qwen check N)
  full)  IDS="" ;;                              # ALL 252 — omit --task-ids (whole bucket)
  *) echo "unknown TIER '$TIER' (tier1|tier2|tier3|full)" >&2; exit 2 ;;
esac
# --task-ids present for tiered subsets; omitted for full so the runner takes the
# whole (source,difficulty) bucket. Resume skips task_ids already in the JSONL,
# so `full` continues from whatever a prior tier already generated.
if [ -n "$IDS" ]; then TASKID_ARGS=(--task-ids $IDS); NTASKS=$(echo $IDS | wc -w)
else TASKID_ARGS=(); NTASKS="all-252"; fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[ -x "$PYTHON" ] || { echo "Error: python '$PYTHON' not executable (set PYTHON=... to a CUDA env)." >&2; exit 3; }

OUT_DIR="$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}"
mkdir -p "$OUT_DIR"
JSONL="$OUT_DIR/pless_think_t1.0_t1.0.jsonl"

echo "=================================================================="
echo " Backend-delta HF run — DeepSeek pless α=2 ($TIER, $NTASKS tasks)"
echo "   model=$MODEL  backend=hf  cap=$MAX_TOKENS  n=$N_SAMPLES"
echo "   baseline (vLLM, NOT regenerated): $BASELINE_DIR/pless_think_t1.0_t1.0.jsonl"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${GPUS:-0}" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" -m bench.apps \
  --model "$MODEL" --backend hf --method pless \
  --source "$SOURCE" --difficulty "$DIFFICULTY" \
  --enable-thinking \
  --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  "${TASKID_ARGS[@]}" \
  --results-dir "$RESULTS_DIR"

echo ">>> scoring the HF run through the standard pipeline (bench.eval)"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m bench.eval \
  --results-file "$JSONL" --dataset apps

echo ">>> paired backend-delta report (HF vs vLLM on the SAME task_ids)"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/compare_backend_delta.py \
  --hf-metrics   "$OUT_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
  --hf-jsonl     "$JSONL" \
  --vllm-metrics "$BASELINE_DIR/metrics/pless_think_t1.0_t1.0_metrics.json" \
  --vllm-jsonl   "$BASELINE_DIR/pless_think_t1.0_t1.0.jsonl" \
  "${TASKID_ARGS[@]}" \
  --out "$OUT_DIR/backend_delta_${TIER}.md"
