#!/usr/bin/env bash
# External-scaffold reasoning-transfer experiment (TODO A42) on APPS ATCODER-interview.
#
# The 26 tasks Qwen3-8B never solved across any full-252 config (all thinking-ON)
# are re-run with thinking OFF in two conditions that differ ONLY by the prompt:
#   * baseline_nothink : plain instruct prompt (control)
#   * claude_scaffold  : prompt augmented with the Claude-Opus algorithm scaffold
#                        (results/scaffold_transfer/scaffolds.jsonl, committed)
# Headline = newly_recovered = treatment_solved - baseline_solved.
#
# Built/validated on MPS; this script is tuned for a CUDA box (RTX 4090) — default
# BACKEND=vllm (much faster than HF). Override any knob via env var.
#
#   BACKEND=vllm N_SAMPLES=5 MAX_NEW_TOKENS=4096 ./run_scaffold_transfer_apps_qwen3.sh
#
# After it finishes, evaluate + analyze (see the tail of this script).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SOURCE="${SOURCE:-ATCODER}"
DIFFICULTY="${DIFFICULTY:-interview}"
BACKEND="${BACKEND:-vllm}"            # vllm on CUDA; hf elsewhere
N_SAMPLES="${N_SAMPLES:-10}"          # pilot with 5, then 10
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"   # thinking OFF -> pure code; 4096 is ample
# Qwen3-8B NON-thinking recommended sampling (HF model card): t0.7 / p0.8 / k20.
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
OUT_ROOT="${OUT_ROOT:-results/scaffold_transfer}"
SCAFFOLD_FILE="${SCAFFOLD_FILE:-$OUT_ROOT/scaffolds.jsonl}"

# The 26 never-solved task_ids (see docs/theory/todos.md A42).
TASK_IDS=(117 257 326 370 454 455 512 661 929 962 1122 1175 1223 1368 1469 1471 \
          1581 1717 2374 2390 2500 2503 2659 2715 2749 2886)

common=(--model "$MODEL" --backend "$BACKEND" --source "$SOURCE" --difficulty "$DIFFICULTY" \
        --method temp --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
        --task-ids "${TASK_IDS[@]}" --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS")
# NOTE: thinking is OFF by default (no --enable-thinking) — that is the experiment.

echo "=== CONDITION 1/2: baseline_nothink (no scaffold) ==="
uv run python -m bench.apps "${common[@]}" \
  --results-dir "$OUT_ROOT/baseline_nothink"

echo "=== CONDITION 2/2: claude_scaffold (Opus scaffold) ==="
uv run python -m bench.apps "${common[@]}" \
  --scaffold-file "$SCAFFOLD_FILE" \
  --results-dir "$OUT_ROOT/claude_scaffold"

echo
echo "Generation done. Now evaluate + analyze:"
echo "  base=\$(ls $OUT_ROOT/baseline_nothink/*/${SOURCE}_${DIFFICULTY}/*.jsonl)"
echo "  treat=\$(ls $OUT_ROOT/claude_scaffold/*/${SOURCE}_${DIFFICULTY}/*.jsonl)"
echo "  uv run python -m bench.eval --results-file \"\$base\"  --dataset apps --k 1,5,10 --skip-diversity"
echo "  uv run python -m bench.eval --results-file \"\$treat\" --dataset apps --k 1,5,10 --skip-diversity"
echo "  uv run python -m bench.eval.scaffold_transfer_analysis \\"
echo "      --baseline-metrics <baseline _metrics.json> --treatment-metrics <treatment _metrics.json> \\"
echo "      --out $OUT_ROOT/analysis/transfer.md"
