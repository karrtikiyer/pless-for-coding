#!/bin/bash
# Combined Qwen3-8B Rényi-α sweep across MBPP-500 + HumanEval-164.
#
# Chains the two per-benchmark scripts in one launch so you can `tmux
# new -s qwen3`, kick this off, and walk away. Both scripts respect the
# same env vars (THINKING, BACKEND, GPUS, VLLM_VENV, etc.).
#
# Convention: script lives at the model level (no benchmark qualifier
# in the filename) because it dispatches across multiple benchmarks —
# see CLAUDE.md "Script naming convention" section.
#
# Typical launches:
#
#   # Decisive test (Qwen3 with thinking DISABLED — distinguishes
#   # saturation from a thinking-phase mechanism):
#   BACKEND=vllm THINKING=off VLLM_VENV=/workspace/vllm_env/.venv \
#     GPUS="0 1" ./run_pless_alpha_qwen3_all.sh
#
#   # Default sweep (thinking ON, HF backend):
#   ./run_pless_alpha_qwen3_all.sh
#
# Resume-safe: each per-benchmark script's --no-resume default is off,
# so re-running picks up where it left off. If you Ctrl+C mid-MBPP and
# re-launch this wrapper, MBPP resumes from its last completed task
# and HumanEval then runs cleanly afterwards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Qwen3-8B Rényi-α sweep — MBPP + HumanEval combined runner"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Start time:  $(date)"
echo "  BACKEND:     ${BACKEND:-hf}  (default 'hf'; set BACKEND=vllm for vLLM)"
echo "  THINKING:    ${THINKING:-on} (default 'on'; set THINKING=off for the"
echo "                                decisive saturation-vs-thinking test)"
echo "  GPUS:        ${GPUS:-<auto-detected via nvidia-smi -L>}"
[ -n "${VLLM_VENV:-}" ] && echo "  VLLM_VENV:   $VLLM_VENV"
[ -n "${MAX_PROBLEMS:-}" ] && echo "  MAX_PROBLEMS: $MAX_PROBLEMS (smoke)"
[ -n "${ONLY_ALPHA:-}" ] && echo "  ONLY_ALPHA:  $ONLY_ALPHA"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# ── Stage 1: MBPP-500 ───────────────────────────────────────────────────────
echo "─── Stage 1/2: MBPP-500 ──────────────────────────────────────────────"
echo "  $(date)"
echo
./run_pless_alpha_qwen3_mbpp.sh
echo
echo "─── Stage 1 done at $(date) ──────────────────────────────────────────"
echo

# ── Stage 2: HumanEval-164 ─────────────────────────────────────────────────
echo "─── Stage 2/2: HumanEval-164 ────────────────────────────────────────"
echo "  $(date)"
echo
./run_pless_alpha_qwen3_humaneval.sh
echo
echo "─── Stage 2 done at $(date) ──────────────────────────────────────────"

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Both benchmarks complete at $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo
echo "Next steps (CPU-only eval):"
echo "  for f in results/pless_alpha_full/Qwen--Qwen3-8B/pless_alpha*_t1.0.jsonl; do"
echo "    uv run python -m bench.eval --dataset mbpp --results-file \"\$f\""
echo "  done"
echo "  for f in results/pless_alpha_full_humaneval/Qwen--Qwen3-8B/pless_alpha*_t1.0.jsonl; do"
echo "    uv run python -m bench.eval --dataset humaneval --results-file \"\$f\""
echo "  done"
