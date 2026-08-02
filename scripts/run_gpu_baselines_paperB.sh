#!/usr/bin/env bash
# =============================================================================
# Paper 2 (paper/paperB) — the three pending GPU baselines, one session.
#
# Produces the data for:
#   (A) GREEDY baseline           -> unlocks the "collapse to greedy" title +
#                                     closes the reviewer's obvious baseline hole.
#   (B) UNFILTERED temp @ T=1.0   -> the missing same-temperature control for
#                                     Figure 1 (pure temperature, no top-p/top-k).
#   (C) SURVIVOR-SET entropy probe-> real per-step data for the replacement
#                                     Figure 2 (survivor-set size on looping vs
#                                     healthy steps). Replaces the OLD, useless
#                                     results/pless_alpha_entropy/* (wrong models
#                                     + wrong datasets).
#
# Everything uses the EXISTING, proven bench.apps.runner. No new runner code.
#   - greedy      = --method temp --top-k 1        (only the argmax survives)
#   - unfilt temp = --method temp --top-p 1.0 --top-k 0
#   - probe       = --method pless --log-entropy    (HF backend only; writes a
#                   <out>.entropy.jsonl sidecar with sigma_p2/max_p/top32 via
#                   bench/generator.py:_log_entropy_batch)
#
# RUN ON A CUDA POD.
#   # full-precision vLLM venv for (A)/(B); HF works from either venv for (C)
#   VLLM_VENV=/workspace/vllm_env/.venv GPUS=0 ./scripts/run_gpu_baselines_paperB.sh
#
# Scope knobs (env): N_SAMPLES (default 10), MAX_TOKENS (32768, matches paper),
#   PROBE_PROBLEMS (30) and PROBE_SAMPLES (3) keep the HF probe tractable.
# =============================================================================
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
PYTHON="$VLLM_VENV/bin/python"
GPUS="${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_USE_FLASHINFER_SAMPLER=0     # baseline parity, matches paper runs

SOURCE="ATCODER"
DIFFICULTY="interview"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
RESULTS_DIR="${RESULTS_DIR:-results/_paperB_baselines}"

# HF probe subset (generation on HF is slow at 32k tokens; a subset is enough to
# compare survivor-set size on looping vs healthy steps).
PROBE_PROBLEMS="${PROBE_PROBLEMS:-30}"
PROBE_SAMPLES="${PROBE_SAMPLES:-3}"

MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "Qwen/Qwen3-8B"
)

common_vllm=( --source "$SOURCE" --difficulty "$DIFFICULTY" --enable-thinking
              --backend vllm --n-samples "$N_SAMPLES"
              --max-new-tokens "$MAX_TOKENS" --results-dir "$RESULTS_DIR" )

for MODEL in "${MODELS[@]}"; do
  echo "======================================================================"
  echo "MODEL: $MODEL"
  echo "======================================================================"

  # ---- (A) GREEDY (deterministic; top-k 1 == argmax). n-samples 1. ----------
  echo "[A] greedy (top-k 1)"
  "$PYTHON" -m bench.apps.runner --model "$MODEL" \
      --method temp --temperature 1.0 --top-p 1.0 --top-k 1 \
      --source "$SOURCE" --difficulty "$DIFFICULTY" --enable-thinking \
      --backend vllm --n-samples 1 --max-new-tokens "$MAX_TOKENS" \
      --results-dir "$RESULTS_DIR"

  # ---- (B) UNFILTERED temperature @ T=1.0 (no top-p / no top-k) -------------
  echo "[B] unfiltered temp T=1.0 (top-p 1.0, top-k 0)"
  "$PYTHON" -m bench.apps.runner --model "$MODEL" \
      --method temp --temperature 1.0 --top-p 1.0 --top-k 0 \
      "${common_vllm[@]}"

  # ---- (C) SURVIVOR-SET probe: pless a=2 + --log-entropy (HF backend) -------
  # HF backend required (log-entropy is unsupported on vLLM). pless a=2 loops
  # ~42% (DeepSeek) / ~15% (Qwen), so the subset naturally contains looping and
  # healthy traces to compare. Writes <out>.entropy.jsonl (sigma_p2/max_p/top32).
  echo "[C] survivor-set probe: pless (a=2) + --log-entropy, HF, subset=$PROBE_PROBLEMS x $PROBE_SAMPLES"
  "$PYTHON" -m bench.apps.runner --model "$MODEL" \
      --method pless --temperature 1.0 \
      --source "$SOURCE" --difficulty "$DIFFICULTY" --enable-thinking \
      --backend hf --n-samples "$PROBE_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
      --max-problems "$PROBE_PROBLEMS" --log-entropy \
      --results-dir "$RESULTS_DIR"
done

echo
echo "DONE. Next:"
echo "  1. Score (A)/(B):  python -m bench.eval --dataset apps --skip-diversity \\"
echo "                       --results-file $RESULTS_DIR/<model>/${SOURCE}_${DIFFICULTY}/<jsonl>"
echo "  2. Figure 2:       uv run python scripts/survivor_set_figure.py \\"
echo "                       --root $RESULTS_DIR"
