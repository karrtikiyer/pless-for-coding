#!/bin/bash
# Forced-</think> recovery at n=10 with real sampling (single GPU, vLLM).
#
# The post-hoc RESCUE question: for the 27 solvable pless-truncated ATCODER-interview
# tasks, cut each pless ramble at its loop onset, force </think> + ```python, and
# generate n=10 code completions with STANDARD temperature sampling. Does a correct
# program come out, and at what per-task rate? (Distinct from A31's loop-PREVENTION
# sweep, which changes the SAMPLER to avoid the ramble in the first place.)
#
# TWO arms (the gap is the finding):
#   PLESS (primary, faithful): code phase uses pless @ temp 1.0 — the SAME sampler the
#     original run used. Answers "does PLESS recover if forced to stop thinking at the
#     loop onset?" pless samples with real diversity wherever the distribution isn't
#     peaked, so n>1 is meaningful (the n=1 MPS "determinism" was partly an MPS artifact).
#   TEMP (ceiling): standard temp 0.8 / top_p 0.95. Answers "is the solution recoverable
#     AT ALL?" temp-recovers-but-pless-can't ⇒ fixable sampler-reachability problem.
# Set ARMS="pless" for pless only, ARMS="temp" for ceiling only (default: both).
#
# RUN ON A CUDA GPU (not macOS). Single GPU is enough (one 8B engine). Requires the
# vLLM venv (pyproject-vllm.toml):
#   uv venv .venv-vllm --python 3.12
#   UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --project pyproject-vllm.toml
#
# Usage:
#   ./run_forced_think_recovery_apps_qwen3.sh
# Env overrides: MODEL, ARMS ("pless temp"), N_SAMPLES (10), PLESS_TEMP (1.0),
#   TEMP (0.8), TOP_P (0.95), MAX_CODE_TOKENS (1024), MAX_MODEL_LEN (24576),
#   GPU_MEM_UTIL (0.90), VLLM_VENV (.venv-vllm), OUT, CUDA_VISIBLE_DEVICES (pin a GPU).

set -euo pipefail

VLLM_VENV="${VLLM_VENV:-.venv-vllm}"

# vLLM env hygiene. VLLM_USE_FLASHINFER_SAMPLER=0 is REQUIRED here too: we use
# top_p sampling, and FlashInfer's top-p kernel JIT-compiles via `ninja` (often
# absent on pods → FileNotFoundError at engine start). =0 forces the native path.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Do NOT set VLLM_WORKER_MULTIPROC_METHOD=spawn (factory-local processor class is
# unpicklable). load_engine here uses register_pless_logitsproc=False, but keep the
# convention. vLLM uses Linux fork by default.

if [ ! -x "$VLLM_VENV/bin/python" ]; then
  echo "Error: vLLM venv not found at '$VLLM_VENV/bin/python'. Bootstrap:" >&2
  echo "  uv venv $VLLM_VENV --python 3.12" >&2
  echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2
  exit 2
fi
"$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null || {
  echo "Error: 'import vllm' failed inside $VLLM_VENV." >&2; exit 3; }

echo "Running forced-</think> recovery (arms='${ARMS:-pless temp}', n=${N_SAMPLES:-10}) on the 27 tasks..."
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$VLLM_VENV/bin/python" scripts/forced_think_recovery_vllm.py

echo
echo "Done. Per-task recovery rates + pass@k: ${OUT:-results/forced_think_recovery/recovery_n10.json}"
echo "Compare vs the n=1 MPS screen (11/27): does the rate hold, and do tasks 711/1277"
echo "(MPS-exception, inconclusive) resolve? Cross-tab recovery vs the R2 'concrete code"
echo "at loop onset' predictor — R2 was 5/5 at n=1; does it stay the cleanest signal?"
