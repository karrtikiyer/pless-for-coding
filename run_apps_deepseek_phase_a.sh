#!/bin/bash
# Phase A — Deepseek-6.7B-Instruct apples-to-apples comparison vs paper Table 2.
#
# Goal: can our pless_alpha @ α=5 beat the paper's reported pass@10 = 0.1993
# for Deepseek-6.7B-Instruct on CODEFORCES_introductory?
#
# Design (4 cells on CODEFORCES introductory, 299 problems):
#   Cell 1 — REPLICA baseline @ N=100, nucleus T=1.0 top_p=0.95
#            (validates the pipeline: should reproduce paper's 0.1993)
#   Cell 2 — REPLICA baseline @ N=10, nucleus T=1.0 top_p=0.95
#            (checks pass@10 estimator robustness across N)
#   Cell 3 — Our α=5 @ N=10, pless_alpha T=1.0
#            (the "we beat paper" candidate at our standard N)
#   Cell 4 — Our α=5 @ N=100, pless_alpha T=1.0
#            (the "we beat paper" claim with paper-comparable statistical power)
#
# All four cells use the PAPER'S EXACT PROMPT (loaded via
# bench.apps.paper_replica from sh0416/outputs-apps). This isolates the
# sampler effect from any prompt-format effect.
#
# Estimated wallclock on 1× H100 with vLLM batching:
#   Cell 1 (N=100): ~20-25 min
#   Cell 2 (N=10):  ~2-3 min
#   Cell 3 (N=10):  ~2-3 min
#   Cell 4 (N=100): ~20-25 min
#   Total: ~50-60 min
#
# Pre-flight (run before the full 4-cell launch):
#   SMOKE=1 ./run_apps_deepseek_phase_a.sh
# → 1 problem × 2 samples × nucleus + α=5, prints outputs for visual inspection
#
# Usage:
#   ./run_apps_deepseek_phase_a.sh             # full 4-cell run (BACKEND=vllm default)
#   SMOKE=1 ./run_apps_deepseek_phase_a.sh     # pre-flight only
#   FORCE=1 ./run_apps_deepseek_phase_a.sh     # re-run cells already completed
#   BACKEND=hf ./run_apps_deepseek_phase_a.sh  # fall back to HF if vLLM has issues

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-coder-6.7b-instruct}"
SOURCE="${SOURCE:-CODEFORCES}"
DIFFICULTY="${DIFFICULTY:-introductory}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_apps_deepseek_phase_a}"
LOG_DIR="${LOG_DIR:-/tmp/deepseek_phase_a_logs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"   # match paper
SEED="${SEED:-0}"
BACKEND="${BACKEND:-vllm}"   # default vLLM (much faster); BACKEND=hf to fall back
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"  # vLLM lives in a separate venv (incompatible deps with main .venv)
mkdir -p "$LOG_DIR"

# NOTE: do NOT export VLLM_WORKER_MULTIPROC_METHOD=spawn here. The
# PlessSplitLogitsProcessor class in bench/generator_vllm.py is defined
# inside a factory function (to keep the file Mac-importable when vLLM
# isn't installed), and spawn-mode workers can't pickle local classes
# (`AttributeError: Can't pickle local object ...PlessSplitLogitsProcessor`).
# The working APPS vLLM drivers (run_apps_qwen3_top_configs_vllm.sh,
# run_pless_alpha_apps_all_models.sh) intentionally leave this unset
# so vLLM uses its Linux default (fork), which doesn't pickle. If a
# future vLLM release forces spawn, the proper fix is to lift the
# processor class to module level (deferred — non-blocking for Phase A).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Disable vLLM's FlashInfer top-k/top-p sampler. FlashInfer JIT-compiles
# its CUDA kernel via `ninja` on first use; if ninja isn't installed in
# PATH the engine startup crashes with
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
# (verified on the H100 pod 2026-05-26). With this flag set to 0, vLLM
# falls back to the PyTorch-native sampler (Triton for batch >= 8) — per
# vLLM 0.21 envs.py: "set to 0 to opt out explicitly, which forces the
# PyTorch-native (Triton for bs>=8) path". For our 6.7B-class models the
# sampler speed difference is negligible vs the forward pass.
# Alternative if you want the FlashInfer kernel: `pip install ninja` in
# the vLLM venv. Setting this flag is cheap insurance either way.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Pick the python interpreter based on backend: vLLM needs its dedicated venv.
if [ "$BACKEND" = "vllm" ]; then
  if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo "ERROR: BACKEND=vllm requires the vLLM venv at '$VLLM_VENV/bin/python'." >&2
    echo "Bootstrap it once with:" >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2
    echo "Or fall back to HF with: BACKEND=hf $0" >&2
    exit 2
  fi
  if ! "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm import failed inside $VLLM_VENV. Re-sync the venv from" >&2
    echo "       pyproject-vllm.toml and try again." >&2
    exit 3
  fi
  PYBIN=("$VLLM_VENV/bin/python")
else
  PYBIN=(uv run python)
fi

if [ "${SMOKE:-0}" = "1" ]; then
  echo "═══ SMOKE MODE ═══"
  echo "Running 1 problem × 2 samples under nucleus + α=5 to visually inspect."
  "${PYBIN[@]}" -m bench.apps \
    --model "$MODEL" \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --method temp --temperature 1.0 \
    --n-samples 2 --max-new-tokens "$MAX_NEW_TOKENS" --max-problems 1 \
    --backend "$BACKEND" --dtype bfloat16 \
    --paper-replica-model "$MODEL" \
    --results-dir "${RESULTS_DIR}_smoke" \
    2>&1 | tee "$LOG_DIR/smoke_nucleus.log"
  echo
  "${PYBIN[@]}" -m bench.apps \
    --model "$MODEL" \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --method pless_alpha --alpha 5.0 --temperature 1.0 \
    --n-samples 2 --max-new-tokens "$MAX_NEW_TOKENS" --max-problems 1 \
    --backend "$BACKEND" --dtype bfloat16 \
    --paper-replica-model "$MODEL" \
    --results-dir "${RESULTS_DIR}_smoke" \
    2>&1 | tee "$LOG_DIR/smoke_alpha5.log"
  echo
  echo "── Smoke complete. Inspect outputs at ${RESULTS_DIR}_smoke/ ──"
  exit 0
fi

run_cell() {
  local cell_name="$1"
  local method="$2"
  local n_samples="$3"
  local extra_args="$4"

  local log="$LOG_DIR/${cell_name}.log"
  local out_dir="$RESULTS_DIR/$cell_name"

  echo
  echo "───── Cell: $cell_name  (method=$method, N=$n_samples) ─────"
  if [ -d "$out_dir" ] && [ "${FORCE:-0}" != "1" ]; then
    # Skip if a non-empty JSONL exists
    if find "$out_dir" -name "*.jsonl" -size +0 -print -quit | grep -q .; then
      echo "[skip] $cell_name already has results at $out_dir (FORCE=1 to re-run)"
      return 0
    fi
  fi

  # shellcheck disable=SC2086
  "${PYBIN[@]}" -m bench.apps \
    --model "$MODEL" \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --method "$method" \
    --temperature 1.0 \
    --n-samples "$n_samples" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --backend "$BACKEND" --dtype bfloat16 \
    --paper-replica-model "$MODEL" \
    --results-dir "$out_dir" \
    $extra_args \
    2>&1 | tee "$log"
}

echo "═══════════════════════════════════════════════════════════════════════"
echo "Phase A — Deepseek paper-replica comparison"
echo "  Model:       $MODEL"
echo "  Bucket:      $SOURCE / $DIFFICULTY"
echo "  Backend:     $BACKEND"
echo "  Results dir: $RESULTS_DIR/{cell_<n>}"
echo "  Log dir:     $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

# Cell 1+2: paper-replica baseline (nucleus T=1.0 top_p=0.95 — matches paper).
# Cell 3+4: our α=5 candidate (pless_alpha T=1.0, top_p irrelevant).
run_cell "cell1_nucleus_n100" "temp" 100 "--top-p 0.95"
run_cell "cell2_nucleus_n10"  "temp" 10  "--top-p 0.95"
run_cell "cell3_alpha5_n10"   "pless_alpha" 10  "--alpha 5.0"
run_cell "cell4_alpha5_n100"  "pless_alpha" 100 "--alpha 5.0"

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "All cells complete. Next: evaluate each cell with bench.eval --dataset apps"
echo "  for D in $RESULTS_DIR/cell*; do"
echo "    uv run python -m bench.eval --results-file \$D/*.jsonl --dataset apps --workers 8"
echo "  done"
echo
echo "Then compare pass@10 across cells against paper's CODEFORCES intro: 0.1993"
echo "═══════════════════════════════════════════════════════════════════════"
