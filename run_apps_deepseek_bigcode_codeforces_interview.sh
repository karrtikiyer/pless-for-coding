#!/bin/bash
# Backend-effect isolation experiment: Deepseek-6.7B-Instruct on
# CODEFORCES_interview, nucleus T=1.0 top_p=0.95, N=100, run twice —
# once with HF transformers backend, once with vLLM 0.21 backend.
#
# Prompt format: bigcode-evaluation-harness's bare default
#   "\nQUESTION:\n{question}\nUse Standard Input format\nANSWER:\n"
# No chat template, no system prompt. This matches the prompt
# bigcode-eval-harness ships out of the box (verified verbatim against
# bigcode_eval/tasks/apps.py::get_prompt 2026-05-27). Paper's stored
# prompt differs (chat-template-wrapped) — that's their custom
# modification, not bigcode's default. Using bigcode's bare format
# here isolates the BACKEND effect cleanly.
#
# Why this experiment exists:
# In Phase A, our nucleus N=100 pass@10 (0.126) lagged the paper's
# reported number (0.199) by 7.4 pp. After eliminating extraction,
# executor, prompt-string, problem-set, and sample-count as causes,
# the dominant remaining hypothesis is generation framework: paper
# used bigcode-eval-harness (HF transformers + accelerate); we use
# vLLM. Running both HF and vLLM with bigcode's prompt should tell
# us how much of the gap is framework choice.
#
# Bucket: CODEFORCES interview (2,386 problems × N=100 = 238,600 gens
# per backend × 2 backends = 477,200 total generations).
#
# Cost estimates (single H100):
#   vLLM cell: ~30-50 min wallclock
#   HF cell:   ~5-10 hours wallclock (HF is much slower than vLLM
#              for batched code generation; depends on accelerate
#              config and batch size)
#
# Outputs:
#   results/pless_alpha_apps_deepseek_bigcode/{hf,vllm}/
#     deepseek-ai--deepseek-coder-6.7b-instruct/
#     CODEFORCES_interview/temp_t1.0.jsonl
#
# Usage:
#   ./run_apps_deepseek_bigcode_codeforces_interview.sh             # both backends
#   BACKENDS=vllm  ./run_apps_deepseek_bigcode_codeforces_interview.sh   # vllm only
#   BACKENDS=hf    ./run_apps_deepseek_bigcode_codeforces_interview.sh   # hf only
#   SMOKE=1 ./run_apps_deepseek_bigcode_codeforces_interview.sh    # 3 problems × 2 samples per backend
#
# Pre-flight (verify the model card chat template / generation works):
#   Run SMOKE=1 first. Check the output JSONL has sensible code samples.
#   The bigcode prompt is bare-completion style, so the model is being
#   used as a base completion model on this task — output quality may
#   differ noticeably from instruct-mode usage.

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-coder-6.7b-instruct}"
SOURCE="${SOURCE:-CODEFORCES}"
DIFFICULTY="${DIFFICULTY:-interview}"
N_SAMPLES="${N_SAMPLES:-100}"
TOP_P="${TOP_P:-0.95}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"   # match paper
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_apps_deepseek_bigcode_chat}"
LOG_DIR="${LOG_DIR:-/tmp/deepseek_bigcode_logs}"
BACKENDS="${BACKENDS:-vllm hf}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
SEED="${SEED:-0}"

mkdir -p "$LOG_DIR"

# Env hygiene (matches other vLLM drivers — see Phase A iterative fixes
# in commits 8bc1b77 → bbd6f20 → 9608094).
#
# NOTE: do NOT export VLLM_WORKER_MULTIPROC_METHOD=spawn here. Our
# bench.generator_vllm.load_engine attaches PlessSplitLogitsProcessor as
# a default logits processor on every engine, even for --method temp.
# That class is defined inside a factory function (to keep the module
# Mac-importable) and spawn-mode workers can't pickle local classes
# (AttributeError: Can't pickle local object ...). Linux's default
# `fork` works because the child inherits parent memory. Leave the
# variable unset; vLLM will pick fork by default.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel via `ninja`
# on first use. Pods often don't ship ninja in PATH, which crashes engine
# startup with FileNotFoundError. VLLM_USE_FLASHINFER_SAMPLER=0 forces
# the PyTorch-native (Triton for bs≥8) sampler instead — negligible
# perf hit for our 6.7B-class workload.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Deepseek-Coder hallucinates `import matplotlib.pyplot` in ~0.5% of
# CODEFORCES competitive-programming samples. On macOS this pops a GUI
# window when each subprocess test runs. Agg backend is in-memory only.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Smoke override
if [ "${SMOKE:-0}" = "1" ]; then
  N_SAMPLES=2
  MAX_PROBLEMS_FLAG="--max-problems 3"
  echo "[SMOKE MODE] 3 problems × 2 samples per backend"
else
  MAX_PROBLEMS_FLAG=""
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "Backend-effect experiment — Deepseek-6.7B-Instruct CODEFORCES_interview"
echo "  Model:           $MODEL"
echo "  Bucket:          $SOURCE / $DIFFICULTY"
echo "  Sampler:         T=$TEMPERATURE, top_p=$TOP_P (nucleus)"
echo "  N samples/task:  $N_SAMPLES"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Prompt format:   bigcode-chat (bigcode bare + apply_chat_template)"
echo "  Backends:        $BACKENDS"
echo "  Results:         $RESULTS_DIR/<backend>/<slug>/<bucket>/temp_t1.0.jsonl"
echo "═══════════════════════════════════════════════════════════════════════"

declare -a FAILED_CELLS=()

for BACKEND in $BACKENDS; do
  CELL_RESULTS="$RESULTS_DIR/$BACKEND"
  LOG="$LOG_DIR/${BACKEND}.log"

  # Pick the right python interpreter for the backend
  if [ "$BACKEND" = "vllm" ]; then
    if [ ! -x "$VLLM_VENV/bin/python" ]; then
      echo "ERROR: vLLM venv missing at $VLLM_VENV — see pyproject-vllm.toml" >&2
      echo "  Bootstrap with:" >&2
      echo "    uv venv $VLLM_VENV --python 3.12" >&2
      echo "    UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2
      echo "Skipping vLLM cell." >&2
      FAILED_CELLS+=("$BACKEND")
      continue
    fi
    # Verify vllm is actually importable inside that venv (catches a
    # broken/incomplete sync before we waste 30 min loading the model).
    if ! "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null; then
      echo "ERROR: vllm import failed inside $VLLM_VENV." >&2
      echo "  Re-sync from pyproject-vllm.toml and try again." >&2
      echo "Skipping vLLM cell." >&2
      FAILED_CELLS+=("$BACKEND")
      continue
    fi
    PYBIN=("$VLLM_VENV/bin/python")
  else
    PYBIN=(uv run python)
  fi

  echo
  echo "── BACKEND=$BACKEND  ─────────────────────────────────────"
  echo "   python: ${PYBIN[*]}"
  echo "   log:    $LOG"
  echo "   out:    $CELL_RESULTS/"

  # shellcheck disable=SC2086
  if "${PYBIN[@]}" -m bench.apps \
    --model "$MODEL" \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --method temp \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --backend "$BACKEND" --dtype bfloat16 \
    --prompt-format bigcode-chat \
    --results-dir "$CELL_RESULTS" \
    $MAX_PROBLEMS_FLAG \
    2>&1 | tee "$LOG"; then
    echo "   ✓ $BACKEND cell completed"
  else
    rc=$?
    echo "   ✗ $BACKEND cell FAILED (exit $rc) — see $LOG"
    FAILED_CELLS+=("$BACKEND")
  fi
done

echo
echo "═══════════════════════════════════════════════════════════════════════"
if [ "${#FAILED_CELLS[@]}" -gt 0 ]; then
  echo "DONE — ${#FAILED_CELLS[@]} cell(s) FAILED: ${FAILED_CELLS[*]}"
  exit 1
fi
echo "DONE — both backends completed. Next: evaluate pass@k."
echo
echo "  for D in $RESULTS_DIR/*/$(echo $MODEL | tr / -)/${SOURCE}_${DIFFICULTY}; do"
echo "    uv run python -m bench.eval --results-file \$D/*.jsonl \\"
echo "      --dataset apps --workers 8 --timeout 5.0 --skip-diversity"
echo "  done"
echo
echo "Then compare pass@10 across (hf, vllm). If they diverge substantially,"
echo "the framework choice IS a major source of the Phase A gap to paper's"
echo "reported number (0.221 for nucleus N=100 on this exact bucket per Table 2)."
echo "═══════════════════════════════════════════════════════════════════════"
