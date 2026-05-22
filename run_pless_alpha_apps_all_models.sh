#!/bin/bash
# Rényi-α p-less APPS sweep — 3 models × 6 (source, difficulty) buckets × 4 α arms.
#
# Adds APPS as a third benchmark alongside MBPP and HumanEval. Uses the
# vLLM backend (3–5× faster than HF on long APPS programs).
#
# Configuration:
#   * Models: Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct, m-a-p OCI-1.3B
#   * Buckets: ATCODER × {introductory, interview, competition},
#              CODEFORCES × {introductory, interview, competition}
#   * α-arms:  {2.0, 2.5, 3.0, 5.0}
#   * Temperature: 1.0
#   * Samples: 10 per problem
#   * Backend: vLLM (override with BACKEND=hf)
#
# Loop order: models (outer) → buckets (middle) → α-arms (inner, one per GPU).
# This keeps a single model loaded across all 24 (bucket × α) combos before
# moving to the next model.
#
# Total: 3 × 6 × 4 = 72 distinct runs. 3649 problems/model × 10 samples × 4 α
#       = ~146k generations/model, ~438k total.
#
# Usage:
#   ./run_pless_alpha_apps_all_models.sh
#
# Env overrides:
#   MODELS           space-separated HF IDs (default: 3 models above)
#   ALPHAS           space-separated α       (default "2.0 2.5 3.0 5.0")
#   SOURCES          (default "ATCODER CODEFORCES")
#   DIFFICULTIES     (default "introductory interview competition")
#   N_SAMPLES                                (default 10)
#   TEMPERATURE                              (default 1.0)
#   MAX_NEW_TOKENS                           (default 8192 — matches existing APPS Qwen3-8B runs;
#                                            APPS programs can be long even without thinking)
#   RESULTS_DIR                              (default results/pless_alpha_apps)
#   BACKEND          hf | vllm                (default vllm)
#   VLLM_VENV        path to vLLM venv root   (default .venv-vllm)
#                    Only used when BACKEND=vllm. See pyproject-vllm.toml
#                    for setup instructions.
#   GPUS                                      (default: auto-detect)
#   MAX_PROBLEMS     cap per bucket (smoke)  (default: unset → full)
#   ONLY_ALPHA                                (default: unset)
#   ONLY_MODEL                                (default: unset)
#   ONLY_BUCKET      e.g. "CODEFORCES competition" (default: unset → all buckets)
#   LOG_DIR                                  (default /tmp/alpha_apps_logs)

set -euo pipefail

MODELS_DEFAULT="Qwen/Qwen2.5-Coder-7B-Instruct codellama/CodeLlama-7b-Instruct-hf m-a-p/OpenCodeInterpreter-DS-1.3B Qwen/Qwen3-8B"
ALPHAS_DEFAULT="2.0 2.5 3.0 5.0"
SOURCES_DEFAULT="ATCODER CODEFORCES"
DIFFICULTIES_DEFAULT="introductory interview competition"

MODELS="${MODELS:-$MODELS_DEFAULT}"
ALPHAS="${ALPHAS:-$ALPHAS_DEFAULT}"
SOURCES="${SOURCES:-$SOURCES_DEFAULT}"
DIFFICULTIES="${DIFFICULTIES:-$DIFFICULTIES_DEFAULT}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# MAX_NEW_TOKENS default lowered 8192 → 2048 since 2026-05-22.
# Rationale (verified on-disk):
#   - Qwen3-NoThink completions on MBPP α=5 are median 20 tokens, p95 72 tokens
#     (n=5000); on HumanEval median 33 tok, p95 75 tok (n=1640).
#   - APPS programs are ~5-10× longer than MBPP function bodies → estimated
#     p95 ≈ 720 tokens for Qwen3-NoThink on APPS. 2048 gives ~3× headroom.
#   - All 4 default models (Qwen2.5-Coder, CodeLlama, m-a-p OCI, Qwen3-8B)
#     run in non-thinking mode by default in this sweep, so 8192 was wasteful.
#   - If you want to also run Qwen3-8B WITH thinking, set both
#     THINKING=on AND MAX_NEW_TOKENS=8192 (median thinking-trace ≈ 7290 tok,
#     p95 ≈ 8697 tok per measured Qwen3-think APPS data; will hit the cap
#     even at 8192).
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
# Qwen3 thinking control. Default "off" — non-thinking mode for Qwen3 models.
# Only applied to model ids containing "Qwen3" (other models silently
# ignore the flag, but we skip it to keep logs clean).
THINKING="${THINKING:-off}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_alpha_apps}"
BACKEND="${BACKEND:-vllm}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
LOG_DIR="${LOG_DIR:-/tmp/alpha_apps_logs}"

# ── vLLM venv preflight ─────────────────────────────────────────────────────
# vLLM lives in a separate venv (.venv-vllm) per project convention
# (see pyproject-vllm.toml). The main .venv does NOT have vllm installed.
if [ "$BACKEND" = "vllm" ]; then
  if [ ! -x "$VLLM_VENV/bin/python" ]; then
    cat >&2 <<EOF
Error: vLLM venv not found at '$VLLM_VENV/bin/python'.

Setup once on the GPU box:
  uv venv $VLLM_VENV --python 3.12
  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync \\
      --project pyproject-vllm.toml --no-install-package torch

Or override VLLM_VENV=/path/to/your/venv if it lives elsewhere.
Alternatively, run with BACKEND=hf to skip vLLM entirely (slower).
EOF
    exit 4
  fi
fi

MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

# ── ONLY_* filters ──────────────────────────────────────────────────────────
if [ -n "${ONLY_ALPHA:-}" ]; then ALPHAS="$ONLY_ALPHA"; fi
if [ -n "${ONLY_MODEL:-}" ]; then MODELS="$ONLY_MODEL"; fi
if [ -n "${ONLY_BUCKET:-}" ]; then
  # Expect "SOURCE DIFFICULTY"
  set -- $ONLY_BUCKET
  SOURCES="$1"; DIFFICULTIES="$2"
fi

# ── GPU auto-detection ──────────────────────────────────────────────────────
if [ -n "${GPUS:-}" ]; then
  GPU_LIST=($GPUS)
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found and GPUS not set." >&2; exit 2
  fi
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
  [ "$GPU_COUNT" -eq 0 ] && { echo "Error: 0 GPUs." >&2; exit 2; }
  GPU_LIST=()
  for ((i=0; i<GPU_COUNT; i++)); do GPU_LIST+=($i); done
fi
N_GPUS=${#GPU_LIST[@]}

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Rényi-α p-less APPS sweep — 3 models × 6 buckets × 4 α arms"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Models:        $MODELS"
echo "  Sources:       $SOURCES"
echo "  Difficulties:  $DIFFICULTIES"
echo "  α grid:        $ALPHAS"
echo "  N samples:     $N_SAMPLES per problem"
echo "  Temperature:   $TEMPERATURE"
echo "  Max tokens:    $MAX_NEW_TOKENS"
echo "  Backend:       $BACKEND"
echo "  Results dir:   $RESULTS_DIR"
echo "  GPUs:          ${GPU_LIST[*]} (count: $N_GPUS)"
echo "  Logs:          $LOG_DIR"
[ -n "${MAX_PROBLEMS:-}" ] && echo "  Max problems:  $MAX_PROBLEMS per bucket (smoke)"
echo "═══════════════════════════════════════════════════════════════════════"
echo

run_arm() {
  local gpu="$1"
  local model="$2"
  local source="$3"
  local difficulty="$4"
  local alpha="$5"
  local model_slug; model_slug=$(echo "$model" | tr '/' '-')
  local log="$LOG_DIR/${model_slug}_${source}_${difficulty}_a${alpha}.log"
  # Chat-tuned models whose HF id lacks 'Instruct'/'Chat' need an opt-in
  # flag so the APPS runner accepts them. Currently just OpenCodeInterpreter.
  local instruct_flag=""
  case "$model" in
    *OpenCodeInterpreter*) instruct_flag="--treat-as-instruct" ;;
  esac
  # Qwen3 thinking is opt-in via THINKING=on. Only Qwen3 models support it;
  # for other models the flag would be silently ignored, but we omit it so
  # the method_key in the output filename stays clean.
  local thinking_flag=""
  case "$model" in
    *Qwen3*)
      if [ "$THINKING" = "on" ]; then thinking_flag="--enable-thinking"; fi
      ;;
  esac
  echo "[GPU $gpu] start $model_slug / $source / $difficulty / α=$alpha"\
" thinking=${thinking_flag:-off} → $log"
  if [ "$BACKEND" = "vllm" ]; then
    # vLLM runs in its parallel venv. PYTHONPATH=$PWD so the source tree
    # wins over any installed copy of bench/.
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    "$VLLM_VENV/bin/python" -m bench.apps \
      --model "$model" \
      --source "$source" --difficulty "$difficulty" \
      --method pless_alpha --alpha "$alpha" \
      --temperature "$TEMPERATURE" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend vllm \
      --results-dir "$RESULTS_DIR" \
      $instruct_flag \
      $thinking_flag \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" uv run python -m bench.apps \
      --model "$model" \
      --source "$source" --difficulty "$difficulty" \
      --method pless_alpha --alpha "$alpha" \
      --temperature "$TEMPERATURE" \
      --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --backend hf \
      --results-dir "$RESULTS_DIR" \
      $instruct_flag \
      $thinking_flag \
      $MAX_PROBLEMS_FLAG \
      > "$log" 2>&1
  fi
  echo "[GPU $gpu] done $model_slug / $source / $difficulty / α=$alpha at $(date +%H:%M:%S)"
}

ALPHA_ARR=($ALPHAS)
N_ALPHAS=${#ALPHA_ARR[@]}

for MODEL in $MODELS; do
  echo "═══ Model: $MODEL ═══════════════════════════════════════════════════"
  for SOURCE in $SOURCES; do
    for DIFFICULTY in $DIFFICULTIES; do
      echo "─── $SOURCE / $DIFFICULTY ───"
      # Distribute α arms across GPUs.
      QUEUE=()
      for ((lane=0; lane<N_GPUS; lane++)); do QUEUE[$lane]=""; done
      for ((i=0; i<N_ALPHAS; i++)); do
        lane=$((i % N_GPUS))
        QUEUE[$lane]="${QUEUE[$lane]} ${ALPHA_ARR[$i]}"
      done
      for ((lane=0; lane<N_GPUS; lane++)); do
        echo "    GPU ${GPU_LIST[$lane]}: α =${QUEUE[$lane]}"
      done

      PIDS=()
      for ((lane=0; lane<N_GPUS; lane++)); do
        gpu="${GPU_LIST[$lane]}"
        alphas_for_lane="${QUEUE[$lane]}"
        (
          for alpha in $alphas_for_lane; do
            run_arm "$gpu" "$MODEL" "$SOURCE" "$DIFFICULTY" "$alpha"
          done
        ) &
        PIDS+=($!)
      done
      wait "${PIDS[@]}"
      echo "─── $SOURCE / $DIFFICULTY done at $(date) ───"
    done
  done
  echo "═══ Model $MODEL done at $(date) ═══"
  echo
done

echo "═══════════════════════════════════════════════════════════════════════"
echo "  APPS α-sweep COMPLETE at $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo
echo "Next steps (CPU-only eval, run on Mac or pod):"
echo "  for f in $RESULTS_DIR/*/{ATCODER,CODEFORCES}_*/pless_alpha*.jsonl; do"
echo "    uv run python -m bench.eval --dataset apps --results-file \"\$f\""
echo "  done"
