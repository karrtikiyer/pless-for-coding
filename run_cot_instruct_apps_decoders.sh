#!/bin/bash
# Within-model decoder comparison on APPS, under INDUCED CoT, for an *instruct*
# (non-reasoning) model. Question: under induced CoT (<think> prefill), do the
# hyperparameter-free pless / pless_norm samplers give better pass@k AND lower
# token consumption than well-tuned standard stochastic decoders?
#
# All arms use --prompt-format cot-prefill (CoT is the fixed condition; the
# decoder is the only variable). NOT --enable-thinking — the model has no native
# think phase; CoT is induced via the <think> prefill (see
# bench/apps/prompts.py::format_prompt_apps_cot_prefill). TODO A29.
#
# Configs (per model):
#   PLESS  — pless           @ t1.0          (hyperparameter-free, under test)
#   PNORM  — pless_norm       @ t1.0          (hyperparameter-free, under test)
#   A      — provider combo   temp0.7 + top_p0.8 + top_k20 + rep_penalty
#            (verified from the shipped generation_config.json: 1.1 for 7B,
#             1.05 for 3B; auto-derived from MODEL, override with REP_PENALTY)
#   B      — literature pass@1 temp0.2 + top_p0.95  (BigCode / code-gen std)
#   C      — diversity        temp0.8
#
# Usage:
#   ./run_cot_instruct_apps_decoders.sh ATCODER interview            # 7B (default)
#   MODEL=Qwen/Qwen2.5-Coder-3B-Instruct ./run_cot_instruct_apps_decoders.sh ATCODER interview
#
# Env overrides:
#   MODEL           HF model id (default: Qwen/Qwen2.5-Coder-7B-Instruct)
#   N_SAMPLES       samples/problem (default: 10)
#   MAX_NEW_TOKENS  token budget incl. CoT (default: 8192)
#   RESULTS_DIR     output root (default: results/pless_cot_efficiency_instruct)
#   BACKEND         hf | vllm (default: hf; use vllm on a CUDA pod)
#   VLLM_VENV       vLLM venv root, used when BACKEND=vllm (default: .venv-vllm)
#   VLLM_USE_FLASHINFER_SAMPLER  default 0 (off) — avoids the ninja/FlashInfer
#                   JIT crash on the top_p/top_k arms; export 1 to re-enable
#   REP_PENALTY     provider rep penalty for arm A (default: auto from MODEL)
#   CONFIGS         comma-separated subset (default: PLESS,PNORM,A,B,C)
#   ONLY            run a single config (e.g. ONLY=A)
#   MAX_PROBLEMS    cap for smoke testing; empty = full bucket
set -euo pipefail

SOURCE="${1:-ATCODER}"
DIFFICULTY="${2:-interview}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
RESULTS_DIR="${RESULTS_DIR:-results/pless_cot_efficiency_instruct}"
BACKEND="${BACKEND:-hf}"
CONFIGS="${CONFIGS:-PLESS,PNORM,A,B,C}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"

# FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel via `ninja`; if
# ninja is absent it crashes vLLM at startup with FileNotFoundError. Arms A
# (top_p+top_k) and B (top_p) hit exactly that path, so force it off (same as
# run_apps_deepseek_*.sh). Harmless under --backend hf. Override by exporting it.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Provider rep_penalty (arm A): from each model's shipped generation_config.json.
if [ -z "${REP_PENALTY:-}" ]; then
  case "$MODEL" in
    *3B*) REP_PENALTY=1.05 ;;
    *)    REP_PENALTY=1.1  ;;
  esac
fi

# Pick the interpreter by backend: vLLM needs the parallel .venv-vllm (no vLLM
# wheels on Mac); hf runs under the default uv venv (incl. local MPS smokes).
if [ "$BACKEND" = "vllm" ]; then
  if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo "Error: vLLM venv not found at $VLLM_VENV (set VLLM_VENV). Bootstrap:" >&2
    echo "  uv venv $VLLM_VENV --python 3.12" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$VLLM_VENV uv sync --project pyproject-vllm.toml" >&2
    exit 2
  fi
  if ! "$VLLM_VENV/bin/python" -c "import vllm" 2>/dev/null; then
    echo "Error: 'import vllm' failed inside $VLLM_VENV — re-sync from pyproject-vllm.toml." >&2
    exit 3
  fi
  PYBIN=("$VLLM_VENV/bin/python")
else
  PYBIN=(uv run python)
fi

MAX_PROBLEMS_FLAG=""
if [ -n "${MAX_PROBLEMS:-}" ]; then
  MAX_PROBLEMS_FLAG="--max-problems $MAX_PROBLEMS"
fi

run_one() {
  local cfg="$1"; shift
  echo
  echo "──── $cfg : $MODEL on $SOURCE/$DIFFICULTY (backend=$BACKEND) ────"
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "${PYBIN[@]}" -m bench.apps \
    --model "$MODEL" \
    --source "$SOURCE" --difficulty "$DIFFICULTY" \
    --prompt-format cot-prefill \
    --backend "$BACKEND" \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --results-dir "$RESULTS_DIR" \
    $MAX_PROBLEMS_FLAG \
    "$@"
}

# Portable membership test (no bash-4 associative arrays — macOS ships bash 3.2,
# so this also runs for local MPS smokes, not just Linux pods).
filter() {
  local cfg="$1"
  [ -n "${ONLY:-}" ] && [ "$cfg" != "$ONLY" ] && return 1
  case ",$CONFIGS," in *",$cfg,"*) return 0 ;; *) return 1 ;; esac
}

# ─── pless / pless_norm @ t1.0 (hyperparameter-free, under test) ────────────
if filter PLESS; then
  run_one PLESS --method pless      --temperature 1.0
fi
if filter PNORM; then
  run_one PNORM --method pless_norm --temperature 1.0
fi

# ─── A: provider combo (temp0.7 + top_p0.8 + top_k20 + rep_penalty) ─────────
if filter A; then
  run_one A --method temp --temperature 0.7 --top-p 0.8 --top-k 20 \
    --repetition-penalty "$REP_PENALTY"
fi

# ─── B: literature pass@1 (temp0.2 + top_p0.95) ─────────────────────────────
if filter B; then
  run_one B --method temp --temperature 0.2 --top-p 0.95
fi

# ─── C: diversity (temp0.8) ─────────────────────────────────────────────────
if filter C; then
  run_one C --method temp --temperature 0.8
fi

echo
echo "All requested configs done for $MODEL on $SOURCE/$DIFFICULTY."
echo "Eval:  uv run python -m bench.eval --results-file <jsonl> --dataset apps"
echo "Report: uv run python -m bench.eval.cot_efficiency --results-dir $RESULTS_DIR \\"
echo "          --dataset apps --max-tokens $MAX_NEW_TOKENS \\"
echo "          --tokenizer $MODEL --think-delimiter fence"
