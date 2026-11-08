# scripts/mbpp_staging_common.sh — sourced (not executed) by the 2026-11 MBPP staging scripts:
#   run_bigcode_mbpp_qwen25coder7b.sh, run_chat_mbpp_qwen25coder7binstruct.sh, run_bos_check_mbpp_chat.sh,
#   run_seeded_mbpp_all_models.sh (several models per invocation: sets MODEL_ID / CONFIGS / SINGLE_BOS per run_configs)
# Provides the shared config tables MBPP_TABLE1_BIGCODE / MBPP_HASAN_BIGCODE / MBPP_TABLE1_DEFAULT_PROMPT (assign CONFIGS after
# sourcing) and append_extra_pless_configs.
#
# Runs `uv run python -m bench` (main uv env, HF backend, runner defaults: n-samples 10, max-new-tokens 512,
# resume ON) once per config, exactly like the historical MBPP-full runs (run_new_samplers_mbpp.sh,
# run_bench.sh, run_top_p_mbpp_all_models.sh, run_bigcode_mbpp_qwen25coder{15b,3b}.sh).
#
# Caller sets: MODEL_ID, RESULTS_DIR, LOG_TAG, PROMPT_ARGS (array, may be empty), CONFIGS (array of
#   "<file stem>|<bench args...>").
# Environment:
#   CUDA_VISIBLE_DEVICES  honoured as given by the caller (never set here)
#   MAX_PROBLEMS=N        passthrough to --max-problems N (smoke run); completeness check is skipped
#   DRY_RUN=1             print the commands only (no model load, no uv env check)
#   EXTRA_PLESS_TEMPS=1   run ONLY p-less / p-less-norm at T=1.0 and T=0.6 (user request 2026-11-08), derived from
#                         the caller's pless / pless_norm T=0.7 configs by changing --temperature and the stem's
#                         _t0.7 suffix only (same flags, same root, the runner's standard file names). Unset / 0 /
#                         empty: the caller's CONFIGS unchanged.
#   SEED=N                passthrough: append --seed N (N a non-negative integer; per-task reseeding, see
#                         bench/seeding.py). Unset / empty: unseeded, as all historical runs. Same file names: the
#                         runner refuses to resume a file generated with a different seed (use a new RESULTS_DIR).
#   RESULTS_DIR=PATH      output root override, honoured by the run_*_mbpp_qwen25coder7b*.sh callers (they assign
#                         RESULTS_DIR="${RESULTS_DIR:-<their default root>}"). Unset / empty: the caller's root.
#                         Seeded / single-BOS reruns must use a NEW root, never the old unseeded files.
#   SINGLE_BOS=1          passthrough: append --single-bos (one BOS for chat templates that start with it). Unset / 0 /
#                         empty: unchanged. Same file-name / resume rule as SEED.
#
# Stops on the first failing config (set -euo pipefail + pipefail through tee). bench/runner.py catches
# per-task exceptions and keeps going ("Error on task_id=..."); this helper turns any such line, or a file
# that is not exactly 500 unique task_ids on a full run, into a failure.
# Evaluation is NOT run here: run `python -m bench.eval` locally (project CLAUDE.md gotcha 9).

set -euo pipefail

: "${MODEL_ID:?}" "${RESULTS_DIR:?}" "${LOG_TAG:?}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_PLESS_TEMPS="${EXTRA_PLESS_TEMPS:-}"
SEED="${SEED:-}"
SINGLE_BOS="${SINGLE_BOS:-}"
LOG_DIR="logs"
MODEL_DIR="${MODEL_ID//\//--}"
EXPECTED_TASKS=500

if [[ -n "$MAX_PROBLEMS" && ! "$MAX_PROBLEMS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PROBLEMS must be a positive integer; got '$MAX_PROBLEMS'" >&2
  exit 2
fi

case "$EXTRA_PLESS_TEMPS" in
  ""|0|1) ;;
  *) echo "EXTRA_PLESS_TEMPS must be 1, 0 or unset; got '$EXTRA_PLESS_TEMPS'" >&2; exit 2 ;;
esac

if [[ -n "$SEED" && ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SEED must be a non-negative integer; got '$SEED'" >&2
  exit 2
fi

case "$SINGLE_BOS" in
  ""|0|1) ;;
  *) echo "SINGLE_BOS must be 1, 0 or unset; got '$SINGLE_BOS'" >&2; exit 2 ;;
esac

ist() { TZ=Asia/Kolkata date '+%F %T IST'; }

# Shared config tables ("<runner file stem>|<bench args>"; the stem is the runner's own file name). Used by
# run_bigcode_mbpp_qwen25coder7b.sh, run_chat_mbpp_qwen25coder7binstruct.sh and run_seeded_mbpp_all_models.sh.
# MBPP Table I configs, base models with --prompt-style bigcode (runner appends "_bigcode" to the stem):
MBPP_TABLE1_BIGCODE=(
  "greedy_bigcode_t1.0|--method greedy --temperature 1.0 --n-samples 1"
  "beam4_bigcode_t1.0|--method beam --num-beams 4 --temperature 1.0 --n-samples 1"
  "temp_bigcode_t0.7|--method temp --temperature 0.7"
  "top_p0.9_bigcode_t1.0|--method top_p --top-p 0.9 --temperature 1.0"
  "pless_bigcode_t0.7|--method pless --temperature 0.7"
  "pless_norm_bigcode_t0.7|--method pless_norm --temperature 0.7"
)
MBPP_HASAN_BIGCODE="top_p0.95_bigcode_t0.2|--method top_p --top-p 0.95 --temperature 0.2"   # Hasan et al. config
# MBPP Table I configs with the runner's default prompt (no --prompt-style, no stem suffix): chat/instruct models
# (chat template) and the 3-shot "paper"-prompt base models (CodeLlama-7b-hf, Llama-2-7b-hf; same flags as their April
# rows from run_bench.sh / run_new_samplers_mbpp.sh / run_top_p_mbpp_all_models.sh):
MBPP_TABLE1_DEFAULT_PROMPT=(
  "greedy_t1.0|--method greedy --temperature 1.0 --n-samples 1"
  "beam4_t1.0|--method beam --num-beams 4 --temperature 1.0 --n-samples 1"
  "temp_t0.7|--method temp --temperature 0.7"
  "top_p0.9_t1.0|--method top_p --top-p 0.9 --temperature 1.0"
  "pless_t0.7|--method pless --temperature 0.7"
  "pless_norm_t0.7|--method pless_norm --temperature 0.7"
)

extra_pless_configs() {   # print the four extra configs, one per line, derived from CONFIGS' T=0.7 p-less entries
  local t m cfg stem args found
  for t in 1.0 0.6; do
    for m in pless pless_norm; do
      found=""
      for cfg in "${CONFIGS[@]}"; do
        stem="${cfg%%|*}"; args="${cfg#*|}"
        [[ " $args " == *" --method $m --temperature 0.7 "* ]] || continue
        [[ -z "$found" ]] || { echo "more than one $m T=0.7 config in CONFIGS" >&2; return 1; }
        [[ "$stem" == *_t0.7 ]] || { echo "stem '$stem' of the $m T=0.7 config does not end in _t0.7" >&2; return 1; }
        found="${stem%_t0.7}_t${t}|${args/--temperature 0.7/--temperature $t}"
      done
      [[ -n "$found" ]] || { echo "no $m T=0.7 config in CONFIGS" >&2; return 1; }
      echo "$found"
    done
  done
}

append_extra_pless_configs() {   # CONFIGS += the four extra p-less configs (T=1.0 / T=0.6), same builder as above
  local extra line
  extra=$(extra_pless_configs) || exit 7
  while IFS= read -r line; do CONFIGS+=("$line"); done <<< "$extra"
}

check_env() {   # transformers>=5 in the main uv env (Qwen2.5); never mutates pyproject/uv.lock
  uv run python -c "import sys, transformers, torch; v = transformers.__version__; \
print(f'transformers {v}  torch {torch.__version__}  cuda_available {torch.cuda.is_available()}'); \
sys.exit(0 if int(v.split('.')[0]) >= 5 else 1)" \
    || { echo "main uv env needs transformers>=5 for $MODEL_ID" >&2; exit 3; }
}

check_complete() {   # out_file log_file
  local out="$1" log="$2"
  if grep -q "Error on task_id=" "$log"; then
    echo "FAIL: per-task errors in $log:" >&2
    grep "Error on task_id=" "$log" | head -5 >&2
    exit 4
  fi
  [[ -n "$MAX_PROBLEMS" ]] && return 0
  uv run python -I -c "import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = [r['task_id'] for r in rows]
ok = len(ids) == len(set(ids)) == int(sys.argv[2])
print(f'{sys.argv[1]}: {len(ids)} rows, {len(set(ids))} unique task_ids', '(OK)' if ok else '(INCOMPLETE)')
sys.exit(0 if ok else 1)" "$out" "$EXPECTED_TASKS" || { echo "FAIL: $out incomplete" >&2; exit 5; }
}

run_configs() {
  MODEL_DIR="${MODEL_ID//\//--}"   # recomputed per call: run_seeded_mbpp_all_models.sh changes MODEL_ID between calls
  if [[ "$EXTRA_PLESS_TEMPS" == 1 ]]; then
    local extra line
    extra=$(extra_pless_configs) || exit 7
    CONFIGS=()
    while IFS= read -r line; do CONFIGS+=("$line"); done <<< "$extra"
  fi
  mkdir -p "$LOG_DIR"
  [[ "$DRY_RUN" == 1 ]] || check_env
  echo "[$(ist)] $MODEL_ID -> ${RESULTS_DIR}/${MODEL_DIR}/  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}," \
       "MAX_PROBLEMS=${MAX_PROBLEMS:-<all 500>}, commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
  [[ "$EXTRA_PLESS_TEMPS" == 1 ]] && echo "    EXTRA_PLESS_TEMPS=1: only p-less / p-less-norm at T=1.0 and T=0.6"
  [[ -n "$SEED" ]] && echo "    SEED=$SEED: --seed $SEED"
  [[ "$SINGLE_BOS" == 1 ]] && echo "    SINGLE_BOS=1: --single-bos"
  local cfg stem args
  for cfg in "${CONFIGS[@]}"; do
    stem="${cfg%%|*}"
    read -ra args <<< "${cfg#*|}"
    local cmd=(uv run python -m bench --model "$MODEL_ID" "${args[@]}" ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}
               --mbpp-config full --results-dir "$RESULTS_DIR")
    [[ -n "$SEED" ]] && cmd+=(--seed "$SEED")
    [[ "$SINGLE_BOS" == 1 ]] && cmd+=(--single-bos)
    [[ -n "$MAX_PROBLEMS" ]] && cmd+=(--max-problems "$MAX_PROBLEMS")
    local out="${RESULTS_DIR}/${MODEL_DIR}/${stem}.jsonl"
    local log="${LOG_DIR}/${LOG_TAG}_${stem}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "--- [$(ist)] ${stem} -> ${out}"
    echo "    CMD: ${cmd[*]}"
    [[ "$DRY_RUN" == 1 ]] && continue
    echo "    log: ${log}"
    "${cmd[@]}" 2>&1 | tee "$log"
    # the runner derives the file name itself; make sure it is the one we expect
    [[ -f "$out" ]] || { echo "FAIL: expected output $out not found" >&2; exit 6; }
    check_complete "$out" "$log"
    echo "--- [$(ist)] finished ${stem}"
  done
  echo ""
  echo "[$(ist)] all ${#CONFIGS[@]} configs done$([[ "$DRY_RUN" == 1 ]] && echo ' (DRY RUN)'). Evaluate locally:"
  echo "  uv run python -m bench.eval --results-file ${RESULTS_DIR}/${MODEL_DIR}/<config>.jsonl --dataset mbpp"
}
