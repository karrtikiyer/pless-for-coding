# scripts/mbpp_staging_common.sh — sourced (not executed) by the 2026-11 MBPP staging scripts:
#   run_bigcode_mbpp_qwen25coder7b.sh, run_chat_mbpp_qwen25coder7binstruct.sh
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
#
# Stops on the first failing config (set -euo pipefail + pipefail through tee). bench/runner.py catches
# per-task exceptions and keeps going ("Error on task_id=..."); this helper turns any such line, or a file
# that is not exactly 500 unique task_ids on a full run, into a failure.
# Evaluation is NOT run here: run `python -m bench.eval` locally (project CLAUDE.md gotcha 9).

set -euo pipefail

: "${MODEL_ID:?}" "${RESULTS_DIR:?}" "${LOG_TAG:?}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="logs"
MODEL_DIR="${MODEL_ID//\//--}"
EXPECTED_TASKS=500

if [[ -n "$MAX_PROBLEMS" && ! "$MAX_PROBLEMS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PROBLEMS must be a positive integer; got '$MAX_PROBLEMS'" >&2
  exit 2
fi

ist() { TZ=Asia/Kolkata date '+%F %T IST'; }

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
  mkdir -p "$LOG_DIR"
  [[ "$DRY_RUN" == 1 ]] || check_env
  echo "[$(ist)] $MODEL_ID -> ${RESULTS_DIR}/${MODEL_DIR}/  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}," \
       "MAX_PROBLEMS=${MAX_PROBLEMS:-<all 500>}, commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
  local cfg stem args
  for cfg in "${CONFIGS[@]}"; do
    stem="${cfg%%|*}"
    read -ra args <<< "${cfg#*|}"
    local cmd=(uv run python -m bench --model "$MODEL_ID" "${args[@]}" ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}
               --mbpp-config full --results-dir "$RESULTS_DIR")
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
