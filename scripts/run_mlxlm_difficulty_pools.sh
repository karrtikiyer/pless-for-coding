#!/bin/bash
# Relaunch scripts/run_mlxlm_difficulty_pools.py once per phase (easy ->
# medium -> hard), so mlx's allocator resets between phases. The python
# script runs one phase and exits; resume skips finished tasks/samples.
# Usage (on the 96GB Mac, from the repo root):
#   nohup caffeinate -i bash scripts/run_mlxlm_difficulty_pools.sh > /tmp/mlxlm_pools_wrapper.out 2>&1 < /dev/null & disown
set -uo pipefail  # NOT -e: one crashed phase must not kill the wrapper

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/scripts/run_mlxlm_difficulty_pools.py"
LOG="${MLXLM_POOLS_LOG:-/tmp/mlxlm_difficulty_pools.log}"
OUT="${MLXLM_POOLS_OUT_DIR:-$REPO/results/pless_variants_mlxlm_pools/Qwen--Qwen3-8B/LCB_v6_stratified60}"

MAX_ATTEMPTS=40  # 3 phases + crash-retries: medium 16 / hard 10 may OOM; each crash resumes from banked samples
last_progress=-1
no_progress=0

progress() {
    # full records x10 + banked samples: monotonic forward-progress proxy
    local d b
    d=$(cat "$OUT"/pless_default_*_think_t1.0.jsonl 2>/dev/null | wc -l | tr -d ' ')
    b=$(cat "$OUT"/_partial/*.partial.jsonl 2>/dev/null | wc -l | tr -d ' ')
    echo $(( d * 10 + b ))
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[$(date)] Attempt $attempt/$MAX_ATTEMPTS" | tee -a "$LOG"
    "$RUN_PY" "$SCRIPT" >> "$LOG" 2>&1
    code=$?
    echo "[$(date)] Process exited with code $code." | tee -a "$LOG"

    if [ "$code" -eq 0 ] && tail -n 3 "$LOG" | grep -q "ALL_PHASES_DONE"; then
        echo "[$(date)] All phases done -- stopping wrapper." | tee -a "$LOG"
        exit 0
    fi

    p=$(progress)
    if [ "$p" -le "$last_progress" ]; then
        no_progress=$((no_progress + 1))
    else
        no_progress=0
    fi
    last_progress=$p
    if [ "$no_progress" -ge 2 ]; then
        echo "[$(date)] STOPPING: no progress across 2 attempts (proxy=$p) -- investigate." | tee -a "$LOG"
        exit 1
    fi
done
echo "[$(date)] STOPPING: reached MAX_ATTEMPTS=$MAX_ATTEMPTS -- investigate." | tee -a "$LOG"
exit 1
