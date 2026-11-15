#!/usr/bin/env bash
# msi_queue_topk_after_recipe.sh -- MSI queue launcher (Ministral LCB full1046 batch 0).
#
# Waits for the running ministral_recipe runner to exit, verifies the recipe
# finished (every batch-0 task_id present exactly once, all lines parseable),
# verifies the code (branch, clean tracked tree, optional ff-only pull, pinned
# sha256 of the runner + bench/generator_vllm.py, pinned ancestor commit,
# top_k configs present), waits for the GPU to be free (capped), runs two tiny
# top_k_15 smoke tests (FlashInfer sampler on, then off) into the stratified-60
# dir and deletes their output, then nohup-launches the full
# top_k_25 top_k_20 top_k_15 run (batch 0, no time limit), waits for its
# "Engine ready" line, and exits 0.
# Any failed gate -> "STOP" in the log and a non-zero exit; nothing launched.
#
# Start once on MSI:
#   cd ~/pless-for-coding && nohup bash scripts/msi_queue_topk_after_recipe.sh \
#       >/dev/null 2>&1 &
#   tail -f ~/pless-for-coding/logs/msi_queue_topk.log
#
# Dry run (prints mutating actions instead of doing them; read-only gates still
# run). Locally, point every external command at stubs, e.g.:
#   DRY_RUN=1 WAIT_PID=<pid> REPO=/tmp/fake_repo PYTHON_BIN=/tmp/stubs/python \
#     JSON_PYTHON=$(uv run python -c 'import sys;print(sys.executable)') \
#     NVIDIA_SMI_BIN=/tmp/stubs/nvidia-smi EXPECT_RUNNER_SHA256=any \
#     EXPECT_GENVLLM_SHA256=any PIN_COMMIT=any bash scripts/msi_queue_topk_after_recipe.sh
# Full stub-based tests: uv run pytest tests/test_msi_queue_topk_script.py
#
# Overridable env: WAIT_PID REPO BRANCH DRY_RUN POLL_SECONDS LOG_DIR PYTHON_BIN
#   JSON_PYTHON GIT_BIN NVIDIA_SMI_BIN PGREP_BIN FLOCK_BIN GPU_FREE_MIB CUDA_HOME_VAL
#   PULL GPU_WAIT_MAX_MINUTES LAUNCH_VERIFY_MINUTES PIN_COMMIT
#   EXPECT_RUNNER_SHA256 EXPECT_GENVLLM_SHA256 EXPECT_ATCODER EXPECT_LEETCODE
#   PIN_DIFF TIMEOUT_BIN NVSMI_TIMEOUT_SECONDS GIT_PULL_TIMEOUT_SECONDS
set -uo pipefail

main() {
WAIT_PID="${WAIT_PID:-59539}"
REPO="${REPO:-$HOME/pless-for-coding}"
BRANCH="${BRANCH:-feat/pless-variants-mps}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == "1" ]]; then POLL_SECONDS="${POLL_SECONDS:-1}"; else POLL_SECONDS="${POLL_SECONDS:-60}"; fi
LOG_DIR="${LOG_DIR:-$REPO/logs}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
JSON_PYTHON="${JSON_PYTHON:-$REPO/.venv/bin/python}"   # only for count/field-printing one-liners
GIT_BIN="${GIT_BIN:-git}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
PGREP_BIN="${PGREP_BIN:-pgrep}"
FLOCK_BIN="${FLOCK_BIN:-$(command -v flock || true)}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
CUDA_HOME_VAL="${CUDA_HOME_VAL:-/usr/local/cuda-13.1}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
NVSMI_TIMEOUT_SECONDS="${NVSMI_TIMEOUT_SECONDS:-60}"
GIT_PULL_TIMEOUT_SECONDS="${GIT_PULL_TIMEOUT_SECONDS:-300}"
# Diff pin: HEAD must not differ from PIN_COMMIT in bench/ (except bench/eval)
# or the runner. Only the exact value "any" disables it (loudly).
PIN_DIFF="${PIN_DIFF:-on}"
PULL="${PULL:-0}"                                  # 1 = git pull --ff-only before the pin gate
GPU_WAIT_MAX_MINUTES="${GPU_WAIT_MAX_MINUTES:-30}" # cap on GPU-idle wait once the runner is gone
LAUNCH_VERIFY_MINUTES="${LAUNCH_VERIFY_MINUTES:-25}"
# Code pins -- reviewed at eff5f26 ("Ministral LCB runner: add pure top-k
# configs (top_k_25/20/15) via native vLLM sampling"). Hashes computed with
# `shasum -a 256` on the files in a worktree at eff5f26 (identical to
# `git show eff5f26:<path> | shasum -a 256`). Set to "any" to disable (loudly).
PIN_COMMIT="${PIN_COMMIT:-eff5f26}"
EXPECT_RUNNER_SHA256="${EXPECT_RUNNER_SHA256:-c9cbc712afe93bb4311b668bc56ec25932c5b1727480560978634b39015421b8}"
EXPECT_GENVLLM_SHA256="${EXPECT_GENVLLM_SHA256:-d89df1e94ff564cb4f9b51b68276d377e0508267a1e6d8a9ef2224ab3f064410}"
EXPECT_ATCODER="${EXPECT_ATCODER:-121}"
EXPECT_LEETCODE="${EXPECT_LEETCODE:-90}"

MODEL="mistralai/Ministral-3-3B-Reasoning-2512"
MODEL_SLUG="mistralai--Ministral-3-3B-Reasoning-2512"
RUNNER="scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
GENVLLM="bench/generator_vllm.py"
FULL_DIR="$REPO/results/pless_variants_cuda/$MODEL_SLUG/LCB_v6_full1046"
BATCH_CSV="$REPO/results/lcb_v6_full1046/batch_assignment.csv"   # cols: task_id,platform,difficulty,batch
# Non --full-dataset runs write to OUT_ROOT_TEMPLATE in the runner:
#   "results/pless_variants_cuda/{model_slug}/LCB_v6_stratified60"
SMOKE_DIR="$REPO/results/pless_variants_cuda/$MODEL_SLUG/LCB_v6_stratified60"
RECIPE_ATCODER="$FULL_DIR/ministral_recipe_atcoder_think_t0.7.jsonl"
RECIPE_LEETCODE="$FULL_DIR/ministral_recipe_leetcode_think_t0.7.jsonl"
RECIPE_LOG="$LOG_DIR/ministral_fulldataset_batch0_ministral_recipe.log"
# Smoke task: first row of results/lcb_v6_stratified60/task_selection.csv
# (abc304_b, atcoder, easy) -- the runner's own stratified-60 selection read by
# _load_task_selection(). Only the atcoder platform matches it, so the leetcode
# pass writes nothing.
SMOKE_TASK_ID="abc304_b"
SMOKE_SELECTION_CSV="$REPO/results/lcb_v6_stratified60/task_selection.csv"
FULL_LOG="$LOG_DIR/ministral_fulldataset_batch0_topk.log"
QLOG="$LOG_DIR/msi_queue_topk.log"
LOCKFILE="$LOG_DIR/msi_queue_topk.lock"
TOPK_RUNNER_PAT="run_pless_variants_ministral_3b_lcb_cuda.py.*--configs top_k_"
RECIPE_RUNNER_PAT="run_pless_variants_ministral_3b_lcb_cuda.py.*--configs ministral_recipe"

mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR" >&2; exit 2; }

# ---------------------------------------------------------------- 1. lock
if [[ -n "$FLOCK_BIN" ]]; then
  exec 9>>"$LOCKFILE"
  "$FLOCK_BIN" -n 9 || stop "another launcher instance holds $LOCKFILE (flock)"
else
  # Fallback (no flock): atomic noclobber pidfile; a dead holder is stale.
  if ! ( set -o noclobber; echo "$$" > "$LOCKFILE" ) 2>/dev/null; then
    local holder; holder="$(cat "$LOCKFILE" 2>/dev/null || true)"
    if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
      stop "another launcher instance (pid $holder) holds $LOCKFILE (pidfile)"
    fi
    log "stale pidfile lock (holder '$holder' not alive); taking it over"
    echo "$$" > "$LOCKFILE"
  fi
  trap '[[ "$(cat "$LOCKFILE" 2>/dev/null)" == "$$" ]] && rm -f "$LOCKFILE"' EXIT
fi

log "==== launcher start pid=$$ WAIT_PID=$WAIT_PID REPO=$REPO BRANCH=$BRANCH DRY_RUN=$DRY_RUN PULL=$PULL POLL_SECONDS=$POLL_SECONDS GPU_WAIT_MAX_MINUTES=$GPU_WAIT_MAX_MINUTES lock=${FLOCK_BIN:+flock}${FLOCK_BIN:-pidfile}"
cd "$REPO" || stop "cannot cd to REPO=$REPO"

if topk_running; then
  stop "a top-k runner is already running ($(proc_list "$TOPK_RUNNER_PAT")) -- refusing to start a second"
fi

# ---------------------------------------------------------------- 2a. wait for runner exit
local use_wait_pid=1 wargs
if pid_alive "$WAIT_PID"; then
  wargs="$(ps -o args= -p "$WAIT_PID" 2>/dev/null)"
  log "WAIT_PID $WAIT_PID is alive: ${wargs:0:300}"
  if [[ ! "$wargs" =~ $RECIPE_RUNNER_PAT ]]; then
    log "WARNING: WAIT_PID $WAIT_PID cmdline does not match /$RECIPE_RUNNER_PAT/ -- ignoring WAIT_PID, relying on the pattern wait only"
    use_wait_pid=0
  fi
else
  log "WAIT_PID $WAIT_PID is NOT alive at launcher start (completion gate will decide)"
fi
local n=0
while :; do
  n=$((n+1))
  if (( use_wait_pid )) && pid_alive "$WAIT_PID"; then
    # Guard against PID reuse / exec into another program while we wait.
    wargs="$(ps -o args= -p "$WAIT_PID" 2>/dev/null)"
    if [[ ! "$wargs" =~ $RECIPE_RUNNER_PAT ]]; then
      log "WARNING: WAIT_PID $WAIT_PID cmdline no longer matches /$RECIPE_RUNNER_PAT/ (now: '${wargs:0:200}') -- PID reused/replaced; no longer waiting on WAIT_PID, relying on the pattern wait only"
      use_wait_pid=0
      continue
    fi
    (( n % 30 == 1 )) && log "[recipe] waiting: WAIT_PID $WAIT_PID alive"
    sleep "$POLL_SECONDS"; continue
  fi
  if "$PGREP_BIN" -f "$RECIPE_RUNNER_PAT" >/dev/null 2>&1; then
    (( n % 30 == 1 )) && log "[recipe] waiting: recipe runner process still present ($(proc_list "$RECIPE_RUNNER_PAT"))"
    sleep "$POLL_SECONDS"; continue
  fi
  break
done
log "[recipe] runner gone (WAIT_PID dead/ignored, no '--configs ministral_recipe' process)"

# ---------------------------------------------------------------- 3. completion gate
local raw_at raw_lc
raw_at="$(count_lines "$RECIPE_ATCODER")"; raw_lc="$(count_lines "$RECIPE_LEETCODE")"
log "recipe raw wc -l: atcoder=$raw_at leetcode=$raw_lc (expect $EXPECT_ATCODER / $EXPECT_LEETCODE)"
local cg_out cg_rc
cg_out="$("$JSON_PYTHON" -c '
import csv, json, sys
csv_path, at_path, lc_path, exp_at, exp_lc = sys.argv[1:6]
want = {"atcoder": set(), "leetcode": set()}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row["batch"].strip() == "0" and row["platform"] in want:
            want[row["platform"]].add(row["task_id"])
ok = True
for plat, path, exp in (("atcoder", at_path, int(exp_at)), ("leetcode", lc_path, int(exp_lc))):
    ids, bad = [], 0
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue  # runner never writes blank lines; tolerate them
                try:
                    ids.append(str(json.loads(line)["task_id"]))
                except Exception:
                    bad += 1
    except FileNotFoundError:
        print(f"{plat}: FILE MISSING {path}"); ok = False; continue
    seen, dups = set(), []
    for t in ids:
        if t in seen and t not in dups: dups.append(t)
        seen.add(t)
    missing = sorted(want[plat] - seen)
    extra = sorted(seen - want[plat])
    print(f"{plat}: batch0_expected={len(want[plat])} (sanity {exp}) distinct={len(seen)} "
          f"parsed_lines={len(ids)} unparseable={bad} duplicated={len(dups)} "
          f"missing={len(missing)} not_in_batch0={len(extra)}")
    if missing: print(f"  missing (<=10): {missing[:10]}")
    if dups: print(f"  duplicated (<=10): {sorted(dups)[:10]}")
    if extra: print(f"  WARNING not in batch 0 (<=10): {extra[:10]}")
    if len(want[plat]) != exp or missing or dups or bad:
        ok = False
sys.exit(0 if ok else 1)
' "$BATCH_CSV" "$RECIPE_ATCODER" "$RECIPE_LEETCODE" "$EXPECT_ATCODER" "$EXPECT_LEETCODE" 2>&1)"
cg_rc=$?
printf '%s\n' "$cg_out" | sed 's/^/    /' | tee -a "$QLOG"
if (( cg_rc != 0 )); then
  log "last 20 lines of $RECIPE_LOG:"
  if [[ -f "$RECIPE_LOG" ]]; then
    tail -n 20 "$RECIPE_LOG" | sed 's/^/    | /' | tee -a "$QLOG"
    log "last 10 'crashed' lines of $RECIPE_LOG:"
    grep -n crashed "$RECIPE_LOG" | tail -n 10 | sed 's/^/    | /' | tee -a "$QLOG"
  else
    log "    (log file missing)"
  fi
  stop "ministral_recipe did NOT finish cleanly (raw wc -l atcoder=$raw_at leetcode=$raw_lc; see per-platform check above). Not launching top-k. Recipe is resumable; user decides."
fi
log "completion gate PASSED"

# ---------------------------------------------------------------- 4. code gate
local cur_branch dirty HEAD_SHA got_r got_g
cur_branch="$("$GIT_BIN" -C "$REPO" branch --show-current 2>&1)" || stop "git branch failed: $cur_branch"
[[ "$cur_branch" == "$BRANCH" ]] || stop "branch is '$cur_branch', expected '$BRANCH'"
dirty="$("$GIT_BIN" -C "$REPO" status --porcelain --untracked-files=no 2>&1)" || stop "git status failed: $dirty"
[[ -z "$dirty" ]] || stop "tracked files are dirty:
$dirty"
log "branch=$cur_branch, tracked tree clean; HEAD: $("$GIT_BIN" -C "$REPO" rev-parse HEAD)"
if [[ "$PULL" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: would run: $GIT_BIN -C $REPO pull --ff-only origin $BRANCH"
  else
    log "run: $GIT_BIN -C $REPO pull --ff-only origin $BRANCH"
    GIT_TERMINAL_PROMPT=0 "$TIMEOUT_BIN" "$GIT_PULL_TIMEOUT_SECONDS" \
      "$GIT_BIN" -C "$REPO" pull --ff-only origin "$BRANCH" >>"$QLOG" 2>&1 \
      || stop "git pull --ff-only origin $BRANCH failed (see above); no merge/rebase attempted"
  fi
else
  log "pull skipped (PULL=0); using the checked-out code as-is, pinned below"
fi
HEAD_SHA="$("$GIT_BIN" -C "$REPO" rev-parse HEAD)"
if [[ "$PIN_COMMIT" == "any" ]]; then
  log "WARNING: PIN_COMMIT=any -- ancestor pin DISABLED"
else
  "$GIT_BIN" -C "$REPO" merge-base --is-ancestor "$PIN_COMMIT" HEAD >/dev/null 2>&1 \
    || stop "pinned commit $PIN_COMMIT is not an ancestor of HEAD $HEAD_SHA"
  log "pin: $PIN_COMMIT is an ancestor of HEAD $HEAD_SHA"
fi
got_r="$(sha256_of "$REPO/$RUNNER")"; got_g="$(sha256_of "$REPO/$GENVLLM")"
check_pin "$RUNNER" "$EXPECT_RUNNER_SHA256" "$got_r"
check_pin "$GENVLLM" "$EXPECT_GENVLLM_SHA256" "$got_g"
if [[ "$PIN_DIFF" == "any" ]]; then
  log "WARNING: PIN_DIFF=any -- diff pin (bench/ minus bench/eval, runner) vs $PIN_COMMIT DISABLED"
elif [[ "$PIN_COMMIT" == "any" ]]; then
  log "WARNING: diff pin skipped because PIN_COMMIT=any (nothing to diff against)"
else
  local dpaths=( bench ':(exclude)bench/eval' "$RUNNER" )
  if ! "$GIT_BIN" -C "$REPO" diff --quiet "$PIN_COMMIT" HEAD -- "${dpaths[@]}"; then
    log "diff vs $PIN_COMMIT in pinned paths:"
    "$GIT_BIN" -C "$REPO" diff --stat "$PIN_COMMIT" HEAD -- "${dpaths[@]}" 2>&1 | sed 's/^/    | /' | tee -a "$QLOG"
    stop "HEAD differs from pinned $PIN_COMMIT in bench/ (excl. bench/eval) or $RUNNER"
  fi
  log "pin: no diff vs $PIN_COMMIT in bench/ (excl. bench/eval) + $RUNNER"
fi
local c
for c in top_k_25 top_k_20 top_k_15; do
  grep -q "\"$c\"" "$REPO/$RUNNER" || stop "runner lacks config \"$c\" -- old code, refusing to launch"
done
log "code gate PASSED: HEAD=$HEAD_SHA runner_sha256=$got_r generator_vllm_sha256=$got_g"

# ---------------------------------------------------------------- 5. smoke tests
[[ -f "$SMOKE_SELECTION_CSV" ]] || stop "stratified-60 selection file missing: $SMOKE_SELECTION_CSV"
grep -q "^$SMOKE_TASK_ID," "$SMOKE_SELECTION_CSV" || stop "smoke task $SMOKE_TASK_ID not in $SMOKE_SELECTION_CSV"
log "smoke output dir (runner OUT_ROOT_TEMPLATE, no --full-dataset): $SMOKE_DIR"
local pre
shopt -s nullglob
pre=( "$SMOKE_DIR"/top_k_15_*_think_t1.0.jsonl )
shopt -u nullglob
(( ${#pre[@]} == 0 )) || stop "pre-existing smoke-target file(s) found, will not overwrite/delete: ${pre[*]}"

local smoke_fail=0 fis
for fis in 1 0; do
  wait_gpu_idle "pre-smoke-fi$fis"
  smoke_once "$fis" || { smoke_fail=1; log "smoke FAILED (VLLM_USE_FLASHINFER_SAMPLER=$fis)"; }
  # Delete after EACH run: the runner resumes by task_id, so a leftover file
  # from run 1 would make run 2 skip the task and test nothing.
  cleanup_smoke
  (( smoke_fail == 0 )) || break
done
(( smoke_fail == 0 )) || stop "smoke test failed -- full run NOT launched"
log "smoke tests PASSED (flashinfer=1 and =0)"

# ---------------------------------------------------------------- 6. full run
wait_gpu_idle "pre-full"
topk_running && stop "a top-k runner appeared meanwhile -- refusing to launch a second"
local f
for c in top_k_25 top_k_20 top_k_15; do
  for f in "$FULL_DIR/${c}_"*"_think_t1.0.jsonl"; do
    [[ -f "$f" ]] && log "pre-existing full1046 file (will be RESUMED by task_id): $f records=$(count_lines "$f")"
  done
done
local TS LDIR
TS="$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S)"
LDIR="$LOG_DIR/topk_launch_$TS"
local CMD=( "$PYTHON_BIN" "$RUNNER" --model "$MODEL" --dtype bfloat16 --gpu-memory-utilization 0.92
      --max-model-len 37500 --n-samples 10 --max-new-tokens 35000
      --configs top_k_25 top_k_20 top_k_15 --full-dataset --batch 0
      --budget-minutes inf --per-config-minutes inf --seed 42 )
local ENVV=( CUDA_HOME="$CUDA_HOME_VAL" PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=1 )
if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY: would write $LDIR/{cmdline.txt,environ.txt,resume_topk.sh}"
  log "DRY: would run: cd $REPO && nohup env ${ENVV[*]} ${CMD[*]} >> $FULL_LOG 2>&1 &"
  log "DRY: would wait <= ${LAUNCH_VERIFY_MINUTES} min for a new '^Engine ready in' line in $FULL_LOG"
  log "==== DRY RUN complete (exit 0)"; exit 0
fi
mkdir -p "$LDIR" || stop "cannot create $LDIR"
{
  printf 'cd %q && ' "$REPO"; printf 'nohup env'; printf ' %q' "${ENVV[@]}" "${CMD[@]}"
  printf ' >> %q 2>&1 &\n' "$FULL_LOG"
} > "$LDIR/cmdline.txt"
cat > "$LDIR/resume_topk.sh" <<EOF
#!/usr/bin/env bash
# Resume the Ministral full1046 batch-0 top-k run (resume is by task_id).
# Generated by scripts/msi_queue_topk_after_recipe.sh at $(ist); HEAD=$HEAD_SHA
set -u
if pgrep -f '$TOPK_RUNNER_PAT' >/dev/null; then echo "top-k runner already running; refusing" >&2; exit 1; fi
$(cat "$LDIR/cmdline.txt")
echo "launched PID \$!"
EOF
chmod +x "$LDIR/resume_topk.sh"
local log_off=0
[[ -f "$FULL_LOG" ]] && log_off="$(wc -c < "$FULL_LOG" | tr -d ' ')"
log "launch: $(cat "$LDIR/cmdline.txt")   (FULL_LOG byte offset before launch: $log_off)"
# 9>&- : do not let the long-running child inherit the flock fd.
nohup env "${ENVV[@]}" "${CMD[@]}" >>"$FULL_LOG" 2>&1 9>&- &
local PID=$!
log "full top-k runner started: PID=$PID; waiting <= ${LAUNCH_VERIFY_MINUTES} min for '^Engine ready in' (runner prints: Engine ready in <N>s)"
# Fallback environ.txt now; replaced from /proc after 'Engine ready' (by then
# nohup/env have exec'd into the python process, so /proc shows python, not the shell).
{ printf '%s\n' "${ENVV[@]}"; echo "# (fallback: explicitly set vars; full launcher env follows)"; env; } > "$LDIR/environ.txt"
{ echo "HEAD=$HEAD_SHA"; echo "PID=$PID"; } >> "$LDIR/cmdline.txt"
local vmax t0 ready lp
vmax="$(minutes_to_seconds "$LAUNCH_VERIFY_MINUTES")"; t0=$SECONDS
lp=$(( POLL_SECONDS < 10 ? POLL_SECONDS : 10 ))
while :; do
  ready="$(tail -c +$((log_off+1)) "$FULL_LOG" 2>/dev/null | grep -m1 '^Engine ready in' || true)"
  if [[ -n "$ready" ]]; then
    log "launch verified: '$ready' (PID=$PID log=$FULL_LOG launch_dir=$LDIR)"
    if pid_alive "$PID" && [[ -r "/proc/$PID/environ" ]]; then
      tr '\0' '\n' < "/proc/$PID/environ" > "$LDIR/environ.txt"
      { printf 'proc_cmdline: '; tr '\0' ' ' < "/proc/$PID/cmdline"; echo; } >> "$LDIR/cmdline.txt"
      log "captured /proc/$PID/{environ,cmdline} into $LDIR"
    else
      log "no /proc/$PID (non-Linux or process gone); kept fallback environ.txt"
    fi
    break
  fi
  if ! pid_alive "$PID"; then
    log "FAIL: PID $PID died before 'Engine ready'. New FULL_LOG tail:"
    tail -c +$((log_off+1)) "$FULL_LOG" | tail -n 20 | sed 's/^/    | /' | tee -a "$QLOG"
    exit 1
  fi
  if (( SECONDS - t0 >= vmax )); then
    log "FAIL: no 'Engine ready' within ${LAUNCH_VERIFY_MINUTES} min (PID $PID still alive, NOT killed). New FULL_LOG tail:"
    tail -c +$((log_off+1)) "$FULL_LOG" | tail -n 20 | sed 's/^/    | /' | tee -a "$QLOG"
    exit 1
  fi
  sleep "$lp"
done
log "==== launcher done (exit 0); not waiting for the run"
exit 0
}

# ======================================================================= helpers
ist() { TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST'; }
log() { printf '[%s] %s\n' "$(ist)" "$*" | tee -a "$QLOG"; }
stop() { log "STOP: $*"; log "launcher exiting non-zero; nothing further launched"; exit 1; }
# act: mutating action; printed instead of executed when DRY_RUN=1.
act() {
  if [[ "$DRY_RUN" == "1" ]]; then log "DRY: would run: $*"; return 0; fi
  log "run: $*"; "$@"
}
# alive = kill -0 succeeds AND not a zombie (a zombie still answers kill -0).
pid_alive() { kill -0 "$1" 2>/dev/null || return 1; [[ "$(ps -o stat= -p "$1" 2>/dev/null)" != Z* ]]; }
topk_running() { "$PGREP_BIN" -f "$TOPK_RUNNER_PAT" >/dev/null 2>&1; }
proc_list() {  # "pid args; pid args" for processes matching $1
  local p out=""
  for p in $("$PGREP_BIN" -f "$1" 2>/dev/null); do
    out+="$p $(ps -o args= -p "$p" 2>/dev/null | cut -c1-200); "
  done
  printf '%s' "${out:-none}"
}
count_lines() { if [[ -f "$1" ]]; then wc -l < "$1" | tr -d ' '; else echo "MISSING"; fi; }
minutes_to_seconds() { awk -v m="$1" 'BEGIN{printf "%d", m*60}'; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
check_pin() {  # $1 path, $2 expected, $3 actual
  if [[ "$2" == "any" ]]; then
    log "WARNING: sha256 pin for $1 DISABLED (expected=any); actual=$3"; return 0
  fi
  [[ "$2" == "$3" ]] || stop "sha256 mismatch for $1: expected=$2 actual=$3"
  log "pin: sha256 $1 = $3 (matches expected)"
}
gpu_used_mib() {  # max memory.used over all GPUs; empty on failure or timeout
  local out rc
  out="$("$TIMEOUT_BIN" "$NVSMI_TIMEOUT_SECONDS" "$NVIDIA_SMI_BIN" \
          --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; rc=$?
  if (( rc != 0 )); then
    (( rc == 124 )) && log "nvidia-smi timed out after ${NVSMI_TIMEOUT_SECONDS}s" >&2 \
                    || log "nvidia-smi exited rc=$rc" >&2
    return 0
  fi
  printf '%s\n' "$out" | awk 'BEGIN{m=""} /^[[:space:]]*[0-9]+/{v=$1+0; if(m==""||v>m)m=v} END{print m}'
}
wait_gpu_idle() {  # $1 label. Waits for no EngineCore + GPU mem < threshold, capped.
  local label="$1" fails=0 n=0 mem cap t0=$SECONDS why
  cap="$(minutes_to_seconds "$GPU_WAIT_MAX_MINUTES")"
  while :; do
    n=$((n+1)); why=""
    mem="$(gpu_used_mib)"
    if [[ -z "$mem" ]]; then
      fails=$((fails+1)); log "[$label] nvidia-smi query failed ($fails/5)"
      (( fails >= 5 )) && stop "[$label] nvidia-smi failed 5 times in a row"
      sleep "$POLL_SECONDS"; continue
    fi
    fails=0
    if "$PGREP_BIN" -f "VLLM::EngineCore" >/dev/null 2>&1; then
      why="VLLM::EngineCore still present ($(proc_list 'VLLM::EngineCore'))"
    elif (( mem >= GPU_FREE_MIB )); then
      why="GPU memory.used=${mem} MiB >= ${GPU_FREE_MIB}"
    fi
    if [[ -z "$why" ]]; then
      log "[$label] GPU idle: no EngineCore, memory.used=${mem} MiB < ${GPU_FREE_MIB}"
      return 0
    fi
    if (( SECONDS - t0 >= cap )); then
      stop "[$label] GPU still held after ${GPU_WAIT_MAX_MINUTES} min: $why; EngineCore procs: $(proc_list 'VLLM::EngineCore'); nvidia-smi memory.used=${mem} MiB"
    fi
    (( n % 30 == 1 )) && log "[$label] waiting: $why"
    sleep "$POLL_SECONDS"
  done
}
smoke_once() {  # $1 = VLLM_USE_FLASHINFER_SAMPLER value; returns 0 on pass
  local fis="$1" slog rc ok=0 f le
  slog="$LOG_DIR/topk_smoke_$(date +%Y%m%d_%H%M%S)_flashinfer${fis}.log"
  log "smoke (VLLM_USE_FLASHINFER_SAMPLER=$fis) -> $slog"
  act env CUDA_HOME="$CUDA_HOME_VAL" PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER="$fis" \
    "$PYTHON_BIN" "$RUNNER" --model "$MODEL" --dtype bfloat16 --gpu-memory-utilization 0.92 \
    --max-model-len 37500 --n-samples 2 --max-new-tokens 512 --configs top_k_15 \
    --only-task-ids "$SMOKE_TASK_ID" --budget-minutes inf --per-config-minutes inf --seed 42 \
    >"$slog" 2>&1 9>&-
  rc=$?
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: would assert exit 0, \"'register_pless_logitsproc': False\" in Loading-engine line, record fields"
    return 0
  fi
  log "smoke exit code: $rc"; (( rc == 0 )) || ok=1
  # Runner prints: Loading engine: <model> (kwargs={..., 'register_pless_logitsproc': False})
  le="$(grep -m1 '^Loading engine:' "$slog" || true)"
  log "engine line: $le"
  if [[ "$le" != *"'register_pless_logitsproc': False"* ]]; then log "FAIL: custom logits processor not disabled"; ok=1; fi
  f="$SMOKE_DIR/top_k_15_atcoder_think_t1.0.jsonl"
  if [[ -f "$f" ]]; then
    # NOTE: the record's top_k comes from the runner's _record_top_k() lookup
    # table (_NATIVE_TOP_P_TOP_K), not from vLLM; this check proves plumbing
    # (config routing, record schema, 2 samples written) only, NOT that vLLM
    # actually sampled with top_k=15.
    "$JSON_PYTHON" -c '
import json,sys
recs=[json.loads(l) for l in open(sys.argv[1])]
bad=0
print("n_records", len(recs))
for r in recs:
    f={k:r.get(k) for k in ("task_id","top_k","backend","sampler_name","temperature","top_p")}
    f["n_samples"]=len(r.get("samples") or [])
    print(f)
    bad |= not (f["top_k"]==15 and f["backend"]=="vllm" and f["sampler_name"]=="top_k_15" and f["n_samples"]==2 and f["task_id"]==sys.argv[2])
sys.exit(1 if bad or len(recs)!=1 else 0)' "$f" "$SMOKE_TASK_ID" 2>&1 | tee -a "$QLOG"
    (( PIPESTATUS[0] == 0 )) || { log "FAIL: record field check"; ok=1; }
  else
    log "FAIL: no smoke output $f"; ok=1
  fi
  return $ok
}
cleanup_smoke() {
  shopt -s nullglob
  local files=( "$SMOKE_DIR"/top_k_15_*_think_t1.0.jsonl )
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then log "no smoke output files to delete"; return 0; fi
  log "deleting smoke output (created by this launcher; verified absent before): ${files[*]}"
  act rm -f -- "${files[@]}"
}

# Whole body is parsed before anything runs, so editing this file mid-run
# (e.g. a git pull) cannot change what the running launcher executes.
main "$@"; exit $?
