#!/usr/bin/env bash
# gpu2_wait_then_launch_cf748_deepseek.sh -- unattended GPU2 waiter (todos A76 -> A77): waits for the
# seeded 2-GPU Qwen3-8B CF-748 pless_effk_nofloor run to finish CLEANLY, then starts the DeepSeek run
# ONCE with scripts/gpu2_launch_cf748_effk_nofloor.sh:
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B MAX_MODEL_LEN=40960 EXPECT_HEAD=<HEAD at waiter start>
#   EXPECT_RUNNER_SHA256=<pin> bash scripts/gpu2_launch_cf748_effk_nofloor.sh
# The launcher is run in the FOREGROUND (GPU2_LAUNCHER_DETACHED=1; this waiter is itself detached) so its
# real exit code and output are captured; all of the launcher's own gates still apply.
#
# Every POLL_SECONDS (default 300):
#   STOP file present (default logs/cf748_deepseek_autostart.STOP)  -> exit 0, nothing launched
#   per Qwen half: state = LAST "WRAPPER START|DONE|FAILED|ABORT" marker in its wrapper log
#     (logs are appended across restarts, so only the last marker counts)
#   either half FAILED/ABORT                                           -> ABORT (exit 3), nothing launched
#   "Traceback" in a python log: WARNING only (file, count, first line no.); the authoritative gate is
#     both halves DONE rc=0 + the --check-only completeness check
#   heartbeat (record counts of both halves) at least every HEARTBEAT_SECONDS (default 3600)
#   both halves "WRAPPER DONE rc=0"                                    -> continue below
#   MAX_WAIT_HOURS (default 96) exceeded                               -> ABORT (exit 3)
# Then: cf748_merge_validate.py --check-only (each half == its split ids, 10 samples, no duplicates,
#   union == 748) else ABORT; wait up to PROC_CLEAR_TIMEOUT_SECONDS (default 900, poll PROC_POLL_SECONDS=30)
#   until no runner / run script / wrapper / VLLM::EngineCore process remains and both GPUs are below
#   GPU_FREE_MIB (the launcher's own gates), else ABORT; STOP file + abort conditions + the full preflight
#   (branch, clean tree, HEAD == EXPECT_HEAD, runner pin, files) re-checked -- failure = ABORT, no marker,
#   restart needed; marker logs/cf748_deepseek_autostart.launched written (exclusive) -> launcher run with
#   only MODEL/MAX_MODEL_LEN/EXPECT_HEAD/EXPECT_RUNNER_SHA256/REPO/BRANCH set and the launcher's other knobs
#   (paths, bins, PGREP_SCOPE, SMOKE/DRY_RUN, ...) UNSET; its exit code and output tail are logged and become
#   this script's exit code. rc != 0 with the launcher's 'nothing launched' line, no 'launch incomplete' /
#   'already launched' and nothing busy -> marker removed (re-armed); otherwise the marker is KEPT.
# Safety: single instance (flock, else pidfile with liveness check); refuses to start if the .launched
#   marker exists (never launches twice -- delete it deliberately to re-arm); config validated at start
#   (exit 2; MODEL=Qwen/Qwen3-8B is rejected) and preflight (launcher, python, id files, branch, clean
#   tree, runner sha256 pin, EXPECT_HEAD == HEAD if given) at start.
#   DRY_RUN=1: one poll, prints what it would do, never launches, exit 0. No environment is dumped.
#
# Run on GPU2 (detached):
#   cd /home/karrtik/pless-for-coding && setsid nohup bash scripts/gpu2_wait_then_launch_cf748_deepseek.sh \
#     >/dev/null 2>&1 < /dev/null &
#   tail -f logs/cf748_deepseek_autostart.log ; touch logs/cf748_deepseek_autostart.STOP   # to cancel
# Tests: uv run pytest tests/test_gpu2_wait_then_launch_cf748_deepseek.py
#
# Overridable env: REPO BRANCH LOG_DIR LAUNCHER CHECK_PYTHON MODEL MAX_MODEL_LEN EXPECT_HEAD
#   EXPECT_RUNNER_SHA256 POLL_SECONDS HEARTBEAT_SECONDS MAX_WAIT_HOURS PROC_CLEAR_TIMEOUT_SECONDS
#   PROC_POLL_SECONDS GPU_FREE_MIB STOP_FILE DRY_RUN GIT_BIN PGREP_BIN NVIDIA_SMI_BIN (empty = skip GPU
#   memory check) TIMEOUT_BIN FLOCK_BIN. Test hook: PGREP_SCOPE (regex prefix for every pgrep pattern).
# Bash 3.2 compatible (macOS tests) -- no associative arrays, no ${x,,}, no mapfile.
set -uo pipefail

main() {
REPO="${REPO:-/home/karrtik/pless-for-coding}"
BRANCH="${BRANCH:-feat/pless-variants-mps}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
LAUNCHER="${LAUNCHER:-$REPO/scripts/gpu2_launch_cf748_effk_nofloor.sh}"
CHECK_PYTHON="${CHECK_PYTHON:-$HOME/vllm032_env/.venv/bin/python}"   # stdlib-only helper, run with -I
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
EXPECT_HEAD="${EXPECT_HEAD:-}"
EXPECT_RUNNER_SHA256="${EXPECT_RUNNER_SHA256:-d8177701bb2e53083e19c1aa79c93ff0fa89c1b0f086347e80f8047b2edcb48f}"
POLL_SECONDS="${POLL_SECONDS:-300}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-3600}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-96}"
PROC_CLEAR_TIMEOUT_SECONDS="${PROC_CLEAR_TIMEOUT_SECONDS:-900}"
PROC_POLL_SECONDS="${PROC_POLL_SECONDS:-30}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
STOP_FILE="${STOP_FILE:-$LOG_DIR/cf748_deepseek_autostart.STOP}"
DRY_RUN="${DRY_RUN:-0}"
GIT_BIN="${GIT_BIN:-git}"
PGREP_BIN="${PGREP_BIN:-pgrep}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN-nvidia-smi}"
TIMEOUT_BIN="${TIMEOUT_BIN-$(command -v timeout || true)}"
FLOCK_BIN="${FLOCK_BIN-$(command -v flock || true)}"
PGREP_SCOPE="${PGREP_SCOPE:-}"

WLOG="$LOG_DIR/cf748_deepseek_autostart.log"
LOCKFILE="$LOG_DIR/cf748_deepseek_autostart.lock"
LAUNCHED_MARKER="$LOG_DIR/cf748_deepseek_autostart.launched"
# The Qwen run being waited for (fixed: these are the A76 launcher's default paths for Qwen/Qwen3-8B).
QWEN_MODEL="Qwen/Qwen3-8B"
QRES="$REPO/results/_cf748_effk_nofloor_split"
QLOG_DIR="$REPO/logs"   # the A76 launcher's LOG_DIR default: wrapper logs cf748_effk_nofloor_gpu{0,1}.log
QSUB="Qwen--Qwen3-8B/CODEFORCES_interview"
ARM="pless_effk_nofloor"
IDS_FILE="$REPO/data/cf_interview_748_ids.txt"
IDS_EVEN="$REPO/data/cf748_split/cf748_even_ids.txt"
IDS_ODD="$REPO/data/cf748_split/cf748_odd_ids.txt"
MERGE_SCRIPT="$REPO/scripts/cf748_merge_validate.py"
RUNNER="$REPO/bench/apps/runner.py"
# Same patterns as the launcher's BUSY_PATS, plus the vLLM engine-core process title.
BUSY_PATS=( "$(scoped ' -m bench\.apps')" "$(scoped 'VLLM::EngineCore')" "$(scoped 'run_cf748_apps_qwen3\.sh')"
            "$(scoped 'run_pless_variants_')" "$(scoped 'cf748_effk_launch_.*/run_gpu[01]\.sh')" )
LAUNCHED=0

mkdir -p "$LOG_DIR" || { echo "cannot create LOG_DIR" >&2; exit 2; }

# ---------------------------------------------------------------- config (exit 2)
[[ "$MODEL" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] && [[ "$MODEL" != *..* ]] \
  || cfg_err "MODEL must match ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\$ without '..' (got $(printf '%q' "$MODEL"))"
[[ "$MODEL" != "$QWEN_MODEL" ]] || cfg_err "MODEL must not be the Qwen default $QWEN_MODEL (this waiter only starts the follow-up run)"
[[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || cfg_err "MAX_MODEL_LEN must be a positive integer"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || cfg_err "DRY_RUN must be 0 or 1"
local v
for v in POLL_SECONDS HEARTBEAT_SECONDS PROC_POLL_SECONDS; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || cfg_err "$v must be a positive integer (seconds)"
done
for v in PROC_CLEAR_TIMEOUT_SECONDS GPU_FREE_MIB; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || cfg_err "$v must be a non-negative integer"
done
[[ "$MAX_WAIT_HOURS" =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v h="$MAX_WAIT_HOURS" 'BEGIN{exit !(h>0)}' \
  || cfg_err "MAX_WAIT_HOURS must be a number > 0"
local MAX_WAIT_SECONDS; MAX_WAIT_SECONDS="$(awk -v h="$MAX_WAIT_HOURS" 'BEGIN{s=h*3600; i=int(s); if (i<s) i++; print i}')"
local want_pin; want_pin="$(printf '%s' "$EXPECT_RUNNER_SHA256" | tr 'A-F' 'a-f')"
[[ "$want_pin" =~ ^[0-9a-f]{64}$ ]] || cfg_err "EXPECT_RUNNER_SHA256 must be 64 hex chars ('any' is not allowed unattended)"
if [[ -n "$EXPECT_HEAD" ]]; then
  [[ "$EXPECT_HEAD" =~ ^[0-9a-f]{40}$ ]] || cfg_err "EXPECT_HEAD must be a full 40-hex commit sha"
fi

log "==== waiter start pid=$$ REPO=$REPO DRY_RUN=$DRY_RUN poll=${POLL_SECONDS}s heartbeat=${HEARTBEAT_SECONDS}s max_wait=${MAX_WAIT_HOURS}h proc_clear_timeout=${PROC_CLEAR_TIMEOUT_SECONDS}s stop_file=$STOP_FILE"

# ---------------------------------------------------------------- never twice; single instance
[[ -e "$LAUNCHED_MARKER" ]] && preflight_err "already launched once ($LAUNCHED_MARKER exists: $(head -c 300 "$LAUNCHED_MARKER" | tr '\n' ' ')); delete it deliberately to re-arm"
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ -n "$FLOCK_BIN" ]]; then
    exec 8>>"$LOCKFILE"
    "$FLOCK_BIN" -n 8 || preflight_err "another waiter instance holds $LOCKFILE (flock)"
  else
    if ! ( set -o noclobber; echo "$$" > "$LOCKFILE" ) 2>/dev/null; then
      local holder; holder="$(cat "$LOCKFILE" 2>/dev/null || true)"
      if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
        preflight_err "another waiter instance (pid $holder) holds $LOCKFILE (pidfile)"
      fi
      log "stale pidfile lock (holder '$holder' not alive); taking it over"
      echo "$$" > "$LOCKFILE"
    fi
    trap '[[ "$(cat "$LOCKFILE" 2>/dev/null)" == "$$" ]] && rm -f "$LOCKFILE"' EXIT
  fi
  log "lock: ${FLOCK_BIN:+flock}${FLOCK_BIN:-pidfile} $LOCKFILE"
fi
trap 'log "terminated by signal; launched=$LAUNCHED"; exit 143' TERM INT

# ---------------------------------------------------------------- preflight (exit 2)
local start_head; start_head="$("$GIT_BIN" -C "$REPO" rev-parse HEAD 2>/dev/null)" || preflight_err "git rev-parse HEAD failed"
if [[ -z "$EXPECT_HEAD" ]]; then
  EXPECT_HEAD="$start_head"
  log "EXPECT_HEAD=$EXPECT_HEAD (captured once at waiter start from git rev-parse HEAD)"
else
  [[ "$EXPECT_HEAD" == "$start_head" ]] || preflight_err "EXPECT_HEAD=$EXPECT_HEAD but HEAD at waiter start is $start_head"
  log "EXPECT_HEAD=$EXPECT_HEAD (from env; equals HEAD at waiter start)"
fi
preflight_checks || preflight_err "$PF_ERR"
log "preflight OK (launcher, python, id files, branch, clean tree, HEAD, runner pin $want_pin); will launch: MODEL=$MODEL MAX_MODEL_LEN=$MAX_MODEL_LEN EXPECT_HEAD=$EXPECT_HEAD EXPECT_RUNNER_SHA256=$want_pin bash $LAUNCHER (foreground, GPU2_LAUNCHER_DETACHED=1)"

# ---------------------------------------------------------------- wait for both halves DONE
local t0 now last_hb=-1 prev_states=""
t0="$(date +%s)"
while :; do
  check_stop
  poll_once
  [[ -n "$ABORT_REASON" ]] && abort "$ABORT_REASON"
  now="$(date +%s)"
  if (( last_hb < 0 || now - last_hb >= HEARTBEAT_SECONDS )) || [[ "$ST0/$ST1" != "$prev_states" ]]; then
    (( last_hb >= 0 )) && TB_FORCE=1 tb_repeat
    log "HEARTBEAT gpu0: state=$ST0 records=$N0 | gpu1: state=$ST1 records=$N1 (of 374 each) | waited $(( now - t0 ))s of ${MAX_WAIT_SECONDS}s"
    last_hb=$now; prev_states="$ST0/$ST1"
  fi
  [[ "$ST0" == DONE && "$ST1" == DONE ]] && break
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY: Qwen run not finished; would wait (poll every ${POLL_SECONDS}s) for 'WRAPPER DONE rc=0' in both wrapper logs, then check-only + process/GPU clear, then launch. Exit 0, nothing launched."
    exit 0
  fi
  (( now - t0 >= MAX_WAIT_SECONDS )) && abort "MAX_WAIT_HOURS=$MAX_WAIT_HOURS exceeded while waiting for the Qwen run (gpu0=$ST0 gpu1=$ST1)"
  sleep "$POLL_SECONDS"
done
log "both Qwen halves report 'WRAPPER DONE rc=0'"

# ---------------------------------------------------------------- completeness check
local chk rc
chk="$(cd "$REPO" && "$CHECK_PYTHON" -I "$MERGE_SCRIPT" --check-only \
        --in-a "$QRES/gpu0" --in-b "$QRES/gpu1" --split-a "$IDS_EVEN" --split-b "$IDS_ODD" \
        --ids-file "$IDS_FILE" --expected-count 748 --method "$ARM" --temperature 1.0 --n-samples 10 \
        --model "$QWEN_MODEL" 2>&1)"; rc=$?
if (( rc != 0 )); then
  [[ "$DRY_RUN" == 1 ]] && { log "DRY: would ABORT: completeness check (cf748_merge_validate.py --check-only) failed rc=$rc: $(oneline "$chk")"; exit 0; }
  abort "completeness check (cf748_merge_validate.py --check-only) failed rc=$rc: $(oneline "$chk")"
fi
log "completeness: $(oneline "$chk")"

# ---------------------------------------------------------------- processes + GPUs clear
local waited=0
while busy_now; do
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY: would wait up to ${PROC_CLEAR_TIMEOUT_SECONDS}s for leftover processes / GPU memory to clear: $BUSY_DESC"
    break
  fi
  (( waited >= PROC_CLEAR_TIMEOUT_SECONDS )) && abort "leftover processes / GPU memory did not clear within ${PROC_CLEAR_TIMEOUT_SECONDS}s: $BUSY_DESC"
  log "waiting for leftover processes / GPU memory to clear (${waited}s of ${PROC_CLEAR_TIMEOUT_SECONDS}s): $BUSY_DESC"
  sleep "$PROC_POLL_SECONDS"; waited=$(( waited + PROC_POLL_SECONDS ))
  check_stop
done
[[ "$DRY_RUN" == 1 ]] || log "no runner / wrapper / EngineCore process left; GPUs below ${GPU_FREE_MIB} MiB"

if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY: would launch: MODEL=$MODEL MAX_MODEL_LEN=$MAX_MODEL_LEN EXPECT_HEAD=$EXPECT_HEAD EXPECT_RUNNER_SHA256=$want_pin GPU2_LAUNCHER_DETACHED=1 bash $LAUNCHER. Exit 0, nothing launched."
  exit 0
fi

# ---------------------------------------------------------------- last checks, then launch ONCE
check_stop
poll_once
[[ -n "$ABORT_REASON" ]] && abort "$ABORT_REASON (re-check right before launch)"
[[ "$ST0" == DONE && "$ST1" == DONE ]] || abort "Qwen wrapper state changed right before launch (gpu0=$ST0 gpu1=$ST1)"
preflight_checks || abort "pre-launch re-check failed: $PF_ERR -- nothing launched, no .launched marker written; fix it and restart the waiter"
log "pre-launch re-check OK (branch, clean tree, HEAD == EXPECT_HEAD, runner pin, files)"
if ! ( set -o noclobber; printf 'launched by waiter pid %s at %s; EXPECT_HEAD=%s MODEL=%s\n' "$$" "$(ist)" "$EXPECT_HEAD" "$MODEL" > "$LAUNCHED_MARKER" ) 2>/dev/null; then
  abort "$LAUNCHED_MARKER appeared meanwhile -- refusing to launch twice"
fi
LAUNCHED=1
local out; out="$LOG_DIR/cf748_deepseek_autostart_launcher_$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S).out"
log "LAUNCH: MODEL=$MODEL MAX_MODEL_LEN=$MAX_MODEL_LEN EXPECT_HEAD=$EXPECT_HEAD EXPECT_RUNNER_SHA256=$want_pin bash $LAUNCHER (output: $out)"
local -a SCRUB=( -u PGREP_SCOPE -u RESULTS_ROOT -u SMOKE_RESULTS_ROOT -u LOG_DIR -u GPU_FREE_MIB -u STAGGER_SECONDS
                 -u IDS_FILE -u IDS_EVEN -u IDS_ODD -u SPLIT_DIR -u GIT_BIN -u NVIDIA_SMI_BIN -u PGREP_BIN -u TIMEOUT_BIN
                 -u SETSID_BIN -u FLOCK_BIN -u VLLM_VENV -u CHECK_PYTHON -u LAUNCH_PATH -u LAUNCH_CHECK_SECONDS
                 -u NVSMI_TIMEOUT_SECONDS -u EXPECT_VLLM_VERSION -u EXPECT_TORCH_VERSION -u EXPECT_FLASHINFER_VERSION
                 -u DRY_RUN -u SMOKE -u SMOKE_IDS -u SMOKE_GPU )
( cd "$REPO" && env "${SCRUB[@]}" GPU2_LAUNCHER_DETACHED=1 REPO="$REPO" BRANCH="$BRANCH" \
    MODEL="$MODEL" MAX_MODEL_LEN="$MAX_MODEL_LEN" EXPECT_HEAD="$EXPECT_HEAD" EXPECT_RUNNER_SHA256="$want_pin" \
    bash "$LAUNCHER" > "$out" 2>&1 < /dev/null 8>&- ); rc=$?
log "launcher exit code: $rc; output tail ($out):"
tail -n 40 "$out" 2>/dev/null | sed 's/^/    | /' | tee -a "$WLOG"
if (( rc != 0 )); then
  if grep -q 'launcher exiting non-zero; nothing launched' "$out" 2>/dev/null \
     && ! grep -qE 'already launched: |==== launch incomplete|WRAPPER_PID=' "$out" 2>/dev/null && ! busy_now; then
    rm -f "$LAUNCHED_MARKER"
    log "launcher reported 'nothing launched' and no runner/wrapper/EngineCore process or GPU memory is in use: removed $LAUNCHED_MARKER (re-armed; restart the waiter to retry)"
  else
    log "launcher rc=$rc and a launch may have (partly) happened (${BUSY_DESC:-launcher output does not say 'nothing launched'}): marker KEPT. Check the launcher log; rm $LAUNCHED_MARKER to re-arm only after confirming nothing of the new run is running"
  fi
fi
log "==== waiter done (launched once; launcher rc=$rc)"
exit "$rc"
}

# ======================================================================= helpers
ist() { TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST'; }
log() { printf '[%s] %s\n' "$(ist)" "$*" | tee -a "$WLOG"; }
cfg_err() { log "CONFIG ERROR: $*; nothing launched"; exit 2; }
preflight_err() { log "PREFLIGHT ERROR: $*; nothing launched"; exit 2; }
abort() { log "ABORT: $*; NOT launching the DeepSeek run (exit 3)"; exit 3; }
oneline() { printf '%s' "$1" | tr '\n' ' ' | cut -c1-1500; }
scoped() { if [[ -n "${PGREP_SCOPE:-}" ]]; then printf '%s.*%s' "$PGREP_SCOPE" "$1"; else printf '%s' "$1"; fi; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
preflight_checks() {  # sets PF_ERR on failure; used at start and immediately before launching
  local f br dt hd gp
  PF_ERR=""
  [[ -f "$LAUNCHER" ]] || { PF_ERR="LAUNCHER not found: $LAUNCHER"; return 1; }
  [[ -x "$CHECK_PYTHON" ]] || { PF_ERR="CHECK_PYTHON not executable: $CHECK_PYTHON"; return 1; }
  for f in "$MERGE_SCRIPT" "$IDS_FILE" "$IDS_EVEN" "$IDS_ODD" "$RUNNER"; do
    [[ -f "$f" ]] || { PF_ERR="required file missing: $f"; return 1; }
  done
  br="$("$GIT_BIN" -C "$REPO" branch --show-current 2>&1)" || { PF_ERR="git branch failed"; return 1; }
  [[ "$br" == "$BRANCH" ]] || { PF_ERR="branch is '$br', expected '$BRANCH' (the launcher would refuse)"; return 1; }
  dt="$("$GIT_BIN" -C "$REPO" status --porcelain --untracked-files=no 2>&1)" || { PF_ERR="git status failed"; return 1; }
  [[ -z "$dt" ]] || { PF_ERR="tracked files are dirty (the launcher would refuse): $(oneline "$dt")"; return 1; }
  hd="$("$GIT_BIN" -C "$REPO" rev-parse HEAD 2>/dev/null)" || { PF_ERR="git rev-parse HEAD failed"; return 1; }
  [[ "$hd" == "$EXPECT_HEAD" ]] || { PF_ERR="HEAD is $hd, expected EXPECT_HEAD=$EXPECT_HEAD"; return 1; }
  gp="$(sha256_of "$RUNNER")"
  [[ "$gp" == "$want_pin" ]] || { PF_ERR="sha256 mismatch for bench/apps/runner.py: expected=$want_pin actual=$gp"; return 1; }
  return 0
}
tb_repeat() { local g; for g in 0 1; do tb_warn "$g" "$QRES/gpu$g/$QSUB/cf748_gpu$g.log"; done; }
check_stop() {
  if [[ -e "$STOP_FILE" ]]; then
    log "STOP file present ($STOP_FILE): exiting cleanly, not launching (exit 0)"
    exit 0
  fi
}
records_of() {  # $1 jsonl
  if [[ -f "$1" ]]; then wc -l < "$1" | tr -d ' '; else echo 0; fi
}
wrapper_state() {  # $1 wrapper log -> NONE | START | DONE | FAILED | ABORT (last marker wins)
  local last
  last="$(grep -E 'WRAPPER (START|DONE|FAILED|ABORT)' "$1" 2>/dev/null | tail -n 1)"
  case "$last" in
    *"WRAPPER DONE rc=0"*) echo DONE ;;
    *"WRAPPER FAILED"*) echo FAILED ;;
    *"WRAPPER ABORT"*) echo ABORT ;;
    *"WRAPPER START"*) echo START ;;
    *) echo NONE ;;
  esac
}
tb_warn() {  # $1 gpu, $2 python log. 'Traceback' is only a WARNING: DONE rc=0 + --check-only is the gate.
  local c first key
  c="$(grep -c 'Traceback' "$2" 2>/dev/null)" || c=0
  (( c > 0 )) || return 0
  first="$(grep -n -m1 'Traceback' "$2" | cut -d: -f1)"
  key="$1:$c:$first"
  case " ${TB_SEEN:-} " in *" $key "*) [[ "${TB_FORCE:-0}" == 1 ]] || return 0 ;; esac
  TB_SEEN="${TB_SEEN:-} $key"
  log "WARNING: gpu$1 python log $2 contains $c 'Traceback' line(s), first at line $first (not an abort: completeness check decides)"
}
poll_once() {  # sets ST0 ST1 N0 N1 ABORT_REASON
  local g st n wl pl
  ABORT_REASON=""
  for g in 0 1; do
    wl="$QLOG_DIR/cf748_effk_nofloor_gpu$g.log"
    pl="$QRES/gpu$g/$QSUB/cf748_gpu$g.log"
    st="$(wrapper_state "$wl")"
    n="$(records_of "$QRES/gpu$g/$QSUB/${ARM}_think_t1.0_t1.0.jsonl")"
    if [[ "$g" == 0 ]]; then ST0="$st"; N0="$n"; else ST1="$st"; N1="$n"; fi
    if [[ "$st" == FAILED || "$st" == ABORT ]]; then
      ABORT_REASON="${ABORT_REASON}gpu$g wrapper log $wl ends with WRAPPER $st ($(oneline "$(grep -E 'WRAPPER (FAILED|ABORT)' "$wl" | tail -n 1)")); "
    fi
    tb_warn "$g" "$pl"
  done
  if [[ -n "$ABORT_REASON" && "$DRY_RUN" == 1 ]]; then
    log "DRY: would ABORT: $ABORT_REASON"
    exit 0
  fi
}
nvsmi() {
  if [[ -n "$TIMEOUT_BIN" ]]; then "$TIMEOUT_BIN" 60 "$NVIDIA_SMI_BIN" "$@"; else "$NVIDIA_SMI_BIN" "$@"; fi
}
busy_now() {  # returns 0 (busy) and sets BUSY_DESC, or 1 (clear)
  local pat out g0 g1
  BUSY_DESC=""
  for pat in "${BUSY_PATS[@]}"; do
    if "$PGREP_BIN" -f "$pat" >/dev/null 2>&1; then
      BUSY_DESC="${BUSY_DESC}process matching '$pat' (pids $("$PGREP_BIN" -f "$pat" 2>/dev/null | tr '\n' ' ')); "
    fi
  done
  if [[ -n "$NVIDIA_SMI_BIN" ]]; then
    out="$(nvsmi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || out=""
    g0="$(printf '%s\n' "$out" | awk -F', *' '$1 ~ /^ *[0-9]/ && $1+0==0 {print $2+0; exit}')"
    g1="$(printf '%s\n' "$out" | awk -F', *' '$1 ~ /^ *[0-9]/ && $1+0==1 {print $2+0; exit}')"
    if [[ -z "$g0" || -z "$g1" ]]; then
      BUSY_DESC="${BUSY_DESC}nvidia-smi did not report GPU0 and GPU1; "
    else
      (( g0 < GPU_FREE_MIB )) || BUSY_DESC="${BUSY_DESC}GPU0 memory.used=${g0} MiB >= ${GPU_FREE_MIB}; "
      (( g1 < GPU_FREE_MIB )) || BUSY_DESC="${BUSY_DESC}GPU1 memory.used=${g1} MiB >= ${GPU_FREE_MIB}; "
    fi
  fi
  [[ -n "$BUSY_DESC" ]]
}

# Whole body is parsed before anything runs, so editing this file mid-run cannot change what runs.
main "$@"; exit $?
