#!/usr/bin/env bash
# gpu2_launch_ministral_gk_batches.sh -- GPU2 launcher (Ministral LCB full1046, batches 1+2).
#
# Launches, in parallel (one per GPU), the Ministral-3-3B-Reasoning-2512 LCB
# full1046 generation for batch 1 (CUDA_VISIBLE_DEVICES=0) and batch 2
# (CUDA_VISIBLE_DEVICES=1) for the configs in GK_CONFIGS (default gk_0_45 then
# gk_0_35, custom pless_renyi k=0.45 / 0.35), no time limit. Mirrors the 2026-10-14 batch1/2 launch
# recorded in docs/theory/gpu2_batch1_batch2_pause_checkpoint.md (same env
# vars: CUDA_VISIBLE_DEVICES, PYTHONPATH, PATH; VLLM_USE_FLASHINFER_SAMPLER and
# CUDA_HOME explicitly REMOVED via `env -u`) plus PYTHONUNBUFFERED=1 so the
# runner's "Engine ready in" line reaches the log promptly (I/O only; set
# UNBUFFERED=0 to drop it).
#
# Config list: GK_CONFIGS (space-separated, default "gk_0_45 gk_0_35"; order =
# run order). Each name must match ^gk_[0-9]+(_[0-9]+)*$, appear once, and have
# a ("<name>", row in the runner. Examples: GK_CONFIGS=gk_0_25 (single config),
# GK_CONFIGS="gk_0_25 gk_0_15" (chain).
#
# Config order: the runner loops `for platform in (atcoder, leetcode): for
# config in --configs`, so a single `--configs A B` would interleave per
# platform. To finish each config on BOTH platforms first, each batch runs a
# generated chain (logs/<prefix><IST>/batch<N>/chain_batchN.sh):
#   1) --configs <cfg 1>  (atcoder, then leetcode)
#   2) read-only completeness check: every batch-N task_id present exactly
#      once in the <cfg 1> atcoder+leetcode files, no unparseable lines
#   3) --configs <cfg 2>  -- ONLY if 1) exited 0 AND 2) passed; and so on.
#   last) after the LAST config exits 0, the same check runs once more
#      ("FINAL COMPLETE verified (a/l)" or "FINAL INCOMPLETE ..."; chain exit 3
#      if incomplete), so a single-config run is verified too.
# Marker lines "[gk-chain batch N] ..." go to the per-batch log either way.
# Batches are independent: batch 1 may start config 2 while batch 2 is still on config 1.
#
# Names derived from GK_CONFIGS (the default keeps the original names):
#   launch dir  logs/gk045_launch_<IST>/ when GK_CONFIGS is the default,
#               logs/gk_launch_<IST>/ otherwise (the "no chain running" gate
#               looks for BOTH prefixes);
#   batch log   logs/ministral_fulldataset_batch<N>_<tags>.log, tag = config
#               name with "_" removed, joined by "_" (gk_0_45 gk_0_35 ->
#               gk045_gk035, the original name; gk_0_25 -> gk025).
#
# Gates (any failure -> "STOP" in the log, non-zero exit, nothing launched):
#   valid GK_CONFIGS; lock; no runner/chain process; both GPUs < GPU_FREE_MIB;
#   branch + clean tracked tree; runner has a row per config; REQUIRED runner sha256
#   pin (EXPECT_RUNNER_SHA256, 'any' = loud warning); optional
#   generator_vllm.py pin; vllm version == EXPECT_VLLM_VERSION (0.21.0);
#   batch_assignment.csv has batch 1 and 2 rows. No git pull (pull manually).
#
# Run on GPU2 (the launcher detaches itself with setsid+nohup, so a dropped
# ssh session cannot kill it; it prints the log path and returns at once):
#   cd /home/karrtik/pless-for-coding
#   EXPECT_RUNNER_SHA256=<sha256 of the reviewed runner> bash scripts/gpu2_launch_ministral_gk_batches.sh
#   GK_CONFIGS=gk_0_25 EXPECT_RUNNER_SHA256=<sha256> bash scripts/gpu2_launch_ministral_gk_batches.sh
#   tail -f logs/gpu2_launch_gk.log
# Dry run (stays in the foreground; read-only gates run, mutating actions
# printed): prefix DRY_RUN=1.
# Stub-based tests: uv run pytest tests/test_gpu2_launch_ministral_gk_batches.py
#
# Overridable env: REPO BRANCH DRY_RUN POLL_SECONDS LOG_DIR PYTHON_BIN
#   JSON_PYTHON GIT_BIN NVIDIA_SMI_BIN PGREP_BIN FLOCK_BIN TIMEOUT_BIN SETSID_BIN
#   NVSMI_TIMEOUT_SECONDS GPU_FREE_MIB LAUNCH_VERIFY_MINUTES STAGGER
#   EXPECT_RUNNER_SHA256 EXPECT_GENVLLM_SHA256 EXPECT_VLLM_VERSION UNBUFFERED
#   GK_CONFIGS
# Test hooks: GPU2_LAUNCHER_DETACHED=1 (stay in foreground),
#   LAUNCH_VERIFY_SECONDS (overrides the minutes value), PGREP_SCOPE (regex
#   prefix prepended to every pgrep pattern; empty = production patterns).
set -uo pipefail

main() {
REPO="${REPO:-/home/karrtik/pless-for-coding}"
BRANCH="${BRANCH:-feat/pless-variants-mps}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == "1" ]]; then POLL_SECONDS="${POLL_SECONDS:-1}"; else POLL_SECONDS="${POLL_SECONDS:-10}"; fi
LOG_DIR="${LOG_DIR:-$REPO/logs}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
JSON_PYTHON="${JSON_PYTHON:-$REPO/.venv/bin/python}"   # read-only count/completeness checks
GIT_BIN="${GIT_BIN:-git}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
PGREP_BIN="${PGREP_BIN:-pgrep}"
FLOCK_BIN="${FLOCK_BIN:-$(command -v flock || true)}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
SETSID_BIN="${SETSID_BIN:-$(command -v setsid || true)}"
NVSMI_TIMEOUT_SECONDS="${NVSMI_TIMEOUT_SECONDS:-60}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
LAUNCH_VERIFY_MINUTES="${LAUNCH_VERIFY_MINUTES:-25}"
LAUNCH_VERIFY_SECONDS="${LAUNCH_VERIFY_SECONDS:-}"
# STAGGER=ready (default): start batch 2 only after batch 1's engine is ready
# (or batch 1 died / timed out). STAGGER=<seconds>: fixed delay instead.
STAGGER="${STAGGER:-ready}"
EXPECT_RUNNER_SHA256="${EXPECT_RUNNER_SHA256:-}"   # REQUIRED (64 hex or 'any')
EXPECT_GENVLLM_SHA256="${EXPECT_GENVLLM_SHA256:-}" # optional (empty = no pin, hash still logged)
EXPECT_VLLM_VERSION="${EXPECT_VLLM_VERSION:-0.21.0}"
UNBUFFERED="${UNBUFFERED:-1}"
PGREP_SCOPE="${PGREP_SCOPE:-}"

MODEL="mistralai/Ministral-3-3B-Reasoning-2512"
MODEL_SLUG="mistralai--Ministral-3-3B-Reasoning-2512"
RUNNER="scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
GENVLLM="bench/generator_vllm.py"
DEFAULT_GK_CONFIGS="gk_0_45 gk_0_35"
GK_CONFIGS="${GK_CONFIGS:-$DEFAULT_GK_CONFIGS}"   # one SEQUENTIAL runner invocation per config (see header)
local -a CONFIGS=()
read -r -a CONFIGS <<< "$GK_CONFIGS"
NCFG=${#CONFIGS[@]}
BATCHES=( 1 2 )
FULL_DIR="$REPO/results/pless_variants_cuda/$MODEL_SLUG/LCB_v6_full1046"
BATCH_CSV="$REPO/results/lcb_v6_full1046/batch_assignment.csv"  # cols: task_id,platform,difficulty,batch
QLOG="$LOG_DIR/gpu2_launch_gk.log"
LOCKFILE="$LOG_DIR/gpu2_launch_gk.lock"
ANY_RUNNER_PAT="$(scoped "run_pless_variants_ministral_3b_lcb_cuda.py")"
ANY_CHAIN_PAT="$(scoped 'gk045_launch_.*/chain_batch[0-9]+\.sh')"   # default-config launches
ANY_CHAIN_PAT2="$(scoped 'gk_launch_.*/chain_batch[0-9]+\.sh')"     # any other GK_CONFIGS
LAUNCH_PATH="$REPO/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
ENV_UNSET=( -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME )
LAUNCHED_NOTE=""

mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR" >&2; exit 2; }

# ---------------------------------------------------------------- GK_CONFIGS (validated before detaching)
(( NCFG > 0 )) || stop "GK_CONFIGS is empty/blank (got '$GK_CONFIGS'); give space-separated config names, e.g. GK_CONFIGS=gk_0_25"
local ci cj
for (( ci = 0; ci < NCFG; ci++ )); do
  [[ "${CONFIGS[$ci]}" =~ ^gk_[0-9]+(_[0-9]+)*$ ]] \
    || stop "GK_CONFIGS entry '${CONFIGS[$ci]}' is not a valid name (must match ^gk_[0-9]+(_[0-9]+)*\$); GK_CONFIGS='$GK_CONFIGS'"
  for (( cj = 0; cj < ci; cj++ )); do
    [[ "${CONFIGS[$ci]}" != "${CONFIGS[$cj]}" ]] || stop "GK_CONFIGS lists '${CONFIGS[$ci]}' more than once; GK_CONFIGS='$GK_CONFIGS'"
  done
done
if [[ "${CONFIGS[*]}" == "$DEFAULT_GK_CONFIGS" ]]; then IS_DEFAULT=1; LAUNCH_PREFIX="gk045_launch_"
else IS_DEFAULT=0; LAUNCH_PREFIX="gk_launch_"; fi
RUNLOG_TAG=""
for (( ci = 0; ci < NCFG; ci++ )); do RUNLOG_TAG+="${RUNLOG_TAG:+_}${CONFIGS[$ci]//_/}"; done
local LAST_CFG="${CONFIGS[$((NCFG-1))]}"
if (( IS_DEFAULT )); then
  ORDER_MSG="Config order per batch (1 on GPU0, 2 on GPU1): invocation 1 --configs gk_0_45 (atcoder then leetcode) to completion, THEN invocation 2 --configs gk_0_35 (atcoder then leetcode), started only if invocation 1 exits 0 AND that batch's gk_0_45 atcoder+leetcode files pass the completeness check. Batches are independent: batch 1 may start gk_0_35 while batch 2 is still on gk_0_45 (both GPUs stay busy)."
else
  ORDER_MSG="Config order per batch (1 on GPU0, 2 on GPU1): $NCFG sequential invocation(s):"
  for (( ci = 0; ci < NCFG; ci++ )); do ORDER_MSG+=" $((ci+1))) --configs ${CONFIGS[$ci]} (atcoder then leetcode)"; done
  ORDER_MSG+="; each invocation after the first starts only if the previous one exits 0 AND that batch's previous-config atcoder+leetcode files pass the completeness check. Batches are independent (both GPUs stay busy)."
fi
ORDER_MSG+=" After the last config ($LAST_CFG) exits 0, a FINAL completeness check runs (chain exits 3 if incomplete)."

# ---------------------------------------------------------------- 0. detach (survive ssh drop)
# The lock is taken below, i.e. in the detached child, never in this parent.
if [[ "$DRY_RUN" != "1" && -z "${GPU2_LAUNCHER_DETACHED:-}" ]]; then
  local self; self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  if [[ -n "$SETSID_BIN" ]]; then
    GPU2_LAUNCHER_DETACHED=1 "$SETSID_BIN" nohup bash "$self" "$@" >/dev/null 2>&1 </dev/null &
  else
    echo "WARNING: setsid not found; detaching with nohup only"
    GPU2_LAUNCHER_DETACHED=1 nohup bash "$self" "$@" >/dev/null 2>&1 </dev/null &
  fi
  echo "launcher detached (pid $!); follow it with: tail -f $QLOG"
  exit 0
fi

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

log "==== launcher start pid=$$ REPO=$REPO BRANCH=$BRANCH DRY_RUN=$DRY_RUN STAGGER=$STAGGER LAUNCH_VERIFY_MINUTES=$LAUNCH_VERIFY_MINUTES lock=${FLOCK_BIN:+flock}${FLOCK_BIN:-pidfile} detached=${GPU2_LAUNCHER_DETACHED:-0}"
log "pgrep patterns: runner='$ANY_RUNNER_PAT' chain='$ANY_CHAIN_PAT' chain2='$ANY_CHAIN_PAT2'"
log "GK_CONFIGS: ${CONFIGS[*]} ($NCFG config(s)$([[ $IS_DEFAULT == 1 ]] && printf ', default')); launch dir prefix $LAUNCH_PREFIX; batch log tag $RUNLOG_TAG"
[[ -n "${VLLM_USE_FLASHINFER_SAMPLER+x}" ]] && log "WARNING: VLLM_USE_FLASHINFER_SAMPLER='$VLLM_USE_FLASHINFER_SAMPLER' is set in the launcher env; it is REMOVED (env -u) from every runner command"
[[ -n "${CUDA_HOME+x}" ]] && log "WARNING: CUDA_HOME='$CUDA_HOME' is set in the launcher env; it is REMOVED (env -u) from every runner command"
[[ "$LAUNCH_VERIFY_MINUTES" =~ ^[0-9]+$ ]] || stop "LAUNCH_VERIFY_MINUTES must be a whole number of minutes (got '$LAUNCH_VERIFY_MINUTES')"
local VERIFY_SECS=$(( LAUNCH_VERIFY_MINUTES * 60 ))
if [[ -n "$LAUNCH_VERIFY_SECONDS" ]]; then
  [[ "$LAUNCH_VERIFY_SECONDS" =~ ^[0-9]+$ ]] || stop "LAUNCH_VERIFY_SECONDS must be a whole number (got '$LAUNCH_VERIFY_SECONDS')"
  VERIFY_SECS="$LAUNCH_VERIFY_SECONDS"
fi
[[ "$STAGGER" == "ready" || "$STAGGER" =~ ^[0-9]+$ ]] || stop "STAGGER must be 'ready' or whole seconds (got '$STAGGER')"
cd "$REPO" || stop "cannot cd to REPO=$REPO"

# ---------------------------------------------------------------- 2. no existing run
if "$PGREP_BIN" -f "$ANY_RUNNER_PAT" >/dev/null 2>&1; then
  stop "a runner process is already running ($(proc_list "$ANY_RUNNER_PAT")) -- refusing to launch"
fi
local pat
for pat in "$ANY_CHAIN_PAT" "$ANY_CHAIN_PAT2"; do
  if "$PGREP_BIN" -f "$pat" >/dev/null 2>&1; then
    stop "a gk chain wrapper is already running ($(proc_list "$pat")) -- refusing to launch"
  fi
done
log "no runner process and no gk chain wrapper running"

# ---------------------------------------------------------------- 3. both GPUs free
check_gpus_free

# ---------------------------------------------------------------- 4. code gate
local cur_branch dirty HEAD_SHA got_r got_g c
cur_branch="$("$GIT_BIN" -C "$REPO" branch --show-current 2>&1)" || stop "git branch failed: $cur_branch"
[[ "$cur_branch" == "$BRANCH" ]] || stop "branch is '$cur_branch', expected '$BRANCH'"
dirty="$("$GIT_BIN" -C "$REPO" status --porcelain --untracked-files=no 2>&1)" || stop "git status failed: $dirty"
[[ -z "$dirty" ]] || stop "tracked files are dirty:
$dirty"
HEAD_SHA="$("$GIT_BIN" -C "$REPO" rev-parse HEAD)"
log "branch=$cur_branch, tracked tree clean; HEAD=$HEAD_SHA (no pull performed by this launcher)"
[[ -f "$REPO/$RUNNER" ]] || stop "runner missing: $REPO/$RUNNER"
for c in "${CONFIGS[@]}"; do
  grep -q "(\"$c\"," "$REPO/$RUNNER" || stop "runner lacks a CONFIGS row for \"$c\" -- old code, refusing to launch"
done
log "runner has CONFIGS rows for: ${CONFIGS[*]}"
got_r="$(sha256_of "$REPO/$RUNNER")"
got_g="$(sha256_of "$REPO/$GENVLLM" 2>/dev/null || true)"
log "sha256 $RUNNER = $got_r"
log "sha256 $GENVLLM = ${got_g:-MISSING}"
if [[ -z "$EXPECT_RUNNER_SHA256" ]]; then
  stop "EXPECT_RUNNER_SHA256 is REQUIRED (64 hex chars, or 'any'); actual sha256 of $RUNNER is $got_r"
fi
check_pin "$RUNNER" "$EXPECT_RUNNER_SHA256" "$got_r"
if [[ -z "$EXPECT_GENVLLM_SHA256" ]]; then
  log "no pin for $GENVLLM (EXPECT_GENVLLM_SHA256 unset); actual=$got_g"
else
  check_pin "$GENVLLM" "$EXPECT_GENVLLM_SHA256" "$got_g"
fi
if ! grep -q 'append_result\|flock' "$REPO/$RUNNER"; then
  log "WARNING: $RUNNER appends records with a plain open('a') (no append_result/flock); batches 1 and 2 write the SAME shared jsonl files concurrently. Check line integrity after the run."
fi

# ---------------------------------------------------------------- 5. vllm version
local vv
vv="$("$PYTHON_BIN" -c "import importlib.metadata as m; print(m.version('vllm'))" 2>&1)" \
  || stop "could not read vllm version via $PYTHON_BIN: $vv"
vv="$(printf '%s' "$vv" | tail -n 1 | tr -d '[:space:]')"
[[ "$vv" == "$EXPECT_VLLM_VERSION" ]] \
  || stop "vllm version is '$vv', expected '$EXPECT_VLLM_VERSION' (batches 1/2 were generated on 0.21.0; results must be comparable)"
log "vllm version $vv (matches expected)"

# ---------------------------------------------------------------- 6. batch assignment
[[ -f "$BATCH_CSV" ]] || stop "batch assignment missing: $BATCH_CSV"
local b counts
for b in "${BATCHES[@]}"; do
  counts="$(awk -F, -v b="$b" '
    NR==1 { for (i=1;i<=NF;i++) { gsub(/\r/,"",$i); col[$i]=i }; next }
    { gsub(/\r/,""); v=$col["batch"]; gsub(/^[ \t]+|[ \t]+$/,"",v); if (v==b) { n[$col["platform"]]++; t++ } }
    END { printf "total=%d atcoder=%d leetcode=%d", t, n["atcoder"], n["leetcode"] }' "$BATCH_CSV")"
  log "batch_assignment batch $b: $counts"
  [[ "$counts" == total=0* ]] && stop "batch_assignment.csv has no rows for batch $b"
done

# ---------------------------------------------------------------- 7. existing results (log only)
log_existing_results

# ---------------------------------------------------------------- 8. launch
local TS LROOT
TS="$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S)"
LROOT="$LOG_DIR/$LAUNCH_PREFIX$TS"
CHECK_PY="$LROOT/check_step_complete.py"
if [[ "$DRY_RUN" == "1" ]]; then
  if (( IS_DEFAULT )); then
    log "DRY: would write $CHECK_PY (read-only gk_0_45 completeness check run by each chain before step 2; also run after the last step as the FINAL check)"
  else
    log "DRY: would write $CHECK_PY (read-only completeness check run by each chain after each of: ${CONFIGS[*]}; the one after $LAST_CFG is the FINAL check)"
  fi
else
  mkdir -p "$LROOT" || stop "cannot create $LROOT"
  write_check_py "$CHECK_PY" || stop "cannot write $CHECK_PY"
fi
local -a PIDS=() T0S=() OFFS=() STATUS=()
local i
for i in 0 1; do
  b="${BATCHES[$i]}"
  if [[ "$DRY_RUN" != "1" ]] && batch_running "$b"; then
    stop "a runner or chain for batch $b appeared meanwhile -- refusing to launch a second"
  fi
  launch_batch "$b" "$LROOT" "$HEAD_SHA" || exit 1
  PIDS[$i]="$LAST_PID"; T0S[$i]="$SECONDS"; OFFS[$i]="$LAST_OFF"
  if (( i == 0 )); then
    if [[ "$STAGGER" == "ready" ]]; then
      if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY: would wait for batch $b 'Engine ready in' (<= ${VERIFY_SECS}s) before launching batch ${BATCHES[1]} (STAGGER=ready)"
      else
        log "STAGGER=ready: waiting for batch $b engine before starting batch ${BATCHES[1]} (serialized engine init avoids concurrent vLLM compile/JIT cache writes)"
        verify_batch "$b" "${PIDS[0]}" "${T0S[0]}" "${OFFS[0]}" "$LROOT/batch$b"; STATUS[0]=$?
        (( STATUS[0] == 0 )) || log "batch $b not ready (status ${STATUS[0]}); launching batch ${BATCHES[1]} anyway (independent GPU)"
      fi
    else
      act sleep "$STAGGER"
    fi
  fi
done
write_integrity_check "$LROOT"
if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY: would capture /proc/<runner python pid>/{environ,cmdline} (child of each chain wrapper) into $LROOT/batch<N>/ after the first invocation's 'Engine ready in'"
  log "==== DRY RUN complete (exit 0); nothing launched. $ORDER_MSG"; exit 0
fi
[[ -n "${STATUS[0]:-}" ]] || { verify_batch "${BATCHES[0]}" "${PIDS[0]}" "${T0S[0]}" "${OFFS[0]}" "$LROOT/batch${BATCHES[0]}"; STATUS[0]=$?; }
verify_batch "${BATCHES[1]}" "${PIDS[1]}" "${T0S[1]}" "${OFFS[1]}" "$LROOT/batch${BATCHES[1]}"; STATUS[1]=$?

if (( STATUS[0] == 0 && STATUS[1] == 0 )); then
  log "==== both engines ready (batch 1 chain PID ${PIDS[0]}, batch 2 chain PID ${PIDS[1]}); launch dir $LROOT; exit 0. $ORDER_MSG"
  exit 0
fi
local msg=""
for i in 0 1; do
  b="${BATCHES[$i]}"
  if (( STATUS[i] == 0 )); then
    msg+="batch $b OK (chain PID ${PIDS[$i]}, KEEPS RUNNING, not killed); "
  elif (( STATUS[i] == 2 )); then
    msg+="batch $b FAILED: no 'Engine ready' within ${VERIFY_SECS}s (chain PID ${PIDS[$i]} still alive, NOT killed); "
  else
    msg+="batch $b FAILED: chain PID ${PIDS[$i]} exited before 'Engine ready' ($(rest_msg)); "
  fi
done
log "==== launch incomplete: $msg resume scripts under $LROOT/batch<N>/; exit 1. $ORDER_MSG"
exit 1
}

# ======================================================================= helpers
ist() { TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST'; }
log() { printf '[%s] %s\n' "$(ist)" "$*" | tee -a "$QLOG"; }
stop() {
  log "STOP: $*"
  if [[ -z "${LAUNCHED_NOTE:-}" ]]; then log "launcher exiting non-zero; nothing launched"
  else log "launcher exiting non-zero; already launched: ${LAUNCHED_NOTE}-- it KEEPS RUNNING (not killed)"; fi
  exit 1
}
act() {  # mutating action; printed instead of executed when DRY_RUN=1
  if [[ "$DRY_RUN" == "1" ]]; then log "DRY: would run: $*"; return 0; fi
  log "run: $*"; "$@"
}
scoped() {  # prepend the PGREP_SCOPE test hook (empty in production = pattern unchanged)
  if [[ -n "${PGREP_SCOPE:-}" ]]; then printf '%s.*%s' "$PGREP_SCOPE" "$1"; else printf '%s' "$1"; fi
}
pid_alive() { kill -0 "$1" 2>/dev/null || return 1; [[ "$(ps -o stat= -p "$1" 2>/dev/null)" != Z* ]]; }
batch_pat() { printf '%s' "$ANY_RUNNER_PAT.*--batch $1( |\$)"; }
chain_pat() { scoped "(gk045|gk)_launch_.*/chain_batch$1\\.sh"; }   # both launch-dir prefixes
runlog_of() { printf '%s' "$LOG_DIR/ministral_fulldataset_batch${1}_${RUNLOG_TAG}.log"; }
rest_msg() {  # configs after the first (never started if step 1 died)
  local k out=""
  for (( k = 1; k < NCFG; k++ )); do out+="${out:+ }${CONFIGS[$k]}"; done
  if [[ -n "$out" ]]; then printf '%s NOT started' "$out"; else printf 'single config %s; no further step' "${CONFIGS[0]}"; fi
}
batch_running() {  # a runner OR a chain wrapper for batch $1
  "$PGREP_BIN" -f "$(batch_pat "$1")" >/dev/null 2>&1 || "$PGREP_BIN" -f "$(chain_pat "$1")" >/dev/null 2>&1
}
runner_child_of() {  # $1 chain pid, $2 batch -> pid of the runner python process under the chain
  local p
  for p in $("$PGREP_BIN" -P "$1" 2>/dev/null); do
    if ps -o args= -p "$p" 2>/dev/null | grep -Eq "$(batch_pat "$2")"; then echo "$p"; return 0; fi
  done
  return 1
}
proc_list() {
  local p out=""
  for p in $("$PGREP_BIN" -f "$1" 2>/dev/null); do
    out+="$p $(ps -o args= -p "$p" 2>/dev/null | cut -c1-200); "
  done
  printf '%s' "${out:-none}"
}
count_lines() { if [[ -f "$1" ]]; then wc -l < "$1" | tr -d ' '; else echo "MISSING"; fi; }
sha256_of() {
  [[ -f "$1" ]] || return 1
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
check_pin() {  # $1 path, $2 expected, $3 actual
  if [[ "$2" == "any" ]]; then
    log "WARNING: ******** sha256 pin for $1 DISABLED (expected=any) -- launching UNPINNED code; actual=$3 ********"
    return 0
  fi
  local want; want="$(printf '%s' "$2" | tr 'A-F' 'a-f')"
  [[ "$want" =~ ^[0-9a-f]{64}$ ]] || stop "expected sha256 for $1 is not 64 hex chars (or 'any'): '$2'; actual=$3"
  [[ "$want" == "$3" ]] || stop "sha256 mismatch for $1: expected=$want actual=$3"
  log "pin: sha256 $1 = $3 (matches expected)"
}
check_gpus_free() {  # STOP unless GPU 0 and GPU 1 both report memory.used < GPU_FREE_MIB
  local fails=0 out rc g0 g1
  while :; do
    out="$("$TIMEOUT_BIN" "$NVSMI_TIMEOUT_SECONDS" "$NVIDIA_SMI_BIN" \
            --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)"; rc=$?
    if (( rc == 0 )) && [[ -n "$out" ]]; then break; fi
    fails=$((fails+1))
    if (( rc == 124 )); then log "nvidia-smi timed out after ${NVSMI_TIMEOUT_SECONDS}s ($fails/5)"
    else log "nvidia-smi query failed rc=$rc ($fails/5)"; fi
    (( fails >= 5 )) && stop "nvidia-smi failed 5 times in a row"
    sleep "$POLL_SECONDS"
  done
  g0="$(printf '%s\n' "$out" | awk -F', *' '$1+0==0 && $1 ~ /^ *[0-9]/ {print $2+0; exit}')"
  g1="$(printf '%s\n' "$out" | awk -F', *' '$1+0==1 {print $2+0; exit}')"
  log "nvidia-smi memory.used: GPU0=${g0:-?} MiB GPU1=${g1:-?} MiB (threshold < $GPU_FREE_MIB)"
  [[ -n "$g0" && -n "$g1" ]] || stop "nvidia-smi did not report both GPU 0 and GPU 1: '$(printf '%s' "$out" | tr '\n' ';')'"
  (( g0 < GPU_FREE_MIB )) || stop "GPU0 busy: memory.used=${g0} MiB >= ${GPU_FREE_MIB}"
  (( g1 < GPU_FREE_MIB )) || stop "GPU1 busy: memory.used=${g1} MiB >= ${GPU_FREE_MIB}"
  log "both GPUs free"
}
log_existing_results() {  # informational only: runner resumes by task_id
  local c plat f out
  for c in "${CONFIGS[@]}"; do
    for plat in atcoder leetcode; do
      f="$FULL_DIR/${c}_${plat}_think_t1.0.jsonl"
      if [[ ! -f "$f" ]]; then log "existing results: $c/$plat none ($f absent)"; continue; fi
      out="$("$JSON_PYTHON" -I -c '
import csv, json, sys
with open(sys.argv[2], newline="") as fh:
    bmap = {}
    for r in csv.DictReader(fh):
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        bmap[r.get("task_id", "")] = r.get("batch", "")
n = {"1": 0, "2": 0}; other = bad = lines = 0
with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if not line.strip():
            continue
        lines += 1
        try:
            b = bmap.get(str(json.loads(line)["task_id"]))
        except Exception:
            bad += 1; continue
        if b in n: n[b] += 1
        else: other += 1
print("lines=%d batch1=%d batch2=%d other_batches=%d unparseable=%d" % (lines, n["1"], n["2"], other, bad))
' "$f" "$BATCH_CSV" 2>&1)" || out="(count failed: $out) wc -l=$(count_lines "$f")"
      log "existing results (will be RESUMED by task_id, not a gate): $c/$plat $out"
    done
  done
}
# write_check_py PATH: read-only completeness check run by each chain after
# every step (between steps, and once more after the last step = FINAL check).
# Streams the jsonl line by line; prints only task_ids/counts.
# argv: batch_csv batch done_config next_config [--step=K/N] platform=path...
#   next_config "-" = FINAL check after the last step (labels "FINAL ...").
#   --step=K/N (default 1/2, omitted then) numbers the "STEP K ..." lines.
# exit 0 = complete, 3 = incomplete (missing / duplicated / unparseable).
write_check_py() {
  cat > "$1" <<'PY'
import csv, json, sys
csv_path, batch, done_cfg, next_cfg = sys.argv[1:5]
step, nsteps = 1, 2
files = []
for a in sys.argv[5:]:
    if a.startswith("--step="):
        step, nsteps = (int(x) for x in a[len("--step="):].split("/", 1))
    else:
        files.append(a.split("=", 1))
final = next_cfg == "-"
want = {plat: set() for plat, _ in files}
with open(csv_path, newline="") as fh:          # csv module handles CRLF
    for row in csv.DictReader(fh):
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        if row.get("batch") == batch.strip() and row.get("platform") in want:
            want[row["platform"]].add(row["task_id"])
tot_missing = tot_dup = tot_bad = 0
present = []
for plat, path in files:
    seen, dups, bad, other_dups = set(), set(), 0, 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:                       # stream; never print sample text
                if not line.strip():
                    continue
                try:
                    tid = str(json.loads(line)["task_id"])
                except Exception:
                    bad += 1
                    continue
                if tid in seen:
                    if tid in want[plat]:
                        dups.add(tid)
                    else:
                        other_dups += 1
                seen.add(tid)
    except FileNotFoundError:
        print(f"{plat}: FILE MISSING {path}")
    missing = sorted(want[plat] - seen)
    if not want[plat]:
        print(f"{plat}: WARNING batch {batch} has 0 {plat} tasks in {csv_path}")
        tot_missing += 1
    have = len(want[plat] & seen)
    present.append(have)
    print(f"{plat}: {done_cfg} batch{batch}_expected={len(want[plat])} present={have} "
          f"missing={len(missing)} dup={len(dups)} bad={bad}"
          + (f" (WARNING other-batch duplicate lines={other_dups})" if other_dups else ""))
    if missing:
        print(f"{plat}: missing (<=10): {missing[:10]}")
    if dups:
        print(f"{plat}: duplicated (<=10): {sorted(dups)[:10]}")
    tot_missing += len(missing); tot_dup += len(dups); tot_bad += bad
label = "FINAL" if final else f"STEP {step}"
if tot_missing or tot_dup or tot_bad:
    tail = (f" -- {done_cfg} batch {batch} NOT complete" if final
            else f" -- step {step + 1}/{nsteps} ({next_cfg}) NOT started")
    print(f"{label} INCOMPLETE: missing={tot_missing} dup={tot_dup} bad={tot_bad}{tail}")
    sys.exit(3)
print(f"{label} COMPLETE verified (" + "/".join(str(n) for n in present) + ")")
sys.exit(0)
PY
}
# write_integrity_check LROOT: post-run line-integrity check for the SHARED
# gk jsonl files (both batches append to the same files; runner uses a plain
# open('a'), no flock). Informational; never a gate.
write_integrity_check() {
  local lroot="$1" f="$1/integrity_check.sh" files="" c plat
  for c in "${CONFIGS[@]}"; do for plat in atcoder leetcode; do
    files+=" $(printf '%q' "$FULL_DIR/${c}_${plat}_think_t1.0.jsonl")"
  done; done
  local body
  body="$(cat <<EOF
#!/usr/bin/env bash
# Post-run integrity check of the shared ${CONFIGS[*]} full1046 jsonl files:
# every non-blank line must parse as JSON with a task_id, and no task_id may repeat.
cd $(printf '%q' "$REPO") && $(printf '%q' "$JSON_PYTHON") -I -c '
import json, sys
bad_any = 0
for p in sys.argv[1:]:
    ids, bad = [], 0
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for l in fh:
                if not l.strip():
                    continue
                try:
                    ids.append(str(json.loads(l)["task_id"]))
                except Exception:
                    bad += 1
    except FileNotFoundError:
        print(p, "MISSING"); continue
    dup = len(ids) - len(set(ids))
    bad_any |= bool(bad or dup)
    print(p, "records=%d distinct=%d unparseable=%d duplicate_task_ids=%d" % (len(ids), len(set(ids)), bad, dup))
sys.exit(1 if bad_any else 0)
'$files
EOF
)"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: would write $f (post-run shared-file integrity check)"
  else
    printf '%s\n' "$body" > "$f" && chmod +x "$f"
  fi
  log "HINT: shared-file integrity (runner appends without flock; NOT a gate): after the runs (or any time), run: bash $f"
}
# launch_batch N LROOT HEAD -> sets LAST_PID (chain wrapper pid), LAST_OFF.
launch_batch() {
  local b="$1" lroot="$2" head="$3" ldir runlog gpu chain c k
  gpu=$(( b - 1 ))
  ldir="$lroot/batch$b"
  runlog="$(runlog_of "$b")"
  chain="$ldir/chain_batch$b.sh"
  local -a ENVV=( CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" PATH="$LAUNCH_PATH" )
  [[ "$UNBUFFERED" == "1" ]] && ENVV+=( PYTHONUNBUFFERED=1 )
  local -a CMDS=() GATES=()
  for c in "${CONFIGS[@]}"; do
    local -a CMD=( "$PYTHON_BIN" "$RUNNER" --model "$MODEL" --dtype bfloat16
          --gpu-memory-utilization 0.92 --max-model-len 37500 --n-samples 10
          --max-new-tokens 35000 --configs "$c" --full-dataset --batch "$b"
          --budget-minutes inf --per-config-minutes inf --seed 42 )
    CMDS+=( "$( printf 'env'; printf ' %s' "${ENV_UNSET[@]}"; printf ' %q' "${ENVV[@]}" "${CMD[@]}"; printf ' < /dev/null' )" )
  done
  # GATES[k-1] = completeness check of config k, run after step k. Between steps
  # it is prefixed with `env -u ...` (as before); --step=K/N only when K/N != 1/2
  # (so the default's gate command is unchanged). The FINAL check (after the last
  # step, next_config "-") calls the stdlib-only check directly: the chain itself
  # already runs without VLLM_USE_FLASHINFER_SAMPLER/CUDA_HOME (launched via env -u).
  local -a STEPARG
  for (( k = 1; k <= NCFG; k++ )); do
    c="${CONFIGS[$((k-1))]}"
    if (( k < NCFG )); then
      STEPARG=(); [[ "$k/$NCFG" == "1/2" ]] || STEPARG=( "--step=$k/$NCFG" )
      GATES+=( "$( printf 'env'; printf ' %s' "${ENV_UNSET[@]}"; printf ' %q' "$JSON_PYTHON" -I "$CHECK_PY" "$BATCH_CSV" "$b" "$c" "${CONFIGS[$k]}" \
                   ${STEPARG[@]+"${STEPARG[@]}"} \
                   "atcoder=$FULL_DIR/${c}_atcoder_think_t1.0.jsonl" "leetcode=$FULL_DIR/${c}_leetcode_think_t1.0.jsonl" )" )
    else
      GATES+=( "$( printf '%q' "$JSON_PYTHON"; printf ' %q' -I "$CHECK_PY" "$BATCH_CSV" "$b" "$c" - "--step=$k/$NCFG" \
                   "atcoder=$FULL_DIR/${c}_atcoder_think_t1.0.jsonl" "leetcode=$FULL_DIR/${c}_leetcode_think_t1.0.jsonl" )" )
    fi
  done
  local check_cmd="${GATES[0]}" final_cmd="${GATES[$((NCFG-1))]}"
  # Chain wrapper: step k+1 starts only if step k exits 0 AND its completeness
  # check passes; after the last step the FINAL check runs (exit 3 if incomplete).
  local body cl
  body="#!/usr/bin/env bash
# gk chain for Ministral LCB full1046 batch $b on GPU $gpu: ${CONFIGS[*]} (in this order, each on both platforms), then a FINAL completeness check.
# Generated by scripts/gpu2_launch_ministral_gk_batches.sh at $(ist); HEAD=$head
# Runner resumes by task_id, so re-running this chain skips finished tasks.
set -u
mark() { printf '[%s] [gk-chain batch $b] %s\n' \"\$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST')\" \"\$*\"; }
cd $(printf '%q' "$REPO") || { mark \"CHAIN ABORT: cannot cd to repo\"; exit 1; }"
  for (( k = 1; k <= NCFG; k++ )); do
    c="${CONFIGS[$((k-1))]}"
    if (( k == 1 )); then body+="
mark \"CHAIN START pid=\$\$ step 1/$NCFG: --configs $c\""
    else body+="
mark \"starting step $k/$NCFG: --configs $c\""
    fi
    if (( k < NCFG )); then
      body+="
${CMDS[$((k-1))]}
rc=\$?
if (( rc != 0 )); then
  mark \"STEP $k/$NCFG ($c) FAILED rc=\$rc -- step $((k+1))/$NCFG (${CONFIGS[$k]}) NOT started\"
  exit \"\$rc\"
fi
mark \"STEP $k/$NCFG ($c) finished rc=0; verifying batch-$b completeness of $c atcoder+leetcode\"
chk=\"\$(${GATES[$((k-1))]} 2>&1)\"
crc=\$?
while IFS= read -r l; do mark \"\$l\"; done <<< \"\$chk\"
if (( crc == 3 )); then exit 3; fi
if (( crc != 0 )); then mark \"STEP $k CHECK ERROR rc=\$crc -- step $((k+1))/$NCFG (${CONFIGS[$k]}) NOT started\"; exit 3; fi"
    else
      body+="
${CMDS[$((k-1))]}
rc=\$?
if (( rc != 0 )); then mark \"STEP $k/$NCFG ($c) FAILED rc=\$rc\"; exit \"\$rc\"; fi
mark \"STEP $k/$NCFG ($c) runner exited rc=0; FINAL check: verifying batch-$b completeness of $c atcoder+leetcode\"
chk=\"\$($final_cmd 2>&1)\"
crc=\$?
while IFS= read -r l; do mark \"\$l\"; done <<< \"\$chk\"
if (( crc == 3 )); then exit 3; fi
if (( crc != 0 )); then mark \"FINAL CHECK ERROR rc=\$crc\"; exit 3; fi
mark \"STEP $k/$NCFG ($c) finished rc=0; CHAIN DONE\"
exit 0"
    fi
  done
  local launch_line
  launch_line="$( printf 'nohup env'; printf ' %s' "${ENV_UNSET[@]}"; printf ' %q' "${ENVV[@]}"
                  printf ' bash %q >> %q 2>&1 < /dev/null &' "$chain" "$runlog" )"
  LAST_PID=""; LAST_OFF=0
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: batch $b (GPU $gpu) env: ${ENV_UNSET[*]} ${ENVV[*]}"
    log "DRY: batch $b step 1/$NCFG would run: ${CMDS[0]}"
    for (( k = 2; k <= NCFG; k++ )); do
      if (( k == 2 )); then log "DRY: batch $b completeness gate would run: ${GATES[0]}"
      else log "DRY: batch $b completeness gate after step $((k-1))/$NCFG would run: ${GATES[$((k-2))]}"; fi
      log "DRY: batch $b step $k/$NCFG would run (only if step $((k-1)) exits 0 and the gate passes): ${CMDS[$((k-1))]}"
    done
    log "DRY: batch $b final completeness gate would run (only if step $NCFG/$NCFG exits 0; chain exits 3 if incomplete): $final_cmd"
    log "DRY: batch $b would write $chain, launched as: cd $REPO; $launch_line"
    log "DRY: batch $b would write $ldir/{cmdline.txt,environ.txt,resume_batch$b.sh}"
    return 0
  fi
  mkdir -p "$ldir" || stop "cannot create $ldir"
  { printf '%s\n' "$body" > "$chain" && chmod +x "$chain"; } || stop "cannot write $chain"
  { echo "launch: cd $REPO; $launch_line"
    for (( k = 1; k <= NCFG; k++ )); do
      echo "step $k: ${CMDS[$((k-1))]}"
      if (( k < NCFG )); then
        if (( k == 1 )); then echo "gate: ${GATES[0]}"; else echo "gate after step $k: ${GATES[$((k-1))]}"; fi
      fi
    done
    echo "final gate (after step $NCFG): $final_cmd"; } > "$ldir/cmdline.txt"
  cat > "$ldir/resume_batch$b.sh" <<EOF
#!/usr/bin/env bash
# Resume the gk chain for Ministral LCB full1046 batch $b on GPU $gpu (re-runs the whole
# chain; the runner resumes by task_id, so finished tasks are skipped).
# Chain configs, in order (each followed by its completeness check; FINAL after the last): ${CONFIGS[*]}
# Chain: $chain
# Generated by scripts/gpu2_launch_ministral_gk_batches.sh at $(ist); HEAD=$head
# Re-applies the full launch environment (CUDA_VISIBLE_DEVICES, PYTHONPATH, PATH;
# VLLM_USE_FLASHINFER_SAMPLER and CUDA_HOME removed).
set -u
if pgrep -f '$(batch_pat "$b")' >/dev/null; then echo "a runner for batch $b is already running; refusing" >&2; exit 1; fi
if pgrep -f '$(chain_pat "$b")' >/dev/null; then echo "a gk chain for batch $b is already running; refusing" >&2; exit 1; fi
gout="\$($(printf '%q' "$TIMEOUT_BIN") $NVSMI_TIMEOUT_SECONDS $(printf '%q' "$NVIDIA_SMI_BIN") --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || gout=""
if [[ -z "\$gout" ]]; then
  echo "WARNING: nvidia-smi query failed; cannot confirm both GPUs are free (continuing)" >&2
else
  printf '%s\n' "\$gout" | awk -F', *' -v t=$GPU_FREE_MIB '\$1 ~ /^ *[0-9]/ && \$2+0 >= t {print "WARNING: GPU " \$1+0 " memory.used=" \$2+0 " MiB >= " t " (continuing)"}' >&2
fi
# NOTE: cd on its own line -- "cd X && nohup ... &" would background a subshell that
# keeps the caller's stdout open and makes \$! the subshell, not the chain.
cd $(printf '%q' "$REPO") || exit 1
$launch_line
echo "launched batch $b chain PID \$!"
EOF
  chmod +x "$ldir/resume_batch$b.sh"
  [[ -f "$runlog" ]] && LAST_OFF="$(wc -c < "$runlog" | tr -d ' ')"
  log "batch $b launch (GPU $gpu): cd $REPO; $launch_line   (log byte offset before launch: $LAST_OFF)"
  log "batch $b step 1/$NCFG: ${CMDS[0]}"
  for (( k = 2; k <= NCFG; k++ )); do
    if (( k == 2 )); then log "batch $b gate:     ${GATES[0]}"
    else log "batch $b gate after step $((k-1))/$NCFG: ${GATES[$((k-2))]}"; fi
    log "batch $b step $k/$NCFG (only if step $((k-1)) exits 0 and the gate passes): ${CMDS[$((k-1))]}"
  done
  log "batch $b final gate (only if step $NCFG/$NCFG exits 0): $final_cmd"
  # 9>&- : the long-running chain/runner must not inherit the flock fd.
  nohup env "${ENV_UNSET[@]}" "${ENVV[@]}" bash "$chain" >>"$runlog" 2>&1 </dev/null 9>&- &
  LAST_PID=$!
  LAUNCHED_NOTE+="batch $b (chain PID $LAST_PID) "
  { echo "HEAD=$head"; echo "CHAIN_PID=$LAST_PID"; } >> "$ldir/cmdline.txt"
  { printf '%s\n' "${ENVV[@]}"; echo "# (fallback: explicitly set vars; launcher env minus ${ENV_UNSET[*]} follows)"
    env "${ENV_UNSET[@]}" env; } > "$ldir/environ.txt"
  log "batch $b chain started: PID=$LAST_PID log=$runlog"
  return 0
}
# verify_batch N CHAIN_PID T0 OFFSET LDIR -> 0 ready, 1 died, 2 timeout (alive, not killed)
# Only step 1 can print 'Engine ready in' first: step 2 starts after step 1 exits.
verify_batch() {
  local b="$1" pid="$2" t0="$3" off="$4" ldir="$5" runlog ready lp rpid
  runlog="$(runlog_of "$b")"
  lp=$(( POLL_SECONDS < 10 ? POLL_SECONDS : 10 ))
  while :; do
    # split on \r too: a tqdm progress line may precede the message on the same "line"
    ready="$(tail -c +$((off+1)) "$runlog" 2>/dev/null | tr '\r' '\n' | grep -m1 'Engine ready in' || true)"
    if [[ -n "$ready" ]]; then
      log "batch $b launch verified (step 1/$NCFG, ${CONFIGS[0]}): '$ready' (chain PID=$pid log=$runlog)"
      rpid="$(runner_child_of "$pid" "$b" || true)"
      if [[ -z "$rpid" ]]; then
        log "batch $b: could not find the runner python child of chain $pid; kept fallback environ.txt"
      elif [[ -r "/proc/$rpid/environ" ]]; then
        tr '\0' '\n' < "/proc/$rpid/environ" > "$ldir/environ.txt"
        { printf 'proc_cmdline (runner PID %s): ' "$rpid"; tr '\0' ' ' < "/proc/$rpid/cmdline"; echo; } >> "$ldir/cmdline.txt"
        log "batch $b: captured /proc/$rpid/{environ,cmdline} (runner PID $rpid, child of chain $pid) into $ldir"
      else
        echo "RUNNER_PID=$rpid" >> "$ldir/cmdline.txt"
        log "batch $b: runner PID $rpid (child of chain $pid); no /proc (non-Linux), kept fallback environ.txt"
      fi
      return 0
    fi
    if ! pid_alive "$pid"; then
      log "FAIL: batch $b chain PID $pid exited before 'Engine ready'. New log tail ($runlog):"
      tail -c +$((off+1)) "$runlog" | tail -n 20 | sed 's/^/    | /' | tee -a "$QLOG"
      return 1
    fi
    if (( SECONDS - t0 >= VERIFY_SECS )); then
      log "FAIL: batch $b no 'Engine ready' within ${VERIFY_SECS}s (chain PID $pid still alive, NOT killed). New log tail:"
      tail -c +$((off+1)) "$runlog" | tail -n 20 | sed 's/^/    | /' | tee -a "$QLOG"
      return 2
    fi
    sleep "$lp"
  done
}

# Whole body is parsed before anything runs, so editing this file mid-run
# cannot change what the running launcher executes.
main "$@"; exit $?
