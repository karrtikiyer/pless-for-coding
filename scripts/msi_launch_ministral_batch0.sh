#!/usr/bin/env bash
# msi_launch_ministral_batch0.sh -- gated MSI launcher: Ministral-3-3B-Reasoning-2512, LCB v6 full1046,
# BATCH 0, configs from CONFIGS (default "top_p_0_9 gk_0_45", in that order on the command line).
#
# The runner (scripts/run_pless_variants_ministral_3b_lcb_cuda.py) is PLATFORM-MAJOR: AtCoder for every
# config in CONFIGS order, then LeetCode for every config. Batch 0 = 121 AtCoder + 90 LeetCode tasks
# (results/lcb_v6_full1046/batch_assignment.csv). Outputs (shared with batches 1/2, appended by task_id):
#   results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/
#     <config>_{atcoder,leetcode}_think_t<T>.jsonl
# Recipe = the MSI top-k run (scripts/msi_queue_topk_after_recipe.sh): --max-model-len 37500 --n-samples 10
#   --max-new-tokens 35000 --full-dataset --batch 0 --budget-minutes inf --per-config-minutes inf --seed 42,
#   env CUDA_HOME=/usr/local/cuda-13.1 PYTHONPATH=$REPO PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=1, and
#   PATH=/usr/local/cuda-13.1/bin:$REPO/.venv/bin:$PATH (FlashInfer JIT needs ninja + nvcc).
#
# Gates (any failure -> "STOP" line in the log, non-zero exit, nothing launched):
#   env/config validation (CONFIGS names ^[a-z0-9_]+$, no duplicates, then each in the runner's CONFIGS);
#   single-instance lock (flock, else pidfile with liveness check); no Ministral runner / msi_queue
#   launcher running; GPU free (no VLLM::EngineCore, nvidia-smi memory.used < GPU_FREE_MIB, optional capped
#   wait GPU_WAIT_MAX_MINUTES, default 0 = check once); branch == BRANCH, tracked tree clean, optional
#   ff-only pull (PULL=1), HEAD == EXPECT_HEAD (REQUIRED); REQUIRED sha256 pins of the runner and
#   bench/generator_vllm.py; venv python + vllm/torch/flashinfer present (versions logged; optional
#   EXPECT_VLLM_VERSION / EXPECT_TORCH_VERSION / EXPECT_FLASHINFER_VERSION pins); ninja + nvcc on the launch
#   PATH; batch-0 counts == EXPECT_ATCODER / EXPECT_LEETCODE; existing non-empty outputs for the requested
#   configs -> STOP unless ALLOW_EXISTING=1 (and they must pass the integrity check either way).
# DRY_RUN=1 runs every read-only gate and prints the exact command; it never launches or writes a launch dir.
# No smoke test (SMOKE=1 is refused, see the report in the commit / PR).
#
# On MSI (values below are for origin/feat/pless-variants-mps @ 4040f15):
#   cd ~/pless-for-coding && git pull --ff-only origin feat/pless-variants-mps
#   DRY_RUN=1 EXPECT_HEAD=<full 40-hex sha of HEAD> \
#     EXPECT_RUNNER_SHA256=b1ce0a53fc276a57dd1dbc37c3b33217f42cd456a5ae176b1576418b65517f07 \
#     EXPECT_GENVLLM_SHA256=d89df1e94ff564cb4f9b51b68276d377e0508267a1e6d8a9ef2224ab3f064410 \
#     bash scripts/msi_launch_ministral_batch0.sh
#   (same without DRY_RUN=1 to launch; tail -f logs/msi_launch_ministral_batch0.log)
# Stub-based tests: uv run pytest tests/test_msi_launch_ministral_batch0.py
#
# Overridable env: REPO BRANCH CONFIGS DRY_RUN PULL ALLOW_EXISTING EXPECT_HEAD EXPECT_RUNNER_SHA256
#   EXPECT_GENVLLM_SHA256 EXPECT_VLLM_VERSION EXPECT_TORCH_VERSION EXPECT_FLASHINFER_VERSION EXPECT_ATCODER
#   EXPECT_LEETCODE GPU_FREE_MIB GPU_WAIT_MAX_MINUTES LAUNCH_VERIFY_MINUTES POLL_SECONDS CUDA_HOME_VAL LOG_DIR
#   PYTHON_BIN GIT_BIN NVIDIA_SMI_BIN PGREP_BIN FLOCK_BIN SETSID_BIN TIMEOUT_BIN NVSMI_TIMEOUT_SECONDS
#   GIT_PULL_TIMEOUT_SECONDS
# Test hook: PGREP_SCOPE (regex prefix for every pgrep pattern; empty = production patterns).
set -uo pipefail

main() {
REPO="${REPO:-$HOME/pless-for-coding}"
BRANCH="${BRANCH:-feat/pless-variants-mps}"
CONFIGS="${CONFIGS-top_p_0_9 gk_0_45}"   # unset -> default; set-but-empty -> STOP
DRY_RUN="${DRY_RUN:-0}"
PULL="${PULL:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
SMOKE="${SMOKE:-0}"
EXPECT_HEAD="${EXPECT_HEAD:-}"
EXPECT_RUNNER_SHA256="${EXPECT_RUNNER_SHA256:-}"
EXPECT_GENVLLM_SHA256="${EXPECT_GENVLLM_SHA256:-}"
EXPECT_VLLM_VERSION="${EXPECT_VLLM_VERSION:-}"
EXPECT_TORCH_VERSION="${EXPECT_TORCH_VERSION:-}"
EXPECT_FLASHINFER_VERSION="${EXPECT_FLASHINFER_VERSION:-}"
EXPECT_ATCODER="${EXPECT_ATCODER:-121}"
EXPECT_LEETCODE="${EXPECT_LEETCODE:-90}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
GPU_WAIT_MAX_MINUTES="${GPU_WAIT_MAX_MINUTES:-0}"
LAUNCH_VERIFY_MINUTES="${LAUNCH_VERIFY_MINUTES:-25}"
POLL_SECONDS="${POLL_SECONDS:-10}"
CUDA_HOME_VAL="${CUDA_HOME_VAL:-/usr/local/cuda-13.1}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
GIT_BIN="${GIT_BIN:-git}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
PGREP_BIN="${PGREP_BIN:-pgrep}"
FLOCK_BIN="${FLOCK_BIN-$(command -v flock || true)}"
SETSID_BIN="${SETSID_BIN-$(command -v setsid || true)}"
TIMEOUT_BIN="${TIMEOUT_BIN-$(command -v timeout || true)}"
NVSMI_TIMEOUT_SECONDS="${NVSMI_TIMEOUT_SECONDS:-60}"
GIT_PULL_TIMEOUT_SECONDS="${GIT_PULL_TIMEOUT_SECONDS:-300}"
PGREP_SCOPE="${PGREP_SCOPE:-}"

MODEL="mistralai/Ministral-3-3B-Reasoning-2512"
MODEL_SLUG="mistralai--Ministral-3-3B-Reasoning-2512"
RUNNER="scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
GENVLLM="bench/generator_vllm.py"
SEED_VALUE=42
BATCH_VALUE=0
N_SAMPLES=10
QLOG="$LOG_DIR/msi_launch_ministral_batch0.log"
LOCKFILE="$LOG_DIR/msi_launch_ministral_batch0.lock"
TMPD=""
LOCK_MODE=""

# ---------------------------------------------------------------- 0. validation (before anything else)
# Paths reach generated scripts: restrict to a safe charset (they are also %q-quoted there).
[[ "$REPO" =~ ^/[A-Za-z0-9._/+-]+$ ]] || { echo "STOP: REPO must be an absolute path of [A-Za-z0-9._/+-] (got $(printf '%q' "$REPO")); nothing launched" >&2; exit 1; }
[[ "$LOG_DIR" =~ ^/[A-Za-z0-9._/+-]+$ ]] || { echo "STOP: LOG_DIR must be an absolute path of [A-Za-z0-9._/+-]; nothing launched" >&2; exit 1; }
mkdir -p "$LOG_DIR" || { echo "STOP: cannot create $LOG_DIR; nothing launched" >&2; exit 2; }
trap cleanup EXIT

[[ "$CUDA_HOME_VAL" =~ ^/[A-Za-z0-9._/+-]+$ ]] || stop "CUDA_HOME_VAL must be an absolute path of [A-Za-z0-9._/+-] (got $(printf '%q' "$CUDA_HOME_VAL"))"
[[ "$PYTHON_BIN" =~ ^/[A-Za-z0-9._/+-]+$ ]] || stop "PYTHON_BIN must be an absolute path of [A-Za-z0-9._/+-] (got $(printf '%q' "$PYTHON_BIN"))"
[[ "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ ]] || stop "BRANCH must match ^[A-Za-z0-9._/-]+\$ (got $(printf '%q' "$BRANCH"))"
local flag
for flag in DRY_RUN PULL ALLOW_EXISTING SMOKE; do
  [[ "${!flag}" == 0 || "${!flag}" == 1 ]] || stop "$flag must be 0 or 1 (got $(printf '%q' "${!flag}"))"
done
[[ "$SMOKE" == 0 ]] || stop "SMOKE=1 is not supported by this launcher (a smoke run would share the stratified-60 output dir, where top_p_0_9 files may already exist and the runner would resume over them); run without SMOKE"
local num
for num in GPU_FREE_MIB EXPECT_ATCODER EXPECT_LEETCODE POLL_SECONDS NVSMI_TIMEOUT_SECONDS GIT_PULL_TIMEOUT_SECONDS; do
  [[ "${!num}" =~ ^[0-9]+$ ]] || stop "$num must be a non-negative integer (got $(printf '%q' "${!num}"))"
done
(( POLL_SECONDS >= 1 )) || stop "POLL_SECONDS must be >= 1"
for num in GPU_WAIT_MAX_MINUTES LAUNCH_VERIFY_MINUTES; do
  [[ "${!num}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || stop "$num must be a non-negative number of minutes (got $(printf '%q' "${!num}"))"
done
for num in EXPECT_VLLM_VERSION EXPECT_TORCH_VERSION EXPECT_FLASHINFER_VERSION; do
  [[ "${!num}" =~ ^[A-Za-z0-9.+_-]*$ ]] || stop "$num must match ^[A-Za-z0-9.+_-]*\$ (got $(printf '%q' "${!num}"))"
done
[[ -n "$EXPECT_HEAD" ]] || stop "EXPECT_HEAD is REQUIRED (the full 40-hex sha you reviewed; get it with: git rev-parse HEAD)"
[[ "$EXPECT_HEAD" =~ ^[0-9a-f]{40}$ ]] || stop "EXPECT_HEAD must be a full 40-hex lower-case sha (got $(printf '%q' "$EXPECT_HEAD"))"
local pinvar
for pinvar in EXPECT_RUNNER_SHA256 EXPECT_GENVLLM_SHA256; do
  [[ -n "${!pinvar}" ]] || stop "$pinvar is REQUIRED (64 hex chars; no default)"
  [[ "${!pinvar}" =~ ^[0-9a-f]{64}$ ]] || stop "$pinvar is not 64 hex chars (lower-case): $(printf '%q' "${!pinvar}")"
done
# CONFIGS: space-separated names. Check the raw string first (no globbing / no other separators), then
# split with read -a (no pathname expansion).
[[ "$CONFIGS" =~ ^[a-z0-9_\ ]*$ ]] || stop "CONFIGS entry must match ^[a-z0-9_]+\$ (space-separated names); got $(printf '%q' "$CONFIGS")"
local -a CFGS=()
read -r -a CFGS <<< "$CONFIGS"
(( ${#CFGS[@]} > 0 )) || stop "CONFIGS is empty"
local i j
for i in "${!CFGS[@]}"; do
  [[ "${CFGS[$i]}" =~ ^[a-z0-9_]+$ ]] || stop "CONFIGS entry $(printf '%q' "${CFGS[$i]}") must match ^[a-z0-9_]+\$"
  for j in "${!CFGS[@]}"; do
    (( j < i )) && [[ "${CFGS[$j]}" == "${CFGS[$i]}" ]] && stop "duplicate config in CONFIGS: ${CFGS[$i]}"
  done
done
local CFG_LIST="${CFGS[*]}"
local CFG_TAG; CFG_TAG="$(printf '%s__' "${CFGS[@]}")"; CFG_TAG="${CFG_TAG%__}"
FULL_DIR="$REPO/results/pless_variants_cuda/$MODEL_SLUG/LCB_v6_full1046"
BATCH_CSV="$REPO/results/lcb_v6_full1046/batch_assignment.csv"
FULL_LOG="$LOG_DIR/ministral_fulldataset_batch0_${CFG_TAG}.log"
LAUNCH_PREFIX="$CUDA_HOME_VAL/bin:$REPO/.venv/bin"
LAUNCH_PATH="$LAUNCH_PREFIX:$PATH"
RUNNER_PAT="$(scoped 'run_pless_variants_ministral_3b_lcb_cuda\.py')"
QUEUE_PAT="$(scoped 'msi_queue[A-Za-z0-9_]*\.sh')"   # the launcher script, not e.g. 'tail -f logs/msi_queue_topk.log'
ENGINE_PAT="$(scoped 'VLLM::EngineCore')"

# ---------------------------------------------------------------- 1. lock
if [[ -n "$FLOCK_BIN" ]]; then
  exec 9>>"$LOCKFILE"
  "$FLOCK_BIN" -n 9 || stop "another launcher instance holds $LOCKFILE (flock)"
  LOCK_MODE=flock
else
  if ! ( set -o noclobber; echo "$$" > "$LOCKFILE" ) 2>/dev/null; then
    local holder; holder="$(cat "$LOCKFILE" 2>/dev/null | tr -cd '0-9')"
    if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
      stop "another launcher instance (pid $holder) holds $LOCKFILE (pidfile)"
    fi
    log "stale pidfile lock (holder '$holder' not alive); taking it over"
    echo "$$" > "$LOCKFILE"
  fi
  LOCK_MODE=pidfile
fi
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/msi_launch_b0.XXXXXX")" || stop "mktemp failed"

log "==== launcher start pid=$$ REPO=$REPO BRANCH=$BRANCH DRY_RUN=$DRY_RUN PULL=$PULL ALLOW_EXISTING=$ALLOW_EXISTING seed=$SEED_VALUE batch=$BATCH_VALUE lock=$LOCK_MODE"
log "config order: $CFG_LIST"
cd "$REPO" || stop "cannot cd to REPO=$REPO"

# ---------------------------------------------------------------- 2. nothing already running
local pat
for pat in "$RUNNER_PAT" "$QUEUE_PAT"; do
  if "$PGREP_BIN" -f "$pat" >/dev/null 2>&1; then
    stop "a runner / queue launcher is already running (pattern '$pat': $(proc_list "$pat")) -- refusing to launch"
  fi
done
log "no Ministral runner / msi_queue launcher running"

# ---------------------------------------------------------------- 3. code gate
local cur_branch dirty HEAD_SHA got_r got_g
cur_branch="$("$GIT_BIN" -C "$REPO" branch --show-current 2>&1)" || stop "git branch failed: $cur_branch"
[[ "$cur_branch" == "$BRANCH" ]] || stop "branch is '$cur_branch', expected '$BRANCH'"
dirty="$("$GIT_BIN" -C "$REPO" status --porcelain --untracked-files=no 2>&1)" || stop "git status failed: $dirty"
[[ -z "$dirty" ]] || stop "tracked files are dirty:
$dirty"
log "branch=$cur_branch, tracked tree clean; HEAD before pull: $("$GIT_BIN" -C "$REPO" rev-parse HEAD)"
if [[ "$PULL" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: would pull: $GIT_BIN -C $REPO pull --ff-only origin $BRANCH (HEAD below is the pre-pull HEAD)"
  else
    log "run: $GIT_BIN -C $REPO pull --ff-only origin $BRANCH"
    if [[ -n "$TIMEOUT_BIN" ]]; then
      GIT_TERMINAL_PROMPT=0 "$TIMEOUT_BIN" "$GIT_PULL_TIMEOUT_SECONDS" "$GIT_BIN" -C "$REPO" pull --ff-only origin "$BRANCH" >>"$QLOG" 2>&1 \
        || stop "git pull --ff-only origin $BRANCH failed (see above); no merge/rebase attempted"
    else
      GIT_TERMINAL_PROMPT=0 "$GIT_BIN" -C "$REPO" pull --ff-only origin "$BRANCH" >>"$QLOG" 2>&1 \
        || stop "git pull --ff-only origin $BRANCH failed (see above); no merge/rebase attempted"
    fi
  fi
else
  log "pull skipped (PULL=0); using the checked-out code as-is"
fi
HEAD_SHA="$("$GIT_BIN" -C "$REPO" rev-parse HEAD)"
[[ "$HEAD_SHA" == "$EXPECT_HEAD" ]] || stop "HEAD is $HEAD_SHA, expected EXPECT_HEAD=$EXPECT_HEAD"
log "HEAD matches EXPECT_HEAD ($HEAD_SHA)"
[[ -f "$REPO/$RUNNER" ]] || stop "runner missing: $REPO/$RUNNER"
[[ -f "$REPO/$GENVLLM" ]] || stop "generator missing: $REPO/$GENVLLM"
got_r="$(sha256_of "$REPO/$RUNNER")"; got_g="$(sha256_of "$REPO/$GENVLLM")"
check_pin "$RUNNER" "$EXPECT_RUNNER_SHA256" "$got_r"
check_pin "$GENVLLM" "$EXPECT_GENVLLM_SHA256" "$got_g"

# ---------------------------------------------------------------- 4. venv + versions
[[ -x "$PYTHON_BIN" ]] || stop "venv python missing or not executable: $PYTHON_BIN"
local vprobe vout V_PY V_VLLM V_TORCH V_FI V_FI2 V_TF
vprobe='import importlib.metadata as m, platform
print("python=" + platform.python_version())
for d in ("vllm", "torch", "flashinfer-python", "flashinfer", "transformers"):
    try:
        print(d + "=" + m.version(d))
    except m.PackageNotFoundError:
        print(d + "=MISSING")'
vout="$("$PYTHON_BIN" -I -c "$vprobe" 2>&1)" || stop "could not read package versions via $PYTHON_BIN: $vout"
V_PY="$(pkgv python "$vout")"; V_VLLM="$(pkgv vllm "$vout")"; V_TORCH="$(pkgv torch "$vout")"
V_FI="$(pkgv flashinfer-python "$vout")"; V_FI2="$(pkgv flashinfer "$vout")"; V_TF="$(pkgv transformers "$vout")"
log "versions: python=$V_PY vllm=$V_VLLM torch=$V_TORCH flashinfer-python=$V_FI flashinfer=$V_FI2 transformers=$V_TF (read from $PYTHON_BIN)"
[[ -n "$V_VLLM" && "$V_VLLM" != MISSING ]] || stop "vllm is not installed in $PYTHON_BIN's environment"
[[ -n "$V_TORCH" && "$V_TORCH" != MISSING ]] || stop "torch is not installed in $PYTHON_BIN's environment"
[[ "$V_FI" != MISSING || "$V_FI2" != MISSING ]] || stop "neither flashinfer-python nor flashinfer is installed (VLLM_USE_FLASHINFER_SAMPLER=1 is part of the recipe)"
[[ -z "$EXPECT_VLLM_VERSION" || "$V_VLLM" == "$EXPECT_VLLM_VERSION" ]] || stop "vllm version $V_VLLM != EXPECT_VLLM_VERSION=$EXPECT_VLLM_VERSION"
[[ -z "$EXPECT_TORCH_VERSION" || "$V_TORCH" == "$EXPECT_TORCH_VERSION" ]] || stop "torch version $V_TORCH != EXPECT_TORCH_VERSION=$EXPECT_TORCH_VERSION"
[[ -z "$EXPECT_FLASHINFER_VERSION" || "$V_FI" == "$EXPECT_FLASHINFER_VERSION" ]] || stop "flashinfer-python version $V_FI != EXPECT_FLASHINFER_VERSION=$EXPECT_FLASHINFER_VERSION"

# ---------------------------------------------------------------- 5. requested configs exist in the runner
local cfgparse cfgout c T
cfgparse='import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
rows = None
for node in tree.body:
    tgt = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
    if isinstance(tgt, ast.Name) and tgt.id == "CONFIGS" and getattr(node, "value", None) is not None:
        rows = ast.literal_eval(node.value)
if rows is None:
    sys.exit("no CONFIGS literal in runner")
for r in rows:
    print("%s %r" % (r[0], float(r[2])))'
cfgout="$("$PYTHON_BIN" -I -c "$cfgparse" "$REPO/$RUNNER" 2>&1)" || stop "could not parse CONFIGS from $RUNNER: $cfgout"
grep -q '^CONFIG_NAMES = \[c\[0\] for c in CONFIGS\]$' "$REPO/$RUNNER" || stop "runner's CONFIG_NAMES is no longer derived from CONFIGS -- refusing"
local -a OUTS=() OUT_LABELS=()
for c in "${CFGS[@]}"; do
  T="$(printf '%s\n' "$cfgout" | awk -v n="$c" '$1 == n {print $2; exit}')"
  [[ -n "$T" ]] || stop "config $c not in the runner's CONFIG_NAMES ($RUNNER at $HEAD_SHA)"
  [[ "$T" =~ ^[0-9]+\.[0-9]+$ ]] || stop "unexpected temperature '$T' for config $c"
  OUTS+=( "$FULL_DIR/${c}_atcoder_think_t${T}.jsonl" ); OUT_LABELS+=( "$c:atcoder" )
done
for c in "${CFGS[@]}"; do
  T="$(printf '%s\n' "$cfgout" | awk -v n="$c" '$1 == n {print $2; exit}')"
  OUTS+=( "$FULL_DIR/${c}_leetcode_think_t${T}.jsonl" ); OUT_LABELS+=( "$c:leetcode" )
done
local at_order lc_order
at_order="$(printf '%s -> ' "${CFGS[@]}")"; at_order="${at_order% -> }"
log "runner has every requested config; platform-major: atcoder[$at_order] then leetcode[$at_order]"
for i in "${!OUTS[@]}"; do log "output ${OUT_LABELS[$i]}: ${OUTS[$i]}"; done

# ---------------------------------------------------------------- 6. ninja + nvcc on the launch PATH
local ninja nvcc
ninja="$( PATH="$LAUNCH_PATH"; command -v ninja || true )"
nvcc="$( PATH="$LAUNCH_PATH"; command -v nvcc || true )"
[[ -n "$ninja" ]] || stop "ninja not found on the launch PATH (prefix $LAUNCH_PREFIX); FlashInfer JIT needs it"
[[ -n "$nvcc" ]] || stop "nvcc not found on the launch PATH (prefix $LAUNCH_PREFIX); FlashInfer JIT needs it"
log "ninja: $ninja ; nvcc: $nvcc (launch PATH prefix $LAUNCH_PREFIX)"

# ---------------------------------------------------------------- 7. batch-0 id file + counts
[[ -f "$BATCH_CSV" ]] || stop "batch assignment CSV missing: $BATCH_CSV"
local bc n_at n_lc
bc="$("$PYTHON_BIN" -I -c 'import csv, sys
c = {"atcoder": 0, "leetcode": 0}
for r in csv.DictReader(open(sys.argv[1])):
    if r["batch"].strip() == "0" and r["platform"] in c:
        c[r["platform"]] += 1
print(c["atcoder"], c["leetcode"])' "$BATCH_CSV" 2>&1)" || stop "could not read $BATCH_CSV: $bc"
n_at="${bc% *}"; n_lc="${bc#* }"
local bmsg="batch 0: atcoder=$n_at leetcode=$n_lc (expected $EXPECT_ATCODER / $EXPECT_LEETCODE)"
[[ "$n_at" == "$EXPECT_ATCODER" && "$n_lc" == "$EXPECT_LEETCODE" ]] || stop "$bmsg"
log "$bmsg"

# ---------------------------------------------------------------- 8. existing outputs
write_integrity_py "$TMPD/integrity_check.py"
local nonempty=0 ic_out ic_rc
for i in "${!OUTS[@]}"; do
  if [[ -s "${OUTS[$i]}" ]]; then nonempty=$((nonempty+1))
  elif [[ -f "${OUTS[$i]}" ]]; then log "existing output ${OUT_LABELS[$i]} exists but is empty: ${OUTS[$i]}"; fi
done
if (( nonempty > 0 )); then
  ic_out="$(run_integrity "$TMPD/integrity_check.py" 2>&1)"; ic_rc=$?
  printf '%s\n' "$ic_out" | sed 's/^/    /' | tee -a "$QLOG"
  (( ic_rc == 0 )) || stop "existing output failed the integrity check (see above) -- not resuming over it, fix by hand"
  [[ "$ALLOW_EXISTING" == "1" ]] || stop "$nonempty existing non-empty output file(s) for the requested configs (listed above); the runner would RESUME them by task_id. Re-run with ALLOW_EXISTING=1 if that is intended"
  log "ALLOW_EXISTING=1: existing output will be RESUMED by task_id ($nonempty non-empty file(s))"
else
  log "no existing non-empty output for the requested configs"
fi

# ---------------------------------------------------------------- 9. GPU free
wait_gpu_idle

# ---------------------------------------------------------------- 10. plan
local -a ENVV=( CUDA_HOME="$CUDA_HOME_VAL" PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=1 PATH="$LAUNCH_PATH" )
local -a CMD=( "$PYTHON_BIN" "$RUNNER" --model "$MODEL" --dtype bfloat16 --gpu-memory-utilization 0.92
               --max-model-len 37500 --n-samples "$N_SAMPLES" --max-new-tokens 35000
               --configs "${CFGS[@]}" --full-dataset --batch "$BATCH_VALUE"
               --budget-minutes inf --per-config-minutes inf --seed "$SEED_VALUE" )
local CMDLINE SETSID_Q=""
[[ -n "$SETSID_BIN" ]] && SETSID_Q="$(printf '%q ' "$SETSID_BIN")"
CMDLINE="$( printf 'cd %q && %snohup env' "$REPO" "$SETSID_Q"; printf ' %q' "${ENVV[@]}" "${CMD[@]}"; printf ' >> %q 2>&1 < /dev/null &' "$FULL_LOG" )"
local TS LDIR
TS="$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S)"
LDIR="$LOG_DIR/msi_launch_$TS"
if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY: would run: $CMDLINE"
  log "DRY: would write $LDIR/{manifest.json,cmdline.txt,resume.sh,integrity_check.sh,integrity_check.py,launched.txt}"
  log "DRY: would wait <= ${LAUNCH_VERIFY_MINUTES} min for a new '^Engine ready in' line in $FULL_LOG"
  log "==== DRY RUN complete (exit 0); nothing launched."
  exit 0
fi

# ---------------------------------------------------------------- 11. launch dir
"$PGREP_BIN" -f "$RUNNER_PAT" >/dev/null 2>&1 && stop "a Ministral runner appeared meanwhile -- refusing to launch a second"
mkdir "$LDIR" 2>/dev/null || stop "cannot create launch dir $LDIR (exists?)"
local START_IST; START_IST="$(ist)"
printf '%s\n' "$CMDLINE" > "$LDIR/cmdline.txt" || stop "cannot write $LDIR/cmdline.txt"
cp "$TMPD/integrity_check.py" "$LDIR/integrity_check.py" || stop "cannot write $LDIR/integrity_check.py"
write_integrity_sh "$LDIR/integrity_check.sh" "$LDIR/integrity_check.py" || stop "cannot write $LDIR/integrity_check.sh"
write_resume "$LDIR/resume.sh" || stop "cannot write $LDIR/resume.sh"
write_manifest "$LDIR/manifest.json" || stop "cannot write $LDIR/manifest.json"
log "launch dir: $LDIR (manifest.json, cmdline.txt, resume.sh, integrity_check.sh)"

# ---------------------------------------------------------------- 12. launch (detached) + Engine-ready wait
local log_off=0
[[ -f "$FULL_LOG" ]] && log_off="$(wc -c < "$FULL_LOG" | tr -d ' ')"
log "launch: $CMDLINE   (FULL_LOG byte offset before launch: $log_off)"
# 9>&- : the long-running child must not inherit the flock fd.
if [[ -n "$SETSID_BIN" ]]; then
  "$SETSID_BIN" nohup env "${ENVV[@]}" "${CMD[@]}" >>"$FULL_LOG" 2>&1 </dev/null 9>&- &
else
  nohup env "${ENVV[@]}" "${CMD[@]}" >>"$FULL_LOG" 2>&1 </dev/null 9>&- &
fi
local PID=$!
{ echo "PID=$PID"; echo "FULL_LOG=$FULL_LOG"; echo "START_IST=$START_IST"; echo "HEAD=$HEAD_SHA"; } > "$LDIR/launched.txt"
log "runner started: PID=$PID log=$FULL_LOG; waiting <= ${LAUNCH_VERIFY_MINUTES} min for '^Engine ready in'"
local vmax t0 ready lp
vmax="$(minutes_to_seconds "$LAUNCH_VERIFY_MINUTES")"; t0=$SECONDS
lp=$(( POLL_SECONDS < 10 ? POLL_SECONDS : 10 ))
while :; do
  ready="$(tail -c +$((log_off+1)) "$FULL_LOG" 2>/dev/null | grep -m1 '^Engine ready in' || true)"
  if [[ -n "$ready" ]]; then
    log "launch verified: '$ready' (PID=$PID log=$FULL_LOG launch_dir=$LDIR)"
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
log "HINT: progress/integrity (read-only): bash $LDIR/integrity_check.sh ; resume: bash $LDIR/resume.sh"
log "==== launcher done (exit 0); not waiting for the run"
exit 0
}

# ======================================================================= helpers
ist() { TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST'; }
log() { printf '[%s] %s\n' "$(ist)" "$*" | tee -a "$QLOG"; }
stop() { log "STOP: $*"; log "launcher exiting non-zero; nothing launched"; exit 1; }
cleanup() {
  [[ -n "${TMPD:-}" && -d "$TMPD" ]] && rm -rf -- "$TMPD"
  if [[ "${LOCK_MODE:-}" == pidfile && "$(cat "$LOCKFILE" 2>/dev/null)" == "$$" ]]; then rm -f -- "$LOCKFILE"; fi
  return 0
}
scoped() { if [[ -n "${PGREP_SCOPE:-}" ]]; then printf '%s.*%s' "$PGREP_SCOPE" "$1"; else printf '%s' "$1"; fi; }
pid_alive() { kill -0 "$1" 2>/dev/null || return 1; [[ "$(ps -o stat= -p "$1" 2>/dev/null)" != Z* ]]; }
proc_list() {
  local p out=""
  for p in $("$PGREP_BIN" -f "$1" 2>/dev/null); do out+="$p $(ps -o args= -p "$p" 2>/dev/null | cut -c1-200); "; done
  printf '%s' "${out:-none}"
}
pkgv() { printf '%s\n' "$2" | sed -n "s/^$1=//p" | tail -n 1; }
minutes_to_seconds() { awk -v m="$1" 'BEGIN{printf "%d", m*60}'; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
check_pin() {  # $1 path, $2 expected (validated 64-hex), $3 actual
  [[ "$2" == "$3" ]] || stop "sha256 mismatch for $1: expected=$2 actual=$3"
  log "pin: sha256 $1 = $3 (matches expected)"
}
gpu_used_mib() {  # max memory.used over all GPUs; empty on failure
  local out rc
  if [[ -n "$TIMEOUT_BIN" ]]; then
    out="$("$TIMEOUT_BIN" "$NVSMI_TIMEOUT_SECONDS" "$NVIDIA_SMI_BIN" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; rc=$?
  else
    out="$("$NVIDIA_SMI_BIN" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; rc=$?
  fi
  (( rc == 0 )) || return 0
  printf '%s\n' "$out" | awk 'BEGIN{m=""} /^[[:space:]]*[0-9]+/{v=$1+0; if(m==""||v>m)m=v} END{print m}'
}
wait_gpu_idle() {  # single check when GPU_WAIT_MAX_MINUTES=0, else poll up to the cap
  local n=0 mem cap t0=$SECONDS why
  cap="$(minutes_to_seconds "$GPU_WAIT_MAX_MINUTES")"
  while :; do
    n=$((n+1)); why=""
    mem="$(gpu_used_mib)"
    if [[ -z "$mem" ]]; then
      why="nvidia-smi query failed ($NVIDIA_SMI_BIN)"
    elif "$PGREP_BIN" -f "$ENGINE_PAT" >/dev/null 2>&1; then
      why="VLLM::EngineCore still present ($(proc_list "$ENGINE_PAT"))"
    elif (( mem >= GPU_FREE_MIB )); then
      why="GPU memory.used=${mem} MiB >= ${GPU_FREE_MIB}"
    fi
    if [[ -z "$why" ]]; then
      log "GPU idle: no EngineCore, memory.used=${mem} MiB < ${GPU_FREE_MIB}"
      return 0
    fi
    if (( SECONDS - t0 >= cap )); then
      if (( cap == 0 )); then stop "GPU not free: $why"; fi
      stop "GPU still held after ${GPU_WAIT_MAX_MINUTES} min: $why"
    fi
    (( n % 30 == 1 )) && log "[gpu] waiting: $why"
    sleep "$POLL_SECONDS"
  done
}
# Integrity arguments: csv, n_samples, then (config, platform, path) triples in launch order.
integrity_args() {
  local i
  printf '%s\n' "$BATCH_CSV" "$N_SAMPLES"
  for i in "${!OUTS[@]}"; do printf '%s\n' "${OUT_LABELS[$i]%%:*}" "${OUT_LABELS[$i]##*:}" "${OUTS[$i]}"; done
}
run_integrity() {  # $1 integrity_check.py (launcher-side, before the launch dir exists)
  local -a a=(); local line
  while IFS= read -r line; do a+=( "$line" ); done < <(integrity_args)
  "$PYTHON_BIN" -I "$1" "${a[@]}"
}
write_integrity_py() {  # $1 path; stdlib-only, read-only checker
  cat > "$1" <<'PY'
"""Read-only integrity / progress check of the Ministral full1046 batch-0 outputs.

argv: batch_assignment.csv n_samples (config platform path)...
Per (config, platform): lines, distinct task_ids, unparseable, duplicated, trailing newline, wrong
method, wrong sample count, ids unknown to the CSV or of the other platform (problems), records of
other batches (tolerated: the full1046 files are shared by all batches), batch-0 done / missing.
Exit 1 on any problem; with REQUIRE_COMPLETE=1 also when any batch-0 id is missing.
"""
import csv
import json
import os
import sys

csv_path, n_samples = sys.argv[1], int(sys.argv[2])
triples = [sys.argv[i:i + 3] for i in range(3, len(sys.argv), 3)]
plat_of, batch_of = {}, {}
with open(csv_path) as f:
    for r in csv.DictReader(f):
        plat_of[r["task_id"]] = r["platform"]
        batch_of[r["task_id"]] = r["batch"].strip()
problems = incomplete = 0
for cfg, plat, path in triples:
    want = {t for t, p in plat_of.items() if p == plat and batch_of[t] == "0"}
    if not os.path.exists(path):
        print(f"{cfg}/{plat}: MISSING {path} batch0_done=0/{len(want)}")
        incomplete += 1
        continue
    data = open(path, "rb").read()
    trailing = "n/a" if not data else ("yes" if data.endswith(b"\n") else "no")
    ids, bad, wrong_method, bad_samples = [], 0, 0, 0
    for raw in data.split(b"\n"):
        if not raw.strip():
            continue
        try:
            r = json.loads(raw)
            ids.append(str(r["task_id"]))
        except Exception:
            bad += 1
            continue
        wrong_method += r.get("method") != cfg
        bad_samples += len(r.get("samples") or []) != n_samples
    seen, dups = set(), set()
    for t in ids:
        (dups if t in seen else seen).add(t)
    unknown = sorted(t for t in seen if t not in plat_of)
    wrong_plat = sorted(t for t in seen if t in plat_of and plat_of[t] != plat)
    other = sorted(t for t in seen if t in plat_of and plat_of[t] == plat and batch_of[t] != "0")
    done = want & seen
    missing = len(want) - len(done)
    print(f"{cfg}/{plat}: lines={len(ids) + bad} distinct={len(seen)} unparseable={bad} duplicated={len(dups)} "
          f"trailing_newline={trailing} wrong_method={wrong_method} bad_samples={bad_samples} "
          f"unknown_ids={len(unknown)} wrong_platform={len(wrong_plat)} other_batches={len(other)} "
          f"batch0_done={len(done)}/{len(want)} missing={missing}  ({path})")
    if dups:
        print(f"  duplicated (<=10): {sorted(dups)[:10]}")
    if unknown or wrong_plat:
        print(f"  unknown/wrong-platform (<=10): {(unknown + wrong_plat)[:10]}")
    if bad or dups or trailing == "no" or wrong_method or bad_samples or unknown or wrong_plat:
        problems += 1
    if missing:
        incomplete += 1
req = os.environ.get("REQUIRE_COMPLETE", "0") == "1"
status = "PROBLEMS" if problems else ("INCOMPLETE" if (incomplete and req) else "OK")
print(f"INTEGRITY {status}: files_with_problems={problems} incomplete_files={incomplete} require_complete={int(req)}")
sys.exit(1 if problems or (req and incomplete) else 0)
PY
}
integrity_cmd() {  # the full checker command line, %q-quoted; $1 = path of integrity_check.py
  local line
  printf '%q -I %q' "$PYTHON_BIN" "$1"
  while IFS= read -r line; do printf ' %q' "$line"; done < <(integrity_args)
}
write_integrity_sh() {  # $1 path, $2 integrity_check.py
  cat > "$1" <<EOF
#!/usr/bin/env bash
# Read-only integrity / progress check of the Ministral full1046 batch-0 run ($CFG_LIST).
# Generated by scripts/msi_launch_ministral_batch0.sh at $(ist); HEAD=$HEAD_SHA
# Exit 1 on problems (unparseable / duplicate / missing trailing newline / wrong method / wrong sample count /
# unknown ids); REQUIRE_COMPLETE=1 also fails while any batch-0 id is missing.
$(integrity_cmd "$2")
EOF
  chmod +x "$1"
}
write_resume() {  # $1 path
  local q_repo q_git q_pgrep q_nv
  q_repo="$(printf '%q' "$REPO")"; q_git="$(printf '%q' "$GIT_BIN")"; q_pgrep="$(printf '%q' "$PGREP_BIN")"
  q_nv="$(printf '%q' "$NVIDIA_SMI_BIN")"
  local i outs_q=""
  for i in "${!OUTS[@]}"; do outs_q+=" $(printf '%q' "${OUTS[$i]}")"; done
  local launch_q
  launch_q="$( printf '%snohup env' "$SETSID_Q"; printf ' %q' "${ENVV[@]:0:4}"; printf ' PATH=%q:"$PATH"' "$LAUNCH_PREFIX"
               printf ' %q' "${CMD[@]}"; printf ' >> %q 2>&1 < /dev/null &' "$FULL_LOG" )"
  cat > "$1" <<EOF
#!/usr/bin/env bash
# Resume the Ministral full1046 batch-0 run ($CFG_LIST); the runner resumes by task_id.
# Generated by scripts/msi_launch_ministral_batch0.sh at $(ist); HEAD=$HEAD_SHA
# Refuses if: a Ministral runner / msi_queue launcher is running, HEAD or the pinned sha256s changed, the
# integrity check reports problems, the GPU is busy (FORCE=1 overrides only the GPU check), or existing
# non-empty outputs are present without ALLOW_EXISTING=1.
set -u
cd $q_repo || exit 1
if $q_pgrep -f $(printf '%q' "$RUNNER_PAT") >/dev/null 2>&1; then echo "a Ministral runner is already running; refusing" >&2; exit 1; fi
if $q_pgrep -f $(printf '%q' "$QUEUE_PAT") >/dev/null 2>&1; then echo "an msi_queue launcher is already running; refusing" >&2; exit 1; fi
head="\$($q_git rev-parse HEAD 2>/dev/null)"
if [[ "\$head" != $(printf '%q' "$HEAD_SHA") ]]; then echo "HEAD is \$head, launch HEAD was $HEAD_SHA; refusing" >&2; exit 1; fi
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "\$1" | awk '{print \$1}'; else shasum -a 256 "\$1" | awk '{print \$1}'; fi; }
[[ "\$(sha $(printf '%q' "$RUNNER"))" == $got_r ]] || { echo "runner sha256 changed since launch; refusing" >&2; exit 1; }
[[ "\$(sha $(printf '%q' "$GENVLLM"))" == $got_g ]] || { echo "generator_vllm sha256 changed since launch; refusing" >&2; exit 1; }
if ! $(integrity_cmd "$LDIR/integrity_check.py") >&2; then echo "integrity check reported problems; refusing" >&2; exit 1; fi
used="\$($q_nv --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{m=""} /^[[:space:]]*[0-9]+/{v=\$1+0; if(m==""||v>m)m=v} END{print m}')"
if [[ -z "\$used" ]]; then echo "cannot read GPU memory from nvidia-smi; refusing" >&2; [[ "\${FORCE:-0}" == 1 ]] || exit 1
elif (( used >= $GPU_FREE_MIB )); then echo "GPU busy: memory.used=\$used MiB >= $GPU_FREE_MIB; refusing" >&2; [[ "\${FORCE:-0}" == 1 ]] || exit 1; fi
if $q_pgrep -f $(printf '%q' "$ENGINE_PAT") >/dev/null 2>&1; then echo "VLLM::EngineCore still present; refusing" >&2; [[ "\${FORCE:-0}" == 1 ]] || exit 1; fi
n=0; for f in$outs_q; do [[ -s "\$f" ]] && n=\$((n+1)); done
if (( n > 0 )) && [[ "\${ALLOW_EXISTING:-0}" != 1 ]]; then echo "\$n existing non-empty output file(s); set ALLOW_EXISTING=1 to resume them; refusing" >&2; exit 1; fi
$launch_q
echo "relaunched PID \$! (log $(printf '%q' "$FULL_LOG"))"
EOF
  chmod +x "$1"
}
write_manifest() {  # $1 path. Values as argv; the environment snapshot is read by python (filtered).
  local -a kv=( "seed=$SEED_VALUE" "batch=$BATCH_VALUE" "model=$MODEL" "start_time_ist=$START_IST"
    "launch_dir=$LDIR" "repo=$REPO" "branch=$BRANCH" "git_head=$HEAD_SHA" "full_log=$FULL_LOG"
    "sha256.$RUNNER=$got_r" "sha256.$GENVLLM=$got_g"
    "versions.python=$V_PY" "versions.vllm=$V_VLLM" "versions.torch=$V_TORCH" "versions.flashinfer-python=$V_FI"
    "versions.flashinfer=$V_FI2" "versions.transformers=$V_TF" "python_bin=$PYTHON_BIN" "ninja=$ninja" "nvcc=$nvcc"
    "launch_path_prefix=$LAUNCH_PREFIX" "batch0_counts.atcoder=$n_at" "batch0_counts.leetcode=$n_lc"
    "command=$CMDLINE" "allow_existing=$ALLOW_EXISTING" "hostname=$(hostname 2>/dev/null || echo '?')" )
  local c i
  for c in "${CFGS[@]}"; do kv+=( "configs[]=$c" ); done
  for i in "${!OUTS[@]}"; do kv+=( "outputs[]=${OUTS[$i]}" ); done
  env "${ENVV[@]}" "$PYTHON_BIN" -I -c '
import json, os, re, sys
out, m = sys.argv[1], {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    if k.endswith("[]"):
        m.setdefault(k[:-2], []).append(v)
        continue
    if k in ("seed", "batch") or k.startswith("batch0_counts."):
        v = int(v)
    head, _, rest = k.partition(".")
    if rest and head in ("sha256", "versions", "batch0_counts"):
        m.setdefault(head, {})[rest] = v
    else:
        m[k] = v
cred = re.compile("TOKEN|KEY|SECRET|PASS|CRED|AUTH")
keep = re.compile(r"^(VLLM_|CUDA_|HF_|PYTHON)")
m["env_snapshot"] = sorted("%s=%s" % (k, v) for k, v in os.environ.items()
                           if keep.match(k) and not cred.search(k.upper()))
m["env_snapshot_note"] = ("runner environment filtered to VLLM_/CUDA_/HF_/PYTHON names; any name matching "
                          "TOKEN|KEY|SECRET|PASS|CRED|AUTH is dropped entirely (name and value)")
with open(out, "x") as fh:
    fh.write(json.dumps(m, indent=2) + "\n")
' "$1" "${kv[@]}"
}

# Whole body is parsed before anything runs, so editing this file mid-run cannot change what runs.
main "$@"; exit $?
