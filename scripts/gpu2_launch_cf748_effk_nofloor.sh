#!/usr/bin/env bash
# gpu2_launch_cf748_effk_nofloor.sh -- GPU2 launcher: Qwen3-8B CF-interview-748, arm pless_effk_nofloor,
# seed 42, tasks split across the two GPUs (plan: notes/effk_nofloor_cf748_plan.md, Option A).
#
# Launches two detached processes (setsid + nohup), one per GPU, each running the UNCHANGED P1 pipeline
#   run_cf748_apps_qwen3.sh  (-> python -m bench.apps --backend vllm ... --method pless_effk_nofloor)
# with GPU0 <- the EVEN half and GPU1 <- the ODD half of data/cf_interview_748_ids.txt (sorted ids,
# even/odd by position; make the halves first with scripts/cf748_split_ids.py split), each into its own
# RESULTS_DIR, and env:
#   VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=1 SKIP_EVAL=1 SEED=42 ARMS=pless_effk_nofloor
#   GPU_MEM_UTIL (default 0.96, the run script's --gpu-memory-utilization passthrough)
#   MODEL=Qwen/Qwen3-8B SOURCE=CODEFORCES DIFFICULTY=interview N_SAMPLES=10 MAX_TOKENS=32768 (explicit =
#   the run script's defaults); MULTI_ARM / MAX_PROBLEMS / CUDA_VISIBLE_DEVICES /
#   VLLM_WORKER_MULTIPROC_METHOD are removed (env -u). PATH=LAUNCH_PATH (venv bin first; FlashInfer JIT
#   needs ninja).
# The output records do NOT store the seed: manifest.json in the launch dir is the record of seed,
# versions, git HEAD, sha256 pins and split hashes.
#
# Gates (any failure -> "STOP" in the log, non-zero exit, nothing launched):
#   config sane; lock; VLLM_WORKER_MULTIPROC_METHOD != spawn (register() runs in the parent only);
#   no runner / run script / Ministral runner / earlier wrapper running; GPU0 and GPU1 memory.used <
#   GPU_FREE_MIB; branch == BRANCH, tracked tree clean, HEAD == EXPECT_HEAD (if set); runner has
#   pless_effk_nofloor + register(); run script has the pless_effk_nofloor arm + SEED passthrough;
#   REQUIRED runner sha256 pin (EXPECT_RUNNER_SHA256, 'any' = loud warning); venv python exists and
#   vllm / torch / flashinfer-python versions (read from the venv's package metadata) are exactly
#   0.32.0 / 2.13.0+cu130 / 0.7.0.post1; ninja on LAUNCH_PATH; id files exist, disjoint, union == the
#   748 ids (re-checked with scripts/cf748_split_ids.py verify). Existing results are only logged
#   (the runner resumes by task_id).
#
# Modes:
#   DRY_RUN=1  run every read-only gate, print every command / file it would write, launch nothing.
#   SMOKE=1    same gates, but ONE process on SMOKE_GPU (default 0) over SMOKE_IDS (default: the first
#              2 sorted ids of the 748 list; must be a subset of it), into a fresh timestamped dir under
#              SMOKE_RESULTS_ROOT. For the register()-under-vLLM-0.32.0 check and a seeded rerun
#              (run SMOKE=1 twice; integrity_check.sh prints per-task sample hashes to compare).
#
# Run on GPU2 (the launcher detaches itself; it prints the log path and returns at once):
#   cd /home/karrtik/pless-for-coding
#   uv run python scripts/cf748_split_ids.py split --out-dir data/cf748_split   # once
#   EXPECT_RUNNER_SHA256=<sha256 of bench/apps/runner.py> DRY_RUN=1 bash scripts/gpu2_launch_cf748_effk_nofloor.sh
#   EXPECT_RUNNER_SHA256=<sha256> SMOKE=1 bash scripts/gpu2_launch_cf748_effk_nofloor.sh
#   EXPECT_RUNNER_SHA256=<sha256> bash scripts/gpu2_launch_cf748_effk_nofloor.sh
#   tail -f logs/gpu2_launch_cf748_effk.log
# Stub-based tests: uv run pytest tests/test_gpu2_launch_cf748_effk_nofloor.py
#
# Overridable env: REPO BRANCH EXPECT_HEAD DRY_RUN SMOKE SMOKE_IDS SMOKE_GPU VLLM_VENV CHECK_PYTHON
#   EXPECT_VLLM_VERSION EXPECT_TORCH_VERSION EXPECT_FLASHINFER_VERSION EXPECT_RUNNER_SHA256 IDS_FILE
#   SPLIT_DIR IDS_EVEN IDS_ODD RESULTS_ROOT SMOKE_RESULTS_ROOT GPU_MEM_UTIL MAX_MODEL_LEN GPU_FREE_MIB
#   LOG_DIR LAUNCH_PATH STAGGER_SECONDS LAUNCH_CHECK_SECONDS GIT_BIN NVIDIA_SMI_BIN PGREP_BIN TIMEOUT_BIN
#   SETSID_BIN FLOCK_BIN NVSMI_TIMEOUT_SECONDS
# MODEL (default Qwen/Qwen3-8B; every default path / lock / log name is unchanged for it). Any other MODEL
#   (e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-8B) gets a model slug (lower-cased basename) in the default
#   RESULTS_ROOT / SMOKE_RESULTS_ROOT, the launcher log + lock and the per-GPU wrapper log names, so it cannot
#   collide with a Qwen launch; it REQUIRES MAX_MODEL_LEN (passed through to the run script / vLLM). The
#   runner is still run_cf748_apps_qwen3.sh (the pless_effk_nofloor arm is model-neutral; its TOKENIZER is
#   only used by the eval step, which SKIP_EVAL=1 skips).
# ARM (default pless_effk_nofloor; every default path / lock / log name / manifest / generated script is
#   unchanged for it). Allow-list: topp0.9 (run-script arm -> --method temp --top-p 0.9 --top-k 0
#   --temperature 1.0 -> temp_p0.9_think_t1.0_t1.0.jsonl) and renyi_k0.35 (--method pless_renyi --renyi-k
#   0.35 --temperature 1.0 -> pless_renyi_think_t1.0_k0.35_t1.0.jsonl); anything else STOPs. A non-default
#   arm appends "_<arm>" after the model tag to the default RESULTS_ROOT / SMOKE_RESULTS_ROOT, the launcher
#   log + lock, the launch dir and the wrapper logs; its gate checks the runner has the method and the run
#   script has the exact arm_args line (the effk register() gate applies to the effk arm only); the
#   integrity/merge commands get --method <method> --filename <output file>; the manifest gets method /
#   output_file. The busy-process gate is arm/model-neutral: any CF-748 run blocks any other launch.
# Test hooks: GPU2_LAUNCHER_DETACHED=1 (stay in foreground), PGREP_SCOPE (regex prefix for every
#   pgrep pattern; empty = production patterns).
set -uo pipefail

main() {
REPO="${REPO:-/home/karrtik/pless-for-coding}"
BRANCH="${BRANCH:-feat/pless-variants-mps}"
EXPECT_HEAD="${EXPECT_HEAD:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SMOKE_IDS="${SMOKE_IDS:-}"
SMOKE_GPU="${SMOKE_GPU:-0}"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm032_env/.venv}"
CHECK_PYTHON="${CHECK_PYTHON:-$VLLM_VENV/bin/python}"   # stdlib-only helpers, always run with -I
EXPECT_VLLM_VERSION="${EXPECT_VLLM_VERSION:-0.32.0}"
EXPECT_TORCH_VERSION="${EXPECT_TORCH_VERSION:-2.13.0+cu130}"
EXPECT_FLASHINFER_VERSION="${EXPECT_FLASHINFER_VERSION:-0.7.0.post1}"
EXPECT_RUNNER_SHA256="${EXPECT_RUNNER_SHA256:-}"
IDS_FILE="${IDS_FILE:-$REPO/data/cf_interview_748_ids.txt}"
SPLIT_DIR="${SPLIT_DIR:-$REPO/data/cf748_split}"
IDS_EVEN="${IDS_EVEN:-$SPLIT_DIR/cf748_even_ids.txt}"
IDS_ODD="${IDS_ODD:-$SPLIT_DIR/cf748_odd_ids.txt}"
DEFAULT_MODEL="Qwen/Qwen3-8B"
MODEL="${MODEL:-$DEFAULT_MODEL}"
MODEL_DIR="${MODEL//\//--}"   # same derivation as run_cf748_apps_qwen3.sh / cf748_merge_validate.py
MODEL_TAG=""                   # empty for the default model -> paths byte-identical to the Qwen launches
MODEL_ARG=""                   # extra cf748_merge_validate.py args (its --model default is Qwen/Qwen3-8B)
if [[ "$MODEL" != "$DEFAULT_MODEL" ]]; then
  MODEL_TAG="_$(printf '%s' "${MODEL##*/}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9._-' '-')"
  MODEL_ARG=" --model $(printf '%q' "$MODEL")"
fi
DEFAULT_ARM="pless_effk_nofloor"
ARM="${ARM-$DEFAULT_ARM}"       # unset -> default; set-but-empty is rejected below (allow-list)
ARM_TAG=""                     # empty for the default arm -> paths byte-identical to the effk launches
ARM_OK=1
case "$ARM" in   # allow-list; METHOD / OUT_FILE = what bench.apps.runner (_method_key + _output_path) writes
  pless_effk_nofloor) METHOD="pless_effk_nofloor"; OUT_FILE="pless_effk_nofloor_think_t1.0_t1.0.jsonl" ;;
  topp0.9)            METHOD="temp";               OUT_FILE="temp_p0.9_think_t1.0_t1.0.jsonl" ;;
  renyi_k0.35)        METHOD="pless_renyi";        OUT_FILE="pless_renyi_think_t1.0_k0.35_t1.0.jsonl" ;;
  *)                  ARM_OK=0; METHOD=""; OUT_FILE="" ;;
esac
FILE_ARG=""                    # extra cf748_merge_validate.py arg (its default filename only fits plain methods)
if (( ! ARM_OK )); then
  ARM_TAG="_invalid-arm"       # fixed tag: a rejected ARM never reaches a path, log or lock name
elif [[ "$ARM" != "$DEFAULT_ARM" ]]; then
  ARM_TAG="_$(printf '%s' "$ARM" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9._-' '-')"
  FILE_ARG=" --filename $(printf '%q' "$OUT_FILE")"
fi
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results/_cf748_effk_nofloor_split${MODEL_TAG}${ARM_TAG}}"
SMOKE_RESULTS_ROOT="${SMOKE_RESULTS_ROOT:-$REPO/results/_cf748_effk_nofloor_smoke${MODEL_TAG}${ARM_TAG}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.96}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
LAUNCH_PATH="${LAUNCH_PATH:-$VLLM_VENV/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
STAGGER_SECONDS="${STAGGER_SECONDS:-300}"
LAUNCH_CHECK_SECONDS="${LAUNCH_CHECK_SECONDS:-30}"
GIT_BIN="${GIT_BIN:-git}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
PGREP_BIN="${PGREP_BIN:-pgrep}"
TIMEOUT_BIN="${TIMEOUT_BIN-$(command -v timeout || true)}"
SETSID_BIN="${SETSID_BIN-$(command -v setsid || true)}"
FLOCK_BIN="${FLOCK_BIN-$(command -v flock || true)}"
NVSMI_TIMEOUT_SECONDS="${NVSMI_TIMEOUT_SECONDS:-60}"
PGREP_SCOPE="${PGREP_SCOPE:-}"

SEED_VALUE=42
RUNNER="bench/apps/runner.py"
RUN_SCRIPT="run_cf748_apps_qwen3.sh"
PINNED_FILES=( "$RUNNER" "$RUN_SCRIPT" bench/generator_vllm.py bench/pless_variants_vllm.py
               scripts/cf748_split_ids.py scripts/cf748_merge_validate.py scripts/gpu2_launch_cf748_effk_nofloor.sh )
QLOG="$LOG_DIR/gpu2_launch_cf748_effk${MODEL_TAG}${ARM_TAG}.log"
LOCKFILE="$LOG_DIR/gpu2_launch_cf748_effk${MODEL_TAG}${ARM_TAG}.lock"
BUSY_PATS=( "$(scoped ' -m bench\.apps')" "$(scoped 'run_cf748_apps_qwen3\.sh')"
            "$(scoped 'run_pless_variants_')" "$(scoped 'cf748_effk_launch_.*/run_gpu[01]\.sh')" )
ENV_UNSET=( -u MULTI_ARM -u MAX_PROBLEMS -u CUDA_VISIBLE_DEVICES -u VLLM_WORKER_MULTIPROC_METHOD )
LAUNCHED_NOTE=""
ENV_BIN="$(command -v env)"

mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR" >&2; exit 2; }

# ---------------------------------------------------------------- 0. config (validated before detaching)
# MODEL reaches generated scripts (paths): only an "org/name" of [A-Za-z0-9._-] is accepted. (MODEL_TAG above
# is already tr-sanitised, so the log/lock names used by this stop() are safe even for a rejected MODEL.)
[[ "$MODEL" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] \
  || stop "MODEL must match ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\$ (an HF 'org/name' id; got '$MODEL')"
# ARM selects the run script's arm (ARMS=) and reaches paths: only the allow-listed arms are accepted.
(( ARM_OK )) || stop "ARM must be one of: pless_effk_nofloor (default) topp0.9 renyi_k0.35 (got '$ARM')"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || stop "DRY_RUN must be 0 or 1 (got '$DRY_RUN')"
[[ "$SMOKE" == 0 || "$SMOKE" == 1 ]] || stop "SMOKE must be 0 or 1 (got '$SMOKE')"
[[ "$SMOKE_GPU" == 0 || "$SMOKE_GPU" == 1 ]] || stop "SMOKE_GPU must be 0 or 1 (got '$SMOKE_GPU')"
[[ "$GPU_MEM_UTIL" =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v u="$GPU_MEM_UTIL" 'BEGIN{exit !(u>0 && u<=1)}' \
  || stop "GPU_MEM_UTIL must be in (0, 1] (got '$GPU_MEM_UTIL')"
[[ -z "$MAX_MODEL_LEN" || "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || stop "MAX_MODEL_LEN must be empty or an integer (got '$MAX_MODEL_LEN')"
[[ "$MODEL" == "$DEFAULT_MODEL" || -n "$MAX_MODEL_LEN" ]] \
  || stop "MODEL=$MODEL is not the default $DEFAULT_MODEL: set MAX_MODEL_LEN explicitly (prompt + MAX_TOKENS=32768 must fit)"
[[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || stop "STAGGER_SECONDS must be whole seconds (got '$STAGGER_SECONDS')"
[[ "$LAUNCH_CHECK_SECONDS" =~ ^[0-9]+$ ]] || stop "LAUNCH_CHECK_SECONDS must be whole seconds (got '$LAUNCH_CHECK_SECONDS')"
[[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]] || stop "GPU_FREE_MIB must be an integer (got '$GPU_FREE_MIB')"

# ---------------------------------------------------------------- 1. detach (survive ssh drop)
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

# ---------------------------------------------------------------- 2. lock
if [[ -n "$FLOCK_BIN" ]]; then
  exec 9>>"$LOCKFILE"
  "$FLOCK_BIN" -n 9 || stop "another launcher instance holds $LOCKFILE (flock)"
else
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

log "==== launcher start pid=$$ REPO=$REPO BRANCH=$BRANCH DRY_RUN=$DRY_RUN SMOKE=$SMOKE arm=$ARM seed=$SEED_VALUE lock=${FLOCK_BIN:+flock}${FLOCK_BIN:-pidfile}"
log "seed=$SEED_VALUE: the output records do NOT store the seed; the launch-dir manifest.json is the record"
[[ "$ARM" != "$DEFAULT_ARM" ]] && log "arm=$ARM -> method=$METHOD output_file=$OUT_FILE (run script ARMS=$ARM)"
cd "$REPO" || stop "cannot cd to REPO=$REPO"

# ---------------------------------------------------------------- 3. multiprocessing start method
if [[ "${VLLM_WORKER_MULTIPROC_METHOD:-}" == "spawn" ]]; then
  stop "VLLM_WORKER_MULTIPROC_METHOD=spawn: bench.pless_variants_vllm.register() runs only in the parent, a spawned engine core would not see pless_effk_nofloor"
fi
log "VLLM_WORKER_MULTIPROC_METHOD='${VLLM_WORKER_MULTIPROC_METHOD:-}' (removed from the runner env; vLLM default applies; smoke-test the start method on 0.32.0)"

# ---------------------------------------------------------------- 4. nothing already running
local pat
for pat in "${BUSY_PATS[@]}"; do
  if "$PGREP_BIN" -f "$pat" >/dev/null 2>&1; then
    stop "a runner/wrapper process is already running (pattern '$pat': $(proc_list "$pat")) -- refusing to launch"
  fi
done
log "no runner / run script / wrapper process running (patterns: ${BUSY_PATS[*]})"

# ---------------------------------------------------------------- 5. both GPUs free
check_gpus_free

# ---------------------------------------------------------------- 6. code gate
local cur_branch dirty HEAD_SHA got_r f
cur_branch="$("$GIT_BIN" -C "$REPO" branch --show-current 2>&1)" || stop "git branch failed: $cur_branch"
[[ "$cur_branch" == "$BRANCH" ]] || stop "branch is '$cur_branch', expected '$BRANCH'"
dirty="$("$GIT_BIN" -C "$REPO" status --porcelain --untracked-files=no 2>&1)" || stop "git status failed: $dirty"
[[ -z "$dirty" ]] || stop "tracked files are dirty:
$dirty"
HEAD_SHA="$("$GIT_BIN" -C "$REPO" rev-parse HEAD)"
if [[ -n "$EXPECT_HEAD" ]]; then
  [[ "$HEAD_SHA" == "$EXPECT_HEAD" ]] || stop "HEAD is $HEAD_SHA, expected EXPECT_HEAD=$EXPECT_HEAD"
  log "HEAD matches EXPECT_HEAD"
fi
log "branch=$cur_branch, tracked tree clean; HEAD=$HEAD_SHA (no pull performed by this launcher)"
[[ -f "$REPO/$RUNNER" ]] || stop "runner missing: $REPO/$RUNNER"
[[ -f "$REPO/$RUN_SCRIPT" ]] || stop "run script missing: $REPO/$RUN_SCRIPT"
if [[ "$ARM" == "$DEFAULT_ARM" ]]; then
  grep -q "\"$ARM\"" "$REPO/$RUNNER" && grep -q "register()" "$REPO/$RUNNER" \
    || stop "runner lacks '$ARM' / register() -- old code, refusing to launch"
  grep -q "^ *$ARM)" "$REPO/$RUN_SCRIPT" || stop "$RUN_SCRIPT lacks the $ARM arm -- old code, refusing to launch"
  grep -q 'SEED:+--seed' "$REPO/$RUN_SCRIPT" || stop "$RUN_SCRIPT lacks the SEED passthrough -- old code, refusing to launch"
  log "runner has $ARM + register(); $RUN_SCRIPT has the $ARM arm + SEED passthrough"
else
  # Built-in method (no register()): the runner must know the method; the run script's arm_args case
  # line for this arm must be exactly the expected one (glob label + the args the output name assumes).
  local arm_line
  case "$ARM" in
    topp0.9)     arm_line='topp*)      echo "--method temp --top-p ${1#topp} --top-k 0 --temperature 1.0" ;;' ;;
    renyi_k0.35) arm_line='renyi_k*)   echo "--method pless_renyi --renyi-k ${1#renyi_k} --temperature 1.0" ;;' ;;
  esac
  grep -q "\"$METHOD\"" "$REPO/$RUNNER" || stop "runner lacks method '$METHOD' (arm $ARM) -- refusing to launch"
  grep -qxF "    $arm_line" "$REPO/$RUN_SCRIPT" \
    || stop "$RUN_SCRIPT lacks the $ARM arm (expected arm_args line: $arm_line) -- refusing to launch"
  grep -q 'SEED:+--seed' "$REPO/$RUN_SCRIPT" || stop "$RUN_SCRIPT lacks the SEED passthrough -- old code, refusing to launch"
  log "runner has method $METHOD; $RUN_SCRIPT has the $ARM arm ($arm_line) + SEED passthrough"
fi
local -a SHAS=()   # parallel to PINNED_FILES (indexed: bash 3.2 has no associative arrays)
local k
for k in "${!PINNED_FILES[@]}"; do
  SHAS[$k]="$(sha256_of "$REPO/${PINNED_FILES[$k]}" 2>/dev/null || echo MISSING)"
  log "sha256 ${PINNED_FILES[$k]} = ${SHAS[$k]}"
done
got_r="${SHAS[0]}"   # PINNED_FILES[0] is the runner
[[ -n "$EXPECT_RUNNER_SHA256" ]] || stop "EXPECT_RUNNER_SHA256 is REQUIRED (64 hex chars, or 'any'); actual sha256 of $RUNNER is $got_r"
check_pin "$RUNNER" "$EXPECT_RUNNER_SHA256" "$got_r"

# ---------------------------------------------------------------- 7. venv + versions
[[ -x "$VLLM_VENV/bin/python" ]] || stop "VLLM_VENV python missing or not executable: $VLLM_VENV/bin/python"
local vprobe vout v_vllm v_torch v_fi
vprobe='import importlib.metadata as m
for d in ("vllm", "torch", "flashinfer-python"):
    try:
        print(d + "=" + m.version(d))
    except m.PackageNotFoundError:
        print(d + "=MISSING")'
vout="$("$VLLM_VENV/bin/python" -I -c "$vprobe" 2>&1)" || stop "could not read package versions via $VLLM_VENV/bin/python: $vout"
v_vllm="$(printf '%s\n' "$vout" | sed -n 's/^vllm=//p' | tail -n 1)"
v_torch="$(printf '%s\n' "$vout" | sed -n 's/^torch=//p' | tail -n 1)"
v_fi="$(printf '%s\n' "$vout" | sed -n 's/^flashinfer-python=//p' | tail -n 1)"
local vgot="vllm=$v_vllm torch=$v_torch flashinfer-python=$v_fi"
local vwant="vllm=$EXPECT_VLLM_VERSION torch=$EXPECT_TORCH_VERSION flashinfer-python=$EXPECT_FLASHINFER_VERSION"
[[ "$vgot" == "$vwant" ]] || stop "versions: $vgot (expected $vwant)"
log "versions: $vgot (match expected)"

# ---------------------------------------------------------------- 8. ninja on the launch PATH
local ninja; ninja="$(PATH="$LAUNCH_PATH" command -v ninja || true)"
[[ -n "$ninja" ]] || stop "ninja not found on LAUNCH_PATH=$LAUNCH_PATH (FlashInfer JIT needs it)"
log "ninja: $ninja (LAUNCH_PATH=$LAUNCH_PATH)"

# ---------------------------------------------------------------- 9. id files
for f in "$IDS_FILE" "$IDS_EVEN" "$IDS_ODD"; do
  [[ -f "$f" ]] || stop "split id file missing: $f (make it with scripts/cf748_split_ids.py split)"
done
local vrf
vrf="$("$CHECK_PYTHON" -I "$REPO/scripts/cf748_split_ids.py" verify --input "$IDS_FILE" --a "$IDS_EVEN" --b "$IDS_ODD" --expected-count 748 2>&1)" \
  || stop "split verify failed: $(printf '%s' "$vrf" | tr '\n' ' ')"
log "split verify OK: even=$(count_ids "$IDS_EVEN") odd=$(count_ids "$IDS_ODD") input=$(count_ids "$IDS_FILE") (disjoint, union == input)"
local SHA_IN SHA_EVEN SHA_ODD
SHA_IN="$(sha256_of "$IDS_FILE")"; SHA_EVEN="$(sha256_of "$IDS_EVEN")"; SHA_ODD="$(sha256_of "$IDS_ODD")"
log "sha256 ids: input=$SHA_IN even=$SHA_EVEN odd=$SHA_ODD"

# ---------------------------------------------------------------- 10. targets
local TS LROOT
TS="$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S)"
local START_IST; START_IST="$(ist)"
local -a GPU_LIST=() IDF=() RES=() WLOG=() RLOG=() SNAPS=()
if [[ "$SMOKE" == "1" ]]; then
  LROOT="$LOG_DIR/cf748_effk_launch_${TS}${MODEL_TAG}${ARM_TAG}_smoke"
  if [[ -z "$SMOKE_IDS" ]]; then
    SMOKE_IDS="$(tr -s '[:space:]' '\n' < "$IDS_FILE" | grep -v '^$' | sort -n | head -n 2 | tr '\n' ' ' | sed 's/ $//')"
  fi
  local sid
  for sid in $SMOKE_IDS; do
    [[ "$sid" =~ ^[0-9]+$ ]] || stop "SMOKE_IDS entry '$sid' is not an integer"
    tr -s '[:space:]' '\n' < "$IDS_FILE" | grep -qx "$sid" || stop "SMOKE_IDS entry $sid not in $IDS_FILE"
  done
  [[ -n "$SMOKE_IDS" ]] || stop "SMOKE_IDS is empty"
  log "SMOKE ids: $SMOKE_IDS (GPU $SMOKE_GPU)"
  GPU_LIST=( "$SMOKE_GPU" ); IDF=( "$LROOT/smoke_ids.txt" ); RES=( "$SMOKE_RESULTS_ROOT/$TS/gpu$SMOKE_GPU" )
  WLOG=( "$LOG_DIR/cf748_effk_nofloor${MODEL_TAG}${ARM_TAG}_gpu${SMOKE_GPU}_smoke.log" )
else
  LROOT="$LOG_DIR/cf748_effk_launch_${TS}${MODEL_TAG}${ARM_TAG}"
  GPU_LIST=( 0 1 ); IDF=( "$IDS_EVEN" "$IDS_ODD" ); RES=( "$RESULTS_ROOT/gpu0" "$RESULTS_ROOT/gpu1" )
  WLOG=( "$LOG_DIR/cf748_effk_nofloor${MODEL_TAG}${ARM_TAG}_gpu0.log" "$LOG_DIR/cf748_effk_nofloor${MODEL_TAG}${ARM_TAG}_gpu1.log" )
fi
# Python (runner) output: with GPUS=<g> run_cf748_apps_qwen3.sh takes its parallel branch and writes
# "$RESULTS_DIR/$MODEL_DIR/${SOURCE}_${DIFFICULTY}/cf748_gpu<g>.log" (truncated by '>' on every start;
# the wrapper keeps a timestamped copy of the previous one). WLOG only holds the wrapper's own lines
# plus the run script's header echoes.
local i g out_jsonl
for i in "${!GPU_LIST[@]}"; do RLOG[$i]="${RES[$i]}/$MODEL_DIR/CODEFORCES_interview/cf748_gpu${GPU_LIST[$i]}.log"; done
for i in "${!GPU_LIST[@]}"; do
  out_jsonl="${RES[$i]}/$MODEL_DIR/CODEFORCES_interview/$OUT_FILE"
  if [[ -f "$out_jsonl" ]]; then
    log "existing results (RESUMED by task_id, not a gate): GPU${GPU_LIST[$i]} $out_jsonl lines=$(wc -l < "$out_jsonl" | tr -d ' ')"
  else
    log "existing results: GPU${GPU_LIST[$i]} none ($out_jsonl absent)"
  fi
done

# ---------------------------------------------------------------- 11. commands
local -a CMDS=() WRAPPERS=() RESUMES=()
for i in "${!GPU_LIST[@]}"; do
  g="${GPU_LIST[$i]}"
  local -a ENVV=( VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=1 SKIP_EVAL=1 SEED="$SEED_VALUE"
                  ARMS="$ARM" VLLM_VENV="$VLLM_VENV" TASK_IDS_FILE="${IDF[$i]}" RESULTS_DIR="${RES[$i]}"
                  GPUS="$g" GPU_MEM_UTIL="$GPU_MEM_UTIL" MODEL="$MODEL" SOURCE=CODEFORCES DIFFICULTY=interview
                  N_SAMPLES=10 MAX_TOKENS=32768 PATH="$LAUNCH_PATH" PYTHONUNBUFFERED=1 )
  [[ -n "$MAX_MODEL_LEN" ]] && ENVV+=( MAX_MODEL_LEN="$MAX_MODEL_LEN" )
  CMDS[$i]="$( printf 'env'; printf ' %s' "${ENV_UNSET[@]}"; printf ' %q' "${ENVV[@]}" bash "$RUN_SCRIPT" )"
  # Effective env the run script will see (launcher env minus ENV_UNSET plus ENVV), filtered.
  SNAPS[$i]="$(env "${ENV_UNSET[@]}" "${ENVV[@]}" "$ENV_BIN" | env_snapshot_filter)"
  WRAPPERS[$i]="$LROOT/gpu$g/run_gpu$g.sh"
  RESUMES[$i]="$LROOT/gpu$g/resume_gpu$g.sh"
done

if [[ "$DRY_RUN" == "1" ]]; then
  for i in "${!GPU_LIST[@]}"; do
    g="${GPU_LIST[$i]}"
    log "DRY: GPU$g would run: cd $REPO; ${CMDS[$i]}"
    log "DRY: GPU$g would write ${WRAPPERS[$i]} (wrapper), ${RESUMES[$i]} (resume), launched detached: $(launch_line "${WRAPPERS[$i]}" "${WLOG[$i]}")"
    log "DRY: GPU$g python output would go to ${RLOG[$i]} (wrapper log ${WLOG[$i]})"
    (( i == 0 && ${#GPU_LIST[@]} > 1 )) && log "DRY: would sleep STAGGER_SECONDS=$STAGGER_SECONDS before launching the next GPU"
  done
  [[ "$SMOKE" == "1" ]] && log "DRY: would write ${IDF[0]} (smoke ids: $SMOKE_IDS)"
  log "DRY: would write $LROOT/manifest.json (seed, versions, git HEAD, sha256 pins, split hashes, start IST)"
  log "DRY: would write $LROOT/integrity_check.sh"
  log "==== DRY RUN complete (exit 0); nothing launched."
  exit 0
fi

mkdir -p "$LROOT" || stop "cannot create $LROOT"
if [[ "$SMOKE" == "1" ]]; then printf '%s\n' $SMOKE_IDS > "${IDF[0]}" || stop "cannot write ${IDF[0]}"; fi
write_manifest "$LROOT/manifest.json" || stop "cannot write $LROOT/manifest.json"
log "manifest: $LROOT/manifest.json"
write_integrity_check "$LROOT/integrity_check.sh" || stop "cannot write integrity check"
for i in "${!GPU_LIST[@]}"; do
  g="${GPU_LIST[$i]}"
  mkdir -p "$LROOT/gpu$g" || stop "cannot create $LROOT/gpu$g"
  write_wrapper "${WRAPPERS[$i]}" "$g" "${CMDS[$i]}" "${RLOG[$i]}" || stop "cannot write ${WRAPPERS[$i]}"
  write_resume "${RESUMES[$i]}" "$g" "${WRAPPERS[$i]}" "${WLOG[$i]}" "${RES[$i]}" "${RLOG[$i]}" || stop "cannot write ${RESUMES[$i]}"
done

# ---------------------------------------------------------------- 12. launch
local -a PIDS=()
for i in "${!GPU_LIST[@]}"; do
  g="${GPU_LIST[$i]}"
  if (( i > 0 )); then
    log "STAGGER: sleeping ${STAGGER_SECONDS}s before launching GPU$g (serialises engine init / compile-cache writes)"
    sleep "$STAGGER_SECONDS"
    for pat in "$(scoped "cf748_effk_launch_.*/gpu$g/run_gpu$g\\.sh")"; do
      "$PGREP_BIN" -f "$pat" >/dev/null 2>&1 && stop "a wrapper for GPU$g appeared meanwhile -- refusing to launch a second"
    done
  fi
  log "GPU$g launch: cd $REPO; ${CMDS[$i]}"
  if [[ -n "$SETSID_BIN" ]]; then
    "$SETSID_BIN" nohup bash "${WRAPPERS[$i]}" >>"${WLOG[$i]}" 2>&1 </dev/null 9>&- &
  else
    nohup bash "${WRAPPERS[$i]}" >>"${WLOG[$i]}" 2>&1 </dev/null 9>&- &
  fi
  PIDS[$i]=$!
  LAUNCHED_NOTE+="GPU$g (wrapper PID ${PIDS[$i]}) "
  echo "GPU$g WRAPPER_PID=${PIDS[$i]} LOG=${WLOG[$i]}" >> "$LROOT/launched.txt"
  log "GPU$g wrapper started: PID=${PIDS[$i]} wrapper log=${WLOG[$i]} results=${RES[$i]}"
  log "GPU$g python output: tail -f ${RLOG[$i]}"
done

# ---------------------------------------------------------------- 13. quick liveness check
sleep "$LAUNCH_CHECK_SECONDS"
local bad=0
for i in "${!GPU_LIST[@]}"; do
  g="${GPU_LIST[$i]}"
  if pid_alive "${PIDS[$i]}"; then
    log "GPU$g wrapper PID ${PIDS[$i]} alive after ${LAUNCH_CHECK_SECONDS}s"
  elif grep -q "WRAPPER DONE rc=0" "${WLOG[$i]}" 2>/dev/null; then
    log "GPU$g wrapper PID ${PIDS[$i]} already finished rc=0"
  else
    bad=1
    log "FAIL: GPU$g wrapper PID ${PIDS[$i]} exited within ${LAUNCH_CHECK_SECONDS}s without 'WRAPPER DONE rc=0'. Wrapper log tail (${WLOG[$i]}):"
    tail -n 20 "${WLOG[$i]}" 2>/dev/null | sed 's/^/    | /' | tee -a "$QLOG"
    log "GPU$g runner log tail (${RLOG[$i]}):"
    tail -n 20 "${RLOG[$i]}" 2>/dev/null | sed 's/^/    | /' | tee -a "$QLOG"
  fi
done
log "HINT: integrity/progress check (read-only, any time): bash $LROOT/integrity_check.sh"
if (( bad )); then log "==== launch incomplete; launch dir $LROOT; resume scripts under $LROOT/gpu<N>/; exit 1"; exit 1; fi
log "==== launched (${LAUNCHED_NOTE}); launch dir $LROOT; exit 0"
exit 0
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
# Keep VLLM_/CUDA_/HF_/PYTHON* variables for the manifest, but DROP entirely (name and value) any
# variable whose NAME looks like a credential (TOKEN|KEY|SECRET|PASS|CRED|AUTH, case-insensitive),
# e.g. HF_TOKEN / HF_HUB_TOKEN. Values of such variables are never written anywhere by this launcher.
env_snapshot_filter() {
  grep -E '^(VLLM_|CUDA_|HF_|PYTHON)' | awk -F= 'toupper($1) !~ /TOKEN|KEY|SECRET|PASS|CRED|AUTH/' | LC_ALL=C sort || true
}
scoped() { if [[ -n "${PGREP_SCOPE:-}" ]]; then printf '%s.*%s' "$PGREP_SCOPE" "$1"; else printf '%s' "$1"; fi; }
pid_alive() { kill -0 "$1" 2>/dev/null || return 1; [[ "$(ps -o stat= -p "$1" 2>/dev/null)" != Z* ]]; }
proc_list() {
  local p out=""
  for p in $("$PGREP_BIN" -f "$1" 2>/dev/null); do out+="$p $(ps -o args= -p "$p" 2>/dev/null | cut -c1-200); "; done
  printf '%s' "${out:-none}"
}
count_ids() { tr -s '[:space:]' '\n' < "$1" | grep -c -v '^$'; }
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
nvsmi() {
  if [[ -n "$TIMEOUT_BIN" ]]; then "$TIMEOUT_BIN" "$NVSMI_TIMEOUT_SECONDS" "$NVIDIA_SMI_BIN" "$@"
  else "$NVIDIA_SMI_BIN" "$@"; fi
}
check_gpus_free() {
  local out rc g0 g1
  out="$(nvsmi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)"; rc=$?
  (( rc == 0 )) && [[ -n "$out" ]] || stop "nvidia-smi query failed (rc=$rc)"
  g0="$(printf '%s\n' "$out" | awk -F', *' '$1 ~ /^ *[0-9]/ && $1+0==0 {print $2+0; exit}')"
  g1="$(printf '%s\n' "$out" | awk -F', *' '$1 ~ /^ *[0-9]/ && $1+0==1 {print $2+0; exit}')"
  log "nvidia-smi memory.used: GPU0=${g0:-?} MiB GPU1=${g1:-?} MiB (threshold < $GPU_FREE_MIB)"
  [[ -n "$g0" && -n "$g1" ]] || stop "nvidia-smi did not report both GPU 0 and GPU 1: '$(printf '%s' "$out" | tr '\n' ';')'"
  (( g0 < GPU_FREE_MIB )) || stop "GPU0 busy: memory.used=${g0} MiB >= ${GPU_FREE_MIB}"
  (( g1 < GPU_FREE_MIB )) || stop "GPU1 busy: memory.used=${g1} MiB >= ${GPU_FREE_MIB}"
  log "both GPUs free"
}
launch_line() {  # $1 wrapper, $2 log
  if [[ -n "$SETSID_BIN" ]]; then printf '%q nohup bash %q >> %q 2>&1 < /dev/null &' "$SETSID_BIN" "$1" "$2"
  else printf 'nohup bash %q >> %q 2>&1 < /dev/null &' "$1" "$2"; fi
}
write_wrapper() {  # $1 path, $2 gpu, $3 command, $4 runner (python) log
  cat > "$1" <<EOF
#!/usr/bin/env bash
# CF-748 $ARM wrapper for GPU $2. Generated by scripts/gpu2_launch_cf748_effk_nofloor.sh at $(ist).
# The runner resumes by task_id, so re-running this wrapper skips finished tasks.
set -u
mark() { printf '[%s] [cf748-effk gpu $2] %s\n' "\$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST')" "\$*"; }
cd $(printf '%q' "$REPO") || { mark "WRAPPER ABORT: cannot cd to repo"; exit 1; }
mark "WRAPPER START pid=\$\$ HEAD=\$($(printf '%q' "$GIT_BIN") rev-parse HEAD 2>/dev/null) python log: $(printf '%q' "$4")"
# run_cf748_apps_qwen3.sh truncates its per-GPU log on start: keep the previous one.
if [[ -s $(printf '%q' "$4") ]]; then cp $(printf '%q' "$4") $(printf '%q' "$4").\$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S).prev; fi
$3 < /dev/null
rc=\$?
if (( rc == 0 )); then mark "WRAPPER DONE rc=0"
else mark "WRAPPER FAILED rc=\$rc -- see python log $(printf '%q' "$4")"; fi
exit "\$rc"
EOF
  chmod +x "$1"
}
write_resume() {  # $1 path, $2 gpu, $3 wrapper, $4 wrapper log, $5 results dir, $6 runner (python) log
  local wpat rpat
  wpat="$(scoped "cf748_effk_launch_.*/gpu$2/run_gpu$2\\.sh")"
  rpat="$(scoped " -m bench\\.apps .*--results-dir $5( |\$)")"
  cat > "$1" <<EOF
#!/usr/bin/env bash
# Resume the CF-748 $ARM run on GPU $2 (re-runs the wrapper; the runner resumes by task_id).
# Generated by scripts/gpu2_launch_cf748_effk_nofloor.sh at $(ist). Wrapper: $(printf '%q' "$3")
# Refuses if this GPU's wrapper or runner is already running, or if GPU $2 is busy (FORCE=1 overrides
# only the GPU-memory check).
set -u
if $(printf '%q' "$PGREP_BIN") -f $(printf '%q' "$wpat") >/dev/null; then echo "a wrapper for GPU $2 is already running; refusing" >&2; exit 1; fi
if $(printf '%q' "$PGREP_BIN") -f $(printf '%q' "$rpat") >/dev/null; then echo "a runner for $(printf '%q' "$5") is already running; refusing" >&2; exit 1; fi
gout="\$($( [[ -n "$TIMEOUT_BIN" ]] && printf '%q %s ' "$TIMEOUT_BIN" "$NVSMI_TIMEOUT_SECONDS")$(printf '%q' "$NVIDIA_SMI_BIN") --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || gout=""
used="\$(printf '%s\n' "\$gout" | awk -F', *' '\$1 ~ /^ *[0-9]/ && \$1+0==$2 {print \$2+0; exit}')"
if [[ -z "\$used" ]]; then
  echo "cannot read GPU $2 memory from nvidia-smi; refusing" >&2; [[ "\${FORCE:-0}" == 1 ]] || exit 1
elif (( used >= $GPU_FREE_MIB )); then
  echo "GPU $2 busy: memory.used=\$used MiB >= $GPU_FREE_MIB; refusing" >&2; [[ "\${FORCE:-0}" == 1 ]] || exit 1
fi
cd $(printf '%q' "$REPO") || exit 1
$(launch_line "$3" "$4")
echo "relaunched GPU $2 wrapper PID \$! (wrapper log $(printf '%q' "$4"))"
echo "python output: tail -f $(printf '%q' "$6")"
EOF
  chmod +x "$1"
}
progress_block() {  # per-GPU "records written + Completed line in the python log" (zero progress -> prc=1)
  local i j r
  printf 'prc=0\n'
  for i in "${!GPU_LIST[@]}"; do
    j="$(printf '%q' "${RES[$i]}/$MODEL_DIR/CODEFORCES_interview/$OUT_FILE")"
    r="$(printf '%q' "${RLOG[$i]}")"
    cat <<EOF
n=0; [[ -f $j ]] && n=\$(wc -l < $j | tr -d ' ')
c=no; grep -q 'Completed problem_id' $r 2>/dev/null && c=yes
echo "GPU${GPU_LIST[$i]} records=\$n completed_line=\$c (results $j; python log $r)"
(( n > 0 )) || prc=1
EOF
  done
}
write_integrity_check() {  # $1 path
  local cp; cp="$(printf '%q' "$CHECK_PYTHON")"
  if [[ "$SMOKE" == "1" ]]; then
    cat > "$1" <<EOF
#!/usr/bin/env bash
# SMOKE integrity check (read-only): records parse, ids == smoke ids, 10 samples each; prints per-task
# distinct-sample count and a sha256 of the samples (compare two smoke runs for seeded reproducibility).
$(progress_block)
$cp -I -c '
import hashlib, json, sys
path, ids = sys.argv[1], [int(x) for x in open(sys.argv[2]).read().split()]
seen, bad = {}, 0
try:
    fh = open(path, encoding="utf-8")
except FileNotFoundError:
    print("MISSING", path); sys.exit(1)
for line in fh:
    try:
        r = json.loads(line); seen.setdefault(r["task_id"], []).append(r)
    except Exception:
        bad += 1
for t in sorted(seen):
    r = seen[t][-1]; s = r.get("samples", [])
    h = hashlib.sha256(json.dumps(r.get("samples_with_thinking", s)).encode()).hexdigest()[:16]
    print("task_id=%s records=%d n_samples=%d distinct=%d mean_chars=%.0f method=%s samples_sha=%s" % (
        t, len(seen[t]), len(s), len(set(s)), sum(map(len, s)) / max(1, len(s)), r.get("method"), h))
ok = (not bad and sorted(seen) == sorted(ids) and all(len(v) == 1 and len(v[0].get("samples", [])) == 10
      and v[0].get("method") == "$METHOD" for v in seen.values()))
print(("SMOKE CHECK OK" if ok else "SMOKE CHECK FAILED") + ": records=%d expected_ids=%s unparseable=%d" % (len(seen), ids, bad))
sys.exit(0 if ok else 1)
' $(printf '%q' "${RES[0]}/$MODEL_DIR/CODEFORCES_interview/$OUT_FILE") $(printf '%q' "${IDF[0]}")
rc=\$?
(( rc == 0 && prc == 0 ))
EOF
  else
    cat > "$1" <<EOF
#!/usr/bin/env bash
# Integrity / progress check (read-only) of the two per-GPU results of the CF-748 $ARM run: every line
# parses, 10 samples per record, method/temperature match, no duplicates, GPU dirs disjoint and each a
# subset of its split file. Missing ids are only reported (run still in progress).
cd $(printf '%q' "$REPO") || exit 1
$(progress_block)
$cp -I scripts/cf748_merge_validate.py --check-only --allow-incomplete \\
  --in-a $(printf '%q' "${RES[0]}") --in-b $(printf '%q' "${RES[1]}") \\
  --split-a $(printf '%q' "$IDS_EVEN") --split-b $(printf '%q' "$IDS_ODD") --ids-file $(printf '%q' "$IDS_FILE") \\
  --method $METHOD --temperature 1.0 --n-samples 10${MODEL_ARG}${FILE_ARG}
rc=\$?
echo "When complete, merge (refuses to overwrite) with:"
echo "  $cp -I scripts/cf748_merge_validate.py --in-a $(printf '%q' "${RES[0]}") --in-b $(printf '%q' "${RES[1]}") --split-a $(printf '%q' "$IDS_EVEN") --split-b $(printf '%q' "$IDS_ODD") --ids-file $(printf '%q' "$IDS_FILE") --method $METHOD --temperature 1.0 --n-samples 10${MODEL_ARG}${FILE_ARG} --out-results-dir <merged results dir>"
(( rc == 0 && prc == 0 ))
EOF
  fi
  chmod +x "$1"
}
write_manifest() {  # $1 path. Values passed as argv (k=v); python only serialises.
  local -a kv=( "seed=$SEED_VALUE" "arm=$ARM" "model=$MODEL" "smoke=$SMOKE" "smoke_ids=${SMOKE_IDS:-}"
    "start_time_ist=$START_IST" "launch_dir=$LROOT" "repo=$REPO" "branch=$BRANCH" "git_head=$HEAD_SHA"
    "versions.vllm=$v_vllm" "versions.torch=$v_torch" "versions.flashinfer-python=$v_fi"
    "venv=$VLLM_VENV" "ninja=$ninja" "hostname=$(hostname 2>/dev/null || echo '?')"
    "split.input.path=$IDS_FILE" "split.input.sha256=$SHA_IN" "split.input.count=$(count_ids "$IDS_FILE")"
    "split.even.path=$IDS_EVEN" "split.even.sha256=$SHA_EVEN" "split.even.count=$(count_ids "$IDS_EVEN")"
    "split.odd.path=$IDS_ODD" "split.odd.sha256=$SHA_ODD" "split.odd.count=$(count_ids "$IDS_ODD")"
    "env.VLLM_USE_V2_MODEL_RUNNER=0" "env.VLLM_USE_FLASHINFER_SAMPLER=1" "env.SKIP_EVAL=1"
    "env.SEED=$SEED_VALUE" "env.ARMS=$ARM" "env.GPU_MEM_UTIL=$GPU_MEM_UTIL" "env.MAX_MODEL_LEN=$MAX_MODEL_LEN"
    "env.N_SAMPLES=10" "env.MAX_TOKENS=32768" "env.PATH=$LAUNCH_PATH"
    "launcher_env_snapshot=$(env | env_snapshot_filter)"
    "env.unset=${ENV_UNSET[*]}" "env.VLLM_WORKER_MULTIPROC_METHOD_in_launcher=${VLLM_WORKER_MULTIPROC_METHOD:-}" )
  [[ "$ARM" != "$DEFAULT_ARM" ]] && kv+=( "method=$METHOD" "output_file=$OUT_FILE" )   # default-arm manifest unchanged
  local i
  for i in "${!PINNED_FILES[@]}"; do kv+=( "sha256.${PINNED_FILES[$i]}=${SHAS[$i]}" ); done
  for i in "${!GPU_LIST[@]}"; do
    kv+=( "gpus.${GPU_LIST[$i]}.ids_file=${IDF[$i]}" "gpus.${GPU_LIST[$i]}.results_dir=${RES[$i]}"
          "gpus.${GPU_LIST[$i]}.wrapper_log=${WLOG[$i]}" "gpus.${GPU_LIST[$i]}.runner_log=${RLOG[$i]}"
          "gpus.${GPU_LIST[$i]}.command=${CMDS[$i]}" "gpus.${GPU_LIST[$i]}.env_snapshot=${SNAPS[$i]}" )
  done
  "$CHECK_PYTHON" -I -c '
import json, sys
out, m = sys.argv[1], {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    if k in ("seed",) or k.endswith(".count"):
        v = int(v)
    elif k == "smoke":
        v = v == "1"
    elif k == "smoke_ids":
        v = [int(x) for x in v.split()]
    elif k.endswith("env_snapshot"):
        v = [x for x in v.splitlines() if x]
    # keys like "sha256.bench/apps/runner.py": split on the FIRST dot only for sha256./env.
    head, _, rest = k.partition(".")
    if head in ("sha256", "env", "versions"):
        parts = [head, rest] if rest else [head]
    else:
        parts = k.split(".")
    d = m
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v
m["note"] = ("records do not store the seed; this manifest records seed/versions/split. Seeded arm vs "
             "the unseeded existing CF-748 arms is NOT like-for-like.")
with open(out, "x") as fh:
    fh.write(json.dumps(m, indent=2) + "\n")
' "$1" "${kv[@]}"
}

# Whole body is parsed before anything runs, so editing this file mid-run cannot change what runs.
main "$@"; exit $?
