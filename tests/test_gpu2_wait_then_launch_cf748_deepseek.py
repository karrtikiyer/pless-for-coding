"""Stub-based tests for scripts/gpu2_wait_then_launch_cf748_deepseek.sh.

A throwaway git repo mimics the GPU2 checkout (real runner + merge validator + split ids), fixtures
fake the Qwen run's wrapper logs / python logs / per-half results, a fake launcher records the env and
args it was called with, and a fake nvidia-smi reports memory. Every pgrep pattern is scoped to this
test's tmp dir (PGREP_SCOPE). No GPU, no network, no ssh, no real launch.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests import test_gpu2_launch_cf748_effk_nofloor as L

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/gpu2_wait_then_launch_cf748_deepseek.sh"
LAUNCHER_SCRIPT = ROOT / "scripts/gpu2_launch_cf748_effk_nofloor.sh"
BRANCH = "feat/pless-variants-mps"
PIN = "d8177701bb2e53083e19c1aa79c93ff0fa89c1b0f086347e80f8047b2edcb48f"
DS_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
COPY = ["bench/apps/runner.py", "scripts/cf748_merge_validate.py", "scripts/cf748_split_ids.py",
        "data/cf_interview_748_ids.txt"]
SUB = "Qwen--Qwen3-8B/CODEFORCES_interview"
JSONL = "pless_effk_nofloor_think_t1.0_t1.0.jsonl"

# Launcher knobs the waiter must NOT pass through from its own environment (it sets only its own).
SCRUBBED = ["PGREP_SCOPE", "RESULTS_ROOT", "SMOKE_RESULTS_ROOT", "LOG_DIR", "GPU_FREE_MIB", "STAGGER_SECONDS",
            "IDS_FILE", "IDS_EVEN", "IDS_ODD", "SPLIT_DIR", "GIT_BIN", "NVIDIA_SMI_BIN", "PGREP_BIN", "TIMEOUT_BIN",
            "SETSID_BIN", "FLOCK_BIN", "VLLM_VENV", "CHECK_PYTHON", "LAUNCH_PATH", "LAUNCH_CHECK_SECONDS",
            "NVSMI_TIMEOUT_SECONDS", "EXPECT_VLLM_VERSION", "EXPECT_TORCH_VERSION", "EXPECT_FLASHINFER_VERSION",
            "DRY_RUN", "SMOKE", "SMOKE_IDS", "SMOKE_GPU"]

FAKE_LAUNCHER = r"""#!/usr/bin/env bash
{
  printf 'CALL'
  for v in MODEL MAX_MODEL_LEN EXPECT_HEAD EXPECT_RUNNER_SHA256 GPU2_LAUNCHER_DETACHED REPO BRANCH __SCRUBBED__; do
    printf ' %s=%s' "$v" "${!v-UNSET}"
  done
  printf ' ARGC=%s ARGS=%s\n' "$#" "$*"
} >> "$FAKE_CALLS"
echo "fake launcher output line"
echo "${FAKE_LAUNCHER_OUT:-==== fake launched}"
exit "${FAKE_LAUNCHER_RC:-0}"
""".replace("__SCRUBBED__", " ".join(SCRUBBED))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", BRANCH, str(repo)], check=True)
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    for rel in COPY:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    (repo / ".gitignore").write_text("logs/\nresults/\ndata/cf748_split/\n")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    r = subprocess.run([sys.executable, "-I", str(repo / "scripts/cf748_split_ids.py"), "split",
                        "--input", str(repo / "data/cf_interview_748_ids.txt"),
                        "--out-dir", str(repo / "data/cf748_split")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    stubs = tmp_path / "stubs"; stubs.mkdir()
    (stubs / "launcher.sh").write_text(FAKE_LAUNCHER)
    (stubs / "nvidia-smi").write_text('#!/usr/bin/env bash\necho "0, ${STUB_MEM0:-5}"\necho "1, ${STUB_MEM1:-7}"\n')
    for f in stubs.iterdir():
        f.chmod(0o755)
    (repo / "logs").mkdir()
    e = {k: v for k, v in os.environ.items()
         if k not in ("DRY_RUN", "SMOKE", "SMOKE_IDS", "MODEL", "MAX_MODEL_LEN", "EXPECT_HEAD",
                      "EXPECT_RUNNER_SHA256", "GPU2_LAUNCHER_DETACHED", "MAX_WAIT_HOURS", "POLL_SECONDS")}
    e.update(REPO=str(repo), LAUNCHER=str(stubs / "launcher.sh"), FAKE_CALLS=str(tmp_path / "calls.txt"),
             CHECK_PYTHON=sys.executable, NVIDIA_SMI_BIN=str(stubs / "nvidia-smi"), PGREP_SCOPE=str(tmp_path),
             POLL_SECONDS="1", PROC_POLL_SECONDS="1", PROC_CLEAR_TIMEOUT_SECONDS="20", HEARTBEAT_SECONDS="1",
             EXPECT_RUNNER_SHA256=_sha(repo / "bench/apps/runner.py"), TIMEOUT_BIN="", FLOCK_BIN="")
    yield {"repo": repo, "env": e, "tmp": tmp_path}
    subprocess.run(["pkill", "-f", f"{tmp_path}/FAKE_MARKER"], check=False)


# ---------------------------------------------------------------- fixture helpers
def _ids(repo: Path, half: str) -> list[int]:
    return [int(x) for x in (repo / f"data/cf748_split/cf748_{half}_ids.txt").read_text().split()]


def _res(repo: Path, g: int) -> Path:
    return repo / f"results/_cf748_effk_nofloor_split/gpu{g}" / SUB


def _write_half(repo: Path, g: int, ids=None, n_samples: int = 10, dup: bool = False) -> None:
    d = _res(repo, g); d.mkdir(parents=True, exist_ok=True)
    if ids is None:
        ids = _ids(repo, "even" if g == 0 else "odd")
    samples = ",".join(f'"s{i}"' for i in range(n_samples))
    lines = [('{"model": "Qwen/Qwen3-8B", "backend": "vllm", "method": "pless_effk_nofloor", '
              f'"temperature": 1.0, "task_id": {t}, "source": "CODEFORCES", "difficulty": "interview", '
              f'"samples": [{samples}]}}') for t in ids]
    if dup:
        lines.append(lines[0])
    (d / JSONL).write_text("\n".join(lines) + "\n")


def _wlog(repo: Path, g: int, *lines: str) -> None:
    p = repo / f"logs/cf748_effk_nofloor_gpu{g}.log"
    with open(p, "a") as fh:
        for l in lines:
            fh.write(f"[2026-11-18 10:00:00 IST] [cf748-effk gpu {g}] {l}\n")


def _pylog(repo: Path, g: int, text: str) -> None:
    d = _res(repo, g); d.mkdir(parents=True, exist_ok=True)
    with open(d / f"cf748_gpu{g}.log", "a") as fh:
        fh.write(text)


def _running(repo: Path) -> None:
    for g in (0, 1):
        _wlog(repo, g, "WRAPPER START pid=1 HEAD=abc")
        _pylog(repo, g, "INFO starting\n")


def _done(repo: Path, **half_kw) -> None:
    for g in (0, 1):
        _write_half(repo, g, **half_kw.get(f"g{g}", {}))
        _wlog(repo, g, "WRAPPER DONE rc=0")


def _run(e: dict, timeout: int = 60, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], env={**e, **extra}, capture_output=True, text=True, timeout=timeout)


def _start(e: dict, **extra: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", str(SCRIPT)], env={**e, **extra}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def _log(repo: Path) -> str:
    p = repo / "logs/cf748_deepseek_autostart.log"
    return p.read_text() if p.exists() else ""


def _calls(env) -> list[str]:
    p = Path(env["env"]["FAKE_CALLS"])
    return p.read_text().splitlines() if p.exists() else []


def _wait_for(pred, secs: float = 15.0) -> bool:
    end = time.time() + secs
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.2)
    return False


def _fake_proc(env, args: str, secs: int = 30) -> None:
    subprocess.run(["bash", "-c", f"(exec -a '{env['tmp']}/FAKE_MARKER {args}' sleep {secs}) >/dev/null 2>&1 &"],
                   check=True)
    time.sleep(0.3)


def _abort(env, needle: str, **extra) -> str:
    r = _run(env["env"], **extra)
    log = _log(env["repo"])
    assert r.returncode != 0, log + r.stdout + r.stderr
    assert "ABORT" in log and needle in log, log
    assert _calls(env) == []
    assert not (env["repo"] / "logs/cf748_deepseek_autostart.launched").exists()
    return log


# ---------------------------------------------------------------- waiting / launching
def test_waits_while_not_done_then_stop_file_during(env):
    repo = env["repo"]
    _running(repo)
    p = _start(env["env"])
    try:
        assert _wait_for(lambda: _log(repo).count("HEARTBEAT") >= 2), _log(repo)
        assert _calls(env) == []
        assert "gpu0: state=START records=0" in _log(repo) and "gpu1: state=START records=0" in _log(repo)
        (repo / "logs/cf748_deepseek_autostart.STOP").write_text("")
        assert p.wait(timeout=15) == 0
    finally:
        p.kill()
    log = _log(repo)
    assert "STOP file present" in log and "not launching" in log
    assert _calls(env) == []


def test_heartbeat_reports_record_counts(env):
    repo = env["repo"]
    _running(repo)
    _write_half(repo, 0, ids=_ids(repo, "even")[:5])
    _write_half(repo, 1, ids=_ids(repo, "odd")[:7])
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "gpu0: state=START records=5" in _log(repo) and "gpu1: state=START records=7" in _log(repo)


def test_both_done_and_valid_launches_once_with_exact_env(env):
    repo = env["repo"]
    head = _git(repo, "rev-parse", "HEAD")
    _running(repo); _done(repo)
    r = _run(env["env"])
    log = _log(repo)
    assert r.returncode == 0, log + r.stdout + r.stderr
    calls = _calls(env)
    assert len(calls) == 1, calls
    c = calls[0]
    for want in (f"MODEL={DS_MODEL}", "MAX_MODEL_LEN=40960", f"EXPECT_HEAD={head}",
                 f"EXPECT_RUNNER_SHA256={env['env']['EXPECT_RUNNER_SHA256']}", "GPU2_LAUNCHER_DETACHED=1",
                 f"REPO={repo}", "DRY_RUN=UNSET", "SMOKE=UNSET", "SMOKE_IDS=UNSET", "ARGC=0"):
        assert f" {want} " in c + " ", (want, c)
    assert "CHECK OK (complete)" in log
    assert "launcher exit code: 0" in log and "fake launcher output line" in log
    assert (repo / "logs/cf748_deepseek_autostart.launched").exists()
    # never launch twice: a second waiter refuses at start
    r2 = _run(env["env"])
    assert r2.returncode != 0 and "already launched" in _log(repo)
    assert len(_calls(env)) == 1


def test_inherited_smoke_dry_env_not_forwarded(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], SMOKE="1", SMOKE_IDS="1 2")
    assert r.returncode == 0, _log(repo)
    c = _calls(env)[0]
    assert "SMOKE=UNSET" in c and "SMOKE_IDS=UNSET" in c and "DRY_RUN=UNSET" in c


def test_becomes_done_while_waiting_then_launches(env):
    repo = env["repo"]
    _running(repo)
    p = _start(env["env"])
    try:
        assert _wait_for(lambda: "HEARTBEAT" in _log(repo))
        assert _calls(env) == []
        _done(repo)
        assert p.wait(timeout=30) == 0, _log(repo)
    finally:
        p.kill()
    assert len(_calls(env)) == 1


def test_expect_head_captured_once_at_start_and_drift_aborts(env):
    """HEAD captured once at start; a later commit makes the pre-launch re-check abort (no marker)."""
    repo = env["repo"]
    head0 = _git(repo, "rev-parse", "HEAD")
    _running(repo)
    p = _start(env["env"])
    try:
        assert _wait_for(lambda: "HEARTBEAT" in _log(repo))
        (repo / "newfile.txt").write_text("x\n")
        _git(repo, "add", "newfile.txt"); _git(repo, "commit", "-qm", "later")
        _done(repo)
        assert p.wait(timeout=30) == 3, _log(repo)
    finally:
        p.kill()
    log = _log(repo)
    assert f"EXPECT_HEAD={head0}" in log.split("HEARTBEAT")[0]   # logged at start
    assert f"expected EXPECT_HEAD={head0}" in log and "restart" in log
    assert _calls(env) == [] and not (repo / "logs/cf748_deepseek_autostart.launched").exists()


def test_expect_head_env_matching_accepted(env):
    _running(env["repo"]); _done(env["repo"])
    head = _git(env["repo"], "rev-parse", "HEAD")
    r = _run(env["env"], EXPECT_HEAD=head)
    assert r.returncode == 0, _log(env["repo"])
    assert f"EXPECT_HEAD={head} " in _calls(env)[0]


def test_expect_head_env_mismatch_exits_2_at_start(env):
    _running(env["repo"]); _done(env["repo"])
    r = _run(env["env"], EXPECT_HEAD="a" * 40)
    assert r.returncode == 2 and "PREFLIGHT ERROR" in _log(env["repo"]) and "EXPECT_HEAD" in _log(env["repo"])
    assert _calls(env) == []


@pytest.mark.skipif(_sha(ROOT / "bench/apps/runner.py") != PIN, reason="runner.py no longer matches the pinned sha")
def test_default_runner_pin_forwarded(env):
    _running(env["repo"]); _done(env["repo"])
    e = {k: v for k, v in env["env"].items() if k != "EXPECT_RUNNER_SHA256"}
    r = _run(e)
    assert r.returncode == 0, _log(env["repo"])
    assert f"EXPECT_RUNNER_SHA256={PIN} " in _calls(env)[0]


def test_launcher_failure_exit_code_propagates_marker_kept(env):
    """rc != 0 without the launcher's 'nothing launched' line -> marker KEPT, manual re-arm hint."""
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], FAKE_LAUNCHER_RC="7")
    assert r.returncode == 7
    log = _log(repo)
    assert "launcher exit code: 7" in log and len(_calls(env)) == 1
    assert (repo / "logs/cf748_deepseek_autostart.launched").exists()
    assert "rm " in log and "to re-arm" in log


def test_launcher_nothing_launched_rearms(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], FAKE_LAUNCHER_RC="1",
             FAKE_LAUNCHER_OUT="[x] STOP: GPU0 busy\n[x] launcher exiting non-zero; nothing launched")
    assert r.returncode == 1
    log = _log(repo)
    assert not (repo / "logs/cf748_deepseek_autostart.launched").exists(), log
    assert "re-armed" in log
    r2 = _run(env["env"])           # a restarted waiter can launch again
    assert r2.returncode == 0 and len(_calls(env)) == 2


@pytest.mark.parametrize("out", ["[x] ==== launch incomplete; launch dir d; exit 1",
                                 "[x] launcher exiting non-zero; already launched: GPU0 (wrapper PID 1) -- it KEEPS RUNNING"])
def test_launcher_partial_launch_keeps_marker(env, out):
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], FAKE_LAUNCHER_RC="1", FAKE_LAUNCHER_OUT=out)
    assert r.returncode == 1
    assert (repo / "logs/cf748_deepseek_autostart.launched").exists()
    assert "KEPT" in _log(repo)


def test_launcher_nothing_launched_but_process_running_keeps_marker(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    launcher = Path(env["env"]["LAUNCHER"])
    # the fake launcher leaves a runner-looking process behind but claims nothing was launched
    launcher.write_text(launcher.read_text().replace(
        'echo "fake launcher output line"',
        f"(exec -a '{env['tmp']}/FAKE_MARKER python -m bench.apps' sleep 8) >/dev/null 2>&1 &\nsleep 0.3"))
    r = _run(env["env"], FAKE_LAUNCHER_RC="1", FAKE_LAUNCHER_OUT="launcher exiting non-zero; nothing launched")
    assert r.returncode == 1
    assert (repo / "logs/cf748_deepseek_autostart.launched").exists()
    assert "KEPT" in _log(repo)


@pytest.mark.parametrize("change", ["dirty", "branch", "launcher_gone", "runner_pin"])
def test_prelaunch_preflight_failure_aborts_without_marker(env, change):
    repo = env["repo"]
    _running(repo)
    p = _start(env["env"])
    try:
        assert _wait_for(lambda: "HEARTBEAT" in _log(repo))
        if change == "dirty":
            (repo / "scripts/cf748_split_ids.py").write_text("# changed\n")
        elif change == "branch":
            _git(repo, "checkout", "-q", "-b", "other")
        elif change == "launcher_gone":
            Path(env["env"]["LAUNCHER"]).unlink()
        else:   # committed runner swap (moves HEAD too; either re-check must abort)
            (repo / "bench/apps/runner.py").write_text("# other runner\n")
            _git(repo, "commit", "-qam", "swap runner")
        _done(repo)
        assert p.wait(timeout=30) == 3, _log(repo)
    finally:
        p.kill()
    log = _log(repo)
    assert "ABORT" in log and "pre-launch" in log and "restart the waiter" in log, log
    assert _calls(env) == [] and not (repo / "logs/cf748_deepseek_autostart.launched").exists()


# ---------------------------------------------------------------- aborts (no launch)
@pytest.mark.parametrize("g", [0, 1])
def test_wrapper_failed_aborts(env, g):
    _running(env["repo"])
    _wlog(env["repo"], g, "WRAPPER FAILED rc=1 -- see python log x")
    _abort(env, "WRAPPER FAILED")


def test_one_done_other_failed_aborts(env):
    repo = env["repo"]
    _running(repo)
    _write_half(repo, 0); _wlog(repo, 0, "WRAPPER DONE rc=0")
    _wlog(repo, 1, "WRAPPER FAILED rc=1")
    _abort(env, "gpu1")


def test_stale_failed_before_restart_does_not_abort(env):
    """Wrapper logs are appended: a FAILED from an earlier attempt followed by a new START is running."""
    repo = env["repo"]
    _wlog(repo, 0, "WRAPPER START pid=1", "WRAPPER FAILED rc=1")
    _running(repo)
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _log(repo)
    assert "ABORT" not in _log(repo) and "would wait" in _log(repo)


def test_stale_done_before_restart_is_not_done(env):
    repo = env["repo"]
    _done(repo)          # an earlier DONE ...
    _running(repo)       # ... then the wrapper was started again
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0 and "would wait" in _log(repo), _log(repo)


@pytest.mark.parametrize("g", [0, 1])
def test_traceback_while_running_is_warning_not_abort(env, g):
    repo = env["repo"]
    _running(repo)
    _pylog(repo, g, "Traceback (most recent call last):\n  File x\nRuntimeError: boom\n")
    r = _run(env["env"], DRY_RUN="1")
    log = _log(repo)
    assert r.returncode == 0 and "ABORT" not in log and "would wait" in log, log
    pl = _res(repo, g) / f"cf748_gpu{g}.log"
    # INFO starting is line 1, the Traceback line 2
    assert f"WARNING: gpu{g} python log {pl} contains 1 'Traceback' line(s), first at line 2" in log, log


def test_traceback_done_and_complete_launches_with_warning(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    _pylog(repo, 1, "Traceback (most recent call last):\nTraceback (most recent call last):\n")
    r = _run(env["env"])
    log = _log(repo)
    assert r.returncode == 0, log
    assert "contains 2 'Traceback' line(s), first at line 2" in log
    assert len(_calls(env)) == 1


def test_traceback_done_and_incomplete_aborts(env):
    repo = env["repo"]
    _running(repo); _done(repo, g0={"ids": _ids(repo, "even")[:-3]})
    _pylog(repo, 0, "Traceback (most recent call last):\n")
    log = _abort(env, "check-only")
    assert "WARNING" in log and "Traceback" in log


def test_incomplete_half_aborts(env):
    repo = env["repo"]
    _running(repo); _done(repo, g1={"ids": _ids(repo, "odd")[:-1]})
    log = _abort(env, "check-only")
    assert "do not match split file" in log


def test_duplicate_record_aborts(env):
    repo = env["repo"]
    _running(repo); _done(repo, g0={"dup": True})
    log = _abort(env, "check-only")
    assert "duplicate task_id" in log


def test_wrong_sample_count_aborts(env):
    _running(env["repo"]); _done(env["repo"], g0={"n_samples": 9})
    _abort(env, "check-only")


def test_results_missing_aborts(env):
    repo = env["repo"]
    _running(repo)
    for g in (0, 1):
        _wlog(repo, g, "WRAPPER DONE rc=0")
    _abort(env, "check-only")


def test_leftover_process_blocks_then_clears(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    _fake_proc(env, "python -m bench.apps --backend vllm", secs=3)
    r = _run(env["env"])
    log = _log(repo)
    assert r.returncode == 0, log
    assert "waiting for leftover processes" in log and "-m bench" in log
    assert len(_calls(env)) == 1


@pytest.mark.parametrize("args", ["python -m bench.apps --backend vllm", "VLLM::EngineCore"])
def test_leftover_process_never_clears_aborts(env, args):
    _running(env["repo"]); _done(env["repo"])
    _fake_proc(env, args, secs=30)
    _abort(env, "did not clear", PROC_CLEAR_TIMEOUT_SECONDS="3")


def test_gpu_memory_busy_never_clears_aborts(env):
    _running(env["repo"]); _done(env["repo"])
    _abort(env, "GPU1", PROC_CLEAR_TIMEOUT_SECONDS="3", STUB_MEM1="30000")


def test_max_wait_aborts(env):
    _running(env["repo"])
    log = _abort(env, "MAX_WAIT_HOURS", MAX_WAIT_HOURS="0.0005")
    assert "HEARTBEAT" in log


# ---------------------------------------------------------------- safety
def test_stop_file_before_start(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    (repo / "logs/cf748_deepseek_autostart.STOP").write_text("")
    r = _run(env["env"])
    assert r.returncode == 0
    assert "STOP file present" in _log(repo) and _calls(env) == []


def test_stop_file_during_process_wait_blocks_launch(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    _fake_proc(env, "python -m bench.apps", secs=6)
    p = _start(env["env"])
    try:
        assert _wait_for(lambda: "waiting for leftover processes" in _log(repo))
        (repo / "logs/cf748_deepseek_autostart.STOP").write_text("")
        assert p.wait(timeout=30) == 0
    finally:
        p.kill()
    assert _calls(env) == [] and "STOP file present" in _log(repo)


def test_second_instance_refused(env):
    repo = env["repo"]
    _running(repo)
    p = _start(env["env"])
    try:
        assert _wait_for(lambda: "HEARTBEAT" in _log(repo))
        r = _run(env["env"])
        assert r.returncode != 0
        assert "another waiter instance" in _log(repo)
        (repo / "logs/cf748_deepseek_autostart.STOP").write_text("")
        assert p.wait(timeout=15) == 0
    finally:
        p.kill()
    assert _calls(env) == []


def test_stale_pidfile_taken_over(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    (repo / "logs/cf748_deepseek_autostart.lock").write_text("999999\n")
    r = _run(env["env"])
    assert r.returncode == 0, _log(repo)
    assert "stale" in _log(repo) and len(_calls(env)) == 1


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock not installed (macOS)")
def test_second_instance_refused_flock(env):
    repo = env["repo"]
    _running(repo)
    e = {**env["env"], "FLOCK_BIN": shutil.which("flock")}
    p = _start(e)
    try:
        assert _wait_for(lambda: "HEARTBEAT" in _log(repo))
        r = _run(e)
        assert r.returncode != 0 and "another waiter instance" in _log(repo)
        (repo / "logs/cf748_deepseek_autostart.STOP").write_text("")
        assert p.wait(timeout=15) == 0
    finally:
        p.kill()


def test_dry_run_done_does_not_launch(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], DRY_RUN="1")
    log = _log(repo)
    assert r.returncode == 0, log
    assert "DRY: would launch" in log and DS_MODEL in log and "CHECK OK" in log
    assert _calls(env) == [] and not (repo / "logs/cf748_deepseek_autostart.launched").exists()


def test_dry_run_not_done_reports_wait(env):
    _running(env["repo"])
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0 and "would wait" in _log(env["repo"])
    assert _calls(env) == []


@pytest.mark.parametrize("bad", ["x/y;touch PWNED", "x/$(touch PWNED)", "x/`touch PWNED`", "a/b c", "../x/y",
                                 "noslash", "a/b/c"])
def test_model_injection_rejected(env, bad):
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], MODEL=bad)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "MODEL must match" in _log(repo)
    assert _calls(env) == []
    assert not list(env["tmp"].rglob("PWNED")) and not list(Path.cwd().glob("PWNED"))


def test_model_override_accepted(env):
    _running(env["repo"]); _done(env["repo"])
    r = _run(env["env"], MODEL="org/Other-Model_1.5", MAX_MODEL_LEN="32768")
    assert r.returncode == 0, _log(env["repo"])
    assert "MODEL=org/Other-Model_1.5 " in _calls(env)[0] and "MAX_MODEL_LEN=32768 " in _calls(env)[0]


@pytest.mark.parametrize("var,val", [
    ("POLL_SECONDS", "0"), ("POLL_SECONDS", "5;x"), ("MAX_WAIT_HOURS", "abc"), ("MAX_WAIT_HOURS", "0"),
    ("DRY_RUN", "yes"), ("MAX_MODEL_LEN", "40k"), ("EXPECT_HEAD", "deadbeef"),
    ("EXPECT_RUNNER_SHA256", "any"), ("PROC_CLEAR_TIMEOUT_SECONDS", "-1"), ("HEARTBEAT_SECONDS", "x"),
])
def test_bad_config_rejected(env, var, val):
    _running(env["repo"]); _done(env["repo"])
    r = _run(env["env"], **{var: val})
    assert r.returncode == 2, _log(env["repo"]) + r.stderr
    assert var in _log(env["repo"]) and _calls(env) == []


def test_preflight_runner_pin_mismatch(env):
    _running(env["repo"]); _done(env["repo"])
    r = _run(env["env"], EXPECT_RUNNER_SHA256="0" * 64)
    assert r.returncode == 2 and "sha256 mismatch" in _log(env["repo"]) and _calls(env) == []


def test_preflight_launcher_missing(env):
    r = _run(env["env"], LAUNCHER=str(env["tmp"] / "nope.sh"))
    assert r.returncode == 2 and "LAUNCHER" in _log(env["repo"])


def test_env_values_not_written(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    r = _run(env["env"], HF_TOKEN="hf_SUPERSECRET123")
    assert r.returncode == 0
    assert "SUPERSECRET" not in _log(repo) + r.stdout + r.stderr


# ---------------------------------------------------------------- launcher env hygiene
def test_launcher_knobs_not_passed_through(env):
    repo = env["repo"]
    _running(repo); _done(repo)
    own = {"DRY_RUN": "0", "LOG_DIR": str(repo / "logs"), "GPU_FREE_MIB": "2000", "GIT_BIN": "git",
           "PGREP_BIN": "pgrep"}   # waiter's own knobs: valid values (still must not reach the launcher)
    extra = {k: "LEAKED" for k in SCRUBBED if k not in env["env"] and k not in own}
    extra.update(own)
    r = _run(env["env"], **extra)       # PGREP_SCOPE / NVIDIA_SMI_BIN / CHECK_PYTHON etc. are set by the fixture too
    assert r.returncode == 0, _log(repo)
    c = _calls(env)[0]
    for k in SCRUBBED:
        assert f" {k}=UNSET " in c + " ", (k, c)
    assert f" BRANCH={BRANCH} " in c and "LEAKED" not in c
    assert {"SMOKE", "PGREP_SCOPE", "RESULTS_ROOT", "VLLM_VENV"} <= set(extra) | set(env["env"])


def test_model_qwen_default_rejected(env):
    _running(env["repo"]); _done(env["repo"])
    r = _run(env["env"], MODEL="Qwen/Qwen3-8B")
    assert r.returncode == 2 and "MODEL" in _log(env["repo"]) and _calls(env) == []


# ---------------------------------------------------------------- contract with the real launcher
@pytest.mark.parametrize("mark", ['mark "WRAPPER START ', 'mark "WRAPPER DONE rc=0"', 'mark "WRAPPER FAILED rc=',
                                  'mark "WRAPPER ABORT:', 'log "launcher exiting non-zero; nothing launched"',
                                  "==== launch incomplete", "already launched: "])
def test_launcher_still_emits_parsed_strings(mark):
    assert mark in LAUNCHER_SCRIPT.read_text(), mark


def _real_launcher_wrapper_logs(tmp: Path, **extra: str) -> Path:
    """Run the REAL launcher with the stub harness of tests/test_gpu2_launch_cf748_effk_nofloor.py."""
    venv = tmp / "lvenv"; (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text(L.PY_STUB)
    (venv / "bin/ninja").write_text("#!/usr/bin/env bash\nexit 0\n")
    stubs = tmp / "lstubs"; stubs.mkdir()
    (stubs / "nvidia-smi").write_text('#!/usr/bin/env bash\necho "0, 5"\necho "1, 7"\n')
    for f in list(stubs.iterdir()) + list((venv / "bin").iterdir()):
        f.chmod(0o755)
    repo = tmp / "lrepo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", L.BRANCH, str(repo)], check=True)
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    for rel in L.COPY:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    (repo / ".gitignore").write_text("logs/\nresults/\ndata/cf748_split/\n")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    subprocess.run([sys.executable, "-I", str(repo / "scripts/cf748_split_ids.py"), "split",
                    "--input", str(repo / "data/cf_interview_748_ids.txt"),
                    "--out-dir", str(repo / "data/cf748_split")], check=True, capture_output=True)
    e = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "SMOKE", "MODEL", "MAX_MODEL_LEN", "EXPECT_HEAD")}
    e.update(REPO=str(repo), BRANCH=L.BRANCH, VLLM_VENV=str(venv), NVIDIA_SMI_BIN=str(stubs / "nvidia-smi"),
             EXPECT_RUNNER_SHA256=_sha(repo / "bench/apps/runner.py"), GPU2_LAUNCHER_DETACHED="1",
             PGREP_SCOPE=str(tmp), LAUNCH_PATH=f"{venv}/bin:/usr/bin:/bin", STAGGER_SECONDS="0",
             LAUNCH_CHECK_SECONDS="1", HF_HOME=str(tmp / "hf"), TIMEOUT_BIN="", SETSID_BIN="", **extra)
    subprocess.run(["bash", str(LAUNCHER_SCRIPT)], env=e, capture_output=True, text=True, timeout=120)
    for g in (0, 1):
        wl = repo / f"logs/cf748_effk_nofloor_gpu{g}.log"
        assert _wait_for(lambda: wl.exists() and ("WRAPPER DONE" in wl.read_text() or "WRAPPER FAILED" in wl.read_text()),
                         60), (wl.read_text() if wl.exists() else "MISSING")
    return repo


@pytest.mark.parametrize("fail", [False, True])
def test_waiter_parses_real_launcher_wrapper_logs(env, fail):
    lrepo = _real_launcher_wrapper_logs(env["tmp"], **({"STUB_RC": "1"} if fail else {}))
    try:
        repo = env["repo"]
        for g in (0, 1):
            shutil.copy2(lrepo / f"logs/cf748_effk_nofloor_gpu{g}.log", repo / f"logs/cf748_effk_nofloor_gpu{g}.log")
        if (lrepo / "results").exists():
            shutil.copytree(lrepo / "results", repo / "results")
        r = _run(env["env"], DRY_RUN="1")
        log = _log(repo)
        assert r.returncode == 0, log
        if fail:
            assert "DRY: would ABORT" in log and "WRAPPER FAILED" in log, log
        else:
            assert "state=DONE records=374" in log and "CHECK OK (complete)" in log and "DRY: would launch" in log, log
    finally:
        subprocess.run(["pkill", "-f", f"{lrepo}/logs/cf748_effk_launch_"], check=False)
