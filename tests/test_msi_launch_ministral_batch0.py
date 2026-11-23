"""Stub-based tests for scripts/msi_launch_ministral_batch0.sh.

Builds a throwaway git repo that mimics the MSI checkout (the REAL Ministral LCB runner,
bench/generator_vllm.py and results/lcb_v6_full1046/batch_assignment.csv are copied in), a fake
`.venv/bin/python` that answers the version probe and fakes the runner (it never imports vllm/torch),
fake ninja / nvcc / nvidia-smi. No GPU, no network, no ssh, no real vLLM launch. Every pgrep pattern
is scoped to this test's tmp dir (PGREP_SCOPE). Runs under /bin/bash (3.2 on macOS) when present.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/msi_launch_ministral_batch0.sh"
BRANCH = "feat/pless-variants-mps"
SLUG = "mistralai--Ministral-3-3B-Reasoning-2512"
RUNNER = "scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
GENVLLM = "bench/generator_vllm.py"
CSV = "results/lcb_v6_full1046/batch_assignment.csv"
COPY = [RUNNER, GENVLLM, CSV]
BASH = "/bin/bash" if Path("/bin/bash").exists() else "bash"
QLOG = "logs/msi_launch_ministral_batch0.log"
FULL_LOG = "logs/ministral_fulldataset_batch0_top_p_0_9__gk_0_45.log"

PY_STUB = r"""#!/usr/bin/env bash
# Fake MSI venv python. Version probe -> env-controlled answers; the Ministral runner -> print args,
# selected env and PATH, the runner's 'Engine ready in' line (unless STUB_NO_READY), then sleep;
# anything else (stdlib helpers run with -I) -> the real interpreter.
if [[ "$1" == "-I" && "$2" == "-c" && "$3" == *importlib.metadata* ]]; then
  echo "python=3.12.3"; echo "vllm=${STUB_VLLM-0.21.0}"; echo "torch=${STUB_TORCH-2.11.0}"
  echo "flashinfer-python=${STUB_FI-0.5.2}"; echo "flashinfer=MISSING"; echo "transformers=5.8.1"; exit 0
fi
if [[ "$1" == "scripts/run_pless_variants_ministral_3b_lcb_cuda.py" ]]; then
  echo "STUBARGS $*"
  echo "STUBENV FI=${VLLM_USE_FLASHINFER_SAMPLER-unset} CUDA_HOME=${CUDA_HOME-unset} PYTHONPATH=${PYTHONPATH-unset} PYTHONUNBUFFERED=${PYTHONUNBUFFERED-unset}"
  echo "STUBPATH $PATH"
  echo "Loading engine: x (kwargs={})"
  if [[ -n "${STUB_DIE:-}" ]]; then echo "Traceback: boom"; exit 3; fi
  [[ -z "${STUB_NO_READY:-}" ]] && echo "Engine ready in 1s"
  sleep "${STUB_SLEEP:-3}"
  echo "STUBDONE"
  exit 0
fi
exec __REAL_PY__ "$@"
""".replace("__REAL_PY__", sys.executable)

SECRETS = {"HF_TOKEN": "hf_SECRET123", "HF_HUB_TOKEN": "hf_SECRET456", "MY_API_KEY": "k_SECRET789",
           "PYTHON_PASSWORD": "p_SECRET000", "CUDA_auth_cred": "c_SECRET111", "VLLM_API_KEY": "v_SECRET222"}

DROP_ENV = ("DRY_RUN", "CONFIGS", "ALLOW_EXISTING", "PULL", "EXPECT_HEAD", "EXPECT_RUNNER_SHA256",
            "EXPECT_GENVLLM_SHA256", "EXPECT_VLLM_VERSION", "EXPECT_TORCH_VERSION",
            "EXPECT_FLASHINFER_VERSION", "EXPECT_ATCODER", "EXPECT_LEETCODE", "GPU_FREE_MIB",
            "GPU_WAIT_MAX_MINUTES", "LAUNCH_VERIFY_MINUTES", "VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME",
            "PYTHONPATH", "SMOKE", "REQUIRE_COMPLETE", "FORCE")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _batch0_ids(repo: Path, platform: str) -> list[str]:
    import csv
    with (repo / CSV).open() as f:
        return sorted(r["task_id"] for r in csv.DictReader(f) if r["batch"].strip() == "0" and r["platform"] == platform)


def _rec(cfg: str, tid: str, n: int = 10) -> str:
    return json.dumps({"task_id": tid, "method": cfg, "samples": ["s"] * n}) + "\n"


@pytest.fixture
def env(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", BRANCH, str(repo)], check=True)
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    for rel in COPY:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    (repo / ".gitignore").write_text("logs/\n.venv/\nresults/pless_variants_cuda/\n")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    vbin = repo / ".venv/bin"; vbin.mkdir(parents=True)
    (vbin / "python").write_text(PY_STUB)
    (vbin / "ninja").write_text("#!/usr/bin/env bash\nexit 0\n")
    cuda = tmp_path / "cuda"; (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").write_text("#!/usr/bin/env bash\nexit 0\n")
    stubs = tmp_path / "stubs"; stubs.mkdir()
    (stubs / "nvidia-smi").write_text('#!/usr/bin/env bash\necho "${STUB_MEM:-5}"\n')
    for f in [*vbin.iterdir(), cuda / "bin/nvcc", *stubs.iterdir()]:
        f.chmod(0o755)
    full = repo / f"results/pless_variants_cuda/{SLUG}/LCB_v6_full1046"
    e = {k: v for k, v in os.environ.items() if k not in DROP_ENV and k not in SECRETS}
    e.update(REPO=str(repo), NVIDIA_SMI_BIN=str(stubs / "nvidia-smi"), CUDA_HOME_VAL=str(cuda),
             EXPECT_HEAD=_git(repo, "rev-parse", "HEAD"), EXPECT_RUNNER_SHA256=_sha(repo / RUNNER),
             EXPECT_GENVLLM_SHA256=_sha(repo / GENVLLM), PGREP_SCOPE=str(tmp_path), POLL_SECONDS="1",
             SETSID_BIN="", FLOCK_BIN="", TIMEOUT_BIN="", HF_HOME=str(tmp_path / "hf"))
    yield {"repo": repo, "env": e, "tmp": tmp_path, "full": full, "cuda": cuda}
    subprocess.run(["pkill", "-f", f"{repo}/.venv/bin/python"], check=False)
    subprocess.run(["pkill", "-f", f"{tmp_path}/FAKE_MARKER"], check=False)


def _run(e: dict, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(SCRIPT)], env={**e, **extra}, capture_output=True, text=True, timeout=120)


def _qlog(repo: Path) -> str:
    p = repo / QLOG
    return p.read_text() if p.exists() else ""


def _launch_dirs(repo: Path) -> list[Path]:
    return sorted(p for p in repo.glob("logs/msi_launch_*") if p.is_dir())


def _assert_nothing_launched(env) -> None:
    assert not _launch_dirs(env["repo"])
    assert not (env["repo"] / FULL_LOG).exists()
    for p in (env["repo"] / "logs").glob("*.log"):
        assert "STUBARGS" not in p.read_text()


def _stop(env, needle: str, **extra) -> str:
    r = _run(env["env"], **extra)
    log = _qlog(env["repo"]) + r.stdout + r.stderr
    assert r.returncode != 0, log
    assert "STOP:" in log and needle in log, log
    assert "nothing launched" in log
    _assert_nothing_launched(env)
    return log


def _fake_proc(env, args: str, secs: int = 30) -> None:
    subprocess.run([BASH, "-c", f"(exec -a '{env['tmp']}/FAKE_MARKER {args}' sleep {secs}) >/dev/null 2>&1 &"],
                   check=True)
    time.sleep(0.3)


def _wait(path: Path, needle: str, secs: float = 30) -> str:
    t0 = time.time()
    while time.time() - t0 < secs:
        if path.exists() and needle in path.read_text():
            return path.read_text()
        time.sleep(0.2)
    raise AssertionError(f"{needle!r} not in {path}:\n{path.read_text() if path.exists() else 'MISSING'}")


def _commit(repo: Path, rel: str, text: str | None) -> None:
    if text is None:
        _git(repo, "rm", "-q", rel)
    else:
        (repo / rel).write_text(text); _git(repo, "add", rel)
    _git(repo, "commit", "-qm", f"edit {rel}")


# ---------------------------------------------------------------- dry run
def test_dry_run_prints_exact_plan_launches_nothing(env):
    repo, cuda = env["repo"], env["cuda"]
    r = _run(env["env"], DRY_RUN="1")
    log = _qlog(repo)
    assert r.returncode == 0, log + r.stderr
    line = next(l for l in log.splitlines() if "DRY: would run:" in l)
    want = (f"--model mistralai/Ministral-3-3B-Reasoning-2512 --dtype bfloat16 --gpu-memory-utilization 0.92 "
            f"--max-model-len 37500 --n-samples 10 --max-new-tokens 35000 --configs top_p_0_9 gk_0_45 "
            f"--full-dataset --batch 0 --budget-minutes inf --per-config-minutes inf --seed 42")
    assert want in line, line
    assert f"{repo}/.venv/bin/python scripts/run_pless_variants_ministral_3b_lcb_cuda.py" in line
    for kv in (f"CUDA_HOME={cuda}", f"PYTHONPATH={repo}", "PYTHONUNBUFFERED=1", "VLLM_USE_FLASHINFER_SAMPLER=1",
               f"PATH={cuda}/bin:{repo}/.venv/bin:"):
        assert kv in line, (kv, line)
    assert f"cd {repo}" in line and f">> {repo}/{FULL_LOG}" in line
    assert "config order: top_p_0_9 gk_0_45" in log
    assert "platform-major: atcoder[top_p_0_9 -> gk_0_45] then leetcode[top_p_0_9 -> gk_0_45]" in log
    assert "batch 0: atcoder=121 leetcode=90 (expected 121 / 90)" in log
    assert "versions: python=3.12.3 vllm=0.21.0 torch=2.11.0 flashinfer-python=0.5.2" in log
    assert f"ninja: {repo}/.venv/bin/ninja" in log and f"nvcc: {cuda}/bin/nvcc" in log
    assert "HEAD matches EXPECT_HEAD" in log and "(matches expected)" in log
    assert "DRY RUN complete" in log
    _assert_nothing_launched(env)


def test_dry_run_respects_config_order(env):
    r = _run(env["env"], DRY_RUN="1", CONFIGS="gk_0_45 top_p_0_9")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "--configs gk_0_45 top_p_0_9 --full-dataset" in _qlog(env["repo"])
    assert "ministral_fulldataset_batch0_gk_0_45__top_p_0_9.log" in _qlog(env["repo"])


# ---------------------------------------------------------------- config validation
@pytest.mark.parametrize("bad,needle", [
    ("x$(touch {tmp}/PWNED)", "CONFIGS entry"),
    ("top_p_0_9 `touch {tmp}/PWNED`", "CONFIGS entry"),
    ("not_a_config", "not in the runner's CONFIG_NAMES"),
    ("", "CONFIGS is empty"),
    ("   ", "CONFIGS is empty"),
    ("TOP_P_0_9", "CONFIGS entry"),
    ("top-p-0.9", "CONFIGS entry"),
    ("top_p_0_9 top_p_0_9", "duplicate"),
])
def test_invalid_configs_rejected(env, bad, needle):
    tmp = env["tmp"]
    log = _stop(env, needle, CONFIGS=bad.format(tmp=tmp), DRY_RUN="1")
    assert not (tmp / "PWNED").exists() and not (env["repo"] / "PWNED").exists()
    assert not list(tmp.rglob("PWNED"))
    if "PWNED" in bad:
        assert "launcher start" not in log  # rejected before the lock / any other gate


@pytest.mark.parametrize("var,val", [("DRY_RUN", "2"), ("ALLOW_EXISTING", "yes"), ("PULL", "true"),
                                     ("GPU_FREE_MIB", "abc"), ("GPU_WAIT_MAX_MINUTES", "1;id"),
                                     ("LAUNCH_VERIFY_MINUTES", "x"), ("EXPECT_ATCODER", "12a"),
                                     ("POLL_SECONDS", "0.5")])
def test_invalid_numeric_or_flag_env(env, var, val):
    _stop(env, var, **{var: val})


# ---------------------------------------------------------------- code gates
def test_expect_head_required(env):
    e = dict(env["env"]); e.pop("EXPECT_HEAD")
    r = _run(e, DRY_RUN="1")
    assert r.returncode != 0 and "STOP: EXPECT_HEAD is REQUIRED" in _qlog(env["repo"])


@pytest.mark.parametrize("val", ["abc123", "0" * 40, "HEAD"])
def test_expect_head_mismatch_or_malformed(env, val):
    _stop(env, "EXPECT_HEAD", EXPECT_HEAD=val, DRY_RUN="1")


@pytest.mark.parametrize("var", ["EXPECT_RUNNER_SHA256", "EXPECT_GENVLLM_SHA256"])
def test_sha_pins_required(env, var):
    e = dict(env["env"]); e.pop(var)
    r = _run(e, DRY_RUN="1")
    assert r.returncode != 0 and f"STOP: {var} is REQUIRED" in _qlog(env["repo"])


@pytest.mark.parametrize("var,path", [("EXPECT_RUNNER_SHA256", RUNNER), ("EXPECT_GENVLLM_SHA256", GENVLLM)])
def test_sha_pin_mismatch(env, var, path):
    _stop(env, f"sha256 mismatch for {path}", **{var: "0" * 64})


@pytest.mark.parametrize("val", ["any", "xyz", "0" * 63])
def test_sha_pin_malformed(env, val):
    _stop(env, "not 64 hex chars", EXPECT_RUNNER_SHA256=val)


def test_wrong_branch(env):
    _git(env["repo"], "checkout", "-qb", "other")
    _stop(env, "branch is 'other'")


def test_dirty_tree(env):
    (env["repo"] / GENVLLM).write_text("# modified\n")
    _stop(env, "tracked files are dirty")


def test_runner_lacks_config(env):
    repo = env["repo"]
    txt = (repo / RUNNER).read_text()
    _commit(repo, RUNNER, txt.replace('("gk_0_45",          "pless_renyi",      1.0, 0.45),\n', ""))
    _stop(env, "gk_0_45 not in the runner's CONFIG_NAMES", EXPECT_HEAD=_git(repo, "rev-parse", "HEAD"),
          EXPECT_RUNNER_SHA256=_sha(repo / RUNNER))


def test_runner_configs_parsed_from_real_runner(env):
    r = _run(env["env"], DRY_RUN="1", CONFIGS="ministral_recipe top_k_15")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    # ministral_recipe has T=0.7 in CONFIGS -> its output file name carries t0.7
    assert "ministral_recipe_atcoder_think_t0.7.jsonl" in log and "top_k_15_leetcode_think_t1.0.jsonl" in log


def test_venv_python_missing(env):
    (env["repo"] / ".venv/bin/python").unlink()
    _stop(env, "venv python missing")


def test_vllm_missing(env):
    _stop(env, "vllm is not installed", STUB_VLLM="MISSING")


def test_optional_version_pin(env):
    _stop(env, "vllm version 0.21.0 != EXPECT_VLLM_VERSION=0.32.0", EXPECT_VLLM_VERSION="0.32.0")
    r = _run(env["env"], DRY_RUN="1", EXPECT_VLLM_VERSION="0.21.0", EXPECT_TORCH_VERSION="2.11.0")
    assert r.returncode == 0, _qlog(env["repo"])


def test_ninja_missing(env):
    (env["repo"] / ".venv/bin/ninja").unlink()
    _stop(env, "ninja not found on the launch PATH", PATH="/usr/bin:/bin")


def test_nvcc_missing(env):
    (env["cuda"] / "bin/nvcc").unlink()
    _stop(env, "nvcc not found on the launch PATH", PATH="/usr/bin:/bin")


def test_smoke_not_supported(env):
    _stop(env, "SMOKE=1 is not supported", SMOKE="1", DRY_RUN="1")


def test_batch_csv_counts(env):
    _stop(env, "batch 0: atcoder=121 leetcode=90 (expected 120 / 90)", EXPECT_ATCODER="120")


def test_batch_csv_missing(env):
    repo = env["repo"]
    _commit(repo, CSV, None)
    _stop(env, "batch assignment CSV missing", EXPECT_HEAD=_git(repo, "rev-parse", "HEAD"))


# ---------------------------------------------------------------- busy / GPU gates
@pytest.mark.parametrize("args", ["python scripts/run_pless_variants_ministral_3b_lcb_cuda.py --configs top_k_15",
                                  "bash scripts/msi_queue_topk_after_recipe.sh"])
def test_busy_process_refused(env, args):
    _fake_proc(env, args)
    _stop(env, "already running", DRY_RUN="1")


def test_tail_of_queue_log_does_not_block(env):
    _fake_proc(env, "tail -f logs/msi_queue_topk.log")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])


def test_gpu_busy(env):
    _stop(env, "memory.used=9000 MiB >= 2000", STUB_MEM="9000")


def test_gpu_enginecore_lingering(env):
    _fake_proc(env, "VLLM::EngineCore")
    _stop(env, "VLLM::EngineCore still present")


def test_gpu_wait_cap(env):
    t0 = time.time()
    _stop(env, "GPU still held after 0.05 min", STUB_MEM="9000", GPU_WAIT_MAX_MINUTES="0.05")
    assert time.time() - t0 >= 2.5


def test_nvidia_smi_failing(env):
    (env["tmp"] / "stubs/nvidia-smi").write_text("#!/usr/bin/env bash\nexit 9\n")
    _stop(env, "nvidia-smi")


# ---------------------------------------------------------------- existing outputs
def test_existing_output_refused_then_allowed(env):
    full = env["full"]; full.mkdir(parents=True)
    at = _batch0_ids(env["repo"], "atcoder")
    f = full / "top_p_0_9_atcoder_think_t1.0.jsonl"
    f.write_text("".join(_rec("top_p_0_9", t) for t in at[:3]))
    log = _stop(env, "ALLOW_EXISTING=1", DRY_RUN="1")
    assert "top_p_0_9_atcoder_think_t1.0.jsonl" in log and "distinct=3" in log
    r = _run(env["env"], DRY_RUN="1", ALLOW_EXISTING="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "ALLOW_EXISTING=1: existing output will be RESUMED by task_id" in log
    assert f.read_text().count("\n") == 3  # untouched


def _other_batch_ids(repo: Path, platform: str) -> list[str]:
    import csv
    with (repo / CSV).open() as f:
        return sorted(r["task_id"] for r in csv.DictReader(f) if r["batch"].strip() != "0" and r["platform"] == platform)


def test_other_batch_records_tolerated_with_allow(env):
    """The full1046 files are shared by all batches (the runner appends batch 0 to the batch-1/2 file)."""
    full = env["full"]; full.mkdir(parents=True)
    ob = _other_batch_ids(env["repo"], "leetcode")[:5]
    (full / "gk_0_45_leetcode_think_t1.0.jsonl").write_text("".join(_rec("gk_0_45", t) for t in ob))
    _stop(env, "ALLOW_EXISTING=1", DRY_RUN="1")
    r = _run(env["env"], DRY_RUN="1", ALLOW_EXISTING="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "gk_0_45/leetcode: lines=5 distinct=5" in log and "other_batches=5" in log and "batch0_done=0/90" in log


def test_empty_existing_file_is_only_logged(env):
    full = env["full"]; full.mkdir(parents=True)
    (full / "gk_0_45_leetcode_think_t1.0.jsonl").write_text("")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "exists but is empty" in _qlog(env["repo"])


@pytest.mark.parametrize("bad", ["no_newline", "dup", "garbage", "unknown_id", "wrong_method", "few_samples"])
def test_corrupt_existing_file_refused_even_with_allow(env, bad):
    full = env["full"]; full.mkdir(parents=True)
    at = _batch0_ids(env["repo"], "atcoder")
    body = "".join(_rec("top_p_0_9", t) for t in at[:3])
    body = {"no_newline": body + _rec("top_p_0_9", at[3]).rstrip("\n"), "dup": body + _rec("top_p_0_9", at[0]),
            "garbage": body + '{"task_id": "trunc\n', "unknown_id": body + _rec("top_p_0_9", "zzz_not_in_csv"),
            "wrong_method": body + _rec("gk_0_45", at[3]), "few_samples": body + _rec("top_p_0_9", at[3], 2)}[bad]
    (full / "top_p_0_9_atcoder_think_t1.0.jsonl").write_text(body)
    _stop(env, "existing output failed the integrity check", DRY_RUN="1", ALLOW_EXISTING="1")


# ---------------------------------------------------------------- lock
def test_second_instance_refused_pidfile(env):
    (env["repo"] / "logs").mkdir()
    holder = subprocess.Popen(["sleep", "30"])
    try:
        (env["repo"] / "logs/msi_launch_ministral_batch0.lock").write_text(f"{holder.pid}\n")
        _stop(env, "another launcher instance", DRY_RUN="1")
    finally:
        holder.kill()


def test_stale_pidfile_taken_over(env):
    (env["repo"] / "logs").mkdir()
    (env["repo"] / "logs/msi_launch_ministral_batch0.lock").write_text("999999\n")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0 and "stale pidfile lock" in _qlog(env["repo"])


def test_concurrent_instance_refused(env):
    p = subprocess.Popen([BASH, str(SCRIPT)], env={**env["env"], "STUB_MEM": "9000", "GPU_WAIT_MAX_MINUTES": "1"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(env["repo"] / QLOG, "waiting: GPU memory.used=9000")
        r = _run(env["env"], DRY_RUN="1")
        assert r.returncode != 0 and "another launcher instance" in _qlog(env["repo"])
    finally:
        p.kill(); p.wait()


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock not installed (macOS)")
def test_second_instance_refused_flock(env):
    p = subprocess.Popen([BASH, str(SCRIPT)], env={**env["env"], "FLOCK_BIN": shutil.which("flock"),
                         "STUB_MEM": "9000", "GPU_WAIT_MAX_MINUTES": "1"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(env["repo"] / QLOG, "waiting: GPU memory.used=9000")
        r = _run(env["env"], DRY_RUN="1", FLOCK_BIN=shutil.which("flock"))
        assert r.returncode != 0 and "(flock)" in _qlog(env["repo"])
    finally:
        p.kill(); p.wait()


# ---------------------------------------------------------------- real (stub) launch
def test_stub_launch_writes_launch_dir_and_verifies_engine_ready(env):
    repo, cuda = env["repo"], env["cuda"]
    r = _run(env["env"], STUB_SLEEP="20")
    log = _qlog(repo)
    assert r.returncode == 0, log + r.stderr
    assert "launch verified: 'Engine ready in 1s'" in log
    L = _launch_dirs(repo)
    assert len(L) == 1 and re.search(r"msi_launch_\d{8}_\d{6}$", L[0].name)
    L = L[0]
    for f in ("manifest.json", "resume.sh", "integrity_check.sh", "cmdline.txt", "launched.txt"):
        assert (L / f).exists(), f
    for f in ("resume.sh", "integrity_check.sh"):
        subprocess.run([BASH, "-n", str(L / f)], check=True)
    man = json.loads((L / "manifest.json").read_text())
    assert man["configs"] == ["top_p_0_9", "gk_0_45"] and man["seed"] == 42 and man["batch"] == 0
    assert man["git_head"] == _git(repo, "rev-parse", "HEAD")
    assert man["sha256"][RUNNER] == _sha(repo / RUNNER) and man["sha256"][GENVLLM] == _sha(repo / GENVLLM)
    assert man["versions"]["python"] == "3.12.3" and man["versions"]["vllm"] == "0.21.0"
    assert man["versions"]["torch"] == "2.11.0" and man["versions"]["flashinfer-python"] == "0.5.2"
    assert "IST" in man["start_time_ist"]
    assert man["launch_path_prefix"] == f"{cuda}/bin:{repo}/.venv/bin"
    assert man["batch0_counts"] == {"atcoder": 121, "leetcode": 90}
    assert "--seed 42" in man["command"] and "--configs top_p_0_9 gk_0_45" in man["command"]
    assert len(man["outputs"]) == 4 and man["outputs"][0].endswith("top_p_0_9_atcoder_think_t1.0.jsonl")
    snap = man["env_snapshot"]
    for want in ("VLLM_USE_FLASHINFER_SAMPLER=1", "PYTHONUNBUFFERED=1", f"CUDA_HOME={cuda}", f"PYTHONPATH={repo}",
                 f"HF_HOME={env['tmp'] / 'hf'}"):
        assert want in snap, snap
    assert all(re.match(r"^(VLLM_|CUDA_|HF_|PYTHON)", x) for x in snap), snap
    flog = _wait(repo / FULL_LOG, "STUBARGS")
    assert ("STUBARGS scripts/run_pless_variants_ministral_3b_lcb_cuda.py --model mistralai/Ministral-3-3B-Reasoning-2512 "
            "--dtype bfloat16 --gpu-memory-utilization 0.92 --max-model-len 37500 --n-samples 10 --max-new-tokens 35000 "
            "--configs top_p_0_9 gk_0_45 --full-dataset --batch 0 --budget-minutes inf --per-config-minutes inf --seed 42") in flog
    assert f"STUBENV FI=1 CUDA_HOME={cuda} PYTHONPATH={repo} PYTHONUNBUFFERED=1" in flog
    assert f"STUBPATH {cuda}/bin:{repo}/.venv/bin:" in flog
    # a second launch while the runner is alive is refused
    r2 = _run(env["env"], DRY_RUN="1")
    assert r2.returncode != 0 and "already running" in _qlog(repo)


def test_launch_dies_before_ready(env):
    r = _run(env["env"], STUB_DIE="1")
    log = _qlog(env["repo"])
    assert r.returncode == 1 and "died before 'Engine ready'" in log and "Traceback: boom" in log


def test_launch_ready_timeout_ignores_old_lines(env):
    (env["repo"] / "logs").mkdir()
    (env["repo"] / FULL_LOG).write_text("Engine ready in 99s\n")
    r = _run(env["env"], STUB_NO_READY="1", STUB_SLEEP="20", LAUNCH_VERIFY_MINUTES="0.05")
    log = _qlog(env["repo"])
    assert r.returncode == 1 and "no 'Engine ready' within 0.05 min" in log and "launch verified" not in log


# ---------------------------------------------------------------- secrets
def _grep_tree(root: Path, needles) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            hits += [f"{p}: {n}" for n in needles if n.encode() in data]
    return hits


@pytest.mark.parametrize("mode", [{"DRY_RUN": "1"}, {}])
def test_secrets_never_written(env, mode):
    repo = env["repo"]
    r = _run(env["env"], **SECRETS, **mode, STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(repo) + r.stderr
    if not mode:
        _wait(repo / FULL_LOG, "STUBDONE")
        man = json.loads((_launch_dirs(repo)[0] / "manifest.json").read_text())
        names = [x.split("=", 1)[0] for x in man["env_snapshot"]]
        assert not [n for n in names if re.search("TOKEN|KEY|SECRET|PASS|CRED|AUTH", n.upper())], names
        assert "HF_HOME" in names
    vals = list(SECRETS.values())
    assert _grep_tree(repo / "logs", vals) == []
    assert not any(v in r.stdout + r.stderr for v in vals)
    for k in SECRETS:
        assert k not in _qlog(repo)


# ---------------------------------------------------------------- generated scripts
def _launch(env, **extra) -> Path:
    r = _run(env["env"], STUB_SLEEP="1", **extra)
    assert r.returncode == 0, _qlog(env["repo"]) + r.stderr
    _wait(env["repo"] / FULL_LOG, "STUBDONE")
    time.sleep(0.5)
    return _launch_dirs(env["repo"])[0]


def test_resume_refuses_while_running_then_relaunches(env):
    repo = env["repo"]
    r = _run(env["env"], STUB_SLEEP="15")
    assert r.returncode == 0, _qlog(repo)
    L = _launch_dirs(repo)[0]
    rr = subprocess.run([BASH, str(L / "resume.sh")], env=env["env"], capture_output=True, text=True, timeout=60)
    assert rr.returncode == 1 and "already running" in rr.stderr
    subprocess.run(["pkill", "-f", f"{repo}/.venv/bin/python"], check=False)
    time.sleep(1)
    full_before = (repo / FULL_LOG).read_text().count("STUBARGS")
    rr = subprocess.run([BASH, str(L / "resume.sh")], env={**env["env"], "STUB_SLEEP": "1"},
                        capture_output=True, text=True, timeout=60)
    assert rr.returncode == 0, rr.stdout + rr.stderr
    assert "relaunched PID" in rr.stdout
    t0 = time.time()
    while (repo / FULL_LOG).read_text().count("STUBARGS") <= full_before and time.time() - t0 < 30:
        time.sleep(0.2)
    flog = (repo / FULL_LOG).read_text()
    assert flog.count("STUBARGS") == full_before + 1
    assert flog.count("--configs top_p_0_9 gk_0_45 --full-dataset --batch 0") >= 2
    assert flog.count(f"STUBPATH {env['cuda']}/bin:{repo}/.venv/bin:") >= 2
    assert flog.count("STUBENV FI=1") >= 2


def test_resume_refuses_on_head_change(env):
    repo = env["repo"]
    L = _launch(env)
    _commit(repo, "README", "x\n")
    rr = subprocess.run([BASH, str(L / "resume.sh")], env=env["env"], capture_output=True, text=True, timeout=60)
    assert rr.returncode == 1 and "HEAD" in rr.stderr


def test_resume_refuses_on_corrupt_output(env):
    L = _launch(env)
    full = env["full"]; full.mkdir(parents=True, exist_ok=True)
    (full / "gk_0_45_leetcode_think_t1.0.jsonl").write_text('{"task_id": "trunc')
    rr = subprocess.run([BASH, str(L / "resume.sh")], env=env["env"], capture_output=True, text=True, timeout=60)
    assert rr.returncode == 1 and "integrity" in rr.stderr


def test_integrity_check_on_fake_results(env):
    repo, full = env["repo"], env["full"]
    L = _launch(env)
    ic = L / "integrity_check.sh"
    # nothing yet: in progress, not a problem; REQUIRE_COMPLETE=1 fails
    r = subprocess.run([BASH, str(ic)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("MISSING") == 4
    r = subprocess.run([BASH, str(ic)], env={**os.environ, "REQUIRE_COMPLETE": "1"}, capture_output=True, text=True)
    assert r.returncode == 1
    full.mkdir(parents=True, exist_ok=True)
    at, lc = _batch0_ids(repo, "atcoder"), _batch0_ids(repo, "leetcode")
    for cfg in ("top_p_0_9", "gk_0_45"):
        (full / f"{cfg}_atcoder_think_t1.0.jsonl").write_text("".join(_rec(cfg, t) for t in at))
        (full / f"{cfg}_leetcode_think_t1.0.jsonl").write_text("".join(_rec(cfg, t) for t in lc[:50]))
    r = subprocess.run([BASH, str(ic)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "top_p_0_9/atcoder: lines=121 distinct=121 unparseable=0 duplicated=0" in r.stdout
    assert "gk_0_45/leetcode: lines=50 distinct=50" in r.stdout and "missing=40" in r.stdout
    r = subprocess.run([BASH, str(ic)], env={**os.environ, "REQUIRE_COMPLETE": "1"}, capture_output=True, text=True)
    assert r.returncode == 1
    for cfg in ("top_p_0_9", "gk_0_45"):
        (full / f"{cfg}_leetcode_think_t1.0.jsonl").write_text("".join(_rec(cfg, t) for t in lc))
    r = subprocess.run([BASH, str(ic)], env={**os.environ, "REQUIRE_COMPLETE": "1"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    # problems: duplicate, garbage, missing trailing newline
    p = full / "gk_0_45_atcoder_think_t1.0.jsonl"
    p.write_text(p.read_text() + _rec("gk_0_45", at[0]) + "not json\n" + _rec("gk_0_45", "x").rstrip("\n"))
    r = subprocess.run([BASH, str(ic)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "gk_0_45/atcoder:" in r.stdout and "duplicated=1" in r.stdout and "unparseable=1" in r.stdout
    assert "trailing_newline=no" in r.stdout and "unknown_ids=1" in r.stdout
    # read-only: the check never modifies result files
    before = {f.name: f.read_bytes() for f in full.iterdir()}
    subprocess.run([BASH, str(ic)], capture_output=True, text=True, timeout=60)
    assert before == {f.name: f.read_bytes() for f in full.iterdir()}


# ---------------------------------------------------------------- pull
def test_pull_opt_in_ff_only(env):
    repo, tmp = env["repo"], env["tmp"]
    origin = tmp / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    _git(repo, "remote", "add", "origin", str(origin)); _git(repo, "push", "-q", "origin", BRANCH)
    other = tmp / "other"
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, str(origin), str(other)], check=True)
    _git(other, "config", "user.email", "t@t"); _git(other, "config", "user.name", "t")
    (other / "README").write_text("new\n"); _git(other, "add", "README"); _git(other, "commit", "-qm", "up")
    _git(other, "push", "-q", "origin", BRANCH)
    new = _git(other, "rev-parse", "HEAD")
    old = _git(repo, "rev-parse", "HEAD")
    r = _run(env["env"], DRY_RUN="1", PULL="1")  # dry: prints, does not pull
    assert r.returncode == 0 and "DRY: would pull: " in _qlog(repo) and _git(repo, "rev-parse", "HEAD") == old
    r = _run(env["env"], DRY_RUN="1")  # PULL default 0
    assert r.returncode == 0 and "pull skipped (PULL=0)" in _qlog(repo)
    r = _run(env["env"], PULL="1", EXPECT_HEAD=new, STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(repo)
    assert _git(repo, "rev-parse", "HEAD") == new


# ---------------------------------------------------------------- portability
@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="no /bin/bash")
def test_bash_3_2_portability(env):
    ver = subprocess.run(["/bin/bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout.strip()
    if ver != "3":
        pytest.skip(f"/bin/bash is bash {ver}, not 3.2")
    subprocess.run(["/bin/bash", "-n", str(SCRIPT)], check=True)
    L = _launch(env)
    for f in ("resume.sh", "integrity_check.sh"):
        subprocess.run(["/bin/bash", "-n", str(L / f)], check=True)
    r = subprocess.run(["/bin/bash", str(L / "integrity_check.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "unbound variable" not in _qlog(env["repo"]) + r.stderr
