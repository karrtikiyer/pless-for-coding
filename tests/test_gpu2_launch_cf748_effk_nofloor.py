"""Stub-based tests for scripts/gpu2_launch_cf748_effk_nofloor.sh.

Builds a throwaway git repo mimicking the GPU2 checkout (with the REAL runner / run script / split +
merge helpers copied in), a fake vLLM venv whose `python` answers the version probe from env vars and
otherwise records `-m bench.apps` invocations (it never imports vllm/torch), and a fake nvidia-smi.
No GPU, no network, no ssh, no real vLLM launch. Every pgrep pattern is scoped to this test's tmp dir
(PGREP_SCOPE).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/gpu2_launch_cf748_effk_nofloor.sh"
BRANCH = "feat/pless-variants-mps"
COPY = ["bench/apps/runner.py", "run_cf748_apps_qwen3.sh", "bench/generator_vllm.py",
        "bench/pless_variants_vllm.py", "scripts/cf748_split_ids.py", "scripts/cf748_merge_validate.py", "scripts/gpu2_launch_cf748_effk_nofloor.sh",
        "data/cf_interview_748_ids.txt"]
OUT_SUB = "Qwen--Qwen3-8B/CODEFORCES_interview"
JSONL = "pless_effk_nofloor_think_t1.0_t1.0.jsonl"

PY_STUB = r"""#!/usr/bin/env bash
# Fake vLLM-venv python. Version probe -> env-controlled answers; `-c` (run script's `import vllm`
# probe) -> ok; `-m bench.apps ...` -> record env + args, append one fake record per task id; any other
# invocation (stdlib helper scripts) -> the real interpreter.
if [[ "$1" == "-I" && "$2" == "-c" && "$3" == *importlib.metadata* ]]; then
  echo "vllm=${STUB_VLLM-0.32.0}"; echo "torch=${STUB_TORCH-2.13.0+cu130}"
  echo "flashinfer-python=${STUB_FI-0.7.0.post1}"; exit 0
fi
if [[ "$1" == "-c" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "bench.apps" ]]; then
  echo "STUBRUN CVD=${CUDA_VISIBLE_DEVICES-unset} V2=${VLLM_USE_V2_MODEL_RUNNER-unset} FI=${VLLM_USE_FLASHINFER_SAMPLER-unset} MP=${VLLM_WORKER_MULTIPROC_METHOD-unset} PATH=$PATH"
  echo "STUBARGS $*"
  ids=""; res=""; prev=""
  for a in "$@"; do [[ "$prev" == "--task-ids-file" ]] && ids="$a"; [[ "$prev" == "--results-dir" ]] && res="$a"; prev="$a"; done
  sleep "${STUB_SLEEP:-0}"
  if [[ -n "${STUB_RC:-}" ]]; then echo "STUB TRACEBACK boom"; exit "$STUB_RC"; fi
  d="$res/Qwen--Qwen3-8B/CODEFORCES_interview"; mkdir -p "$d"
  for t in $(cat "$ids"); do
    echo "Completed problem_id=$t"
    printf '{"model": "Qwen/Qwen3-8B", "backend": "vllm", "method": "pless_effk_nofloor", "temperature": 1.0, "task_id": %s, "source": "CODEFORCES", "difficulty": "interview", "samples": ["a","b","c","d","e","f","g","h","i","j"]}\n' "$t" >> "$d/pless_effk_nofloor_think_t1.0_t1.0.jsonl"
  done
  exit 0
fi
exec __REAL_PY__ "$@"
""".replace("__REAL_PY__", sys.executable)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path: Path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text(PY_STUB)
    (venv / "bin/ninja").write_text("#!/usr/bin/env bash\nexit 0\n")
    stubs = tmp_path / "stubs"; stubs.mkdir()
    (stubs / "nvidia-smi").write_text(
        '#!/usr/bin/env bash\necho "0, ${STUB_MEM0:-5}"\n[[ -n "${STUB_ONE_GPU:-}" ]] || echo "1, ${STUB_MEM1:-7}"\n')
    for f in list(stubs.iterdir()) + list((venv / "bin").iterdir()):
        f.chmod(0o755)
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
    e = {k: v for k, v in os.environ.items()
         if k not in ("DRY_RUN", "SMOKE", "SMOKE_IDS", "SEED", "ARMS", "GPUS", "CUDA_VISIBLE_DEVICES",
                      "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_WORKER_MULTIPROC_METHOD",
                      "EXPECT_HEAD", "MAX_MODEL_LEN", "GPU_MEM_UTIL")}
    e.update(REPO=str(repo), BRANCH=BRANCH, VLLM_VENV=str(venv), NVIDIA_SMI_BIN=str(stubs / "nvidia-smi"),
             EXPECT_RUNNER_SHA256=_sha(repo / "bench/apps/runner.py"), GPU2_LAUNCHER_DETACHED="1",
             PGREP_SCOPE=str(tmp_path), LAUNCH_PATH=f"{venv}/bin:/usr/bin:/bin", STAGGER_SECONDS="0",
             LAUNCH_CHECK_SECONDS="1", HF_HOME=str(tmp_path / "hf"), TIMEOUT_BIN="", SETSID_BIN="")
    yield {"repo": repo, "env": e, "tmp": tmp_path, "venv": venv}
    subprocess.run(["pkill", "-f", f"{repo}/logs/cf748_effk_launch_"], check=False)
    subprocess.run(["pkill", "-f", f"{tmp_path}/FAKE_MARKER"], check=False)


def _run(e: dict, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], env={**e, **extra}, capture_output=True, text=True, timeout=120)


def _qlog(repo: Path) -> str:
    p = repo / "logs/gpu2_launch_cf748_effk.log"
    return p.read_text() if p.exists() else ""


def _fake_proc(env, args: str, secs: int = 30) -> None:
    subprocess.run(["bash", "-c", f"(exec -a '{env['tmp']}/FAKE_MARKER {args}' sleep {secs}) >/dev/null 2>&1 &"],
                   check=True)
    time.sleep(0.3)


def _assert_nothing_launched(env) -> None:
    assert not list(env["repo"].glob("logs/cf748_effk_launch_*"))
    assert not (env["repo"] / "results").exists()


def _stop(env, needle: str, **extra) -> str:
    r = _run(env["env"], **extra)
    log = _qlog(env["repo"])
    assert r.returncode != 0, log + r.stdout + r.stderr
    assert "STOP:" in log and needle in log, log
    assert "nothing launched" in log
    _assert_nothing_launched(env)
    return log


# ---------------------------------------------------------------- dry run
def test_dry_run_full_prints_everything_launches_nothing(env):
    r = _run(env["env"], DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log + r.stderr
    _assert_nothing_launched(env)
    assert "versions: vllm=0.32.0 torch=2.13.0+cu130 flashinfer-python=0.7.0.post1 (match expected)" in log
    assert "split verify OK" in log and "374" in log
    assert "both GPUs free" in log and "ninja:" in log
    assert f"pin: sha256 bench/apps/runner.py = {_sha(env['repo'] / 'bench/apps/runner.py')}" in log
    repo = env["repo"]
    for g, half in ((0, "even"), (1, "odd")):
        line = next(l for l in log.splitlines() if f"DRY: GPU{g} would run:" in l)
        for want in ("VLLM_USE_V2_MODEL_RUNNER=0", "VLLM_USE_FLASHINFER_SAMPLER=1", "SKIP_EVAL=1", "SEED=42",
                     "ARMS=pless_effk_nofloor", f"GPUS={g}", "GPU_MEM_UTIL=0.96", f"VLLM_VENV={env['venv']}",
                     f"TASK_IDS_FILE={repo}/data/cf748_split/cf748_{half}_ids.txt",
                     f"RESULTS_DIR={repo}/results/_cf748_effk_nofloor_split/gpu{g}",
                     "MODEL=Qwen/Qwen3-8B", "N_SAMPLES=10", "MAX_TOKENS=32768", "run_cf748_apps_qwen3.sh"):
            assert want in line, (want, line)
        assert "MULTI_ARM" not in line.split("run_cf748")[0].replace("-u MULTI_ARM", "")
    assert "DRY: would write" in log and "manifest.json" in log and "resume_gpu0.sh" in log
    assert "DRY RUN complete" in log


def test_dry_run_smoke_default_two_ids(env):
    r = _run(env["env"], DRY_RUN="1", SMOKE="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log + r.stderr
    _assert_nothing_launched(env)
    ids = sorted(int(x) for x in (env["repo"] / "data/cf_interview_748_ids.txt").read_text().split())
    assert f"SMOKE ids: {ids[0]} {ids[1]}" in log
    lines = [l for l in log.splitlines() if "would run:" in l]
    assert len(lines) == 1 and "GPU0 would run:" in lines[0]
    assert "results/_cf748_effk_nofloor_smoke/" in lines[0] and "SEED=42" in lines[0]
    assert "_cf748_effk_nofloor_split" not in lines[0]


def test_dry_run_smoke_override_ids_and_gpu(env):
    r = _run(env["env"], DRY_RUN="1", SMOKE="1", SMOKE_IDS="6 8 9", SMOKE_GPU="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "SMOKE ids: 6 8 9" in log and "DRY: GPU1 would run:" in log and "GPUS=1" in log


def test_smoke_id_not_in_list_stops(env):
    _stop(env, "not in", DRY_RUN="1", SMOKE="1", SMOKE_IDS="1 3")  # 3 is not in the 748 list


# ---------------------------------------------------------------- gates
def test_gate_venv_missing(env):
    _stop(env, "VLLM_VENV", DRY_RUN="1", VLLM_VENV=str(env["tmp"] / "nope"))


@pytest.mark.parametrize("var,val,needle", [
    ("STUB_VLLM", "0.21.0", "vllm=0.21.0"),
    ("STUB_TORCH", "2.11.0+cu130", "torch=2.11.0+cu130"),
    ("STUB_FI", "MISSING", "flashinfer-python=MISSING"),
])
def test_gate_versions(env, var, val, needle):
    _stop(env, needle, DRY_RUN="1", **{var: val})


def test_gate_dirty_tree(env):
    (env["repo"] / "run_cf748_apps_qwen3.sh").write_text("#!/bin/bash\n")
    _stop(env, "dirty", DRY_RUN="1")


def test_gate_wrong_branch(env):
    _stop(env, "branch", DRY_RUN="1", BRANCH="main")


def test_gate_expect_head(env):
    head = _git(env["repo"], "rev-parse", "HEAD")
    r = _run(env["env"], DRY_RUN="1", EXPECT_HEAD=head)
    assert r.returncode == 0, _qlog(env["repo"])
    _stop(env, "HEAD", DRY_RUN="1", EXPECT_HEAD="0" * 40)


def test_gate_runner_pin_required_and_checked(env):
    _stop(env, "EXPECT_RUNNER_SHA256 is REQUIRED", DRY_RUN="1", EXPECT_RUNNER_SHA256="")
    _stop(env, "sha256 mismatch", DRY_RUN="1", EXPECT_RUNNER_SHA256="a" * 64)


def test_gate_runner_lacks_variant(env):
    p = env["repo"] / "bench/apps/runner.py"
    p.write_text(p.read_text().replace("pless_effk_nofloor", "pless_effk_xx"))
    _git(env["repo"], "commit", "-qam", "old runner")
    _stop(env, "runner lacks", DRY_RUN="1", EXPECT_RUNNER_SHA256=_sha(p))


def test_gate_script_lacks_arm(env):
    p = env["repo"] / "run_cf748_apps_qwen3.sh"
    p.write_text(p.read_text().replace("pless_effk_nofloor)", "pless_effk_xx)"))
    _git(env["repo"], "commit", "-qam", "old script")
    _stop(env, "lacks the pless_effk_nofloor arm", DRY_RUN="1")


def test_gate_split_file_missing(env):
    (env["repo"] / "data/cf748_split/cf748_odd_ids.txt").unlink()
    _stop(env, "cf748_odd_ids.txt", DRY_RUN="1")


def test_gate_split_overlap(env):
    d = env["repo"] / "data/cf748_split"
    odd = d / "cf748_odd_ids.txt"
    odd.write_text(odd.read_text() + (d / "cf748_even_ids.txt").read_text().split()[0] + "\n")
    _stop(env, "overlap", DRY_RUN="1")


def test_gate_split_union(env):
    odd = env["repo"] / "data/cf748_split/cf748_odd_ids.txt"
    odd.write_text("\n".join(odd.read_text().split()[:-1]) + "\n")
    _stop(env, "union", DRY_RUN="1")


def test_gate_gpu0_busy(env):
    _stop(env, "GPU0 busy", DRY_RUN="1", STUB_MEM0="9000")


def test_gate_gpu1_busy(env):
    _stop(env, "GPU1 busy", DRY_RUN="1", STUB_MEM1="9000")


def test_gate_gpu1_missing(env):
    _stop(env, "did not report both", DRY_RUN="1", STUB_ONE_GPU="1")


@pytest.mark.parametrize("cmd", ["python -m bench.apps --model x", "bash run_cf748_apps_qwen3.sh",
                                 "python scripts/run_pless_variants_ministral_3b_lcb_cuda.py"])
def test_gate_runner_already_running(env, cmd):
    _fake_proc(env, cmd)
    _stop(env, "already running", DRY_RUN="1")


def test_gate_ninja_missing(env):
    (env["venv"] / "bin/ninja").unlink()
    _stop(env, "ninja", DRY_RUN="1")


def test_gate_spawn_refused(env):
    _stop(env, "spawn", DRY_RUN="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")


def test_gate_bad_gpu_mem_util(env):
    _stop(env, "GPU_MEM_UTIL", DRY_RUN="1", GPU_MEM_UTIL="1.5")


# ---------------------------------------------------------------- stub launch (fake python; no vLLM)
def _wait(path: Path, needle: str, secs: float = 30) -> str:
    t0 = time.time()
    while time.time() - t0 < secs:
        if path.exists() and needle in path.read_text():
            return path.read_text()
        time.sleep(0.3)
    raise AssertionError(f"{needle!r} not in {path}:\n{path.read_text() if path.exists() else 'MISSING'}")


def test_stub_launch_full(env):
    repo = env["repo"]
    r = _run(env["env"])
    log = _qlog(repo)
    assert r.returncode == 0, log + r.stderr
    ldirs = list(repo.glob("logs/cf748_effk_launch_*"))
    assert len(ldirs) == 1
    L = ldirs[0]
    man = json.loads((L / "manifest.json").read_text())
    assert man["seed"] == 42 and man["arm"] == "pless_effk_nofloor" and man["smoke"] is False
    assert man["versions"] == {"vllm": "0.32.0", "torch": "2.13.0+cu130", "flashinfer-python": "0.7.0.post1"}
    assert man["git_head"] == _git(repo, "rev-parse", "HEAD")
    assert man["sha256"]["bench/apps/runner.py"] == _sha(repo / "bench/apps/runner.py")
    assert man["split"]["even"]["sha256"] == _sha(repo / "data/cf748_split/cf748_even_ids.txt")
    assert man["split"]["odd"]["count"] == 374 and man["split"]["input"]["count"] == 748
    assert "IST" in man["start_time_ist"]
    assert man["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0" and man["env"]["GPU_MEM_UTIL"] == "0.96"
    # env snapshots: effective runner env per GPU + the launcher's own env
    assert isinstance(man["launcher_env_snapshot"], list)
    assert f"HF_HOME={env['tmp'] / 'hf'}" in man["launcher_env_snapshot"]
    snap = man["gpus"]["0"]["env_snapshot"]
    for want in ("VLLM_USE_V2_MODEL_RUNNER=0", "VLLM_USE_FLASHINFER_SAMPLER=1", "PYTHONUNBUFFERED=1"):
        assert want in snap, snap
    assert not any(x.startswith("CUDA_VISIBLE_DEVICES=") or x.startswith("VLLM_WORKER_MULTIPROC_METHOD=")
                   for x in snap)
    for g, half in ((0, "even"), (1, "odd")):
        res = repo / f"results/_cf748_effk_nofloor_split/gpu{g}"
        glog = _wait(res / OUT_SUB / f"cf748_gpu{g}.log", "STUBARGS")
        assert f"CVD={g} V2=0 FI=1 MP=unset" in glog
        assert f"--task-ids-file {repo}/data/cf748_split/cf748_{half}_ids.txt" in glog
        assert "--method pless_effk_nofloor --temperature 1.0 --seed 42" in glog
        assert "--gpu-memory-utilization 0.96" in glog and "--n-samples 10 --max-new-tokens 32768" in glog
        assert man["gpus"][str(g)]["results_dir"] == str(res)
        # the python output log is the run script's per-GPU log inside RESULTS_DIR
        rlog = res / OUT_SUB / f"cf748_gpu{g}.log"
        assert man["gpus"][str(g)]["runner_log"] == str(rlog)
        assert f"tail -f {rlog}" in log
        assert str(rlog) in (L / f"gpu{g}" / f"resume_gpu{g}.sh").read_text()
        assert str(rlog) in (L / "integrity_check.sh").read_text()
        for f in (f"run_gpu{g}.sh", f"resume_gpu{g}.sh"):
            assert (L / f"gpu{g}" / f).exists()
            subprocess.run(["bash", "-n", str(L / f"gpu{g}" / f)], check=True)
    # wait for both wrappers to finish, then the integrity check validates the union
    for g in (0, 1):
        _wait(repo / f"logs/cf748_effk_nofloor_gpu{g}.log", "WRAPPER DONE rc=0")
    ic = subprocess.run(["bash", str(L / "integrity_check.sh")], capture_output=True, text=True, timeout=60)
    assert ic.returncode == 0, ic.stdout + ic.stderr
    assert "CHECK OK" in ic.stdout and "a=374 b=374 union=748" in ic.stdout
    assert "GPU0 records=374 completed_line=yes" in ic.stdout
    assert "GPU1 records=374 completed_line=yes" in ic.stdout
    # second launch refuses: results exist is only logged, but the launch dir lock / gates still run
    assert "existing results" in log


def test_resume_refuses_while_running(env):
    repo = env["repo"]
    r = _run(env["env"], STUB_SLEEP="20")
    assert r.returncode == 0, _qlog(repo)
    L = next(repo.glob("logs/cf748_effk_launch_*"))
    rr = subprocess.run(["bash", str(L / "gpu0/resume_gpu0.sh")], capture_output=True, text=True, timeout=30,
                        env={**env["env"]})
    assert rr.returncode == 1 and "already running" in rr.stderr
    # and a second launcher refuses because a runner is running
    r2 = _run(env["env"], DRY_RUN="1")
    assert r2.returncode != 0 and "already running" in _qlog(repo)


def test_resume_relaunches_after_exit(env):
    repo = env["repo"]
    r = _run(env["env"], SMOKE="1")
    assert r.returncode == 0, _qlog(repo)
    L = next(repo.glob("logs/cf748_effk_launch_*"))
    glog = repo / "logs/cf748_effk_nofloor_gpu0_smoke.log"
    _wait(glog, "WRAPPER DONE rc=0")
    rr = subprocess.run(["bash", str(L / "gpu0/resume_gpu0.sh")], capture_output=True, text=True, timeout=30,
                        env={**env["env"]})
    assert rr.returncode == 0, rr.stdout + rr.stderr
    assert "relaunched" in rr.stdout
    t0 = time.time()
    while glog.read_text().count("WRAPPER DONE rc=0") < 2 and time.time() - t0 < 30:
        time.sleep(0.3)
    assert glog.read_text().count("WRAPPER DONE rc=0") == 2


def test_stub_launch_smoke(env):
    repo = env["repo"]
    r = _run(env["env"], SMOKE="1", SMOKE_IDS="6 8")
    log = _qlog(repo)
    assert r.returncode == 0, log + r.stderr
    L = next(repo.glob("logs/cf748_effk_launch_*_smoke"))
    man = json.loads((L / "manifest.json").read_text())
    assert man["smoke"] is True and man["smoke_ids"] == [6, 8] and list(man["gpus"]) == ["0"]
    res = Path(man["gpus"]["0"]["results_dir"])
    assert "_cf748_effk_nofloor_smoke" in str(res)
    _wait(repo / "logs/cf748_effk_nofloor_gpu0_smoke.log", "WRAPPER DONE rc=0")
    recs = [json.loads(l) for l in (res / OUT_SUB / JSONL).read_text().splitlines()]
    assert [x["task_id"] for x in recs] == [6, 8]
    ic = subprocess.run(["bash", str(L / "integrity_check.sh")], capture_output=True, text=True, timeout=60)
    assert ic.returncode == 0 and "SMOKE CHECK OK" in ic.stdout, ic.stdout + ic.stderr
    assert "GPU0 records=2 completed_line=yes" in ic.stdout
    assert not (repo / "results/_cf748_effk_nofloor_split").exists()


def test_runner_failure_is_visible(env):
    """Non-zero runner rc -> run script rc 4 -> wrapper logs FAILED, launcher exits 1 and tails the PYTHON log."""
    repo = env["repo"]
    r = _run(env["env"], STUB_RC="1", LAUNCH_CHECK_SECONDS="4")
    log = _qlog(repo)
    assert r.returncode == 1, log
    for g in (0, 1):
        wl = _wait(repo / f"logs/cf748_effk_nofloor_gpu{g}.log", "WRAPPER FAILED rc=")
        assert "WRAPPER FAILED rc=4" in wl and "WRAPPER DONE rc=0" not in wl
        assert f"FAIL: GPU{g}" in log
        rlog = repo / f"results/_cf748_effk_nofloor_split/gpu{g}" / OUT_SUB / f"cf748_gpu{g}.log"
        assert f"runner log tail ({rlog})" in log
    assert "STUB TRACEBACK boom" in log
    L = next(repo.glob("logs/cf748_effk_launch_*"))
    ic = subprocess.run(["bash", str(L / "integrity_check.sh")], capture_output=True, text=True, timeout=60)
    assert ic.returncode != 0
    assert "GPU0 records=0 completed_line=no" in ic.stdout and "GPU1 records=0 completed_line=no" in ic.stdout


# ---------------------------------------------------------------- secrets never written
SECRETS = {"HF_TOKEN": "hf_SECRET123", "HF_HUB_TOKEN": "hf_SECRET456", "MY_API_KEY": "k_SECRET789",
           "PYTHON_PASSWORD": "p_SECRET000", "CUDA_auth_cred": "c_SECRET111"}


def _grep_tree(root: Path, needles) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            hits += [f"{p}: {n}" for n in needles if n.encode() in data]
    return hits


@pytest.mark.parametrize("mode", [{}, {"DRY_RUN": "1"}, {"SMOKE": "1"}])
def test_secret_env_values_never_written(env, mode):
    repo = env["repo"]
    r = _run(env["env"], **SECRETS, **mode)
    assert r.returncode == 0, _qlog(repo) + r.stderr
    if "DRY_RUN" not in mode:
        wl = "logs/cf748_effk_nofloor_gpu0" + ("_smoke" if "SMOKE" in mode else "") + ".log"
        _wait(repo / wl, "WRAPPER DONE rc=0")
        L = next(repo.glob("logs/cf748_effk_launch_*"))
        man = json.loads((L / "manifest.json").read_text())
        names = [x.split("=", 1)[0] for x in man["launcher_env_snapshot"]]
        names += [x.split("=", 1)[0] for g in man["gpus"].values() for x in g["env_snapshot"]]
        assert not [n for n in names if any(k in n.upper() for k in ("TOKEN", "KEY", "SECRET", "PASS", "CRED", "AUTH"))]
        assert any(n == "HF_HOME" for n in names)          # non-secret HF_ vars still recorded
    vals = list(SECRETS.values())
    hits = _grep_tree(repo / "logs", vals) + _grep_tree(repo / "results", vals) if (repo / "results").exists() \
        else _grep_tree(repo / "logs", vals)
    assert hits == [], hits
    assert not any(v in r.stdout + r.stderr for v in vals)
