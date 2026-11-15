"""Stub-based tests for scripts/gpu2_launch_ministral_gk_batches.sh.

Builds a throwaway git repo that mimics the GPU2 checkout, stubs python /
nvidia-smi, and runs the launcher end to end. No GPU, no network, no ssh.
Every pgrep pattern is scoped to this test's tmp dir (PGREP_SCOPE), so two
concurrent runs of this file cannot see each other's fake processes.
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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/gpu2_launch_ministral_gk_batches.sh"
SLUG = "mistralai--Ministral-3-3B-Reasoning-2512"
BRANCH = "feat/pless-variants-mps"
RUNNER_REL = "scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
RUNNER_TXT = ('CONFIGS = [\n    ("gk_0_35",          "pless_renyi",      1.0, 0.35),\n'
              '    ("gk_0_45",          "pless_renyi",      1.0, 0.45),\n]\n')
CSV_ROWS = ["abc0_a,atcoder,easy,0", "abc1_a,atcoder,easy,1", "abc1_b,atcoder,hard,1",
            "lc1_a,leetcode,easy,1", "abc2_a,atcoder,easy,2", "lc2_a,leetcode,easy,2"]
SAMPLE_SENTINEL = "SAMPLE_BODY_SENTINEL"

PY_STUB = r"""#!/usr/bin/env bash
# Fake .venv python. `-c ...` = the vllm-version probe. Otherwise a fake runner:
# prints its env/args + the engine-ready line (unless told to die/hang), sleeps,
# then appends one record per batch task_id (like the real runner) and exits 0.
if [[ "$1" == "-c" ]]; then echo "${STUB_VLLM:-0.21.0}"; exit 0; fi
batch=""; cfg=""; prev=""
for a in "$@"; do
  [[ "$prev" == "--batch" ]] && batch="$a"; [[ "$prev" == "--configs" ]] && cfg="$a"; prev="$a"
done
echo "ARGS: $*"
echo "CVD=$CUDA_VISIBLE_DEVICES PATH=$PATH PYTHONPATH=$PYTHONPATH FI=${VLLM_USE_FLASHINFER_SAMPLER-unset} CUDA_HOME=${CUDA_HOME-unset}"
echo "Loading engine: x"
if [[ "${STUB_DIE_BATCH:-}" == "$batch" ]]; then echo "Traceback: boom batch $batch"; exit 3; fi
sleep "${STUB_READY_DELAY:-0}"
if [[ "${STUB_NO_READY_BATCH:-}" != "$batch" ]]; then
  if [[ -n "${STUB_CR:-}" ]]; then printf 'Loading safetensors 100%%|####|\rEngine ready in 1s\n'
  else echo "Engine ready in 1s"; fi
fi
sleep "${STUB_SLEEP:-20}"
if [[ "${STUB_FAIL_CONFIG:-}" == "$cfg" ]]; then echo "Traceback: $cfg failed"; exit 5; fi
d="results/pless_variants_cuda/__SLUG__/LCB_v6_full1046"; mkdir -p "$d"
while IFS=, read -r tid plat diff bb; do
  bb="${bb%$'\r'}"
  [[ "$bb" == "$batch" ]] || continue
  [[ "$tid" == "${STUB_SKIP_TASK:-}" ]] && continue
  echo "{\"task_id\": \"$tid\", \"samples\": [\"__SENTINEL__\"]}" >> "$d/${cfg}_${plat}_think_t1.0.jsonl"
  [[ "$tid" == "${STUB_DUP_TASK:-}" ]] && echo "{\"task_id\": \"$tid\"}" >> "$d/${cfg}_${plat}_think_t1.0.jsonl"
done < <(tail -n +2 results/lcb_v6_full1046/batch_assignment.csv)
[[ -n "${STUB_BAD_LINE:-}" ]] && echo '{"task_id": "trunc' >> "$d/${cfg}_atcoder_think_t1.0.jsonl"
echo "STUB DONE $cfg batch $batch"
""".replace("__SLUG__", SLUG).replace("__SENTINEL__", SAMPLE_SENTINEL)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path: Path):
    stubs = tmp_path / "stubs"; stubs.mkdir()
    (stubs / "python").write_text(PY_STUB)
    (stubs / "nvidia-smi").write_text(
        '#!/usr/bin/env bash\necho "0, ${STUB_MEM0:-5}"\n'
        '[[ -n "${STUB_ONE_GPU:-}" ]] || echo "1, ${STUB_MEM1:-7}"\n')
    for f in stubs.iterdir():
        f.chmod(0o755)
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", BRANCH, str(repo)], check=True)
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "scripts").mkdir(); (repo / "bench").mkdir()
    runner = repo / RUNNER_REL
    runner.write_text(RUNNER_TXT)
    (repo / "bench/generator_vllm.py").write_text("# stub\n")
    ba = repo / "results/lcb_v6_full1046"; ba.mkdir(parents=True)
    (ba / "batch_assignment.csv").write_text("task_id,platform,difficulty,batch\n" + "\n".join(CSV_ROWS) + "\n")
    (repo / ".gitignore").write_text("logs/\nresults/pless_variants_cuda/\n")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    full = repo / f"results/pless_variants_cuda/{SLUG}/LCB_v6_full1046"; full.mkdir(parents=True)
    e = dict(os.environ, REPO=str(repo), BRANCH=BRANCH,
             PYTHON_BIN=str(stubs / "python"), JSON_PYTHON=sys.executable,
             NVIDIA_SMI_BIN=str(stubs / "nvidia-smi"), POLL_SECONDS="1",
             EXPECT_RUNNER_SHA256=_sha(runner), GPU2_LAUNCHER_DETACHED="1",
             PGREP_SCOPE=str(tmp_path))
    for k in ("DRY_RUN", "EXPECT_GENVLLM_SHA256", "VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME"):
        e.pop(k, None)
    yield {"repo": repo, "env": e, "full": full, "tmp": tmp_path, "stubs": stubs}
    subprocess.run(["pkill", "-f", f"{repo}/logs/gk045_launch_"], check=False)  # chains first
    subprocess.run(["pkill", "-f", f"{repo}/logs/gk_launch_"], check=False)     # non-default GK_CONFIGS
    subprocess.run(["pkill", "-f", f"{stubs}/python"], check=False)
    subprocess.run(["pkill", "-f", f"{tmp_path}/FAKE_GK_MARKER"], check=False)


def _run(e: dict, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], env={**e, **extra},
                          capture_output=True, text=True, timeout=180)


def _qlog(repo: Path) -> str:
    p = repo / "logs/gpu2_launch_gk.log"
    return p.read_text() if p.exists() else ""


def _stub_pids(stubs: Path, batch: int) -> list[str]:
    r = subprocess.run(["pgrep", "-f", f"{stubs}/python .*--batch {batch} "],
                       capture_output=True, text=True)
    return r.stdout.split()


def _fake_proc(env, args: str, secs: int = 30) -> None:
    """Background process whose cmdline is '<tmp>/FAKE_GK_MARKER <args>' (in PGREP_SCOPE)."""
    subprocess.run(["bash", "-c", f"(exec -a '{env['tmp']}/FAKE_GK_MARKER {args}' sleep {secs}) >/dev/null 2>&1 &"],
                   check=True)
    time.sleep(0.3)


def _assert_nothing_launched(env) -> None:
    assert not list(env["repo"].glob("logs/gk045_launch_*/batch*"))
    assert not list(env["repo"].glob("logs/ministral_fulldataset_batch*_gk045_gk035.log"))
    assert not _stub_pids(env["stubs"], 1) and not _stub_pids(env["stubs"], 2)


def _batch_log(env, b: int) -> str:
    p = env["repo"] / f"logs/ministral_fulldataset_batch{b}_gk045_gk035.log"
    return p.read_text() if p.exists() else ""


def _wait_for(env, b: int, needle: str, secs: float = 40) -> str:
    t0 = time.time()
    while time.time() - t0 < secs:
        txt = _batch_log(env, b)
        if needle in txt:
            return txt
        time.sleep(0.3)
    raise AssertionError(f"'{needle}' not in batch {b} log:\n{_batch_log(env, b)}")


# ---------------------------------------------------------------- happy path
def test_full_stub_run_both_ready(env):
    r = _run(env["env"], STUB_SLEEP="30")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log + r.stderr
    assert "both engines ready" in log
    assert "vllm version 0.21.0 (matches expected)" in log
    assert "batch_assignment batch 1: total=3 atcoder=2 leetcode=1" in log
    assert "batch_assignment batch 2: total=2 atcoder=1 leetcode=1" in log
    # STAGGER=ready: batch 2 only starts after batch 1's engine was verified
    assert log.index("batch 1 launch verified") < log.index("batch 2 launch (GPU 1)")
    assert "child of chain" in log  # runner python found under the wrapper bash
    assert "Config order per batch" in log and "Batches are independent" in log
    assert "HINT: shared-file integrity" in log
    ldirs = list(env["repo"].glob("logs/gk045_launch_*"))
    assert len(ldirs) == 1
    assert (ldirs[0] / "integrity_check.sh").exists() and (ldirs[0] / "check_step_complete.py").exists()
    for b, gpu in ((1, 0), (2, 1)):
        d = ldirs[0] / f"batch{b}"
        for f in ("cmdline.txt", "environ.txt", f"resume_batch{b}.sh", f"chain_batch{b}.sh"):
            assert (d / f).exists(), f
        runlog = _batch_log(env, b)
        # per-batch GPU, exact PATH, no FlashInfer / CUDA_HOME
        assert f"CVD={gpu} PATH={env['repo']}/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin " in runlog
        assert "FI=unset CUDA_HOME=unset" in runlog
        assert f"PYTHONPATH={env['repo']}" in runlog
        # first invocation: gk_0_45 ONLY, full mirrored argument list
        assert (f"ARGS: {RUNNER_REL} --model mistralai/Ministral-3-3B-Reasoning-2512 --dtype bfloat16 "
                "--gpu-memory-utilization 0.92 --max-model-len 37500 --n-samples 10 "
                f"--max-new-tokens 35000 --configs gk_0_45 --full-dataset --batch {b} "
                "--budget-minutes inf --per-config-minutes inf --seed 42") in runlog
        assert "--configs gk_0_35" not in runlog  # step 2 not started while step 1 runs
        assert f"[gk-chain batch {b}] CHAIN START" in runlog
        ch = (d / f"chain_batch{b}.sh").read_text()
        i45, igate, i35 = ch.index("--configs gk_0_45 "), ch.index("check_step_complete.py"), ch.index("--configs gk_0_35 ")
        assert i45 < igate < i35
        assert ch.count(f"CUDA_VISIBLE_DEVICES={gpu} ") == 2 and ch.count("/snap/bin") == 2
        assert ch.count("env -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME ") == 3  # 2 steps + gate
        rs = (d / f"resume_batch{b}.sh").read_text()
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in rs and "PATH=" in rs and "/snap/bin" in rs
        assert "PYTHONPATH=" in rs and f"chain_batch{b}.sh" in rs and f"--batch {b}" in rs
        assert "nohup env -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME " in rs
        assert "< /dev/null" in rs and "nvidia-smi" in rs
        for s in (f"resume_batch{b}.sh", f"chain_batch{b}.sh"):
            subprocess.run(["bash", "-n", str(d / s)], check=True)
    # resume script refuses while that batch's runner is alive
    rr = subprocess.run(["bash", str(ldirs[0] / "batch1/resume_batch1.sh")],
                        capture_output=True, text=True, timeout=30)
    assert rr.returncode == 1 and "already running" in rr.stderr
    # second launcher instance refuses (runner running)
    r2 = _run(env["env"])
    assert r2.returncode != 0
    assert "STOP: a runner process is already running" in _qlog(env["repo"])


# ---------------------------------------------------------------- chain + completeness gate
def test_chain_runs_gk045_to_completion_then_gk035(env):
    r = _run(env["env"], STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(env["repo"])
    for b, counts in ((1, "2/1"), (2, "1/1")):
        txt = _wait_for(env, b, f"[gk-chain batch {b}] STEP 2/2 (gk_0_35) finished rc=0; CHAIN DONE")
        order = [txt.index(s) for s in (
            f"[gk-chain batch {b}] CHAIN START", "--configs gk_0_45 --full-dataset",
            f"STUB DONE gk_0_45 batch {b}",
            f"[gk-chain batch {b}] STEP 1/2 (gk_0_45) finished rc=0; verifying",
            f"[gk-chain batch {b}] STEP 1 COMPLETE verified ({counts})",
            f"[gk-chain batch {b}] starting step 2/2: --configs gk_0_35",
            "--configs gk_0_35 --full-dataset", f"STUB DONE gk_0_35 batch {b}")]
        assert order == sorted(order)
        assert txt.count("ARGS: ") == 2
        assert SAMPLE_SENTINEL not in txt  # the gate never prints sample text


def test_first_step_fails_second_not_started(env):
    r = _run(env["env"], STUB_SLEEP="1", STUB_FAIL_CONFIG="gk_0_45")
    assert r.returncode == 0, _qlog(env["repo"])  # engines were ready; failure is later
    for b in (1, 2):
        _wait_for(env, b, f"[gk-chain batch {b}] STEP 1/2 (gk_0_45) FAILED rc=5 -- step 2/2 (gk_0_35) NOT started")
        time.sleep(1.5)
        txt = _batch_log(env, b)
        assert "--configs gk_0_35" not in txt and txt.count("ARGS: ") == 1


def test_gate_missing_task_stops_step2_other_batch_continues(env):
    r = _run(env["env"], STUB_SLEEP="1", STUB_SKIP_TASK="abc1_b")
    assert r.returncode == 0, _qlog(env["repo"])
    txt = _wait_for(env, 1, "[gk-chain batch 1] STEP 1 INCOMPLETE: missing=1 dup=0 bad=0 -- step 2/2 (gk_0_35) NOT started")
    assert "atcoder: missing (<=10): ['abc1_b']" in txt
    # batch 2 is independent and complete -> proceeds to gk_0_35
    _wait_for(env, 2, "[gk-chain batch 2] STEP 2/2 (gk_0_35) finished rc=0; CHAIN DONE")
    time.sleep(1)
    assert "--configs gk_0_35" not in _batch_log(env, 1)


def test_gate_duplicate_stops_step2(env):
    r = _run(env["env"], STUB_SLEEP="1", STUB_DUP_TASK="lc1_a")
    assert r.returncode == 0
    txt = _wait_for(env, 1, "STEP 1 INCOMPLETE: missing=0 dup=1 bad=0 -- step 2/2 (gk_0_35) NOT started")
    assert "leetcode: duplicated (<=10): ['lc1_a']" in txt
    time.sleep(1)
    assert "--configs gk_0_35" not in _batch_log(env, 1)


def test_gate_unparseable_stops_step2(env):
    r = _run(env["env"], STUB_SLEEP="1", STUB_BAD_LINE="1")
    assert r.returncode == 0
    txt = _wait_for(env, 1, "-- step 2/2 (gk_0_35) NOT started")
    assert re.search(r"STEP 1 INCOMPLETE: missing=0 dup=0 bad=[1-9]", txt), txt
    time.sleep(1)
    assert "--configs gk_0_35" not in _batch_log(env, 1)


def test_gate_crlf_batch_csv(env):
    csvp = env["repo"] / "results/lcb_v6_full1046/batch_assignment.csv"
    csvp.write_bytes(("task_id,platform,difficulty,batch\r\n" + "\r\n".join(CSV_ROWS) + "\r\n").encode())
    _git(env["repo"], "commit", "-qam", "crlf")
    r = _run(env["env"], STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "batch_assignment batch 1: total=3 atcoder=2 leetcode=1" in _qlog(env["repo"])
    _wait_for(env, 1, "[gk-chain batch 1] STEP 1 COMPLETE verified (2/1)")


def test_resume_script_reruns_chain_and_refuses_on_chain(env):
    r = _run(env["env"], STUB_SLEEP="1", STUB_FAIL_CONFIG="gk_0_45")
    assert r.returncode == 0
    _wait_for(env, 1, "NOT started")
    _wait_for(env, 2, "NOT started")
    ldir = next(env["repo"].glob("logs/gk045_launch_*")) / "batch1"
    # nothing running now -> resume launches the full chain again; GPU0 busy only WARNS
    e = {k: v for k, v in env["env"].items() if k != "STUB_FAIL_CONFIG"}
    rr = subprocess.run(["bash", str(ldir / "resume_batch1.sh")],
                        env={**e, "STUB_SLEEP": "3", "STUB_MEM0": "9000"},
                        capture_output=True, text=True, timeout=30)
    assert rr.returncode == 0 and "launched batch 1 chain PID" in rr.stdout, rr.stderr
    assert "WARNING: GPU 0 memory.used=9000 MiB >= 2000 (continuing)" in rr.stderr
    time.sleep(0.5)
    # while the resumed chain runs, a second resume refuses
    rr2 = subprocess.run(["bash", str(ldir / "resume_batch1.sh")], capture_output=True, text=True, timeout=30)
    assert rr2.returncode == 1 and "already running" in rr2.stderr
    txt = _wait_for(env, 1, "CHAIN DONE")
    assert txt.count("[gk-chain batch 1] CHAIN START") == 2
    assert "CVD=0 PATH=" in txt.split("CHAIN START")[-1]


def test_integrity_check_script(env):
    r = _run(env["env"], STUB_SLEEP="30")
    assert r.returncode == 0
    chk = next(env["repo"].glob("logs/gk045_launch_*")) / "integrity_check.sh"
    f = env["full"] / "gk_0_45_atcoder_think_t1.0.jsonl"
    f.write_text('{"task_id": "abc1_a"}\n{"task_id": "abc2_a"}\n')
    ok = subprocess.run(["bash", str(chk)], capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0 and "records=2 distinct=2 unparseable=0 duplicate_task_ids=0" in ok.stdout
    f.write_text('{"task_id": "abc1_a"}\n{"task_id": "abc1_a"}\n{"task_id": "tr\n')
    bad = subprocess.run(["bash", str(chk)], capture_output=True, text=True, timeout=30)
    assert bad.returncode == 1 and "unparseable=1 duplicate_task_ids=1" in bad.stdout


# ---------------------------------------------------------------- env leak
def test_flashinfer_and_cuda_home_are_stripped(env):
    leak = {"VLLM_USE_FLASHINFER_SAMPLER": "1", "CUDA_HOME": "/usr/local/cuda-13.1"}
    r = _run({**env["env"], **leak}, STUB_SLEEP="30")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "WARNING: VLLM_USE_FLASHINFER_SAMPLER='1' is set in the launcher env" in log
    assert "WARNING: CUDA_HOME='/usr/local/cuda-13.1' is set in the launcher env" in log
    ldir = next(env["repo"].glob("logs/gk045_launch_*"))
    for b in (1, 2):
        assert "FI=unset CUDA_HOME=unset" in _batch_log(env, b)
        envtxt = (ldir / f"batch{b}/environ.txt").read_text()
        assert "VLLM_USE_FLASHINFER_SAMPLER=" not in envtxt and "\nCUDA_HOME=" not in "\n" + envtxt
        for s in (f"chain_batch{b}.sh", f"resume_batch{b}.sh", "cmdline.txt"):
            txt = (ldir / f"batch{b}/{s}").read_text()
            assert "env -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME" in txt
            assert "VLLM_USE_FLASHINFER_SAMPLER=" not in txt and "CUDA_HOME=" not in txt


def test_flashinfer_stripped_in_dry_run_commands(env):
    r = _run({**env["env"], "VLLM_USE_FLASHINFER_SAMPLER": "1", "CUDA_HOME": "/x"}, DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    cmds = [l for l in log.splitlines() if "would run: env" in l or "launched as:" in l]
    assert len(cmds) == 6
    for l in cmds:
        assert "env -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME" in l
        assert "VLLM_USE_FLASHINFER_SAMPLER=" not in l and "CUDA_HOME=" not in l


# ---------------------------------------------------------------- detach / failure handling
def test_detaches_and_child_takes_lock(env):
    e = {k: v for k, v in env["env"].items() if k != "GPU2_LAUNCHER_DETACHED"}
    t0 = time.time()
    r = _run(e, STUB_SLEEP="30")
    assert r.returncode == 0 and time.time() - t0 < 10
    assert "launcher detached (pid" in r.stdout
    assert "tail -f " + str(env["repo"] / "logs/gpu2_launch_gk.log") in r.stdout
    t0 = time.time()
    while "both engines ready" not in _qlog(env["repo"]) and time.time() - t0 < 60:
        time.sleep(0.5)
    log = _qlog(env["repo"])
    assert "both engines ready" in log
    assert log.count("==== launcher start") == 1 and "detached=1" in log  # lock+gates ran in the child


def test_one_batch_dies_other_keeps_running(env):
    r = _run(env["env"], STUB_DIE_BATCH="2", STUB_SLEEP="30")
    log = _qlog(env["repo"])
    assert r.returncode == 1
    assert "batch 2 FAILED: chain PID" in log and "exited before 'Engine ready'" in log
    assert "Traceback: boom batch 2" in log
    assert "[gk-chain batch 2] STEP 1/2 (gk_0_45) FAILED rc=3 -- step 2/2 (gk_0_35) NOT started" in log
    assert "batch 1 OK" in log and "KEEPS RUNNING" in log
    assert _stub_pids(env["stubs"], 1), "batch 1 must not be killed"


def test_batch1_dies_batch2_still_launched(env):
    r = _run(env["env"], STUB_DIE_BATCH="1", STUB_SLEEP="30")
    log = _qlog(env["repo"])
    assert r.returncode == 1
    assert "launching batch 2 anyway" in log
    assert "batch 1 FAILED" in log and "batch 2 OK" in log
    assert _stub_pids(env["stubs"], 2)


def test_stop_after_batch1_launched_says_it_keeps_running(env):
    p = subprocess.Popen(["bash", str(SCRIPT)], env={**env["env"], "STUB_SLEEP": "30", "STUB_READY_DELAY": "4"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while "batch 1 chain started" not in _qlog(env["repo"]) and time.time() - t0 < 30:
        time.sleep(0.2)
    _fake_proc(env, "python run_pless_variants_ministral_3b_lcb_cuda.py --batch 2 --x")
    assert p.wait(timeout=60) == 1
    log = _qlog(env["repo"])
    assert "STOP: a runner or chain for batch 2 appeared meanwhile" in log
    assert "already launched: batch 1 (chain PID" in log and "KEEPS RUNNING (not killed)" in log
    assert "nothing launched" not in log
    assert _stub_pids(env["stubs"], 1)


def test_launch_verify_ignores_old_engine_ready_lines(env):
    (env["repo"] / "logs").mkdir(exist_ok=True)
    (env["repo"] / "logs/ministral_fulldataset_batch2_gk045_gk035.log").write_text("Engine ready in 99s\n")
    r = _run(env["env"], STUB_NO_READY_BATCH="2", LAUNCH_VERIFY_SECONDS="3", STUB_SLEEP="30")
    log = _qlog(env["repo"])
    assert r.returncode == 1
    assert "batch 2 FAILED: no 'Engine ready' within 3s" in log and "NOT killed" in log
    assert "batch 2 launch verified" not in log


def test_engine_ready_after_carriage_return(env):
    r = _run(env["env"], STUB_CR="1", STUB_SLEEP="30")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "launch verified (step 1/2, gk_0_45): 'Engine ready in 1s'" in _qlog(env["repo"])


def test_fixed_stagger_mode(env):
    r = _run(env["env"], STAGGER="1", STUB_SLEEP="30")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "run: sleep 1" in _qlog(env["repo"])


def test_existing_results_logged_not_gated(env):
    f = env["full"] / "gk_0_45_atcoder_think_t1.0.jsonl"
    f.write_text("".join(json.dumps({"task_id": t}) + "\n" for t in ("abc1_a", "abc1_b", "abc2_a")))
    r = _run(env["env"], DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "gk_0_45/atcoder lines=3 batch1=2 batch2=1 other_batches=0 unparseable=0" in log
    assert "gk_0_35/leetcode none" in log


# ---------------------------------------------------------------- dry run
def test_dry_run_prints_and_launches_nothing(env):
    e = {k: v for k, v in env["env"].items() if k != "GPU2_LAUNCHER_DETACHED"}  # DRY stays in foreground
    r = _run(e, DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "launcher detached" not in r.stdout
    for b, gpu in ((1, 0), (2, 1)):
        assert f"DRY: batch {b} (GPU {gpu}) env: -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH=" in log
        s1 = log.index(f"DRY: batch {b} step 1/2 would run: env -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME CUDA_VISIBLE_DEVICES={gpu} ")
        sg = log.index(f"DRY: batch {b} completeness gate would run: ")
        s2 = log.index(f"DRY: batch {b} step 2/2 would run (only if step 1 exits 0 and the gate passes): env -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME CUDA_VISIBLE_DEVICES={gpu} ")
        assert s1 < sg < s2
        assert f"--configs gk_0_45 --full-dataset --batch {b}" in log[s1:sg]
        assert f"--configs gk_0_35 --full-dataset --batch {b}" in log[s2:]
        assert f"chain_batch{b}.sh >> " in log
        assert f"ministral_fulldataset_batch{b}_gk045_gk035.log 2>&1 < /dev/null &" in log
    assert "DRY RUN complete" in log and "Config order per batch" in log
    _assert_nothing_launched(env)


# ---------------------------------------------------------------- gates
def _assert_stopped(env, r, needle):
    log = _qlog(env["repo"])
    assert r.returncode != 0
    assert needle in log, log
    _assert_nothing_launched(env)


def test_default_pgrep_patterns(env):
    rec = env["tmp"] / "pgrep_args.txt"
    stub = env["stubs"] / "pgrep"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> {rec}\nexit 1\n'); stub.chmod(0o755)
    e = {k: v for k, v in env["env"].items() if k != "PGREP_SCOPE"}
    r = _run(e, DRY_RUN="1", PGREP_BIN=str(stub))
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert r"pgrep patterns: runner='run_pless_variants_ministral_3b_lcb_cuda.py' chain='gk045_launch_.*/chain_batch[0-9]+\.sh'" in log
    calls = rec.read_text().splitlines()
    assert "-f run_pless_variants_ministral_3b_lcb_cuda.py" in calls
    assert r"-f gk045_launch_.*/chain_batch[0-9]+\.sh" in calls


def test_running_job_stops(env):
    _fake_proc(env, "python scripts/run_pless_variants_ministral_3b_lcb_cuda.py --batch 3")
    _assert_stopped(env, _run(env["env"]), "STOP: a runner process is already running")


def test_running_chain_wrapper_gate(env):
    _fake_proc(env, "bash /x/logs/gk045_launch_1/batch1/chain_batch1.sh")
    _assert_stopped(env, _run(env["env"]), "STOP: a gk chain wrapper is already running")


def test_busy_gpu_stops(env):
    _assert_stopped(env, _run(env["env"], STUB_MEM1="9000"), "STOP: GPU1 busy: memory.used=9000 MiB >= 2000")
    _assert_stopped(env, _run(env["env"], STUB_MEM0="2000"), "STOP: GPU0 busy: memory.used=2000 MiB")


def test_single_gpu_reported_stops(env):
    _assert_stopped(env, _run(env["env"], STUB_ONE_GPU="1"), "did not report both GPU 0 and GPU 1")


def test_nvidia_smi_failing_stops_after_5(env):
    (env["stubs"] / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 9\n")
    r = _run(env["env"])
    _assert_stopped(env, r, "nvidia-smi failed 5 times in a row")
    assert _qlog(env["repo"]).count("nvidia-smi query failed rc=9") == 5


def test_nvidia_smi_hang_hits_timeout(env):
    (env["stubs"] / "nvidia-smi").write_text("#!/usr/bin/env bash\nexec sleep 100\n")
    t0 = time.time()
    r = _run(env["env"], NVSMI_TIMEOUT_SECONDS="1")
    assert time.time() - t0 < 60
    _assert_stopped(env, r, "nvidia-smi failed 5 times in a row")
    assert _qlog(env["repo"]).count("nvidia-smi timed out after 1s") == 5


def test_launch_verify_minutes_must_be_integer(env):
    _assert_stopped(env, _run(env["env"], LAUNCH_VERIFY_MINUTES="0.5"), "LAUNCH_VERIFY_MINUTES must be a whole number")


def test_wrong_branch_stops(env):
    _git(env["repo"], "checkout", "-qb", "other")
    _assert_stopped(env, _run(env["env"]), "STOP: branch is 'other'")


def test_dirty_tree_stops(env):
    (env["repo"] / "bench/generator_vllm.py").write_text("# modified\n")
    _assert_stopped(env, _run(env["env"]), "STOP: tracked files are dirty")


def test_missing_config_row_stops(env):
    runner = env["repo"] / RUNNER_REL
    runner.write_text('CONFIGS = [\n    ("gk_0_35", "pless_renyi", 1.0, 0.35),\n]\n# "gk_0_45" only in a comment\n')
    _git(env["repo"], "commit", "-qam", "old runner")
    _assert_stopped(env, _run(env["env"], EXPECT_RUNNER_SHA256=_sha(runner)),
                    'STOP: runner lacks a CONFIGS row for "gk_0_45"')


def test_runner_sha_mismatch_stops(env):
    r = _run(env["env"], EXPECT_RUNNER_SHA256="0" * 64)
    _assert_stopped(env, r, f"STOP: sha256 mismatch for {RUNNER_REL}: expected={'0' * 64} actual=")


def test_missing_runner_sha_stops(env):
    e = dict(env["env"]); e.pop("EXPECT_RUNNER_SHA256")
    r = _run(e)
    _assert_stopped(env, r, "STOP: EXPECT_RUNNER_SHA256 is REQUIRED")
    assert _sha(env["repo"] / RUNNER_REL) in _qlog(env["repo"])  # actual value shown
    _assert_stopped(env, _run(env["env"], EXPECT_RUNNER_SHA256=""), "EXPECT_RUNNER_SHA256 is REQUIRED")


def test_malformed_runner_sha_stops(env):
    _assert_stopped(env, _run(env["env"], EXPECT_RUNNER_SHA256="abc"), "is not 64 hex chars")


def test_runner_sha_uppercase_accepted(env):
    r = _run(env["env"], DRY_RUN="1", EXPECT_RUNNER_SHA256=_sha(env["repo"] / RUNNER_REL).upper())
    assert r.returncode == 0, _qlog(env["repo"])


def test_runner_sha_any_warns_loudly(env):
    r = _run(env["env"], DRY_RUN="1", EXPECT_RUNNER_SHA256="any")
    assert r.returncode == 0
    assert f"WARNING: ******** sha256 pin for {RUNNER_REL} DISABLED" in _qlog(env["repo"])


def test_genvllm_pin_optional_and_enforced(env):
    r = _run(env["env"], DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0 and "no pin for bench/generator_vllm.py" in log
    assert f"sha256 bench/generator_vllm.py = {_sha(env['repo'] / 'bench/generator_vllm.py')}" in log
    _assert_stopped(env, _run(env["env"], EXPECT_GENVLLM_SHA256="f" * 64),
                    "STOP: sha256 mismatch for bench/generator_vllm.py")


def test_vllm_version_mismatch_stops(env):
    _assert_stopped(env, _run(env["env"], STUB_VLLM="0.32.0"),
                    "STOP: vllm version is '0.32.0', expected '0.21.0'")


def test_unlocked_append_warning(env):
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0
    assert "no append_result/flock" in _qlog(env["repo"])


def test_missing_batch_rows_stop(env):
    csvp = env["repo"] / "results/lcb_v6_full1046/batch_assignment.csv"
    csvp.write_text("task_id,platform,difficulty,batch\nabc0_a,atcoder,easy,0\nabc1_a,atcoder,easy,1\n")
    _git(env["repo"], "commit", "-qam", "csv")
    _assert_stopped(env, _run(env["env"]), "STOP: batch_assignment.csv has no rows for batch 2")


# ---------------------------------------------------------------- lock
def test_pidfile_lock_held_by_live_process_stops(env):
    (env["repo"] / "logs").mkdir(exist_ok=True)
    (env["repo"] / "logs/gpu2_launch_gk.lock").write_text(f"{os.getpid()}\n")
    _assert_stopped(env, _run(env["env"], FLOCK_BIN=""), "holds")
    assert "(pidfile)" in _qlog(env["repo"])


def test_pidfile_stale_lock_taken_over(env):
    (env["repo"] / "logs").mkdir(exist_ok=True)
    (env["repo"] / "logs/gpu2_launch_gk.lock").write_text("999999\n")
    r = _run(env["env"], FLOCK_BIN="", DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "stale pidfile lock" in _qlog(env["repo"])


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock not installed (Linux path)")
def test_flock_lock_held_stops(env):
    (env["repo"] / "logs").mkdir(exist_ok=True)
    lock = env["repo"] / "logs/gpu2_launch_gk.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "30"])
    try:
        time.sleep(0.5)
        _assert_stopped(env, _run(env["env"]), "holds")
    finally:
        holder.kill()


# ================================================================ GK_CONFIGS generalisation
EXPECTED_DEFAULT_DRY = Path(__file__).resolve().parent / "data/gpu2_launch_gk_default_dry_expected.txt"
NEW_ROWS = ('    ("gk_0_25",          "pless_renyi",      1.0, 0.25),\n'
            '    ("gk_0_15",          "pless_renyi",      1.0, 0.15),\n')


def _add_rows(env, rows: str = NEW_ROWS) -> dict:
    """Commit a runner with extra CONFIGS rows; return env with the matching sha pin."""
    runner = env["repo"] / RUNNER_REL
    runner.write_text(RUNNER_TXT.replace("]\n", rows + "]\n"))
    _git(env["repo"], "commit", "-qam", "runner + rows")
    return {**env["env"], "EXPECT_RUNNER_SHA256": _sha(runner)}


def _blog(env, b: int, tag: str) -> str:
    p = env["repo"] / f"logs/ministral_fulldataset_batch{b}_{tag}.log"
    return p.read_text() if p.exists() else ""


def _wait_tag(env, b: int, tag: str, needle: str, secs: float = 60) -> str:
    t0 = time.time()
    while time.time() - t0 < secs:
        txt = _blog(env, b, tag)
        if needle in txt:
            return txt
        time.sleep(0.3)
    raise AssertionError(f"'{needle}' not in batch {b} log ({tag}):\n{_blog(env, b, tag)}")


def _norm_dry(env, log: str) -> list[str]:
    out = []
    for l in log.splitlines():
        m = re.match(r"^\[[^\]]+ IST\] (DRY: .*)$", l)
        if m:
            s = m.group(1).replace(str(env["tmp"]), "<TMP>").replace(sys.executable, "<PY>")
            out.append(re.sub(r"_launch_\d{8}_\d{6}", "_launch_<TS>", s))
    return out


def _assert_nothing_launched_any(env) -> None:
    _assert_nothing_launched(env)
    assert not list(env["repo"].glob("logs/gk_launch_*/batch*"))
    assert not [p for p in env["repo"].glob("logs/ministral_fulldataset_batch*.log")]


def test_default_dry_run_commands_match_recorded_pre_change_output(env):
    """Recorded from the launcher BEFORE GK_CONFIGS existed. The only allowed
    differences: the two NEW final-gate lines, and the check-script description
    line gains a suffix. Every printed command line is byte-identical."""
    expected = EXPECTED_DEFAULT_DRY.read_text().splitlines()
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])
    got = _norm_dry(env, _qlog(env["repo"]))
    finals = [l for l in got if "final completeness gate" in l]
    rest = [l for l in got if "final completeness gate" not in l]
    assert len(rest) == len(expected)
    assert rest[0].startswith(expected[0][:-1]) and rest[0].endswith(")")
    assert rest[1:] == expected[1:]
    assert len(finals) == 2
    for b, l in zip((1, 2), finals):
        assert l == (f"DRY: batch {b} final completeness gate would run (only if step 2/2 exits 0; chain exits 3 if incomplete): "
                     f"<PY> -I <TMP>/repo/logs/gk045_launch_<TS>/check_step_complete.py "
                     f"<TMP>/repo/results/lcb_v6_full1046/batch_assignment.csv {b} gk_0_35 - --step=2/2 "
                     f"atcoder=<TMP>/repo/results/pless_variants_cuda/{SLUG}/LCB_v6_full1046/gk_0_35_atcoder_think_t1.0.jsonl "
                     f"leetcode=<TMP>/repo/results/pless_variants_cuda/{SLUG}/LCB_v6_full1046/gk_0_35_leetcode_think_t1.0.jsonl")
    # the final line sits between step 2/2 and "would write chain" for its batch
    i_s2 = got.index(next(l for l in got if l.startswith("DRY: batch 1 step 2/2")))
    assert got[i_s2 + 1] == finals[0] and got[i_s2 + 2].startswith("DRY: batch 1 would write ")


def test_explicit_default_list_equals_default(env):
    r = _run(env["env"], DRY_RUN="1", GK_CONFIGS="  gk_0_45   gk_0_35 ")
    assert r.returncode == 0, _qlog(env["repo"])
    log = _qlog(env["repo"])
    assert "/logs/gk045_launch_" in log and "/logs/gk_launch_" not in log
    assert "ministral_fulldataset_batch1_gk045_gk035.log" in log


def test_default_chain_runs_final_gate_after_gk035(env):
    r = _run(env["env"], STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(env["repo"])
    for b, counts in ((1, "2/1"), (2, "1/1")):
        txt = _wait_for(env, b, f"[gk-chain batch {b}] STEP 2/2 (gk_0_35) finished rc=0; CHAIN DONE")
        order = [txt.index(s) for s in (
            f"STUB DONE gk_0_35 batch {b}",
            f"[gk-chain batch {b}] STEP 2/2 (gk_0_35) runner exited rc=0; FINAL check",
            f"[gk-chain batch {b}] FINAL COMPLETE verified ({counts})",
            f"[gk-chain batch {b}] STEP 2/2 (gk_0_35) finished rc=0; CHAIN DONE")]
        assert order == sorted(order)


def test_single_config_gk_0_25(env):
    e = _add_rows(env)
    r = _run(e, GK_CONFIGS="gk_0_25", STUB_SLEEP="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    assert "GK_CONFIGS: gk_0_25 (1 config(s)); launch dir prefix gk_launch_; batch log tag gk025" in log
    assert "launch verified (step 1/1, gk_0_25): 'Engine ready in 1s'" in log
    assert not list(env["repo"].glob("logs/gk045_launch_*"))
    assert not list(env["repo"].glob("logs/*gk045_gk035.log"))
    ldirs = list(env["repo"].glob("logs/gk_launch_*"))
    assert len(ldirs) == 1
    for b, gpu, counts in ((1, 0, "2/1"), (2, 1, "1/1")):
        txt = _wait_tag(env, b, "gk025", f"[gk-chain batch {b}] STEP 1/1 (gk_0_25) finished rc=0; CHAIN DONE")
        order = [txt.index(s) for s in (
            f"[gk-chain batch {b}] CHAIN START", "step 1/1: --configs gk_0_25",
            "--configs gk_0_25 --full-dataset", f"STUB DONE gk_0_25 batch {b}",
            f"[gk-chain batch {b}] STEP 1/1 (gk_0_25) runner exited rc=0; FINAL check: verifying batch-{b} completeness of gk_0_25",
            f"[gk-chain batch {b}] atcoder: gk_0_25 batch{b}_expected=",
            f"[gk-chain batch {b}] FINAL COMPLETE verified ({counts})",
            f"[gk-chain batch {b}] STEP 1/1 (gk_0_25) finished rc=0; CHAIN DONE")]
        assert order == sorted(order)
        assert txt.count("ARGS: ") == 1 and SAMPLE_SENTINEL not in txt
        assert f"CVD={gpu} " in txt and "FI=unset CUDA_HOME=unset" in txt
        d = ldirs[0] / f"batch{b}"
        ch = (d / f"chain_batch{b}.sh").read_text()
        assert ch.count("--configs gk_0_25 ") == 1 and "--configs gk_0_45" not in ch
        assert "gk_0_25 - --step=1/1" in ch and "NOT started" not in ch
        rs = (d / f"resume_batch{b}.sh").read_text()
        assert "Chain configs, in order" in rs and "gk_0_25" in rs and f"chain_batch{b}.sh" in rs
        assert "(gk045|gk)_launch_" in rs and f"_gk025.log" in rs
        cl = (d / "cmdline.txt").read_text()
        assert "step 1: " in cl and "final gate (after step 1): " in cl and "\ngate: " not in cl
        for s in (f"resume_batch{b}.sh", f"chain_batch{b}.sh"):
            subprocess.run(["bash", "-n", str(d / s)], check=True)
    ic = (ldirs[0] / "integrity_check.sh").read_text()
    assert "gk_0_25_atcoder_think_t1.0.jsonl" in ic and "gk_0_25_leetcode_think_t1.0.jsonl" in ic
    assert "gk_0_45" not in ic and "gk_0_35" not in ic
    ok = subprocess.run(["bash", str(ldirs[0] / "integrity_check.sh")], capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0, ok.stdout
    assert "After the last config (gk_0_25) exits 0, a FINAL completeness check runs" in log


def test_single_config_final_incomplete_exit3(env):
    e = _add_rows(env)
    r = _run(e, GK_CONFIGS="gk_0_25", STUB_SLEEP="1", STUB_SKIP_TASK="abc1_b")
    assert r.returncode == 0, _qlog(env["repo"])
    txt = _wait_tag(env, 1, "gk025", "[gk-chain batch 1] FINAL INCOMPLETE: missing=1 dup=0 bad=0 -- gk_0_25 batch 1 NOT complete")
    assert "atcoder: missing (<=10): ['abc1_b']" in txt
    _wait_tag(env, 2, "gk025", "[gk-chain batch 2] FINAL COMPLETE verified (1/1)")
    time.sleep(1)
    assert "CHAIN DONE" not in _blog(env, 1, "gk025")
    # chain exit code is 3 when the final check fails (re-run the chain directly)
    chain = next(env["repo"].glob("logs/gk_launch_*")) / "batch1/chain_batch1.sh"
    rr = subprocess.run(["bash", str(chain)], env={**e, "STUB_SLEEP": "0", "STUB_SKIP_TASK": "abc1_b"},
                        capture_output=True, text=True, timeout=60)
    assert rr.returncode == 3, rr.stdout
    assert "FINAL INCOMPLETE" in rr.stdout


def test_single_config_dry_run(env):
    e = _add_rows(env)
    r = _run(e, GK_CONFIGS="gk_0_25", DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    got = _norm_dry(env, log)
    for b in (1, 2):
        mine = [l for l in got if l.startswith(f"DRY: batch {b} ")]
        assert [l.split(" would")[0] for l in mine] == [
            f"DRY: batch {b} (GPU {b - 1}) env: -u VLLM_USE_FLASHINFER_SAMPLER -u CUDA_HOME CUDA_VISIBLE_DEVICES={b - 1} PYTHONPATH=<TMP>/repo PATH=<TMP>/repo/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin PYTHONUNBUFFERED=1",
            f"DRY: batch {b} step 1/1", f"DRY: batch {b} final completeness gate",
            f"DRY: batch {b}", f"DRY: batch {b}"]
        assert f"--configs gk_0_25 --full-dataset --batch {b} " in mine[1]
        assert f" {b} gk_0_25 - --step=1/1 atcoder=" in mine[2] and "gk_0_25_leetcode_think_t1.0.jsonl" in mine[2]
        assert f"logs/gk_launch_<TS>/batch{b}/chain_batch{b}.sh >> <TMP>/repo/logs/ministral_fulldataset_batch{b}_gk025.log 2>&1 < /dev/null &" in mine[3]
    assert not any("completeness gate would run:" in l or "step 2/" in l for l in got)
    _assert_nothing_launched_any(env)


def test_three_config_chain_order(env):
    e = _add_rows(env)
    cfgs = "gk_0_25 gk_0_15 gk_0_35"
    r = _run(e, GK_CONFIGS=cfgs, STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(env["repo"])
    tag = "gk025_gk015_gk035"
    for b, counts in ((1, "2/1"), (2, "1/1")):
        txt = _wait_tag(env, b, tag, f"[gk-chain batch {b}] STEP 3/3 (gk_0_35) finished rc=0; CHAIN DONE", secs=90)
        order = [txt.index(s) for s in (
            "CHAIN START", "step 1/3: --configs gk_0_25", f"STUB DONE gk_0_25 batch {b}",
            "STEP 1/3 (gk_0_25) finished rc=0; verifying", f"STEP 1 COMPLETE verified ({counts})",
            "starting step 2/3: --configs gk_0_15", f"STUB DONE gk_0_15 batch {b}",
            "STEP 2/3 (gk_0_15) finished rc=0; verifying", f"STEP 2 COMPLETE verified ({counts})",
            "starting step 3/3: --configs gk_0_35", f"STUB DONE gk_0_35 batch {b}",
            "STEP 3/3 (gk_0_35) runner exited rc=0; FINAL check", f"FINAL COMPLETE verified ({counts})",
            "STEP 3/3 (gk_0_35) finished rc=0; CHAIN DONE")]
        assert order == sorted(order)
        assert txt.count("ARGS: ") == 3


def test_three_config_step2_failure_stops_step3(env):
    e = _add_rows(env)
    r = _run(e, GK_CONFIGS="gk_0_25 gk_0_15 gk_0_35", STUB_SLEEP="1", STUB_FAIL_CONFIG="gk_0_15")
    assert r.returncode == 0
    txt = _wait_tag(env, 1, "gk025_gk015_gk035", "STEP 2/3 (gk_0_15) FAILED rc=5 -- step 3/3 (gk_0_35) NOT started")
    time.sleep(1)
    assert "--configs gk_0_35" not in _blog(env, 1, "gk025_gk015_gk035") and "FINAL" not in txt


def test_three_config_dry_run_gates(env):
    e = _add_rows(env)
    r = _run(e, GK_CONFIGS="gk_0_25 gk_0_15 gk_0_35", DRY_RUN="1")
    log = _qlog(env["repo"])
    assert r.returncode == 0, log
    got = [l for l in _norm_dry(env, log) if l.startswith("DRY: batch 1 ")]
    heads = ["DRY: " + l[5:].split(": ", 1)[0] if " env:" not in l else "env" for l in got]
    assert heads[:7] == ["env", "DRY: batch 1 step 1/3 would run",
                         "DRY: batch 1 completeness gate would run",
                         "DRY: batch 1 step 2/3 would run (only if step 1 exits 0 and the gate passes)",
                         "DRY: batch 1 completeness gate after step 2/3 would run",
                         "DRY: batch 1 step 3/3 would run (only if step 2 exits 0 and the gate passes)",
                         "DRY: batch 1 final completeness gate would run (only if step 3/3 exits 0; chain exits 3 if incomplete)"]
    assert " 1 gk_0_25 gk_0_15 --step=1/3 atcoder=" in got[2] and "/gk_0_25_atcoder" in got[2]
    assert " 1 gk_0_15 gk_0_35 --step=2/3 atcoder=" in got[4] and "/gk_0_15_leetcode" in got[4]
    assert " 1 gk_0_35 - --step=3/3 atcoder=" in got[6]
    assert "--configs gk_0_25 " in got[1] and "--configs gk_0_15 " in got[3] and "--configs gk_0_35 " in got[5]
    _assert_nothing_launched_any(env)


@pytest.mark.parametrize("bad", ["gk_0_25x", "GK_0_25", "gk_", "gk_0__25", "gk0_25", "top_p_0_95",
                                 "gk_0_25;id", "gk_0_.25", "gk_0_45 gk_0_3-5"])
def test_invalid_gk_config_names_stop(env, bad):
    e = _add_rows(env)
    r = _run(e, GK_CONFIGS=bad, DRY_RUN="1")
    assert r.returncode != 0
    assert "is not a valid name (must match ^gk_[0-9]+(_[0-9]+)*$)" in _qlog(env["repo"])
    _assert_nothing_launched_any(env)


def test_invalid_name_stops_before_detaching(env):
    e = {k: v for k, v in _add_rows(env).items() if k != "GPU2_LAUNCHER_DETACHED"}
    r = _run(e, GK_CONFIGS="gk_x")
    assert r.returncode != 0 and "launcher detached" not in r.stdout
    assert "STOP: GK_CONFIGS entry 'gk_x' is not a valid name" in r.stdout


def test_blank_gk_configs_stops(env):
    r = _run(env["env"], GK_CONFIGS="   ", DRY_RUN="1")
    assert r.returncode != 0 and "STOP: GK_CONFIGS is empty/blank" in _qlog(env["repo"])


def test_duplicate_gk_configs_stop(env):
    e = _add_rows(env)
    for dup in ("gk_0_25 gk_0_25", "gk_0_45 gk_0_25 gk_0_45"):
        r = _run(e, GK_CONFIGS=dup, DRY_RUN="1")
        assert r.returncode != 0
        assert "lists 'gk_0_" in _qlog(env["repo"]) and "more than once" in _qlog(env["repo"])
    _assert_nothing_launched_any(env)


def test_missing_runner_row_for_new_config_stops(env):
    # default fake runner has only gk_0_35 / gk_0_45 rows; a comment mention does not count
    runner = env["repo"] / RUNNER_REL
    runner.write_text(RUNNER_TXT + '# "gk_0_25" planned\n')
    _git(env["repo"], "commit", "-qam", "comment only")
    r = _run(env["env"], GK_CONFIGS="gk_0_25", EXPECT_RUNNER_SHA256=_sha(runner))
    log = _qlog(env["repo"])
    assert r.returncode != 0 and 'STOP: runner lacks a CONFIGS row for "gk_0_25"' in log
    _assert_nothing_launched_any(env)
    # second config missing also stops (checked per config)
    r = _run(env["env"], GK_CONFIGS="gk_0_45 gk_0_25", EXPECT_RUNNER_SHA256=_sha(runner))
    assert r.returncode != 0 and _qlog(env["repo"]).count('lacks a CONFIGS row for "gk_0_25"') == 2


def test_new_prefix_chain_blocks_any_launch(env):
    _fake_proc(env, "bash /x/logs/gk_launch_1/batch1/chain_batch1.sh")
    _assert_stopped(env, _run(env["env"]), "STOP: a gk chain wrapper is already running")


def test_non_default_still_requires_runner_pin(env):
    e = dict(_add_rows(env)); e.pop("EXPECT_RUNNER_SHA256")
    r = _run(e, GK_CONFIGS="gk_0_25")
    assert r.returncode != 0 and "STOP: EXPECT_RUNNER_SHA256 is REQUIRED" in _qlog(env["repo"])
    _assert_nothing_launched_any(env)
