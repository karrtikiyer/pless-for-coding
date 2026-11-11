"""Stub-based tests for scripts/msi_queue_topk_after_recipe.sh.

Builds a throwaway git repo (with a bare 'origin') that mimics the MSI checkout,
stubs python / nvidia-smi, and runs the launcher end to end. No GPU, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/msi_queue_topk_after_recipe.sh"
SLUG = "mistralai--Ministral-3-3B-Reasoning-2512"
BRANCH = "feat/pless-variants-mps"
AT_IDS = [f"abc{i:03d}_a" for i in range(121)]
LC_IDS = [f"lc{i:04d}" for i in range(90)]

PY_STUB = r"""#!/usr/bin/env bash
# Fake runner. Smoke (--n-samples 2): print engine line + write one record.
# Full run: print the engine-ready line (unless STUB_NO_READY) then sleep.
if [[ " $* " == *" --n-samples 2 "* ]]; then
  echo "Loading engine: x (kwargs={'dtype': 'bfloat16', 'register_pless_logitsproc': ${STUB_REGISTER:-False}})"
  d="results/pless_variants_cuda/__SLUG__/LCB_v6_stratified60"; mkdir -p "$d"
  echo '{"task_id": "abc304_b", "top_k": 15, "backend": "vllm", "sampler_name": "top_k_15", "temperature": 1.0, "top_p": 1.0, "samples": ["a", "b"]}' >> "$d/top_k_15_atcoder_think_t1.0.jsonl"
  exit 0
fi
echo "Loading engine: full FI=$VLLM_USE_FLASHINFER_SAMPLER"
if [[ -n "${STUB_DIE:-}" ]]; then echo "Traceback: boom"; exit 3; fi
[[ -z "${STUB_NO_READY:-}" ]] && echo "Engine ready in 1s"
sleep "${STUB_SLEEP:-20}"
""".replace("__SLUG__", SLUG)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_recipe(path: Path, ids: list[str], extra_lines: str = "") -> None:
    path.write_text("".join(json.dumps({"task_id": t}) + "\n" for t in ids) + extra_lines)


@pytest.fixture
def env(tmp_path: Path):
    stubs = tmp_path / "stubs"; stubs.mkdir()
    (stubs / "python").write_text(PY_STUB)
    (stubs / "nvidia-smi").write_text("#!/usr/bin/env bash\necho \"${STUB_MEM:-5}\"\n")
    for f in stubs.iterdir():
        f.chmod(0o755)

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", BRANCH, str(repo)], check=True)
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "scripts").mkdir(); (repo / "bench").mkdir()
    runner = repo / "scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
    runner.write_text('X = ["top_k_25", "top_k_20", "top_k_15"]\n')
    (repo / "bench/generator_vllm.py").write_text("# stub\n")
    (repo / "bench/eval").mkdir(); (repo / "bench/eval/metrics.py").write_text("# eval\n")
    (repo / "bench/other.py").write_text("# other\n")
    sel = repo / "results/lcb_v6_stratified60"; sel.mkdir(parents=True)
    (sel / "task_selection.csv").write_text(
        "task_id,platform,difficulty,contest_date\nabc304_b,atcoder,easy,2023-06-03\n")
    ba = repo / "results/lcb_v6_full1046"; ba.mkdir(parents=True)
    rows = ([f"{t},atcoder,easy,0" for t in AT_IDS] + [f"{t},leetcode,easy,0" for t in LC_IDS]
            + ["zzz_b1,atcoder,easy,1", "lc_b1,leetcode,hard,1"])
    (ba / "batch_assignment.csv").write_text("task_id,platform,difficulty,batch\n" + "\n".join(rows) + "\n")
    (repo / ".gitignore").write_text("logs/\nresults/pless_variants_cuda/\n")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    pin = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", BRANCH)

    full = repo / f"results/pless_variants_cuda/{SLUG}/LCB_v6_full1046"; full.mkdir(parents=True)
    _write_recipe(full / "ministral_recipe_atcoder_think_t0.7.jsonl", AT_IDS)
    _write_recipe(full / "ministral_recipe_leetcode_think_t0.7.jsonl", LC_IDS)
    (repo / "logs").mkdir()
    (repo / "logs/ministral_fulldataset_batch0_ministral_recipe.log").write_text(
        "".join(f"recipe line {i}\n" for i in range(30))
        + "[ministral_recipe/atcoder] task abc007_a crashed: RuntimeError('x')\n")

    e = dict(os.environ, REPO=str(repo), BRANCH=BRANCH, WAIT_PID="999999",
             PYTHON_BIN=str(stubs / "python"), JSON_PYTHON=sys.executable,
             NVIDIA_SMI_BIN=str(stubs / "nvidia-smi"), POLL_SECONDS="1",
             PIN_COMMIT=pin, EXPECT_RUNNER_SHA256=_sha(runner),
             EXPECT_GENVLLM_SHA256=_sha(repo / "bench/generator_vllm.py"))
    procs: list[subprocess.Popen] = []
    yield {"repo": repo, "env": e, "full": full, "procs": procs, "origin": origin, "tmp": tmp_path}
    subprocess.run(["pkill", "-f", f"{stubs}/python"], check=False)
    subprocess.run(["pkill", "-f", "FAKE_RECIPE_MARKER"], check=False)
    for p in procs:
        p.kill()


def _run(e: dict, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], env={**e, **extra},
                          capture_output=True, text=True, timeout=120)


def _qlog(repo: Path) -> str:
    return (repo / "logs/msi_queue_topk.log").read_text()


def _fake_recipe_proc(seconds: int) -> str:
    """Orphaned process (reaped by init, like the nohup-ed MSI runner) whose
    cmdline matches the recipe-runner pattern."""
    return subprocess.run(
        ["bash", "-c",
         f"(exec -a 'python run_pless_variants_ministral_3b_lcb_cuda.py --configs ministral_recipe FAKE_RECIPE_MARKER' sleep {seconds}) >/dev/null 2>&1 & echo $!"],
        capture_output=True, text=True, check=True).stdout.strip()


def test_dry_run_waits_for_pid_then_passes(env):
    spid = _fake_recipe_proc(3)
    t0 = time.time()
    r = _run(env["env"], DRY_RUN="1", WAIT_PID=spid)
    assert r.returncode == 0, r.stdout + r.stderr
    assert time.time() - t0 >= 2.5  # actually waited for the fake recipe runner
    log = _qlog(env["repo"])
    assert "WARNING: WAIT_PID" not in log
    assert "pull skipped (PULL=0)" in log and "completion gate PASSED" in log
    assert "atcoder: batch0_expected=121 (sanity 121) distinct=121" in log
    assert "code gate PASSED" in log and "matches expected" in log
    assert "DRY: would run: cd" in log and "--configs top_k_25 top_k_20 top_k_15" in log
    assert not list(env["repo"].glob("logs/topk_launch_*"))


def test_wait_pid_cmdline_mismatch_warns(env):
    spid = subprocess.run(["bash", "-c", "sleep 30 >/dev/null 2>&1 & echo $!"],
                          capture_output=True, text=True, check=True).stdout.strip()
    try:
        r = _run(env["env"], DRY_RUN="1", WAIT_PID=spid)
    finally:
        os.kill(int(spid), signal.SIGTERM)
    assert r.returncode == 0
    assert "does not match" in _qlog(env["repo"])


def test_full_stub_run_then_second_instance_refuses(env):
    r = _run(env["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    log = _qlog(env["repo"])
    assert log.count("smoke exit code: 0") == 2
    assert "'n_samples': 2" in log and "smoke tests PASSED" in log
    smoke_dir = env["repo"] / f"results/pless_variants_cuda/{SLUG}/LCB_v6_stratified60"
    assert not list(smoke_dir.glob("top_k_15_*_think_t1.0.jsonl"))  # cleaned up
    ldirs = list(env["repo"].glob("logs/topk_launch_*"))
    assert len(ldirs) == 1
    for f in ("cmdline.txt", "environ.txt", "resume_topk.sh"):
        assert (ldirs[0] / f).exists()
    assert "launch verified: 'Engine ready in 1s'" in log
    r2 = _run(env["env"])
    assert r2.returncode != 0
    assert "top-k runner is already running" in _qlog(env["repo"])


def test_launch_verify_ignores_old_engine_ready_lines(env):
    (env["repo"] / "logs/ministral_fulldataset_batch0_topk.log").write_text("Engine ready in 99s\n")
    r = _run(env["env"], STUB_NO_READY="1", LAUNCH_VERIFY_MINUTES="0.05")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "FAIL: no 'Engine ready' within 0.05 min" in log and "launch verified" not in log


def test_launch_dies_before_ready(env):
    r = _run(env["env"], STUB_DIE="1")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "died before 'Engine ready'" in log and "Traceback: boom" in log


def test_incomplete_recipe_stops(env):
    _write_recipe(env["full"] / "ministral_recipe_atcoder_think_t0.7.jsonl", AT_IDS[:120])
    r = _run(env["env"])
    assert r.returncode != 0
    log = _qlog(env["repo"])
    assert "raw wc -l: atcoder=120 leetcode=90" in log
    assert "missing=1" in log and "'abc120_a'" in log
    assert "STOP: ministral_recipe did NOT finish cleanly" in log
    assert "recipe line 29" in log and "recipe line 10\n" not in log  # last 20 only
    assert "task abc007_a crashed" in log  # grep crashed
    assert "smoke (VLLM" not in log and "launched" not in log.split("STOP:")[0]


def test_duplicate_and_unparseable_lines_stop(env):
    # right line count (121) but one duplicate + one missing
    _write_recipe(env["full"] / "ministral_recipe_atcoder_think_t0.7.jsonl", AT_IDS[:120] + [AT_IDS[0]])
    _write_recipe(env["full"] / "ministral_recipe_leetcode_think_t0.7.jsonl", LC_IDS, '{"task_id": "trunc')
    r = _run(env["env"])
    assert r.returncode != 0
    log = _qlog(env["repo"])
    assert "raw wc -l: atcoder=121" in log  # wc -l alone would have passed
    assert "duplicated=1" in log and "duplicated (<=10): ['abc000_a']" in log
    assert "leetcode:" in log and "unparseable=1" in log
    assert "STOP: ministral_recipe did NOT finish cleanly" in log


def test_dirty_tracked_tree_stops(env):
    (env["repo"] / "bench/generator_vllm.py").write_text("# modified\n")
    r = _run(env["env"])
    assert r.returncode != 0
    assert "STOP: tracked files are dirty" in _qlog(env["repo"])


def test_wrong_branch_stops(env):
    _git(env["repo"], "checkout", "-qb", "other")
    r = _run(env["env"])
    assert r.returncode != 0
    assert "STOP: branch is 'other'" in _qlog(env["repo"])


def test_sha_pin_mismatch_stops(env):
    r = _run(env["env"], EXPECT_RUNNER_SHA256="0" * 64)
    assert r.returncode != 0
    log = _qlog(env["repo"])
    assert "STOP: sha256 mismatch for scripts/run_pless_variants_ministral_3b_lcb_cuda.py" in log
    assert "expected=" + "0" * 64 in log and "actual=" in log


def test_sha_pin_any_warns_loudly(env):
    r = _run(env["env"], DRY_RUN="1", EXPECT_GENVLLM_SHA256="any")
    assert r.returncode == 0
    assert "WARNING: sha256 pin for bench/generator_vllm.py DISABLED" in _qlog(env["repo"])


def test_pin_commit_not_ancestor_stops(env):
    # commit on a side branch, not an ancestor of HEAD
    _git(env["repo"], "checkout", "-qb", "side")
    (env["repo"] / "side.txt").write_text("x\n"); _git(env["repo"], "add", "side.txt")
    _git(env["repo"], "commit", "-qm", "side")
    side = _git(env["repo"], "rev-parse", "HEAD")
    _git(env["repo"], "checkout", "-q", BRANCH)
    r = _run(env["env"], PIN_COMMIT=side)
    assert r.returncode != 0
    assert f"STOP: pinned commit {side} is not an ancestor" in _qlog(env["repo"])


def test_pull_opt_in_fast_forwards(env):
    other = env["tmp"] / "other"
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, str(env["origin"]), str(other)], check=True)
    _git(other, "config", "user.email", "t@t"); _git(other, "config", "user.name", "t")
    (other / "README").write_text("new\n"); _git(other, "add", "README")
    _git(other, "commit", "-qm", "upstream"); _git(other, "push", "-q", "origin", BRANCH)
    new_head = _git(other, "rev-parse", "HEAD")
    r = _run(env["env"], DRY_RUN="1")  # PULL default 0: HEAD unchanged
    assert r.returncode == 0 and _git(env["repo"], "rev-parse", "HEAD") != new_head
    r = _run(env["env"], DRY_RUN="0", PULL="1", STUB_SLEEP="1")
    assert r.returncode == 0, _qlog(env["repo"])
    assert _git(env["repo"], "rev-parse", "HEAD") == new_head


def test_preexisting_smoke_file_is_preserved(env):
    d = env["repo"] / f"results/pless_variants_cuda/{SLUG}/LCB_v6_stratified60"
    d.mkdir(parents=True)
    keep = d / "top_k_15_atcoder_think_t1.0.jsonl"; keep.write_text("precious\n")
    r = _run(env["env"])
    assert r.returncode != 0
    assert "pre-existing smoke-target file" in _qlog(env["repo"])
    assert keep.read_text() == "precious\n"


def test_logitsproc_registered_fails_smoke_and_cleans_up(env):
    r = _run(env["env"], STUB_REGISTER="True")
    assert r.returncode != 0
    log = _qlog(env["repo"])
    assert "FAIL: custom logits processor not disabled" in log
    assert "STOP: smoke test failed" in log
    assert "full top-k runner started" not in log
    d = env["repo"] / f"results/pless_variants_cuda/{SLUG}/LCB_v6_stratified60"
    assert not list(d.glob("top_k_15_*_think_t1.0.jsonl"))


def test_gpu_busy_keeps_waiting(env):
    p = subprocess.Popen(["bash", str(SCRIPT)], env={**env["env"], "STUB_MEM": "9000"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env["procs"].append(p)
    time.sleep(4)
    assert p.poll() is None  # still waiting (default 30-min cap)
    p.send_signal(signal.SIGTERM); p.wait(timeout=10)
    log = _qlog(env["repo"])
    assert "completion gate PASSED" in log  # count gate ran BEFORE the GPU wait
    assert "memory.used=9000 MiB >= 2000" in log


def test_gpu_wait_cap_stops(env):
    r = _run(env["env"], STUB_MEM="9000", GPU_WAIT_MAX_MINUTES="0.05")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "STOP: [pre-smoke-fi1] GPU still held after 0.05 min" in log
    assert "nvidia-smi memory.used=9000 MiB" in log


# ---------------------------------------------------------------- round-2 review
def _commit_change(repo: Path, rel: str, text: str) -> None:
    (repo / rel).write_text(text); _git(repo, "add", rel); _git(repo, "commit", "-qm", f"edit {rel}")


def test_nvidia_smi_failing_stops_after_5(env):
    (env["tmp"] / "stubs/nvidia-smi").write_text("#!/usr/bin/env bash\nexit 9\n")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "nvidia-smi exited rc=9" in log and "nvidia-smi failed 5 times in a row" in log


def test_nvidia_smi_hang_hits_timeout(env):
    (env["tmp"] / "stubs/nvidia-smi").write_text("#!/usr/bin/env bash\nexec sleep 100\n")
    t0 = time.time()
    r = _run(env["env"], DRY_RUN="1", NVSMI_TIMEOUT_SECONDS="1")
    assert r.returncode == 1
    assert time.time() - t0 < 60
    log = _qlog(env["repo"])
    assert log.count("nvidia-smi timed out after 1s") == 5
    assert "nvidia-smi failed 5 times in a row" in log


def test_enginecore_lingers_then_clears(env):
    subprocess.run(["bash", "-c", "(exec -a 'VLLM::EngineCore FAKE_RECIPE_MARKER' sleep 4) >/dev/null 2>&1 &"], check=True)
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])
    log = _qlog(env["repo"])
    assert "EngineCore still present" in log and "GPU idle" in log


def test_enginecore_orphan_hits_cap(env):
    subprocess.run(["bash", "-c", "(exec -a 'VLLM::EngineCore FAKE_RECIPE_MARKER' sleep 60) >/dev/null 2>&1 &"], check=True)
    r = _run(env["env"], DRY_RUN="1", GPU_WAIT_MAX_MINUTES="0.05")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "GPU still held" in log and "VLLM::EngineCore FAKE_RECIPE_MARKER" in log


def test_empty_pin_vars_fall_back_to_defaults(env):
    # empty string must NOT disable a pin; defaults (eff5f26) apply -> fake repo fails
    r = _run(env["env"], DRY_RUN="1", EXPECT_RUNNER_SHA256="", PIN_COMMIT="")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "DISABLED" not in log and "STOP: pinned commit eff5f26 is not an ancestor" in log


def test_blank_trailing_line_is_tolerated(env):
    p = env["full"] / "ministral_recipe_atcoder_think_t0.7.jsonl"
    p.write_text(p.read_text() + "\n")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "atcoder: batch0_expected=121 (sanity 121) distinct=121 parsed_lines=121 unparseable=0" in _qlog(env["repo"])


def test_nonblank_garbage_line_still_stops(env):
    p = env["full"] / "ministral_recipe_atcoder_think_t0.7.jsonl"
    p.write_text(p.read_text() + "not json\n")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 1 and "unparseable=1" in _qlog(env["repo"])


def test_pull_bringing_runner_drift_stops_at_sha_pin(env):
    other = env["tmp"] / "other"
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, str(env["origin"]), str(other)], check=True)
    _git(other, "config", "user.email", "t@t"); _git(other, "config", "user.name", "t")
    rp = other / "scripts/run_pless_variants_ministral_3b_lcb_cuda.py"
    rp.write_text(rp.read_text() + "# drift\n"); _git(other, "commit", "-qam", "drift")
    _git(other, "push", "-q", "origin", BRANCH)
    r = _run(env["env"], PULL="1")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "sha256 mismatch" in log and "smoke (VLLM" not in log


def test_wait_pid_replaced_midwait_stops_waiting_on_it(env):
    # matches the recipe pattern for 2 s, then exec()s into an unrelated 'sleep 40'
    spid = subprocess.run(
        ["bash", "-c",
         "(exec -a 'python run_pless_variants_ministral_3b_lcb_cuda.py --configs ministral_recipe FAKE_RECIPE_MARKER' "
         "bash -c 'sleep 2; exec sleep 40') >/dev/null 2>&1 & echo $!"],
        capture_output=True, text=True, check=True).stdout.strip()
    try:
        t0 = time.time()
        r = _run(env["env"], DRY_RUN="1", WAIT_PID=spid)
        took = time.time() - t0
    finally:
        os.kill(int(spid), signal.SIGTERM)
    assert r.returncode == 0, _qlog(env["repo"])
    assert took < 30  # did not wait for the replaced process
    log = _qlog(env["repo"])
    assert "WAIT_PID" in log and "no longer matches" in log


def test_bench_diff_outside_eval_stops(env):
    _commit_change(env["repo"], "bench/other.py", "# changed\n")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 1
    log = _qlog(env["repo"])
    assert "STOP: HEAD differs from pinned" in log and "bench/other.py" in log


def test_bench_eval_diff_is_allowed(env):
    _commit_change(env["repo"], "bench/eval/metrics.py", "# changed\n")
    r = _run(env["env"], DRY_RUN="1")
    assert r.returncode == 0, _qlog(env["repo"])
    assert "pin: no diff vs" in _qlog(env["repo"])


def test_pin_diff_any_disables_loudly(env):
    _commit_change(env["repo"], "bench/other.py", "# changed\n")
    r = _run(env["env"], DRY_RUN="1", PIN_DIFF="any")
    assert r.returncode == 0
    assert "WARNING: PIN_DIFF=any" in _qlog(env["repo"])
    r = _run(env["env"], DRY_RUN="1", PIN_DIFF="ANY")  # only exact 'any' disables
    assert r.returncode == 1


def test_environ_fallback_written_when_no_proc(env):
    r = _run(env["env"])
    assert r.returncode == 0, _qlog(env["repo"])
    ldir = next(env["repo"].glob("logs/topk_launch_*"))
    txt = (ldir / "environ.txt").read_text()
    if Path("/proc/self/environ").exists():
        assert "VLLM_USE_FLASHINFER_SAMPLER=1" in txt and "captured /proc/" in _qlog(env["repo"])
    else:
        assert "fallback" in txt and "VLLM_USE_FLASHINFER_SAMPLER=1" in txt


def test_git_pull_hang_hits_timeout(env):
    g = env["tmp"] / "stubs/git"
    g.write_text('#!/usr/bin/env bash\n'
                 'for a in "$@"; do [[ "$a" == pull ]] && exec sleep 100; done\n'
                 'exec git "$@"\n')
    g.chmod(0o755)
    t0 = time.time()
    r = _run(env["env"], PULL="1", GIT_BIN=str(g), GIT_PULL_TIMEOUT_SECONDS="1")
    assert r.returncode == 1 and time.time() - t0 < 60
    assert "STOP: git pull --ff-only origin" in _qlog(env["repo"])
