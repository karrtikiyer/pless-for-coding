"""Behavioural tests for run_cf748_apps_qwen3.sh's opt-in SKIP_EVAL=1 (generation only).

Fake venv python + fake `uv` record their argv, so no GPU/vLLM is needed. SKIP_EVAL=1 must run generation
and skip the trailing `uv run ... bench.eval` / `cot_efficiency` steps (which must not run on gpu2, and which
would eval another arm's still-growing file when two single-GPU invocations share one output dir). The
default (unset) must still run them.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "run_cf748_apps_qwen3.sh"
SCRIPT_VARS = ("ARMS", "MULTI_ARM", "MAX_PROBLEMS", "MODEL", "SOURCE", "DIFFICULTY", "RESULTS_DIR",
               "TASK_IDS_FILE", "N_SAMPLES", "MAX_TOKENS", "VLLM_VENV", "WORKERS", "TOKENIZER", "GPUS",
               "CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_V2_MODEL_RUNNER", "SKIP_EVAL")


def _exe(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path: Path, **env):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py_log, uv_log = tmp_path / "py.log", tmp_path / "uv.log"
    _exe(venv_bin / "python", 'if [ "$1" = "-c" ]; then exit 0; fi\necho "$*" >> "$FAKE_PY_LOG"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "uv", 'echo "uv $*" >> "$FAKE_UV_LOG"\n')
    ids = tmp_path / "ids.txt"
    ids.write_text("1 2\n")
    out_dir = tmp_path / "res" / "Qwen--Qwen3-8B" / "CODEFORCES_interview"
    out_dir.mkdir(parents=True)
    (out_dir / "pless_renyi_think_t1.0_k0.25_t1.0.jsonl").write_text("{}\n")  # something for eval to find
    full_env = {k: v for k, v in os.environ.items() if k not in SCRIPT_VARS}
    full_env.update({
        "VLLM_VENV": str(tmp_path / "venv"), "RESULTS_DIR": str(tmp_path / "res"), "TASK_IDS_FILE": str(ids),
        "FAKE_PY_LOG": str(py_log), "FAKE_UV_LOG": str(uv_log), "ARMS": "renyi_k0.25",
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "HF_HOME": str(tmp_path / "hf"), **env,
    })
    r = subprocess.run(["bash", str(SCRIPT)], cwd=REPO_ROOT, env=full_env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    py = py_log.read_text().splitlines() if py_log.exists() else []
    uv = uv_log.read_text().splitlines() if uv_log.exists() else []
    return py, uv, r


def test_default_runs_generation_then_eval(tmp_path):
    py, uv, _ = _run(tmp_path)
    assert len(py) == 1 and "-m bench.apps" in py[0] and "--renyi-k 0.25" in py[0]
    assert any("bench.eval " in u or u.endswith("bench.eval") or "-m bench.eval --results-file" in u for u in uv)
    assert any("bench.eval.cot_efficiency" in u for u in uv)


def test_skip_eval_runs_generation_only(tmp_path):
    py, uv, r = _run(tmp_path, SKIP_EVAL="1")
    assert len(py) == 1 and "-m bench.apps" in py[0] and "--renyi-k 0.25" in py[0]
    assert uv == [], f"no uv eval/cot_efficiency calls expected, got {uv}"
    assert "SKIP_EVAL=1" in r.stdout


def test_skip_eval_with_gpus_pins_and_skips(tmp_path):
    py, uv, _ = _run(tmp_path, SKIP_EVAL="1", GPUS="1")
    assert len(py) == 1 and "-m bench.apps" in py[0]
    assert uv == []
