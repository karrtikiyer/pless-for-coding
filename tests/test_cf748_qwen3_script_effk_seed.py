"""run_cf748_apps_qwen3.sh: `pless_effk_nofloor` arm + opt-in SEED passthrough.

Fake venv python records its argv (no GPU/vLLM). Golden argv lines for the 16 default arms and the
MULTI_ARM invocation were captured from the script BEFORE this change (fake-python dry capture,
2026-11-16); with SEED unset they must be reproduced byte-for-byte.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "run_cf748_apps_qwen3.sh"
SCRIPT_VARS = ("ARMS", "MULTI_ARM", "MAX_PROBLEMS", "MODEL", "SOURCE", "DIFFICULTY", "RESULTS_DIR",
               "TASK_IDS_FILE", "N_SAMPLES", "MAX_TOKENS", "VLLM_VENV", "WORKERS", "TOKENIZER", "GPUS",
               "CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_V2_MODEL_RUNNER", "SKIP_EVAL",
               "MAX_MODEL_LEN", "GPU_MEM_UTIL", "SEED")

PRE = ("-m bench.apps --model Qwen/Qwen3-8B --backend vllm --source CODEFORCES --difficulty interview "
       "--task-ids-file IDS --enable-thinking --n-samples 10 --max-new-tokens 32768 --results-dir RES ")
GOLDEN_DEFAULT_ARMS = [PRE + s for s in (
    "--method pless --temperature 1.0",
    "--method pless_norm --temperature 1.0",
    "--method pless_renyi --renyi-k 1.6 --temperature 1.0",
    "--method pless_renyi --renyi-k 0.8 --temperature 1.0",
    "--method pless_renyi --renyi-k 0.4 --temperature 1.0",
    "--method pless_renyi --renyi-k 0.2 --temperature 1.0",
    "--method pless_renyi --renyi-k 0.1 --temperature 1.0",
    "--method pless_renyi --renyi-k 0.05 --temperature 1.0",
    "--method temp --temperature 0.6 --top-p 0.95 --top-k 20",
    "--method temp --top-p 0.7 --top-k 0 --temperature 1.0",
    "--method temp --top-p 0.75 --top-k 0 --temperature 1.0",
    "--method temp --top-p 0.8 --top-k 0 --temperature 1.0",
    "--method temp --top-p 0.85 --top-k 0 --temperature 1.0",
    "--method temp --top-p 0.9 --top-k 0 --temperature 1.0",
    "--method temp --top-p 0.95 --top-k 0 --temperature 1.0",
    "--method temp --temperature 1.0",
)]
GOLDEN_MULTI_ARM = (
    "-m bench.apps.multi_arm_runner --model Qwen/Qwen3-8B --backend vllm --source CODEFORCES "
    "--difficulty interview --task-ids-file IDS --enable-thinking --n-samples 10 --max-new-tokens 32768 "
    "--results-dir RES --arms pless pless_norm renyi_k1.6 renyi_k0.8 renyi_k0.4 renyi_k0.2 renyi_k0.1 "
    "renyi_k0.05 rec topp0.7 topp0.75 topp0.8 topp0.85 topp0.9 topp0.95 temp")
EFFK = PRE + "--method pless_effk_nofloor --temperature 1.0"


def _exe(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path: Path, ok: bool = True, **env):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py_log = tmp_path / "py.log"
    _exe(venv_bin / "python", 'if [ "$1" = "-c" ]; then exit 0; fi\necho "$*" >> "$FAKE_PY_LOG"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "uv", "exit 0\n")
    ids = tmp_path / "ids.txt"
    ids.write_text("1 2\n")
    res = tmp_path / "res"
    full_env = {k: v for k, v in os.environ.items() if k not in SCRIPT_VARS}
    full_env.update({"VLLM_VENV": str(tmp_path / "venv"), "RESULTS_DIR": str(res), "TASK_IDS_FILE": str(ids),
                     "FAKE_PY_LOG": str(py_log), "SKIP_EVAL": "1", "PATH": f"{bin_dir}:{os.environ['PATH']}",
                     "HF_HOME": str(tmp_path / "hf"), **env})
    r = subprocess.run(["bash", str(SCRIPT)], cwd=REPO_ROOT, env=full_env, capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    lines = py_log.read_text().splitlines() if py_log.exists() else []
    return [l.replace(str(ids), "IDS").replace(str(res), "RES") for l in lines], r


def test_default_arms_byte_identical_when_seed_unset(tmp_path):
    lines, _ = _run(tmp_path)
    assert lines == GOLDEN_DEFAULT_ARMS


def test_multi_arm_byte_identical_when_seed_unset(tmp_path):
    lines, _ = _run(tmp_path, MULTI_ARM="1")
    assert lines == [GOLDEN_MULTI_ARM]


def test_empty_seed_is_unset(tmp_path):
    lines, _ = _run(tmp_path, SEED="")
    assert lines == GOLDEN_DEFAULT_ARMS


def test_effk_nofloor_arm(tmp_path):
    lines, _ = _run(tmp_path, ARMS="pless_effk_nofloor")
    assert lines == [EFFK]


def test_seed_appended_to_every_arm_and_nothing_else_changes(tmp_path):
    lines, _ = _run(tmp_path, SEED="42")
    assert len(lines) == len(GOLDEN_DEFAULT_ARMS)
    for got, want in zip(lines, GOLDEN_DEFAULT_ARMS):
        assert got == want + " --seed 42"


def test_seed_with_effk_arm(tmp_path):
    lines, _ = _run(tmp_path, SEED="42", ARMS="pless_effk_nofloor")
    assert lines == [EFFK + " --seed 42"]


def test_seed_with_gpus_single_gpu(tmp_path):
    lines, _ = _run(tmp_path, SEED="42", ARMS="pless_effk_nofloor", GPUS="1")
    assert lines == [EFFK + " --seed 42"]


def test_seed_reaches_multi_arm(tmp_path):
    lines, _ = _run(tmp_path, SEED="42", MULTI_ARM="1")
    assert lines == [GOLDEN_MULTI_ARM.replace(" --arms ", " --seed 42 --arms ")]


@pytest.mark.parametrize("bad", ["abc", "4.2", "-1", "42 43"])
def test_non_integer_seed_rejected_before_generation(tmp_path, bad):
    lines, r = _run(tmp_path, ok=False, SEED=bad, ARMS="pless_effk_nofloor")
    assert r.returncode == 6, r.stdout + r.stderr
    assert "SEED" in r.stderr
    assert lines == []
