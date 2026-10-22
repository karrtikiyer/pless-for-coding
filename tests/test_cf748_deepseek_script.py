"""Behavioural tests for run_cf748_apps_deepseek.sh (MULTI_ARM opt-in + single-arm default).

The script is run against a fake venv python and a fake `uv` that only record their argv (and the GPU
pin), so no GPU/vLLM is needed. The key hazard under test: multi_arm_runner's preset `rec` hardcodes
top_k=20 (Qwen), but DeepSeek's recommended config has NO top-k, so `rec` must be passed via --arm-raw,
never via --arms. The host environment is scrubbed of the script's own variables so a stray `ARMS` /
`MODEL` in the developer's shell cannot change what is being tested.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "run_cf748_apps_deepseek.sh"
DEFAULT_ARMS = ("pless pless_norm renyi_k1.6 renyi_k0.8 renyi_k0.4 renyi_k0.2 renyi_k0.1 renyi_k0.05 "
                "rec topp0.7 topp0.75 topp0.8 topp0.85 topp0.9 topp0.95 temp").split()
RAW_REC = "name=rec,method=temp,temperature=0.6,top_p=0.95"
# Every variable the script reads or exports; removed from the inherited environment before each run.
SCRIPT_VARS = ("ARMS", "MULTI_ARM", "MAX_PROBLEMS", "MODEL", "SOURCE", "DIFFICULTY", "RESULTS_DIR",
               "TASK_IDS_FILE", "N_SAMPLES", "MAX_TOKENS", "VLLM_VENV", "WORKERS", "TOKENIZER", "GPUS",
               "CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_V2_MODEL_RUNNER")


def _exe(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_full(tmp_path: Path, **env):
    """Run the script with fake python/uv. Returns (python invocations as argv lists, CompletedProcess,
    CUDA_VISIBLE_DEVICES seen by each invocation)."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    log, env_log = tmp_path / "calls.log", tmp_path / "env.log"
    _exe(venv_bin / "python",
         'if [ "$1" = "-c" ]; then exit 0; fi\n'
         'for a in "$@"; do printf "%s\\n" "$a" >> "$FAKE_LOG"; done\n'
         'printf -- "---\\n" >> "$FAKE_LOG"\n'
         'printf "%s\\n" "${CUDA_VISIBLE_DEVICES-<unset>}" >> "$FAKE_ENV_LOG"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "uv", 'echo "uv $*" >> "$FAKE_UV_LOG"\n')
    ids = tmp_path / "ids.txt"
    ids.write_text("1 2\n")
    full_env = {k: v for k, v in os.environ.items() if k not in SCRIPT_VARS}
    full_env.update({
        "VLLM_VENV": str(tmp_path / "venv"), "RESULTS_DIR": str(tmp_path / "res"), "TASK_IDS_FILE": str(ids),
        "FAKE_LOG": str(log), "FAKE_ENV_LOG": str(env_log), "FAKE_UV_LOG": str(tmp_path / "uv.log"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "HF_HOME": str(tmp_path / "hf"), **env,
    })
    r = subprocess.run(["bash", str(SCRIPT)], cwd=REPO_ROOT, env=full_env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    blocks, cur = [], []
    if log.exists():
        for line in log.read_text().splitlines():
            if line == "---":
                blocks.append(cur)
                cur = []
            else:
                cur.append(line)
    cvd = env_log.read_text().splitlines() if env_log.exists() else []
    return blocks, r, cvd


def _run(tmp_path: Path, **env) -> list[list[str]]:
    return _run_full(tmp_path, **env)[0]


def _flag_values(argv: list[str], flag: str) -> list[str]:
    """Values following `flag` up to the next token starting with '--' (for nargs='+' flags)."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                out.append(argv[j])
                j += 1
            i = j
        else:
            i += 1
    return out


def test_multi_arm_default_sends_rec_via_arm_raw_not_arms(tmp_path):
    calls = _run(tmp_path, MULTI_ARM="1")
    assert len(calls) == 1, "multi-arm must launch exactly ONE python process"
    argv = calls[0]
    assert argv[:2] == ["-m", "bench.apps.multi_arm_runner"]
    assert _flag_values(argv, "--arms") == [a for a in DEFAULT_ARMS if a != "rec"]
    assert "rec" not in _flag_values(argv, "--arms")
    raw = _flag_values(argv, "--arm-raw")
    assert raw == [RAW_REC]
    assert not any("top_k" in v for v in raw)


def test_multi_arm_passes_deepseek_settings_results_dir_and_no_resume_flag(tmp_path):
    argv = _run(tmp_path, MULTI_ARM="1")[0]
    assert argv[:2] == ["-m", "bench.apps.multi_arm_runner"]
    assert _flag_values(argv, "--model") == ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    assert _flag_values(argv, "--n-samples") == ["10"]
    assert _flag_values(argv, "--max-new-tokens") == ["32768"]
    assert _flag_values(argv, "--source") == ["CODEFORCES"] and _flag_values(argv, "--difficulty") == ["interview"]
    assert _flag_values(argv, "--results-dir") == [str(tmp_path / "res")]
    assert "--enable-thinking" in argv
    assert "--no-resume" not in argv


def test_multi_arm_only_rec_has_no_arms_flag(tmp_path):
    argv = _run(tmp_path, MULTI_ARM="1", ARMS="rec")[0]
    assert "--arms" not in argv
    assert _flag_values(argv, "--arm-raw") == [RAW_REC]


def test_multi_arm_without_rec_has_no_arm_raw(tmp_path):
    argv = _run(tmp_path, MULTI_ARM="1", ARMS="pless renyi_k0.4")[0]
    assert _flag_values(argv, "--arms") == ["pless", "renyi_k0.4"]
    assert "--arm-raw" not in argv


def test_multi_arm_pins_first_gpu_and_reports_ignored_gpus(tmp_path):
    calls, proc, cvd = _run_full(tmp_path, MULTI_ARM="1", GPUS="0,1")
    assert len(calls) == 1
    assert cvd == ["0"], f"engine process must see CUDA_VISIBLE_DEVICES=0, saw {cvd}"
    assert "using GPU 0, ignoring the rest: 1" in proc.stderr


def test_multi_arm_single_gpu_and_no_gpus_set(tmp_path):
    _, proc, cvd = _run_full(tmp_path / "one", MULTI_ARM="1", GPUS="3")
    assert cvd == ["3"] and "ignoring the rest" not in proc.stderr
    _, _, cvd = _run_full(tmp_path / "none", MULTI_ARM="1")
    assert cvd == ["<unset>"]


def test_default_path_is_still_single_arm_loop_with_rec_without_top_k(tmp_path):
    calls = _run(tmp_path, ARMS="pless rec")
    assert [c[:2] for c in calls] == [["-m", "bench.apps"]] * 2
    rec = calls[1]
    assert _flag_values(rec, "--top-p") == ["0.95"] and _flag_values(rec, "--temperature") == ["0.6"]
    assert "--top-k" not in rec


def test_rec_settings_identical_in_single_and_multi_arm_paths(tmp_path):
    """`rec` is defined twice in the script (arm_args for single-arm, --arm-raw for multi-arm); they must agree."""
    single = _run(tmp_path / "single", ARMS="rec")[0]
    multi = _run(tmp_path / "multi", MULTI_ARM="1", ARMS="rec")[0]
    raw = dict(kv.split("=", 1) for kv in _flag_values(multi, "--arm-raw")[0].split(","))
    assert raw["method"] == _flag_values(single, "--method")[0]
    assert float(raw["temperature"]) == float(_flag_values(single, "--temperature")[0])
    assert float(raw["top_p"]) == float(_flag_values(single, "--top-p")[0])
    assert "top_k" not in raw and "--top-k" not in single


def test_arm_raw_rec_filename_matches_deepseek_table_expectation():
    from argparse import Namespace
    from bench.apps.multi_arm_runner import _arm_output_path, _parse_arm_raw, _preset_arm_spec
    base = Namespace(results_dir="r", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", source="CODEFORCES",
                     difficulty="interview", enable_thinking=True)
    spec = _parse_arm_raw(RAW_REC)
    assert spec["top_k"] == 0
    assert _arm_output_path(spec, base).name == "temp_p0.95_think_t0.6_t0.6.jsonl"
    # documents the trap this avoids: the Qwen-oriented preset would write a different (k20) file
    assert _arm_output_path(_preset_arm_spec("rec"), base).name == "temp_p0.95_k20_think_t0.6_t0.6.jsonl"


def test_stray_host_variables_do_not_leak_into_the_script(tmp_path, monkeypatch):
    for k, v in (("ARMS", "pless"), ("MODEL", "other/model"), ("N_SAMPLES", "99"), ("MULTI_ARM", "0")):
        monkeypatch.setenv(k, v)
    argv = _run(tmp_path, MULTI_ARM="1")[0]
    assert _flag_values(argv, "--arms") == [a for a in DEFAULT_ARMS if a != "rec"]
    assert _flag_values(argv, "--model") == ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    assert _flag_values(argv, "--n-samples") == ["10"]


def test_header_no_longer_claims_22_arms():
    head = SCRIPT.read_text().split("set -euo pipefail")[0]
    assert "22 arms" not in head
    assert "16" in head and "MULTI_ARM" in head
