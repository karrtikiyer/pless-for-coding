"""Commands issued by the 2026-11 MBPP staging scripts (DRY_RUN=1; nothing is generated, no uv env check).

run_bigcode_mbpp_qwen25coder7b.sh / run_chat_mbpp_qwen25coder7binstruct.sh + scripts/mbpp_staging_common.sh.
The default command lines are locked to the ones issued by commit ce09f99 (the runs on GPU2); EXTRA_PLESS_TEMPS=1
runs only p-less / p-less-norm at T=1.0 and T=0.6 with the T=0.7 configs' other flags.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = "run_bigcode_mbpp_qwen25coder7b.sh"
CHAT = "run_chat_mbpp_qwen25coder7binstruct.sh"
BASE_ROOT = "results/mbpp_qwen25coder7b_2026_11"
CHAT_ROOT = "results/mbpp_qwen25coder7binstruct_2026_11"
B = "uv run python -m bench --model Qwen/Qwen2.5-Coder-7B"
C = "uv run python -m bench --model Qwen/Qwen2.5-Coder-7B-Instruct"
BT = f"--prompt-style bigcode --mbpp-config full --results-dir {BASE_ROOT}"
CT = f"--mbpp-config full --results-dir {CHAT_ROOT}"

# As printed by `DRY_RUN=1 ./<script>` at ce09f99 (captured before any change).
BASE_DEFAULT = [
    ("greedy_bigcode_t1.0", f"{B} --method greedy --temperature 1.0 --n-samples 1 {BT}"),
    ("beam4_bigcode_t1.0", f"{B} --method beam --num-beams 4 --temperature 1.0 --n-samples 1 {BT}"),
    ("temp_bigcode_t0.7", f"{B} --method temp --temperature 0.7 {BT}"),
    ("top_p0.9_bigcode_t1.0", f"{B} --method top_p --top-p 0.9 --temperature 1.0 {BT}"),
    ("pless_bigcode_t0.7", f"{B} --method pless --temperature 0.7 {BT}"),
    ("pless_norm_bigcode_t0.7", f"{B} --method pless_norm --temperature 0.7 {BT}"),
    ("top_p0.95_bigcode_t0.2", f"{B} --method top_p --top-p 0.95 --temperature 0.2 {BT}"),
]
CHAT_DEFAULT = [
    ("greedy_t1.0", f"{C} --method greedy --temperature 1.0 --n-samples 1 {CT}"),
    ("beam4_t1.0", f"{C} --method beam --num-beams 4 --temperature 1.0 --n-samples 1 {CT}"),
    ("temp_t0.7", f"{C} --method temp --temperature 0.7 {CT}"),
    ("top_p0.9_t1.0", f"{C} --method top_p --top-p 0.9 --temperature 1.0 {CT}"),
    ("pless_t0.7", f"{C} --method pless --temperature 0.7 {CT}"),
    ("pless_norm_t0.7", f"{C} --method pless_norm --temperature 0.7 {CT}"),
]
EXTRA_ORDER = [("pless", "1.0"), ("pless_norm", "1.0"), ("pless", "0.6"), ("pless_norm", "0.6")]
BASE_EXTRA = [(f"{m}_bigcode_t{t}", f"{B} --method {m} --temperature {t} {BT}") for m, t in EXTRA_ORDER]
CHAT_EXTRA = [(f"{m}_t{t}", f"{C} --method {m} --temperature {t} {CT}") for m, t in EXTRA_ORDER]


def _dry_run(tmp_path, script, env_extra=None):
    """Run a copy of the script (and the shared helper) in tmp_path with DRY_RUN=1; returns (rc, [(stem, cmd)], out)."""
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / script, work / script)
    shutil.copy(REPO / "scripts" / "mbpp_staging_common.sh", work / "scripts" / "mbpp_staging_common.sh")
    env = {k: v for k, v in os.environ.items()
           if k not in ("EXTRA_PLESS_TEMPS", "MAX_PROBLEMS", "DRY_RUN", "SEED", "SINGLE_BOS", "RESULTS_DIR")}
    env.update({"DRY_RUN": "1"}, **(env_extra or {}))
    p = subprocess.run(["bash", str(work / script)], capture_output=True, text=True, env=env, cwd=work)
    out = p.stdout + p.stderr
    stems = re.findall(r"^--- \[[^]]*\] (\S+) -> (\S+)$", p.stdout, re.M)
    cmds = re.findall(r"^    CMD: (.*)$", p.stdout, re.M)
    assert len(stems) == len(cmds), out
    for (stem, path), _ in zip(stems, cmds):   # the announced output path is <root>/<model dir>/<stem>.jsonl
        assert path.endswith(f"/{stem}.jsonl"), path
    return p.returncode, [(s, c) for (s, _), c in zip(stems, cmds)], out


@pytest.mark.parametrize("script,expected", [(BASE, BASE_DEFAULT), (CHAT, CHAT_DEFAULT)])
@pytest.mark.parametrize("env", [None, {"EXTRA_PLESS_TEMPS": "0"}, {"EXTRA_PLESS_TEMPS": ""}, {"SEED": ""},
                                 {"SINGLE_BOS": "0"}, {"SINGLE_BOS": ""}, {"RESULTS_DIR": ""}])
def test_default_commands_are_unchanged(tmp_path, script, expected, env):
    rc, cmds, out = _dry_run(tmp_path, script, env)
    assert rc == 0, out
    assert cmds == expected
    assert f"all {len(expected)} configs done (DRY RUN)" in out
    assert "EXTRA_PLESS_TEMPS" not in out   # default output carries no trace of the new options
    assert "SEED" not in out and "SINGLE_BOS" not in out and "--seed" not in out and "--single-bos" not in out


@pytest.mark.parametrize("script,expected", [(BASE, BASE_EXTRA), (CHAT, CHAT_EXTRA)])
def test_extra_pless_temps_runs_only_the_four_extras(tmp_path, script, expected):
    rc, cmds, out = _dry_run(tmp_path, script, {"EXTRA_PLESS_TEMPS": "1"})
    assert rc == 0, out
    assert cmds == expected
    assert "all 4 configs done (DRY RUN)" in out


@pytest.mark.parametrize("script,default,extra", [(BASE, BASE_DEFAULT, BASE_EXTRA), (CHAT, CHAT_DEFAULT, CHAT_EXTRA)])
def test_extras_equal_the_t07_pless_commands_except_temperature(script, default, extra):
    ref = dict(default)
    for stem, cmd in extra:
        method, t = re.search(r"--method (\S+) --temperature (\S+)", cmd).groups()
        src = ref[stem.rsplit("_t", 1)[0] + "_t0.7"]
        assert cmd == src.replace("--temperature 0.7", f"--temperature {t}"), stem
        assert method in ("pless", "pless_norm") and t in ("1.0", "0.6")


@pytest.mark.parametrize("script,model,extra", [(BASE, "Qwen/Qwen2.5-Coder-7B", BASE_EXTRA),
                                                (CHAT, "Qwen/Qwen2.5-Coder-7B-Instruct", CHAT_EXTRA)])
def test_extra_stems_equal_the_runner_file_names(tmp_path, script, model, extra):
    # Same derivation as bench/runner.py main(): method key, "_bigcode" for a base model with --prompt-style bigcode,
    # then checkpointing.get_output_path(<results>, model, key, float(--temperature)).
    from bench.checkpointing import get_output_path
    from bench.prompts import is_instruct_model
    for stem, cmd in extra:
        method, t = re.search(r"--method (\S+) --temperature (\S+)", cmd).groups()
        key = method + ("_bigcode" if "--prompt-style bigcode" in cmd and not is_instruct_model(model) else "")
        assert get_output_path(str(tmp_path), model, key, float(t)).name == f"{stem}.jsonl"


def test_base_extra_names_equal_the_existing_3b_files():
    old = REPO / "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B"
    if not old.is_dir():
        pytest.skip("3B MBPP results not present on this machine")
    for stem, _ in BASE_EXTRA:
        assert (old / f"{stem}.jsonl").is_file(), stem


def test_extra_pless_temps_with_max_problems(tmp_path):
    rc, cmds, out = _dry_run(tmp_path, BASE, {"EXTRA_PLESS_TEMPS": "1", "MAX_PROBLEMS": "3"})
    assert rc == 0, out
    assert [c for _, c in cmds] == [c + " --max-problems 3" for _, c in BASE_EXTRA]


@pytest.mark.parametrize("value", ["2", "yes", "true", " 1"])
def test_extra_pless_temps_rejects_other_values(tmp_path, value):
    rc, cmds, out = _dry_run(tmp_path, BASE, {"EXTRA_PLESS_TEMPS": value})
    assert rc == 2 and cmds == [], out


# ---- SEED / SINGLE_BOS passthroughs (2026-11-08): append --seed N / --single-bos to every config's command line
@pytest.mark.parametrize("script,default", [(BASE, BASE_DEFAULT), (CHAT, CHAT_DEFAULT)])
@pytest.mark.parametrize("env,suffix", [({"SEED": "1234"}, " --seed 1234"), ({"SEED": "0"}, " --seed 0"),
                                        ({"SINGLE_BOS": "1"}, " --single-bos"),
                                        ({"SEED": "7", "SINGLE_BOS": "1"}, " --seed 7 --single-bos")])
def test_seed_and_single_bos_are_appended(tmp_path, script, default, env, suffix):
    rc, cmds, out = _dry_run(tmp_path, script, env)
    assert rc == 0, out
    assert cmds == [(s, c + suffix) for s, c in default]   # same stems: the runner does not rename files


def test_seed_single_bos_with_extras_and_max_problems(tmp_path):
    rc, cmds, out = _dry_run(tmp_path, BASE, {"EXTRA_PLESS_TEMPS": "1", "MAX_PROBLEMS": "3", "SEED": "5",
                                              "SINGLE_BOS": "1"})
    assert rc == 0, out
    assert [c for _, c in cmds] == [c + " --seed 5 --single-bos --max-problems 3" for _, c in BASE_EXTRA]


@pytest.mark.parametrize("env", [{"SEED": "-1"}, {"SEED": "abc"}, {"SEED": "1.5"}, {"SEED": " 1"}, {"SEED": "01"},
                                 {"SINGLE_BOS": "2"}, {"SINGLE_BOS": "yes"}, {"SINGLE_BOS": "true"}])
def test_seed_single_bos_reject_other_values(tmp_path, env):
    rc, cmds, out = _dry_run(tmp_path, BASE, env)
    assert rc == 2 and cmds == [], out


# ---- run_bos_check_mbpp_chat.sh MODEL OUT_ROOT: MBPP-full greedy, default (double BOS) vs --single-bos
BOS = "run_bos_check_mbpp_chat.sh"


def _bos_dry_run(tmp_path, args, env_extra=None):
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / BOS, work / BOS)
    shutil.copy(REPO / "scripts" / "mbpp_staging_common.sh", work / "scripts" / "mbpp_staging_common.sh")
    env = {k: v for k, v in os.environ.items()
           if k not in ("EXTRA_PLESS_TEMPS", "MAX_PROBLEMS", "DRY_RUN", "SEED", "SINGLE_BOS", "RESULTS_DIR")}
    env.update({"DRY_RUN": "1"}, **(env_extra or {}))
    p = subprocess.run(["bash", str(work / BOS), *args], capture_output=True, text=True, env=env, cwd=work)
    stems = re.findall(r"^--- \[[^]]*\] (\S+) -> (\S+)$", p.stdout, re.M)
    cmds = re.findall(r"^    CMD: (.*)$", p.stdout, re.M)
    return p.returncode, stems, cmds, p.stdout + p.stderr


@pytest.mark.parametrize("model", ["codellama/CodeLlama-7b-Instruct-hf", "meta-llama/Llama-2-7b-chat-hf"])
def test_bos_check_commands(tmp_path, model):
    rc, stems, cmds, out = _bos_dry_run(tmp_path, [model, "results/bos_check"])
    assert rc == 0, out
    md = model.replace("/", "--")
    greedy = f"uv run python -m bench --model {model} --method greedy --temperature 1.0 --n-samples 1 --mbpp-config full"
    assert cmds == [f"{greedy} --results-dir results/bos_check/double_bos",
                    f"{greedy} --results-dir results/bos_check/single_bos --single-bos"]
    # the double-BOS arm is the exact CHAT_DEFAULT greedy command line, only the root differs
    assert cmds[0] == dict(CHAT_DEFAULT)["greedy_t1.0"].replace(C, f"uv run python -m bench --model {model}") \
        .replace(CHAT_ROOT, "results/bos_check/double_bos")
    assert stems == [("greedy_t1.0", f"results/bos_check/double_bos/{md}/greedy_t1.0.jsonl"),
                     ("greedy_t1.0", f"results/bos_check/single_bos/{md}/greedy_t1.0.jsonl")]


def test_bos_check_max_problems(tmp_path):
    rc, _, cmds, out = _bos_dry_run(tmp_path, ["codellama/CodeLlama-7b-Instruct-hf", "r"], {"MAX_PROBLEMS": "3"})
    assert rc == 0, out
    assert cmds[0].endswith("--results-dir r/double_bos --max-problems 3")
    assert cmds[1].endswith("--results-dir r/single_bos --single-bos --max-problems 3")


@pytest.mark.parametrize("args", [[], ["codellama/CodeLlama-7b-Instruct-hf"], ["codellama/CodeLlama-7b-hf", "r"],
                                  ["codellama/CodeLlama-7b-Instruct-hf", "r", "extra"]])
def test_bos_check_rejects_bad_args(tmp_path, args):
    rc, _, cmds, out = _bos_dry_run(tmp_path, args)
    assert rc == 2 and cmds == [], out


@pytest.mark.parametrize("env", [{"SINGLE_BOS": "1"}, {"SEED": "1"}, {"EXTRA_PLESS_TEMPS": "1"}])
def test_bos_check_refuses_caller_overrides(tmp_path, env):
    rc, _, cmds, out = _bos_dry_run(tmp_path, ["codellama/CodeLlama-7b-Instruct-hf", "r"], env)
    assert rc == 2 and cmds == [], out


# ---- RESULTS_DIR override (2026-11-08): seeded reruns go to a NEW root, never into the old unseeded files
@pytest.mark.parametrize("script,default,root", [(BASE, BASE_DEFAULT, BASE_ROOT), (CHAT, CHAT_DEFAULT, CHAT_ROOT)])
def test_results_dir_override(tmp_path, script, default, root):
    new = "results/mbpp_seeded_2026_11"
    rc, cmds, out = _dry_run(tmp_path, script, {"RESULTS_DIR": new, "SEED": "1234"})
    assert rc == 0, out
    assert cmds == [(s, c.replace(f"--results-dir {root}", f"--results-dir {new}") + " --seed 1234") for s, c in default]
    assert root not in out          # nothing announced under the old root
    model_dir = ("Qwen/Qwen2.5-Coder-7B" if script == BASE else "Qwen/Qwen2.5-Coder-7B-Instruct").replace("/", "--")
    assert f"-> {new}/{model_dir}/" in out


def test_bos_check_refuses_results_dir_env(tmp_path):
    rc, _, cmds, out = _bos_dry_run(tmp_path, ["codellama/CodeLlama-7b-Instruct-hf", "r"], {"RESULTS_DIR": "x"})
    assert rc == 2 and cmds == [], out
