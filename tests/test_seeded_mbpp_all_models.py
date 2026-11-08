"""run_seeded_mbpp_all_models.sh MODEL_ID [MODEL_ID ...] — seeded MBPP-full regeneration (analysis only, 2026-11-08).

Reuses scripts/mbpp_staging_common.sh (config tables, extra p-less builder, completeness / per-task-error / transformers
checks, logging). DRY_RUN=1 tests lock the exact command lines; the "fake run" tests put a stub `uv` on PATH (no model,
no GPU) to exercise the non-dry-run guarantees: stop on first failure, "Error on task_id" check, 500-row check,
transformers check, log names, PYTORCH_CUDA_ALLOC_CONF export.
"""
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from bench.checkpointing import get_output_path
from bench.prompts import is_instruct_model

REPO = Path(__file__).resolve().parents[1]
SCRIPT = "run_seeded_mbpp_all_models.sh"
ROOT = "results/mbpp_seeded_2026_11"

HASAN_BASE = ["m-a-p/OpenCodeInterpreter-DS-1.3B", "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-3B",
              "Qwen/Qwen2.5-Coder-7B"]
OTHER_BASE = ["codellama/CodeLlama-7b-hf", "meta-llama/Llama-2-7b-hf"]   # 3-shot "paper" prompt (their Table I rows)
PAPER_BASE = OTHER_BASE
BIGCODE_BASE = HASAN_BASE
CHAT = ["codellama/CodeLlama-7b-Instruct-hf", "meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-Coder-7B-Instruct"]
ALL = HASAN_BASE + OTHER_BASE + CHAT

TABLE1 = [("greedy", "_t1.0", "--method greedy --temperature 1.0 --n-samples 1"),
          ("beam4", "_t1.0", "--method beam --num-beams 4 --temperature 1.0 --n-samples 1"),
          ("temp", "_t0.7", "--method temp --temperature 0.7"),
          ("top_p0.9", "_t1.0", "--method top_p --top-p 0.9 --temperature 1.0"),
          ("pless", "_t0.7", "--method pless --temperature 0.7"),
          ("pless_norm", "_t0.7", "--method pless_norm --temperature 0.7")]
HASAN = [("top_p0.95", "_t0.2", "--method top_p --top-p 0.95 --temperature 0.2")]
EXTRA = [(m, f"_t{t}", f"--method {m} --temperature {t}")
         for t in ("1.0", "0.6") for m in ("pless", "pless_norm")]


def expected(model, root=ROOT):
    """[(stem, cmd)] exactly as the staging / April scripts build them, + --seed 42 (+ --single-bos for chat).

    bigcode base (OCI, Qwen2.5-Coder base): --prompt-style bigcode, "_bigcode" stems. 3-shot base (CodeLlama-7b-hf,
    Llama-2-7b-hf): runner default prompt (no --prompt-style, n-shots 3), same flags and stems as their April files
    (run_bench.sh / run_new_samplers_mbpp.sh / run_top_p_mbpp_all_models.sh). Chat: no --prompt-style, + --single-bos.
    """
    chat = model in CHAT
    rows = TABLE1 + (HASAN if model in HASAN_BASE else []) + EXTRA
    bigcode = model in BIGCODE_BASE
    tag = "_bigcode" if bigcode else ""
    prompt = " --prompt-style bigcode" if bigcode else ""
    tail = " --seed 42" + (" --single-bos" if chat else "")
    return [(f"{k}{tag}{t}",
             f"uv run python -m bench --model {model} {a}{prompt} --mbpp-config full --results-dir {root}{tail}")
            for k, t, a in rows]


CLEAR = ("EXTRA_PLESS_TEMPS", "MAX_PROBLEMS", "DRY_RUN", "SEED", "SINGLE_BOS", "RESULTS_DIR", "PYTORCH_CUDA_ALLOC_CONF")


def _work(tmp_path):
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / SCRIPT, work / SCRIPT)
    shutil.copy(REPO / "scripts" / "mbpp_staging_common.sh", work / "scripts" / "mbpp_staging_common.sh")
    return work


def _parse(stdout):
    stems = re.findall(r"^--- \[[^]]*\] (\S+) -> (\S+)$", stdout, re.M)
    cmds = re.findall(r"^    CMD: (.*)$", stdout, re.M)
    assert len(stems) == len(cmds), stdout
    return stems, cmds


def _dry_run(tmp_path, models, env_extra=None):
    work = _work(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update({"DRY_RUN": "1"}, **(env_extra or {}))
    p = subprocess.run(["bash", str(work / SCRIPT), *models], capture_output=True, text=True, env=env, cwd=work)
    stems, cmds = _parse(p.stdout)
    return p.returncode, [(s, c) for (s, _), c in zip(stems, cmds)], stems, p.stdout + p.stderr


# ------------------------------------------------------------------ DRY_RUN: exact command lists
@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-Coder-7B",          # base, Hasan, bigcode
                                   "codellama/CodeLlama-7b-hf",      # base, 3-shot
                                   "meta-llama/Llama-2-7b-hf",       # base, 3-shot
                                   "meta-llama/Llama-2-7b-chat-hf",  # chat, Llama family
                                   "Qwen/Qwen2.5-Coder-7B-Instruct"])  # chat, Qwen
def test_exact_commands_representative(tmp_path, model):
    rc, cmds, stems, out = _dry_run(tmp_path, [model])
    assert rc == 0, out
    assert cmds == expected(model)
    md = model.replace("/", "--")
    assert [p for _, p in stems] == [f"{ROOT}/{md}/{s}.jsonl" for s, _ in cmds]


@pytest.mark.parametrize("model", ALL)
def test_every_model_commands_and_count(tmp_path, model):
    rc, cmds, _, out = _dry_run(tmp_path, [model])
    assert rc == 0, out
    assert cmds == expected(model)
    n = 11 if model in HASAN_BASE else 10
    assert len(cmds) == n
    assert f"all {n} configs done (DRY RUN)" in out


@pytest.mark.parametrize("model", ALL)
def test_seed_single_bos_and_hasan_flags(tmp_path, model):
    _, cmds, _, _ = _dry_run(tmp_path, [model])
    for _, c in cmds:
        assert c.split().count("--seed") == 1 and " --seed 42" in c
        assert ("--single-bos" in c) == (model in CHAT)
        assert ("--prompt-style bigcode" in c) == (model in BIGCODE_BASE)
        assert ("--prompt-style" in c) == (model in BIGCODE_BASE)   # 3-shot + chat: runner default, never passed
        assert "--n-shots" not in c
    hasan = [c for _, c in cmds if "--top-p 0.95" in c]
    assert len(hasan) == (1 if model in HASAN_BASE else 0)
    if hasan:
        assert "--temperature 0.2" in hasan[0]


def test_model_table_agrees_with_is_instruct_model():
    for m in ALL:
        assert is_instruct_model(m) == (m in CHAT), m


@pytest.mark.parametrize("model", ALL)
def test_stems_equal_the_runner_file_names(tmp_path, model):
    """Same derivation as bench/runner.py main() for the methods used here."""
    for stem, cmd in expected(model):
        a = cmd.split()
        val = lambda f: a[a.index(f) + 1]  # noqa: E731
        method = val("--method")
        key = {"top_p": f"top_p{val('--top-p') if '--top-p' in a else ''}",
               "beam": f"beam{val('--num-beams') if '--num-beams' in a else ''}"}.get(method, method)
        if not is_instruct_model(model) and "bigcode" in a:
            key += "_bigcode"
        assert get_output_path(str(tmp_path), model, key, float(val("--temperature"))).name == f"{stem}.jsonl"


def test_several_models_in_given_order(tmp_path):
    models = ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-3B", "codellama/CodeLlama-7b-Instruct-hf"]
    rc, cmds, _, out = _dry_run(tmp_path, models)
    assert rc == 0, out
    assert cmds == [x for m in models for x in expected(m)]


def test_full_gpu0_and_gpu1_queues(tmp_path):
    q0 = ["Qwen/Qwen2.5-Coder-7B", "m-a-p/OpenCodeInterpreter-DS-1.3B", "Qwen/Qwen2.5-Coder-1.5B",
          "codellama/CodeLlama-7b-hf", "meta-llama/Llama-2-7b-hf"]
    q1 = ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-3B", "codellama/CodeLlama-7b-Instruct-hf",
          "meta-llama/Llama-2-7b-chat-hf"]
    for q, n in ((q0, 11 * 3 + 10 * 2), (q1, 10 + 11 + 10 + 10)):
        rc, cmds, _, out = _dry_run(tmp_path, q)
        assert rc == 0, out
        assert len(cmds) == n and cmds == [x for m in q for x in expected(m)]


def test_default_root_and_override(tmp_path):
    rc, cmds, _, out = _dry_run(tmp_path, ["Qwen/Qwen2.5-Coder-7B"], {"RESULTS_DIR": ""})
    assert rc == 0 and all(f"--results-dir {ROOT} " in c for _, c in cmds), out
    rc, cmds, _, out = _dry_run(tmp_path, ["Qwen/Qwen2.5-Coder-7B"], {"RESULTS_DIR": "results/x"})
    assert rc == 0, out
    assert cmds == expected("Qwen/Qwen2.5-Coder-7B", root="results/x")
    assert ROOT not in out


def test_max_problems_passthrough(tmp_path):
    rc, cmds, _, out = _dry_run(tmp_path, ["meta-llama/Llama-2-7b-chat-hf"], {"MAX_PROBLEMS": "3"})
    assert rc == 0, out
    assert [c for _, c in cmds] == [c + " --max-problems 3" for _, c in expected("meta-llama/Llama-2-7b-chat-hf")]


def test_alloc_conf_default_and_override(tmp_path):
    _, _, _, out = _dry_run(tmp_path, ["Qwen/Qwen2.5-Coder-7B"])
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in out
    _, _, _, out = _dry_run(tmp_path, ["Qwen/Qwen2.5-Coder-7B"], {"PYTORCH_CUDA_ALLOC_CONF": "backend:native"})
    assert "PYTORCH_CUDA_ALLOC_CONF=backend:native" in out and "expandable_segments" not in out


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("models", [[], ["Qwen/Qwen-7B"], ["Qwen/Qwen2.5-Coder-3B-Instruct"], ["qwen/qwen2.5-coder-7b"],
                                    ["Qwen/Qwen2.5-Coder-7B", "unknown/model"],      # refused before anything runs
                                    ["Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-7B"]])  # duplicate
def test_refuses_unknown_models(tmp_path, models):
    rc, cmds, _, out = _dry_run(tmp_path, models)
    assert rc == 2 and cmds == [], out


@pytest.mark.parametrize("env", [{"SEED": "1"}, {"SINGLE_BOS": "1"}, {"SINGLE_BOS": "0"}, {"EXTRA_PLESS_TEMPS": "1"}])
def test_refuses_caller_overrides_of_fixed_settings(tmp_path, env):
    rc, cmds, _, out = _dry_run(tmp_path, ["Qwen/Qwen2.5-Coder-7B"], env)
    assert rc == 2 and cmds == [], out


# ------------------------------------------------------------------ fake (non-dry) runs with a stub `uv`
FAKE_BENCH = r'''
import json, os, sys
sys.path.insert(0, os.environ["REPO"])
from bench.checkpointing import get_output_path
from bench.prompts import is_instruct_model
a = sys.argv[1:]
v = lambda f: a[a.index(f) + 1]
m = v("--method")
key = {"top_p": "top_p" + (v("--top-p") if "--top-p" in a else ""),
       "beam": "beam" + (v("--num-beams") if "--num-beams" in a else "")}.get(m, m)
if not is_instruct_model(v("--model")) and "bigcode" in a:
    key += "_bigcode"
out = get_output_path(v("--results-dir"), v("--model"), key, float(v("--temperature")))
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps({"stem": out.stem, "argv": a, "alloc": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
if out.stem == os.environ.get("FAKE_FAIL_STEM"):
    print("Traceback: boom"); sys.exit(1)
if out.stem == os.environ.get("FAKE_ERROR_STEM"):
    print("Error on task_id=7: CUDA out of memory")
n = int(os.environ.get("FAKE_ROWS", v("--max-problems") if "--max-problems" in a else "500"))
with open(out, "w") as f:
    for i in range(n):
        f.write(json.dumps({"task_id": i + 1, "samples": ["x"]}) + "\n")
print(f"generated {n}")
'''

FAKE_UV = """#!/bin/bash
# stub: uv run python <...>
shift  # run
shift  # python
if [[ "$1" == "-m" && "$2" == "bench" ]]; then shift 2; exec "$REAL_PY" "$FAKE_BENCH_PY" "$@"; fi
if [[ "$1" == "-c" && "$2" == *"import sys, transformers"* ]]; then
  echo "transformers $FAKE_TF  torch fake  cuda_available False"; [[ "${FAKE_TF%%.*}" -ge 5 ]]; exit $?
fi
exec "$REAL_PY" "$@"
"""


def _fake_run(tmp_path, models, env_extra=None):
    work = _work(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "uv").write_text(FAKE_UV)
    (bin_ / "uv").chmod((bin_ / "uv").stat().st_mode | stat.S_IEXEC)
    (tmp_path / "fake_bench.py").write_text(FAKE_BENCH)
    calls = tmp_path / "calls.jsonl"
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update({"PATH": f"{bin_}:{env['PATH']}", "REAL_PY": sys.executable, "FAKE_BENCH_PY": str(tmp_path / "fake_bench.py"),
                "REPO": str(REPO), "FAKE_CALLS": str(calls), "FAKE_TF": "5.3.0", "CUDA_VISIBLE_DEVICES": "1"},
               **(env_extra or {}))
    p = subprocess.run(["bash", str(work / SCRIPT), *models], capture_output=True, text=True, env=env, cwd=work)
    import json
    rows = [json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []
    return p.returncode, rows, work, p.stdout + p.stderr


def test_fake_run_two_models_complete(tmp_path):
    models = ["Qwen/Qwen2.5-Coder-7B-Instruct", "m-a-p/OpenCodeInterpreter-DS-1.3B"]
    rc, rows, work, out = _fake_run(tmp_path, models)
    assert rc == 0, out
    exp = [x for m in models for x in expected(m)]
    assert [r["stem"] for r in rows] == [s for s, _ in exp]
    assert [" ".join(["uv", "run", "python", "-m", "bench", *r["argv"]]) for r in rows] == [c for _, c in exp]
    assert all(r["alloc"] == "expandable_segments:True" and r["cuda"] == "1" for r in rows)
    for m in models:
        md = m.replace("/", "--")
        for s, _ in expected(m):
            assert (work / ROOT / md / f"{s}.jsonl").is_file()
            logs = list((work / "logs").glob(f"mbpp_seeded_{md}_{s}_*.log"))
            assert len(logs) == 1, (s, logs)
    assert out.count("500 rows, 500 unique task_ids (OK)") == 21


def test_fake_run_stops_on_first_failing_config(tmp_path):
    models = ["Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-3B"]
    rc, rows, _, out = _fake_run(tmp_path, models, {"FAKE_FAIL_STEM": "temp_bigcode_t0.7"})
    assert rc != 0, out
    assert [r["stem"] for r in rows] == ["greedy_bigcode_t1.0", "beam4_bigcode_t1.0", "temp_bigcode_t0.7"]


def test_fake_run_per_task_error_fails(tmp_path):
    rc, rows, _, out = _fake_run(tmp_path, ["codellama/CodeLlama-7b-hf", "meta-llama/Llama-2-7b-hf"],
                                 {"FAKE_ERROR_STEM": "beam4_t1.0"})
    assert rc == 4, out
    assert "FAIL: per-task errors" in out
    assert [r["stem"] for r in rows] == ["greedy_t1.0", "beam4_t1.0"]


def test_fake_run_incomplete_file_fails(tmp_path):
    rc, rows, _, out = _fake_run(tmp_path, ["codellama/CodeLlama-7b-hf"], {"FAKE_ROWS": "499"})
    assert rc == 5 and len(rows) == 1, out


def test_fake_run_transformers_check(tmp_path):
    rc, rows, _, out = _fake_run(tmp_path, ["codellama/CodeLlama-7b-hf"], {"FAKE_TF": "4.57.1"})
    assert rc == 3 and rows == [], out


def test_fake_run_max_problems_skips_completeness(tmp_path):
    rc, rows, _, out = _fake_run(tmp_path, ["meta-llama/Llama-2-7b-chat-hf"], {"MAX_PROBLEMS": "2"})
    assert rc == 0 and len(rows) == 10, out
    assert all(r["argv"][-2:] == ["--max-problems", "2"] for r in rows)


def test_fake_run_alloc_conf_override(tmp_path):
    rc, rows, _, out = _fake_run(tmp_path, ["meta-llama/Llama-2-7b-chat-hf"],
                                 {"MAX_PROBLEMS": "1", "PYTORCH_CUDA_ALLOC_CONF": ""})
    assert rc == 0 and all(r["alloc"] == "" for r in rows), out


# ------------------------------------------------------------------ --single-bos is a no-op for Qwen chat (runner level)
def test_single_bos_is_noop_for_qwen_chat_runner(monkeypatch, tmp_path):
    from tests.test_mbpp_seed_single_bos import QWEN_CHAT, _run, _stub_generators, _tok
    qwen = _tok(QWEN_CHAT)
    assert qwen.bos_token is None
    prompts = {}
    for arm, extra in (("default", []), ("single", ["--single-bos"])):
        calls = []
        _stub_generators(monkeypatch, calls)
        _run(monkeypatch, tmp_path / arm, ["--model", QWEN_CHAT, "--method", "greedy", "--n-samples", "1", "--seed", "42",
                                           *extra], None, qwen)
        prompts[arm] = [kw["prompt_text"] for _, kw in calls]
    assert len(prompts["default"]) == 3
    assert all(isinstance(p, str) for p in prompts["single"])
    assert prompts["single"] == prompts["default"]
    assert [qwen.encode(p) for p in prompts["single"]] == [qwen.encode(p) for p in prompts["default"]]


# ------------------------------------------------------------------ 3-shot base: same stems / flags as the April files
APRIL = REPO / "results/pless_full_mbpp_results"


@pytest.mark.parametrize("model", PAPER_BASE)
def test_paper_base_stems_and_flags_match_april_files(model):
    d = APRIL / model.replace("/", "--")
    if not d.is_dir():
        pytest.skip("April MBPP results not present on this machine")
    import json
    for stem, cmd in expected(model):
        f = d / f"{stem}.jsonl"
        assert f.is_file(), f
        r = json.loads(f.open().readline())
        a = cmd.split()
        val = lambda k: a[a.index(k) + 1]  # noqa: E731
        assert r["model"] == model and r["method"] == val("--method")
        assert r["temperature"] == float(val("--temperature"))
        assert len(r["samples"]) == (1 if r["method"] in ("greedy", "beam") else 10)
        assert r.get("num_beams") == (int(val("--num-beams")) if "--num-beams" in a else None)
        assert r.get("top_p") == (float(val("--top-p")) if "--top-p" in a else None)


def test_llama2_base_prompt_equals_shi_llama2_input_task11(monkeypatch, tmp_path):
    """CPU only: the prompt the runner hands the generator for MBPP task 11 (Llama-2-7b-hf, 3-shot default) vs the
    first line of Shi et al.'s data/mbpp/Llama2_input.jsonl. Known difference: one trailing space in an example."""
    import json
    import bench.runner as R
    from tests.test_mbpp_seed_single_bos import _stub_generators
    shi_path = Path("/tmp/shi_llama2_input.jsonl")
    if not shi_path.is_file():
        pytest.skip("Shi et al. Llama2_input.jsonl not at /tmp/shi_llama2_input.jsonl")
    shi = json.loads(shi_path.open().readline())["instructions"]
    calls = []
    _stub_generators(monkeypatch, calls)
    monkeypatch.setattr(R, "load_model_and_tokenizer", lambda *a, **k: (None, None))
    # The real MBPP-full test split (the runner's own repo/config/split), read from the local HF cache parquet: this
    # machine's datasets 2.21 cannot parse the cached dataset_info written by a newer datasets release.
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    try:
        pq_path = hf_hub_download("google-research-datasets/mbpp", "full/test-00000-of-00001.parquet",
                                  repo_type="dataset", local_files_only=True)
    except Exception as e:  # pragma: no cover - machine without the cache
        pytest.skip(f"MBPP parquet not cached: {e}")
    rows = pq.read_table(pq_path).to_pylist()
    assert len(rows) == 500 and rows[0]["task_id"] == 11
    seen = []
    monkeypatch.setattr(R, "load_dataset", lambda *a, **k: (seen.append((a, k)), Dataset.from_list(rows))[1])
    _, _, cmd = expected("meta-llama/Llama-2-7b-hf")[0][1].partition("-m bench ")
    argv = cmd.split()
    argv[argv.index("--results-dir") + 1] = str(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bench", *argv, "--task-ids", "11"])
    R.main()
    assert seen == [(("google-research-datasets/mbpp", "full"), {"split": "test"})]
    assert len(calls) == 1
    ours = calls[0][1]["prompt_text"]
    assert isinstance(ours, str)
    assert ours != shi                                    # not byte-identical ...
    assert ours.replace("    return (res) \n", "    return (res)\n") == shi.replace("    return (res) \n", "    return (res)\n")
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(ours.splitlines(True), shi.splitlines(True))) if a != b]
    assert len(ours.splitlines()) == len(shi.splitlines()) and len(diff) == 1, diff   # ... exactly one line differs
