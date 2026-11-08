"""Opt-in --seed and --single-bos for the MBPP runner (python -m bench).

Default (neither flag) must be unchanged: same prompt handed to the generators (a str, which they encode with
``tokenizer.encode(prompt_text, return_tensors="pt")``), same generator kwargs, same record keys, no reseeding.

--seed S: before each task the runner reseeds torch / numpy / python from bench.seeding.task_seed(S, task_id), so a
task's samples do not depend on task order or on resume. --single-bos: a chat-template string that already starts with
tokenizer.bos_token is encoded once with add_special_tokens=False (one BOS instead of two).
All model-dependent tests use a tiny random Llama on CPU with the real (locally cached) CodeLlama tokenizer.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

import bench.runner as R
from bench.checkpointing import check_resume_compatible
from bench.prompts import encode_single_bos, format_prompt_instruct
from bench.seeding import seed_task, task_seed

CHAT = "codellama/CodeLlama-7b-Instruct-hf"
BASE = "codellama/CodeLlama-7b-hf"
QWEN_CHAT = "Qwen/Qwen2.5-Coder-7B-Instruct"
TASKS = [
    {"task_id": tid, "text": f"Write a function to add {tid} to x.", "code": "",
     "test_list": [f"assert add{tid}(1) == {1 + tid}", f"assert add{tid}(0) == {tid}"],
     "test_setup_code": "", "challenge_test_list": []}
    for tid in (11, 12, 13)
]
OLD_RECORD_KEYS = {"model", "backend", "method", "temperature", "task_id", "prompt_text", "samples", "test_list",
                   "timestamp"}
SAMPLING = [
    ["--method", "temp", "--temperature", "0.7"],
    ["--method", "top_p", "--top-p", "0.9", "--temperature", "1.0"],
    ["--method", "pless", "--temperature", "0.7"],
    ["--method", "pless_norm", "--temperature", "0.7"],
]


def _tok(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:  # pragma: no cover - machine without the cache
        pytest.skip(f"{name} tokenizer not cached locally: {e}")


@pytest.fixture(scope="module")
def tok():
    return _tok(CHAT)


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    return LlamaForCausalLM(cfg).eval()


def _run(monkeypatch, tmp_path, argv, model, tok, tasks=TASKS):
    from datasets import Dataset
    monkeypatch.setattr(R, "load_dataset", lambda *a, **k: Dataset.from_list(tasks))
    monkeypatch.setattr(R, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    monkeypatch.setattr(sys, "argv", ["bench", *argv, "--mbpp-config", "full", "--results-dir", str(tmp_path)])
    R.main()
    files = list(Path(tmp_path).rglob("*.jsonl"))
    assert len(files) == 1, files
    return files[0], [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


def _strip(recs):
    return [{k: v for k, v in r.items() if k != "timestamp"} for r in recs]


# ---------------------------------------------------------------- argument parsing
def test_defaults_are_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "x", "--method", "temp"])
    a = R.parse_args()
    assert a.seed is None and a.single_bos is False


def test_seed_and_single_bos_parse(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "x", "--method", "temp", "--seed", "7", "--single-bos"])
    a = R.parse_args()
    assert a.seed == 7 and a.single_bos is True


@pytest.mark.parametrize("extra", [["--seed", "-1"], ["--seed", "1", "--backend", "vllm"],
                                   ["--single-bos", "--backend", "vllm"]])
def test_rejected_combinations(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "x", "--method", "temp", *extra])
    with pytest.raises(SystemExit) as e:
        R.parse_args()
    assert e.value.code == 2


# ---------------------------------------------------------------- seeding unit
def test_task_seed_is_deterministic_and_distinct():
    assert task_seed(1234, 11) == task_seed(1234, 11)
    seeds = {task_seed(s, t) for s in range(5) for t in range(11, 511)}
    assert len(seeds) == 5 * 500   # no collisions across tasks or neighbouring base seeds (unlike seed + task_id)
    assert all(0 <= s < 2**32 for s in seeds)


def test_seed_task_reseeds_torch_numpy_python():
    import random
    import numpy as np
    seed_task(3, 11)
    a = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])
    torch.rand(5), np.random.rand(5), random.random()
    seed_task(3, 11)
    b = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])
    assert a == b


# ---------------------------------------------------------------- default byte-identity (stubbed generators)
GEN_FNS = ("generate_samples", "generate_samples_standard", "generate_samples_greedy", "generate_samples_beam")
DEFAULT_CALL_KWARGS = {   # exactly what bench/runner.py passed at 06a5441 (besides model/tokenizer/prompt_text)
    "greedy": ("generate_samples_greedy", {"max_new_tokens", "stop_strings"}),
    "beam": ("generate_samples_beam", {"num_beams", "max_new_tokens", "stop_strings"}),
    "temp": ("generate_samples_standard", {"n_samples", "max_new_tokens", "temperature", "stop_strings", "top_p",
                                           "top_k"}),
    "top_p": ("generate_samples_standard", {"n_samples", "max_new_tokens", "temperature", "stop_strings", "top_p",
                                            "top_k"}),
    "pless": ("generate_samples", {"sampler_fn", "n_samples", "max_new_tokens", "temperature", "stop_strings",
                                   "entropy_log"}),
}


def _stub_generators(monkeypatch, calls):
    for name in GEN_FNS:
        def f(_name=name, **kw):
            calls.append((_name, kw))
            return ["def f():\n    return 1"] * kw.get("n_samples", 1)
        monkeypatch.setattr(R, name, f)


@pytest.mark.parametrize("model", [CHAT, BASE])
@pytest.mark.parametrize("cfg", [["--method", "greedy", "--n-samples", "1"],
                                 ["--method", "beam", "--num-beams", "4", "--n-samples", "1"],
                                 ["--method", "temp", "--temperature", "0.7"],
                                 ["--method", "top_p", "--top-p", "0.9"],
                                 ["--method", "pless", "--temperature", "0.7"]])
def test_default_generator_calls_and_records_unchanged(monkeypatch, tmp_path, tok, model, cfg):
    calls, seeded = [], []
    _stub_generators(monkeypatch, calls)
    monkeypatch.setattr(torch, "manual_seed", lambda *a: seeded.append(a))
    argv = ["--model", model, *cfg] + (["--prompt-style", "bigcode"] if model == BASE else [])
    _, recs = _run(monkeypatch, tmp_path, argv, None, tok)
    fn, keys = DEFAULT_CALL_KWARGS[cfg[1]]
    assert [c[0] for c in calls] == [fn] * 3
    for (_, kw), task in zip(calls, TASKS):
        assert set(kw) == keys | {"model", "tokenizer", "prompt_text"}
        assert isinstance(kw["prompt_text"], str)       # the generator still does tokenizer.encode(prompt_text)
        if model == CHAT:
            assert kw["prompt_text"] == format_prompt_instruct({**task, "prompt": task["text"]}, tok)[0]
            assert tok.encode(kw["prompt_text"])[:2] == [1, 1]   # the historical double BOS, untouched
    extra = {"num_beams"} if cfg[1] == "beam" else {"top_p"} if cfg[1] == "top_p" else set()
    for r in recs:
        assert set(r) == OLD_RECORD_KEYS | extra       # no seed / single_bos keys
    assert seeded == []                                  # no reseeding in a default run


# ---------------------------------------------------------------- --seed with a real tiny model
@pytest.mark.parametrize("cfg", SAMPLING, ids=lambda c: c[1])
def test_same_seed_same_samples_different_seed_different(monkeypatch, tmp_path, tok, tiny, cfg):
    argv = ["--model", CHAT, *cfg, "--n-samples", "3", "--max-new-tokens", "6"]
    torch.manual_seed(111)
    _, a = _run(monkeypatch, tmp_path / "a", argv + ["--seed", "1234"], tiny, tok)
    torch.manual_seed(999)   # a different global state must not matter
    _, b = _run(monkeypatch, tmp_path / "b", argv + ["--seed", "1234"], tiny, tok)
    _, c = _run(monkeypatch, tmp_path / "c", argv + ["--seed", "1235"], tiny, tok)
    assert _strip(a) == _strip(b)
    assert [r["samples"] for r in a] != [r["samples"] for r in c]
    assert all(r["seed"] == 1234 for r in a) and all(r["seed"] == 1235 for r in c)
    assert all(len(set(r["samples"])) > 1 for r in a)   # genuinely stochastic, not a degenerate collapse


@pytest.mark.parametrize("cfg", SAMPLING, ids=lambda c: c[1])
def test_resume_and_task_order_independent(monkeypatch, tmp_path, tok, tiny, cfg):
    argv = ["--model", CHAT, *cfg, "--n-samples", "3", "--max-new-tokens", "6", "--seed", "42"]
    _, full = _run(monkeypatch, tmp_path / "full", argv, tiny, tok)
    _run(monkeypatch, tmp_path / "res", argv + ["--max-problems", "1"], tiny, tok)   # "interrupted" after task 11
    torch.rand(17)                                                                      # global RNG moved on
    _, resumed = _run(monkeypatch, tmp_path / "res", argv, tiny, tok)
    assert _strip(resumed) == _strip(full)
    _, rev = _run(monkeypatch, tmp_path / "rev", argv, tiny, tok, tasks=TASKS[::-1])
    assert sorted(_strip(rev), key=lambda r: r["task_id"]) == _strip(full)


@pytest.mark.parametrize("cfg", [["--method", "greedy", "--n-samples", "1"],
                                 ["--method", "beam", "--num-beams", "2", "--n-samples", "1"]])
def test_seed_is_harmless_for_greedy_and_beam(monkeypatch, tmp_path, tok, tiny, cfg):
    argv = ["--model", CHAT, *cfg, "--max-new-tokens", "5"]
    _, plain = _run(monkeypatch, tmp_path / "p", argv, tiny, tok)
    _, seeded = _run(monkeypatch, tmp_path / "s", argv + ["--seed", "5"], tiny, tok)
    assert [r["samples"] for r in plain] == [r["samples"] for r in seeded]
    assert all(r["seed"] == 5 for r in seeded) and all("seed" not in r for r in plain)


# ---------------------------------------------------------------- resume guard
def _write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


@pytest.mark.parametrize("existing,requested,ok", [
    ({}, {"seed": None, "single_bos": None}, True),                          # every historical file, default run
    ({"seed": 1}, {"seed": 1, "single_bos": None}, True),
    ({}, {"seed": 1, "single_bos": None}, False),                            # seeded run onto an unseeded file
    ({"seed": 1}, {"seed": None, "single_bos": None}, False),
    ({"seed": 1}, {"seed": 2, "single_bos": None}, False),
    ({}, {"seed": None, "single_bos": True}, False),
    ({"single_bos": True}, {"seed": None, "single_bos": None}, False),
])
def test_check_resume_compatible(tmp_path, existing, requested, ok):
    p = tmp_path / "f.jsonl"
    _write(p, [{"task_id": 11, **existing}])
    if ok:
        check_resume_compatible(p, requested)
    else:
        with pytest.raises(SystemExit, match="--no-resume"):
            check_resume_compatible(p, requested)


def test_missing_file_is_compatible(tmp_path):
    check_resume_compatible(tmp_path / "none.jsonl", {"seed": 3, "single_bos": True})


def test_runner_refuses_to_mix_seeded_into_unseeded(monkeypatch, tmp_path, tok, tiny):
    argv = ["--model", CHAT, "--method", "temp", "--n-samples", "2", "--max-new-tokens", "3"]
    f, _ = _run(monkeypatch, tmp_path, argv + ["--max-problems", "1"], tiny, tok)
    before = f.read_text()
    with pytest.raises(SystemExit, match="seed"):
        _run(monkeypatch, tmp_path, argv + ["--seed", "1"], tiny, tok)
    assert f.read_text() == before
    _, recs = _run(monkeypatch, tmp_path, argv + ["--seed", "1", "--no-resume"], tiny, tok)
    assert len(recs) == 3 and all(r["seed"] == 1 for r in recs)


# ---------------------------------------------------------------- --single-bos
def test_codellama_chat_double_vs_single_bos(tok):
    text = format_prompt_instruct({**TASKS[0], "prompt": TASKS[0]["text"]}, tok)[0]
    assert text.startswith(tok.bos_token)
    double = tok.encode(text)
    single = encode_single_bos(text, tok)
    assert double[:2] == [1, 1] and single[0] == 1 and single[1] != 1
    assert single == double[1:]


def test_single_bos_unchanged_without_leading_bos(tok):
    from bench.prompts import format_prompt_base_bigcode
    text = format_prompt_base_bigcode({**TASKS[0], "prompt": TASKS[0]["text"]})[0]
    assert encode_single_bos(text, tok) is text          # base prompt: string returned untouched
    qwen = _tok(QWEN_CHAT)
    assert qwen.bos_token is None
    qtext = format_prompt_instruct({**TASKS[0], "prompt": TASKS[0]["text"]}, qwen)[0]
    assert encode_single_bos(qtext, qwen) is qtext
    assert encode_single_bos([1, 2, 3], tok) == [1, 2, 3]   # pre-tokenised (old Qwen) prompts pass through


@pytest.mark.parametrize("model,expect_list", [(CHAT, True), (BASE, False)])
def test_runner_single_bos_prompt_and_record(monkeypatch, tmp_path, tok, model, expect_list):
    calls = []
    _stub_generators(monkeypatch, calls)
    argv = ["--model", model, "--method", "greedy", "--n-samples", "1", "--single-bos"]
    argv += ["--prompt-style", "bigcode"] if model == BASE else []
    _, recs = _run(monkeypatch, tmp_path, argv, None, tok)
    for (_, kw), task in zip(calls, TASKS):
        pt = kw["prompt_text"]
        if expect_list:
            text = format_prompt_instruct({**task, "prompt": task["text"]}, tok)[0]
            assert pt == tok.encode(text)[1:] and pt[:2] != [1, 1]
        else:
            assert isinstance(pt, str)
    assert all(r["single_bos"] is True for r in recs)
    assert all("seed" not in r for r in recs)


def test_single_bos_end_to_end_input_ids(monkeypatch, tmp_path, tok, tiny):
    """The ids the model actually sees (prefill of the custom loop) start with exactly one BOS."""
    seen = []
    orig = tiny.forward

    def spy(*a, **k):
        if k.get("past_key_values") is None:
            seen.append(k["input_ids"][0].tolist())
        return orig(*a, **k)
    monkeypatch.setattr(tiny, "forward", spy)
    argv = ["--model", CHAT, "--method", "pless", "--n-samples", "2", "--max-new-tokens", "2"]
    _run(monkeypatch, tmp_path / "d", argv + ["--max-problems", "1"], tiny, tok)
    _run(monkeypatch, tmp_path / "s", argv + ["--max-problems", "1", "--single-bos"], tiny, tok)
    assert seen[0][:2] == [1, 1] and seen[1][:2] != [1, 1] and seen[1] == seen[0][1:]


def test_encode_single_bos_guarantees_only_the_extra_bos_is_removed():
    """Runtime guarantee (review 2026-11-08): for any tokenizer, the single-BOS ids must equal the default encoding
    minus its first token. Otherwise refuse, never silently send a different prompt."""
    import pytest

    class OddTok:  # default encoding is NOT "[BOS] + no-special encoding"
        bos_token = "<s>"
        def encode(self, text, add_special_tokens=True):
            return [1, 1, 7, 8] if add_special_tokens else [1, 9, 9]

    with pytest.raises(ValueError, match="single-BOS"):
        encode_single_bos("<s>[INST] hi", OddTok())
