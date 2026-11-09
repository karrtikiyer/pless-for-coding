"""Automatic CUDA-OOM fallback in the MBPP runner (python -m bench).

On a CUDA out-of-memory error while generating a task's n samples, the runner frees memory and retries THAT task in
chunks of 5, then 2, then 1 (only sizes < n_samples), re-applying the per-task reseed before each retry. A successful
retry adds ``hf_batch_size`` and ``oom_retry: true`` to that task's record only. If every size fails, the runner logs
"Error on task_id=..." exactly as before (the staging scripts' per-task-error check stops the queue).

Chunked sampling is reproducible given (seed, task_id, chunk size) but is NOT the same draw as the unchunked call.
Default (no OOM): same generator calls (no hf_batch_size kwarg) and same records.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

import bench.runner as R
from bench.generator import generate_samples, generate_samples_standard
from bench.sampler_bridge import SAMPLERS
from bench.seeding import seed_task

BASE = "codellama/CodeLlama-7b-hf"
CHAT = "codellama/CodeLlama-7b-Instruct-hf"
TASKS = [
    {"task_id": tid, "text": f"Write a function to add {tid} to x.", "code": "",
     "test_list": [f"assert add{tid}(1) == {1 + tid}"], "test_setup_code": "", "challenge_test_list": []}
    for tid in (11, 12, 13)
]
GEN_FNS = ("generate_samples", "generate_samples_standard", "generate_samples_greedy", "generate_samples_beam")
OOM_MSG = "CUDA out of memory. Tried to allocate 2.00 GiB"
SAMPLING_CFGS = {
    "temp": (["--method", "temp", "--temperature", "0.7"], "generate_samples_standard"),
    "top_p": (["--method", "top_p", "--top-p", "0.95", "--temperature", "0.2"], "generate_samples_standard"),
    "pless": (["--method", "pless", "--temperature", "1.0"], "generate_samples"),
    "pless_norm": (["--method", "pless_norm", "--temperature", "0.6"], "generate_samples"),
}


def _run(monkeypatch, tmp_path, argv, model=None, tok=None, tasks=TASKS):
    from datasets import Dataset
    monkeypatch.setattr(R, "load_dataset", lambda *a, **k: Dataset.from_list(tasks))
    monkeypatch.setattr(R, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    monkeypatch.setattr(sys, "argv", ["bench", *argv, "--mbpp-config", "full", "--results-dir", str(tmp_path)])
    R.main()
    files = list(Path(tmp_path).rglob("*.jsonl"))
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


def _stub(monkeypatch, calls, fail):
    """Stub every generator; ``fail(task_marker, kw)`` -> exception to raise (or None)."""
    for name in GEN_FNS:
        def f(_name=name, **kw):
            calls.append((_name, kw))
            exc = fail(kw)
            if exc is not None:
                raise exc
            n = kw.get("n_samples", 1)
            return [f"def f():\n    return {i}" for i in range(n)]
        monkeypatch.setattr(R, name, f)


def _is_task(kw, tid):
    return f"add {tid} " in kw["prompt_text"] if isinstance(kw["prompt_text"], str) else False


BIGCODE = ["--model", BASE, "--prompt-style", "bigcode"]


# ---------------------------------------------------------------- OOM classification
def test_is_cuda_oom():
    assert R._is_cuda_oom(torch.cuda.OutOfMemoryError(OOM_MSG))
    assert R._is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not R._is_cuda_oom(RuntimeError("shape mismatch"))
    assert not R._is_cuda_oom(ValueError("out of memory"))   # only RuntimeError-family counts


# ---------------------------------------------------------------- retry ladder (stubbed generators)
@pytest.mark.parametrize("name", list(SAMPLING_CFGS))
def test_oom_once_retries_with_chunk_5(monkeypatch, tmp_path, capsys, name):
    cfg, fn = SAMPLING_CFGS[name]
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG)
          if _is_task(kw, 12) and kw.get("hf_batch_size") is None else None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + cfg + ["--seed", "42"])
    assert [r["task_id"] for r in recs] == [11, 12, 13]
    by = {r["task_id"]: r for r in recs}
    assert by[12]["hf_batch_size"] == 5 and by[12]["oom_retry"] is True
    assert len(by[12]["samples"]) == 10
    for tid in (11, 13):
        assert "hf_batch_size" not in by[tid] and "oom_retry" not in by[tid]
    assert all(c[0] == fn for c in calls)
    t12 = [kw for _, kw in calls if _is_task(kw, 12)]
    assert [kw.get("hf_batch_size") for kw in t12] == [None, 5]
    assert "hf_batch_size" not in t12[0]                 # first attempt = the unchanged default call
    assert t12[1]["n_samples"] == 10
    for _, kw in calls:
        if not _is_task(kw, 12):
            assert "hf_batch_size" not in kw
    cap = capsys.readouterr()
    out = cap.out + cap.err
    assert "OOM on task_id=12: retried with batch size 5 (ok)" in out
    assert "Error on task_id=" not in out


def test_oom_ladder_5_2_1(monkeypatch, tmp_path, capsys):
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG)
          if _is_task(kw, 11) and kw.get("hf_batch_size") in (None, 5, 2) else None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["pless"][0])
    by = {r["task_id"]: r for r in recs}
    assert by[11]["hf_batch_size"] == 1 and by[11]["oom_retry"] is True and len(by[11]["samples"]) == 10
    assert [kw.get("hf_batch_size") for _, kw in calls if _is_task(kw, 11)] == [None, 5, 2, 1]
    out = capsys.readouterr()
    text = out.out + out.err
    assert "OOM on task_id=11: retried with batch size 1 (ok)" in text
    assert "Error on task_id=" not in text


def test_runtimeerror_out_of_memory_also_retried(monkeypatch, tmp_path):
    calls = []
    _stub(monkeypatch, calls, lambda kw: RuntimeError("CUDA error: out of memory")
          if kw.get("hf_batch_size") is None else None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["temp"][0])
    assert all(r["hf_batch_size"] == 5 and r["oom_retry"] is True for r in recs)


def test_all_sizes_fail_logs_error_as_today(monkeypatch, tmp_path, capsys):
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG) if _is_task(kw, 12) else None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["temp"][0])
    assert [r["task_id"] for r in recs] == [11, 13]
    assert [kw.get("hf_batch_size") for _, kw in calls if _is_task(kw, 12)] == [None, 5, 2, 1]
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Error on task_id=12:" in text and "out of memory" in text
    assert text.count("Error on task_id=") == 1


def test_non_oom_error_not_retried(monkeypatch, tmp_path, capsys):
    calls = []
    _stub(monkeypatch, calls, lambda kw: RuntimeError("shape mismatch") if _is_task(kw, 12) else None)
    _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["pless"][0])
    assert len([1 for _, kw in calls if _is_task(kw, 12)]) == 1
    assert "Error on task_id=12: shape mismatch" in capsys.readouterr().out + ""


@pytest.mark.parametrize("cfg", [["--method", "greedy", "--n-samples", "1"],
                                 ["--method", "beam", "--num-beams", "4", "--n-samples", "1"]])
def test_greedy_beam_oom_not_retried(monkeypatch, tmp_path, capsys, cfg):
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG) if _is_task(kw, 12) else None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + cfg)
    assert [r["task_id"] for r in recs] == [11, 13]
    assert len([1 for _, kw in calls if _is_task(kw, 12)]) == 1
    assert "Error on task_id=12:" in capsys.readouterr().out


def test_ladder_skips_sizes_not_below_n(monkeypatch, tmp_path):
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG)
          if kw.get("hf_batch_size") is None else None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["temp"][0] + ["--n-samples", "3"])
    assert all(r["hf_batch_size"] == 2 for r in recs)
    assert [kw.get("hf_batch_size") for _, kw in calls][:2] == [None, 2]


def test_reseed_before_each_retry(monkeypatch, tmp_path):
    seeds = []
    real = R.seed_task
    monkeypatch.setattr(R, "seed_task", lambda s, t: (seeds.append((s, t)), real(s, t))[1])
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG)
          if _is_task(kw, 12) and kw.get("hf_batch_size") in (None, 5) else None)
    _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["temp"][0] + ["--seed", "7"])
    assert seeds == [(7, 11), (7, 12), (7, 12), (7, 12), (7, 13)]


def test_no_oom_default_calls_and_records_unchanged(monkeypatch, tmp_path):
    calls = []
    _stub(monkeypatch, calls, lambda kw: None)
    recs = _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["pless"][0])
    assert len(calls) == 3 and all("hf_batch_size" not in kw for _, kw in calls)
    assert all("hf_batch_size" not in r and "oom_retry" not in r for r in recs)


def test_oom_memory_freed(monkeypatch, tmp_path):
    freed = []
    monkeypatch.setattr(R, "_free_cuda_memory", lambda: freed.append(1))
    calls = []
    _stub(monkeypatch, calls, lambda kw: torch.cuda.OutOfMemoryError(OOM_MSG)
          if _is_task(kw, 12) and kw.get("hf_batch_size") in (None, 5) else None)
    _run(monkeypatch, tmp_path, BIGCODE + SAMPLING_CFGS["temp"][0])
    assert len(freed) == 2


# ---------------------------------------------------------------- real tiny model (chunked generators)
def _tok(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:  # pragma: no cover
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


PROMPT = "def add(a, b):\n"


@pytest.mark.parametrize("sampler", ["pless", "pless_norm"])
def test_chunked_pless_n_samples_and_reproducible(tok, tiny, sampler):
    kw = dict(model=tiny, tokenizer=tok, prompt_text=PROMPT, sampler_fn=SAMPLERS[sampler], n_samples=5,
              max_new_tokens=6, temperature=1.0)
    seed_task(42, 11)
    a = generate_samples(**kw, hf_batch_size=2)
    torch.rand(13)
    seed_task(42, 11)
    b = generate_samples(**kw, hf_batch_size=2)
    assert len(a) == 5 and a == b
    assert len(set(a)) > 1
    # Semantics: k consecutive chunks from one per-task RNG stream (reseed once, then 2 + 2 + 1).
    seed_task(42, 11)
    manual = []
    for n in (2, 2, 1):
        manual += generate_samples(**{**kw, "n_samples": n})
    assert a == manual


def test_chunked_pless_batch_ge_n_is_unchunked(tok, tiny):
    kw = dict(model=tiny, tokenizer=tok, prompt_text=PROMPT, sampler_fn=SAMPLERS["pless"], n_samples=4,
              max_new_tokens=6, temperature=1.0)
    seed_task(1, 2)
    a = generate_samples(**kw)
    seed_task(1, 2)
    b = generate_samples(**kw, hf_batch_size=4)
    seed_task(1, 2)
    c = generate_samples(**kw, hf_batch_size=None)
    assert a == b == c


def test_chunked_pless_entropy_log_and_token_ids(tok, tiny):
    kw = dict(model=tiny, tokenizer=tok, prompt_text=PROMPT, sampler_fn=SAMPLERS["pless"], n_samples=5,
              max_new_tokens=4, temperature=1.0)
    log = []
    seed_task(3, 4)
    samples, ids, plen = generate_samples(**kw, hf_batch_size=2, entropy_log=log, return_token_ids=True)
    assert len(samples) == 5 and len(ids) == 5
    assert plen == len(tok.encode(PROMPT))
    assert {r["sample_id"] for r in log} == set(range(5))


def test_chunked_standard_reproducible(tok, tiny):
    kw = dict(model=tiny, tokenizer=tok, prompt_text=PROMPT, n_samples=5, max_new_tokens=6, temperature=0.7)
    seed_task(42, 11)
    a = generate_samples_standard(**kw, hf_batch_size=2)
    seed_task(42, 11)
    b = generate_samples_standard(**kw, hf_batch_size=2)
    assert len(a) == 5 and a == b


@pytest.mark.parametrize("name", ["temp", "pless"])
def test_runner_real_model_oom_retry_reproducible(monkeypatch, tmp_path, tok, tiny, name):
    """End to end with the real tiny model: first attempt for task 12 OOMs, the retry runs the real chunked
    generator; two seeded runs give the same records."""
    cfg, fn = SAMPLING_CFGS[name]
    real = getattr(R, fn)

    def flaky(**kw):
        if kw.get("hf_batch_size") is None and "add 12 to x" in str(kw["prompt_text"]):
            raise torch.cuda.OutOfMemoryError(OOM_MSG)
        return real(**kw)
    monkeypatch.setattr(R, fn, flaky)
    argv = ["--model", CHAT, *cfg, "--n-samples", "6", "--max-new-tokens", "5", "--seed", "42"]
    a = _run(monkeypatch, tmp_path / "a", argv, tiny, tok)
    torch.rand(7)
    b = _run(monkeypatch, tmp_path / "b", argv, tiny, tok)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "timestamp"} for r in rs]
    assert strip(a) == strip(b)
    by = {r["task_id"]: r for r in a}
    assert by[12]["hf_batch_size"] == 5 and by[12]["oom_retry"] is True and len(by[12]["samples"]) == 6
    assert "oom_retry" not in by[11] and "oom_retry" not in by[13]
