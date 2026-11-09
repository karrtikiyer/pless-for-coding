"""bench.apps runner: opt-in --gpu-memory-utilization reaches vLLM's load_engine; default unchanged.

Needed to fit Qwen3-8B at a 32768-token cap on a 24GB RTX 4090 (at the default 0.90 only ~4.4 GiB of KV
cache is left, less than one full-length request). A fake `bench.generator_vllm` records load_engine's
kwargs and aborts before any generation, so no vLLM/GPU is needed.
"""

from __future__ import annotations

import sys
import types

import pytest

BASE = ["bench.apps", "--model", "Qwen/Qwen3-8B", "--backend", "vllm", "--method", "pless_renyi",
        "--renyi-k", "0.25", "--temperature", "1.0", "--enable-thinking",
        "--source", "CODEFORCES", "--difficulty", "interview"]


class _Stop(Exception):
    pass


def _capture_load_engine(monkeypatch, tmp_path, extra):
    seen = {}

    def fake_load_engine(model, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setitem(sys.modules, "bench.generator_vllm",
                        types.SimpleNamespace(load_engine=fake_load_engine))
    monkeypatch.setattr(sys, "argv", BASE + ["--results-dir", str(tmp_path)] + extra)
    from bench.apps import runner
    with pytest.raises(_Stop):
        runner.main()
    return seen


def test_parser_default_is_none_and_accepts_value(monkeypatch):
    from bench.apps.runner import _build_argparser
    p = _build_argparser()
    assert p.parse_args(BASE[1:]).gpu_memory_utilization is None
    assert p.parse_args(BASE[1:] + ["--gpu-memory-utilization", "0.95"]).gpu_memory_utilization == 0.95


def test_default_does_not_pass_gpu_memory_utilization(monkeypatch, tmp_path):
    seen = _capture_load_engine(monkeypatch, tmp_path, [])
    assert "gpu_memory_utilization" not in seen  # load_engine keeps its own 0.90 default
    assert seen["max_model_len"] is None


def test_flags_reach_load_engine(monkeypatch, tmp_path):
    seen = _capture_load_engine(monkeypatch, tmp_path,
                                ["--gpu-memory-utilization", "0.95", "--max-model-len", "36000"])
    assert seen["gpu_memory_utilization"] == 0.95
    assert seen["max_model_len"] == 36000


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.1"])
def test_out_of_range_utilization_rejected(monkeypatch, bad):
    monkeypatch.setattr(sys, "argv", BASE + ["--gpu-memory-utilization", bad])
    from bench.apps.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args()
