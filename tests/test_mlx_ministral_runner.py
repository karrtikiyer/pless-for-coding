"""Config and record-format tests for scripts/run_mlxlm_ministral_gk_pools.py.

Needs mlx (the runner imports the mlx logits processors). Run with
  MLX_SITE_PACKAGES=$HOME/.venv-vllm-metal/lib/python3.12/site-packages uv run pytest tests/test_mlx_ministral_runner.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("MLX_SITE_PACKAGES"):
    sys.path.append(os.environ["MLX_SITE_PACKAGES"])
pytest.importorskip("mlx.core", reason="mlx not importable; set MLX_SITE_PACKAGES")

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mgk_runner", REPO / "scripts/run_mlxlm_ministral_gk_pools.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

VLLM_RECORD = REPO / "results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/gk_0_1_atcoder_think_t1.0.jsonl"


def test_requested_k_values_with_anchor_last():
    ks = [k for _, k in runner.CONFIGS]
    assert ks[:-1] == [0.025, 0.075, 0.15, 0.25, 0.3, 0.35, 0.45]
    assert runner.CONFIGS[-1] == ("gk_0_1", 0.1)


@pytest.mark.parametrize("name,k", runner.CONFIGS)
def test_config_name_encodes_its_k(name, k):
    assert name == "gk_" + str(k).replace(".", "_")


def test_production_settings_match_vllm_runner():
    assert runner.MODEL_ID == "mistralai/Ministral-3-3B-Reasoning-2512"
    assert runner.N_SAMPLES == 10 and runner.TEMPERATURE == 1.0 and runner.BASE_SEED == 42
    assert runner.THINK_END_MARKER == "[/THINK]"
    assert runner.BATCHES == (1, 2)
    if "MLXLM_MGK_MAX_NEW_TOKENS" not in os.environ:
        assert runner.MAX_NEW_TOKENS == 35000


def test_system_prompt_is_the_vllm_runners():
    text = (REPO / "scripts/run_pless_variants_ministral_3b_lcb_cuda.py").read_text()
    prompt = runner._vllm_runner_system_prompt()
    assert prompt.startswith("# HOW YOU SHOULD THINK AND ANSWER")
    assert "[THINK]" in prompt and "[/THINK]" in prompt
    assert "You are an expert Python programmer." in prompt and "MINISTRAL_SYSTEM_PROMPT" in text


def test_output_name_matches_vllm_convention():
    assert runner.out_path("gk_0_15", "leetcode").name == "gk_0_15_leetcode_think_t1.0.jsonl"


@pytest.mark.parametrize("name,k", runner.CONFIGS)
def test_record_carries_config_k(name, k):
    p = SimpleNamespace(platform="atcoder", difficulty="easy", contest_date="2024-01-01", question="q")
    f = runner._base_record_fields(name, k, p)
    assert f["method"] == name and f["renyi_k"] == k and f["sampler_name"] == "pless_renyi"


@pytest.mark.skipif(not VLLM_RECORD.exists(), reason="vLLM reference record not present")
def test_record_has_every_vllm_field():
    vllm_keys = set(json.loads(VLLM_RECORD.open().readline()))
    p = SimpleNamespace(platform="atcoder", difficulty="easy", contest_date="2024-01-01", question="q")
    ours = set(runner._base_record_fields("gk_0_1", 0.1, p)) | {
        "task_id", "samples", "samples_with_thinking", "timestamp"}
    assert vllm_keys <= ours, sorted(vllm_keys - ours)


def test_easy_backfill_12_medium_backfill_8_hard_10():
    if not set(os.environ) & {"MLXLM_MGK_EASY_CAP", "MLXLM_MGK_EASY_BACKFILL", "MLXLM_MGK_MEDIUM_CAP",
                              "MLXLM_MGK_MEDIUM_BACKFILL", "MLXLM_MGK_HARD_CAP"}:
        assert runner.EASY_CAP == 120 and runner.EASY_BACKFILL == 12
        assert runner.MEDIUM_CAP == 120 and runner.MEDIUM_BACKFILL == 8
        assert runner.HARD_CAP == 10


def test_plan_gives_easy_backfill_and_no_backfill_elsewhere():
    from bench.mlx_difficulty_pools import build_pool_batches
    tasks = [(f"e{i}", "easy", 10) for i in range(30)] + [(f"m{i}", "medium", 10) for i in range(30)]
    tasks += [(f"h{i}", "hard", 10) for i in range(3)]
    backfill = {"easy": runner.EASY_BACKFILL, "medium": runner.MEDIUM_BACKFILL}
    for b in build_pool_batches(tasks, easy_cap=runner.EASY_CAP, medium_cap=runner.MEDIUM_CAP,
                                hard_cap=runner.HARD_CAP, easy_backfill=runner.EASY_BACKFILL,
                                medium_backfill=runner.MEDIUM_BACKFILL):
        rows = sum(n for _, n in b.rows)
        if b.phase in backfill and rows > backfill[b.phase]:
            assert b.completion_batch_size == backfill[b.phase]
        else:
            assert b.completion_batch_size >= rows


def test_run_batch_passes_backfill_and_prefill_sizes_to_batchgenerator(monkeypatch, tmp_path):
    # If prefill_batch_size is dropped, mlx-lm defaults to 8 and a cap of 12 reaches 14+.
    import importlib
    g = importlib.import_module("mlx_lm.generate")
    from bench.mlx_difficulty_pools import PoolBatch
    seen = {}

    class FakeGen:
        def __init__(self, model, **kw):
            seen.update(kw)
        def insert(self, prompts, max_tokens):
            return list(range(len(prompts)))
        def next_generated(self):
            return []
        def close(self):
            pass

    monkeypatch.setattr(g, "BatchGenerator", FakeGen)
    monkeypatch.setattr(runner, "PARTIAL_DIR", tmp_path)
    batch = PoolBatch("easy", [("t", 20)], completion_batch_size=12, prefill_batch_size=1)
    p = SimpleNamespace(platform="atcoder", difficulty="easy", contest_date="x", question="q", task_id="t")
    try:
        runner.run_batch(None, None, [2], "gk_0_1", 0.1, "easy batch 1/1", batch, [("t", 0, 20)], 1,
                         {"t": p}, {"t": [1, 2]}, {})
    except RuntimeError:
        pass  # fake generator returns nothing, so the post-drain check fires
    assert seen["completion_batch_size"] == 12 and seen["prefill_batch_size"] == 1


def test_plan_passes_backfill_settings_to_scheduler(monkeypatch):
    # Without medium_backfill reaching build_pool_batches, medium would insert 120
    # rows with completion_batch_size=120 (near-certain OOM) and the suite stays green.
    seen = {}

    def fake_build(tasks, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(runner, "build_pool_batches", fake_build)
    monkeypatch.setattr(runner, "_load_done", lambda config: set())
    monkeypatch.setattr(runner, "_load_banked", lambda config: {})
    probs = [SimpleNamespace(task_id="t1", difficulty="medium", platform="atcoder")]
    runner._plan(probs)
    assert seen["easy_backfill"] == runner.EASY_BACKFILL
    assert seen["medium_backfill"] == runner.MEDIUM_BACKFILL
    assert seen["medium_cap"] == runner.MEDIUM_CAP and seen["hard_cap"] == runner.HARD_CAP
