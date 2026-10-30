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
