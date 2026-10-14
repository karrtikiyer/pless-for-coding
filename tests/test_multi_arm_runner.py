"""Tests for ``bench.apps.multi_arm_runner``.

Regression guard: the preset arm → generation params mapping MUST reproduce
``run_cf748_apps_qwen3.sh:arm_args()`` byte-for-byte, and the per-arm output
paths MUST match what ``bench.apps.runner`` would produce for the same arm.

Can't test the actual vLLM generation path without a GPU; those are covered by
the smoke-regression check (small sweep via both paths, compare metrics).
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from bench.apps.multi_arm_runner import (
    _arm_output_path,
    _arm_to_namespace,
    _parse_arm_raw,
    _preset_arm_spec,
)
from bench.apps.runner import _method_key, _output_path


# ---------------------------------------------------------------------------
# Preset arm mapping — mirror of shell arm_args() in run_cf748_apps_qwen3.sh.
# ---------------------------------------------------------------------------


def test_preset_pless():
    s = _preset_arm_spec("pless")
    assert s["method"] == "pless"
    assert s["temperature"] == 1.0
    assert s["top_p"] == 1.0 and s["top_k"] == 0
    assert s["renyi_k"] is None and s["alpha"] is None


def test_preset_pless_norm():
    s = _preset_arm_spec("pless_norm")
    assert s["method"] == "pless_norm"
    assert s["temperature"] == 1.0
    assert s["top_p"] == 1.0 and s["top_k"] == 0


def test_preset_renyi_k():
    for k_str, k_val in [("0.05", 0.05), ("0.075", 0.075), ("0.1", 0.1),
                         ("0.35", 0.35), ("1.6", 1.6)]:
        s = _preset_arm_spec(f"renyi_k{k_str}")
        assert s["method"] == "pless_renyi"
        assert s["temperature"] == 1.0
        assert s["renyi_k"] == k_val
        assert s["top_p"] == 1.0 and s["top_k"] == 0


def test_preset_rec():
    s = _preset_arm_spec("rec")
    assert s["method"] == "temp"
    assert s["temperature"] == 0.6
    assert s["top_p"] == 0.95
    assert s["top_k"] == 20


def test_preset_topp():
    for p_str, p_val in [("0.7", 0.7), ("0.9", 0.9), ("0.95", 0.95)]:
        s = _preset_arm_spec(f"topp{p_str}")
        assert s["method"] == "temp"
        assert s["temperature"] == 1.0
        assert s["top_p"] == p_val
        assert s["top_k"] == 0


def test_preset_temp():
    s = _preset_arm_spec("temp")
    assert s["method"] == "temp"
    assert s["temperature"] == 1.0
    # top_p/top_k left at single-arm defaults
    assert s["top_p"] == 1.0 and s["top_k"] == 0


def test_preset_unknown_raises():
    with pytest.raises(ValueError, match="unknown preset arm"):
        _preset_arm_spec("not_a_preset")


def test_preset_malformed_renyi_k():
    with pytest.raises(ValueError, match="renyi_k arm must look like"):
        _preset_arm_spec("renyi_knotanumber")


def test_preset_malformed_topp():
    with pytest.raises(ValueError, match="topp arm must look like"):
        _preset_arm_spec("toppnotanumber")


# ---------------------------------------------------------------------------
# Raw arm spec parsing.
# ---------------------------------------------------------------------------


def test_arm_raw_minimal():
    s = _parse_arm_raw("name=foo,method=temp")
    assert s["name"] == "foo"
    assert s["method"] == "temp"
    # Defaults from single-arm runner
    assert s["temperature"] == 1.0
    assert s["top_p"] == 1.0
    assert s["top_k"] == 0
    assert s["renyi_k"] is None
    assert s["alpha"] is None


def test_arm_raw_full():
    s = _parse_arm_raw(
        "name=mix,method=temp,temperature=0.8,top_p=0.9,top_k=50,"
        "renyi_k=0.2,alpha=3.0"
    )
    assert s["name"] == "mix"
    assert s["method"] == "temp"
    assert s["temperature"] == 0.8
    assert s["top_p"] == 0.9
    assert s["top_k"] == 50
    assert s["renyi_k"] == 0.2
    assert s["alpha"] == 3.0


def test_arm_raw_with_whitespace():
    s = _parse_arm_raw(" name = x , method = pless , temperature = 1.5 ")
    assert s["name"] == "x"
    assert s["method"] == "pless"
    assert s["temperature"] == 1.5


def test_arm_raw_missing_name():
    with pytest.raises(ValueError, match="requires at least name= and method="):
        _parse_arm_raw("method=temp,temperature=1.0")


def test_arm_raw_missing_method():
    with pytest.raises(ValueError, match="requires at least name= and method="):
        _parse_arm_raw("name=foo,temperature=1.0")


def test_arm_raw_malformed_segment():
    with pytest.raises(ValueError, match="expected key=value"):
        _parse_arm_raw("name=foo,method,temperature=1.0")


# ---------------------------------------------------------------------------
# Output-path parity with single-arm runner.
# For each preset arm, _arm_output_path(arm, base_args) MUST equal what
# bench.apps.runner's _output_path(..., _method_key(single_arm_ns), temp) would
# produce for the equivalent single-arm invocation.
# ---------------------------------------------------------------------------


def _single_arm_ns(method, temperature, top_p=1.0, top_k=0,
                   renyi_k=None, alpha=None, enable_thinking=True):
    """Build the Namespace that _method_key sees on a single-arm invocation."""
    return Namespace(
        method=method, temperature=temperature,
        top_p=top_p, top_k=top_k,
        renyi_k=renyi_k, alpha=alpha,
        repetition_penalty=1.0, post_temperature=None,
        sampler_think=None, sampler_code=None,
        temp_think=None, temp_code=None,
        enable_thinking=enable_thinking,
    )


def _base_args(tmp_path):
    return Namespace(
        model="Qwen/Qwen3-8B", source="CODEFORCES", difficulty="interview",
        results_dir=str(tmp_path),
        enable_thinking=True,
    )


@pytest.mark.parametrize("arm_name,method,temp,top_p,top_k,renyi_k", [
    ("pless",        "pless",       1.0, 1.0, 0,  None),
    ("pless_norm",   "pless_norm",  1.0, 1.0, 0,  None),
    ("renyi_k0.1",   "pless_renyi", 1.0, 1.0, 0,  0.1),
    ("renyi_k0.075", "pless_renyi", 1.0, 1.0, 0,  0.075),
    ("renyi_k0.45",  "pless_renyi", 1.0, 1.0, 0,  0.45),
    ("rec",          "temp",        0.6, 0.95, 20, None),
    ("topp0.9",      "temp",        1.0, 0.9, 0,  None),
    ("topp0.95",     "temp",        1.0, 0.95, 0, None),
    ("temp",         "temp",        1.0, 1.0, 0,  None),
])
def test_arm_output_path_matches_single_arm(
    tmp_path, arm_name, method, temp, top_p, top_k, renyi_k,
):
    """For each preset arm, multi-arm's output path MUST equal single-arm's."""
    base = _base_args(tmp_path)
    arm = _preset_arm_spec(arm_name)
    multi_path = _arm_output_path(arm, base)

    single_ns = _single_arm_ns(method=method, temperature=temp,
                                top_p=top_p, top_k=top_k, renyi_k=renyi_k)
    single_path = _output_path(
        base.results_dir, base.model, base.source, base.difficulty,
        _method_key(single_ns), temp,
    )
    assert multi_path == single_path, (
        f"arm {arm_name!r}: multi={multi_path} vs single={single_path}"
    )


# ---------------------------------------------------------------------------
# _arm_to_namespace fields match what _method_key needs.
# ---------------------------------------------------------------------------


def test_arm_to_namespace_populates_method_key_fields(tmp_path):
    base = _base_args(tmp_path)
    arm = _preset_arm_spec("renyi_k0.1")
    ns = _arm_to_namespace(arm, base)
    # _method_key reads these:
    key = _method_key(ns)
    assert "pless_renyi" in key
    assert "k0.1" in key
    assert "think" in key  # enable_thinking is True
