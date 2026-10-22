"""The multi-arm vs single-arm parity script must also cover the DeepSeek `rec` raw spec."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parity():
    spec = importlib.util.spec_from_file_location(
        "compare_apps_arm_config_parity", REPO_ROOT / "scripts" / "compare_apps_arm_config_parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arm_raw_cases_include_deepseek_rec_without_top_k():
    mod = _parity()
    cases = dict(mod.ARM_RAW_EQUIVALENT_CLI)
    flags = cases["name=rec,method=temp,temperature=0.6,top_p=0.95"]
    assert "--top-k" not in flags and "--top-p" in flags and "0.95" in flags


def test_all_parity_checks_pass_for_both_thinking_settings():
    mod = _parity()
    parser = mod._build_argparser()
    for cli in (mod.BASE_ARGS_CLI, mod.BASE_ARGS_CLI + ["--enable-thinking"]):
        assert mod._run_all_checks(parser.parse_args(cli), "x") is False


def test_parity_check_fails_when_a_raw_spec_disagrees_with_the_runner_flags(monkeypatch):
    """Negative path: a raw spec with top_k=20 paired with CLI flags that have no --top-k must be reported."""
    mod = _parity()
    monkeypatch.setattr(mod, "ARM_RAW_EQUIVALENT_CLI", [
        ("name=rec,method=temp,temperature=0.6,top_p=0.95,top_k=20",
         ["--method", "temp", "--temperature", "0.6", "--top-p", "0.95"]),
    ])
    parser = mod._build_argparser()
    assert mod._run_all_checks(parser.parse_args(mod.BASE_ARGS_CLI), "x") is True


def test_parity_check_fails_when_the_temperature_disagrees(monkeypatch):
    mod = _parity()
    monkeypatch.setattr(mod, "ARM_RAW_EQUIVALENT_CLI", [
        ("name=rec,method=temp,temperature=0.7,top_p=0.95",
         ["--method", "temp", "--temperature", "0.6", "--top-p", "0.95"]),
    ])
    parser = mod._build_argparser()
    assert mod._run_all_checks(parser.parse_args(mod.BASE_ARGS_CLI), "x") is True
