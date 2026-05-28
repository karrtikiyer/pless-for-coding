"""Tests for --log-entropy in bench/apps/runner.py.

Mirrors the GSM8K runner pattern (bench/gsm8k/runner.py:64-72 + 78-83).

Background: the entropy probe sidecar feeds the central figure
(``results/entropy_probe/_central_figure_v2/survival_vs_entropy.png``).
MBPP and GSM8K runners have ``--log-entropy``; APPS does not. Adding it
lets us extend the central figure to APPS for Deepseek-6.7B-Instruct
on CODEFORCES_introductory.

Constraints:
- ``--log-entropy`` only valid with samplers that route through
  ``generate_samples`` (the manual-decode HF path that exposes the
  ``entropy_log`` hook): pless / pless_alpha / pless_norm / split.
- Not valid with ``temp`` (uses ``model.generate`` which doesn't capture
  the raw softmax) or with ``--backend vllm`` (no entropy hook).
"""
from __future__ import annotations
import pytest


def test_apps_runner_accepts_log_entropy_flag():
    """argparser must accept --log-entropy (no error)."""
    from bench.apps.runner import _build_argparser
    p = _build_argparser()
    # Should parse without raising
    args = p.parse_args([
        "--model", "test-model",
        "--source", "CODEFORCES", "--difficulty", "introductory",
        "--method", "pless_alpha", "--alpha", "5.0",
        "--n-samples", "1", "--max-new-tokens", "16",
        "--log-entropy",
    ])
    assert args.log_entropy is True


def test_apps_runner_log_entropy_defaults_false():
    """Backward compat: --log-entropy off by default."""
    from bench.apps.runner import _build_argparser
    p = _build_argparser()
    args = p.parse_args([
        "--model", "test-model",
        "--source", "CODEFORCES", "--difficulty", "introductory",
        "--method", "pless_alpha", "--alpha", "5.0",
    ])
    assert args.log_entropy is False


def test_apps_runner_rejects_log_entropy_with_temp_method():
    """--log-entropy + --method temp must error.

    temp goes through generate_samples_standard (model.generate) which
    doesn't capture per-position softmax. Refuse rather than silently
    skip the sidecar.
    """
    import subprocess
    res = subprocess.run(
        ["uv", "run", "python", "-m", "bench.apps",
         "--model", "test-model",
         "--source", "CODEFORCES", "--difficulty", "introductory",
         "--method", "temp",
         "--n-samples", "1", "--max-new-tokens", "16",
         "--log-entropy",
         "--results-dir", "/tmp/nope"],
        capture_output=True, text=True,
    )
    assert res.returncode != 0, (
        f"--log-entropy + --method temp should error, got rc=0\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}"
    )
    combined = (res.stdout + res.stderr).lower()
    assert ("log-entropy" in combined or "log_entropy" in combined), (
        f"error message should mention --log-entropy, got: {combined[:500]}"
    )


def test_apps_runner_accepts_log_entropy_with_pless_alpha():
    """--log-entropy + --method pless_alpha must not raise at parse time."""
    from bench.apps.runner import _build_argparser
    p = _build_argparser()
    args = p.parse_args([
        "--model", "test-model",
        "--source", "CODEFORCES", "--difficulty", "introductory",
        "--method", "pless_alpha", "--alpha", "2.0",
        "--log-entropy",
    ])
    assert args.log_entropy is True
    assert args.method == "pless_alpha"
