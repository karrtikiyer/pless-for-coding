"""Smoke tests for the cross-domain entropy probe.

These tests do NOT load any model and do NOT hit the network. They
verify the small parts of the probe that have local logic:
  * dip-test classifies bimodal vs unimodal correctly
  * prompts contain the no-code constraint for non-code datasets
  * chat-template plumbing routes the right roles through

The actual model-running smoke is left to the shell driver (which the
user invokes on a GPU pod, not on CI/local machines).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def test_dip_test_strongly_multimodal():
    """Bimodal Gaussian mixture → reject unimodality with high confidence."""
    from bench.entropy_probe.analysis import compute_dip_test
    rng = np.random.default_rng(42)
    bimodal = np.concatenate([
        rng.normal(0.2, 0.15, 800),
        rng.normal(3.0, 0.30, 800),
    ])
    result = compute_dip_test(bimodal.tolist())
    assert result["p_value"] < 0.05, (
        f"expected bimodal rejection, got p={result['p_value']}"
    )
    assert result["interpretation"] in ("multimodal", "strongly_multimodal")
    assert result["n_samples"] == 1600


def test_dip_test_unimodal_gaussian():
    """Pure Gaussian → cannot reject unimodality."""
    from bench.entropy_probe.analysis import compute_dip_test
    rng = np.random.default_rng(123)
    unimodal = rng.normal(1.5, 0.5, 2000)
    result = compute_dip_test(unimodal.tolist())
    assert result["p_value"] > 0.10, (
        f"expected unimodal acceptance, got p={result['p_value']}"
    )
    assert result["interpretation"] == "consistent_with_unimodal"


def test_dip_test_too_few_samples():
    """Need at least 4 samples for the dip test to be meaningful."""
    from bench.entropy_probe.analysis import compute_dip_test
    result = compute_dip_test([1.0, 2.0])
    assert result.get("error") == "too_few_samples"
    assert result["n_samples"] == 2


def test_prompts_forbid_code_for_non_code_datasets():
    """A reviewer-confound guard: math prompts must explicitly forbid code,
    otherwise a coder-tuned model may emit programs and contaminate the
    entropy distribution with code structure."""
    from bench.entropy_probe.prompts import GSM8K_SYSTEM, MATH_SYSTEM
    for name, sys_msg in (("GSM8K", GSM8K_SYSTEM), ("MATH", MATH_SYSTEM)):
        assert "Do not write any code" in sys_msg, (
            f"{name} prompt must explicitly forbid code"
        )
        assert "code fence" in sys_msg or "code block" in sys_msg or \
               "markdown code" in sys_msg, (
            f"{name} prompt must mention code fences (so the model "
            "knows even ``` blocks are off-limits)"
        )


def test_prompts_mbpp_does_not_forbid_code():
    """The code baseline must, of course, allow code."""
    from bench.entropy_probe.prompts import MBPP_SYSTEM
    assert "Do not write any code" not in MBPP_SYSTEM
    assert "python" in MBPP_SYSTEM.lower() or "code" in MBPP_SYSTEM.lower()


def test_format_prompt_routes_through_chat_template():
    """format_prompt(...) must call tokenizer.apply_chat_template with
    a (system, user) pair and add_generation_prompt=True."""
    from bench.entropy_probe.prompts import format_prompt
    fake_tok = MagicMock()
    fake_tok.apply_chat_template.return_value = "FAKE_PROMPT_STRING"
    out = format_prompt("gsm8k", "What is 2+2?", fake_tok)
    assert out == "FAKE_PROMPT_STRING"

    fake_tok.apply_chat_template.assert_called_once()
    args, kwargs = fake_tok.apply_chat_template.call_args
    messages = args[0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "2+2" in messages[1]["content"]
    assert kwargs.get("tokenize") is False
    assert kwargs.get("add_generation_prompt") is True


def test_format_prompt_rejects_unknown_dataset():
    from bench.entropy_probe.prompts import format_prompt
    fake_tok = MagicMock()
    with pytest.raises(ValueError, match="Unknown dataset"):
        format_prompt("not_a_real_dataset", "blah", fake_tok)


def test_dataset_loaders_have_max_problems_param():
    """Each loader must accept max_problems for smoke runs."""
    import inspect
    from bench.entropy_probe.datasets import DATASETS
    for name, loader in DATASETS.items():
        sig = inspect.signature(loader)
        assert "max_problems" in sig.parameters, (
            f"{name} loader missing max_problems parameter"
        )


def test_entropy_probe_problem_dataclass():
    from bench.entropy_probe.datasets import EntropyProbeProblem
    p = EntropyProbeProblem(task_id="t", problem="q", reference=None)
    assert p.task_id == "t"
    assert p.problem == "q"
    assert p.reference is None


def test_runner_cli_help_runs_without_imports_failing():
    """Light integration: parse_args() with --help should not blow up
    on any import issue inside the runner."""
    from bench.entropy_probe.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--help"])


def test_runner_argparse_has_required_flags():
    """parse_args(...) must require --model and --dataset."""
    from bench.entropy_probe.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args([])  # no required args → error
    # Valid minimum:
    ns = parse_args([
        "--model", "fake/model",
        "--dataset", "gsm8k",
        "--max-problems", "1",
    ])
    assert ns.model == "fake/model"
    assert ns.dataset == "gsm8k"
    assert ns.max_problems == 1
