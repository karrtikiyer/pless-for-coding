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
    # Default n_samples must be 1 (preserves greedy / single-sample
    # behavior of the original probe — back-compat with results captured
    # under the previous default).
    assert ns.n_samples == 1


def test_runner_argparse_accepts_n_samples():
    """--n-samples must accept N > 1 for stochastic-sampling cells."""
    from bench.entropy_probe.runner import parse_args
    ns = parse_args([
        "--model", "fake/model",
        "--dataset", "gsm8k",
        "--n-samples", "3",
    ])
    assert ns.n_samples == 3


def test_run_one_problem_returns_list_of_n_samples():
    """run_one_problem(..., n_samples=N) must produce N records, each
    with a distinct sample_idx. Uses mocked model/tokenizer so no GPU
    or HF download is required."""
    from unittest.mock import MagicMock, patch
    import torch
    from bench.entropy_probe.runner import run_one_problem
    from bench.entropy_probe.datasets import EntropyProbeProblem

    # Mock tokenizer that produces a deterministic short prompt.
    fake_tok = MagicMock()
    fake_tok.apply_chat_template.return_value = "PROMPT"
    fake_tok.return_value = MagicMock(
        input_ids=torch.zeros((1, 4), dtype=torch.long),
    )
    # Make `to(...)` return self so generate sees the tensor.
    fake_tok.return_value.to.return_value = fake_tok.return_value
    fake_tok.decode.return_value = "COMPLETION"
    fake_tok.pad_token_id = 0
    fake_tok.eos_token_id = 0

    fake_model = MagicMock()
    fake_model.device = "cpu"
    fake_model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)

    fake_problem = EntropyProbeProblem(task_id="x", problem="q", reference=None)

    with patch("bench.entropy_probe.runner.teacher_forced_entropy",
               return_value=[0.1, 0.2, 0.3, 0.4]):
        recs = run_one_problem(
            fake_model, fake_tok, fake_problem,
            dataset="gsm8k", max_new_tokens=8, n_samples=3,
        )

    assert len(recs) == 3
    assert [r["sample_idx"] for r in recs] == [0, 1, 2]
    assert all(r["task_id"] == "x" for r in recs)
    # Verify do_sample=True was selected for n>1
    sample_calls = [c for c in fake_model.generate.call_args_list]
    for call in sample_calls:
        assert call.kwargs["do_sample"] is True, (
            "n_samples>1 must use multinomial sampling, not greedy"
        )


def test_run_one_problem_n_eq_1_uses_greedy():
    """n_samples=1 must use deterministic greedy decode for back-compat."""
    from unittest.mock import MagicMock, patch
    import torch
    from bench.entropy_probe.runner import run_one_problem
    from bench.entropy_probe.datasets import EntropyProbeProblem

    fake_tok = MagicMock()
    fake_tok.apply_chat_template.return_value = "PROMPT"
    fake_tok.return_value = MagicMock(
        input_ids=torch.zeros((1, 4), dtype=torch.long),
    )
    fake_tok.return_value.to.return_value = fake_tok.return_value
    fake_tok.decode.return_value = "COMPLETION"
    fake_tok.pad_token_id = 0
    fake_tok.eos_token_id = 0

    fake_model = MagicMock()
    fake_model.device = "cpu"
    fake_model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)

    fake_problem = EntropyProbeProblem(task_id="x", problem="q", reference=None)

    with patch("bench.entropy_probe.runner.teacher_forced_entropy",
               return_value=[0.1, 0.2]):
        recs = run_one_problem(
            fake_model, fake_tok, fake_problem,
            dataset="gsm8k", max_new_tokens=4, n_samples=1,
        )

    assert len(recs) == 1
    assert recs[0]["sample_idx"] == 0
    call = fake_model.generate.call_args
    assert call.kwargs["do_sample"] is False, (
        "n_samples=1 must use greedy for deterministic / back-compat behavior"
    )


# ─── pless_alpha CLI plumbing (new in Option C) ─────────────────────────────

def test_parse_args_requires_alpha_when_sampler_is_pless_alpha():
    """--sampler=pless_alpha demands --alpha; argparse should reject."""
    from bench.entropy_probe.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args([
            "--model", "fake/model",
            "--dataset", "gsm8k",
            "--sampler", "pless_alpha",
        ])
    ns = parse_args([
        "--model", "fake/model",
        "--dataset", "gsm8k",
        "--sampler", "pless_alpha",
        "--alpha", "2.0",
    ])
    assert ns.sampler == "pless_alpha"
    assert ns.alpha == 2.0


def test_parse_args_rejects_alpha_when_sampler_is_not_pless_alpha():
    """Passing --alpha with --sampler=multinomial is a misconfig."""
    from bench.entropy_probe.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args([
            "--model", "fake/model",
            "--dataset", "gsm8k",
            "--sampler", "multinomial",
            "--alpha", "2.0",
        ])


def test_parse_args_default_sampler_is_multinomial():
    """Backwards-compat: omitting --sampler preserves the original probe behavior."""
    from bench.entropy_probe.runner import parse_args
    ns = parse_args([
        "--model", "fake/model",
        "--dataset", "gsm8k",
    ])
    assert ns.sampler == "multinomial"
    assert ns.alpha is None


def test_sampler_tag_encodes_method_and_alpha():
    """Output subdir tag must include sampler + α for pless_alpha so the
    Option-C reruns don't clobber the existing multinomial-T=1.0 cells."""
    from bench.entropy_probe.runner import _sampler_tag, parse_args
    # multinomial @ T=1.0 with n>1 → "multinomial_t1.0"
    ns = parse_args([
        "--model", "m", "--dataset", "gsm8k",
        "--sampler", "multinomial", "--n-samples", "3",
    ])
    assert _sampler_tag(ns) == "multinomial_t1.0"
    # multinomial @ T=1.0 with n=1 → "greedy_t1.0"
    ns = parse_args([
        "--model", "m", "--dataset", "gsm8k",
        "--sampler", "multinomial", "--n-samples", "1",
    ])
    assert _sampler_tag(ns) == "greedy_t1.0"
    # pless_alpha @ α=2.0, T=1.0 → "pless_alpha_a2.0_t1.0"
    ns = parse_args([
        "--model", "m", "--dataset", "gsm8k",
        "--sampler", "pless_alpha", "--alpha", "2.0",
    ])
    assert _sampler_tag(ns) == "pless_alpha_a2.0_t1.0"
    # pless_alpha @ α=5.0 → "pless_alpha_a5.0_t1.0"
    ns = parse_args([
        "--model", "m", "--dataset", "gsm8k",
        "--sampler", "pless_alpha", "--alpha", "5.0",
    ])
    assert _sampler_tag(ns) == "pless_alpha_a5.0_t1.0"


def test_run_one_problem_pless_alpha_routes_through_generate_samples():
    """When sampler=pless_alpha, run_one_problem must call
    bench.generator.generate_samples (the production code path), NOT
    model.generate(). And the returned full_ids must reach
    teacher_forced_entropy. This is the methodology-fix invariant for
    Option C."""
    from unittest.mock import MagicMock, patch
    import torch
    from bench.entropy_probe.runner import run_one_problem
    from bench.entropy_probe.datasets import EntropyProbeProblem

    fake_tok = MagicMock()
    fake_tok.apply_chat_template.return_value = "PROMPT"
    fake_tok.pad_token_id = 0
    fake_tok.eos_token_id = 0

    fake_model = MagicMock()
    fake_model.device = "cpu"

    fake_problem = EntropyProbeProblem(task_id="x", problem="q", reference=None)

    # Fake generate_samples returning (texts, ids_list, prompt_len)
    fake_full_ids = [
        torch.zeros(10, dtype=torch.long),
        torch.zeros(11, dtype=torch.long),
    ]
    fake_gs_return = (["text1", "text2"], fake_full_ids, 4)

    with patch("bench.generator.generate_samples",
               return_value=fake_gs_return) as mock_gs, \
         patch("bench.entropy_probe.runner.teacher_forced_entropy",
               return_value=[0.5, 0.5, 0.5]) as mock_tfe:
        recs = run_one_problem(
            fake_model, fake_tok, fake_problem,
            dataset="gsm8k", max_new_tokens=8,
            n_samples=2, sampler="pless_alpha", alpha=2.0,
            temperature=1.0,
        )

    # Two records returned, with sample_idx 0 and 1.
    assert len(recs) == 2
    assert [r["sample_idx"] for r in recs] == [0, 1]
    # generate_samples called exactly once (batched N=2), with
    # return_token_ids=True so we got full_ids back.
    assert mock_gs.call_count == 1
    assert mock_gs.call_args.kwargs["return_token_ids"] is True
    assert mock_gs.call_args.kwargs["n_samples"] == 2
    # teacher_forced_entropy called twice (once per returned sample),
    # each with a (1, seq_len) tensor (unsqueeze(0) of the 1-D id list).
    assert mock_tfe.call_count == 2
    for call in mock_tfe.call_args_list:
        ids_arg = call.args[1]
        assert ids_arg.ndim == 2 and ids_arg.shape[0] == 1
    # And NO calls to model.generate() — the multinomial path is bypassed.
    assert fake_model.generate.call_count == 0
