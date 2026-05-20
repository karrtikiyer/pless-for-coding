"""Tests for pless_alpha wiring in vLLM backend + APPS runner.

Two flavours:

1. **Mac-safe pure-Python tests**: vLLM-side logit-mask reference parity with
   HF-side prob-mask (mirrors test_vllm_parity.py for pless / pless_norm).
2. **CLI parsing tests**: APPS runner accepts --method pless_alpha --alpha N
   and rejects misuse.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Reference: pure-Python pless_alpha on probs (mirrors sampler_bridge.py)
# ---------------------------------------------------------------------------


def _pless_alpha_decode_on_probs(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Reference: post-mask, post-renormalize distribution for pless_alpha.

    Mirrors ``bench/sampler_bridge.py:make_pless_alpha_sampler`` minus the
    multinomial step. Used to check vLLM-side mask-on-logits + softmax
    produces the same distribution as HF-side mask-on-probs + renormalize.
    """
    probs = probs.clone()
    if alpha == 2.0:
        threshold = probs.square().sum(dim=-1, keepdim=True)
    else:
        threshold = probs.pow(alpha).sum(dim=-1, keepdim=True)
    mask = probs < threshold
    all_pruned = mask.all(dim=-1)
    if all_pruned.any():
        fallback_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
        mask[all_pruned] = mask[all_pruned].scatter(-1, fallback_idx, False)
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


# ---------------------------------------------------------------------------
# vLLM logit-mask parity for pless_alpha at α ∈ {2.0, 2.5, 3.0, 5.0}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [2.0, 2.5, 3.0, 5.0])
def test_pless_alpha_logit_mask_matches_prob_mask(alpha: float):
    """vLLM-side _pless_alpha_mask_logits + softmax produces the same
    distribution as the canonical HF-side mask-on-probs."""
    from bench.generator_vllm import _pless_alpha_mask_logits

    torch.manual_seed(int(alpha * 10))
    logits = torch.randn(8, 5000) * 1.5

    # HF reference
    probs_ref = _pless_alpha_decode_on_probs(torch.softmax(logits, dim=-1), alpha)

    # vLLM path
    logits_v = logits.clone()
    logits_v = _pless_alpha_mask_logits(logits_v, alpha=alpha)
    probs_v = torch.softmax(logits_v.float(), dim=-1)

    diff = (probs_ref - probs_v).abs().max().item()
    assert diff < 1e-5, (
        f"pless_alpha (α={alpha}) distributions diverged: max abs diff {diff}"
    )


def test_pless_alpha_at_alpha2_matches_pless():
    """At α=2.0, pless_alpha must produce byte-equivalent distributions to
    plain pless. Sanity check for the regression that motivated the α=2
    fast-path in sampler_bridge.py."""
    from bench.generator_vllm import _pless_alpha_mask_logits, _pless_mask_logits

    torch.manual_seed(42)
    logits = torch.randn(4, 3000) * 2.0

    masked_alpha = _pless_alpha_mask_logits(logits.clone(), alpha=2.0)
    masked_pless = _pless_mask_logits(logits.clone())

    probs_alpha = torch.softmax(masked_alpha.float(), dim=-1)
    probs_pless = torch.softmax(masked_pless.float(), dim=-1)
    diff = (probs_alpha - probs_pless).abs().max().item()
    assert diff < 1e-9, (
        f"pless_alpha (α=2) must byte-match plain pless; got diff {diff}"
    )


# ---------------------------------------------------------------------------
# CLI: APPS runner accepts pless_alpha + --alpha
# ---------------------------------------------------------------------------


def test_apps_runner_accepts_pless_alpha():
    """`bench/apps/runner.py:parse_args` must accept --method pless_alpha
    --alpha 5.0 without erroring."""
    import bench.apps.runner as m

    test_argv = [
        "bench.apps",
        "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--source", "CODEFORCES",
        "--difficulty", "competition",
        "--method", "pless_alpha",
        "--alpha", "5.0",
    ]
    with patch.object(sys, "argv", test_argv):
        args = m.parse_args()
    assert args.method == "pless_alpha"
    assert args.alpha == 5.0


def test_apps_runner_requires_alpha_for_pless_alpha():
    """`--method pless_alpha` without `--alpha` must error."""
    import bench.apps.runner as m

    test_argv = [
        "bench.apps",
        "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--source", "CODEFORCES",
        "--difficulty", "competition",
        "--method", "pless_alpha",
    ]
    with patch.object(sys, "argv", test_argv):
        with pytest.raises(SystemExit):
            m.parse_args()


def test_apps_runner_rejects_alpha_without_pless_alpha():
    """`--alpha 5.0 --method pless` (no _alpha) must error to avoid silent
    misuse where --alpha is passed but ignored."""
    import bench.apps.runner as m

    test_argv = [
        "bench.apps",
        "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--source", "CODEFORCES",
        "--difficulty", "competition",
        "--method", "pless",
        "--alpha", "5.0",
    ]
    with patch.object(sys, "argv", test_argv):
        with pytest.raises(SystemExit):
            m.parse_args()


def test_apps_runner_method_key_encodes_alpha():
    """`_method_key` for pless_alpha must include `_a{α}` suffix."""
    import bench.apps.runner as m

    test_argv = [
        "bench.apps",
        "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--source", "CODEFORCES",
        "--difficulty", "competition",
        "--method", "pless_alpha",
        "--alpha", "2.5",
    ]
    with patch.object(sys, "argv", test_argv):
        args = m.parse_args()
    key = m._method_key(args)
    assert "pless_alpha" in key
    assert "_a2.5" in key, f"Method key missing α suffix: {key!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
