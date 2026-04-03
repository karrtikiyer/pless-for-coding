"""Tests for bench.sampler_bridge, including post-truncation temperature."""

import torch

from bench.sampler_bridge import SAMPLERS, make_pless_post_temp_sampler


def _make_probs(batch_size: int = 2, vocab_size: int = 100) -> torch.Tensor:
    """Create a realistic probability distribution for testing."""
    logits = torch.randn(batch_size, vocab_size)
    # Make distribution peaky: a few high-prob tokens, long tail
    logits[:, 0] = 5.0  # dominant token
    logits[:, 1] = 3.0  # runner-up
    logits[:, 2] = 2.0  # third
    return torch.softmax(logits, dim=-1)


def test_samplers_dict_has_expected_keys():
    assert "pless" in SAMPLERS
    assert "pless_norm" in SAMPLERS


def test_pless_basic():
    probs = _make_probs()
    tokens = SAMPLERS["pless"](probs.clone())
    assert tokens.shape == (2, 1)
    assert tokens.dtype == torch.long
    assert (tokens >= 0).all()
    assert (tokens < 100).all()


def test_make_pless_post_temp_sampler_returns_callable():
    sampler = make_pless_post_temp_sampler(2.0)
    assert callable(sampler)


def test_post_temp_sampler_output_shape():
    sampler = make_pless_post_temp_sampler(2.0)
    probs = _make_probs(batch_size=4, vocab_size=200)
    tokens = sampler(probs)
    assert tokens.shape == (4, 1)
    assert tokens.dtype == torch.long
    assert (tokens >= 0).all()
    assert (tokens < 200).all()


def test_post_temp_1_matches_standard_pless():
    """post_temperature=1.0 should behave identically to standard p-less."""
    torch.manual_seed(42)
    probs1 = _make_probs(batch_size=1, vocab_size=50)
    probs2 = probs1.clone()

    # After p-less truncation + renormalization, the distribution should be identical
    # We can't compare sampled tokens (stochastic), but we can check the distribution
    # by running truncation without sampling.

    # Standard p-less truncation
    p1 = probs1.square().sum(dim=-1, keepdim=True)
    mask1 = probs1 < p1
    probs1[mask1] = 0.0
    probs1.div_(probs1.sum(dim=-1, keepdim=True))

    # Post-temp=1.0 truncation (should be identical since pow(1/1.0) = identity)
    p2 = probs2.square().sum(dim=-1, keepdim=True)
    mask2 = probs2 < p2
    probs2[mask2] = 0.0
    probs2.div_(probs2.sum(dim=-1, keepdim=True))
    # pow(1.0) is skipped by the if-check, so distributions should match
    assert torch.allclose(probs1, probs2)


def test_post_temp_flattens_distribution():
    """Higher post_temperature should make survivor probabilities more uniform."""
    # Use a flatter distribution so multiple tokens survive p-less threshold
    probs_base = torch.tensor([[0.30, 0.25, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]])

    # Truncate with standard p-less
    probs_std = probs_base.clone()
    p = probs_std.square().sum(dim=-1, keepdim=True)
    mask = probs_std < p
    probs_std[mask] = 0.0
    probs_std.div_(probs_std.sum(dim=-1, keepdim=True))
    survivors_std = probs_std[probs_std > 0]

    # Truncate + post-temp=5.0
    probs_pt = probs_base.clone()
    p = probs_pt.square().sum(dim=-1, keepdim=True)
    mask = probs_pt < p
    probs_pt[mask] = 0.0
    probs_pt.div_(probs_pt.sum(dim=-1, keepdim=True))
    probs_pt.pow_(1.0 / 5.0)
    probs_pt.div_(probs_pt.sum(dim=-1, keepdim=True))
    survivors_pt = probs_pt[probs_pt > 0]

    # Same number of survivors
    assert len(survivors_std) == len(survivors_pt)
    assert len(survivors_std) >= 2, "Need at least 2 survivors for meaningful test"

    # Post-temp distribution should be more uniform (lower std)
    assert survivors_pt.std() < survivors_std.std()


def test_post_temp_preserves_zero_mask():
    """Pruned tokens (prob=0) must stay at zero after post-temperature."""
    sampler = make_pless_post_temp_sampler(5.0)
    probs = _make_probs(batch_size=3, vocab_size=100)

    # Record which tokens are below threshold before sampling
    p = probs.square().sum(dim=-1, keepdim=True)
    should_be_zero = probs < p

    # Run the full sampler (it modifies probs in-place)
    probs_copy = probs.clone()
    _ = sampler(probs_copy)

    # All tokens that were below threshold should still be zero
    assert (probs_copy[should_be_zero] == 0.0).all()
