"""Tests for the Rényi G_k sampler (rooted form) in HF + vLLM backends + APPS runner.

G_k = (Σpᵢ^k)^{1/(k-1)} = exp(-H_k). Distinct from pless_alpha's raw power sum
τ_α = Σpᵢ^α; the two coincide only at order 2 (both = Σpᵢ²). Mirrors
test_pless_alpha_vllm_and_apps.py, plus:
  - k=2 byte-equivalence to plain pless (HF and vLLM),
  - k=1 Shannon-limit continuity,
  - loosening monotonicity (lower k admits ≥ tokens).
All CPU/Mac-safe (no GPU / no vLLM engine needed — the mask fns are pure tensor ops).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
import torch

K_GRID = [2.0, 1.6, 0.8, 0.4, 0.2, 0.1, 0.05]


# ---------------------------------------------------------------------------
# References (mirror bench/sampler_bridge.py:make_pless_renyi_sampler)
# ---------------------------------------------------------------------------

def _renyi_threshold(probs: torch.Tensor, k: float) -> torch.Tensor:
    if k == 2.0:
        return probs.square().sum(dim=-1, keepdim=True)
    if k == 1.0:
        logp = torch.where(probs > 0, probs.log(), probs.new_zeros(()))
        return (probs * logp).sum(dim=-1, keepdim=True).exp()
    if k == 0.0:
        return probs.new_full((probs.size(0), 1), 1.0 / probs.size(-1))
    return probs.pow(k).sum(dim=-1, keepdim=True).pow(1.0 / (k - 1.0))


def _pless_renyi_decode_on_probs(probs: torch.Tensor, k: float) -> torch.Tensor:
    """Post-mask, post-renormalize distribution (mirrors the sampler minus multinomial)."""
    probs = probs.clone()
    threshold = _renyi_threshold(probs, k)
    mask = probs < threshold
    all_pruned = mask.all(dim=-1)
    if all_pruned.any():
        fallback_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
        mask[all_pruned] = mask[all_pruned].scatter(-1, fallback_idx, False)
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def _make_probs():
    torch.manual_seed(7)
    return torch.softmax(torch.randn(1, 200) * 2.0, dim=-1)


def _all_pruned_row(k=8):
    # Un-normalized row: Σpᵢ² = 0.25·k > 0.5 = max ⇒ every token pruned (forces the guard).
    return torch.full((1, k), 0.5)


# ---------------------------------------------------------------------------
# vLLM logit-mask parity with the HF reference, across the k grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", K_GRID + [1.0, 0.0, -1.0])
def test_renyi_logit_mask_matches_prob_mask(k: float):
    from bench.generator_vllm import _pless_renyi_mask_logits

    torch.manual_seed(int(k * 100) + 1)
    logits = torch.randn(8, 5000) * 1.5
    probs_ref = _pless_renyi_decode_on_probs(torch.softmax(logits, dim=-1), k)
    probs_v = torch.softmax(_pless_renyi_mask_logits(logits.clone(), k=k).float(), dim=-1)
    diff = (probs_ref - probs_v).abs().max().item()
    assert diff < 1e-5, f"G_k (k={k}) distributions diverged: max abs diff {diff}"


def test_renyi_at_k2_matches_pless_vllm():
    """k=2 must be byte-equivalent to plain pless (root exponent 1/(2-1)=1)."""
    from bench.generator_vllm import _pless_renyi_mask_logits, _pless_mask_logits

    torch.manual_seed(42)
    logits = torch.randn(4, 3000) * 2.0
    p_renyi = torch.softmax(_pless_renyi_mask_logits(logits.clone(), k=2.0).float(), dim=-1)
    p_pless = torch.softmax(_pless_mask_logits(logits.clone()).float(), dim=-1)
    diff = (p_renyi - p_pless).abs().max().item()
    assert diff < 1e-9, f"G_2 must byte-match plain pless; got diff {diff}"


def test_renyi_at_k2_matches_pless_hf_seeded():
    """HF make_pless_renyi_sampler(2) must draw identically to guarded pless on a fixed seed."""
    from bench.sampler_bridge import make_pless_renyi_sampler, make_guarded_pless_sampler

    probs = _make_probs()
    r = make_pless_renyi_sampler(2.0)
    p = make_guarded_pless_sampler()
    torch.manual_seed(123); t_renyi = r(probs.clone())
    torch.manual_seed(123); t_pless = p(probs.clone())
    assert torch.equal(t_renyi, t_pless)


# ---------------------------------------------------------------------------
# k=1 Shannon-limit continuity, loosening monotonicity, degenerate guard
# ---------------------------------------------------------------------------

def test_renyi_k1_shannon_limit_is_continuous():
    """The k=1 branch (exp(Σ pᵢ ln pᵢ)) must match the limit of G_k as k→1."""
    probs = _make_probs()
    g1 = _renyi_threshold(probs, 1.0)
    g_lo = _renyi_threshold(probs, 0.999)
    g_hi = _renyi_threshold(probs, 1.001)
    assert (g1 - g_lo).abs().max().item() < 1e-3
    assert (g1 - g_hi).abs().max().item() < 1e-3


def test_renyi_loosens_as_k_decreases():
    """Lower k ⇒ threshold no higher ⇒ at least as many tokens admitted."""
    probs = _make_probs()
    counts = []
    for k in [2.0, 0.8, 0.2, 0.05]:
        kept = int((_pless_renyi_decode_on_probs(probs, k) > 0).sum().item())
        counts.append(kept)
    assert counts == sorted(counts), f"admitted counts not monotone in k: {counts}"


def test_renyi_argmax_always_survives_and_no_nan():
    """G_k ≤ max pᵢ, so the argmax is never pruned; the degenerate row still returns
    a valid index (never NaN)."""
    from bench.sampler_bridge import make_pless_renyi_sampler

    # normal peaky dist: argmax (index of max prob) must remain nonzero after masking
    probs = _make_probs()
    for k in K_GRID:
        post = _pless_renyi_decode_on_probs(probs, k)
        assert post[0, probs.argmax()] > 0, f"argmax pruned at k={k}"
    # degenerate all-pruned (unnormalized) row with a distinct max at index 7:
    # at k=2, Σpᵢ²=2.11 > max=0.6 ⇒ every token pruned ⇒ guard restores the argmax.
    row = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]])
    tok = make_pless_renyi_sampler(2.0)(row.clone())
    assert tok.shape == (1, 1) and tok.dtype == torch.long
    assert int(tok.item()) == 7  # guard restored the argmax, no NaN/crash
    # a looser k on the same row is not all-pruned; must still return a valid index.
    tok2 = make_pless_renyi_sampler(0.2)(row.clone())
    assert 0 <= int(tok2.item()) < 8


def test_renyi_accepts_k0_and_negative():
    """k=0 (→1/v uniform threshold) and negative k are accepted (match the author's
    p_moment_decode reference); no ValueError, no NaN."""
    from bench.sampler_bridge import make_pless_renyi_sampler

    probs = _make_probs()
    # k=0: threshold is exactly 1/v
    thr0 = _renyi_threshold(probs, 0.0)
    assert abs(thr0[0, 0].item() - 1.0 / probs.size(-1)) < 1e-6  # float32 tolerance
    tok0 = make_pless_renyi_sampler(0.0)(probs.clone())
    assert tok0.shape == (1, 1) and tok0.dtype == torch.long
    assert 0 <= int(tok0.item()) < probs.size(-1)
    # negative order: accepted, loosens further (admits ≥ tokens than k=0.2), no crash
    n_neg = int((_pless_renyi_decode_on_probs(probs, -1.0) > 0).sum().item())
    n_02 = int((_pless_renyi_decode_on_probs(probs, 0.2) > 0).sum().item())
    assert n_neg >= n_02
    tok_neg = make_pless_renyi_sampler(-1.0)(probs.clone())
    assert 0 <= int(tok_neg.item()) < probs.size(-1)


# ---------------------------------------------------------------------------
# CLI: APPS runner accepts pless_renyi + --renyi-k
# ---------------------------------------------------------------------------

_BASE_ARGV = [
    "bench.apps", "--model", "Qwen/Qwen3-8B",
    "--source", "ATCODER", "--difficulty", "interview",
]


def test_apps_runner_accepts_pless_renyi():
    import bench.apps.runner as m
    with patch.object(sys, "argv", _BASE_ARGV + ["--method", "pless_renyi", "--renyi-k", "0.4"]):
        args = m.parse_args()
    assert args.method == "pless_renyi" and args.renyi_k == 0.4


def test_apps_runner_requires_renyi_k():
    import bench.apps.runner as m
    with patch.object(sys, "argv", _BASE_ARGV + ["--method", "pless_renyi"]):
        with pytest.raises(SystemExit):
            m.parse_args()


def test_apps_runner_rejects_renyi_k_without_method():
    import bench.apps.runner as m
    with patch.object(sys, "argv", _BASE_ARGV + ["--method", "pless", "--renyi-k", "0.4"]):
        with pytest.raises(SystemExit):
            m.parse_args()


def test_apps_runner_method_key_encodes_k():
    import bench.apps.runner as m
    with patch.object(sys, "argv", _BASE_ARGV + ["--method", "pless_renyi", "--renyi-k", "0.4"]):
        args = m.parse_args()
    key = m._method_key(args)
    assert "pless_renyi" in key and "_k0.4" in key, f"method key missing k suffix: {key!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
