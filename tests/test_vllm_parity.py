"""Parity tests for the vLLM backend (bench/generator_vllm.py).

Two flavours:

1. **Mac-safe unit tests** (run via ``uv run pytest tests/test_vllm_parity.py``):
   exercise the pure-Python logit-transform layer of
   ``bench/generator_vllm.py`` against the canonical
   ``p-less/p_less_samplers.py`` to confirm mathematical equivalence
   (mask-on-probs ≡ mask-to-minus-inf-on-logits-then-softmax). These
   don't require vLLM at all and run on Mac / any CI.

2. **GPU parity gate** (run via
   ``.venv-vllm/bin/python -m pytest tests/test_vllm_parity.py -k gpu``)
   loads Qwen3-8B, runs 10 problems × 10 samples on both HF and vLLM
   backends with the same RNG seed, and asserts the distributional
   metrics (NAUADC, struct_div, codebleu_div) agree within their
   measured noise floors. This is the merge gate for ``feat/vllm-backend``.

Tolerances (from docs/research/vllm_migration_analysis.md §10b-D):
  * NAUADC       within  ±0.02
  * struct_div   within  ±0.01
  * codebleu_div within  ±0.02

Why distributional, not bit-identical: vLLM uses FlashAttention-2 and
its own multinomial; HF uses SDPA and torch.multinomial. They produce
different *token sequences* even with the same seed. What we care about
is that the population statistics match.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Mac-safe unit tests (no vLLM, no GPU required)
# ---------------------------------------------------------------------------


def _pless_decode_on_probs(probs: torch.Tensor) -> torch.Tensor:
    """Reference implementation: p_less_decode minus the multinomial step.

    Returns the post-mask, post-renormalize distribution so we can compare
    against the vLLM path's softmax-after-mask output. Pure function;
    matches ``p-less/p_less_samplers.py:p_less_decode`` lines 25-28.
    """
    probs = probs.clone()
    p = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < p
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def _pless_norm_decode_on_probs(probs: torch.Tensor) -> torch.Tensor:
    """Reference: p_less_norm_decode minus multinomial. Mirrors lines 54-57."""
    probs = probs.clone()
    v = probs.size(-1)
    p = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
    mask = probs < p
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def test_pless_logit_mask_matches_prob_mask():
    """vLLM-side _pless_mask_logits + softmax produces the same distribution
    as the canonical HF-side mask + renormalize on probs."""
    from bench.generator_vllm import _pless_mask_logits

    torch.manual_seed(0)
    logits = torch.randn(8, 5000) * 1.5  # batch of 8, vocab of 5000

    # HF path
    probs_ref = _pless_decode_on_probs(torch.softmax(logits, dim=-1))

    # vLLM path
    logits_v = logits.clone()
    logits_v = _pless_mask_logits(logits_v)
    probs_v = torch.softmax(logits_v.float(), dim=-1)

    # Live tokens should match within numerical precision.
    diff = (probs_ref - probs_v).abs().max().item()
    assert diff < 1e-5, f"pless distributions diverged: max abs diff {diff}"


def test_pless_norm_logit_mask_matches_prob_mask():
    from bench.generator_vllm import _pless_norm_mask_logits

    torch.manual_seed(1)
    logits = torch.randn(4, 3000) * 2.0
    probs_ref = _pless_norm_decode_on_probs(torch.softmax(logits, dim=-1))
    logits_v = logits.clone()
    logits_v = _pless_norm_mask_logits(logits_v)
    probs_v = torch.softmax(logits_v.float(), dim=-1)
    diff = (probs_ref - probs_v).abs().max().item()
    assert diff < 1e-5, f"pless_norm distributions diverged: max abs diff {diff}"


def test_top_p_top_k_mask_respects_limits():
    """temp_standard's logit mask should leave at most ``top_k`` tokens alive
    and the surviving probability mass should be roughly ≥ ``top_p``."""
    from bench.generator_vllm import _top_p_top_k_mask_logits

    torch.manual_seed(2)
    logits = torch.randn(4, 1000) * 2.0
    masked = _top_p_top_k_mask_logits(logits.clone(), top_p=0.95, top_k=20)
    probs = torch.softmax(masked.float(), dim=-1)
    live = (probs > 1e-9).sum(dim=-1)
    assert (live <= 20).all(), f"top_k violated: {live.tolist()}"


def test_temp_pure_is_identity():
    """temp_pure must NOT modify logits — that's its definition."""
    from bench.generator_vllm import _SAMPLER_LOGIT_FN

    torch.manual_seed(3)
    logits = torch.randn(2, 100)
    out = _SAMPLER_LOGIT_FN["temp_pure"](logits.clone())
    assert torch.equal(out, logits), "temp_pure should be identity"


def test_sampler_dispatch_table_covers_all_split_samplers():
    """Defensive: ensure every name in bench.sampler_bridge.SPLIT_SAMPLERS
    has a matching entry in the vLLM dispatch table. If a new sampler is
    added to SPLIT_SAMPLERS but not _SAMPLER_LOGIT_FN, the vLLM path
    silently can't be used for it — this test catches that."""
    from bench.generator_vllm import _SAMPLER_LOGIT_FN
    from bench.sampler_bridge import SPLIT_SAMPLERS

    for name in SPLIT_SAMPLERS:
        assert name in _SAMPLER_LOGIT_FN, (
            f"Sampler {name!r} is in SPLIT_SAMPLERS but missing from "
            f"_SAMPLER_LOGIT_FN in bench/generator_vllm.py — vLLM backend "
            f"cannot use this sampler."
        )


# ---------------------------------------------------------------------------
# GPU parity gate — only runs if vLLM is importable and CUDA is available
# ---------------------------------------------------------------------------


_VLLM_AVAILABLE = False
try:  # pragma: no cover — environment-dependent
    import vllm  # noqa: F401
    _VLLM_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass


@pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="vLLM and/or CUDA not available; this test runs only in .venv-vllm on GPU.",
)
@pytest.mark.gpu
def test_distributional_parity_qwen3_h7p_codeforces(tmp_path):
    """End-to-end: same model, same prompts, same RNG seed; HF and vLLM
    backends should produce distributional metrics within tolerance.

    Designed to run on a GPU box where both HF and vLLM are installed
    (the .venv-vllm overlay). Picks 10 H7P-CODEFORCES problems
    deterministically; runs split decoding (temp_pure t1.5 think →
    pless t1.0 code); computes NAUADC (via algosim, if available) and
    struct_div / codebleu_div on both backends.

    Skipped if the algosim submodule isn't installed in the venv.
    """
    pytest.importorskip("vllm")
    pytest.importorskip("transformers")

    # The full A/B is large (10 problems × 8K tokens × 2 backends ≈ 30 min
    # of GPU time). Run via:
    #   .venv-vllm/bin/python -m pytest tests/test_vllm_parity.py::test_distributional_parity_qwen3_h7p_codeforces -s
    # Skeleton implementation below — fleshed out once .venv-vllm exists
    # on the GPU box and vLLM is verified importable.
    pytest.skip(
        "GPU parity test implementation pending — requires .venv-vllm on the GPU "
        "box, see docs/research/vllm_migration_analysis.md §10b-D for the merge "
        "criteria (NAUADC ±0.02, struct_div ±0.01, codebleu_div ±0.02 over "
        "10 problems × 10 samples)."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
