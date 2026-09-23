"""vLLM logit-processor for top-nσ sampling (Tang et al., arXiv:2411.07641).

Threshold in *logit* space: keep tokens whose logit is within `n · σ` of the
top logit. Default n=1.0 (author-recommended). Temperature-invariant because
the σ term scales the same way logits do under temperature scaling.

Added as a comparison baseline for our hyperparameter-free variants —
top-nσ is the current-SoTA single-knob decoder for reasoning benchmarks
(ACL 2025 Oral). Registers into ``bench.generator_vllm._SAMPLER_LOGIT_FN``
via :func:`register` so the runner can dispatch it by name.
"""
from __future__ import annotations

import torch


def _top_nsigma_mask_logits(logits: torch.Tensor, n: float = 1.0) -> torch.Tensor:
    """In-place: zero out (set to -inf) tokens whose logit is more than
    ``n * σ`` below the top logit.

    ``logits`` has shape ``(N, V)``. Handles ``-inf`` tokens by ignoring
    them in the std computation (masked with ``finite_mask``).
    """
    # ``logits`` may contain -inf from any prior filter; use only finite
    # entries to compute μ and σ (otherwise σ blows up).
    finite_mask = torch.isfinite(logits)
    # Row-wise μ and σ over finite entries. Use masked_select via clone-and-mask.
    logits_for_stats = logits.clone()
    logits_for_stats[~finite_mask] = float("nan")
    # nanmean / nanstd (torch has nanmean; nanstd needs manual construction)
    mean = torch.nanmean(logits_for_stats, dim=-1, keepdim=True)
    var = torch.nanmean(
        (logits_for_stats - mean).pow(2), dim=-1, keepdim=True
    )
    std = var.clamp(min=1e-12).sqrt()
    max_logit, _ = logits.max(dim=-1, keepdim=True)
    threshold = max_logit - n * std
    mask = logits < threshold
    # Never mask the argmax itself (safety — should already survive threshold).
    argmax_idx = logits.argmax(dim=-1, keepdim=True)
    mask.scatter_(-1, argmax_idx, False)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def top_nsigma_1_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """top-nσ at n=1.0 — the paper's default (§4.2, §5.1)."""
    return _top_nsigma_mask_logits(logits, n=1.0)


def top_nsigma_0_5_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """top-nσ at n=0.5 — lower end of the paper's flat-performance band (§5.4)."""
    return _top_nsigma_mask_logits(logits, n=0.5)


def top_nsigma_1_5_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """top-nσ at n=1.5 — upper end of the flat band before degradation (§5.4)."""
    return _top_nsigma_mask_logits(logits, n=1.5)


def register() -> None:
    """Splice top-nσ variants into ``bench.generator_vllm._SAMPLER_LOGIT_FN``.

    Registers three points from the paper's sensitivity sweep (§5.4):
      - ``top_nsigma``     → n=1.0 (paper default)
      - ``top_nsigma_0_5`` → n=0.5 (lower flat-band endpoint)
      - ``top_nsigma_1_5`` → n=1.5 (upper flat-band endpoint)

    The paper (Tang et al. 2025, arXiv:2411.07641, ACL 2025) documents
    that n ∈ [0.3, ~1.0] gives essentially flat performance on GSM8K
    across T ∈ [0.5, 3.0]; n ≥ 2.0 degrades significantly. This 3-point
    sweep brackets the recommended range without entering the failure
    region.

    Important: the paper's Algorithm 1 filters *before* temperature
    scaling. Since our vLLM runner uses ``temperature=1.0`` at the
    SamplingParams level (temperature is handled per-phase inside the
    PlessSplitLogitsProcessor at T=1.0 for the head-to-head), the
    ordering equivalence holds and n=1.0 here means what the paper's
    n=1.0 means. If you run at T ≠ 1.0, the interpretation shifts.
    """
    from bench import generator_vllm as gv
    gv._SAMPLER_LOGIT_FN["top_nsigma"]     = top_nsigma_1_mask_logits
    gv._SAMPLER_LOGIT_FN["top_nsigma_0_5"] = top_nsigma_0_5_mask_logits
    gv._SAMPLER_LOGIT_FN["top_nsigma_1_5"] = top_nsigma_1_5_mask_logits
