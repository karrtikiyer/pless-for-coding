"""Effective-support top-k: rank-cut sibling of p-less.

Same hyperparameter-free signal as canonical p-less — the collision entropy
``Σpᵢ²`` — but used as a *rank* cut instead of a *probability threshold*:

    N_eff = 1 / Σpᵢ²           (perplexity of order 2)
    k     = max(2, round(N_eff))
    admit top-k tokens, renormalise, sample.

Why this variant is worth trying against the Qwen3-8B pless-α=2 failure list
(see ``results/pless_cot_efficiency_vllm/…/pless_a2_failures_solved_by_others.csv``):

Canonical p-less admits token ``i`` iff ``pᵢ ≥ Σpᵢ²``. When the model is
confident (e.g. mid-loop, generating the same token repeatedly), the
distribution peaks: ``max(pᵢ) → 1`` implies ``Σpᵢ² → 1``, so the threshold
approaches the top-1 probability and only the argmax survives.  Once the
sampler is effectively deterministic the loop is self-perpetuating.

Effective-k breaks that collapse by construction:

* peaked distribution → ``N_eff ≈ 1`` → ``k = max(2, 1) = 2`` → always
  a second option, so the sampler cannot go deterministic;
* flat distribution → ``N_eff`` large → many survivors, mirroring the
  ``Σpᵢ² → 0`` limit of canonical p-less.

No new hyperparameter; no reparameterization of the threshold (this is not
τ_α = Σpᵢ^α nor G_k = (Σpᵢ^k)^{1/(k-1)}); different *mechanism* (rank vs.
probability), same underlying statistic.
"""
from __future__ import annotations

import torch


def p_less_effk_decode(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token per row using effective-support top-k.

    ``probs``: shape (batch, vocab); modified in-place (same convention as
    the upstream ``p_less_decode``). Returns shape (batch, 1) long tensor
    of sampled token ids.
    """
    coll = probs.square().sum(dim=-1, keepdim=True)
    # Guard against extreme peaks where Σpᵢ² > 1 in float32 (parallel
    # reduction rounding); clamp so 1/coll is finite.
    coll = coll.clamp(min=1.0 / probs.size(-1), max=1.0)
    n_eff = (1.0 / coll).round().to(torch.long)          # (batch, 1)
    k_row = n_eff.clamp(min=2, max=probs.size(-1))       # (batch, 1)
    k_max = int(k_row.max().item())

    topk_vals, _ = probs.topk(k_max, dim=-1)             # (batch, k_max)
    # Per-row threshold = the k_row-th largest prob (1-indexed from top).
    gather_idx = (k_row - 1).clamp(min=0, max=k_max - 1)  # (batch, 1)
    threshold = topk_vals.gather(-1, gather_idx)          # (batch, 1)

    probs[probs < threshold] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return torch.multinomial(probs, num_samples=1)
