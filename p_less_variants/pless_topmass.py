"""Cumulative-mass p-less: an adaptive nucleus using the collision as the cutoff.

Sort tokens descending, cumulate mass, admit the shortest prefix with
cumulative mass ≥ ``1 − Σpᵢ²``.  Renormalise, sample.

This is *structurally* nucleus / top-p, but the ``p`` parameter is derived from
the distribution itself (``1 − Σpᵢ²`` = the Gini impurity of order 2), not
tuned.  On peaked distributions ``Σpᵢ² → 1`` so the target mass → 0, admitting
only the top-1 (deterministic collapse, same failure mode as canonical
p-less) — so this variant does not help the *looping* pathology on its own,
but it does provide a very different survivor set on medium-entropy
distributions than either rank-cut or threshold-cut p-less, which is worth
one shot on the failure list.
"""
from __future__ import annotations

import torch


def p_less_topmass_decode(probs: torch.Tensor) -> torch.Tensor:
    coll = probs.square().sum(dim=-1, keepdim=True)                # (B,1)
    target = (1.0 - coll).clamp(min=0.0)                            # (B,1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)  # (B,V)
    cumsum = sorted_probs.cumsum(dim=-1)
    # Include the token that crosses the threshold (shift right).
    mask_sorted = cumsum - sorted_probs > target                    # True → prune
    # Guarantee at least 2 survivors so the sampler is never fully deterministic
    # (the ``target → 0`` edge case on peaked distributions would otherwise
    # collapse to argmax and reproduce the p-less failure mode).
    mask_sorted[..., :2] = False
    sorted_probs = sorted_probs.masked_fill(mask_sorted, 0.0)
    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return torch.multinomial(probs, num_samples=1)
