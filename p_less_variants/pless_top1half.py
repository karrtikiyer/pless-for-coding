"""Relative-to-top-1 threshold: admit ``pᵢ ≥ max(pᵢ) / 2``.

Hyperparameter-free (the ``1/2`` is a structural constant, not a knob — the
factor is uniquely determined by the "always keep a token at least half as
likely as the top token" rule and cannot be tuned without breaking that
invariant).  Not collision-based (no ``Σpᵢ²``), so it's a *different signal*
from canonical p-less and the two collision-based variants we test alongside.

On peaked distributions the top token dominates by a big margin and only it
survives the ``max/2`` filter — that reproduces the p-less looping failure,
so we floor at 2 survivors.  On medium-entropy distributions this variant
admits a much broader set than canonical p-less (which prunes at ``Σpᵢ²``,
often ≪ ``max/2``), giving it a chance on the failure-list tasks where the
model needs more branching diversity mid-reasoning.
"""
from __future__ import annotations

import torch


def p_less_top1half_decode(probs: torch.Tensor) -> torch.Tensor:
    max_p = probs.max(dim=-1, keepdim=True).values                  # (B,1)
    threshold = max_p * 0.5
    mask = probs < threshold                                        # True → prune
    survivors = (~mask).sum(dim=-1)
    need_floor = survivors < 2
    if bool(need_floor.any()):
        top2 = probs[need_floor].topk(2, dim=-1).indices
        new_mask = torch.ones_like(mask[need_floor], dtype=torch.bool)
        new_mask.scatter_(-1, top2, False)
        mask[need_floor] = new_mask
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return torch.multinomial(probs, num_samples=1)
