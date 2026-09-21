"""Gini-inverted p-less: threshold = ``1 − Σpᵢ²`` (Gini impurity of order 2).

Where canonical p-less uses the collision ``Σpᵢ²`` as an *admission floor*
(prune ``pᵢ < Σpᵢ²``), this variant uses the *complement* ``1 − Σpᵢ²`` — the
Gini impurity — as the floor.  The two thresholds behave oppositely with
peakedness:

* peaked (``Σpᵢ² → 1``) → threshold ``→ 0`` → **admit almost everything**,
  breaking the deterministic collapse that produces the p-less loop failure;
* flat (``Σpᵢ² → 0``) → threshold ``→ 1`` → aggressive pruning (only tokens
  with probability ≥ 1 survive, which almost never happens → the min-2 floor
  kicks in on every step).

That flat-regime behaviour is a real concern for open-vocabulary generation
where the entropy is genuinely high most of the time — this variant will
sample essentially at random from the top-2 in those regimes, hurting
coherence.  Included here specifically because on the *failure-list* tasks
the failure signature is *peaked / looping*, exactly the regime this
threshold relaxes.

Uses only ``Σpᵢ²`` as its input signal, but as an *upper*-bound complement
rather than a lower-bound threshold — a genuinely different family from
canonical p-less, τ_α, and G_k.
"""
from __future__ import annotations

import torch


def p_less_gini_decode(probs: torch.Tensor) -> torch.Tensor:
    threshold = 1.0 - probs.square().sum(dim=-1, keepdim=True)      # (B,1)
    mask = probs < threshold                                        # True → prune
    # Min-2 floor: essential for this variant because on flat distributions
    # the threshold approaches 1 and nothing passes.
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
