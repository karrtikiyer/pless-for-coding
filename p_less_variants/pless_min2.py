"""Plain p-less with a hard floor of two survivors.

Canonical ``p_less_decode`` admits token ``i`` iff ``pᵢ ≥ Σpᵢ²``. On peaked
distributions (e.g. once a loop is under way) ``Σpᵢ² → max(pᵢ)`` and only the
argmax survives; the loop then self-perpetuates because the sampler is
effectively deterministic.

This variant is byte-identical to ``p_less_decode`` whenever ≥ 2 tokens pass
the collision threshold, and unmasks the second-largest token whenever only
one does. No new hyperparameter, no threshold reparameterization; the only
behavioural change is the guarantee that the sampler never degenerates to a
deterministic argmax.

Why worth trying on the Qwen3-8B pless-α=2 failure list
(see ``results/pless_cot_efficiency_vllm/…/pless_a2_failures_solved_by_others.csv``):
the failure signature on those tasks is looping / repetition once the CoT
enters a peaked regime. Enforcing a two-token minimum during the peaked
regime is the smallest possible perturbation that provably makes escape
possible on every step.
"""
from __future__ import annotations

import torch


def p_less_min2_decode(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token per row from p-less with a min-2 survivor floor.

    ``probs``: shape (batch, vocab); modified in-place (same convention as
    the upstream ``p_less_decode``). Returns shape (batch, 1) long tensor
    of sampled token ids.
    """
    threshold = probs.square().sum(dim=-1, keepdim=True)     # (batch, 1)
    mask = probs < threshold                                 # True → prune

    # For rows where the mask would leave ≤ 1 survivor, unmask the top-2.
    survivors = (~mask).sum(dim=-1)                          # (batch,)
    need_floor = survivors < 2                               # (batch,)
    if bool(need_floor.any()):
        top2 = probs[need_floor].topk(2, dim=-1).indices     # (n_bad, 2)
        # Rebuild those rows' masks: keep only top-2, prune everything else.
        new_mask = torch.ones_like(mask[need_floor], dtype=torch.bool)
        new_mask.scatter_(-1, top2, False)
        mask[need_floor] = new_mask

    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return torch.multinomial(probs, num_samples=1)
