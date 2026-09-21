"""vLLM logit-processor versions of the p_less_variants/ samplers.

Each function takes raw next-token *logits* (shape (V,)) and returns logits with
non-survivor tokens set to ``-inf``.  Mirrors the pattern in
``bench/generator_vllm.py`` (``_pless_mask_logits`` etc.) so the vLLM sampler
naturally samples from a truncated distribution.

Call :func:`register` once (before ``load_engine``) to splice these variants
into ``bench.generator_vllm._SAMPLER_LOGIT_FN`` — after that the vLLM runner
accepts ``--method pless_effk / pless_min2 / pless_topmass / pless_top1half /
pless_gini`` transparently.
"""
from __future__ import annotations

import torch


def _restore_argmax_on_all_pruned(mask: torch.Tensor, probs: torch.Tensor) -> None:
    """If a row would be fully pruned, unmask its argmax token in place."""
    all_pruned = mask.all(dim=-1)
    if bool(all_pruned.any()):
        argmax_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
        keep = torch.zeros_like(argmax_idx, dtype=torch.bool)
        mask[all_pruned] = mask[all_pruned].scatter(-1, argmax_idx, keep)


def _floor_min2(mask: torch.Tensor, probs: torch.Tensor) -> None:
    """If a row has fewer than 2 survivors, keep its top-2 tokens instead."""
    survivors = (~mask).sum(dim=-1)
    need = survivors < 2
    if bool(need.any()):
        top2 = probs[need].topk(2, dim=-1).indices
        new = torch.ones_like(mask[need], dtype=torch.bool)
        new.scatter_(-1, top2, False)
        mask[need] = new


def pless_effk_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """Effective-support top-k: keep top-⌈1/Σpᵢ²⌉ tokens (min 2)."""
    probs = torch.softmax(logits.float(), dim=-1)
    coll = probs.square().sum(dim=-1, keepdim=True).clamp(
        min=1.0 / probs.size(-1), max=1.0
    )
    n_eff = (1.0 / coll).round().to(torch.long)
    k_row = n_eff.clamp(min=2, max=probs.size(-1))
    k_max = int(k_row.max().item())
    topk_vals, _ = probs.topk(k_max, dim=-1)
    gather_idx = (k_row - 1).clamp(min=0, max=k_max - 1)
    threshold = topk_vals.gather(-1, gather_idx)
    mask = probs < threshold
    _restore_argmax_on_all_pruned(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def pless_min2_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """p-less threshold (Σpᵢ²) with a min-2 survivor floor."""
    probs = torch.softmax(logits.float(), dim=-1)
    threshold = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < threshold
    _floor_min2(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def pless_topmass_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """Cumulative-mass cut at ``1 − Σpᵢ²`` (min 2)."""
    probs = torch.softmax(logits.float(), dim=-1)
    coll = probs.square().sum(dim=-1, keepdim=True)
    target = (1.0 - coll).clamp(min=0.0)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask_sorted = cumsum - sorted_probs > target
    mask_sorted[..., :2] = False  # min-2 floor in sorted space
    scatter_mask = torch.zeros_like(mask_sorted).scatter_(-1, sorted_idx, mask_sorted)
    logits.masked_fill_(scatter_mask, float("-inf"))
    return logits


def pless_top1half_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """Admit ``pᵢ ≥ max(pᵢ) / 2`` (min 2)."""
    probs = torch.softmax(logits.float(), dim=-1)
    threshold = probs.max(dim=-1, keepdim=True).values * 0.5
    mask = probs < threshold
    _floor_min2(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def pless_gini_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """Inverted threshold ``pᵢ ≥ 1 − Σpᵢ²`` (min 2)."""
    probs = torch.softmax(logits.float(), dim=-1)
    threshold = 1.0 - probs.square().sum(dim=-1, keepdim=True)
    mask = probs < threshold
    _floor_min2(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


VARIANTS_VLLM = {
    "pless_effk":     pless_effk_mask_logits,
    "pless_min2":     pless_min2_mask_logits,
    "pless_topmass":  pless_topmass_mask_logits,
    "pless_top1half": pless_top1half_mask_logits,
    "pless_gini":     pless_gini_mask_logits,
}


def register() -> None:
    """Splice the 5 variants into ``bench.generator_vllm._SAMPLER_LOGIT_FN``.

    Idempotent — safe to call multiple times. Must be invoked *before*
    ``generate_samples_vllm`` first uses the dispatch table.
    """
    from bench import generator_vllm as gv
    for name, fn in VARIANTS_VLLM.items():
        gv._SAMPLER_LOGIT_FN[name] = fn
