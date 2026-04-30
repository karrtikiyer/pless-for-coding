import sys
from pathlib import Path

import torch

# Add p-less/ to sys.path so we can import without modifying it
_pless_dir = str(Path(__file__).resolve().parent.parent / "p-less")
if _pless_dir not in sys.path:
    sys.path.insert(0, _pless_dir)

from p_less_samplers import p_less_decode, p_less_norm_decode  # noqa: E402

SAMPLERS = {
    "pless": p_less_decode,
    "pless_norm": p_less_norm_decode,
}


def make_temperature_sampler(top_p: float = 0.95, top_k: int = 20):
    """Standard temperature sampler with top-p/top-k filtering + multinomial.

    Default parameters match Qwen3's recommended generation config for thinking mode.
    Temperature scaling is handled by the caller (generator loop), so this
    sampler only applies top-k/top-p filtering and samples.
    """
    def sampler(probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clone()
        # top-k: zero out everything outside the top-k tokens
        if top_k > 0:
            topk_vals, _ = probs.topk(min(top_k, probs.shape[-1]), dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            probs[probs < threshold] = 0.0

        # top-p (nucleus): sort descending, cumsum, zero tokens past the threshold
        if top_p < 1.0:
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            # Shift right so the token that crosses the threshold is kept
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

        # Renormalize and sample
        probs.div_(probs.sum(dim=-1, keepdim=True).clamp(min=1e-12))
        return torch.multinomial(probs, num_samples=1)
    return sampler


# Samplers available for the split decoding method's --sampler-think / --sampler-code args
SPLIT_SAMPLERS = {
    "pless": p_less_decode,
    "pless_norm": p_less_norm_decode,
    # temp_standard: temperature + nucleus(0.95) + top-k(20). Matches Qwen3's
    # recommended generation config; the filter is meaningful at high temp.
    "temp_standard": make_temperature_sampler(),
    # temp_pure: pure temperature scaling, no top-p / top-k truncation.
    # Use when a clean temperature ablation is needed.
    "temp_pure": make_temperature_sampler(top_p=1.0, top_k=0),
}


def make_pless_post_temp_sampler(post_temperature: float):
    """P-less truncation followed by post-truncation temperature scaling.

    Decouples the pruning decision (controlled by the pre-temperature in the
    generation loop) from the sampling distribution among survivors (controlled
    by ``post_temperature`` here).

    Math: ``prob^(1/T₂)`` with T₂ > 1 flattens the survivor distribution.
    ``0^(1/T₂) = 0`` so pruned tokens stay at zero probability.
    """
    def sampler(probs: torch.Tensor) -> torch.Tensor:
        # Standard p-less truncation
        p = probs.square().sum(dim=-1, keepdim=True)
        mask = probs < p
        probs[mask] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True))
        # Flatten survivors by post-temperature
        if post_temperature != 1.0:
            probs.pow_(1.0 / post_temperature)
            probs.div_(probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    return sampler
