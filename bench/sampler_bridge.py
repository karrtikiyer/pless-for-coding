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
