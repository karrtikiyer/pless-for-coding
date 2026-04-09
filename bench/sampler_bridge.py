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

# Logit-based samplers: take temperature-scaled logits (B, V) → (B, 1) token indices.
# Unlike prob-based SAMPLERS, these handle softmax internally and skip the
# smoothing constant used by p-less methods.
LOGIT_SAMPLERS = {"top_h", "top_nsigma"}


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


# ---------------------------------------------------------------------------
# Top-H decoding (Baghaei Potraghloo et al., arXiv:2509.02510)
# ---------------------------------------------------------------------------

def top_h_decode(logits: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
    """Entropy-constrained mass maximization (ECMM) via greedy token selection.

    Sorts tokens by descending probability, then greedily adds them to the
    sampling set until the entropy of the renormalised subset exceeds
    ``alpha * H(p)``.  Keeps the subset with maximal probability mass while
    bounding the randomness of the distribution we sample from.

    Uses the incremental entropy formula from the paper's Theorem 3 / Appendix
    A.3 for efficient O(V) computation after sorting.

    Args:
        logits: (batch_size, vocab_size) temperature-scaled logits.
        alpha:  entropy threshold coefficient, 0 < alpha < 1 (paper default 0.4).

    Returns:
        (batch_size, 1) sampled token indices.
    """
    probs = torch.softmax(logits, dim=-1)  # (B, V)

    # H(p): entropy of the full distribution
    log_p = torch.log(probs.clamp(min=1e-12))
    H_p = -(probs * log_p).sum(dim=-1)  # (B,)
    threshold = alpha * H_p              # (B,)

    # Sort descending by probability
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

    # Incremental entropy: H(q_j) = log(Γ_j) - h_j / Γ_j
    #   Γ_j = cumulative probability mass of first j tokens
    #   h_j = cumulative sum of p_i · log(p_i)
    gamma = sorted_probs.cumsum(dim=-1)                                      # (B, V)
    h = (sorted_probs * torch.log(sorted_probs.clamp(min=1e-12))).cumsum(dim=-1)  # (B, V)
    H_q = torch.log(gamma.clamp(min=1e-12)) - h / gamma.clamp(min=1e-12)    # (B, V)

    # First position where H(q) exceeds threshold → that token is excluded
    exceeded = H_q > threshold.unsqueeze(-1)  # (B, V)
    any_exceeded = exceeded.any(dim=-1)        # (B,)
    first_exceed_idx = exceeded.float().argmax(dim=-1)  # (B,)  — first True
    vocab_size = logits.shape[-1]
    n_keep = torch.where(any_exceeded, first_exceed_idx,
                         torch.tensor(vocab_size, device=logits.device))
    n_keep = n_keep.clamp(min=1)

    # Mask, renormalise, sample
    range_idx = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
    keep_mask = range_idx < n_keep.unsqueeze(-1)  # (B, V)

    filtered = sorted_probs * keep_mask.float()
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    sampled_sorted_idx = torch.multinomial(filtered, num_samples=1)          # (B, 1)
    next_tokens = sorted_indices.gather(dim=-1, index=sampled_sorted_idx)    # (B, 1)
    return next_tokens


def make_top_h_sampler(alpha: float = 0.4):
    """Factory: returns a logit sampler with the given alpha."""
    def sampler(logits: torch.Tensor) -> torch.Tensor:
        return top_h_decode(logits, alpha=alpha)
    return sampler


# ---------------------------------------------------------------------------
# Top-nσ decoding (Tang et al., arXiv:2411.07641)
# ---------------------------------------------------------------------------

def top_nsigma_decode(logits: torch.Tensor, n: float = 1.0) -> torch.Tensor:
    """Statistical logit filtering based on the noisy/informative region split.

    Keeps tokens whose (temperature-scaled) logits fall within ``n`` standard
    deviations of the maximum logit.  Temperature-invariant by construction:
    the mask ``l_i >= max(l) - n·σ(l)`` is the same before and after dividing
    all logits by T, because both max and σ scale linearly.

    Args:
        logits: (batch_size, vocab_size) temperature-scaled logits.
        n:      threshold multiplier (paper default 1.0).

    Returns:
        (batch_size, 1) sampled token indices.
    """
    M = logits.max(dim=-1, keepdim=True).values    # (B, 1)
    sigma = logits.std(dim=-1, keepdim=True)        # (B, 1)

    # Mask: keep tokens where logit >= M - n·σ
    threshold = M - n * sigma                       # (B, 1)
    mask = logits >= threshold                      # (B, V)

    filtered_logits = logits.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)  # (B, 1)
    return next_tokens


def make_top_nsigma_sampler(n: float = 1.0):
    """Factory: returns a logit sampler with the given n."""
    def sampler(logits: torch.Tensor) -> torch.Tensor:
        return top_nsigma_decode(logits, n=n)
    return sampler
