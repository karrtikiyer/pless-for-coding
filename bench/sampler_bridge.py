import sys
from pathlib import Path

import torch

# Add p-less/ to sys.path so we can import without modifying it
_pless_dir = str(Path(__file__).resolve().parent.parent / "p-less")
if _pless_dir not in sys.path:
    sys.path.insert(0, _pless_dir)

from p_less_samplers import p_less_decode, p_less_norm_decode  # noqa: E402


# ---------------------------------------------------------------------------
# Crash-guarded wrappers around the authors' pless samplers
# ---------------------------------------------------------------------------
#
# The authors' p_less_decode / p_less_norm_decode are correct in exact
# arithmetic: the argmax token always survives because Σpᵢ² ≤ max(pᵢ), so a row
# is never fully pruned. In float32 on GPU, a parallel reduction over a large
# vocab can compute Σpᵢ² slightly above max(pᵢ) at near-deterministic positions,
# pruning *every* token → sum == 0 → divide-by-zero → NaN → torch.multinomial
# crash.
#
# These wrappers add a targeted argmax fallback for exactly that degenerate row,
# without modifying the read-only p-less/ submodule and without depending on
# make_pless_alpha_sampler. They are detect-first: the crash is internal to the
# authors' function, so we must spot the all-pruned row *before* delegating
# (a try/except is unsafe on GPU — a CUDA assert can poison the context).
#
# In the normal case (no row fully pruned, ~100% of real positions) the wrapper
# delegates untouched to the authors' function and is byte-identical to it.


def make_guarded_pless_sampler():
    """Authors' ``p_less_decode`` + argmax fallback for the degenerate
    all-pruned row. Byte-identical to ``p_less_decode`` whenever no row is fully
    pruned. Threshold = Σpᵢ²."""
    def sampler(probs: torch.Tensor) -> torch.Tensor:
        threshold = probs.square().sum(dim=-1, keepdim=True)
        all_pruned = (probs < threshold).all(dim=-1)            # (N,)
        if not bool(all_pruned.any()):
            return p_less_decode(probs)                         # authors' path, unchanged
        out = probs.new_zeros((probs.size(0), 1), dtype=torch.long)
        out[all_pruned] = probs[all_pruned].argmax(dim=-1, keepdim=True)
        good = ~all_pruned
        if bool(good.any()):
            out[good] = p_less_decode(probs[good])              # boolean-index copy → original safe
        return out
    return sampler


def make_guarded_pless_norm_sampler():
    """Authors' ``p_less_norm_decode`` + the same argmax fallback.
    Threshold = (v·Σpᵢ² − 1)/(v − 1)."""
    def sampler(probs: torch.Tensor) -> torch.Tensor:
        v = probs.size(-1)
        threshold = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
        all_pruned = (probs < threshold).all(dim=-1)
        if not bool(all_pruned.any()):
            return p_less_norm_decode(probs)
        out = probs.new_zeros((probs.size(0), 1), dtype=torch.long)
        out[all_pruned] = probs[all_pruned].argmax(dim=-1, keepdim=True)
        good = ~all_pruned
        if bool(good.any()):
            out[good] = p_less_norm_decode(probs[good])
        return out
    return sampler


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


# pless / pless_norm route through the crash-guarded wrappers (byte-identical to
# the authors' samplers in the normal case; argmax fallback only on the
# degenerate all-pruned row).
SAMPLERS = {
    "pless": make_guarded_pless_sampler(),
    "pless_norm": make_guarded_pless_norm_sampler(),
}

# Samplers available for the split decoding method's --sampler-think / --sampler-code args
SPLIT_SAMPLERS = {
    "pless": make_guarded_pless_sampler(),
    "pless_norm": make_guarded_pless_norm_sampler(),
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
        # Standard p-less truncation (same Σpᵢ² threshold as p_less_decode).
        p = probs.square().sum(dim=-1, keepdim=True)
        mask = probs < p
        # Argmax fallback for the degenerate all-pruned row (float32 reduction
        # can push Σpᵢ² above max(pᵢ)); mirrors make_pless_alpha_sampler. Without
        # it, the renormalize below would divide by zero → NaN.
        all_pruned = mask.all(dim=-1)
        if all_pruned.any():
            fallback_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
            mask[all_pruned] = mask[all_pruned].scatter(-1, fallback_idx, False)
        probs[mask] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True))
        # Flatten survivors by post-temperature
        if post_temperature != 1.0:
            probs.pow_(1.0 / post_temperature)
            probs.div_(probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    return sampler


def make_pless_alpha_sampler(alpha: float):
    """Rényi-α-generalized p-less sampler.

    Threshold ``= Σpᵢ^α`` (raw, no root — matches the unrooted ``Σpᵢ²`` in
    the upstream p-less). α=2 reproduces ``p_less_decode`` exactly.
    α > 2 keeps more tokens at high-entropy (semantic) positions while
    preserving tightness at peaked (syntactic) ones. α < 2 is stricter
    and may zero out the whole row at non-peaked positions — falls back
    to argmax for those rows.

    For α ≥ 2 the max-prob token always survives because
    ``Σpᵢ^α ≤ max(pᵢ)^(α-1) ≤ max(pᵢ)``; the argmax fallback is a no-op.

    See ``docs/research/position_aware_code_sampling.md`` for design rationale.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    def sampler(probs: torch.Tensor) -> torch.Tensor:
        # Threshold = Σ pᵢ^α (raw, no root).
        # Special-case α=2 to use the exact ``square()`` call from the
        # upstream p-less, guaranteeing byte-identical behavior at α=2.
        if alpha == 2.0:
            threshold = probs.square().sum(dim=-1, keepdim=True)
        else:
            threshold = probs.pow(alpha).sum(dim=-1, keepdim=True)

        mask = probs < threshold  # True ⇒ prune
        # Fallback: if the mask would prune every token in a row (possible
        # only when α < 2 on non-peaked distributions), unmask its argmax.
        all_pruned = mask.all(dim=-1)
        if all_pruned.any():
            fallback_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
            mask[all_pruned] = mask[all_pruned].scatter(-1, fallback_idx, False)

        probs[mask] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    return sampler


def make_pless_renyi_sampler(k: float):
    """Rényi-order-k threshold sampler — the origin paper's App. B.5 *rooted* form.

    Threshold ``= G_k = (Σpᵢ^k)^{1/(k-1)} = exp(-H_k)``, i.e. the Rényi entropy of
    order k in exponentiated form. This is distinct from ``make_pless_alpha_sampler``'s
    raw power sum ``τ_α = Σpᵢ^α``: the two coincide *only* at order 2 (both = Σpᵢ²).
    Unlike ``τ``, ``G_k`` is a probability-weighted power mean, so it always lies in
    ``[min pᵢ, max pᵢ]``. Lowering k below 2 loosens the filter (admits more tail
    tokens); k=2 reproduces plain p-less byte-for-byte. Same all-pruned argmax
    fallback as the other guarded samplers.

    See ``docs/research/paperA_renyi_nonequivalence.md`` for why τ_α ≠ G_k for order>2.

    Any real order k is accepted (matching the author's ``p_moment_decode`` reference):
    k=0 → 1/v (the uniform-entropy threshold); k=1 → the Shannon limit; k<0 loosens
    further toward ``min pᵢ``. Caveat: for strongly negative k over a large vocabulary,
    ``probs.pow(k)`` on near-zero tail tokens can overflow float32 (same as the author's
    reference); the practically-useful range (k ≳ -3) is safe.
    """

    def sampler(probs: torch.Tensor) -> torch.Tensor:
        if k == 2.0:
            # byte-identical to plain p-less at order 2 (root exponent 1/(2-1)=1)
            threshold = probs.square().sum(dim=-1, keepdim=True)
        elif k == 1.0:
            # Shannon limit: G_1 = exp(Σ pᵢ ln pᵢ). Guard log(0) via where().
            logp = torch.where(probs > 0, probs.log(), probs.new_zeros(()))
            threshold = (probs * logp).sum(dim=-1, keepdim=True).exp()
        elif k == 0.0:
            # G_0 = 1/v (uniform threshold); the general branch also yields this but
            # 0^0 on zero-prob tokens is ambiguous, so special-case it (matches author).
            threshold = 1.0 / probs.size(-1)
        else:
            threshold = probs.pow(k).sum(dim=-1, keepdim=True).pow(1.0 / (k - 1.0))

        mask = probs < threshold  # True ⇒ prune
        # Fallback: unmask the argmax if a row would be fully pruned (avoids ÷0 → NaN).
        all_pruned = mask.all(dim=-1)
        if all_pruned.any():
            fallback_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
            mask[all_pruned] = mask[all_pruned].scatter(-1, fallback_idx, False)

        probs[mask] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    return sampler
