"""pless / pless_norm logits processors for mlx-lm.

mlx-lm's ``logits_processors`` contract (confirmed directly against
mlx_lm/generate.py's ``_step`` for both single-sequence generate and
``GenerationBatch``): each processor is called per-sequence as
``processor(token_history, logits) -> logits`` with ``logits`` shape
``(1, vocab)``. Sampling (softmax + categorical draw) happens afterward in a
separate ``sampler`` callable -- the same two-stage contract vLLM's
``LogitsProcessor.apply()`` uses.

This file does not reimplement p-less. It is a line-for-line translation (torch
-> mx.array) of the already-authored, already-reviewed logits-space masking
functions in ``bench/generator_vllm.py`` (``_pless_mask_logits`` /
``_pless_norm_mask_logits`` / ``_restore_argmax_on_all_pruned``), which are
themselves a verified transcription of the authors' probs-space
``p_less_decode`` / ``p_less_norm_decode`` (``p-less/p_less_samplers.py``) into
"return masked logits, let the caller sample" form. See
``scripts/verify_mlx_pless_logits.py`` for the randomized-trial check that this
translation produces the identical survivor mask as the torch reference on
every trial (same validation pattern used to catch the earlier top-p
inclusive/exclusive bug in ``_top_p_top_k_mask_logits``).
"""
import math
import numbers

import mlx.core as mx


def _restore_argmax_on_all_pruned(mask: mx.array, probs: mx.array) -> mx.array:
    """If a row is fully pruned, un-mask its argmax token.

    mx translation of ``generator_vllm.py``'s ``_restore_argmax_on_all_pruned``.
    In exact arithmetic the pless thresholds never prune every token (the
    argmax always satisfies the threshold), but a float32 reduction over a
    large vocab can push the threshold just above max(pᵢ) and mask the whole
    row -- this restores at least one surviving token for that row.
    """
    all_pruned = mx.all(mask, axis=-1, keepdims=True)  # (N, 1)
    argmax_idx = mx.argmax(probs, axis=-1, keepdims=True)  # (N, 1)
    current = mx.take_along_axis(mask, argmax_idx, axis=-1)  # (N, 1)
    keep = current & ~all_pruned
    return mx.put_along_axis(mask, argmax_idx, keep, axis=-1)


def _pless_mask_logits(logits: mx.array) -> mx.array:
    """Set tokens below the p-less threshold (Σpᵢ²) to -inf.

    mx translation of ``generator_vllm.py:_pless_mask_logits``.
    """
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    p = mx.sum(mx.square(probs), axis=-1, keepdims=True)
    mask = probs < p
    mask = _restore_argmax_on_all_pruned(mask, probs)
    return mx.where(mask, mx.array(-mx.inf, dtype=logits.dtype), logits)


def _pless_norm_mask_logits(logits: mx.array) -> mx.array:
    """Set tokens below the p-less-norm threshold to -inf.

    mx translation of ``generator_vllm.py:_pless_norm_mask_logits``.
    """
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    v = probs.shape[-1]
    p = (v * mx.sum(mx.square(probs), axis=-1, keepdims=True) - 1.0) / (v - 1.0)
    mask = probs < p
    mask = _restore_argmax_on_all_pruned(mask, probs)
    return mx.where(mask, mx.array(-mx.inf, dtype=logits.dtype), logits)


def _pless_effk_mask_logits(logits: mx.array) -> mx.array:
    """Effective-support top-k: keep top-ceil(1/Sum(p_i^2)) tokens (min 2).

    mx translation of ``bench/pless_variants_vllm.py:pless_effk_mask_logits``.
    """
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    vocab = probs.shape[-1]
    coll = mx.clip(mx.sum(mx.square(probs), axis=-1, keepdims=True), 1.0 / vocab, 1.0)
    n_eff = mx.round(1.0 / coll)
    k_row = mx.clip(n_eff, 2, vocab).astype(mx.int32)
    k_max = int(mx.max(k_row).item())

    topk_vals = mx.sort(probs, axis=-1)[:, ::-1][:, :k_max]
    gather_idx = mx.clip(k_row - 1, 0, k_max - 1)
    threshold = mx.take_along_axis(topk_vals, gather_idx, axis=-1)

    mask = probs < threshold
    mask = _restore_argmax_on_all_pruned(mask, probs)
    return mx.where(mask, mx.array(-mx.inf, dtype=logits.dtype), logits)


def _pless_renyi_mask_logits(logits: mx.array, k: float) -> mx.array:
    """Set tokens below the Rényi-order-k threshold G_k = (Σpᵢ^k)^{1/(k-1)} to -inf.

    mx translation of ``generator_vllm.py:_pless_renyi_mask_logits``, same
    special cases: k==2 -> Σpᵢ² (identical to ``_pless_mask_logits``), k==1 ->
    Shannon limit exp(Σpᵢ log pᵢ), k==0 -> 1/v.
    """
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    if k == 2.0:
        threshold = mx.sum(mx.square(probs), axis=-1, keepdims=True)
    elif k == 1.0:
        logp = mx.where(probs > 0, mx.log(probs), mx.zeros_like(probs))
        threshold = mx.exp(mx.sum(probs * logp, axis=-1, keepdims=True))
    elif k == 0.0:
        threshold = mx.array(1.0 / probs.shape[-1], dtype=mx.float32)
    else:
        threshold = mx.power(mx.sum(mx.power(probs, k), axis=-1, keepdims=True), 1.0 / (k - 1.0))
    mask = probs < threshold
    mask = _restore_argmax_on_all_pruned(mask, probs)
    return mx.where(mask, mx.array(-mx.inf, dtype=logits.dtype), logits)


def make_pless_logits_processor():
    """mlx-lm ``logits_processors`` entry for plain p-less."""
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        return _pless_mask_logits(logits)
    return processor


def make_pless_norm_logits_processor():
    """mlx-lm ``logits_processors`` entry for p-less-norm."""
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        return _pless_norm_mask_logits(logits)
    return processor


def make_pless_effk_logits_processor():
    """mlx-lm ``logits_processors`` entry for p-less effective-support top-k."""
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        return _pless_effk_mask_logits(logits)
    return processor


def make_pless_renyi_logits_processor(k: float):
    """mlx-lm ``logits_processors`` entry for G_k (``pless_renyi``) at order ``k``."""
    if isinstance(k, bool) or not isinstance(k, numbers.Real) or not math.isfinite(k):
        raise ValueError(f"renyi k must be a finite number, got {k!r}")
    k = float(k)

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        # float32 out: mlx-lm runs logits - logsumexp(logits) in this dtype next,
        # and vLLM does that normalisation in float32.
        return _pless_renyi_mask_logits(logits.astype(mx.float32), k)
    return processor
