"""vLLM backend for the pless benchmark generation pipeline.

Sibling of ``bench/generator.py`` (the HuggingFace backend). This file
exposes ``generate_samples_split_vllm`` and ``generate_samples_vllm``
with **identical signatures + return types** (``list[str]``) to the HF
counterparts so the runners can swap backends with a single flag.

Backend selection lives in each runner via ``--backend {hf,vllm}``;
this file is never imported when the user picks ``hf``.

Why a parallel file:
  * vLLM is only installable on CUDA Linux boxes (no Mac wheels). The
    main `bench/generator.py` keeps working on Mac for development.
  * vLLM owns model + tokenizer + decode loop. Reusing the HF
    `generate_samples_split` would buy nothing — we replace the whole
    decode loop with one ``LLM.generate(...)`` call backed by a
    custom :class:`PlessSplitLogitsProcessor`.

Concept map (HF backend → vLLM equivalent):
  * ``model + tokenizer + manual decode loop``  → ``vllm.LLM`` instance
  * per-step ``softmax + sampler_fn(probs)``    → :class:`PlessSplitLogitsProcessor.apply`
  * one-way think→code switch on token 151668   → per-request state in ``update_state``
  * temperature application                     → done inside the logits processor (we
    pass SamplingParams.temperature=1.0 so vLLM does not double-scale)

Module-load contract: this file MUST parse on a vLLM-free environment
so the runners can defer-import it inside the `--backend vllm` branch
only. All vLLM imports are pushed inside functions / class factories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:  # pragma: no cover — type hints only, never executed at import
    from vllm import LLM, SamplingParams
    from vllm.v1.sample.logits_processor import LogitsProcessor


# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

# Qwen3 </think> token id. Asserted at runtime against the tokenizer (see
# ``_verify_think_end_id``) before any generation runs, so the same
# correctness check the HF backend does in ``generate_samples_split:517-521``
# is mirrored here.
QWEN3_THINK_END_TOKEN_ID = 151668


def resolve_think_end_id(tokenizer) -> int:
    """Resolve a model's single-token ``</think>`` id from its own tokenizer.

    Model-aware replacement for the hardcoded ``QWEN3_THINK_END_TOKEN_ID``: the
    split / loop-force processor needs the integer id both to detect the
    think→code boundary and to force ``</think>`` on a loop. Qwen3 → 151668,
    DeepSeek-R1-Distill → 128014. Raises if ``</think>`` is not a single token —
    the processor can only force one id, so a multi-token result is a hard error
    (and on a smaller-vocab model the stale Qwen3 default would IndexError).
    """
    ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError(
            f"</think> must be a single token for split/loop-force decoding, "
            f"got {ids} from tokenizer {tokenizer.__class__.__name__}"
        )
    return ids[0]


# Live n-gram loop detection → force </think> (opt-in via cfg["loop_n"]/["loop_k"]).
from bench.loop_detect import ngram_loop_fired
_LOOP_CHECK_EVERY = 8   # run the n-gram scan every N think-tokens (throttle; a loop needs ~n*k tokens to form)


# ---------------------------------------------------------------------------
# Pless logit transforms (vLLM operates on raw logits, not probs)
# ---------------------------------------------------------------------------
#
# The HF backend's ``p_less_decode`` (p-less/p_less_samplers.py) works on the
# *probability* distribution after temperature + softmax:
#
#     p = sum(probs^2)         # collision-entropy threshold
#     probs[probs < p] = 0     # mask
#     probs /= probs.sum()     # renormalize
#     return multinomial(probs, n=1)
#
# vLLM's ``LogitsProcessor.apply`` receives raw logits (post-temperature is
# the caller's job; vLLM's own sampler will softmax + multinomial on our
# return value). So the equivalent transform is:
#
#     1. softmax(logits) → probs
#     2. apply pless mask on probs
#     3. set masked tokens' logits to -inf (so softmax(...) downstream
#        produces a zero on them and the multinomial respects the mask)
#
# Pure renormalization isn't needed because vLLM's downstream softmax
# does it for us — we only need to express "these tokens are dead".
#
# All transforms are batch-aware: ``logits`` shape is (num_rows, vocab).


def _restore_argmax_on_all_pruned(
    mask: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """Branchless argmax fallback for a fully-pruned row (in-place on ``mask``).

    In exact arithmetic the pless thresholds never prune every token (the argmax
    always satisfies Σpᵢ^α ≤ max pᵢ), but a float32 reduction over a large vocab
    can push the threshold just above max(pᵢ) and mask the whole row. This
    un-masks each all-pruned row's argmax so at least one token survives —
    matching the HF guarded samplers (``bench/sampler_bridge.py``).

    Written branchlessly (no ``.any()`` / ``.item()`` / Python ``if``) so it adds
    no per-step CPU-GPU synchronization on vLLM's hot decode path. Its cost is
    negligible once the fast path applies it to the whole batch in one reduction
    (measured ~2% vs no guard at K=80).
    """
    all_pruned = mask.all(dim=-1, keepdim=True)            # (N, 1)
    argmax_idx = probs.argmax(dim=-1, keepdim=True)        # (N, 1)
    # Mask value at argmax becomes False iff the row was all-pruned; otherwise
    # it keeps its current value (no-op for the common, non-degenerate case).
    keep = mask.gather(-1, argmax_idx) & ~all_pruned
    mask.scatter_(-1, argmax_idx, keep)
    return mask


def _pless_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """In-place: zero out (set to -inf) tokens below the p-less threshold."""
    probs = torch.softmax(logits.float(), dim=-1)
    p = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < p
    _restore_argmax_on_all_pruned(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def _pless_norm_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """In-place: zero out tokens below the p-less-norm threshold."""
    probs = torch.softmax(logits.float(), dim=-1)
    v = probs.size(-1)
    p = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
    mask = probs < p
    _restore_argmax_on_all_pruned(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def _pless_alpha_mask_logits(logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """In-place: zero out tokens below the Rényi-α p-less threshold.

    Mirrors ``bench/sampler_bridge.py:make_pless_alpha_sampler``:
      * threshold = Σ pᵢ^α  (raw, no root — matches the unrooted Σpᵢ² in pless)
      * α=2 fast-path uses ``probs.square()`` for byte-equivalence with
        ``_pless_mask_logits``.
      * α < 2 may prune the whole row at non-peaked distributions; the
        argmax-fallback restores its argmax token.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    if alpha == 2.0:
        threshold = probs.square().sum(dim=-1, keepdim=True)
    else:
        threshold = probs.pow(alpha).sum(dim=-1, keepdim=True)
    mask = probs < threshold
    _restore_argmax_on_all_pruned(mask, probs)
    logits.masked_fill_(mask, float("-inf"))
    return logits


def _top_p_top_k_mask_logits(
    logits: torch.Tensor, top_p: float, top_k: int
) -> torch.Tensor:
    """Apply top-p (nucleus) and/or top-k truncation by masking logits to -inf.

    Matches the semantics of ``bench/sampler_bridge.py:make_temperature_sampler``
    — temp_standard's default is top_p=0.95, top_k=20; temp_pure passes
    top_p=1.0, top_k=0 (i.e. no truncation).
    """
    if top_k > 0:
        # Mask everything below the top-k threshold.
        kth_vals = logits.topk(top_k, dim=-1).values[..., -1:].expand_as(logits)
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        # Tokens beyond cumulative top_p are dropped (but always keep at least one).
        sorted_mask = cum > top_p
        sorted_mask[..., 0] = False
        # Scatter the mask back to original token positions.
        scatter_mask = torch.zeros_like(sorted_mask).scatter_(-1, sorted_idx, sorted_mask)
        logits = logits.masked_fill(scatter_mask, float("-inf"))
    return logits


# Dispatch table: sampler name → logit transform.
_SAMPLER_LOGIT_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "pless":         _pless_mask_logits,
    "pless_norm":    _pless_norm_mask_logits,
    "temp_pure":     lambda x: x,  # no truncation
    "temp_standard": lambda x: _top_p_top_k_mask_logits(x, top_p=0.95, top_k=20),
}


# ---------------------------------------------------------------------------
# LogitsProcessor factory
# ---------------------------------------------------------------------------
#
# The class definition has to happen *inside* a function so the
# ``from vllm.v1.sample.logits_processor import LogitsProcessor`` base-class
# import only runs when the user actually picks `--backend vllm`. Otherwise
# this file has to be loadable on Mac.


def _build_pless_split_logits_processor_class() -> type:
    """Define and return the :class:`PlessSplitLogitsProcessor` class.

    The class is built once, lazily, on first use. The result is cached on
    the function (closure-free pattern) so subsequent calls return the
    same type.
    """
    if _build_pless_split_logits_processor_class._cached is not None:  # type: ignore[attr-defined]
        return _build_pless_split_logits_processor_class._cached  # type: ignore[attr-defined]

    # Deferred imports — only loaded on first call.
    from vllm.sampling_params import SamplingParams  # noqa: F401  (used for type hints)
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality

    class PlessSplitLogitsProcessor(LogitsProcessor):
        """Pless / pless-norm / temp samplers + one-way think→code phase switch.

        Per request, the engine hands us:
          * a batch index (position in the in-flight tensor)
          * the request's SamplingParams (we read ``extra_args`` for our
            pless config dict)
          * a reference to the request's growing ``output_token_ids`` list

        We maintain two dictionaries keyed by batch index:
          * ``_cfg[idx]``  → per-request sampler/temperature config
          * ``_out[idx]``  → reference to the output_token_ids list

        At each engine step:
          * ``update_state`` keeps these dictionaries consistent with vLLM's
            batch composition (adds/removes/moves).
          * ``apply(logits)`` walks the active rows, checks whether
            ``QWEN3_THINK_END_TOKEN_ID`` has been emitted yet for each row
            (i.e. row is in code phase), and applies the appropriate
            temperature + sampler logit transform.

        Sampler config schema (passed via ``SamplingParams.extra_args``):

            extra_args = {
                "pless_split": {
                    "t_think":       float,
                    "t_code":        float,
                    "sampler_think": "pless"|"pless_norm"|"temp_pure"|"temp_standard",
                    "sampler_code":  "pless"|"pless_norm"|"temp_pure"|"temp_standard",
                }
            }
        """

        EXTRA_ARG_KEY = "pless_split"

        def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool):  # noqa: D401
            self._device = device
            self._cfg: dict[int, dict] = {}
            self._out: dict[int, list[int]] = {}
            # Once a request has emitted </think>, cache the result so we
            # don't re-scan the (growing) output_token_ids list every step.
            self._in_code: dict[int, bool] = {}
            # Live loop-force: once the n-gram detector fires for a request, keep
            # forcing </think> until it's emitted (then in_code flips to True).
            self._loop_fired: dict[int, bool] = {}

        @classmethod
        def validate_params(cls, sampling_params):
            extra = sampling_params.extra_args
            if extra is None or cls.EXTRA_ARG_KEY not in extra:
                # Caller doesn't want split-decoding behaviour — allow it
                # (the processor will pass logits through unchanged for
                # that request).
                return
            cfg = extra[cls.EXTRA_ARG_KEY]
            for key in ("t_think", "t_code", "sampler_think", "sampler_code"):
                if key not in cfg:
                    raise ValueError(f"pless_split config missing {key!r}")
            allowed = set(_SAMPLER_LOGIT_FN) | {"pless_alpha"}
            for sampler_key, alpha_key in (
                ("sampler_think", "alpha_think"),
                ("sampler_code", "alpha_code"),
            ):
                if cfg[sampler_key] not in allowed:
                    raise ValueError(
                        f"pless_split.{sampler_key}={cfg[sampler_key]!r} not in "
                        f"{sorted(allowed)}"
                    )
                if cfg[sampler_key] == "pless_alpha" and alpha_key not in cfg:
                    raise ValueError(
                        f"pless_split.{sampler_key}='pless_alpha' requires "
                        f"{alpha_key!r} to be set."
                    )

        def is_argmax_invariant(self) -> bool:
            # We change the distribution non-trivially.
            return False

        def update_state(self, batch_update) -> None:  # type: ignore[no-untyped-def]
            if batch_update is None:
                return
            # Added requests
            for idx, params, _prompt_ids, output_ids in batch_update.added:
                extra = params.extra_args if params is not None else None
                cfg = extra.get(self.EXTRA_ARG_KEY) if extra else None
                if cfg is None:
                    # Request opted out of split decoding; ensure we don't
                    # carry stale state from a previous occupant of this index.
                    self._cfg.pop(idx, None)
                    self._out.pop(idx, None)
                    self._in_code.pop(idx, None)
                    self._loop_fired.pop(idx, None)
                    continue
                self._cfg[idx] = cfg
                self._out[idx] = output_ids  # reference; grows automatically
                self._in_code[idx] = False
                self._loop_fired[idx] = False
            # Removed requests
            for idx in batch_update.removed:
                self._cfg.pop(idx, None)
                self._out.pop(idx, None)
                self._in_code.pop(idx, None)
                self._loop_fired.pop(idx, None)
            # Moved requests — adx → bdx; ``direct`` distinguishes move vs swap.
            for adx, bdx, direct in batch_update.moved:
                a_cfg = self._cfg.pop(adx, None)
                b_cfg = self._cfg.pop(bdx, None)
                a_out = self._out.pop(adx, None)
                b_out = self._out.pop(bdx, None)
                a_code = self._in_code.pop(adx, None)
                b_code = self._in_code.pop(bdx, None)
                a_loop = self._loop_fired.pop(adx, None)
                b_loop = self._loop_fired.pop(bdx, None)
                if a_cfg is not None:
                    self._cfg[bdx] = a_cfg
                    self._out[bdx] = a_out  # type: ignore[assignment]
                    self._in_code[bdx] = a_code if a_code is not None else False
                    self._loop_fired[bdx] = a_loop if a_loop is not None else False
                # ``direct`` is a MoveDirectionality enum, NOT the string "swap".
                # On a SWAP, B (the request now at adx) must keep its state too;
                # comparing against "swap" silently dropped it.
                if direct == MoveDirectionality.SWAP and b_cfg is not None:
                    self._cfg[adx] = b_cfg
                    self._out[adx] = b_out  # type: ignore[assignment]
                    self._in_code[adx] = b_code if b_code is not None else False
                    self._loop_fired[adx] = b_loop if b_loop is not None else False

        def _uniform_cfg(self):
            """If every tracked request uses ONE phase-independent sampler config
            (``sampler_think == sampler_code`` and equal temps/alpha, identical
            across all rows), return ``(sampler_name, temp, alpha)``; else None.

            This is the common ``generate_samples_vllm`` case (think==code), where
            the think/code phase is irrelevant — so the whole batch can be masked
            in one call instead of the per-row Python loop, which is launch-bound
            and dominates decode time at large batch (measured 1.5x→2.0x slowdown
            at K=40→80). Returns None for split/mixed configs, which then take the
            unchanged per-row path below.
            """
            first = None
            n = 0
            for _idx, cfg in self._cfg.items():
                name = cfg["sampler_think"]
                if name != cfg["sampler_code"]:
                    return None
                temp = float(cfg["t_think"])
                if temp != float(cfg["t_code"]):
                    return None
                if name == "pless_alpha":
                    if float(cfg["alpha_think"]) != float(cfg["alpha_code"]):
                        return None
                    alpha = float(cfg["alpha_think"])
                else:
                    alpha = None
                key = (name, temp, alpha)
                if first is None:
                    first = key
                elif key != first:
                    return None
                n += 1
            return first if n >= 2 else None

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            # logits shape: (num_requests, vocab)
            if not self._cfg:
                return logits
            # Fast path: one phase-independent config across all rows → mask the
            # whole sub-batch in a single call (avoids the launch-bound per-row
            # loop). Byte-equivalent to the per-row path (same dim=-1 reductions).
            # Loop-force needs per-row n-gram detection, so it cannot use the
            # batched fast path. When no request opts into loop-force (the
            # default), this is False and behaviour is unchanged.
            loop_active = any("loop_n" in c for c in self._cfg.values())
            uniform = None if loop_active else self._uniform_cfg()
            if uniform is not None:
                idxs = [i for i in self._cfg if i < logits.size(0)]
                if len(idxs) >= 2:
                    name, temp, alpha = uniform
                    idx_t = torch.as_tensor(idxs, device=logits.device, dtype=torch.long)
                    sub = logits[idx_t]
                    if temp != 1.0:
                        sub = sub / temp
                    if name == "pless_alpha":
                        sub = _pless_alpha_mask_logits(sub, alpha=alpha)
                    else:
                        sub = _SAMPLER_LOGIT_FN[name](sub)
                    logits[idx_t] = sub
                    return logits
            # General per-row path (split / mixed configs): unchanged.
            for idx, cfg in self._cfg.items():
                if idx >= logits.size(0):
                    # Stale dict entry; should not happen if update_state is
                    # called every step but defensive.
                    continue
                # Model-aware </think> id (set per-request in cfg; falls back to
                # the Qwen3 default if a caller didn't supply one).
                think_end_id = cfg.get("think_end_id", QWEN3_THINK_END_TOKEN_ID)
                in_code = self._in_code.get(idx, False)
                if not in_code:
                    # Cheap check: scan only the last token (the one just
                    # appended this step). If it was THINK_END, flip phase.
                    out = self._out.get(idx)
                    if out and out[-1] == think_end_id:
                        in_code = True
                        self._in_code[idx] = True
                    elif out and think_end_id in out:
                        # Defensive (e.g. tokens appended in bulk after a
                        # restore from snapshot); pay the O(n) scan once.
                        in_code = True
                        self._in_code[idx] = True

                # Live loop-force: in the think phase, if the n-gram detector has
                # fired (or fires now), force </think> this step — mask the row to
                # the THINK_END token only. vLLM then samples </think>, and next
                # step in_code flips to True (code phase). Opt-in via cfg["loop_n"].
                if not in_code and "loop_n" in cfg:
                    out = self._out.get(idx) or []
                    if not self._loop_fired.get(idx, False):
                        if len(out) % _LOOP_CHECK_EVERY == 0 and ngram_loop_fired(
                            out, cfg["loop_n"], cfg["loop_k"], cfg.get("loop_window", 400)
                        ):
                            self._loop_fired[idx] = True
                    if self._loop_fired.get(idx, False):
                        row = logits[idx]
                        row[:] = float("-inf")
                        row[think_end_id] = 0.0
                        logits[idx] = row
                        continue   # skip normal pless masking for this forced row

                if in_code:
                    temp = float(cfg["t_code"])
                    sampler_name = cfg["sampler_code"]
                    alpha_key = "alpha_code"
                else:
                    temp = float(cfg["t_think"])
                    sampler_name = cfg["sampler_think"]
                    alpha_key = "alpha_think"

                row = logits[idx]
                if temp != 1.0:
                    row = row / temp
                if sampler_name == "pless_alpha":
                    row = _pless_alpha_mask_logits(row, alpha=float(cfg[alpha_key]))
                else:
                    row = _SAMPLER_LOGIT_FN[sampler_name](row)
                logits[idx] = row
            return logits

    _build_pless_split_logits_processor_class._cached = PlessSplitLogitsProcessor  # type: ignore[attr-defined]
    return PlessSplitLogitsProcessor


_build_pless_split_logits_processor_class._cached = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


def load_engine(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.90,
    register_pless_logitsproc: bool = True,
    **kwargs,
):
    """Construct a vLLM LLM engine. Deferred-import wrapper.

    Mirrors `bench/generator.py:load_model_and_tokenizer` in role — call
    once before the per-task loop, reuse for all generations.

    vLLM 0.21+ requires custom LogitsProcessor *classes* to be registered
    at engine init via ``LLM(logits_processors=[...])`` rather than
    per-request via ``SamplingParams(logits_processors=...)`` (the
    latter was removed). When ``register_pless_logitsproc=True``
    (default), the PlessSplitLogitsProcessor is auto-registered so the
    pless / pless_norm / pless_alpha samplers work via per-request
    ``extra_args``. Pass ``register_pless_logitsproc=False`` only for
    bare temp-sampling workflows that don't need our custom processor.
    """
    from vllm import LLM
    if register_pless_logitsproc:
        processor_cls = _build_pless_split_logits_processor_class()
        # Use user-provided 'logits_processors' kwarg if any, else default to ours.
        existing = kwargs.pop("logits_processors", None)
        logits_processors = (existing or []) + [processor_cls]
    else:
        logits_processors = kwargs.pop("logits_processors", None)
    engine = LLM(
        model=model_id,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        logits_processors=logits_processors,
        **kwargs,
    )
    # Detect broken byte-level BPE decoder (same bug worked around in
    # bench/generator.py:108-117 for the HF backend). vLLM's internal
    # tokenizer for OCI / DeepSeek-Coder loads as slow LlamaTokenizer and
    # either strips whitespace or emits literal `Ġ`/`Ċ` BPE markers.
    # Stash a known-good PreTrainedTokenizerFast on the engine so callers
    # can re-decode token_ids when needed.
    _maybe_install_safe_tokenizer(engine, model_id)
    return engine


def _maybe_install_safe_tokenizer(engine, model_id: str) -> None:
    """Round-trip 'a b\\nc' through vLLM's tokenizer; if broken, attach a
    PreTrainedTokenizerFast as ``engine._safe_tokenizer`` for callers to
    re-decode generated token ids with.
    """
    test_str = "a b\nc"
    tok = engine.get_tokenizer()
    try:
        ids = tok.encode(test_str, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        decoded = None
    if decoded == test_str.strip():
        return  # tokenizer is fine
    from transformers import PreTrainedTokenizerFast
    safe = PreTrainedTokenizerFast.from_pretrained(model_id)
    safe_decoded = safe.decode(
        safe.encode(test_str, add_special_tokens=False),
        skip_special_tokens=True,
    ).strip()
    if safe_decoded != test_str.strip():
        raise RuntimeError(
            f"vLLM tokenizer for {model_id!r} round-trips whitespace "
            f"incorrectly ({decoded!r}) AND PreTrainedTokenizerFast also "
            f"failed ({safe_decoded!r}); cannot recover."
        )
    engine._safe_tokenizer = safe
    print(
        f"[vllm] Installed safe PreTrainedTokenizerFast for {model_id!r} "
        f"(vLLM's default tokenizer mangled whitespace: {decoded!r})"
    )


def _extract_completion_texts(request_output, engine) -> list[str]:
    """Return decoded text per completion, using engine._safe_tokenizer
    to re-decode token_ids if vLLM's default decoder is broken for this
    model. Falls back to completion.text otherwise.
    """
    safe = getattr(engine, "_safe_tokenizer", None)
    if safe is None:
        return [completion.text for completion in request_output.outputs]
    return [
        safe.decode(list(completion.token_ids), skip_special_tokens=True)
        for completion in request_output.outputs
    ]


def _verify_think_end_id(engine) -> int:
    """Resolve and return the loaded model's single-token ``</think>`` id from the
    engine tokenizer (model-aware; see :func:`resolve_think_end_id`).

    Replaces the old hardcoded-equality assertion: we still fail loudly if
    ``</think>`` is not a single token (the processor can force only one id, and
    would otherwise sample from the wrong phase forever), but the id itself comes
    from whatever model is loaded — Qwen3 → 151668, DeepSeek → 128014.
    """
    return resolve_think_end_id(engine.get_tokenizer())


# ---------------------------------------------------------------------------
# Public generation API — matching the HF backend's signatures
# ---------------------------------------------------------------------------


def generate_samples_split_vllm(
    engine,
    tokenizer,  # unused — vLLM owns its own tokenizer; argument kept for API parity
    prompt_text: str | list[int],
    sampler_fn_think: str,
    sampler_fn_code: str,
    n_samples: int,
    max_new_tokens: int,
    temperature_think: float,
    temperature_code: float,
    stop_strings: list[str] | None = None,
    think_end_token_id: int | None = None,  # default: resolve from the model tokenizer
) -> list[str]:
    """vLLM-backed split decoding. Same return type as the HF version.

    Differences from the HF entry point:
      * ``sampler_fn_think`` / ``sampler_fn_code`` are **strings** here
        (``"pless"``, ``"pless_norm"``, ``"temp_pure"``, ``"temp_standard"``),
        not the callable factories from ``bench/sampler_bridge.py``. The
        vLLM ``LogitsProcessor`` doesn't take Python callbacks per-step;
        the sampler is selected by name inside the processor.
      * ``tokenizer`` is unused here (vLLM owns its tokenizer). Kept in
        the signature so the runner-side dispatch code stays symmetric
        with the HF version.
    """
    from vllm import SamplingParams

    think_end_id = (
        think_end_token_id if think_end_token_id is not None
        else _verify_think_end_id(engine)
    )
    processor_cls = _build_pless_split_logits_processor_class()

    # NOTE: vLLM 0.21+ removed SamplingParams.logits_processors. The
    # processor class must be pre-registered at engine load via
    # ``load_engine(..., register_pless_logitsproc=True)`` (the default)
    # or by passing ``logits_processors=[processor_cls]`` to ``LLM()``.
    # Per-request activation happens via ``extra_args`` below.
    sp = SamplingParams(
        n=n_samples,
        max_tokens=max_new_tokens,
        temperature=1.0,    # our LogitsProcessor handles temperature itself
        top_p=1.0,
        top_k=-1,
        stop=stop_strings or None,
        extra_args={
            processor_cls.EXTRA_ARG_KEY: {
                "t_think":       float(temperature_think),
                "t_code":        float(temperature_code),
                "sampler_think": sampler_fn_think,
                "sampler_code":  sampler_fn_code,
                "think_end_id":  think_end_id,
            },
        },
    )

    # Accept either a string prompt or a pre-tokenized list of ids (parity
    # with the HF backend, which supports both for old-Qwen direct tokenize).
    if isinstance(prompt_text, list):
        from vllm import TokensPrompt
        prompt = TokensPrompt(prompt_token_ids=prompt_text)
    else:
        prompt = prompt_text

    outputs = engine.generate([prompt], sp, use_tqdm=False)
    # outputs is a list of RequestOutput, one per prompt; we sent one prompt.
    request_out = outputs[0]
    return _extract_completion_texts(request_out, engine)


def generate_samples_vllm(
    engine,
    tokenizer,
    prompt_text: str | list[int],
    sampler_name: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
    alpha: float | None = None,
    loop_ngram_n: int | None = None,
    loop_ngram_k: int | None = None,
    loop_window: int = 400,
) -> list[str]:
    """vLLM-backed single-sampler generation (no think/code split).

    Matches the role of ``bench/generator.py:generate_samples`` and is the
    target for MBPP / HumanEval runs that use a uniform sampler.

    ``alpha`` is required when ``sampler_name == "pless_alpha"`` and is
    propagated to both phases (uniform α across think + code).

    When ``loop_ngram_n`` and ``loop_ngram_k`` are set, the processor runs live
    n-gram loop detection in the THINK phase and forces </think> on detection
    (the deployable "detect-rambling → end-thinking" intervention).
    """
    from vllm import SamplingParams

    processor_cls = _build_pless_split_logits_processor_class()
    if sampler_name == "pless_alpha" and alpha is None:
        raise ValueError("alpha is required when sampler_name='pless_alpha'")
    # We reuse the split processor by setting t_think == t_code and
    # sampler_think == sampler_code; the </think> detection becomes a
    # no-op since both phases are identical.
    cfg = {
        "t_think":       float(temperature),
        "t_code":        float(temperature),
        "sampler_think": sampler_name,
        "sampler_code":  sampler_name,
    }
    if sampler_name == "pless_alpha":
        cfg["alpha_think"] = float(alpha)
        cfg["alpha_code"] = float(alpha)
    if loop_ngram_n and loop_ngram_k:
        cfg["loop_n"] = int(loop_ngram_n)
        cfg["loop_k"] = int(loop_ngram_k)
        cfg["loop_window"] = int(loop_window)
    # Model-aware </think> id for think→code detection + loop-force masking
    # (Qwen3 → 151668, DeepSeek-R1-Distill → 128014). Without this the processor
    # falls back to the Qwen3 default and IndexErrors on smaller-vocab models.
    cfg["think_end_id"] = resolve_think_end_id(engine.get_tokenizer())
    # NOTE: vLLM 0.21+ removed SamplingParams.logits_processors; the
    # processor class is pre-registered at engine load (see load_engine).
    sp = SamplingParams(
        n=n_samples,
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        stop=stop_strings or None,
        extra_args={processor_cls.EXTRA_ARG_KEY: cfg},
    )

    if isinstance(prompt_text, list):
        from vllm import TokensPrompt
        prompt = TokensPrompt(prompt_token_ids=prompt_text)
    else:
        prompt = prompt_text

    outputs = engine.generate([prompt], sp, use_tqdm=False)
    return _extract_completion_texts(outputs[0], engine)


def generate_samples_standard_vllm(
    engine,
    tokenizer,
    prompt_text: str | list[int],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """vLLM-backed standard temperature / top-p / top-k sampling. No custom processor.

    Counterpart to ``bench/generator.py:generate_samples_standard``.
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        n=n_samples,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        repetition_penalty=repetition_penalty,
        stop=stop_strings or None,
    )

    if isinstance(prompt_text, list):
        from vllm import TokensPrompt
        prompt = TokensPrompt(prompt_token_ids=prompt_text)
    else:
        prompt = prompt_text

    outputs = engine.generate([prompt], sp, use_tqdm=False)
    return _extract_completion_texts(outputs[0], engine)
