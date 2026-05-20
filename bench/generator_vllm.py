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


def _pless_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """In-place: zero out (set to -inf) tokens below the p-less threshold."""
    probs = torch.softmax(logits.float(), dim=-1)
    p = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < p
    logits.masked_fill_(mask, float("-inf"))
    return logits


def _pless_norm_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """In-place: zero out tokens below the p-less-norm threshold."""
    probs = torch.softmax(logits.float(), dim=-1)
    v = probs.size(-1)
    p = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
    mask = probs < p
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
    all_pruned = mask.all(dim=-1)
    if all_pruned.any():
        fallback_idx = probs[all_pruned].argmax(dim=-1, keepdim=True)
        mask[all_pruned] = mask[all_pruned].scatter(-1, fallback_idx, False)
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
    from vllm.v1.sample.logits_processor import LogitsProcessor

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
                    continue
                self._cfg[idx] = cfg
                self._out[idx] = output_ids  # reference; grows automatically
                self._in_code[idx] = False
            # Removed requests
            for idx in batch_update.removed:
                self._cfg.pop(idx, None)
                self._out.pop(idx, None)
                self._in_code.pop(idx, None)
            # Moved requests — adx → bdx; ``direct`` distinguishes move vs swap.
            for adx, bdx, direct in batch_update.moved:
                a_cfg = self._cfg.pop(adx, None)
                b_cfg = self._cfg.pop(bdx, None)
                a_out = self._out.pop(adx, None)
                b_out = self._out.pop(bdx, None)
                a_code = self._in_code.pop(adx, None)
                b_code = self._in_code.pop(bdx, None)
                if a_cfg is not None:
                    self._cfg[bdx] = a_cfg
                    self._out[bdx] = a_out  # type: ignore[assignment]
                    self._in_code[bdx] = a_code if a_code is not None else False
                if direct == "swap" and b_cfg is not None:
                    self._cfg[adx] = b_cfg
                    self._out[adx] = b_out  # type: ignore[assignment]
                    self._in_code[adx] = b_code if b_code is not None else False

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            # logits shape: (num_requests, vocab)
            if not self._cfg:
                return logits
            for idx, cfg in self._cfg.items():
                if idx >= logits.size(0):
                    # Stale dict entry; should not happen if update_state is
                    # called every step but defensive.
                    continue
                in_code = self._in_code.get(idx, False)
                if not in_code:
                    # Cheap check: scan only the last token (the one just
                    # appended this step). If it was THINK_END, flip phase.
                    out = self._out.get(idx)
                    if out and out[-1] == QWEN3_THINK_END_TOKEN_ID:
                        in_code = True
                        self._in_code[idx] = True
                    elif out and QWEN3_THINK_END_TOKEN_ID in out:
                        # Defensive (e.g. tokens appended in bulk after a
                        # restore from snapshot); pay the O(n) scan once.
                        in_code = True
                        self._in_code[idx] = True

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
    return LLM(
        model=model_id,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        logits_processors=logits_processors,
        **kwargs,
    )


def _verify_think_end_id(engine) -> None:
    """Runtime assertion: tokenizer's </think> id matches QWEN3_THINK_END_TOKEN_ID.

    This is the vLLM equivalent of the assertion in
    ``bench/generator.py:generate_samples_split:517-521``. If the tokenizer
    disagrees, fail loudly so we don't silently sample from the wrong
    phase forever.
    """
    tok = engine.get_tokenizer()
    ids = tok.encode("</think>", add_special_tokens=False)
    if not (len(ids) == 1 and ids[0] == QWEN3_THINK_END_TOKEN_ID):
        raise AssertionError(
            f"Expected </think> to be single token id {QWEN3_THINK_END_TOKEN_ID}, "
            f"got {ids} from tokenizer {tok.__class__.__name__}"
        )


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
    think_end_token_id: int = QWEN3_THINK_END_TOKEN_ID,  # noqa: ARG001 — pinned by Qwen3
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

    _verify_think_end_id(engine)
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
    return [completion.text for completion in request_out.outputs]


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
) -> list[str]:
    """vLLM-backed single-sampler generation (no think/code split).

    Matches the role of ``bench/generator.py:generate_samples`` and is the
    target for MBPP / HumanEval runs that use a uniform sampler.

    ``alpha`` is required when ``sampler_name == "pless_alpha"`` and is
    propagated to both phases (uniform α across think + code).
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
    return [completion.text for completion in outputs[0].outputs]


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
        stop=stop_strings or None,
    )

    if isinstance(prompt_text, list):
        from vllm import TokensPrompt
        prompt = TokensPrompt(prompt_token_ids=prompt_text)
    else:
        prompt = prompt_text

    outputs = engine.generate([prompt], sp, use_tqdm=False)
    return [completion.text for completion in outputs[0].outputs]
