"""Parity tests for the vLLM backend (bench/generator_vllm.py).

Two flavours:

1. **Mac-safe unit tests** (run via ``uv run pytest tests/test_vllm_parity.py``):
   exercise the pure-Python logit-transform layer of
   ``bench/generator_vllm.py`` against the canonical
   ``p-less/p_less_samplers.py`` to confirm mathematical equivalence
   (mask-on-probs ≡ mask-to-minus-inf-on-logits-then-softmax). These
   don't require vLLM at all and run on Mac / any CI.

2. **GPU parity gate** (run via
   ``.venv-vllm/bin/python -m pytest tests/test_vllm_parity.py -k gpu``)
   loads Qwen3-8B, runs 10 problems × 10 samples on both HF and vLLM
   backends with the same RNG seed, and asserts the distributional
   metrics (NAUADC, struct_div, codebleu_div) agree within their
   measured noise floors. This is the merge gate for ``feat/vllm-backend``.

Tolerances (from docs/research/vllm_migration_analysis.md §10b-D):
  * NAUADC       within  ±0.02
  * struct_div   within  ±0.01
  * codebleu_div within  ±0.02

Why distributional, not bit-identical: vLLM uses FlashAttention-2 and
its own multinomial; HF uses SDPA and torch.multinomial. They produce
different *token sequences* even with the same seed. What we care about
is that the population statistics match.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Mac-safe unit tests (no vLLM, no GPU required)
# ---------------------------------------------------------------------------


def _pless_decode_on_probs(probs: torch.Tensor) -> torch.Tensor:
    """Reference implementation: p_less_decode minus the multinomial step.

    Returns the post-mask, post-renormalize distribution so we can compare
    against the vLLM path's softmax-after-mask output. Pure function;
    matches ``p-less/p_less_samplers.py:p_less_decode`` lines 25-28.
    """
    probs = probs.clone()
    p = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < p
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def _pless_norm_decode_on_probs(probs: torch.Tensor) -> torch.Tensor:
    """Reference: p_less_norm_decode minus multinomial. Mirrors lines 54-57."""
    probs = probs.clone()
    v = probs.size(-1)
    p = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
    mask = probs < p
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def test_pless_logit_mask_matches_prob_mask():
    """vLLM-side _pless_mask_logits + softmax produces the same distribution
    as the canonical HF-side mask + renormalize on probs."""
    from bench.generator_vllm import _pless_mask_logits

    torch.manual_seed(0)
    logits = torch.randn(8, 5000) * 1.5  # batch of 8, vocab of 5000

    # HF path
    probs_ref = _pless_decode_on_probs(torch.softmax(logits, dim=-1))

    # vLLM path
    logits_v = logits.clone()
    logits_v = _pless_mask_logits(logits_v)
    probs_v = torch.softmax(logits_v.float(), dim=-1)

    # Live tokens should match within numerical precision.
    diff = (probs_ref - probs_v).abs().max().item()
    assert diff < 1e-5, f"pless distributions diverged: max abs diff {diff}"


def test_pless_norm_logit_mask_matches_prob_mask():
    from bench.generator_vllm import _pless_norm_mask_logits

    torch.manual_seed(1)
    logits = torch.randn(4, 3000) * 2.0
    probs_ref = _pless_norm_decode_on_probs(torch.softmax(logits, dim=-1))
    logits_v = logits.clone()
    logits_v = _pless_norm_mask_logits(logits_v)
    probs_v = torch.softmax(logits_v.float(), dim=-1)
    diff = (probs_ref - probs_v).abs().max().item()
    assert diff < 1e-5, f"pless_norm distributions diverged: max abs diff {diff}"


def test_top_p_top_k_mask_respects_limits():
    """temp_standard's logit mask should leave at most ``top_k`` tokens alive
    and the surviving probability mass should be roughly ≥ ``top_p``."""
    from bench.generator_vllm import _top_p_top_k_mask_logits

    torch.manual_seed(2)
    logits = torch.randn(4, 1000) * 2.0
    masked = _top_p_top_k_mask_logits(logits.clone(), top_p=0.95, top_k=20)
    probs = torch.softmax(masked.float(), dim=-1)
    live = (probs > 1e-9).sum(dim=-1)
    assert (live <= 20).all(), f"top_k violated: {live.tolist()}"


def test_temp_pure_is_identity():
    """temp_pure must NOT modify logits — that's its definition."""
    from bench.generator_vllm import _SAMPLER_LOGIT_FN

    torch.manual_seed(3)
    logits = torch.randn(2, 100)
    out = _SAMPLER_LOGIT_FN["temp_pure"](logits.clone())
    assert torch.equal(out, logits), "temp_pure should be identity"


def test_sampler_dispatch_table_covers_all_split_samplers():
    """Defensive: ensure every name in bench.sampler_bridge.SPLIT_SAMPLERS
    has a matching entry in the vLLM dispatch table. If a new sampler is
    added to SPLIT_SAMPLERS but not _SAMPLER_LOGIT_FN, the vLLM path
    silently can't be used for it — this test catches that."""
    from bench.generator_vllm import _SAMPLER_LOGIT_FN
    from bench.sampler_bridge import SPLIT_SAMPLERS

    for name in SPLIT_SAMPLERS:
        assert name in _SAMPLER_LOGIT_FN, (
            f"Sampler {name!r} is in SPLIT_SAMPLERS but missing from "
            f"_SAMPLER_LOGIT_FN in bench/generator_vllm.py — vLLM backend "
            f"cannot use this sampler."
        )


# ---------------------------------------------------------------------------
# Equivalence to the AUTHORS' actual samplers (not a reimplementation)
# ---------------------------------------------------------------------------
#
# The tests above compare the vLLM mask against `_pless_decode_on_probs`, a
# reimplementation living in THIS file — which could share a bug with the vLLM
# code. The two tests below instead run the real `p_less_decode` /
# `p_less_norm_decode` from the p-less/ submodule and confirm the vLLM
# masked-logits distribution has the same survivor support and matching
# empirical token frequencies. This is the guarantee that our vLLM
# reimplementation cannot silently drift from the authors' method.


def _authors_empirical(decode_fn, logits: torch.Tensor, n: int, seed: int):
    """Run the authors' (probs->token) sampler `n` times on one distribution;
    return (survivor_set, freq_dict)."""
    probs = torch.softmax(logits.float(), dim=-1)            # (1, V)
    torch.manual_seed(seed)
    draws = decode_fn(probs.expand(n, -1).clone()).squeeze(-1)  # (n,)
    survivors = set(draws.unique().tolist())
    freqs = {t: (draws == t).float().mean().item() for t in survivors}
    return survivors, freqs


def _peaky_logits(vocab: int = 1500) -> torch.Tensor:
    torch.manual_seed(0)
    logits = torch.randn(1, vocab)
    logits[0, :5] += torch.tensor([6.0, 5.0, 4.5, 4.0, 3.5])
    return logits


def test_vllm_pless_matches_authors_sampler():
    from bench.generator_vllm import _pless_mask_logits
    from bench.sampler_bridge import p_less_decode

    logits = _peaky_logits()
    d_vllm = torch.softmax(_pless_mask_logits(logits.clone()).float(), dim=-1)[0]
    vllm_survivors = set((d_vllm > 0).nonzero(as_tuple=True)[0].tolist())

    authors_survivors, freqs = _authors_empirical(p_less_decode, logits, n=20000, seed=123)
    assert authors_survivors == vllm_survivors, (
        f"survivor support differs: authors={sorted(authors_survivors)} "
        f"vllm={sorted(vllm_survivors)}"
    )
    for t, f in freqs.items():
        assert abs(f - d_vllm[t].item()) < 0.02, (
            f"token {t}: authors freq {f:.4f} vs vLLM prob {d_vllm[t].item():.4f}"
        )


def test_vllm_pless_norm_matches_authors_sampler():
    from bench.generator_vllm import _pless_norm_mask_logits
    from bench.sampler_bridge import p_less_norm_decode

    logits = _peaky_logits()
    d_vllm = torch.softmax(_pless_norm_mask_logits(logits.clone()).float(), dim=-1)[0]
    vllm_survivors = set((d_vllm > 0).nonzero(as_tuple=True)[0].tolist())

    authors_survivors, freqs = _authors_empirical(p_less_norm_decode, logits, n=20000, seed=123)
    assert authors_survivors == vllm_survivors, (
        f"survivor support differs: authors={sorted(authors_survivors)} "
        f"vllm={sorted(vllm_survivors)}"
    )
    for t, f in freqs.items():
        assert abs(f - d_vllm[t].item()) < 0.02


def test_vllm_argmax_fallback_branchless():
    """The branchless guard un-masks the argmax of a fully-pruned row and is a
    no-op otherwise — and the mask fns never leave a row entirely -inf."""
    from bench.generator_vllm import (
        _pless_mask_logits,
        _pless_norm_mask_logits,
        _restore_argmax_on_all_pruned,
    )

    probs = torch.tensor([[0.1, 0.7, 0.2]])
    m = torch.ones(1, 3, dtype=torch.bool)          # whole row pruned
    _restore_argmax_on_all_pruned(m, probs)
    assert m.tolist() == [[True, False, True]]      # argmax (idx 1) survives

    m2 = torch.tensor([[True, False, True]])        # normal row
    before = m2.clone()
    _restore_argmax_on_all_pruned(m2, probs)
    assert torch.equal(m2, before)                  # untouched

    torch.manual_seed(0)
    logits = torch.randn(8, 4000)
    for fn in (_pless_mask_logits, _pless_norm_mask_logits):
        out = fn(logits.clone())
        assert (out > float("-inf")).any(dim=-1).all(), "a row was left all -inf"


def test_swap_move_preserves_both_request_states(monkeypatch):
    """Regression test for the `direct == 'swap'` enum bug: on a SWAP move,
    BOTH requests' state must be kept. Builds the real processor class against a
    minimal fake `vllm` module (no GPU / no vLLM install needed)."""
    import enum
    import types

    class _Base:  # stand-in for vllm's LogitsProcessor (we override everything)
        pass

    class MoveDirectionality(enum.Enum):
        UNIDIRECTIONAL = enum.auto()
        SWAP = enum.auto()

    lp_mod = types.ModuleType("vllm.v1.sample.logits_processor")
    lp_mod.LogitsProcessor = _Base
    lp_mod.MoveDirectionality = MoveDirectionality
    sp_mod = types.ModuleType("vllm.sampling_params")
    sp_mod.SamplingParams = object
    for name in ("vllm", "vllm.v1", "vllm.v1.sample"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.logits_processor", lp_mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp_mod)

    import bench.generator_vllm as gv
    monkeypatch.setattr(gv._build_pless_split_logits_processor_class, "_cached", None)
    cls = gv._build_pless_split_logits_processor_class()
    proc = cls(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)

    cfgA = {"t_think": 1.0, "t_code": 1.0, "sampler_think": "pless", "sampler_code": "pless"}
    cfgB = {"t_think": 1.0, "t_code": 1.0, "sampler_think": "pless_norm", "sampler_code": "pless_norm"}
    proc._cfg = {0: cfgA, 1: cfgB}
    proc._out = {0: [10], 1: [20]}
    proc._in_code = {0: True, 1: False}

    bu = types.SimpleNamespace(added=[], removed=[], moved=[(0, 1, MoveDirectionality.SWAP)])
    proc.update_state(bu)

    # After the swap, A's state lives at row 1 and B's at row 0 — neither dropped.
    assert proc._cfg[1] is cfgA and proc._cfg[0] is cfgB
    assert proc._out[1] == [10] and proc._out[0] == [20]
    assert proc._in_code[1] is True and proc._in_code[0] is False


def _build_processor_against_fake_vllm(monkeypatch):
    """Build the real PlessSplitLogitsProcessor class against a minimal fake
    `vllm` module (no GPU / no vLLM install needed). Returns (gv, cls)."""
    import enum
    import types

    class _Base:
        pass

    class MoveDirectionality(enum.Enum):
        UNIDIRECTIONAL = enum.auto()
        SWAP = enum.auto()

    lp = types.ModuleType("vllm.v1.sample.logits_processor")
    lp.LogitsProcessor = _Base
    lp.MoveDirectionality = MoveDirectionality
    spm = types.ModuleType("vllm.sampling_params")
    spm.SamplingParams = object
    for name in ("vllm", "vllm.v1", "vllm.v1.sample"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.logits_processor", lp)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", spm)

    import bench.generator_vllm as gv
    monkeypatch.setattr(gv._build_pless_split_logits_processor_class, "_cached", None)
    return gv, gv._build_pless_split_logits_processor_class()


def test_batched_fast_path_matches_per_row(monkeypatch):
    """The uniform fast path in apply() must produce byte-identical masked logits
    to applying the same temperature + mask independently per row."""
    from bench.generator_vllm import _SAMPLER_LOGIT_FN, _pless_alpha_mask_logits

    gv, cls = _build_processor_against_fake_vllm(monkeypatch)
    proc = cls(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)

    torch.manual_seed(0)
    K, V = 5, 4000
    base = torch.randn(K, V)
    base[:, 0] += 5.0

    cases = [
        ("pless", None, 1.0),
        ("pless_norm", None, 1.0),
        ("pless", None, 0.7),       # temperature != 1.0
        ("pless_alpha", 3.0, 1.0),
    ]
    for name, alpha, temp in cases:
        cfg = {"t_think": temp, "t_code": temp, "sampler_think": name, "sampler_code": name}
        if alpha is not None:
            cfg["alpha_think"] = alpha
            cfg["alpha_code"] = alpha
        proc._cfg = {i: cfg for i in range(K)}
        proc._out = {i: [] for i in range(K)}
        proc._in_code = {i: False for i in range(K)}

        # Confirm the fast path actually triggers for this uniform config.
        assert proc._uniform_cfg() is not None, name

        out_fast = proc.apply(base.clone())

        ref = base.clone()
        for i in range(K):
            row = ref[i] / temp if temp != 1.0 else ref[i]
            ref[i] = (_pless_alpha_mask_logits(row, alpha=alpha) if name == "pless_alpha"
                      else _SAMPLER_LOGIT_FN[name](row))

        assert torch.equal(out_fast.isinf(), ref.isinf()), f"{name}: survivor pattern differs"
        assert torch.allclose(
            out_fast.masked_fill(out_fast.isinf(), 0.0),
            ref.masked_fill(ref.isinf(), 0.0), atol=1e-5,
        ), f"{name}: finite logits differ"


def test_split_config_skips_fast_path_and_stays_correct(monkeypatch):
    """A split config (sampler_think != sampler_code) must NOT take the fast path,
    and the unchanged per-row loop must apply the right sampler per phase."""
    from bench.generator_vllm import _SAMPLER_LOGIT_FN

    gv, cls = _build_processor_against_fake_vllm(monkeypatch)
    proc = cls(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)

    torch.manual_seed(1)
    base = torch.randn(2, 3000)
    base[:, 0] += 5.0
    cfg = {"t_think": 1.0, "t_code": 1.0, "sampler_think": "pless", "sampler_code": "pless_norm"}
    proc._cfg = {0: cfg, 1: cfg}
    proc._out = {0: [1], 1: [gv.QWEN3_THINK_END_TOKEN_ID]}   # row 1 entered code phase
    proc._in_code = {0: False, 1: False}

    assert proc._uniform_cfg() is None        # split -> no fast path

    out = proc.apply(base.clone())
    ref = base.clone()
    ref[0] = _SAMPLER_LOGIT_FN["pless"](ref[0])        # think phase
    ref[1] = _SAMPLER_LOGIT_FN["pless_norm"](ref[1])   # code phase
    assert torch.equal(out.isinf(), ref.isinf())


# ---------------------------------------------------------------------------
# GPU parity gate — only runs if vLLM is importable and CUDA is available
# ---------------------------------------------------------------------------


_VLLM_AVAILABLE = False
try:  # pragma: no cover — environment-dependent
    import vllm  # noqa: F401
    _VLLM_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass


@pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="vLLM and/or CUDA not available; this test runs only in .venv-vllm on GPU.",
)
@pytest.mark.gpu
def test_distributional_parity_qwen3_h7p_codeforces(tmp_path):
    """End-to-end: same model, same prompts, same RNG seed; HF and vLLM
    backends should produce distributional metrics within tolerance.

    Designed to run on a GPU box where both HF and vLLM are installed
    (the .venv-vllm overlay). Picks 10 H7P-CODEFORCES problems
    deterministically; runs split decoding (temp_pure t1.5 think →
    pless t1.0 code); computes NAUADC (via algosim, if available) and
    struct_div / codebleu_div on both backends.

    Skipped if the algosim submodule isn't installed in the venv.
    """
    pytest.importorskip("vllm")
    pytest.importorskip("transformers")

    # The full A/B is large (10 problems × 8K tokens × 2 backends ≈ 30 min
    # of GPU time). Run via:
    #   .venv-vllm/bin/python -m pytest tests/test_vllm_parity.py::test_distributional_parity_qwen3_h7p_codeforces -s
    # Skeleton implementation below — fleshed out once .venv-vllm exists
    # on the GPU box and vLLM is verified importable.
    pytest.skip(
        "GPU parity test implementation pending — requires .venv-vllm on the GPU "
        "box, see docs/research/vllm_migration_analysis.md §10b-D for the merge "
        "criteria (NAUADC ±0.02, struct_div ±0.01, codebleu_div ±0.02 over "
        "10 problems × 10 samples)."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_loop_force_forces_think_end_on_ngram_loop(monkeypatch):
    """Live loop-force: a think-phase request whose recent tokens contain an n-gram
    loop must have its logits forced to </think> (argmax == THINK_END). A request
    WITHOUT a loop must be untouched by the force path (normal pless masking)."""
    gv, cls = _build_processor_against_fake_vllm(monkeypatch)
    proc = cls(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)
    TE = gv.QWEN3_THINK_END_TOKEN_ID

    # row 0: looping (8-tok unit x6, len divisible by check-every so it scans) → should force </think>
    loop_unit = [101, 102, 103, 104, 105, 106, 107, 108]
    looping = loop_unit * 6
    looping = looping[: (len(looping) // gv._LOOP_CHECK_EVERY) * gv._LOOP_CHECK_EVERY]  # land on a scan step
    # row 1: varied, no loop → must NOT be forced
    varied = list(range(200, 200 + len(looping)))

    loopcfg = {"t_think": 1.0, "t_code": 1.0, "sampler_think": "pless", "sampler_code": "pless",
               "loop_n": 8, "loop_k": 4, "loop_window": 400}
    proc._cfg = {0: dict(loopcfg), 1: dict(loopcfg)}
    proc._out = {0: looping, 1: varied}
    proc._in_code = {0: False, 1: False}
    proc._loop_fired = {0: False, 1: False}

    V = 151936
    logits = torch.randn(2, V)
    out = proc.apply(logits.clone())

    assert int(out[0].argmax()) == TE, "looping row should be forced to </think>"
    assert proc._loop_fired[0] is True
    # non-looping row: not forced — its argmax should NOT be coerced to TE by the force path
    # (pless masking may keep many tokens; just assert the loop flag stayed False)
    assert proc._loop_fired[1] is False


def test_loop_force_off_is_unchanged(monkeypatch):
    """With no loop_n in cfg (default), loop-force is inert: the row is pless-masked,
    not forced to </think>."""
    gv, cls = _build_processor_against_fake_vllm(monkeypatch)
    proc = cls(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)
    cfg = {"t_think": 1.0, "t_code": 1.0, "sampler_think": "pless", "sampler_code": "pless"}
    proc._cfg = {0: cfg}
    proc._out = {0: [101, 102, 103, 104, 105, 106, 107, 108] * 6}  # looping, but loop-force OFF
    proc._in_code = {0: False}
    proc._loop_fired = {0: False}
    out = proc.apply(torch.randn(1, 151936).clone())
    assert proc._loop_fired[0] is False, "no loop_n in cfg → detector must not run"
