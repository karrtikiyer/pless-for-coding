"""Regression tests for the mlx-lm G_k (Rényi, `pless_renyi`) logits processor.

The mx port must reproduce the torch reference it is translated from,
`bench/generator_vllm.py::_pless_renyi_mask_logits`, i.e. the rooted form
G_k = (Σpᵢ^k)^{1/(k-1)} = exp(-H_k) -- NOT the raw power sum τ_k = Σpᵢ^k (the two
coincide only at k=2).

mlx is not in this repo's uv.lock (macOS-only). Point MLX_SITE_PACKAGES at a
site-packages dir that has mlx (it is APPENDED, never prepended, so this venv's
torch/numpy win), e.g.:
  MLX_SITE_PACKAGES=$HOME/.venv-vllm-metal/lib/python3.12/site-packages uv run pytest tests/test_mlx_pless_renyi.py
Without it, these tests skip.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

if os.environ.get("MLX_SITE_PACKAGES"):
    sys.path.append(os.environ["MLX_SITE_PACKAGES"])
mx = pytest.importorskip("mlx.core", reason="mlx not importable; set MLX_SITE_PACKAGES")

from bench.generator_vllm import _pless_renyi_mask_logits as torch_renyi  # noqa: E402
from bench.mlx_pless_logits_processors import (  # noqa: E402
    _pless_mask_logits as mx_pless,
    _pless_renyi_mask_logits as mx_renyi,
    make_pless_renyi_logits_processor,
)

# Existing Ministral full1046 grid + the requested extra points + the special cases.
K_GRID = [2.0, 1.6, 1.0, 0.8, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
          0.075, 0.05, 0.025, 0.0]
VOCABS = [131072, 32000, 7]  # Ministral-3 vocab, a mid vocab, a tiny edge case
BOUNDARY_RTOL = 1e-4  # measured max |p-G_k|/G_k of torch-vs-mx disagreements: 9.7e-6 (840 rows, vocab 131072)


def _rows(vocab: int, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        regime = i % 3
        if regime == 0:  # flat
            row = rng.normal(0.0, 0.5, vocab)
        elif regime == 1:  # one spike
            row = rng.normal(0.0, 1.0, vocab)
            row[rng.integers(vocab)] += rng.uniform(8.0, 20.0)
        else:  # two spikes
            row = rng.normal(0.0, 0.3, vocab)
            a, b = rng.choice(vocab, 2, replace=False)
            row[a] += rng.uniform(4.0, 10.0)
            row[b] += rng.uniform(4.0, 10.0)
        out.append(row.astype(np.float32))
    return out


def _survivors_torch(row: np.ndarray, k: float) -> np.ndarray:
    out = torch_renyi(torch.from_numpy(row.copy())[None, :], k)
    return np.isfinite(out[0].numpy())


def _survivors_mx(fn, row: np.ndarray, k: float | None = None) -> np.ndarray:
    m = mx.array(row)[None, :]
    out = fn(m) if k is None else fn(m, k)
    return np.isfinite(np.array(out[0].astype(mx.float32)))


def _threshold_f64(row: np.ndarray, k: float) -> tuple[np.ndarray, float]:
    """Independent float64 numpy reference for probs and G_k."""
    z = row.astype(np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    if k == 2.0:
        thr = float(np.sum(p * p))
    elif k == 1.0:
        nz = p[p > 0]
        thr = float(np.exp(np.sum(nz * np.log(nz))))
    elif k == 0.0:
        thr = 1.0 / p.size
    else:
        thr = float(np.sum(p ** k) ** (1.0 / (k - 1.0)))
    return p, thr


def _disagreements_off_boundary(row, k, a, b) -> list[int]:
    """Indices where two survivor masks differ AND the token is not within
    BOUNDARY_RTOL of the threshold (i.e. a real disagreement, not FP noise)."""
    p, thr = _threshold_f64(row, k)
    idx = np.nonzero(a != b)[0]
    return [int(i) for i in idx if abs(p[i] - thr) > BOUNDARY_RTOL * thr]


# ---------------------------------------------------------------------------
# Hand-checked values (independent of both implementations)
# ---------------------------------------------------------------------------

class TestHandChecked:
    # probs = [0.4, 0.35, 0.25] (logits = log probs).
    #   k=2:   Σp² = 0.16 + 0.1225 + 0.0625 = 0.345            -> keep {0, 1}
    #   k=0.5: (Σ√p)^(1/(0.5-1)) = (0.632456+0.591608+0.5)^-2 = 1.724064^-2 = 0.336429 -> keep {0, 1}
    #   unrooted τ_0.5 = Σ√p = 1.72407 > every p              -> would keep only argmax
    PROBS = np.array([0.4, 0.35, 0.25])

    def test_k2_threshold_value(self):
        _, thr = _threshold_f64(np.log(self.PROBS).astype(np.float32), 2.0)
        assert thr == pytest.approx(0.345, abs=1e-6)

    def test_k05_threshold_value(self):
        _, thr = _threshold_f64(np.log(self.PROBS).astype(np.float32), 0.5)
        assert thr == pytest.approx(0.336429, abs=1e-6)

    @pytest.mark.parametrize("k", [2.0, 0.5])
    def test_mx_keeps_top_two(self, k):
        row = np.log(self.PROBS).astype(np.float32)
        assert _survivors_mx(mx_renyi, row, k).tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# Parity with the torch reference
# ---------------------------------------------------------------------------

class TestParityWithTorchReference:
    @pytest.mark.parametrize("vocab", VOCABS)
    @pytest.mark.parametrize("k", K_GRID)
    def test_survivor_sets_agree_except_fp_boundary(self, vocab, k):
        for row in _rows(vocab, 6, seed=hash((vocab, k)) % 2**32):
            t = _survivors_torch(row, k)
            m = _survivors_mx(mx_renyi, row, k)
            assert m.any(), "every row must keep at least one token"
            off = _disagreements_off_boundary(row, k, t, m)
            assert off == [], f"vocab={vocab} k={k}: non-boundary disagreement at {off[:5]}"

    def test_boundary_check_would_catch_the_unrooted_power_sum(self):
        """Guard against a vacuous parity test: the τ_k = Σpᵢ^k bug (missing root)
        must produce off-boundary disagreements on a realistic row."""
        def unrooted(logits, k):
            probs = mx.softmax(logits.astype(mx.float32), axis=-1)
            thr = mx.sum(mx.power(probs, k), axis=-1, keepdims=True)
            return mx.where(probs < thr, mx.array(-mx.inf, dtype=logits.dtype), logits)

        row = _rows(32000, 3, seed=7)[2]
        bad = _survivors_mx(unrooted, row, 0.1)
        assert _disagreements_off_boundary(row, 0.1, _survivors_torch(row, 0.1), bad)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    @pytest.mark.parametrize("vocab", [131072, 32000])
    def test_k2_is_identical_to_plain_pless(self, vocab):
        for row in _rows(vocab, 6, seed=11):
            m = mx.array(row)[None, :]
            assert np.array_equal(np.array(mx_renyi(m, 2.0)), np.array(mx_pless(m)))

    def test_lower_k_never_drops_a_token(self):
        ks = sorted(k for k in K_GRID if k > 0)
        for row in _rows(32000, 6, seed=3):
            prev = None
            for k in ks:  # ascending k = tightening threshold
                cur = _survivors_mx(mx_renyi, row, k)
                if prev is not None:
                    gained = np.nonzero(cur & ~prev)[0]
                    p, thr = _threshold_f64(row, k)
                    assert all(abs(p[i] - thr) <= BOUNDARY_RTOL * thr for i in gained), \
                        f"k={k} admitted a token that a lower k rejected"
                prev = cur

    def test_preserves_shape_and_dtype_including_bf16(self):
        row = _rows(32000, 2, seed=5)[1]
        for dtype in (mx.float32, mx.bfloat16, mx.float16):
            m = mx.array(row)[None, :].astype(dtype)
            out = mx_renyi(m, 0.1)
            assert out.shape == m.shape and out.dtype == dtype
            kept = np.isfinite(np.array(out.astype(mx.float32)))[0]
            ref = np.array(m.astype(mx.float32))[0]
            assert np.array_equal(np.array(out.astype(mx.float32))[0][kept], ref[kept])

    def test_rows_are_masked_independently(self):
        rows = _rows(32000, 3, seed=9)
        batch = mx.array(np.stack(rows))
        out = np.array(mx_renyi(batch, 0.2))
        for i, row in enumerate(rows):
            single = np.array(mx_renyi(mx.array(row)[None, :], 0.2))[0]
            assert np.array_equal(out[i], single)

    def test_flat_row_never_empties_and_differs_only_at_the_threshold(self):
        """On a fully tied row every p = 1/v exactly equals G_k, so whether all
        tokens or only the argmax survive is decided by float rounding alone.
        Torch itself gives both outcomes (keeps 1 at k=2 and k=1, all 1000 at
        k=1.6), and mlx 0.32.4 differs from torch at k=0.5 (1 vs 1000) while
        mlx 0.32.1 matches. Real logits are never fully tied. Require a
        non-empty result and only threshold-tie disagreements."""
        flat = np.zeros(1000, np.float32)
        for k in (2.0, 1.6, 1.0, 0.5, 0.05, 0.0):
            m = _survivors_mx(mx_renyi, flat, k)
            assert m.any()
            assert _disagreements_off_boundary(flat, k, _survivors_torch(flat, k), m) == [], f"k={k}"
        for row in _rows(131072, 3, seed=13):
            for k in (0.025, 0.0):
                assert _survivors_mx(mx_renyi, row, k).any()


# ---------------------------------------------------------------------------
# mlx-lm processor contract
# ---------------------------------------------------------------------------

class TestProcessorFactory:
    def test_processor_signature_and_bound_k(self):
        row = _rows(32000, 2, seed=1)[0]
        logits = mx.array(row)[None, :]
        tokens = mx.array([1, 2, 3])
        for k in (0.1, 0.45):
            proc = make_pless_renyi_logits_processor(k)
            assert np.array_equal(np.array(proc(tokens, logits)), np.array(mx_renyi(logits, k)))

    def test_factories_do_not_share_k(self):
        row = _rows(32000, 2, seed=2)[0]
        logits = mx.array(row)[None, :]
        a = make_pless_renyi_logits_processor(0.025)
        b = make_pless_renyi_logits_processor(1.6)
        out_a, out_b = np.array(a(None, logits)), np.array(b(None, logits))
        assert np.array_equal(out_a, np.array(mx_renyi(logits, 0.025)))
        assert np.array_equal(out_b, np.array(mx_renyi(logits, 1.6)))
        assert not np.array_equal(out_a, out_b)

    @pytest.mark.parametrize("k", [np.float32(0.1), np.float64(0.1), 1, 0.1])
    def test_accepts_any_real_number(self, k):
        row = _rows(32000, 2, seed=4)[0]
        logits = mx.array(row)[None, :]
        out = make_pless_renyi_logits_processor(k)(None, logits)
        assert np.array_equal(np.array(out), np.array(mx_renyi(logits, float(k))))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.1", None])
    def test_rejects_non_finite_or_non_numeric_k(self, bad):
        with pytest.raises((TypeError, ValueError)):
            make_pless_renyi_logits_processor(bad)


class TestBf16ParityWithTorch:
    """Production logits are bf16. Both implementations must upcast to float32
    before softmax; a missing upcast changes ~half of bf16 rows at vocab 131072
    (independent review, 2026-10-30)."""

    @pytest.mark.parametrize("k", [0.025, 0.075, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 2.0])
    def test_bf16_survivors_match_torch_bf16(self, k):
        for row in _rows(131072, 6, seed=int(k * 1e4) + 1):
            bf16_row = torch.from_numpy(row.copy()).to(torch.bfloat16).float().numpy()
            t_out = torch_renyi(torch.from_numpy(row.copy())[None, :].to(torch.bfloat16), k)
            t = torch.isfinite(t_out[0].float()).numpy()
            m_out = mx_renyi(mx.array(row)[None, :].astype(mx.bfloat16), k)
            m = np.isfinite(np.array(m_out[0].astype(mx.float32)))
            off = _disagreements_off_boundary(bf16_row, k, t, m)
            assert off == [], f"k={k}: bf16 non-boundary disagreement at {off[:5]}"


class TestProcessorReturnsFloat32:
    """mlx-lm normalises the processor's output with logsumexp in its dtype
    (generate.py `logits - logsumexp(logits)`); vLLM does that step in float32.
    The G_k processor returns float32 so mlx matches."""

    @pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
    def test_output_is_float32_with_same_survivors(self, dtype):
        row = _rows(131072, 2, seed=21)[1]
        logits = mx.array(row)[None, :].astype(getattr(mx, dtype))
        out = make_pless_renyi_logits_processor(0.1)(None, logits)
        assert out.dtype == mx.float32
        masked = mx_renyi(logits, 0.1)
        assert np.array_equal(np.isfinite(np.array(out)), np.isfinite(np.array(masked.astype(mx.float32))))
        kept = np.isfinite(np.array(out))[0]
        assert np.array_equal(np.array(out)[0][kept], np.array(logits.astype(mx.float32))[0][kept])
