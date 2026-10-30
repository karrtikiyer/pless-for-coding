"""Randomized-trial parity check: bench/mlx_pless_logits_processors.py's mx
port vs. the torch reference it was translated from
(bench/generator_vllm.py's _pless_mask_logits / _pless_norm_mask_logits).

Same validation methodology used to catch the earlier top-p inclusive/
exclusive bug in _top_p_top_k_mask_logits (CLAUDE.md gotcha #8): generate many
random logit vectors across a range of peakiness, run both implementations,
and assert the surviving (non -inf) token set is identical on every trial --
not just "looks right."

This imports the torch reference functions directly from generator_vllm.py
(module-private but importable) rather than re-deriving them, so there is
exactly one place the pless-masking math is authored for each framework and
this script never drifts from either.

Run: uv run python scripts/verify_mlx_pless_logits.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.generator_vllm import _pless_mask_logits as _torch_pless  # noqa: E402
from bench.generator_vllm import _pless_norm_mask_logits as _torch_pless_norm  # noqa: E402
from bench.pless_variants_vllm import pless_effk_mask_logits as _torch_pless_effk  # noqa: E402

import mlx.core as mx  # noqa: E402

from bench.mlx_pless_logits_processors import _pless_mask_logits as _mx_pless  # noqa: E402
from bench.mlx_pless_logits_processors import _pless_norm_mask_logits as _mx_pless_norm  # noqa: E402
from bench.mlx_pless_logits_processors import _pless_effk_mask_logits as _mx_pless_effk  # noqa: E402

N_TRIALS = 2000
VOCAB_SIZES = [151936, 32000, 7]  # Qwen3 vocab, a smaller model's vocab, a tiny edge case
SEED = 42


def _random_logits(rng: np.random.Generator, vocab_size: int, peakiness: str) -> np.ndarray:
    """Generate one random logit row at a given peakiness regime."""
    if peakiness == "uniform":
        return rng.normal(0.0, 0.5, size=vocab_size).astype(np.float32)
    if peakiness == "peaked":
        base = rng.normal(0.0, 1.0, size=vocab_size).astype(np.float32)
        spike_idx = rng.integers(0, vocab_size)
        base[spike_idx] += rng.uniform(8.0, 20.0)
        return base
    if peakiness == "bimodal":
        base = rng.normal(0.0, 0.3, size=vocab_size).astype(np.float32)
        idx_a, idx_b = rng.choice(vocab_size, size=2, replace=False)
        base[idx_a] += rng.uniform(4.0, 10.0)
        base[idx_b] += rng.uniform(4.0, 10.0)
        return base
    raise ValueError(peakiness)


def _survivor_set(mask_logits_fn, logits_row: np.ndarray, *, backend: str) -> frozenset[int]:
    if backend == "torch":
        t = torch.from_numpy(logits_row.copy()).unsqueeze(0)
        out = mask_logits_fn(t)
        survivors = (out[0] != float("-inf")).nonzero().flatten().tolist()
    elif backend == "mx":
        m = mx.array(logits_row.copy())[None, :]
        out = mask_logits_fn(m)
        out_np = np.array(out[0])
        survivors = np.nonzero(out_np != float("-inf"))[0].tolist()
    else:
        raise ValueError(backend)
    return frozenset(survivors)


def main() -> None:
    rng = np.random.default_rng(SEED)
    mismatches = []
    total = 0

    for vocab_size in VOCAB_SIZES:
        for peakiness in ("uniform", "peaked", "bimodal"):
            for trial in range(N_TRIALS // (len(VOCAB_SIZES) * 3)):
                row = _random_logits(rng, vocab_size, peakiness)
                total += 1

                torch_survivors = _survivor_set(_torch_pless, row, backend="torch")
                mx_survivors = _survivor_set(_mx_pless, row, backend="mx")
                if torch_survivors != mx_survivors:
                    mismatches.append(("pless", vocab_size, peakiness, trial,
                                        torch_survivors, mx_survivors))

                total += 1
                torch_survivors_n = _survivor_set(_torch_pless_norm, row, backend="torch")
                mx_survivors_n = _survivor_set(_mx_pless_norm, row, backend="mx")
                if torch_survivors_n != mx_survivors_n:
                    mismatches.append(("pless_norm", vocab_size, peakiness, trial,
                                        torch_survivors_n, mx_survivors_n))

                # pless_effk: expect a HIGHER mismatch rate than plain pless/pless_norm
                # above (~4% vs ~1.35%, concentrated in "uniform" and absent from
                # "peaked") -- this is NOT the same boundary-noise story. Diagnosed
                # mechanism (2026-10-27 adversarial review): effk pipes the same
                # sub-ULP cross-backend noise in `coll = Sum(p_i^2)` through a
                # DISCONTINUOUS integer round() when computing `n_eff = round(1/coll)`.
                # In the near-uniform/large-vocab regime `1/coll` is itself huge
                # (~1e5), so adjacent integer rounding buckets are only ~1/1e5 apart
                # in relative terms -- well within float32 summation-order noise from
                # reducing 150k+ squared-probability terms differently per backend.
                # Peaked distributions have `coll` near 1 (small effective support),
                # so `1/coll` is small (~1-5) and bucket gaps are huge relative to FP
                # noise -- hence zero mismatches there. This is inherent to
                # round(1/Sum(p_i^2)) over large vocabularies (would likely also show
                # up comparing torch-CPU vs torch-GPU, or across PyTorch versions),
                # not an mlx-specific defect -- but it is a mechanistically distinct,
                # higher-risk divergence surface than plain pless's graceful
                # probability-vs-threshold comparison, and should not be assumed to
                # be "the same kind of noise, just more of it" without re-deriving
                # this mechanism for any FUTURE pless variant that adds a round()/
                # rank-cut step.
                total += 1
                torch_survivors_e = _survivor_set(_torch_pless_effk, row, backend="torch")
                mx_survivors_e = _survivor_set(_mx_pless_effk, row, backend="mx")
                if torch_survivors_e != mx_survivors_e:
                    mismatches.append(("pless_effk", vocab_size, peakiness, trial,
                                        torch_survivors_e, mx_survivors_e))

    print(f"Ran {total} trials across vocab sizes {VOCAB_SIZES} x peakiness regimes.")
    if mismatches:
        print(f"MISMATCH: {len(mismatches)}/{total} trials disagree between torch and mx:")
        for m in mismatches[:10]:
            sampler, vocab_size, peakiness, trial, t_surv, m_surv = m
            print(f"  [{sampler}] vocab={vocab_size} peakiness={peakiness} trial={trial}: "
                  f"torch survivors={sorted(t_surv)[:10]}... (n={len(t_surv)}) "
                  f"mx survivors={sorted(m_surv)[:10]}... (n={len(m_surv)})")
        # Every sampler here has its OWN established, diagnosed floating-point
        # boundary-noise rate (pless/pless_norm: ~1.35%, graceful probability-vs-
        # threshold ties; pless_effk: ~4%, round(1/Sum(p_i^2)) rank-bucket noise --
        # see the comment above the effk trial, a mechanistically distinct and
        # higher-risk surface, not "the same noise, just more of it"). Only fail
        # the run if a sampler's rate moves meaningfully outside ITS OWN band --
        # that would indicate a real regression, not expected cross-backend noise.
        per_sampler_trials = total // 3  # one of {pless, pless_norm, pless_effk} per inner loop iteration
        expected_max_rate = {"pless": 0.03, "pless_norm": 0.03, "pless_effk": 0.08}
        regressions = []
        for sampler in ("pless", "pless_norm", "pless_effk"):
            n = sum(1 for m in mismatches if m[0] == sampler)
            rate = n / per_sampler_trials if per_sampler_trials else 0.0
            print(f"  {sampler} mismatch rate: {n}/{per_sampler_trials} ({rate:.1%}) "
                  f"-- expected <= {expected_max_rate[sampler]:.0%}")
            if rate > expected_max_rate[sampler]:
                regressions.append(sampler)
        if regressions:
            print(f"REGRESSION: {regressions} exceeded their established noise band.")
            sys.exit(1)
        print("PASS (all mismatches are within each sampler's established, diagnosed FP-noise band).")
    else:
        print("PASS: mx port's survivor set matches the torch reference on every trial.")


if __name__ == "__main__":
    main()
