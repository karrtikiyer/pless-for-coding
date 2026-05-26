"""Tests for the survival-mass vs entropy analysis.

The math we're testing is on synthetic top-K probability vectors —
no JSONL parsing, no plotting, no GPU. The acceptance criteria for
the final figure live in the module's docstring; here we only
verify per-record correctness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ─── per-record math primitives ─────────────────────────────────────────

def test_compute_survival_keeps_top_token_above_threshold():
    """A token with p >= threshold survives; below threshold gets zeroed."""
    from bench.eval.entropy_survival_curves import compute_survival
    p = np.array([0.5, 0.3, 0.1, 0.1])
    # Threshold = 0.2 → only p[0]=0.5 and p[1]=0.3 survive
    surv, n = compute_survival(p, threshold=0.2)
    assert n == 2
    assert abs(surv - 0.8) < 1e-9


def test_compute_survival_argmax_fallback_when_all_below_threshold():
    """If no token has p >= threshold, the production sampler falls back
    to argmax. Match that: survived_mass = max(p), n_surviving = 1."""
    from bench.eval.entropy_survival_curves import compute_survival
    # Uniform 0.1 across 10 tokens; threshold 0.5 — nothing survives
    p = np.array([0.1] * 10)
    surv, n = compute_survival(p, threshold=0.5)
    assert n == 1
    assert abs(surv - 0.1) < 1e-9  # argmax = 0.1 (any one of them)


def test_compute_survival_threshold_inclusive():
    """`>=` not `>` per the production sampler. p == threshold survives."""
    from bench.eval.entropy_survival_curves import compute_survival
    p = np.array([0.5, 0.25, 0.25])
    surv, n = compute_survival(p, threshold=0.25)
    # All three: 0.5 ≥ 0.25, 0.25 ≥ 0.25, 0.25 ≥ 0.25 → 3 tokens, total 1.0
    assert n == 3
    assert abs(surv - 1.0) < 1e-9


def test_compute_survival_at_alpha_2_threshold_keeps_argmax_only_on_one_hot():
    """For a one-hot distribution, σ_p² = 1 (only top token has p²=1).
    Threshold = 1 → only that one token survives. Matches the
    intuition: 'no diversity, argmax-only behavior'."""
    from bench.eval.entropy_survival_curves import compute_survival
    p = np.array([1.0, 0.0, 0.0, 0.0])
    sigma_p2 = float((p ** 2).sum())  # = 1.0
    surv, n = compute_survival(p, threshold=sigma_p2)
    assert n == 1
    assert abs(surv - 1.0) < 1e-9


def test_compute_survival_at_alpha_2_uniform_keeps_all():
    """Uniform p over K tokens: each p = 1/K, σ_p² = 1/K, threshold = 1/K.
    Every token has p == threshold → all survive (inclusive `>=`)."""
    from bench.eval.entropy_survival_curves import compute_survival
    K = 8
    p = np.full(K, 1.0 / K)
    sigma_p2 = float((p ** 2).sum())  # = 1/K
    surv, n = compute_survival(p, threshold=sigma_p2)
    assert n == K
    assert abs(surv - 1.0) < 1e-9


# ─── entropy math ───────────────────────────────────────────────────────

def test_compute_entropy_one_hot_is_zero():
    from bench.eval.entropy_survival_curves import compute_entropy
    p = np.array([1.0, 0.0, 0.0])
    H = compute_entropy(p)
    assert H == pytest.approx(0.0, abs=1e-9)


def test_compute_entropy_uniform_is_log_K():
    """Uniform over K → H = log(K) in nats."""
    from bench.eval.entropy_survival_curves import compute_entropy
    K = 32
    p = np.full(K, 1.0 / K)
    H = compute_entropy(p)
    assert H == pytest.approx(math.log(K), abs=1e-9)


def test_compute_entropy_does_not_explode_on_zeros():
    """ε-smoothing for log(0); H of [0.9, 0.1, 0, 0] is finite."""
    from bench.eval.entropy_survival_curves import compute_entropy
    p = np.array([0.9, 0.1, 0.0, 0.0])
    H = compute_entropy(p)
    assert math.isfinite(H)
    assert H >= 0


def test_compute_entropy_handles_unnormalized_top_k_via_renorm():
    """Stored top-32 doesn't sum to 1.0 (some mass past top-32).
    `compute_entropy` should accept normalized OR unnormalized and
    return the entropy of the normalized distribution (matches what
    we'd see if the full softmax were truncated to top-32)."""
    from bench.eval.entropy_survival_curves import compute_entropy
    p_unnorm = np.array([0.5, 0.3, 0.1])  # sums to 0.9
    p_norm = p_unnorm / p_unnorm.sum()
    H_unnorm = compute_entropy(p_unnorm)
    H_norm = compute_entropy(p_norm)
    assert H_unnorm == pytest.approx(H_norm, abs=1e-9)


# ─── full per-record pipeline ───────────────────────────────────────────

def test_process_record_returns_4_tuple():
    """`process_record({...})` returns (H, surv_α2, surv_α5, truncation_mass)."""
    from bench.eval.entropy_survival_curves import process_record
    rec = {
        "task_id": 1,
        "sample_id": 0,
        "position": 0,
        "token_id": 100,
        "token_str": "x",
        "sigma_p2": 0.5,
        "sigma_p3": 0.3,
        "sigma_p5": 0.15,
        "max_p": 0.7,
        "top32_probs": [0.7, 0.2, 0.05, 0.05] + [0.0] * 28,
        "top32_indices": list(range(32)),
    }
    out = process_record(rec)
    H, surv_a2, surv_a5, trunc_mass = out
    # Trunc mass should be 0 here (top-4 sum to 1.0)
    assert abs(trunc_mass) < 1e-9
    # σ_p² = 0.5 → only p ≥ 0.5 survives → just 0.7
    assert abs(surv_a2 - 0.7) < 1e-9
    # σ_p⁵ = 0.15 → p ≥ 0.15 survives → 0.7 + 0.2 = 0.9
    assert abs(surv_a5 - 0.9) < 1e-9
    # H finite, non-negative
    assert math.isfinite(H)
    assert H >= 0


def test_process_record_quantifies_top_32_truncation():
    """When top-32 sums to <1, truncation_mass is reported correctly."""
    from bench.eval.entropy_survival_curves import process_record
    # Probs sum to 0.95 — 0.05 truncation
    rec = {
        "task_id": 0, "sample_id": 0, "position": 0,
        "token_id": 0, "token_str": "",
        "sigma_p2": 0.3, "sigma_p3": 0.2, "sigma_p5": 0.1,
        "max_p": 0.5,
        "top32_probs": [0.5, 0.3, 0.15] + [0.0] * 29,
        "top32_indices": list(range(32)),
    }
    out = process_record(rec)
    _, _, _, trunc = out
    assert trunc == pytest.approx(0.05, abs=1e-9)


# ─── binning + aggregation ─────────────────────────────────────────────

def test_bin_records_groups_by_entropy_value():
    """Records get bucketed by H. We expose one bin per ~0.05 nat."""
    from bench.eval.entropy_survival_curves import bin_records
    # Synthetic: 100 records with H uniformly in [0, 1) nats
    records_4tuples = [(h, 0.5, 0.7, 0.0) for h in np.linspace(0, 0.99, 100)]
    bins = bin_records(records_4tuples, bin_width=0.05, h_max=1.0)
    # 0 to 1.0 with 0.05 width → 20 bins
    assert len(bins) == 20
    # Each bin should have ~5 records
    for b in bins:
        assert b["n_positions"] >= 1
    # Mean survival in each bin is 0.5 (α=2) and 0.7 (α=5) (since all records identical)
    for b in bins:
        if b["n_positions"] > 0:
            assert b["mean_survival_alpha2"] == pytest.approx(0.5, abs=1e-9)
            assert b["mean_survival_alpha5"] == pytest.approx(0.7, abs=1e-9)


def test_bin_records_handles_empty_bins():
    """Bins with zero records should be present but marked n_positions=0."""
    from bench.eval.entropy_survival_curves import bin_records
    # All records at H=0.5 — only one bin populated
    records_4tuples = [(0.5, 0.6, 0.8, 0.0) for _ in range(10)]
    bins = bin_records(records_4tuples, bin_width=0.05, h_max=1.0)
    n_populated = sum(1 for b in bins if b["n_positions"] > 0)
    assert n_populated == 1
    # Empty bins still exist (so the plot has a continuous x-axis)
    n_empty = sum(1 for b in bins if b["n_positions"] == 0)
    assert n_empty > 0
