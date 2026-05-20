"""Unit tests for the beta-binomial pass@k closed form."""

from __future__ import annotations

import math

import numpy as np

from bench.eval.validate_pass_at_k_beta_binomial import (
    BetaFit,
    fit_beta_binomial_mom,
    measured_pass_at_k_chen,
    pass_at_k_beta,
)


def test_pass_at_1_equals_mean():
    """pass@1 under Beta(a, b) should equal mean = a/(a+b)."""
    fit = BetaFit(a=2.0, b=3.0, mean=0.4, nu=5.0, method="mom")
    assert math.isclose(pass_at_k_beta(fit, 1), 0.4, rel_tol=1e-9)


def test_uniform_beta_pass_at_k():
    """Under Beta(1, 1) = Uniform[0, 1], pass@k = 1 - 1/(k+1) = k/(k+1)."""
    fit = BetaFit(a=1.0, b=1.0, mean=0.5, nu=2.0, method="mom")
    for k in [1, 2, 3, 5, 10]:
        expected = k / (k + 1)
        got = pass_at_k_beta(fit, k)
        assert math.isclose(got, expected, rel_tol=1e-9), f"k={k}: got {got}, want {expected}"


def test_point_mass_recovers_iid():
    """When all tasks share the same p, BetaBinom collapses to Binomial.
    pass@k should equal 1 - (1-p)^k."""
    # Simulate n=10 tasks where each scored c=7 (so per-task p̂=0.7).
    counts = np.full(500, 7, dtype=int)
    fit = fit_beta_binomial_mom(counts, n=10)
    # Variance is zero (all identical), so MOM falls back to point mass.
    assert fit.method == "degenerate-point-mass"
    for k in [1, 3, 5, 10]:
        expected = 1.0 - (1 - 0.7) ** k
        got = pass_at_k_beta(fit, k)
        assert math.isclose(got, expected, rel_tol=1e-9)


def test_mom_recovers_first_moment_exactly():
    """MOM-fitted Beta should reproduce mean(c)/n exactly for pass@1."""
    rng = np.random.default_rng(42)
    # Sample from BetaBinom(n=10, a=2, b=3).
    p_per_task = rng.beta(2.0, 3.0, size=2000)
    counts = rng.binomial(10, p_per_task)
    fit = fit_beta_binomial_mom(counts, n=10)
    p1_pred = pass_at_k_beta(fit, 1)
    p1_emp = counts.mean() / 10
    assert math.isclose(p1_pred, p1_emp, rel_tol=1e-9)
    # And the fitted (a, b) should be near (2, 3).
    assert abs(fit.a - 2.0) < 0.3
    assert abs(fit.b - 3.0) < 0.3


def test_betabinom_predicts_passk_on_synthetic():
    """When data IS beta-binomial, predicted pass@k should match measured
    within sampling noise."""
    rng = np.random.default_rng(7)
    n_samples = 10
    n_tasks = 2000
    p_per_task = rng.beta(2.0, 3.0, size=n_tasks)
    counts = rng.binomial(n_samples, p_per_task)
    fit = fit_beta_binomial_mom(counts, n=n_samples)
    for k in [1, 3, 5, 10]:
        pred = pass_at_k_beta(fit, k)
        meas = measured_pass_at_k_chen(counts, n=n_samples, k=k)
        # Synthetic data with n_tasks=2000 should be within ~1.5 pp
        assert abs(pred - meas) < 0.015, (
            f"k={k}: pred={pred:.4f}, meas={meas:.4f}, diff={(pred-meas):.4f}"
        )


def test_chen_estimator_endpoints():
    """Chen estimator: c=0 -> 0, c=n -> 1, all-correct cases."""
    n = 10
    # All tasks have c=0 -> pass@k = 0 for any k
    counts = np.zeros(100, dtype=int)
    assert measured_pass_at_k_chen(counts, n=n, k=5) == 0.0
    # All tasks have c=n -> pass@k = 1
    counts = np.full(100, n, dtype=int)
    assert measured_pass_at_k_chen(counts, n=n, k=5) == 1.0
