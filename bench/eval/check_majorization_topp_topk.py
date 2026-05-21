"""Step 0.6 — Brute-force majorization-monotonicity check for top-p and top-k
truncation operators, paralleling the α-truncation checker.

If top-p and top-k satisfy the same majorization-monotonicity that α-truncation
does (which we verified in `check_majorization_trunc_alpha.py`), then the
Phase-1 majorization theorem for α-arms wouldn't be uniquely "α's" — it would
be a property shared by the class of "truncate-and-renormalize" operators
on the probability simplex.

If they DON'T satisfy it, α-arms get genuine theoretical uniqueness: the
first proved monotonicity for a parametric truncation family.

Conventions (parallel to the α-truncation checker):

  Trunc_p(p, p_val):   keep tokens whose sorted-descending cumulative prob
                       up to and including them is <= p_val (standard
                       nucleus sampling: always include the next token
                       that crosses the threshold).
  Trunc_k(p, k):       keep the top-k tokens by probability.

For both, larger parameter → more tokens kept → more spread-out result.
Expected direction (parallel to the corrected α-truncation direction):

  For p1 < p2 in (0, 1], we test:  q_{Trunc_p, p2} <prec q_{Trunc_p, p1}
  For k1 < k2 in {1..V},  we test:  q_{Trunc_k, k2} <prec q_{Trunc_k, k1}

i.e. larger hyperparameter → more spread out → majorized BY the smaller one.

Run:
    uv run python -m bench.eval.check_majorization_topp_topk
"""

from __future__ import annotations

import numpy as np

from bench.eval.check_majorization_trunc_alpha import (
    is_majorized_by,
    is_valid_distribution,
    min_slack,
    violation_index,
)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def trunc_top_p(p: np.ndarray, p_val: float) -> np.ndarray:
    """Standard top-p (nucleus) truncation: keep tokens whose cumulative
    sorted-descending prob is <= p_val, plus always include the first token
    that crosses the threshold (matches HF / vLLM convention).

    Returns the renormalized distribution lifted back to length V, with
    zeros on discarded indices.
    """
    p = np.asarray(p, dtype=np.float64)
    sorted_idx = np.argsort(-p)
    sorted_p = p[sorted_idx]
    cumsum = np.cumsum(sorted_p)
    # Mask in sorted order: keep up to and including the first index where cumsum >= p_val.
    # Convention: always keep at least the top-1, even if it alone exceeds p_val.
    keep_mask_sorted = np.zeros(len(p), dtype=bool)
    keep_mask_sorted[0] = True
    for i in range(1, len(p)):
        if cumsum[i - 1] < p_val:
            keep_mask_sorted[i] = True
        else:
            break
    # Map back to original index order
    keep_mask = np.zeros(len(p), dtype=bool)
    keep_mask[sorted_idx] = keep_mask_sorted
    q = np.zeros_like(p)
    kept_sum = float(p[keep_mask].sum())
    if kept_sum <= 0.0:
        return q
    q[keep_mask] = p[keep_mask] / kept_sum
    return q


def trunc_top_k(p: np.ndarray, k: int) -> np.ndarray:
    """Top-k truncation: keep the top-k by probability, renormalize."""
    p = np.asarray(p, dtype=np.float64)
    if k >= len(p):
        # All kept; return p unchanged (it's already a probability distribution)
        return p.copy()
    if k <= 0:
        return np.zeros_like(p)
    sorted_idx = np.argsort(-p)
    keep_mask = np.zeros(len(p), dtype=bool)
    keep_mask[sorted_idx[:k]] = True
    q = np.zeros_like(p)
    kept_sum = float(p[keep_mask].sum())
    if kept_sum <= 0.0:
        return q
    q[keep_mask] = p[keep_mask] / kept_sum
    return q


# ---------------------------------------------------------------------------
# Sanity check on hand-worked example p = [0.5, 0.3, 0.2]
# ---------------------------------------------------------------------------


def sanity_check() -> None:
    p = np.array([0.5, 0.3, 0.2])

    # Top-p sanity
    q_p03 = trunc_top_p(p, 0.3)  # cumsum 0 < 0.3 then 0.5 >= 0.3 → only top-1 kept
    q_p05 = trunc_top_p(p, 0.5)  # cumsum 0 < 0.5 then 0.5 >= 0.5 → top-1 only
    q_p06 = trunc_top_p(p, 0.6)  # cumsum 0 < 0.6, keep; cumsum 0.5 < 0.6, keep; cumsum 0.8 >= 0.6, stop → {0.5, 0.3}
    q_p09 = trunc_top_p(p, 0.9)  # cumsum 0 < 0.9, keep; 0.5 < 0.9, keep; 0.8 < 0.9, keep → all three
    q_p10 = trunc_top_p(p, 1.0)  # cumsum 0 < 1, keep; 0.5 < 1, keep; 0.8 < 1, keep → all three

    print(f"  p           = {p}")
    print(f"  TopP(0.3)   = {q_p03}  (expect [1, 0, 0])")
    print(f"  TopP(0.5)   = {q_p05}  (expect [1, 0, 0])")
    print(f"  TopP(0.6)   = {q_p06}  (expect [0.625, 0.375, 0])")
    print(f"  TopP(0.9)   = {q_p09}  (expect [0.5, 0.3, 0.2])")
    print(f"  TopP(1.0)   = {q_p10}  (expect [0.5, 0.3, 0.2])")
    np.testing.assert_allclose(q_p03, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q_p05, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q_p06, [0.625, 0.375, 0.0], atol=1e-12)
    np.testing.assert_allclose(q_p09, [0.5, 0.3, 0.2], atol=1e-12)
    np.testing.assert_allclose(q_p10, [0.5, 0.3, 0.2], atol=1e-12)

    # Expected direction: larger p_val produces more spread → majorized BY smaller
    assert is_majorized_by(q_p06, q_p05), "TopP(0.6) should be majorized by TopP(0.5)"
    assert is_majorized_by(q_p09, q_p06), "TopP(0.9) should be majorized by TopP(0.6)"

    # Top-k sanity
    q_k1 = trunc_top_k(p, 1)
    q_k2 = trunc_top_k(p, 2)
    q_k3 = trunc_top_k(p, 3)
    print(f"  TopK(k=1)   = {q_k1}  (expect [1, 0, 0])")
    print(f"  TopK(k=2)   = {q_k2}  (expect [0.625, 0.375, 0])")
    print(f"  TopK(k=3)   = {q_k3}  (expect [0.5, 0.3, 0.2])")
    np.testing.assert_allclose(q_k1, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q_k2, [0.625, 0.375, 0.0], atol=1e-12)
    np.testing.assert_allclose(q_k3, [0.5, 0.3, 0.2], atol=1e-12)

    assert is_majorized_by(q_k2, q_k1), "TopK(k=2) should be majorized by TopK(k=1)"
    assert is_majorized_by(q_k3, q_k2), "TopK(k=3) should be majorized by TopK(k=2)"

    print("  sanity-check PASSED for top-p and top-k.\n")


# ---------------------------------------------------------------------------
# Random sweeps
# ---------------------------------------------------------------------------


# Top-p values to test. Chose a dense grid including extremes.
TOP_P_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
# Top-k values to test. Bounded by V at runtime.
TOP_K_VALUES = [1, 2, 3, 5, 10, 25, 50, 100]

V_LIST = [3, 4, 5, 6, 8, 10, 15]
N_TRIALS_PER_V = 10000
SEED = 42


def run_random_sweep_top_p() -> dict:
    rng = np.random.default_rng(SEED)
    total = 0
    skipped_degenerate = 0
    counterex = []
    slacks = []

    for V in V_LIST:
        for _ in range(N_TRIALS_PER_V):
            p = rng.dirichlet(np.ones(V))
            for i, p1 in enumerate(TOP_P_VALUES):
                for p2 in TOP_P_VALUES[i + 1:]:
                    q1 = trunc_top_p(p, p1)
                    q2 = trunc_top_p(p, p2)
                    if not is_valid_distribution(q1) or not is_valid_distribution(q2):
                        skipped_degenerate += 1
                        continue
                    total += 1
                    # Test: q2 (larger p_val) should be majorized by q1 (smaller p_val)
                    if not is_majorized_by(q2, q1, atol=1e-9):
                        counterex.append({
                            "V": V, "p": p.tolist(), "p1": p1, "p2": p2,
                            "q1": q1.tolist(), "q2": q2.tolist(),
                            "violation_idx": violation_index(q2, q1),
                            "slack": min_slack(q2, q1),
                        })
                    else:
                        slacks.append(min_slack(q2, q1))

    return {
        "total": total,
        "skipped_degenerate": skipped_degenerate,
        "counterex": counterex,
        "slacks": slacks,
    }


def run_random_sweep_top_k() -> dict:
    rng = np.random.default_rng(SEED + 1)
    total = 0
    skipped_degenerate = 0
    counterex = []
    slacks = []

    for V in V_LIST:
        valid_ks = [k for k in TOP_K_VALUES if k <= V]
        valid_ks = sorted(set(valid_ks + [V]))   # include "keep all" as terminal
        for _ in range(N_TRIALS_PER_V):
            p = rng.dirichlet(np.ones(V))
            for i, k1 in enumerate(valid_ks):
                for k2 in valid_ks[i + 1:]:
                    q1 = trunc_top_k(p, k1)
                    q2 = trunc_top_k(p, k2)
                    if not is_valid_distribution(q1) or not is_valid_distribution(q2):
                        skipped_degenerate += 1
                        continue
                    total += 1
                    if not is_majorized_by(q2, q1, atol=1e-9):
                        counterex.append({
                            "V": V, "p": p.tolist(), "k1": k1, "k2": k2,
                            "q1": q1.tolist(), "q2": q2.tolist(),
                            "violation_idx": violation_index(q2, q1),
                            "slack": min_slack(q2, q1),
                        })
                    else:
                        slacks.append(min_slack(q2, q1))

    return {
        "total": total,
        "skipped_degenerate": skipped_degenerate,
        "counterex": counterex,
        "slacks": slacks,
    }


def _slack_quantiles(slacks: list[float]) -> dict:
    if not slacks:
        return {}
    arr = np.asarray(slacks)
    return {
        "n": len(arr),
        "min": float(arr.min()),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def main() -> None:
    print("=" * 80)
    print("Step 0.6 — top-p / top-k majorization-monotonicity check")
    print("=" * 80)
    print()
    print("Sanity check on hand-worked p = [0.5, 0.3, 0.2]:")
    sanity_check()

    print(f"Random sweep: V ∈ {V_LIST}, {N_TRIALS_PER_V} trials per V")
    print()

    print("--- Top-p sweep ---")
    print(f"top_p values tested: {TOP_P_VALUES}")
    res_p = run_random_sweep_top_p()
    print(f"  Total valid trials:        {res_p['total']:>10}")
    print(f"  Skipped (degenerate):      {res_p['skipped_degenerate']:>10}")
    print(f"  Counterexamples:           {len(res_p['counterex']):>10}")
    sl = _slack_quantiles(res_p['slacks'])
    if sl:
        print(f"  Min-slack quantiles (passing trials): "
              f"min={sl['min']:.3e}  p50={sl['p50']:.3e}  p90={sl['p90']:.3e}  "
              f"p99={sl['p99']:.3e}  max={sl['max']:.3e}")
    if res_p['counterex']:
        print(f"\n  First {min(5, len(res_p['counterex']))} counterexamples:")
        for ce in res_p['counterex'][:5]:
            print(f"    V={ce['V']}  p1={ce['p1']}  p2={ce['p2']}  "
                  f"slack={ce['slack']:.3e}  violating_k={ce['violation_idx']}")
            print(f"      p  = {ce['p']}")
            print(f"      q1 = {ce['q1']}")
            print(f"      q2 = {ce['q2']}")
    print()

    print("--- Top-k sweep ---")
    print(f"top_k values tested: {TOP_K_VALUES} (bounded by V)")
    res_k = run_random_sweep_top_k()
    print(f"  Total valid trials:        {res_k['total']:>10}")
    print(f"  Skipped (degenerate):      {res_k['skipped_degenerate']:>10}")
    print(f"  Counterexamples:           {len(res_k['counterex']):>10}")
    sl = _slack_quantiles(res_k['slacks'])
    if sl:
        print(f"  Min-slack quantiles (passing trials): "
              f"min={sl['min']:.3e}  p50={sl['p50']:.3e}  p90={sl['p90']:.3e}  "
              f"p99={sl['p99']:.3e}  max={sl['max']:.3e}")
    if res_k['counterex']:
        print(f"\n  First {min(5, len(res_k['counterex']))} counterexamples:")
        for ce in res_k['counterex'][:5]:
            print(f"    V={ce['V']}  k1={ce['k1']}  k2={ce['k2']}  "
                  f"slack={ce['slack']:.3e}  violating_k={ce['violation_idx']}")
            print(f"      p  = {ce['p']}")
            print(f"      q1 = {ce['q1']}")
            print(f"      q2 = {ce['q2']}")
    print()

    print("=" * 80)
    print("Verdict")
    print("=" * 80)

    if not res_p['counterex'] and not res_k['counterex']:
        print(
            "Both top-p AND top-k satisfy majorization-monotonicity in the same "
            "direction as α-truncation. The Phase-1 theorem (if proved for α) "
            "is NOT uniquely α's — it's a property of the truncate-and-renormalize "
            "family. Paper positioning: 'first proved instance of a class' rather "
            "than 'uniquely principled'."
        )
    elif res_p['counterex'] and res_k['counterex']:
        print(
            f"Both top-p AND top-k have counterexamples ({len(res_p['counterex'])} "
            f"and {len(res_k['counterex'])} respectively). α-truncation is the only "
            "monotonic member of the three. Strong paper positioning: α has a "
            "guarantee that top-p and top-k provably do not."
        )
    elif res_p['counterex']:
        print(
            f"Top-p has {len(res_p['counterex'])} counterexamples; top-k has none. "
            "α and top-k satisfy the monotonicity; top-p does not. The paper claim "
            "can be sharpened: α is the *parametrically smooth* member of the "
            "monotonic family; top-p is excluded."
        )
    else:
        print(
            f"Top-k has {len(res_k['counterex'])} counterexamples; top-p has none. "
            "α and top-p satisfy the monotonicity; top-k does not."
        )


if __name__ == "__main__":
    main()
