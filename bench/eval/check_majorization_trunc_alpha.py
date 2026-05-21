"""Brute-force checker for the truncation-alpha majorization conjecture.

Conjecture: For any probability vector p over V tokens and 1 <= alpha < alpha',
    Trunc_alpha(p) <prec Trunc_alpha'(p)
where Trunc_alpha keeps tokens with p_i >= sum_j p_j^alpha and renormalizes.

If true, every Schur-concave functional (e.g. Shannon entropy) is monotone
non-decreasing in alpha on the truncated renormalized distribution.

Run:
    uv run python -m bench.eval.check_majorization_trunc_alpha
"""

from __future__ import annotations

import numpy as np


def trunc_alpha(p: np.ndarray, alpha: float) -> np.ndarray:
    """Return q_alpha (same length as p), zero on discarded indices.

    Note: when alpha is close to 1 and p is near-uniform, the kept set CAN
    be empty (e.g. uniform V=3 at alpha=1.0001: T = 3*(1/3)^1.0001 ~ 0.9997
    > 1/3, so no p_i >= T). In that case we return the all-zero vector;
    callers should treat this as "Trunc_alpha is undefined for this (p, alpha)"
    and exclude it from the majorization comparison (zero vector is not a
    probability distribution).
    """
    p = np.asarray(p, dtype=np.float64)
    threshold = float(np.sum(p ** alpha))
    kept_mask = p >= threshold
    q = np.zeros_like(p)
    kept_sum = float(p[kept_mask].sum())
    if kept_sum <= 0.0:
        return q
    q[kept_mask] = p[kept_mask] / kept_sum
    return q


def is_valid_distribution(q: np.ndarray, atol: float = 1e-9) -> bool:
    """Returns True if q sums to 1 (within atol). Excludes the degenerate
    empty-kept-set case where trunc_alpha returned all zeros."""
    return abs(float(np.sum(q)) - 1.0) <= atol


def is_majorized_by(x: np.ndarray, y: np.ndarray, atol: float = 1e-9) -> bool:
    """Return True iff x <prec y (x is majorized by y).

    Definition: sort both descending. Then for every k,
        sum_{i=1..k} x^down_i  <=  sum_{i=1..k} y^down_i
    with sums equal at k = len. Allow tolerance atol.
    """
    x_sorted = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    y_sorted = np.sort(np.asarray(y, dtype=np.float64))[::-1]
    assert x_sorted.shape == y_sorted.shape
    cx = np.cumsum(x_sorted)
    cy = np.cumsum(y_sorted)
    # Totals must match (within tolerance)
    if abs(cx[-1] - cy[-1]) > atol:
        return False
    # Every prefix sum of x must be <= prefix sum of y
    return bool(np.all(cx <= cy + atol))


def violation_index(x: np.ndarray, y: np.ndarray, atol: float = 1e-9) -> int:
    """Return first k (0-indexed) where prefix sum of x exceeds y, else -1."""
    x_sorted = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    y_sorted = np.sort(np.asarray(y, dtype=np.float64))[::-1]
    cx = np.cumsum(x_sorted)
    cy = np.cumsum(y_sorted)
    diffs = cx - cy
    bad = np.where(diffs > atol)[0]
    return int(bad[0]) if bad.size > 0 else -1


def min_slack(x: np.ndarray, y: np.ndarray) -> float:
    """Return min_k [S_k(y) - S_k(x)] over sorted-descending prefix sums.

    Positive => majorization holds with that much room. Negative => violated.
    """
    x_sorted = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    y_sorted = np.sort(np.asarray(y, dtype=np.float64))[::-1]
    cx = np.cumsum(x_sorted)
    cy = np.cumsum(y_sorted)
    return float(np.min(cy - cx))


# ---------------------------------------------------------------------------
# Sanity check on hand-worked example
# ---------------------------------------------------------------------------

def sanity_check() -> None:
    # NOTE on the hand-worked example in the spec:
    #
    # The spec said that for p=[0.5,0.3,0.2], alpha=3 gives q_3=[0.625,0.375,0].
    # That is arithmetically wrong: T_3 = 0.5^3+0.3^3+0.2^3 = 0.160, and the
    # kept set {i: p_i >= 0.160} is all three indices. So q_3 = p exactly.
    # Using alpha = 2.5 gives T~0.244, kept={0.5,0.3}, q_2.5=[0.625,0.375,0]
    # -- that's the value the spec attributed (incorrectly) to alpha=3.
    #
    # MORE IMPORTANTLY: the spec's "expected" majorization direction
    # (q_2 < q_3 < q_5) is BACKWARDS. Sorted-desc partial sums:
    #     q_2   sorted = [1, 0, 0]      => prefix [1.0, 1.0, 1.0]
    #     q_2.5 sorted = [0.625, 0.375, 0] => prefix [0.625, 1.0, 1.0]
    #     q_3=q_5 sorted = [0.5, 0.3, 0.2] => prefix [0.5, 0.8, 1.0]
    # For q_2 <prec q_2.5 we need prefix(q_2) <= prefix(q_2.5) componentwise.
    # At k=1: 1.0 <= 0.625 is FALSE. So q_2 is NOT majorized by q_2.5.
    # Direction is reversed: q_5 <prec q_3 <prec q_2.5 <prec q_2. The MORE
    # PEAKED distribution majorizes; the more spread-out one is majorized.
    #
    # That means the conjecture "q_alpha <prec q_alpha' for alpha < alpha'"
    # as literally written in the spec is dead on this example: the larger
    # alpha keeps more tokens and is therefore more spread out, so it is
    # MAJORIZED BY the smaller-alpha distribution, not the other way around.
    #
    # We verify both directions below and let the sweep decide.

    p = np.array([0.5, 0.3, 0.2])
    q2 = trunc_alpha(p, 2.0)
    q25 = trunc_alpha(p, 2.5)
    q3 = trunc_alpha(p, 3.0)
    q5 = trunc_alpha(p, 5.0)

    print(f"  p      = {p}")
    print(f"  q_2    = {q2}    (T=0.38; expect [1.0, 0, 0])")
    print(f"  q_2.5  = {q25}    (T~0.244; expect [0.625, 0.375, 0])")
    print(f"  q_3    = {q3}    (T=0.16; expect [0.5, 0.3, 0.2])")
    print(f"  q_5    = {q5}    (T~0.034; expect [0.5, 0.3, 0.2])")

    # Verify the Trunc_alpha output values
    np.testing.assert_allclose(q2, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q25, [0.625, 0.375, 0.0], atol=1e-12)
    np.testing.assert_allclose(q3, [0.5, 0.3, 0.2], atol=1e-12)
    np.testing.assert_allclose(q5, [0.5, 0.3, 0.2], atol=1e-12)

    # The CORRECT direction: larger-alpha distribution is majorized by
    # smaller-alpha distribution (the more peaked majorizes the more spread).
    assert is_majorized_by(q25, q2), "q_2.5 should be majorized by q_2"
    assert is_majorized_by(q3, q25), "q_3 should be majorized by q_2.5"
    assert is_majorized_by(q5, q2), "q_5 should be majorized by q_2"

    # And the LITERAL spec direction (q_alpha <prec q_alpha' for alpha<alpha')
    # is FALSE here (where the two differ):
    assert not is_majorized_by(q2, q25), "spec-direction must fail: q_2 NOT <prec q_2.5"
    assert not is_majorized_by(q25, q3), "spec-direction must fail: q_2.5 NOT <prec q_3"

    # Cross-check the prompt's warning case: [0.6,0.4] NOT majorized by [0.5,0.5]
    assert not is_majorized_by(np.array([0.6, 0.4]), np.array([0.5, 0.5])), \
        "[0.6,0.4] should NOT be majorized by [0.5,0.5]"
    assert is_majorized_by(np.array([0.5, 0.5]), np.array([0.6, 0.4])), \
        "[0.5,0.5] should be majorized by [0.6,0.4]"

    print("  sanity-check PASSED. NOTE: the spec's literal direction "
          "(q_alpha <prec q_alpha') is REVERSED on the hand example;")
    print("  the correct direction is q_alpha' <prec q_alpha (larger alpha "
          "= more spread = majorized). The sweep below checks BOTH.\n")

    # Cross-check the warning case from the prompt: [0.6, 0.4] is NOT
    # majorized by [0.5, 0.5]. (Peaked vector has LARGER prefix sums, so it
    # majorizes; therefore peaked is not majorized BY the flatter one.)
    assert not is_majorized_by(np.array([0.6, 0.4]), np.array([0.5, 0.5])), \
        "[0.6,0.4] should NOT be majorized by [0.5,0.5]"
    assert is_majorized_by(np.array([0.5, 0.5]), np.array([0.6, 0.4])), \
        "[0.5,0.5] should be majorized by [0.6,0.4]"

    print("  sanity-check PASSED.\n")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

EDGE_ALPHAS = [1.0001, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]


def make_edge_distributions() -> list[tuple[str, np.ndarray]]:
    cases: list[tuple[str, np.ndarray]] = []
    for V in [3, 4, 5, 8, 15]:
        cases.append((f"uniform V={V}", np.full(V, 1.0 / V)))
        eps = 1e-3
        peaked = np.full(V, eps / (V - 1))
        peaked[0] = 1.0 - eps
        cases.append((f"peaked V={V} eps={eps}", peaked))
        eps2 = 1e-8
        peaked2 = np.full(V, eps2 / (V - 1))
        peaked2[0] = 1.0 - eps2
        cases.append((f"peaked V={V} eps={eps2}", peaked2))
    # Two-mode
    for V in [4, 5, 8, 15]:
        small = (1.0 - 0.8) / (V - 2)
        tm = np.full(V, small)
        tm[0] = 0.4
        tm[1] = 0.4
        cases.append((f"two-mode V={V}", tm))
    return cases


def check_edge_cases() -> tuple[int, int, list[dict], list[dict]]:
    """Return (trials_total, trials_skipped_degenerate, ce_spec, ce_rev)."""
    cases = make_edge_distributions()
    ce_spec: list[dict] = []
    ce_rev: list[dict] = []
    trials = 0
    skipped = 0
    for name, p in cases:
        for i, a in enumerate(EDGE_ALPHAS):
            for ap in EDGE_ALPHAS[i + 1:]:
                qa = trunc_alpha(p, a)
                qap = trunc_alpha(p, ap)
                if not (is_valid_distribution(qa) and is_valid_distribution(qap)):
                    skipped += 1
                    continue
                trials += 1
                # Spec direction: q_alpha <prec q_alpha'
                if not is_majorized_by(qa, qap):
                    k = violation_index(qa, qap)
                    ce_spec.append({
                        "where": "edge", "name": name, "p": p.copy(),
                        "alpha": a, "alpha_prime": ap, "k": k,
                        "q_a": qa.copy(), "q_ap": qap.copy(),
                    })
                # Reverse direction: q_alpha' <prec q_alpha
                if not is_majorized_by(qap, qa):
                    k = violation_index(qap, qa)
                    ce_rev.append({
                        "where": "edge", "name": name, "p": p.copy(),
                        "alpha": a, "alpha_prime": ap, "k": k,
                        "q_a": qa.copy(), "q_ap": qap.copy(),
                    })
    return trials, skipped, ce_spec, ce_rev


# ---------------------------------------------------------------------------
# Random sweep
# ---------------------------------------------------------------------------

V_LIST = [3, 4, 5, 6, 7, 8, 10, 15]
ALPHA_GRID = [1.1, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 50.0]
N_TRIALS_PER_V = 10_000


def random_sweep(
    seed: int = 20260521,
) -> tuple[int, int, list[dict], list[dict], list[float], list[float]]:
    """Run the random sweep checking BOTH directions.

    Skips trials where either trunc_alpha output is degenerate
    (empty kept set; the all-zero vector is not a probability distribution
    and the conjecture is undefined there).

    Returns:
        total_checks (non-degenerate), n_skipped_degenerate,
        ce_spec, ce_rev, slacks_spec, slacks_rev
    """
    rng = np.random.default_rng(seed)
    ce_spec: list[dict] = []
    ce_rev: list[dict] = []
    slacks_spec: list[float] = []
    slacks_rev: list[float] = []
    total_checks = 0
    n_skipped = 0

    alpha_pairs = [
        (a, ap)
        for i, a in enumerate(ALPHA_GRID)
        for ap in ALPHA_GRID[i + 1:]
    ]

    for V in V_LIST:
        # Dirichlet(1,...,1) is uniform on the simplex
        samples = rng.dirichlet(np.ones(V), size=N_TRIALS_PER_V)
        for p in samples:
            cache: dict[float, np.ndarray] = {}
            for a, ap in alpha_pairs:
                if a not in cache:
                    cache[a] = trunc_alpha(p, a)
                if ap not in cache:
                    cache[ap] = trunc_alpha(p, ap)
                qa = cache[a]
                qap = cache[ap]
                if not (is_valid_distribution(qa) and is_valid_distribution(qap)):
                    n_skipped += 1
                    continue
                total_checks += 1

                # Spec direction: q_a <prec q_ap
                slack_s = min_slack(qa, qap)
                slacks_spec.append(slack_s)
                if slack_s < -1e-9:
                    k = violation_index(qa, qap)
                    ce_spec.append({
                        "where": "random", "V": V, "p": p.copy(),
                        "alpha": a, "alpha_prime": ap, "k": k,
                        "q_a": qa.copy(), "q_ap": qap.copy(), "slack": slack_s,
                    })

                # Reverse direction: q_ap <prec q_a
                slack_r = min_slack(qap, qa)
                slacks_rev.append(slack_r)
                if slack_r < -1e-9:
                    k = violation_index(qap, qa)
                    ce_rev.append({
                        "where": "random", "V": V, "p": p.copy(),
                        "alpha": a, "alpha_prime": ap, "k": k,
                        "q_a": qa.copy(), "q_ap": qap.copy(), "slack": slack_r,
                    })
    return total_checks, n_skipped, ce_spec, ce_rev, slacks_spec, slacks_rev


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    print("=" * 72)
    print("Sanity check on hand-worked example p=[0.5, 0.3, 0.2]")
    print("=" * 72)
    sanity_check()

    print("=" * 72)
    print("Edge-case sweep")
    print("=" * 72)
    edge_trials, edge_skipped, edge_ce_spec, edge_ce_rev = check_edge_cases()
    print(f"  edge trials (non-degenerate)         : {edge_trials}")
    print(f"  edge skipped (degenerate Trunc)      : {edge_skipped}")
    print(f"  edge counterexamples (spec direction): {len(edge_ce_spec)}")
    print(f"  edge counterexamples (reverse dir)   : {len(edge_ce_rev)}\n")

    print("=" * 72)
    print(
        f"Random sweep: V in {V_LIST}, "
        f"alphas in {ALPHA_GRID}, N={N_TRIALS_PER_V} per V"
    )
    print("=" * 72)
    rand_checks, rand_skipped, rand_ce_spec, rand_ce_rev, slacks_spec, slacks_rev = random_sweep()
    print(f"  random trials (non-degenerate)         : {rand_checks}")
    print(f"  random skipped (degenerate Trunc)      : {rand_skipped}")
    print(f"  random counterexamples (spec direction): {len(rand_ce_spec)}")
    print(f"  random counterexamples (reverse dir)   : {len(rand_ce_rev)}\n")

    total_checks = edge_trials + rand_checks
    all_ce_spec = edge_ce_spec + rand_ce_spec
    all_ce_rev = edge_ce_rev + rand_ce_rev
    print("=" * 72)
    print(f"TOTAL checks per direction: {total_checks}")
    print(f"TOTAL spec-direction counterexamples   (q_a <prec q_ap)  : {len(all_ce_spec)}")
    print(f"TOTAL reverse-direction counterexamples (q_ap <prec q_a) : {len(all_ce_rev)}")
    print("=" * 72)

    def _print_ce(label: str, ces: list[dict]) -> None:
        if not ces:
            return
        print(f"\nFirst (up to) 5 {label} counterexamples:")
        for i, ce in enumerate(ces[:5]):
            print(f"\n--- {label} counterexample #{i + 1} ({ce['where']}) ---")
            if "V" in ce:
                print(f"  V = {ce['V']}")
            if "name" in ce:
                print(f"  name = {ce['name']}")
            print(f"  p          = {ce['p']}")
            print(f"  alpha      = {ce['alpha']}")
            print(f"  alpha'     = {ce['alpha_prime']}")
            print(f"  k (0-idx)  = {ce['k']}")
            print(f"  q_alpha    = {ce['q_a']}")
            print(f"  q_alpha'   = {ce['q_ap']}")
            if label == "spec":
                x, y = ce['q_a'], ce['q_ap']
                xn, yn = "q_alpha", "q_alpha'"
            else:
                x, y = ce['q_ap'], ce['q_a']
                xn, yn = "q_alpha'", "q_alpha"
            x_sorted = np.sort(x)[::-1]
            y_sorted = np.sort(y)[::-1]
            cx = np.cumsum(x_sorted)
            cy = np.cumsum(y_sorted)
            k = ce['k']
            print(f"  testing: {xn} <prec {yn}")
            print(f"  S_k({xn})  = {cx[k]:.12f}")
            print(f"  S_k({yn})  = {cy[k]:.12f}")
            print(f"  delta = S_k({xn}) - S_k({yn}) = {cx[k] - cy[k]:.3e}")

    _print_ce("spec", all_ce_spec)
    _print_ce("reverse", all_ce_rev)

    def _slack_summary(name: str, slacks: list[float]) -> None:
        if not slacks:
            return
        arr = np.array(slacks)
        passing = arr[arr >= -1e-9]
        print(f"\nMin-slack distribution ({name} direction):")
        print(f"  n passing / n total : {passing.size} / {arr.size}")
        if passing.size:
            for q in (0.01, 0.10, 0.50, 0.90, 0.99):
                print(f"  quantile {q:>4.0%} : {np.quantile(passing, q):.6e}")
            print(f"  min          : {passing.min():.6e}")
            print(f"  max          : {passing.max():.6e}")
        if (arr < -1e-9).any():
            failing = arr[arr < -1e-9]
            print(f"  n failing    : {failing.size}")
            print(f"  worst slack  : {failing.min():.6e}")

    _slack_summary("spec (q_a <prec q_ap)", slacks_spec)
    _slack_summary("reverse (q_ap <prec q_a)", slacks_rev)

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    spec_holds = not all_ce_spec
    rev_holds = not all_ce_rev
    if spec_holds:
        print("CONJECTURE (spec direction, q_alpha <prec q_alpha'): HOLDS")
    else:
        print(
            f"CONJECTURE (spec direction, q_alpha <prec q_alpha'): "
            f"FALSE -- {len(all_ce_spec)} counterexamples"
        )
    if rev_holds:
        print("CONJECTURE (reverse direction, q_alpha' <prec q_alpha): HOLDS")
    else:
        print(
            f"CONJECTURE (reverse direction, q_alpha' <prec q_alpha): "
            f"FALSE -- {len(all_ce_rev)} counterexamples"
        )

    print()
    if spec_holds:
        print("PHASE-1 PROCEED")
    else:
        print("PHASE-1 ABORT")


if __name__ == "__main__":
    main()
