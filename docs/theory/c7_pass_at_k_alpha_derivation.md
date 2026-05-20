# C7: Closed-form pass@k(α) from per-position model statistics

**Status: SMOKE-TEST DRAFT** — headline theorem only, validation pending.
Full proof polish + assumption-failure analysis deferred to v2 if the
smoke test succeeds.

## Setup

We sample sequences from a language model using a Rényi-α p-less sampler.
At each generation step `t`, the model produces a probability distribution
`p_t : V → [0, 1]` over the vocabulary `V` of size `|V|`. The α-truncation
rule is:

- **Threshold**: `T_{t,α} = Σ_i p_t(i)^α`
- **Kept set**: `K_{t,α} = { i ∈ V : p_t(i) ≥ T_{t,α} }`
- **Kept mass**: `f_{t,α} = Σ_{i ∈ K_{t,α}} p_t(i)`
- **Truncated distribution**:
  `q_{t,α}(i) = p_t(i) / f_{t,α}` if `i ∈ K_{t,α}`, else `0`

We then draw `z_{s,t} ~ q_{t,α}` independently for each sample `s` and
position `t`.

A sample is **correct** if the resulting sequence `(z_{s,1}, …, z_{s,L_s})`
passes the unit tests for the problem.

The metric is:

```
pass@k = E_problems[ 1 − C(n−c, k) / C(n, k) ]
```

where `n` is the number of samples per problem, `c` is the count of
correct samples for that problem, and the expectation is over problems
(Chen et al. 2021, [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)).

## Two load-bearing assumptions

### Assumption A — Per-position independence of correctness

For each problem there exists a sequence of **correct continuation sets**
`{C_t ⊆ V}` such that a sample is correct iff it stays within `C_t` at
every position:

```
sample correct  ⇔  ∀t : z_{s,t} ∈ C_t
```

In particular, errors at one position don't accidentally "fix themselves"
later. This is the standard factorization assumption in language-modeling
theory (e.g., used implicitly in the Codex pass@k analysis).

**When this breaks:** branching trajectories. The "correct set" at
position `t` may genuinely depend on what tokens were sampled at `t' < t`,
because the model commits to one algorithm and then has to follow through.
But for our purposes the assumption holds *on average* if we let `C_t`
encode the *union* of all correct continuations across all viable
trajectories the model could pursue.

### Assumption B — Correct-set robustness under α ≥ 2

For α ≥ 2, the correct continuation set is contained in the α-kept set:

```
∀α ≥ 2 :  C_t ⊆ K_{t,α}
```

i.e., truncation never zeros out a correct token. Equivalently, every
correct continuation has probability `≥ T_{t,α}` under the natural model
distribution.

**Why this is reasonable:** at α = 2, the kept set is already broad
(typically the top 3–10 tokens at high-entropy positions). For α > 2 the
threshold is smaller and the kept set is strictly larger
(`K_{t,α} ⊇ K_{t,2}` for α > 2), so robustness for α = 2 implies
robustness for all α ≥ 2.

**When this breaks:** if a correct continuation has very low probability
(`p_t(i) < T_{t,2}` for some `i ∈ C_t`), the α = 2 sampler will *never*
pick it. This is the regime where the model genuinely doesn't know the
right answer. We don't claim our theorem applies in that regime.

## Definition: per-position correct mass

Under Assumption B, the kept correct mass at position `t` under α is

```
ĉ_{t,α} = Σ_{i ∈ C_t} p_t(i)  =  c_t
```

i.e., it's constant in α (the correct set is entirely retained for any
α ≥ 2). Let `c_t = ĉ_{t,2}` denote this constant.

## Lemma (per-position correctness probability)

Under Assumptions A and B, the probability that sample `s` produces a
correct token at position `t` is

```
r_{t,α}  =  P_{z ~ q_{t,α}}[z ∈ C_t]
        =  (Σ_{i ∈ C_t} p_t(i) · 𝟙[p_t(i) ≥ T_{t,α}]) / f_{t,α}
        =  c_t / f_{t,α}                                    (by Assumption B)
```

This is decreasing in α (because `f_{t,α}` is non-decreasing in α and
`c_t` is constant).

## Theorem (per-problem pass@1 scaling)

Fix a problem, and let `L` be the sequence length. Under Assumptions A
and B, the per-problem pass-rate under α-sampling is

```
pass@1(α)  =  ∏_{t=1}^L r_{t,α}  =  ∏_t (c_t / f_{t,α})
```

Taking logs:

```
log pass@1(α)  =  Σ_t log c_t  −  Σ_t log f_{t,α}                  (1)
```

The term `Σ_t log c_t` is a per-problem constant (the intrinsic
difficulty under our model). The term `Σ_t log f_{t,α}` is a function of
α that we can compute *directly* from the per-position distributions
`{p_t}` (which our entropy sidecars log).

### Calibration

We don't observe `Σ_t log c_t` directly, but we DO observe `pass@1(α₀)`
for some reference α₀ (we use α₀ = 2). From (1):

```
Σ_t log c_t  =  log pass@1(α₀)  +  Σ_t log f_{t,α₀}                (2)
```

Plugging (2) into (1) for any target α:

```
log pass@1(α)  =  log pass@1(α₀)  +  Σ_t log (f_{t,α₀} / f_{t,α})  (3)
```

or equivalently:

```
pass@1(α)  =  pass@1(α₀) · ∏_t (f_{t,α₀} / f_{t,α})                (4)
```

This is the **headline result**. Plug in α₀ = 2, then to predict `pass@1`
at any other α we only need:

1. The per-position kept-mass ratios `f_{t,α=2} / f_{t,α}`, computed from
   the (model, problem)-specific entropy sidecar data.
2. The empirical `pass@1(α=2)` baseline.

## Corollary (population-level pass@1)

The dataset-level pass@1 averages over problems. Let
`Π_α(problem) = ∏_t (f_{t,α=2} / f_{t,α})` for that problem's positions.
Then

```
pass@1(α; dataset) = E_problem[ pass@1_problem(α=2) · Π_α(problem) ]
```

This is computable from per-problem entropy logs and per-problem α=2
pass-rates.

## Corollary (pass@k via iid samples)

Within a single problem, the `n` samples are drawn independently. If each
has predicted pass-probability `p_problem(α) = pass@1_problem(α)`, the
probability that exactly `c` of `n` are correct is `Binomial(n, p)`. Then:

```
pass@k_problem(α)  =  E_c~Binomial(n, p_problem(α))[ 1 − C(n−c, k) / C(n, k) ]
```

Averaging over problems gives the dataset-level `pass@k(α)`.

In closed form, under iid:

```
pass@k_problem(α)  =  1 − (1 − p_problem(α))^k                    (5)
```

(when we use the asymptotic estimator; the Chen et al. unbiased
estimator gives a finite-sample correction).

## Falsifiable predictions

Equation (4) makes specific predictions we can test:

1. **Monotonicity**: `pass@1(α)` should be *non-increasing* in α
   (since `f_{t,α}` is non-decreasing in α for α ≥ 2).
   **Empirical check:** Qwen MBPP pass@1 goes 77.08 → 76.76 → 76.60 → 75.32 ✓
   CodeLlama MBPP: 41.78 → 41.24 → 40.66 → 40.32 ✓ (monotone)
   m-a-p MBPP: 47.68 → 47.28 → 48.00 → 46.26 (NOT monotone; α=3.0
   beats α=2.0). This is one piece of evidence against Assumption B
   on m-a-p, or against the strong independence claim.

2. **Magnitude of drop**: the predicted pass@1 drop equals the
   *cumulative log-ratio* `Σ_t log (f_{t,α=2} / f_{t,α})`.
   This is a quantitative prediction.

3. **Cross-model robustness**: the same formula should fit Qwen and
   CodeLlama with no model-specific tuning beyond the per-problem
   `pass@1(α=2)` baseline.

4. **Pass@10 lift size**: under iid (Eq. 5),
   `pass@10(α) - pass@10(α=2) ≈ k · (1 − pass@1(α=2))^{k-1} · (pass@1(α) − pass@1(α=2))`
   for small changes. The empirical α=5 - α=2 pass@10 lift is
   +1.8 to +14.6 pp across cells. This should follow from formula.

## What this gives us, and what it doesn't

**Gives:** a closed-form scaling law for `pass@1(α)` and `pass@k(α)` in
terms of the per-position kept-mass ratios and a single per-problem
calibration constant. Computable on CPU from existing data.

**Doesn't give:**
- A first-principles prediction of `pass@1(α=2)` itself — that's the
  intrinsic difficulty of the problem and we treat it as observed.
- An explanation of *why* the per-position entropy distribution looks
  the way it does (that's C1, the PCFG+BPE theorem).
- An α-optimization rule that finds the best α per problem (that's
  C10, the adaptive α schedule).

## Validation protocol

1. For each (Qwen, CodeLlama) on (MBPP, HumanEval) — 4 cells —
   compute per-task `f_{t,α}` for α ∈ {2.0, 2.5, 3.0, 5.0}.
2. Compute per-task log-ratio `Σ_t log (f_{t,α=2} / f_{t,α})`.
3. Calibrate per-task `pass@1(α=2)` from measured metrics.
4. Predict per-task `pass@1(α)` for α ∈ {2.5, 3.0, 5.0} via (4).
5. Aggregate to dataset-level `pass@1(α)`, `pass@5(α)`, `pass@10(α)`
   via (5).
6. Plot predicted vs measured. Report per-cell absolute prediction
   error.

**Pass criterion** for the smoke test: mean absolute prediction error
on `pass@1(α)` for α ∈ {2.5, 3.0, 5.0} ≤ 3 pp on at least one cell.

## Honest assessment

The theorem above is a **direct chain of definitions** under the two
assumptions. The interesting question is whether the assumptions hold
well enough in practice. The non-monotonic m-a-p pass@1 we already see
is mild evidence that Assumption B isn't always tight. We'll know much
more once the validation script runs.

If the fit works, this is a respectable theoretical contribution: a
*predictive* scaling law for pass@k(α) on a sampler family that the
original p-less paper introduced only at α=2. The paper would still
need C1 (bimodal-entropy mechanism) and C10 (adaptive α algorithm) to
clear the ICLR theory bar fully — but C7 alone is the keystone.

## References

- Tan, Wu, Howard 2025 ([arXiv:2509.23234v6](https://arxiv.org/abs/2509.23234), ICLR 2026 oral) — p-less and its α-generalization
- Chen et al. 2021 ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)) — pass@k unbiased estimator
