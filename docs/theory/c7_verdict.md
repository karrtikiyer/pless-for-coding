# C7 Verdict (v2): naive formula fails, beta-binomial refit warranted

**Update 2026-05-19 (v2):** the original verdict (v1) prematurely
recommended workshop after the naive formula failed. A self-critique
identified six errors in the v1 analysis, with #1 being load-bearing:
the recent literature has a standard Bayesian framework
([arXiv:2510.05197](https://arxiv.org/abs/2510.05197)) for exactly
this prediction problem, which I didn't apply. This v2 recommends a
3–5 day beta-binomial refit BEFORE deciding ICLR vs workshop.

---

## What we did (smoke-test C7)

Implemented the headline naive prediction:

```
pass@1_task(α) = pass@1_task(α=2) · ∏_t (f_{t,α=2} / f_{t,α})
```

Then aggregated to pass@k via iid binomial. Ran on Qwen MBPP and
CodeLlama MBPP.

## What we observed (the failure)

| Cell | α | Predicted pass@1 | Measured pass@1 | err (pp) |
|---|---|---:|---:|---:|
| Qwen MBPP | 2.0 (calib) | 77.08% | 77.08% | 0.00 |
| Qwen MBPP | 5.0 | 21.20% | 75.32% | **54.12** |
| CodeLlama MBPP | 2.0 (calib) | 41.78% | 41.78% | 0.00 |
| CodeLlama MBPP | 5.0 | 22.11% | 40.32% | **18.21** |

Naive formula gets the direction right but the magnitude is off by
~30×. More damningly, predicted pass@10 *decreases* with α; measured
pass@10 *increases* with α.

## Six things I got wrong in v1

### 1. (Worst) Picked a strawman parametric model when the right framework was already in the 2025 literature

**[Efficient Prediction of Pass@k Scaling in LLMs, arXiv:2510.05197](https://arxiv.org/abs/2510.05197)** (Oct 2025) models per-problem pass-rate as a **beta-binomial distribution**, not a point estimate. They demonstrate this is the principled approach to predicting pass@k at higher k from limited samples. **[Don't Pass@k: A Bayesian Framework, arXiv:2510.04265](https://arxiv.org/abs/2510.04265)** (Oct 2025) uses a Dirichlet prior with closed-form posterior.

I should have started from beta-binomial. The naive multiplicative formula was nearly guaranteed to fail — per-problem `p` is *distributed*, not a point.

### 2. The "constant correct set" assumption was textbook strawman

Multiple correct continuations are obvious at high-entropy positions in code (variable naming, helper structure, syntactically equivalent algorithms). Assuming `C_t` is constant across α was almost certain to fail before any data ran. I should have flagged this strongly upfront rather than building a theorem around it.

### 3. The "sub-iid correlation" interpretation in v1 is incoherent

I claimed pass@10 rising while pass@1 falling implies "samples are MORE diverse than iid." Mathematically wrong: iid is the *maximum* independence — you can't be "less correlated" than iid.

The real mechanism is **per-problem pass-rate heterogeneity**: at high α, the *distribution* of per-problem `p` widens. Some problems gain `p > 0` (newly solvable via broader sampling); most problems' `p` stays roughly constant. Dataset-level `pass@10 = E_problem[1-(1-p)^10]` benefits from the long-tail shift even though average `p` barely moves.

This is exactly what beta-binomial captures.

### 4. The cumulative log-product is exponentially sensitive

`pass@1(α) = pass@1(α=2) · ∏_t (f_{α=2}/f_{α})` over ~150 positions per sample. A 10% per-position error in the multiplicand compounds to e^15 ≈ 3M-fold error in the product. No error budget for the load-bearing assumption to be even slightly off. Bad design.

### 5. Implementation: used top-32 approximation when exact data was logged

The sidecar already has `sigma_p2`, `sigma_p3`, `sigma_p5` (over full vocab). I computed thresholds from top-32 instead, which underestimates at high-entropy positions — exactly where the prediction needs accuracy. Sloppy.

### 6. Decision in 4 hours was overconfident

I declared "workshop path" after the first naive theorem failed. The literature has the right framework; trying it before pivoting is the correct research move. My time-to-decision was about 5× too fast.

## What's actually salvageable

Two empirical observations from the failure remain valuable:

1. **Per-position iid factorization fails for α-sweep on code.** A falsifiable claim with quantitative magnitude (off by ~30× on Qwen). Stands as a negative result.
2. **The pass@10 lift comes from per-problem heterogeneity, not per-sample factorization.** Specifically: long-tail problems become solvable at high α. Beta-binomial framework directly captures this.

## The right next step (v2 recommendation)

**Implement the beta-binomial framework on our existing data** (~3–5 days CPU work).

Concretely:

1. **Fit** `Beta(a_α, b_α)` to the per-problem pass-rate at each α via method of moments. This requires per-problem `num_correct` (already in the metrics JSONs).
2. **Compute** pass@k under the fitted Beta in closed form:
   `pass@k = 1 - B(a, b+k) / B(a, b) = 1 - ∏_{i=0}^{k-1} (b+i)/(a+b+i)`
3. **Sanity check**: does the fitted Beta reproduce the measured pass@k for k > 1? (If yes, the framework fits the data; if not, the per-problem distribution isn't actually beta-distributed and we need a different parametric family.)
4. **Look at how** `(a_α, b_α)` shifts as α grows. Hypothesis: `a/(a+b)` stays roughly constant (matches small pass@1 drop) while concentration `a+b` decreases (per-problem variance grows).
5. **If the shift has structure**, try to PREDICT `(a_α, b_α)` from `(a_2, b_2)` + per-position entropy statistics. This is the actual theorem candidate.

If step 4 shows clean structure (e.g., concentration decays as a function of mean entropy under α), we have a meaningful theoretical contribution. If not, fall back to workshop with the negative result.

## Revised decision rule

After the beta-binomial fit (~3–5 days):

| Outcome | Action |
|---|---|
| Step 3 succeeds (fitted Beta reproduces measured pass@k within sampling noise) AND step 4 shows clear `α → (a, b)` structure | Continue to ICLR theory-track |
| Step 3 succeeds, step 4 shows no clean structure | Workshop with a richer empirical "beta-binomial decomposition" section |
| Step 3 fails (beta-binomial doesn't fit) | Try Dirichlet-multinomial (arXiv:2510.04265); else workshop |

## Cost

CPU-only. No GPU. ~3–5 days of focused theory + Python. Total path-to-decision: under a week.

## Time spent on v1 (the failed naive attempt)

About 4 hours from "let us start C7" to v1 verdict. Bounded and informative. The cost of the v2 refit (3–5 days) is justified by the prior-work check showing this is a solved class of problem in the literature.

## References

- Tan, Wu, Howard 2025 ([arXiv:2509.23234v6](https://arxiv.org/abs/2509.23234), ICLR 2026 oral) — p-less and its α-generalization
- Chen et al. 2021 ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)) — pass@k unbiased estimator
- **Efficient Prediction of Pass@k Scaling, [arXiv:2510.05197](https://arxiv.org/abs/2510.05197)** (Oct 2025) — beta-binomial framework
- **Don't Pass@k: A Bayesian Framework, [arXiv:2510.04265](https://arxiv.org/abs/2510.04265)** (Oct 2025) — Dirichlet-based alternative
