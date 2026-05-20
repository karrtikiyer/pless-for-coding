# C7 v3 Step 5: predicting per-task Δp from α=2 entropy — partial signal, no clean mechanism

**Status (2026-05-19):** Step 5 of the v2 roadmap shows entropy
features add a small but real lift in R² (≈+3–5 pp absolute) over the
naive "p at α=2 alone" baseline, but the lift is well below the
noise-corrected ceiling and the fitted coefficients flip sign across
models. Per-task prediction from per-position entropy is not the
right vehicle for an ICLR theorem.

## What we did

For Qwen2.5-Coder-7B-Instruct and CodeLlama-7B-Instruct on MBPP-500,
we have entropy sidecars logged at α=2 (σ_p2, σ_p3, σ_p5, max_p,
top-32 per position; 500 tasks × 10 samples each). We regressed
per-task `Δp_i = c_i^(α)/10 − c_i^(α=2)/10` on per-task aggregate
entropy features for α ∈ {2.5, 3.0, 5.0}.

Models tested:
- **null**: intercept only — sanity check
- **baseline**: `Δp ~ intercept + β·p_α=2` ("how much room to grow")
- **full**: baseline + 10 entropy features (mean Rényi entropies at
  α∈{2,3,5}, variance of log σ_p2, kept-mass log-ratio means, mean
  max_p, sample length)
- **Δ-only**: entropy features without p_α=2
- **single**: baseline + a single `mean_log_ratio` feature

## Results

| Model | α | null | baseline | full | Δ-only | single |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder | 2.5 | 0 | 0.113 | 0.140 | 0.020 | 0.113 |
| Qwen2.5-Coder | 3.0 | 0 | 0.129 | 0.174 | 0.042 | 0.129 |
| Qwen2.5-Coder | 5.0 | 0 | 0.156 | 0.190 | 0.030 | 0.160 |
| CodeLlama | 2.5 | 0 | 0.084 | 0.095 | 0.009 | 0.085 |
| CodeLlama | 3.0 | 0 | 0.122 | 0.140 | 0.016 | 0.128 |
| CodeLlama | 5.0 | 0 | 0.178 | 0.202 | 0.025 | 0.184 |

**R² ceiling** (computed from per-task binomial noise on `Δp`):
Qwen MBPP α=2→5: **0.80**. So perfect features could reach R²≈0.80;
we capture 0.19/0.80 ≈ 24% of recoverable signal.

## What this means

### Three findings, ordered by importance

**1. Most "predictability" of Δp comes from `p_α=2` itself.** Tasks
already at `c=10/10` can't grow further; tasks at `c=0/10` are most
likely to gain. The baseline-only R² (0.08–0.18) captures this.
Entropy features add a modest 2–5 pp R² on top.

**2. The R² ceiling is ~0.80, not ~0.20.** Sample noise alone
wouldn't cap us at 0.20. There is signal we could in principle
recover with better features — but our current α=2 entropy summary
statistics aren't capturing it.

**3. Coefficient signs flip between Qwen and CodeLlama for
key features.** Example: `mean_log_ratio_23` is +20.9 for Qwen α=5
but −4.8 for CodeLlama α=5. This is the smoking gun: the model
isn't identifying a stable mechanism — it's exploiting feature
correlations differently per model. Not a theorem candidate.

### Why our entropy features aren't enough

The features we logged at α=2 are vocab-level **distribution
statistics** at each position. To predict per-task Δp well, we'd
also need:

1. **Per-position correctness alignment**: which top-k tokens at
   each position actually lead to passing samples. We don't have
   this — we'd need to align passing-sample token traces with
   per-position distributions. *Doable from our data with more
   work.*
2. **Counterfactual kept-mass at higher α**: f_α values for
   α ∈ {2.5, 3, 5} per position. We have top-32, which suffices
   for low-entropy positions but truncates at high-entropy ones.
   The σ_pα values give us the *threshold*, not the *kept mass*.
3. **Per-sample features, not per-task aggregates**: averaging
   over 10 samples washes out the bimodal-entropy signature at
   key high-leverage positions.

## Decision

**Per-task entropy-based prediction is not the path to an ICLR
theorem.** Even if pursued with the feature improvements above,
mechanism identifiability is in doubt given Qwen/CodeLlama sign
flips.

**The C7 v3 main finding holds (Steps 3 + 4):**
- Beta-binomial fits per-task pass-rate within ~0.3 pp.
- `ν(α) = a_α + b_α` grows monotonically across all 6 (model, dataset)
  cells while mean `p_α` is nearly flat.

That regularity is independently publishable as an *empirical*
characterization of α-family decoding for code, even without the
predictive entropy theorem.

## What to recommend next

Three options, decreasing in ambition:

### (a) Workshop-paper path (recommended)

- C7 v3 finding (beta-binomial decomposition + ν(α) growth + flat
  mean) as the **empirical headline**.
- Bimodal-entropy result (Hartigan dip test) as **mechanism evidence**.
- T-envelope/Pareto-dominance + NAUADC + 3-model × 2-benchmark sweep
  as **breadth evidence**.
- Honest "no closed-form predictor of (a_α, b_α) from per-position
  entropy at our sample budget" caveat.
- Target: code generation workshop at ICLR or NeurIPS.

### (b) Push to ICLR theory-track with token-level traces (1–2 weeks)

- Augment data with per-passing-sample token traces aligned to
  per-position distributions.
- Compute exact per-position correct-set retention rate under each α.
- This is the v1 theorem's *correct* implementation. Whether it
  succeeds depends on whether the per-sample correct-set assumption
  holds; bimodal entropy distribution suggests it might.
- Risk: another 1–2 weeks could yield another negative result.

### (c) Switch to predicting (a_α, b_α) from population-level
features (no extra data needed)

- 6 cells × 4 α = 24 data points.
- 2 cells have entropy data (Qwen MBPP, CodeLlama MBPP) = 8 data
  points for entropy-conditional prediction.
- Probably too few data points for credible fit.

**Recommendation:** go with (a) — write up the workshop paper. The
C7 v3 finding is solid, novel, and well-supported. Skip the theorem
chase.

## Output artifacts

- `results/c7_validation/step5_entropy_prediction/regression_summary.{md,json}`
- `results/c7_validation/step5_entropy_prediction/per_task_features_*.json`
- `results/c7_validation/step5_entropy_prediction/scatter_*.png`

## References

- [arXiv:2509.23234](https://arxiv.org/abs/2509.23234) Tan, Wu, Howard 2025 (upstream)
- [arXiv:2510.05197](https://arxiv.org/abs/2510.05197) beta-binomial pass@k
- [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) Chen et al. pass@k unbiased estimator
