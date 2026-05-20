# C7 — Closed-form pass@k(α): Plan

## What C7 is

A closed-form expression `pass@k(α) = F(α, per-position model statistics)`
derived from first principles, predicting our empirical α-curves on
code generation. If the prediction fits within ~3 pp on at least one
(model, benchmark) cell, C7 is the keystone for an ICLR theory-track
submission. Otherwise, we fall back to workshop with the empirical
contribution.

## Three-phase plan

### Phase 1 — Theory (target: 1 day for smoke-test version)

Document: `docs/theory/c7_pass_at_k_alpha_derivation.md`

**Headline theorem (informal):** Under two assumptions —
(A) per-position independence of correctness, and
(B) correct-set robustness (the correct continuation set is preserved
across α≥2 truncations) — the per-problem pass-rate at α scales as

```
pass@1_problem(α) = pass@1_problem(α₀) · ∏_t (f_{t,α₀} / f_{t,α})
```

where `f_{t,α} = Σ_{i : p_t(i) ≥ T_{t,α}} p_t(i)` is the kept mass at
position t under α-truncation. The full theorem extends this to pass@k
via the standard binomial aggregation (Chen et al. 2021).

This is computable purely from per-position model statistics we
already log (the entropy sidecars).

### Phase 2 — Empirical validation (target: 1 day for smoke-test)

Script: `bench/eval/validate_pass_at_k_prediction.py`

Inputs:
- `results/pless_alpha_entropy/{Qwen,CodeLlama}/pless_t1.0.jsonl.entropy.jsonl`
  (280K+ per-position records with `sigma_p2/3/5`, `max_p`, top-32)
- `results/pless_alpha_full/{model}/metrics/pless_alpha_a*_t1.0_metrics.json`
  (per-task pass@k for α ∈ {2.0, 2.5, 3.0, 5.0})

Process:
1. For each position record, compute `f_{t,α}` for α ∈ {2.0, 2.5, 3.0, 5.0}
2. For each (task, sample): compute log-ratio `Σ_t log(f_{t,α=2} / f_{t,α})`
3. Average over samples per task → per-task log-ratio
4. Calibrate `pass@1_task(α=2)` from observed metrics
5. Predicted `pass@1_task(α) = pass@1_task(α=2) · exp(log-ratio)`
6. Aggregate to pass@1, pass@5, pass@10 across tasks
7. Compare predicted vs measured per (model, α)

Output:
- `results/c7_validation/fit_summary.json` — per-cell errors
- `results/c7_validation/predicted_vs_measured_*.png` — fit quality plots
- `results/c7_validation/summary.md` — verdict

### Phase 3 — Decision (target: same day)

Decision rule:

| Phase 2 outcome | Next action |
|---|---|
| pass@1 fit ≤ 3 pp absolute on ≥1 cell | Continue to C1 + C10 for ICLR push |
| pass@1 fit ≤ 5 pp on ≥1 cell, qualitatively right | Refine assumptions, ~1 more week |
| Fit fails everywhere (>10 pp) | Workshop path, honest pivot |

## Risk register

| Risk | Mitigation |
|---|---|
| Independence assumption breaks | Have a "correlated-errors" refinement v2 ready |
| Correct-set-robustness fails | Estimate kept correct mass from passing-samples directly (Phase 2 fallback estimator) |
| m-a-p has no entropy logged | 2-model validation is enough; m-a-p adds a 30-min GPU run if needed |
| Per-position f_{t,α} approximation from top-32 is inaccurate | Tail beyond top-32 contributes <0.01% of mass; rounding error is below noise floor |

## What we'll need

- CPU only (no GPU, no API). Existing data sufficient for smoke test.
- Optional: 30 min GPU later for m-a-p entropy logging (extends C7 to 3rd model).

## Smoke-test vs full plan

Smoke test = 1-day Phase 1 + 1-day Phase 2 on Qwen MBPP alone.
Full plan = 7-day Phase 1 + 5-day Phase 2 across all 4 covered cells.

Start with smoke test. If predicted pass@10(α) on Qwen MBPP lands
within ±3 pp of measured at α=5, commit to full plan. Otherwise pivot.
