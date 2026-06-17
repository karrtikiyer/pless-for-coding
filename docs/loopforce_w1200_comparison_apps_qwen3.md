# Loop-force (w1200) vs baselines — ATCODER-interview, Qwen3-8B (thinking on, n=10, full 252)

Live + cross-verified by `scripts/loopforce_w1200_comparison.py`. pass@k = unbiased estimator (Chen 2021) recomputed from raw pass_results and checked vs stored; trunc%/compl%/cond-correctness from `</think>` presence + pass_results; cb_div via the project's `add_self_codebleu` (CodeBLEU, correct-only, **no re-execution**); mean think tok from the cot_efficiency CSV.

Loop-force = live n-gram detect (n=30/k=6) → force `</think>` at **window=1200**. Baseline = same sampler, no loop-force.

| Config | n | trunc% | compl% | cond-corr | pass@1 | pass@5 | pass@10 | cb_div | (cb n≥2) | mean think tok |
|---|---|---|---|---|---|---|---|---|---|---|
| temp p0.95 @T1.0 | 252 | 0.2 | 99.8 | 0.706 | 0.705 | 0.806 | 0.821 | 0.4967 | 202 | 11,062 |
| temp k20 @T1.0 | 252 | 0.0 | 100.0 | 0.700 | 0.700 | 0.815 | 0.841 | 0.5018 | 202 | 11,209 |
| temp @T0.6 (unfilt) | 252 | 0.4 | 99.6 | 0.702 | 0.699 | 0.812 | 0.841 | 0.4758 | 201 | 11,112 |
| pless loop-force w1200 | 252 | 1.5 | 98.5 | 0.663 | 0.653 | 0.783 | 0.813 | 0.4555 | 194 | 10,058 |
| pless_norm loop-force w1200 | 252 | 1.6 | 98.4 | 0.661 | 0.651 | 0.783 | 0.806 | 0.4710 | 196 | 10,078 |
| pless_norm @α2 (no-force base) | 252 | 16.0 | 84.0 | 0.748 | 0.629 | 0.791 | 0.829 | 0.4576 | 195 | 13,573 |
| pless @α2 (no-force base) | 252 | 14.5 | 85.5 | 0.732 | 0.625 | 0.792 | 0.825 | 0.4527 | 196 | 13,485 |

## Cross-verification

✓ pass@k (recomputed vs stored) and trunc% (recomputed vs CSV) agree to tolerance.

## Caveats

- **cb_div** is correct-only over each config's own ≥2-correct subset (see `(cb n≥2)`); a config that solves fewer tasks computes diversity over a smaller/harder set — mild cross-config confound, ranking robust.
- **mean think tok** counts truncated samples at their cut length (≈cap), so it is biased UP for high-truncation configs — exactly the rambling cost loop-force removes.
- **structural (zss) diversity** omitted — intractable on APPS-CoT code (the reason these runs use `--skip-diversity`); CodeBLEU only.
- The α=5 / T2.0 prevention winners (`pless_recovery_full252/`) are not on local disk, so they're omitted here rather than quoted from memory.
