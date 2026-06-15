# Decoder comparison — ATCODER-interview, Qwen3-8B (thinking on, n=10)

All values pulled live and cross-verified by `scripts/build_decoder_comparison_table.py`. pass@k recomputed from raw pass_results (unbiased estimator, Chen 2021) and checked vs stored; trunc% recomputed from `</think>` presence and checked vs the cot_efficiency CSV; mean think tok from the cot_efficiency CSV; cb_div via the project's `add_self_codebleu`.

**Sources:** full252 `pless_recovery_full252/`, canonical `pless_cot_efficiency_vllm/.../ATCODER_interview_all_252/`, T0.6 decoders `decoders_t0.6/`. All n_tasks=252 except where noted.

| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |
|---|---|---|---|---|---|---|
| temp p0.95 @T1.0 | 252 | 0.705 | 0.821 | 0.4965 | 11,062 | 0.2 |
| temp k20 @T1.0 | 252 | 0.700 | 0.841 | 0.5017 | 11,209 | 0.0 |
| temp @T0.6 (unfilt) | 252 | 0.699 | 0.841 | 0.4757 | 11,112 | 0.4 |
| top_k @T0.6 | 252 | 0.698 | 0.821 | 0.4789 | 11,248 | 0.8 |
| pless α=4 | 252 | 0.696 | 0.821 | 0.4689 | 11,279 | 1.4 |
| top_p @T0.6 | 252 | 0.695 | 0.841 | 0.4797 | 11,253 | 1.2 |
| pless T2.0 | 252 | 0.694 | 0.821 | 0.4609 | 11,027 | 0.2 |
| pless α=5 | 252 | 0.686 | 0.833 | 0.4746 | 11,124 | 0.6 |
| temp p+k @T0.6 | 252 | 0.680 | 0.829 | 0.4681 | 11,127 | 1.0 |
| pless α=3 | 252 | 0.676 | 0.806 | 0.4663 | 11,495 | 2.7 |
| pless_norm @α2 | 252 | 0.629 | 0.829 | 0.4573 | 13,573 | 16.0 |
| pless @α2 (base) | 252 | 0.625 | 0.825 | 0.4528 | 13,485 | 14.5 |
| pless_norm @T0.6 | 252 | 0.619 | 0.806 | 0.4513 | 14,149 | 18.4 |
| pless @T0.6 | 252 | 0.615 | 0.825 | 0.4550 | 14,304 | 19.0 |

## Cross-verification

✓ All pass@k (recomputed vs stored) and trunc% (recomputed vs CSV) agree to tolerance. No mismatches.

## Caveats

- **cb_div** is correct-only over each config's own ≥2-correct subset (~195–202 tasks, ~95% overlapping) — mild cross-config confound; ranking robust, small gaps may shift on a common set.
- **structural_diversity** omitted: zss tree-edit is intractable on APPS-CoT code (a single ~2000-AST-node pair times out >60s) — the reason these runs use `--skip-diversity`.
- **mean think tok** counts truncated samples at their cut length (≈cap), so it is biased UP for the high-truncation configs — which is exactly the rambling cost it exposes.
