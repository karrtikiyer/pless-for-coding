# Decoder comparison — ATCODER-interview, Qwen3-8B (thinking on, n=10)

All values pulled live by `scripts/build_decoder_comparison_table.py`. pass@k recomputed from raw pass_results (unbiased, Chen 2021) & checked vs stored; trunc% from `</think>` presence; mean think tok from the cot CSV when present else re-tokenized from the jsonl; cb_div via the project's `add_self_codebleu` (correct-only, no execution).

| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |
|---|---|---|---|---|---|---|
| temp p0.95 @T1.0 | 252 | 0.705 | 0.821 | 0.4967 | 11,062 | 0.2 |
| temp k20 @T1.0 | 252 | 0.700 | 0.841 | 0.5017 | 11,209 | 0.0 |
| temp @T0.6 (unfilt) | 252 | 0.699 | 0.841 | 0.4757 | 11,112 | 0.4 |
| top_k @T0.6 | 252 | 0.698 | 0.821 | 0.4789 | 11,248 | 0.8 |
| pless α=4 | 252 | 0.696 | 0.821 | 0.4689 | 11,279 | 1.4 |
| top_p @T0.6 | 252 | 0.695 | 0.841 | 0.4798 | 11,253 | 1.2 |
| pless T2.0 | 252 | 0.694 | 0.821 | 0.4609 | 11,027 | 0.2 |
| pless α=5 | 252 | 0.686 | 0.833 | 0.4744 | 11,124 | 0.6 |
| adaptive (1-chop) | 252 | 0.682 | 0.845 | 0.4593 | 11,229 | 2.7 |
| temp p+k @T0.6 | 252 | 0.680 | 0.829 | 0.4681 | 11,127 | 1.0 |
| pless α=3 | 252 | 0.676 | 0.806 | 0.4664 | 11,495 | 2.7 |
| pless_norm @α2 | 252 | 0.629 | 0.829 | 0.4575 | 13,573 | 16.0 |
| pless @α2 (base) | 252 | 0.625 | 0.825 | 0.4525 | 13,485 | 14.5 |
| pless_norm @T0.6 | 252 | 0.619 | 0.806 | 0.4513 | 14,149 | 18.4 |
| pless @T0.6 | 252 | 0.615 | 0.825 | 0.4549 | 14,304 | 19.0 |

## Cross-verification

✓ pass@k (recomputed vs stored) and trunc% agree to tolerance.

## Caveats
- **cb_div** correct-only over each config's own ≥2-correct subset (mild cross-config confound).
- **mean think tok** counts truncated samples at their cut length (≈cap) → biased UP for high-truncation configs (the rambling cost). Re-tokenized estimate when no CSV (±few %).
- **structural_diversity** omitted (zss intractable on APPS-CoT — the reason for `--skip-diversity`).
