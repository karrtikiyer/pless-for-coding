# Decoder comparison — CODEFORCES-interview (748), Qwen3-8B (CODEFORCES-interview, 748) (thinking on, n=10)

All values pulled live by `scripts/build_decoder_comparison_table.py`. pass@k recomputed from raw pass_results (unbiased, Chen 2021) & checked vs stored; trunc% from `</think>` presence; mean think tok from the cot CSV when present else re-tokenized from the jsonl; cb_div via the project's `add_self_codebleu` (correct-only, no execution).

| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |
|---|---|---|---|---|---|---|
| G_k=0.05 | 748 | 0.648 | 0.795 | 0.4679 | 11,185 | 0.1 |
| G_k=0.1 | 748 | 0.647 | 0.794 | 0.4728 | 11,410 | 0.1 |
| G_k=0.2 | 748 | 0.646 | 0.789 | 0.4671 | 11,387 | 0.2 |
| top-p 0.9 | 748 | 0.644 | 0.790 | 0.4656 | 11,262 | 0.3 |
| G_k=0.4 | 748 | 0.641 | 0.782 | 0.4483 | 11,155 | 0.2 |
| top-p 0.85 | 748 | 0.640 | 0.775 | 0.4584 | 10,948 | 0.5 |
| top-p 0.8 | 748 | 0.638 | 0.778 | 0.4501 | 11,130 | 0.7 |
| top-p 0.7 | 748 | 0.634 | 0.773 | 0.4527 | 11,007 | 1.8 |
| top-p 0.75 | 748 | 0.633 | 0.778 | 0.4505 | 11,299 | 1.2 |
| temp T0.6 p0.95 k20 (rec) | 748 | 0.629 | 0.781 | 0.4441 | 11,279 | 1.4 |
| G_k=0.8 | 748 | 0.612 | 0.769 | 0.4379 | 11,944 | 7.6 |
| G_k=1.6 | 748 | 0.590 | 0.763 | 0.4305 | 13,381 | 14.6 |
| pless @k2 (base) | 748 | 0.579 | 0.747 | 0.4334 | 13,724 | 16.2 |
| pless_norm | 748 | 0.572 | 0.754 | 0.4371 | 13,279 | 16.1 |

## Cross-verification

✓ pass@k (recomputed vs stored) and trunc% agree to tolerance.

## Caveats
- **cb_div** correct-only over each config's own ≥2-correct subset (mild cross-config confound).
- **mean think tok** counts truncated samples at their cut length (≈cap) → biased UP for high-truncation configs (the rambling cost). Re-tokenized estimate when no CSV (±few %).
- **structural_diversity** omitted (zss intractable on APPS-CoT — the reason for `--skip-diversity`).
