# Decoder comparison — CODEFORCES-interview (748), DeepSeek-R1-Distill-Llama-8B (CODEFORCES-interview, 748) (thinking on, n=10)

All values pulled live by `scripts/build_decoder_comparison_table.py`. pass@k recomputed from raw pass_results (unbiased, Chen 2021) & checked vs stored; trunc% from `</think>` presence; mean think tok from the cot CSV when present else re-tokenized from the jsonl; cb_div via the project's `add_self_codebleu` (correct-only, no execution).

| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |
|---|---|---|---|---|---|---|
| top-p 0.9 | 748 | 0.484 | 0.702 | 0.5061 | 8,927 | 0.0 |
| G_k=0.35 | 748 | 0.472 | 0.694 | 0.5149 | 9,386 | 0.1 |
| G_k=0.8 | 748 | 0.439 | 0.655 | 0.4572 | 14,381 | 28.7 |
| G_k=1.6 | 748 | 0.396 | 0.623 | 0.4451 | 17,618 | 44.1 |
| pless_norm | 748 | 0.386 | 0.618 | 0.4417 | 18,136 | 47.0 |
| pless @k2 (base) | 748 | 0.382 | 0.611 | 0.4416 | 17,559 | 47.0 |

**Pending (not scored yet):** temp T0.6 p0.95 (rec).

## Cross-verification

✓ pass@k (recomputed vs stored) and trunc% agree to tolerance.

## Caveats
- **cb_div** correct-only over each config's own ≥2-correct subset (mild cross-config confound).
- **mean think tok** counts truncated samples at their cut length (≈cap) → biased UP for high-truncation configs (the rambling cost). Re-tokenized estimate when no CSV (±few %).
- **structural_diversity** omitted (zss intractable on APPS-CoT — the reason for `--skip-diversity`).
