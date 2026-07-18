# Decoder comparison — ATCODER-interview, DeepSeek-R1-Distill-Llama-8B (fixed vLLM, post-#45488) (thinking on, n=10)

All values pulled live by `scripts/build_decoder_comparison_table.py`. pass@k recomputed from raw pass_results (unbiased, Chen 2021) & checked vs stored; trunc% from `</think>` presence; mean think tok from the cot CSV when present else re-tokenized from the jsonl; cb_div via the project's `add_self_codebleu` (correct-only, no execution).

| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |
|---|---|---|---|---|---|---|
| pless α=5 (prevention) | 252 | 0.483 | 0.714 | 0.5525 | 9,424 | 0.3 |
| temp t1.0 (p0.95) | 252 | 0.480 | 0.726 | 0.5563 | 9,683 | 0.0 |
| temp t0.6 (p0.95) [rec] | 252 | 0.475 | 0.714 | 0.5341 | 9,866 | 6.2 |
| pless α=4 | 252 | 0.473 | 0.710 | 0.5308 | 9,220 | 1.4 |
| temp t0.6 (unfilt) | 252 | 0.467 | 0.702 | 0.5433 | 9,640 | 1.8 |
| temp t0.6 (p0.95+k20) | 252 | 0.464 | 0.683 | 0.5458 | 10,096 | 6.5 |
| temp t1.0 (k20) | 252 | 0.459 | 0.698 | 0.5813 | 9,791 | 0.0 |
| pless α=3 | 252 | 0.457 | 0.710 | 0.5380 | 10,504 | 10.6 |
| adaptive (1-chop) | 252 | 0.457 | 0.687 | 0.5099 | 10,428 | 7.1 |
| adaptive (3-chop) | 252 | 0.452 | 0.687 | 0.5043 | 9,952 | 5.4 |
| pless α=2 | 252 | 0.392 | 0.627 | 0.4885 | 17,269 | 41.8 |
| pless_norm | 252 | 0.392 | 0.663 | 0.4934 | 17,116 | 41.7 |

## Cross-verification

✓ pass@k (recomputed vs stored) and trunc% agree to tolerance.

## Caveats
- **cb_div** correct-only over each config's own ≥2-correct subset (mild cross-config confound).
- **mean think tok** counts truncated samples at their cut length (≈cap) → biased UP for high-truncation configs (the rambling cost). Re-tokenized estimate when no CSV (±few %).
- **structural_diversity** omitted (zss intractable on APPS-CoT — the reason for `--skip-diversity`).
