# MBPP: P-Less vs arXiv 2507.03160 Baseline (Qwen2.5-Coder-3B, 1.5B & OCI-DS-1.3B)

Comparison of p-less sampling against the top_p=0.95/temp=0.2 baseline from "Assessing Small Language Models for Code Generation" (arXiv:2507.03160). BigCode zero-shot docstring format, MBPP full (500 tasks), n=10, unbiased pass@k estimator. Models: Qwen2.5-Coder-3B, Qwen2.5-Coder-1.5B, OpenCodeInterpreter-DS-1.3B.

## pass@1 Comparison

### Qwen2.5-Coder-3B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=0.6) **←** | Ours | 59.4 |
| 2 | P-Less (t=0.6) **←** | Ours | 59.3 |
| 3 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 58.2 |
| 4 | P-Less Norm (t=1.0) **←** | Ours | 57.6 |
| 5 | Top-p (paper, t=0.2) | Paper | 57.0 |
| 6 | P-Less (t=1.0) **←** | Ours | 56.5 |
| 7 | Temperature (t=0.7) **←** | Ours | 42.6 |
| 8 | Top-p p=0.9 (t=1.0) **←** | Ours | 34.7 |

### Qwen2.5-Coder-1.5B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less (t=0.6) **←** | Ours | 53.1 |
| 2 | P-Less Norm (t=0.6) **←** | Ours | 52.8 |
| 3 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 52.5 |
| 4 | P-Less Norm (t=1.0) **←** | Ours | 51.9 |
| 5 | P-Less (t=1.0) **←** | Ours | 51.3 |
| 6 | Top-p (paper, t=0.2) | Paper | 51.0 |
| 7 | Temperature (t=0.7) **←** | Ours | 38.1 |
| 8 | Top-p p=0.9 (t=1.0) **←** | Ours | 32.7 |

### OCI-DS-1.3B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | Top-p (paper, t=0.2) | Paper | 44.0 |
| 2 | P-Less Norm (t=1.0) **←** | Ours | 26.8 |
| 3 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 26.5 |
| 4 | P-Less (t=0.6) **←** | Ours | 26.5 |
| 5 | P-Less Norm (t=0.6) **←** | Ours | 26.4 |
| 6 | P-Less (t=1.0) **←** | Ours | 26.4 |
| 7 | Temperature (t=0.7) **←** | Ours | 24.5 |
| 8 | Top-p p=0.9 (t=1.0) **←** | Ours | 21.3 |


## Extended Metrics (Our Methods Only)

### Qwen2.5-Coder-3B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less (t=0.6) | 59.3 | 63.4 | 64.8 | 66.2 | 66.2 | 65.4 | 63.4 | 11.6 | 60.0 | 1.8 | 57.2 | 0.0 |
| P-Less (t=1.0) | 56.5 | 66.4 | 69.4 | 72.2 | 72.2 | 72.2 | 65.8 | 36.6 | 59.4 | 12.2 | 51.6 | 3.8 |
| P-Less Norm (t=0.6) | 59.4 | 63.7 | 65.2 | 66.6 | 66.6 | 66.2 | 63.4 | 10.8 | 60.0 | 2.2 | 57.2 | 0.2 |
| P-Less Norm (t=1.0) | 57.6 | 66.9 | 69.8 | 72.4 | 72.4 | 72.2 | 66.8 | 37.4 | 59.4 | 13.2 | 53.4 | 3.0 |
| Temperature (t=0.7) | 42.6 | 63.9 | 70.6 | 77.6 | 77.6 | 77.4 | 61.6 | 59.2 | 48.2 | 40.2 | 34.2 | 17.0 |
| Top-p p=0.95 (paper replication) (t=0.2) | 58.2 | 66.6 | 69.2 | 71.6 | 71.6 | 71.6 | 65.6 | 30.8 | 61.8 | 9.6 | 54.4 | 2.2 |
| Top-p p=0.9 (t=1.0) | 34.7 | 58.4 | 67.3 | 77.0 | 77.0 | 76.8 | 53.6 | 52.4 | 37.4 | 33.2 | 20.6 | 13.8 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Qwen2.5-Coder-1.5B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less (t=0.6) | 53.1 | 57.7 | 59.2 | 60.8 | 60.8 | 60.8 | 57.4 | 10.4 | 53.8 | 1.4 | 51.0 | 0.0 |
| P-Less (t=1.0) | 51.3 | 61.2 | 64.7 | 68.8 | 68.8 | 68.6 | 60.2 | 35.4 | 52.6 | 15.8 | 47.2 | 4.0 |
| P-Less Norm (t=0.6) | 52.8 | 58.0 | 59.8 | 61.4 | 61.4 | 61.4 | 57.8 | 11.4 | 53.4 | 1.4 | 49.8 | 0.0 |
| P-Less Norm (t=1.0) | 51.9 | 61.4 | 64.5 | 67.8 | 67.8 | 67.8 | 60.2 | 34.2 | 54.4 | 15.4 | 48.4 | 4.2 |
| Temperature (t=0.7) | 38.1 | 58.0 | 64.7 | 71.0 | 71.0 | 71.0 | 56.0 | 53.8 | 41.4 | 36.0 | 29.0 | 16.2 |
| Top-p p=0.95 (paper replication) (t=0.2) | 52.5 | 61.2 | 64.4 | 68.6 | 68.6 | 68.4 | 59.4 | 31.4 | 54.8 | 10.8 | 48.8 | 2.6 |
| Top-p p=0.9 (t=1.0) | 32.7 | 53.4 | 61.4 | 70.2 | 70.2 | 70.2 | 49.2 | 48.6 | 34.2 | 32.0 | 20.4 | 15.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### OCI-DS-1.3B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less (t=0.6) | 26.5 | 30.3 | 31.5 | 32.0 | 32.0 | 29.6 | 31.2 | 17.8 | 26.6 | 9.4 | 23.8 | 4.0 |
| P-Less (t=1.0) | 26.4 | 31.2 | 32.7 | 33.8 | 33.8 | 31.4 | 31.2 | 23.8 | 27.4 | 13.8 | 23.0 | 5.2 |
| P-Less Norm (t=0.6) | 26.4 | 29.9 | 31.0 | 31.8 | 31.8 | 28.2 | 29.8 | 16.6 | 27.0 | 9.2 | 24.0 | 4.6 |
| P-Less Norm (t=1.0) | 26.8 | 31.4 | 32.9 | 34.2 | 34.2 | 31.2 | 30.6 | 23.2 | 28.0 | 14.2 | 24.4 | 6.0 |
| Temperature (t=0.7) | 24.5 | 36.5 | 41.6 | 48.2 | 48.2 | 43.2 | 32.8 | 23.6 | 24.6 | 14.2 | 18.6 | 6.2 |
| Top-p p=0.95 (paper replication) (t=0.2) | 26.5 | 31.3 | 33.4 | 36.0 | 36.0 | 32.2 | 30.0 | 23.2 | 26.6 | 13.0 | 23.8 | 5.0 |
| Top-p p=0.9 (t=1.0) | 21.3 | 34.7 | 40.5 | 47.8 | 47.8 | 41.4 | 31.6 | 23.2 | 21.8 | 11.4 | 13.6 | 4.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Qwen2.5-Coder-3B

- **P-Less Norm (t=0.6)**: rank 1/8
- **P-Less (t=0.6)**: rank 2/8
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 3/8
- **P-Less Norm (t=1.0)**: rank 4/8
- **P-Less (t=1.0)**: rank 6/8
- **Temperature (t=0.7)**: rank 7/8
- **Top-p p=0.9 (t=1.0)**: rank 8/8

### Qwen2.5-Coder-1.5B

- **P-Less (t=0.6)**: rank 1/8
- **P-Less Norm (t=0.6)**: rank 2/8
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 3/8
- **P-Less Norm (t=1.0)**: rank 4/8
- **P-Less (t=1.0)**: rank 5/8
- **Temperature (t=0.7)**: rank 7/8
- **Top-p p=0.9 (t=1.0)**: rank 8/8

### OCI-DS-1.3B

- **P-Less Norm (t=1.0)**: rank 2/8
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 3/8
- **P-Less (t=0.6)**: rank 4/8
- **P-Less Norm (t=0.6)**: rank 5/8
- **P-Less (t=1.0)**: rank 6/8
- **Temperature (t=0.7)**: rank 7/8
- **Top-p p=0.9 (t=1.0)**: rank 8/8

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.

### OCI-DS-1.3B: Results Pending Re-run

**The OCI-DS-1.3B numbers above (26.x%) used a broken tokenizer and do not reflect true performance.**

Root cause: transformers 5.x loads `LlamaTokenizer` (declared in `tokenizer_config.json`) which overrides the `ByteLevel` decoder from `tokenizer.json`. This destroys whitespace in decoded text, causing BPE artifacts (`Ċ"""` instead of `\n"""`) that prevent stop sequences from firing. The model generated very long repetitive samples (mean ~803 chars vs ~214 for Qwen models), depressing pass@1.

The paper authors (arXiv 2507.03160) used transformers 4.x which did not have this regression, giving them 44% pass@1 with the same BigCode format and `AutoTokenizer`.

**Fix applied:** `bench/generator.py:load_model_and_tokenizer()` now auto-detects broken whitespace round-trip and reloads as `PreTrainedTokenizerFast`, which respects the `ByteLevel` decoder from `tokenizer.json`. **Action required:** re-run OCI-DS-1.3B on H100 using `run_bigcode_mbpp_oci13b_rerun.sh`. Expected pass@1 after fix: ~44% for the paper-replication config.
