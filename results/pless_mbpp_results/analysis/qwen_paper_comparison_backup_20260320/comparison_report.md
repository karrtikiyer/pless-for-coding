# MBPP: P-Less vs Paper Decoding Methods (Qwen-7B)

Comparison of p-less sampling against 12 decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 26.

## pass@1 Comparison

### Qwen-7B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | Beam Search | Paper | 34.4 |
| 2 | FSD-d | Paper | 33.8 |
| 3 | P-Less Norm (t=1.0) **←** | Ours | 33.3 |
| 4 | Diverse Beam Search | Paper | 33.2 |
| 5 | Greedy | Paper | 33.0 |
| 6 | FSD | Paper | 33.0 |
| 7 | Temperature | Paper | 33.0 |
| 8 | P-Less (t=1.0) **←** | Ours | 32.6 |
| 9 | Contrastive Search | Paper | 28.4 |
| 10 | Top-p | Paper | 27.4 |
| 11 | Typical | Paper | 27.0 |
| 12 | Temperature (t=0.7) **←** | Ours | 26.2 |
| 13 | η-Sampling | Paper | 25.8 |
| 14 | Top-k | Paper | 19.8 |
| 15 | Mirostat | Paper | 18.4 |


## Extended Metrics (Our Methods Only)

### Qwen-7B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=1.0) | 33.3 | 50.7 | 55.7 | 60.3 | 60.3 | 28.8 | 49.4 | 2.7 | 40.1 | 0.0 | 24.5 | 0.0 |
| P-Less (t=1.0) | 32.6 | 49.1 | 54.3 | 59.1 | 59.1 | 28.0 | 47.5 | 1.9 | 36.6 | 0.0 | 26.1 | 0.0 |
| Temperature (t=0.7) | 26.2 | 46.4 | 54.7 | 63.0 | 63.0 | 59.5 | 41.6 | 33.9 | 28.8 | 10.9 | 11.3 | 1.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Qwen-7B

- **P-Less (t=1.0)**: rank 8/15
- **P-Less Norm (t=1.0)**: rank 3/15
- **Temperature (t=0.7)**: rank 12/15
- P-Less vs paper's Temperature sampling: 0.4pp below (32.6% vs 33.0%)
- Our temp_0.7 vs paper's Temperature: 26.2% vs 33.0% (Δ=-6.8pp — sanity check for setup alignment)

### Limitations

- We ran only 3 methods (pless, pless_norm, temp_0.7) vs the paper's 14. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
