# MBPP: P-Less vs Paper Decoding Methods (Qwen-7B)

Comparison of p-less sampling against 12 decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 26.

## pass@1 Comparison

### Qwen-7B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=1.0) **←** | Ours | 35.7 |
| 2 | P-Less Norm (t=0.6) **←** | Ours | 35.5 |
| 3 | P-Less (t=0.6) **←** | Ours | 35.4 |
| 4 | P-Less (t=1.0) **←** | Ours | 35.0 |
| 5 | Beam Search | Paper | 34.4 |
| 6 | Temperature | Paper | 33.8 |
| 7 | FSD-d | Paper | 33.6 |
| 8 | Diverse Beam Search | Paper | 33.2 |
| 9 | Greedy | Paper | 33.0 |
| 10 | FSD | Paper | 33.0 |
| 11 | Temperature (t=0.7) **←** | Ours | 29.8 |
| 12 | Contrastive Search | Paper | 28.4 |
| 13 | Top-p (ours) (t=1.0) **←** | Ours | 27.5 |
| 14 | Top-p | Paper | 27.4 |
| 15 | Typical | Paper | 27.0 |
| 16 | η-Sampling | Paper | 25.8 |
| 17 | Top-k | Paper | 19.8 |
| 18 | Mirostat | Paper | 18.4 |


## Extended Metrics (Our Methods Only)

### Qwen-7B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 35.5 | 38.9 | 39.8 | 40.4 | 40.4 | 40.4 | 39.4 | 3.0 | 35.8 | 0.0 | 33.6 | 0.0 |
| P-Less Norm (t=1.0) | 35.7 | 43.8 | 47.0 | 50.2 | 50.2 | 50.2 | 42.6 | 15.0 | 35.8 | 3.0 | 30.6 | 0.6 |
| P-Less (t=0.6) | 35.4 | 38.7 | 39.6 | 40.4 | 40.4 | 40.4 | 38.6 | 2.8 | 36.8 | 0.2 | 32.8 | 0.0 |
| P-Less (t=1.0) | 35.0 | 43.0 | 45.8 | 48.6 | 48.6 | 48.4 | 41.8 | 13.6 | 37.4 | 3.0 | 29.6 | 0.4 |
| Temperature (t=0.7) | 29.8 | 44.7 | 50.9 | 58.1 | 58.1 | 58.1 | 41.7 | 35.7 | 29.9 | 17.6 | 21.4 | 3.8 |
| Top-p (ours) (t=1.0) | 27.5 | 42.3 | 48.6 | 55.9 | 55.9 | 55.9 | 38.3 | 33.9 | 27.9 | 15.8 | 19.0 | 4.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Qwen-7B

- **P-Less Norm (t=1.0)**: rank 1/18
- **P-Less Norm (t=0.6)**: rank 2/18
- **P-Less (t=0.6)**: rank 3/18
- **P-Less (t=1.0)**: rank 4/18
- **Temperature (t=0.7)**: rank 11/18
- **Top-p (ours) (t=1.0)**: rank 13/18
- Best P-Less vs paper's Temperature sampling: 1.6pp above (35.4% vs 33.8%)
- Our temp_0.7 vs paper's Temperature: 29.8% vs 33.8% (Δ=-4.0pp — sanity check for setup alignment)

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
