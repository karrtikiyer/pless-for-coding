# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (CodeLlama-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 26.

## pass@1 Comparison

### CodeLlama-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=0.6) **←** | Ours | 41.7 |
| 2 | P-Less (t=0.6) **←** | Ours | 41.7 |
| 3 | P-Less Norm (t=1.0) **←** | Ours | 41.5 |
| 4 | P-Less (t=1.0) **←** | Ours | 41.4 |
| 5 | FSD-d | Paper | 39.6 |
| 6 | FSD | Paper | 37.0 |
| 7 | Temperature (t=0.7) **←** | Ours | 36.8 |
| 8 | Contrastive Search | Paper | 36.0 |
| 9 | Greedy | Paper | 35.4 |
| 10 | Diverse Beam Search | Paper | 35.0 |
| 11 | Temperature | Paper | 35.0 |
| 12 | Top-p (ours) (t=1.0) **←** | Ours | 34.6 |
| 13 | Beam Search | Paper | 34.2 |
| 14 | Top-p | Paper | 32.8 |
| 15 | Typical | Paper | 31.8 |
| 16 | Top-k | Paper | 25.4 |
| 17 | η-Sampling | Paper | 23.6 |
| 18 | Mirostat | Paper | 21.2 |

### CodeLlama-7B-Instruct

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less (t=1.0) **←** | Ours | 41.6 |
| 2 | Diverse Beam Search | Paper | 41.6 |
| 3 | P-Less Norm (t=1.0) **←** | Ours | 41.4 |
| 4 | P-Less (t=0.6) **←** | Ours | 41.2 |
| 5 | P-Less Norm (t=0.6) **←** | Ours | 41.1 |
| 6 | Beam Search | Paper | 40.8 |
| 7 | Temperature | Paper | 39.0 |
| 8 | Temperature (t=0.7) **←** | Ours | 38.3 |
| 9 | Top-p (ours) (t=1.0) **←** | Ours | 38.3 |
| 10 | Typical | Paper | 38.2 |
| 11 | Top-p | Paper | 37.6 |
| 12 | FSD | Paper | 37.2 |
| 13 | Contrastive Search | Paper | 37.0 |
| 14 | Greedy | Paper | 36.8 |
| 15 | FSD-d | Paper | 36.6 |
| 16 | Top-k | Paper | 35.6 |
| 17 | η-Sampling | Paper | 35.4 |
| 18 | Mirostat | Paper | 34.4 |


## Extended Metrics (Our Methods Only)

### CodeLlama-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 41.7 | 46.3 | 47.9 | 49.0 | 49.0 | 49.0 | 45.8 | 4.8 | 41.8 | 0.4 | 38.8 | 0.0 |
| P-Less Norm (t=1.0) | 41.5 | 50.4 | 53.8 | 57.4 | 57.4 | 57.4 | 49.0 | 21.4 | 42.0 | 6.0 | 37.6 | 0.8 |
| P-Less (t=0.6) | 41.7 | 46.5 | 48.2 | 49.4 | 49.4 | 49.4 | 46.4 | 4.6 | 41.8 | 0.4 | 39.2 | 0.0 |
| P-Less (t=1.0) | 41.4 | 50.3 | 53.6 | 57.2 | 57.2 | 57.2 | 48.8 | 22.4 | 42.2 | 4.8 | 36.8 | 1.6 |
| Temperature (t=0.7) | 36.8 | 52.9 | 58.9 | 65.2 | 65.2 | 65.2 | 48.8 | 45.6 | 39.4 | 26.2 | 29.0 | 8.4 |
| Top-p (ours) (t=1.0) | 34.6 | 52.1 | 59.5 | 68.2 | 68.2 | 68.2 | 47.6 | 45.0 | 35.0 | 28.8 | 25.6 | 13.8 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### CodeLlama-7B-Instruct — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 41.1 | 41.8 | 42.1 | 42.2 | 42.2 | 0.0 | 41.8 | 0.0 | 40.8 | 0.0 | 40.6 | 0.0 |
| P-Less Norm (t=1.0) | 41.4 | 43.1 | 43.5 | 43.8 | 43.8 | 0.2 | 43.2 | 0.0 | 42.0 | 0.0 | 40.4 | 0.0 |
| P-Less (t=0.6) | 41.2 | 41.7 | 42.0 | 42.2 | 42.2 | 0.0 | 41.6 | 0.0 | 41.2 | 0.0 | 41.0 | 0.0 |
| P-Less (t=1.0) | 41.6 | 43.4 | 43.8 | 44.2 | 44.2 | 0.2 | 43.4 | 0.0 | 42.4 | 0.0 | 40.2 | 0.0 |
| Temperature (t=0.7) | 38.3 | 48.2 | 51.6 | 55.2 | 55.2 | 22.2 | 47.4 | 0.0 | 39.4 | 0.0 | 34.4 | 0.0 |
| Top-p (ours) (t=1.0) | 38.3 | 49.4 | 53.7 | 59.0 | 59.0 | 32.0 | 46.0 | 1.0 | 39.8 | 0.0 | 33.8 | 0.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### CodeLlama-7B (base)

- **P-Less Norm (t=0.6)**: rank 1/18
- **P-Less (t=0.6)**: rank 2/18
- **P-Less Norm (t=1.0)**: rank 3/18
- **P-Less (t=1.0)**: rank 4/18
- **Temperature (t=0.7)**: rank 7/18
- **Top-p (ours) (t=1.0)**: rank 12/18
- Best P-Less vs paper's Temperature sampling: 6.7pp above (41.7% vs 35.0%)
- Our temp_0.7 vs paper's Temperature: 36.8% vs 35.0% (Δ=+1.8pp — sanity check for setup alignment)

### CodeLlama-7B-Instruct

- **P-Less (t=1.0)**: rank 1/18
- **P-Less Norm (t=1.0)**: rank 3/18
- **P-Less (t=0.6)**: rank 4/18
- **P-Less Norm (t=0.6)**: rank 5/18
- **Temperature (t=0.7)**: rank 8/18
- **Top-p (ours) (t=1.0)**: rank 9/18
- Best P-Less vs paper's Temperature sampling: 2.6pp above (41.6% vs 39.0%)
- Our temp_0.7 vs paper's Temperature: 38.3% vs 39.0% (Δ=-0.7pp — sanity check for setup alignment)

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
