# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (CodeLlama-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 26.

## pass@1 Comparison

### CodeLlama-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=0.6) **←** | Ours | 41.8 |
| 2 | P-Less (t=0.6) **←** | Ours | 41.4 |
| 3 | P-Less Norm (t=1.0) **←** | Ours | 40.2 |
| 4 | P-Less (t=1.0) **←** | Ours | 40.1 |
| 5 | FSD-d | Paper | 39.6 |
| 6 | FSD | Paper | 37.0 |
| 7 | Contrastive Search | Paper | 36.0 |
| 8 | Greedy | Paper | 35.4 |
| 9 | Diverse Beam Search | Paper | 35.0 |
| 10 | Temperature | Paper | 35.0 |
| 11 | Beam Search | Paper | 34.2 |
| 12 | Top-p | Paper | 32.8 |
| 13 | Typical | Paper | 31.8 |
| 14 | Temperature (t=0.7) **←** | Ours | 29.8 |
| 15 | Top-k | Paper | 25.4 |
| 16 | η-Sampling | Paper | 23.6 |
| 17 | Mirostat | Paper | 21.2 |

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
| 9 | Typical | Paper | 38.2 |
| 10 | Top-p | Paper | 37.6 |
| 11 | FSD | Paper | 37.2 |
| 12 | Contrastive Search | Paper | 37.0 |
| 13 | Greedy | Paper | 36.8 |
| 14 | FSD-d | Paper | 36.6 |
| 15 | Top-k | Paper | 35.6 |
| 16 | η-Sampling | Paper | 35.4 |
| 17 | Mirostat | Paper | 34.4 |


## Extended Metrics (Our Methods Only)

### CodeLlama-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 41.8 | 49.4 | 52.0 | 54.4 | 54.4 | 53.4 | 48.4 | 27.6 | 43.2 | 10.8 | 37.4 | 5.2 |
| P-Less Norm (t=1.0) | 40.2 | 51.9 | 56.7 | 62.6 | 62.6 | 61.6 | 49.0 | 44.6 | 41.2 | 31.2 | 34.8 | 18.8 |
| P-Less (t=0.6) | 41.4 | 49.5 | 52.2 | 54.8 | 54.8 | 53.4 | 48.8 | 28.2 | 43.0 | 12.2 | 36.8 | 5.4 |
| P-Less (t=1.0) | 40.1 | 51.0 | 55.3 | 60.4 | 60.4 | 59.0 | 49.0 | 44.4 | 41.0 | 32.2 | 34.6 | 18.0 |
| Temperature (t=0.7) | 29.8 | 48.0 | 55.4 | 63.8 | 63.8 | 63.0 | 42.4 | 42.0 | 31.8 | 29.4 | 20.6 | 13.8 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### CodeLlama-7B-Instruct — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 41.1 | 41.8 | 42.1 | 42.2 | 42.2 | 0.0 | 41.8 | 0.0 | 40.8 | 0.0 | 40.6 | 0.0 |
| P-Less Norm (t=1.0) | 41.4 | 43.1 | 43.5 | 43.8 | 43.8 | 0.2 | 43.2 | 0.0 | 42.0 | 0.0 | 40.4 | 0.0 |
| P-Less (t=0.6) | 41.2 | 41.7 | 42.0 | 42.2 | 42.2 | 0.0 | 41.6 | 0.0 | 41.2 | 0.0 | 41.0 | 0.0 |
| P-Less (t=1.0) | 41.6 | 43.4 | 43.8 | 44.2 | 44.2 | 0.2 | 43.4 | 0.0 | 42.4 | 0.0 | 40.2 | 0.0 |
| Temperature (t=0.7) | 38.3 | 48.2 | 51.6 | 55.2 | 55.2 | 22.2 | 47.4 | 0.0 | 39.4 | 0.0 | 34.4 | 0.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### CodeLlama-7B (base)

- **P-Less Norm (t=0.6)**: rank 1/17
- **P-Less (t=0.6)**: rank 2/17
- **P-Less Norm (t=1.0)**: rank 3/17
- **P-Less (t=1.0)**: rank 4/17
- **Temperature (t=0.7)**: rank 14/17
- Best P-Less vs paper's Temperature sampling: 6.4pp above (41.4% vs 35.0%)
- Our temp_0.7 vs paper's Temperature: 29.8% vs 35.0% (Δ=-5.2pp — sanity check for setup alignment)

### CodeLlama-7B-Instruct

- **P-Less (t=1.0)**: rank 1/17
- **P-Less Norm (t=1.0)**: rank 3/17
- **P-Less (t=0.6)**: rank 4/17
- **P-Less Norm (t=0.6)**: rank 5/17
- **Temperature (t=0.7)**: rank 8/17
- Best P-Less vs paper's Temperature sampling: 2.6pp above (41.6% vs 39.0%)
- Our temp_0.7 vs paper's Temperature: 38.3% vs 39.0% (Δ=-0.7pp — sanity check for setup alignment)

### Limitations

- We ran 5 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
