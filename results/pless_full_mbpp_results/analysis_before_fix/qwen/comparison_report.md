# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Qwen-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 26.

## pass@1 Comparison

### Qwen-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | Beam Search | Paper | 34.4 |
| 2 | Temperature | Paper | 33.8 |
| 3 | FSD-d | Paper | 33.6 |
| 4 | Diverse Beam Search | Paper | 33.2 |
| 5 | Greedy | Paper | 33.0 |
| 6 | FSD | Paper | 33.0 |
| 7 | P-Less Norm (t=0.6) **←** | Ours | 31.4 |
| 8 | P-Less (t=0.6) **←** | Ours | 31.4 |
| 9 | P-Less Norm (t=1.0) **←** | Ours | 30.3 |
| 10 | P-Less (t=1.0) **←** | Ours | 30.1 |
| 11 | Contrastive Search | Paper | 28.4 |
| 12 | Top-p | Paper | 27.4 |
| 13 | Typical | Paper | 27.0 |
| 14 | η-Sampling | Paper | 25.8 |
| 15 | Temperature (t=0.7) **←** | Ours | 22.1 |
| 16 | Top-k | Paper | 19.8 |
| 17 | Mirostat | Paper | 18.4 |

### Qwen-7B-Chat

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=1.0) **←** | Ours | 34.5 |
| 2 | P-Less Norm (t=0.6) **←** | Ours | 34.4 |
| 3 | P-Less (t=1.0) **←** | Ours | 34.4 |
| 4 | P-Less (t=0.6) **←** | Ours | 34.0 |
| 5 | Diverse Beam Search | Paper | 33.6 |
| 6 | Beam Search | Paper | 30.8 |
| 7 | FSD | Paper | 30.8 |
| 8 | Greedy | Paper | 30.4 |
| 9 | Temperature | Paper | 30.0 |
| 10 | FSD-d | Paper | 29.8 |
| 11 | Top-p | Paper | 28.8 |
| 12 | Temperature (t=0.7) **←** | Ours | 28.7 |
| 13 | Typical | Paper | 27.2 |
| 14 | Top-k | Paper | 26.8 |
| 15 | Contrastive Search | Paper | 25.8 |
| 16 | Mirostat | Paper | 25.0 |
| 17 | η-Sampling | Paper | 24.2 |


## Extended Metrics (Our Methods Only)

### Qwen-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 31.4 | 37.5 | 39.8 | 42.0 | 42.0 | 41.8 | 36.6 | 8.2 | 31.6 | 1.0 | 28.4 | 0.0 |
| P-Less Norm (t=1.0) | 30.3 | 42.0 | 47.0 | 53.0 | 53.0 | 52.6 | 37.8 | 25.6 | 30.6 | 8.0 | 24.6 | 1.8 |
| P-Less (t=0.6) | 31.4 | 37.8 | 39.9 | 41.8 | 41.8 | 41.8 | 37.4 | 7.8 | 33.0 | 0.6 | 26.8 | 0.2 |
| P-Less (t=1.0) | 30.1 | 41.9 | 46.4 | 51.4 | 51.4 | 51.4 | 39.4 | 25.4 | 32.4 | 8.8 | 23.6 | 0.8 |
| Temperature (t=0.7) | 22.1 | 37.1 | 44.0 | 53.4 | 53.4 | 51.6 | 30.6 | 27.2 | 22.6 | 12.8 | 14.6 | 4.4 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Qwen-7B-Chat — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 34.4 | 36.6 | 37.3 | 37.6 | 37.6 | 37.6 | 36.4 | 0.6 | 35.2 | 0.0 | 32.8 | 0.0 |
| P-Less Norm (t=1.0) | 34.5 | 38.6 | 39.9 | 40.8 | 40.8 | 40.8 | 38.6 | 3.2 | 35.2 | 0.4 | 31.8 | 0.0 |
| P-Less (t=0.6) | 34.0 | 35.7 | 36.4 | 37.0 | 37.0 | 37.0 | 35.4 | 0.4 | 34.6 | 0.0 | 32.8 | 0.0 |
| P-Less (t=1.0) | 34.4 | 38.3 | 39.6 | 40.6 | 40.6 | 40.6 | 38.4 | 2.2 | 34.6 | 0.0 | 32.6 | 0.0 |
| Temperature (t=0.7) | 28.7 | 40.0 | 44.7 | 50.4 | 50.4 | 50.4 | 36.8 | 15.8 | 29.4 | 5.6 | 22.8 | 1.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Qwen-7B (base)

- **P-Less Norm (t=0.6)**: rank 7/17
- **P-Less (t=0.6)**: rank 8/17
- **P-Less Norm (t=1.0)**: rank 9/17
- **P-Less (t=1.0)**: rank 10/17
- **Temperature (t=0.7)**: rank 15/17
- Best P-Less vs paper's Temperature sampling: 2.4pp below (31.4% vs 33.8%)
- Our temp_0.7 vs paper's Temperature: 22.1% vs 33.8% (Δ=-11.7pp — sanity check for setup alignment)

### Qwen-7B-Chat

- **P-Less Norm (t=1.0)**: rank 1/17
- **P-Less Norm (t=0.6)**: rank 2/17
- **P-Less (t=1.0)**: rank 3/17
- **P-Less (t=0.6)**: rank 4/17
- **Temperature (t=0.7)**: rank 12/17
- Best P-Less vs paper's Temperature sampling: 4.4pp above (34.4% vs 30.0%)
- Our temp_0.7 vs paper's Temperature: 28.7% vs 30.0% (Δ=-1.3pp — sanity check for setup alignment)

### Limitations

- We ran 5 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
