# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Qwen-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 26.

## pass@1 Comparison

### Qwen-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) **←** | Ours | 39.8 |
| 2 | beam8 (t=1.0) **←** | Ours | 39.6 |
| 3 | greedy (t=1.0) **←** | Ours | 36.8 |
| 4 | P-Less (t=0.7) **←** | Ours | 36.2 |
| 5 | P-Less Norm (t=1.0) **←** | Ours | 35.7 |
| 6 | P-Less Norm (t=0.6) **←** | Ours | 35.5 |
| 7 | P-Less (t=0.6) **←** | Ours | 35.4 |
| 8 | P-Less Norm (t=0.7) **←** | Ours | 35.3 |
| 9 | P-Less (t=1.0) **←** | Ours | 35.0 |
| 10 | Beam Search | Paper | 34.4 |
| 11 | Temperature | Paper | 33.8 |
| 12 | FSD-d | Paper | 33.6 |
| 13 | Diverse Beam Search | Paper | 33.2 |
| 14 | Greedy | Paper | 33.0 |
| 15 | FSD | Paper | 33.0 |
| 16 | P-Less (t=0.6) **←** | Ours | 31.4 |
| 17 | P-Less Norm (t=0.6) **←** | Ours | 31.3 |
| 18 | Temperature (t=0.7) **←** | Ours | 29.8 |
| 19 | Contrastive Search | Paper | 28.4 |
| 20 | Top-p (ours) (t=1.0) **←** | Ours | 27.5 |
| 21 | Top-p | Paper | 27.4 |
| 22 | Typical | Paper | 27.0 |
| 23 | η-Sampling | Paper | 25.8 |
| 24 | Top-k | Paper | 19.8 |
| 25 | Mirostat | Paper | 18.4 |
| 26 | P-Less (t=0.6) **←** | Ours | 14.0 |
| 27 | P-Less (t=0.6) **←** | Ours | 12.8 |
| 28 | Temperature (t=0.7) **←** | Ours | 9.7 |

### Qwen-7B-Chat

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam8 (t=1.0) **←** | Ours | 36.2 |
| 2 | beam4 (t=1.0) **←** | Ours | 36.0 |
| 3 | P-Less Norm (t=1.0) **←** | Ours | 34.5 |
| 4 | P-Less Norm (t=0.6) **←** | Ours | 34.4 |
| 5 | P-Less (t=1.0) **←** | Ours | 34.4 |
| 6 | P-Less Norm (t=0.7) **←** | Ours | 34.1 |
| 7 | P-Less (t=0.6) **←** | Ours | 34.0 |
| 8 | P-Less (t=0.7) **←** | Ours | 33.9 |
| 9 | Diverse Beam Search | Paper | 33.6 |
| 10 | greedy (t=1.0) **←** | Ours | 31.4 |
| 11 | Beam Search | Paper | 30.8 |
| 12 | FSD | Paper | 30.8 |
| 13 | Greedy | Paper | 30.4 |
| 14 | Temperature | Paper | 30.0 |
| 15 | FSD-d | Paper | 29.8 |
| 16 | Top-p | Paper | 28.8 |
| 17 | Temperature (t=0.7) **←** | Ours | 28.7 |
| 18 | Typical | Paper | 27.2 |
| 19 | Top-p (ours) (t=1.0) **←** | Ours | 27.1 |
| 20 | Top-k | Paper | 26.8 |
| 21 | Contrastive Search | Paper | 25.8 |
| 22 | Mirostat | Paper | 25.0 |
| 23 | η-Sampling | Paper | 24.2 |


## Extended Metrics (Our Methods Only)

### Qwen-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 |
| beam8 (t=1.0) | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 | 39.6 |
| greedy (t=1.0) | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 | 36.8 |
| P-Less (t=0.6) | 12.8 | 23.4 | 28.2 | 32.6 | 32.6 | 32.6 | 20.9 | 2.3 | 11.6 | 0.0 | 4.7 | 0.0 |
| P-Less (t=0.6) | 31.4 | 36.0 | 37.3 | 38.2 | 38.2 | 38.2 | 36.2 | 2.6 | 32.6 | 0.0 | 27.8 | 0.0 |
| P-Less Norm (t=0.6) | 31.3 | 35.9 | 37.2 | 38.2 | 38.2 | 38.2 | 35.6 | 2.2 | 32.8 | 0.0 | 28.6 | 0.0 |
| P-Less Norm (t=0.6) | 35.5 | 38.9 | 39.8 | 40.4 | 40.4 | 40.4 | 39.4 | 3.0 | 35.8 | 0.0 | 33.6 | 0.0 |
| P-Less Norm (t=0.7) | 35.3 | 39.9 | 41.6 | 42.8 | 42.8 | 42.8 | 40.4 | 4.8 | 35.2 | 0.4 | 32.6 | 0.0 |
| P-Less Norm (t=1.0) | 35.7 | 43.8 | 47.0 | 50.2 | 50.2 | 50.2 | 42.6 | 15.0 | 35.8 | 3.0 | 30.6 | 0.6 |
| P-Less (t=0.6) | 14.0 | 21.4 | 24.7 | 28.2 | 28.2 | 24.6 | 19.6 | 4.2 | 14.0 | 0.6 | 9.0 | 0.0 |
| P-Less (t=0.6) | 35.4 | 38.7 | 39.6 | 40.4 | 40.4 | 40.4 | 38.6 | 2.8 | 36.8 | 0.2 | 32.8 | 0.0 |
| P-Less (t=0.7) | 36.2 | 40.4 | 41.7 | 42.8 | 42.8 | 42.8 | 40.4 | 4.8 | 37.0 | 0.8 | 34.4 | 0.2 |
| P-Less (t=1.0) | 35.0 | 43.0 | 45.8 | 48.6 | 48.6 | 48.4 | 41.8 | 13.6 | 37.4 | 3.0 | 29.6 | 0.4 |
| Temperature (t=0.7) | 9.7 | 22.4 | 30.1 | 39.8 | 39.8 | 36.4 | 16.0 | 6.2 | 3.6 | 0.6 | 0.6 | 0.2 |
| Temperature (t=0.7) | 29.8 | 44.7 | 50.9 | 58.1 | 58.1 | 58.1 | 41.7 | 35.7 | 29.9 | 17.6 | 21.4 | 3.8 |
| Top-p (ours) (t=1.0) | 27.5 | 42.3 | 48.6 | 55.9 | 55.9 | 55.9 | 38.3 | 33.9 | 27.9 | 15.8 | 19.0 | 4.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Qwen-7B-Chat — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 |
| beam8 (t=1.0) | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 | 36.2 |
| greedy (t=1.0) | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 | 31.4 |
| P-Less Norm (t=0.6) | 34.4 | 36.6 | 37.3 | 37.6 | 37.6 | 37.6 | 36.4 | 0.6 | 35.2 | 0.0 | 32.8 | 0.0 |
| P-Less Norm (t=0.7) | 34.1 | 36.6 | 37.4 | 38.0 | 38.0 | 38.0 | 36.4 | 1.0 | 34.6 | 0.2 | 32.2 | 0.0 |
| P-Less Norm (t=1.0) | 34.5 | 38.6 | 39.9 | 40.8 | 40.8 | 40.8 | 38.6 | 3.2 | 35.2 | 0.4 | 31.8 | 0.0 |
| P-Less (t=0.6) | 34.0 | 35.7 | 36.4 | 37.0 | 37.0 | 37.0 | 35.4 | 0.4 | 34.6 | 0.0 | 32.8 | 0.0 |
| P-Less (t=0.7) | 33.9 | 36.6 | 37.4 | 37.8 | 37.8 | 37.8 | 36.8 | 0.6 | 34.8 | 0.0 | 31.8 | 0.0 |
| P-Less (t=1.0) | 34.4 | 38.3 | 39.6 | 40.6 | 40.6 | 40.6 | 38.4 | 2.2 | 34.6 | 0.0 | 32.6 | 0.0 |
| Temperature (t=0.7) | 28.7 | 40.0 | 44.7 | 50.4 | 50.4 | 50.4 | 36.8 | 15.8 | 29.2 | 5.4 | 22.8 | 1.0 |
| Top-p (ours) (t=1.0) | 27.1 | 38.7 | 43.7 | 50.6 | 50.6 | 50.4 | 35.2 | 16.8 | 28.0 | 4.0 | 20.4 | 0.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Qwen-7B (base)

- **beam4 (t=1.0)**: rank 1/28
- **beam8 (t=1.0)**: rank 2/28
- **greedy (t=1.0)**: rank 3/28
- **P-Less (t=0.7)**: rank 4/28
- **P-Less Norm (t=1.0)**: rank 5/28
- **P-Less Norm (t=0.6)**: rank 6/28
- **P-Less (t=0.6)**: rank 7/28
- **P-Less Norm (t=0.7)**: rank 8/28
- **P-Less (t=1.0)**: rank 9/28
- **P-Less (t=0.6)**: rank 7/28
- **P-Less Norm (t=0.6)**: rank 6/28
- **Temperature (t=0.7)**: rank 18/28
- **Top-p (ours) (t=1.0)**: rank 20/28
- **P-Less (t=0.6)**: rank 7/28
- **P-Less (t=0.6)**: rank 7/28
- **Temperature (t=0.7)**: rank 18/28
- Best P-Less vs paper's Temperature sampling: 2.4pp above (36.2% vs 33.8%)
- Our temp_0.7 vs paper's Temperature: 29.8% vs 33.8% (Δ=-4.0pp — sanity check for setup alignment)

### Qwen-7B-Chat

- **beam8 (t=1.0)**: rank 1/23
- **beam4 (t=1.0)**: rank 2/23
- **P-Less Norm (t=1.0)**: rank 3/23
- **P-Less Norm (t=0.6)**: rank 4/23
- **P-Less (t=1.0)**: rank 5/23
- **P-Less Norm (t=0.7)**: rank 6/23
- **P-Less (t=0.6)**: rank 7/23
- **P-Less (t=0.7)**: rank 8/23
- **greedy (t=1.0)**: rank 10/23
- **Temperature (t=0.7)**: rank 17/23
- **Top-p (ours) (t=1.0)**: rank 19/23
- Best P-Less vs paper's Temperature sampling: 4.4pp above (34.4% vs 30.0%)
- Our temp_0.7 vs paper's Temperature: 28.7% vs 30.0% (Δ=-1.3pp — sanity check for setup alignment)

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
