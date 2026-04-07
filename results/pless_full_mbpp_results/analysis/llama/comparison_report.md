# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Llama-2-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 1.

## pass@1 Comparison

### Llama-2-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) **←** | Ours | 25.2 |
| 2 | beam8 (t=1.0) **←** | Ours | 24.6 |
| 3 | greedy (t=1.0) **←** | Ours | 23.0 |
| 4 | P-Less (t=0.7) **←** | Ours | 23.0 |
| 5 | P-Less Norm (t=0.6) **←** | Ours | 22.6 |
| 6 | P-Less (t=0.6) **←** | Ours | 22.5 |
| 7 | P-Less Norm (t=0.7) **←** | Ours | 22.5 |
| 8 | P-Less (t=1.0) **←** | Ours | 22.4 |
| 9 | P-Less Norm (t=0.6) **←** | Ours | 22.2 |
| 10 | P-Less (t=0.6) **←** | Ours | 21.9 |
| 11 | P-Less Norm (t=1.0) **←** | Ours | 21.8 |
| 12 | FSD-d | Paper | 21.2 |
| 13 | Beam Search | Paper | 19.4 |
| 14 | FSD | Paper | 19.2 |
| 15 | Diverse Beam Search | Paper | 18.4 |
| 16 | DoLa | Paper | 18.4 |
| 17 | Contrastive Decoding | Paper | 18.2 |
| 18 | Greedy | Paper | 17.8 |
| 19 | Contrastive Search | Paper | 17.4 |
| 20 | Temperature | Paper | 17.2 |
| 21 | Temperature (t=0.7) **←** | Ours | 17.1 |
| 22 | Top-p | Paper | 14.8 |
| 23 | Top-p (ours) (t=1.0) **←** | Ours | 14.0 |
| 24 | P-Less (t=0.6) **←** | Ours | 13.8 |
| 25 | Typical | Paper | 12.0 |
| 26 | Top-k | Paper | 10.2 |
| 27 | η-Sampling | Paper | 9.4 |
| 28 | Mirostat | Paper | 7.8 |
| 29 | Temperature (t=0.7) **←** | Ours | 4.0 |

### Llama-2-7B-Chat

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) **←** | Ours | 22.4 |
| 2 | beam8 (t=1.0) **←** | Ours | 22.4 |
| 3 | Beam Search | Paper | 21.6 |
| 4 | Diverse Beam Search | Paper | 21.2 |
| 5 | greedy (t=1.0) **←** | Ours | 20.6 |
| 6 | P-Less (t=0.7) **←** | Ours | 20.6 |
| 7 | P-Less Norm (t=0.7) **←** | Ours | 20.5 |
| 8 | P-Less (t=0.6) **←** | Ours | 20.5 |
| 9 | P-Less Norm (t=0.6) **←** | Ours | 20.4 |
| 10 | P-Less Norm (t=1.0) **←** | Ours | 20.2 |
| 11 | P-Less (t=1.0) **←** | Ours | 20.1 |
| 12 | Temperature | Paper | 20.0 |
| 13 | Top-p (ours) (t=1.0) **←** | Ours | 18.0 |
| 14 | DoLa | Paper | 18.0 |
| 15 | Typical | Paper | 18.0 |
| 16 | FSD | Paper | 17.8 |
| 17 | FSD-d | Paper | 17.8 |
| 18 | Temperature (t=0.7) **←** | Ours | 17.8 |
| 19 | Top-p | Paper | 17.6 |
| 20 | Contrastive Search | Paper | 17.4 |
| 21 | Contrastive Decoding | Paper | 17.4 |
| 22 | Greedy | Paper | 17.2 |
| 23 | η-Sampling | Paper | 17.0 |
| 24 | Top-k | Paper | 16.0 |
| 25 | Mirostat | Paper | 16.0 |


## Extended Metrics (Our Methods Only)

### Llama-2-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 25.2 | 25.2 | 25.2 | 25.2 | 25.2 | 25.0 | 25.2 | 25.0 | 25.2 | 25.0 | 25.2 | 25.0 |
| beam8 (t=1.0) | 24.6 | 24.6 | 24.6 | 24.6 | 24.6 | 24.2 | 24.6 | 24.2 | 24.6 | 24.2 | 24.6 | 24.2 |
| greedy (t=1.0) | 23.0 | 23.0 | 23.0 | 23.0 | 23.0 | 22.8 | 23.0 | 22.8 | 23.0 | 22.8 | 23.0 | 22.8 |
| P-Less (t=0.6) | 21.9 | 25.1 | 26.2 | 27.4 | 27.4 | 27.4 | 25.0 | 2.6 | 22.2 | 0.4 | 20.0 | 0.0 |
| P-Less Norm (t=0.6) | 22.2 | 25.2 | 26.3 | 27.2 | 27.2 | 27.2 | 25.0 | 3.2 | 22.6 | 0.4 | 20.4 | 0.0 |
| P-Less Norm (t=0.6) | 22.6 | 26.6 | 28.1 | 30.0 | 30.0 | 29.6 | 25.0 | 3.0 | 23.8 | 0.2 | 20.0 | 0.0 |
| P-Less Norm (t=0.7) | 22.5 | 27.1 | 28.8 | 30.4 | 30.4 | 30.2 | 27.0 | 3.2 | 22.8 | 0.4 | 19.4 | 0.0 |
| P-Less Norm (t=1.0) | 21.8 | 29.8 | 33.1 | 37.2 | 37.2 | 37.0 | 27.8 | 7.0 | 23.0 | 1.2 | 18.0 | 0.2 |
| P-Less (t=0.6) | 13.8 | 20.6 | 24.0 | 27.8 | 27.8 | 25.6 | 18.6 | 3.0 | 12.8 | 0.4 | 8.8 | 0.0 |
| P-Less (t=0.6) | 22.5 | 26.4 | 27.8 | 29.2 | 29.2 | 29.0 | 26.2 | 2.6 | 23.2 | 0.2 | 20.6 | 0.0 |
| P-Less (t=0.7) | 23.0 | 27.8 | 29.4 | 31.2 | 31.2 | 30.8 | 26.8 | 3.4 | 23.8 | 0.4 | 21.0 | 0.0 |
| P-Less (t=1.0) | 22.4 | 30.5 | 33.6 | 37.0 | 37.0 | 36.8 | 29.4 | 9.2 | 22.6 | 1.0 | 17.2 | 0.0 |
| Temperature (t=0.7) | 4.0 | 10.6 | 15.6 | 24.2 | 24.2 | 17.2 | 4.0 | 1.6 | 0.2 | 0.0 | 0.0 | 0.0 |
| Temperature (t=0.7) | 17.1 | 29.7 | 36.0 | 44.6 | 44.6 | 44.6 | 24.8 | 21.2 | 15.8 | 8.2 | 9.6 | 1.4 |
| Top-p (ours) (t=1.0) | 14.0 | 25.9 | 32.0 | 39.4 | 39.4 | 39.4 | 21.0 | 17.6 | 11.4 | 5.8 | 6.4 | 1.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Llama-2-7B-Chat — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 |
| beam8 (t=1.0) | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 |
| greedy (t=1.0) | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 | 20.6 |
| P-Less Norm (t=0.6) | 20.4 | 21.1 | 21.3 | 21.4 | 21.4 | 21.4 | 21.0 | 0.0 | 20.2 | 0.0 | 20.0 | 0.0 |
| P-Less Norm (t=0.7) | 20.5 | 21.3 | 21.5 | 21.6 | 21.6 | 21.6 | 21.4 | 0.0 | 20.6 | 0.0 | 19.6 | 0.0 |
| P-Less Norm (t=1.0) | 20.2 | 21.5 | 22.0 | 22.2 | 22.2 | 22.2 | 21.4 | 0.0 | 20.0 | 0.0 | 19.4 | 0.0 |
| P-Less (t=0.6) | 20.5 | 21.2 | 21.3 | 21.4 | 21.4 | 21.4 | 21.4 | 0.0 | 20.6 | 0.0 | 20.2 | 0.0 |
| P-Less (t=0.7) | 20.6 | 21.3 | 21.5 | 21.6 | 21.6 | 21.6 | 21.4 | 0.0 | 20.6 | 0.0 | 20.2 | 0.0 |
| P-Less (t=1.0) | 20.1 | 21.4 | 21.9 | 22.4 | 22.4 | 22.4 | 21.6 | 0.0 | 20.2 | 0.0 | 19.2 | 0.0 |
| Temperature (t=0.7) | 17.8 | 24.3 | 27.1 | 30.2 | 30.2 | 30.2 | 22.8 | 7.2 | 17.4 | 0.4 | 15.0 | 0.0 |
| Top-p (ours) (t=1.0) | 18.0 | 25.5 | 29.0 | 33.6 | 33.6 | 33.6 | 23.0 | 9.4 | 17.8 | 1.4 | 14.0 | 0.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Llama-2-7B (base)

- **beam4 (t=1.0)**: rank 1/29
- **beam8 (t=1.0)**: rank 2/29
- **greedy (t=1.0)**: rank 3/29
- **P-Less (t=0.7)**: rank 4/29
- **P-Less Norm (t=0.6)**: rank 5/29
- **P-Less (t=0.6)**: rank 6/29
- **P-Less Norm (t=0.7)**: rank 7/29
- **P-Less (t=1.0)**: rank 8/29
- **P-Less Norm (t=0.6)**: rank 5/29
- **P-Less (t=0.6)**: rank 6/29
- **P-Less Norm (t=1.0)**: rank 11/29
- **Temperature (t=0.7)**: rank 21/29
- **Top-p (ours) (t=1.0)**: rank 23/29
- **P-Less (t=0.6)**: rank 6/29
- **Temperature (t=0.7)**: rank 21/29
- Best P-Less vs paper's Temperature sampling: 5.8pp above (23.0% vs 17.2%)
- Our temp_0.7 vs paper's Temperature: 17.1% vs 17.2% (Δ=-0.1pp — sanity check for setup alignment)

### Llama-2-7B-Chat

- **beam4 (t=1.0)**: rank 1/25
- **beam8 (t=1.0)**: rank 2/25
- **greedy (t=1.0)**: rank 5/25
- **P-Less (t=0.7)**: rank 6/25
- **P-Less Norm (t=0.7)**: rank 7/25
- **P-Less (t=0.6)**: rank 8/25
- **P-Less Norm (t=0.6)**: rank 9/25
- **P-Less Norm (t=1.0)**: rank 10/25
- **P-Less (t=1.0)**: rank 11/25
- **Top-p (ours) (t=1.0)**: rank 13/25
- **Temperature (t=0.7)**: rank 18/25
- Best P-Less vs paper's Temperature sampling: 0.6pp above (20.6% vs 20.0%)
- Our temp_0.7 vs paper's Temperature: 17.8% vs 20.0% (Δ=-2.2pp — sanity check for setup alignment)

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
