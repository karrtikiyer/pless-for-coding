# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Llama-2-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 1.

## pass@1 Comparison

### Llama-2-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) | Repro | 25.2 |
| 2 | beam8 (t=1.0) | Repro | 24.6 |
| 3 | greedy (t=1.0) | Repro | 23.6 |
| 4 | greedy (t=1.0) | Repro | 23.0 |
| 5 | P-Less (t=0.7) **←** | Ours | 23.0 |
| 6 | P-Less Norm (t=0.6) **←** | Ours | 22.6 |
| 7 | P-Less (t=0.6) **←** | Ours | 22.5 |
| 8 | P-Less Norm (t=0.7) **←** | Ours | 22.5 |
| 9 | P-Less (t=1.0) **←** | Ours | 22.4 |
| 10 | P-Less Norm (t=0.6) **←** | Ours | 22.2 |
| 11 | P-Less (t=0.6) **←** | Ours | 21.9 |
| 12 | P-Less Norm (t=1.0) **←** | Ours | 21.8 |
| 13 | Temperature (t=0.3) | Repro | 21.7 |
| 14 | FSD-d | Paper | 21.2 |
| 15 | Beam Search | Paper | 19.4 |
| 16 | FSD | Paper | 19.2 |
| 17 | Diverse Beam Search | Paper | 18.4 |
| 18 | DoLa | Paper | 18.4 |
| 19 | Contrastive Decoding | Paper | 18.2 |
| 20 | Greedy | Paper | 17.8 |
| 21 | Top-p 0.8 (t=1.0) | Repro | 17.5 |
| 22 | Contrastive Search | Paper | 17.4 |
| 23 | Temperature | Paper | 17.2 |
| 24 | Temperature (t=0.7) **←** | Ours | 17.1 |
| 25 | Top-p | Paper | 14.8 |
| 26 | Top-p 0.9 (t=1.0) **←** | Ours | 14.0 |
| 27 | P-Less (t=0.6) **←** | Ours | 13.8 |
| 28 | Top-k 5 (t=1.0) | Repro | 12.5 |
| 29 | Typical | Paper | 12.0 |
| 30 | Top-k | Paper | 10.2 |
| 31 | η-Sampling | Paper | 9.4 |
| 32 | Mirostat | Paper | 7.8 |
| 33 | Temperature (t=0.7) **←** | Ours | 4.0 |

### Llama-2-7B-Chat

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) | Repro | 22.4 |
| 2 | beam8 (t=1.0) | Repro | 22.4 |
| 3 | Beam Search | Paper | 21.6 |
| 4 | Diverse Beam Search | Paper | 21.2 |
| 5 | greedy (t=1.0) | Repro | 20.6 |
| 6 | P-Less (t=0.7) **←** | Ours | 20.6 |
| 7 | P-Less Norm (t=0.7) **←** | Ours | 20.5 |
| 8 | P-Less (t=0.6) **←** | Ours | 20.5 |
| 9 | P-Less Norm (t=0.6) **←** | Ours | 20.4 |
| 10 | P-Less Norm (t=1.0) **←** | Ours | 20.2 |
| 11 | P-Less (t=1.0) **←** | Ours | 20.1 |
| 12 | Temperature | Paper | 20.0 |
| 13 | Temperature (t=0.3) | Repro | 19.5 |
| 14 | Top-p 0.8 (t=1.0) | Repro | 18.5 |
| 15 | Top-p 0.9 (t=1.0) **←** | Ours | 18.0 |
| 16 | DoLa | Paper | 18.0 |
| 17 | Typical | Paper | 18.0 |
| 18 | FSD | Paper | 17.8 |
| 19 | FSD-d | Paper | 17.8 |
| 20 | Temperature (t=0.7) **←** | Ours | 17.8 |
| 21 | Top-p | Paper | 17.6 |
| 22 | Contrastive Search | Paper | 17.4 |
| 23 | Contrastive Decoding | Paper | 17.4 |
| 24 | Top-k 5 (t=1.0) | Repro | 17.3 |
| 25 | Greedy | Paper | 17.2 |
| 26 | η-Sampling | Paper | 17.0 |
| 27 | Top-k | Paper | 16.0 |
| 28 | Mirostat | Paper | 16.0 |


## Extended Metrics (Our Methods Only)

### Llama-2-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 25.2 | 25.2 | 25.2 | 25.2 | 25.2 | 25.0 | 25.2 | 25.0 | 25.2 | 25.0 | 25.2 | 25.0 |
| beam8 (t=1.0) | 24.6 | 24.6 | 24.6 | 24.6 | 24.6 | 24.2 | 24.6 | 24.2 | 24.6 | 24.2 | 24.6 | 24.2 |
| greedy (t=1.0) | 23.6 | 23.6 | 23.6 | 23.6 | 23.6 | 23.4 | 23.6 | 23.4 | 23.6 | 23.4 | 23.6 | 23.4 |
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
| Temperature (t=0.3) | 21.7 | 30.2 | 33.8 | 38.4 | 38.4 | 38.4 | 28.2 | 15.2 | 21.6 | 3.8 | 17.6 | 0.4 |
| Temperature (t=0.7) | 17.1 | 29.7 | 36.0 | 44.6 | 44.6 | 44.6 | 24.8 | 21.2 | 15.8 | 8.2 | 9.6 | 1.4 |
| Top-k (t=1.0) | 12.5 | 24.3 | 30.7 | 40.0 | 40.0 | 40.0 | 19.2 | 16.8 | 10.4 | 5.6 | 4.6 | 1.2 |
| Top-p (t=1.0) | 17.5 | 30.2 | 36.6 | 45.2 | 45.2 | 45.2 | 25.0 | 22.4 | 15.8 | 8.2 | 9.6 | 1.2 |
| Top-p (t=1.0) | 14.0 | 25.9 | 32.0 | 39.4 | 39.4 | 39.4 | 21.0 | 17.6 | 11.4 | 5.8 | 6.4 | 1.6 |

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
| Temperature (t=0.3) | 19.5 | 23.6 | 25.1 | 27.0 | 27.0 | 27.0 | 22.4 | 3.0 | 20.4 | 0.0 | 17.0 | 0.0 |
| Temperature (t=0.7) | 17.8 | 24.3 | 27.1 | 30.2 | 30.2 | 30.2 | 22.8 | 7.2 | 17.4 | 0.4 | 15.0 | 0.0 |
| Top-k (t=1.0) | 17.3 | 25.7 | 29.3 | 34.2 | 34.2 | 34.2 | 23.6 | 10.0 | 16.8 | 2.0 | 12.8 | 0.4 |
| Top-p (t=1.0) | 18.5 | 25.0 | 27.3 | 30.2 | 30.2 | 30.2 | 23.8 | 6.8 | 19.8 | 0.6 | 14.8 | 0.0 |
| Top-p (t=1.0) | 18.0 | 25.5 | 29.0 | 33.6 | 33.6 | 33.6 | 23.0 | 9.4 | 17.8 | 1.4 | 14.0 | 0.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Llama-2-7B (base)

- **P-Less (t=0.7)**: rank 5/33
- **P-Less Norm (t=0.6)**: rank 6/33
- **P-Less (t=0.6)**: rank 7/33
- **P-Less Norm (t=0.7)**: rank 8/33
- **P-Less (t=1.0)**: rank 9/33
- **P-Less Norm (t=0.6)**: rank 6/33
- **P-Less (t=0.6)**: rank 7/33
- **P-Less Norm (t=1.0)**: rank 12/33
- **Temperature (t=0.7)**: rank 24/33
- **Top-p 0.9 (t=1.0)**: rank 26/33
- **P-Less (t=0.6)**: rank 7/33
- **Temperature (t=0.7)**: rank 24/33
- Best P-Less vs paper's Temperature sampling: 5.8pp above (23.0% vs 17.2%)
- Our temp_0.7 vs paper's Temperature: 17.1% vs 17.2% (Δ=-0.1pp — sanity check for setup alignment)

### Llama-2-7B-Chat

- **P-Less (t=0.7)**: rank 6/28
- **P-Less Norm (t=0.7)**: rank 7/28
- **P-Less (t=0.6)**: rank 8/28
- **P-Less Norm (t=0.6)**: rank 9/28
- **P-Less Norm (t=1.0)**: rank 10/28
- **P-Less (t=1.0)**: rank 11/28
- **Top-p 0.9 (t=1.0)**: rank 15/28
- **Temperature (t=0.7)**: rank 20/28
- Best P-Less vs paper's Temperature sampling: 0.6pp above (20.6% vs 20.0%)
- Our temp_0.7 vs paper's Temperature: 17.8% vs 20.0% (Δ=-2.2pp — sanity check for setup alignment)

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
