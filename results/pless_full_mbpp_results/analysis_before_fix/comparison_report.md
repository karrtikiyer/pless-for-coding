# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Llama-2-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925).

## pass@1 Comparison

### Llama-2-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=0.6) **←** | Ours | 22.3 |
| 2 | P-Less (t=0.6) **←** | Ours | 22.2 |
| 3 | FSD-d | Paper | 21.2 |
| 4 | P-Less (t=1.0) **←** | Ours | 19.8 |
| 5 | Beam Search | Paper | 19.4 |
| 6 | FSD | Paper | 19.2 |
| 7 | P-Less Norm (t=1.0) **←** | Ours | 19.1 |
| 8 | Diverse Beam Search | Paper | 18.4 |
| 9 | DoLa | Paper | 18.4 |
| 10 | Contrastive Decoding | Paper | 18.2 |
| 11 | Greedy | Paper | 17.8 |
| 12 | Contrastive Search | Paper | 17.4 |
| 13 | Temperature | Paper | 17.2 |
| 14 | Top-p | Paper | 14.8 |
| 15 | Temperature (t=0.7) **←** | Ours | 13.2 |
| 16 | Typical | Paper | 12.0 |
| 17 | Top-k | Paper | 10.2 |
| 18 | η-Sampling | Paper | 9.4 |
| 19 | Mirostat | Paper | 7.8 |

### Llama-2-7B-Chat

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | Beam Search | Paper | 21.6 |
| 2 | Diverse Beam Search | Paper | 21.2 |
| 3 | P-Less (t=0.6) **←** | Ours | 20.5 |
| 4 | P-Less Norm (t=0.6) **←** | Ours | 20.4 |
| 5 | P-Less Norm (t=1.0) **←** | Ours | 20.2 |
| 6 | P-Less (t=1.0) **←** | Ours | 20.1 |
| 7 | Temperature | Paper | 20.0 |
| 8 | Typical | Paper | 18.0 |
| 9 | DoLa | Paper | 18.0 |
| 10 | Temperature (t=0.7) **←** | Ours | 17.8 |
| 11 | FSD | Paper | 17.8 |
| 12 | FSD-d | Paper | 17.8 |
| 13 | Top-p | Paper | 17.6 |
| 14 | Contrastive Search | Paper | 17.4 |
| 15 | Contrastive Decoding | Paper | 17.4 |
| 16 | Greedy | Paper | 17.2 |
| 17 | η-Sampling | Paper | 17.0 |
| 18 | Top-k | Paper | 16.0 |
| 19 | Mirostat | Paper | 16.0 |


## Extended Metrics (Our Methods Only)

### Llama-2-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 22.3 | 28.9 | 31.2 | 33.0 | 33.0 | 32.4 | 28.8 | 14.0 | 22.6 | 5.6 | 18.6 | 2.2 |
| P-Less (t=0.6) | 22.2 | 28.6 | 30.8 | 33.0 | 33.0 | 32.6 | 28.2 | 15.6 | 23.0 | 6.0 | 19.0 | 1.2 |
| P-Less (t=1.0) | 19.8 | 31.3 | 35.5 | 40.0 | 40.0 | 39.6 | 30.4 | 21.2 | 21.0 | 7.4 | 12.2 | 1.6 |
| P-Less Norm (t=1.0) | 19.1 | 30.0 | 34.3 | 38.8 | 38.8 | 37.4 | 27.8 | 19.0 | 20.2 | 7.4 | 11.8 | 2.0 |
| Temperature (t=0.7) | 13.2 | 24.9 | 30.9 | 39.0 | 39.0 | 38.0 | 20.2 | 16.4 | 10.8 | 6.8 | 6.0 | 1.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Llama-2-7B-Chat — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less (t=0.6) | 20.5 | 21.2 | 21.3 | 21.4 | 21.4 | 21.4 | 21.4 | 0.0 | 20.6 | 0.0 | 20.2 | 0.0 |
| P-Less Norm (t=0.6) | 20.4 | 21.1 | 21.3 | 21.4 | 21.4 | 21.4 | 21.0 | 0.0 | 20.2 | 0.0 | 20.0 | 0.0 |
| P-Less Norm (t=1.0) | 20.2 | 21.5 | 22.0 | 22.2 | 22.2 | 22.2 | 21.4 | 0.0 | 20.0 | 0.0 | 19.4 | 0.0 |
| P-Less (t=1.0) | 20.1 | 21.4 | 21.9 | 22.4 | 22.4 | 22.4 | 21.6 | 0.0 | 20.2 | 0.0 | 19.2 | 0.0 |
| Temperature (t=0.7) | 17.8 | 24.3 | 27.1 | 30.2 | 30.2 | 30.2 | 22.8 | 7.2 | 17.4 | 0.4 | 15.0 | 0.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Llama-2-7B (base)

- **P-Less Norm (t=0.6)**: rank 1/19
- **P-Less (t=0.6)**: rank 2/19
- **P-Less (t=1.0)**: rank 4/19
- **P-Less Norm (t=1.0)**: rank 7/19
- **Temperature (t=0.7)**: rank 15/19
- Our temp_0.7 vs paper's Temperature: 13.2% vs 17.2% (Δ=-4.0pp — sanity check for setup alignment)

### Llama-2-7B-Chat

- **P-Less (t=0.6)**: rank 3/19
- **P-Less Norm (t=0.6)**: rank 4/19
- **P-Less Norm (t=1.0)**: rank 5/19
- **P-Less (t=1.0)**: rank 6/19
- **Temperature (t=0.7)**: rank 10/19
- Our temp_0.7 vs paper's Temperature: 17.8% vs 20.0% (Δ=-2.2pp — sanity check for setup alignment)

### Limitations

- We ran 5 configs per model (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7) vs the paper's 14 methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
