# MBPP: P-Less vs Paper Decoding Methods (Llama-2-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925).

## pass@1 Comparison

### Llama-2-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=1.0) **←** | Ours | 28.6 |
| 2 | P-Less (t=1.0) **←** | Ours | 28.5 |
| 3 | FSD-d | Paper | 21.2 |
| 4 | Temperature (t=0.7) **←** | Ours | 20.2 |
| 5 | Beam Search | Paper | 19.4 |
| 6 | FSD | Paper | 19.2 |
| 7 | Diverse Beam Search | Paper | 18.4 |
| 8 | DoLa | Paper | 18.4 |
| 9 | Contrastive Decoding | Paper | 18.2 |
| 10 | Greedy | Paper | 17.8 |
| 11 | Contrastive Search | Paper | 17.4 |
| 12 | Temperature | Paper | 17.2 |
| 13 | Top-p | Paper | 14.8 |
| 14 | Typical | Paper | 12.0 |
| 15 | Top-k | Paper | 10.2 |
| 16 | η-Sampling | Paper | 9.4 |
| 17 | Mirostat | Paper | 7.8 |

### Llama-2-7B-Chat

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=1.0) **←** | Ours | 29.0 |
| 2 | P-Less (t=1.0) **←** | Ours | 28.6 |
| 3 | Temperature (t=0.7) **←** | Ours | 27.3 |
| 4 | Beam Search | Paper | 21.6 |
| 5 | Diverse Beam Search | Paper | 21.2 |
| 6 | Temperature | Paper | 20.0 |
| 7 | DoLa | Paper | 18.0 |
| 8 | Typical | Paper | 18.0 |
| 9 | FSD | Paper | 17.8 |
| 10 | FSD-d | Paper | 17.8 |
| 11 | Top-p | Paper | 17.6 |
| 12 | Contrastive Search | Paper | 17.4 |
| 13 | Contrastive Decoding | Paper | 17.4 |
| 14 | Greedy | Paper | 17.2 |
| 15 | η-Sampling | Paper | 17.0 |
| 16 | Top-k | Paper | 16.0 |
| 17 | Mirostat | Paper | 16.0 |


## Extended Metrics (Our Methods Only)

### Llama-2-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=1.0) | 28.6 | 41.9 | 46.5 | 51.4 | 51.4 | 48.2 | 39.7 | 30.0 | 31.5 | 14.4 | 21.0 | 3.5 |
| P-Less (t=1.0) | 28.5 | 42.0 | 47.2 | 52.1 | 52.1 | 50.6 | 38.5 | 31.9 | 30.0 | 14.0 | 20.6 | 5.8 |
| Temperature (t=0.7) | 20.2 | 36.2 | 43.5 | 52.1 | 52.1 | 49.8 | 31.1 | 28.0 | 19.5 | 13.2 | 10.1 | 3.1 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Llama-2-7B-Chat — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=1.0) | 29.0 | 31.2 | 31.8 | 32.3 | 32.3 | 32.3 | 31.5 | 0.0 | 30.0 | 0.0 | 27.6 | 0.0 |
| P-Less (t=1.0) | 28.6 | 30.8 | 31.8 | 33.1 | 33.1 | 33.1 | 30.4 | 0.0 | 28.4 | 0.0 | 26.8 | 0.0 |
| Temperature (t=0.7) | 27.3 | 35.2 | 38.3 | 42.8 | 42.8 | 42.8 | 32.7 | 10.5 | 28.4 | 1.6 | 24.1 | 0.4 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Llama-2-7B (base)

- **P-Less (t=1.0)**: rank 2/17
- **P-Less Norm (t=1.0)**: rank 1/17
- **Temperature (t=0.7)**: rank 4/17
- P-Less vs paper's Temperature sampling: 11.3pp above (28.5% vs 17.2%)
- Our temp_0.7 vs paper's Temperature: 20.2% vs 17.2% (Δ=+3.0pp — sanity check for setup alignment)

### Llama-2-7B-Chat

- **P-Less (t=1.0)**: rank 2/17
- **P-Less Norm (t=1.0)**: rank 1/17
- **Temperature (t=0.7)**: rank 3/17
- P-Less vs paper's Temperature sampling: 8.6pp above (28.6% vs 20.0%)
- Our temp_0.7 vs paper's Temperature: 27.3% vs 20.0% (Δ=+7.3pp — sanity check for setup alignment)

### Limitations

- We ran only 3 methods (pless, pless_norm, temp_0.7) vs the paper's 14. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
