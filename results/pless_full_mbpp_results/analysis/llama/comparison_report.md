# Full MBPP (500 problems): P-Less vs Paper Decoding Methods (Llama-2-7B)

Comparison of p-less sampling against decoding methods from "A Thorough Examination of Decoding Methods in the Era of LLMs" (arXiv:2402.06925), Table 1.

## pass@1 Comparison

### Llama-2-7B (base)

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=0.6) **←** | Ours | 22.6 |
| 2 | P-Less (t=0.6) **←** | Ours | 22.5 |
| 3 | P-Less (t=1.0) **←** | Ours | 22.4 |
| 4 | P-Less Norm (t=1.0) **←** | Ours | 21.8 |
| 5 | FSD-d | Paper | 21.2 |
| 6 | Beam Search | Paper | 19.4 |
| 7 | FSD | Paper | 19.2 |
| 8 | Diverse Beam Search | Paper | 18.4 |
| 9 | DoLa | Paper | 18.4 |
| 10 | Contrastive Decoding | Paper | 18.2 |
| 11 | Greedy | Paper | 17.8 |
| 12 | Contrastive Search | Paper | 17.4 |
| 13 | Temperature | Paper | 17.2 |
| 14 | Temperature (t=0.7) **←** | Ours | 17.1 |
| 15 | Top-p | Paper | 14.8 |
| 16 | Top-p (ours) (t=1.0) **←** | Ours | 14.0 |
| 17 | Typical | Paper | 12.0 |
| 18 | Top-k | Paper | 10.2 |
| 19 | η-Sampling | Paper | 9.4 |
| 20 | Mirostat | Paper | 7.8 |

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
| 8 | Top-p (ours) (t=1.0) **←** | Ours | 18.0 |
| 9 | DoLa | Paper | 18.0 |
| 10 | Typical | Paper | 18.0 |
| 11 | FSD | Paper | 17.8 |
| 12 | FSD-d | Paper | 17.8 |
| 13 | Temperature (t=0.7) **←** | Ours | 17.8 |
| 14 | Top-p | Paper | 17.6 |
| 15 | Contrastive Search | Paper | 17.4 |
| 16 | Contrastive Decoding | Paper | 17.4 |
| 17 | Greedy | Paper | 17.2 |
| 18 | η-Sampling | Paper | 17.0 |
| 19 | Top-k | Paper | 16.0 |
| 20 | Mirostat | Paper | 16.0 |


## Extended Metrics (Our Methods Only)

### Llama-2-7B (base) — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 22.6 | 26.6 | 28.1 | 30.0 | 30.0 | 29.6 | 25.0 | 3.0 | 23.8 | 0.2 | 20.0 | 0.0 |
| P-Less Norm (t=1.0) | 21.8 | 29.8 | 33.1 | 37.2 | 37.2 | 37.0 | 27.8 | 7.0 | 23.0 | 1.2 | 18.0 | 0.2 |
| P-Less (t=0.6) | 22.5 | 26.4 | 27.8 | 29.2 | 29.2 | 29.0 | 26.2 | 2.6 | 23.2 | 0.2 | 20.6 | 0.0 |
| P-Less (t=1.0) | 22.4 | 30.5 | 33.6 | 37.0 | 37.0 | 36.8 | 29.4 | 9.2 | 22.6 | 1.0 | 17.2 | 0.0 |
| Temperature (t=0.7) | 17.1 | 29.7 | 36.0 | 44.6 | 44.6 | 44.6 | 24.8 | 21.2 | 15.8 | 8.2 | 9.6 | 1.4 |
| Top-p (ours) (t=1.0) | 14.0 | 25.9 | 32.0 | 39.4 | 39.4 | 39.4 | 21.0 | 17.6 | 11.4 | 5.8 | 6.4 | 1.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Llama-2-7B-Chat — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less Norm (t=0.6) | 20.4 | 21.1 | 21.3 | 21.4 | 21.4 | 21.4 | 21.0 | 0.0 | 20.2 | 0.0 | 20.0 | 0.0 |
| P-Less Norm (t=1.0) | 20.2 | 21.5 | 22.0 | 22.2 | 22.2 | 22.2 | 21.4 | 0.0 | 20.0 | 0.0 | 19.4 | 0.0 |
| P-Less (t=0.6) | 20.5 | 21.2 | 21.3 | 21.4 | 21.4 | 21.4 | 21.4 | 0.0 | 20.6 | 0.0 | 20.2 | 0.0 |
| P-Less (t=1.0) | 20.1 | 21.4 | 21.9 | 22.4 | 22.4 | 22.4 | 21.6 | 0.0 | 20.2 | 0.0 | 19.2 | 0.0 |
| Temperature (t=0.7) | 17.8 | 24.3 | 27.1 | 30.2 | 30.2 | 30.2 | 22.8 | 7.2 | 17.4 | 0.4 | 15.0 | 0.0 |
| Top-p (ours) (t=1.0) | 18.0 | 25.5 | 29.0 | 33.6 | 33.6 | 33.6 | 23.0 | 9.4 | 17.8 | 1.4 | 14.0 | 0.0 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Llama-2-7B (base)

- **P-Less Norm (t=0.6)**: rank 1/20
- **P-Less (t=0.6)**: rank 2/20
- **P-Less (t=1.0)**: rank 3/20
- **P-Less Norm (t=1.0)**: rank 4/20
- **Temperature (t=0.7)**: rank 14/20
- **Top-p (ours) (t=1.0)**: rank 16/20
- Best P-Less vs paper's Temperature sampling: 5.3pp above (22.5% vs 17.2%)
- Our temp_0.7 vs paper's Temperature: 17.1% vs 17.2% (Δ=-0.1pp — sanity check for setup alignment)

### Llama-2-7B-Chat

- **P-Less (t=0.6)**: rank 3/20
- **P-Less Norm (t=0.6)**: rank 4/20
- **P-Less Norm (t=1.0)**: rank 5/20
- **P-Less (t=1.0)**: rank 6/20
- **Top-p (ours) (t=1.0)**: rank 8/20
- **Temperature (t=0.7)**: rank 13/20
- Best P-Less vs paper's Temperature sampling: 0.5pp above (20.5% vs 20.0%)
- Our temp_0.7 vs paper's Temperature: 17.8% vs 20.0% (Δ=-2.2pp — sanity check for setup alignment)

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
