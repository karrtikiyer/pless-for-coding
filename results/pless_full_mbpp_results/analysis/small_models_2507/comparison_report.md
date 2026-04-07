# MBPP: P-Less vs arXiv 2507.03160 Baseline (Qwen2.5-Coder-3B, Qwen2.5-Coder-1.5B, OCI-DS-1.3B)

Comparison of p-less sampling against the top_p=0.95/temp=0.2 baseline from "Assessing Small Language Models for Code Generation" (arXiv:2507.03160). BigCode zero-shot docstring format, MBPP full (500 tasks), n=10, unbiased pass@k estimator. Models: Qwen2.5-Coder-3B, Qwen2.5-Coder-1.5B, OCI-DS-1.3B.

## pass@1 Comparison

### Qwen2.5-Coder-3B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) **←** | Ours | 61.0 |
| 2 | greedy (t=1.0) **←** | Ours | 60.2 |
| 3 | P-Less Norm (t=0.6) **←** | Ours | 59.4 |
| 4 | P-Less (t=0.6) **←** | Ours | 59.3 |
| 5 | beam8 (t=1.0) **←** | Ours | 59.2 |
| 6 | P-Less Norm (t=0.7) **←** | Ours | 59.0 |
| 7 | P-Less (t=0.7) **←** | Ours | 58.6 |
| 8 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 58.2 |
| 9 | P-Less Norm (t=1.0) **←** | Ours | 57.6 |
| 10 | Top-p (paper, t=0.2) | Paper | 57.0 |
| 11 | P-Less (t=1.0) **←** | Ours | 56.5 |
| 12 | Temperature (t=0.7) **←** | Ours | 42.6 |
| 13 | Top-p p=0.9 (t=1.0) **←** | Ours | 34.7 |

### Qwen2.5-Coder-1.5B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) **←** | Ours | 54.4 |
| 2 | greedy (t=1.0) **←** | Ours | 54.4 |
| 3 | beam8 (t=1.0) **←** | Ours | 53.8 |
| 4 | P-Less (t=0.6) **←** | Ours | 53.1 |
| 5 | P-Less (t=0.7) **←** | Ours | 53.0 |
| 6 | P-Less Norm (t=0.7) **←** | Ours | 52.9 |
| 7 | P-Less Norm (t=0.6) **←** | Ours | 52.8 |
| 8 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 52.5 |
| 9 | P-Less Norm (t=1.0) **←** | Ours | 51.9 |
| 10 | P-Less (t=1.0) **←** | Ours | 51.3 |
| 11 | Top-p (paper, t=0.2) | Paper | 51.0 |
| 12 | Temperature (t=0.7) **←** | Ours | 38.1 |
| 13 | Top-p p=0.9 (t=1.0) **←** | Ours | 32.7 |

### OCI-DS-1.3B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | beam4 (t=1.0) **←** | Ours | 46.4 |
| 2 | beam8 (t=1.0) **←** | Ours | 45.4 |
| 3 | P-Less (t=0.7) **←** | Ours | 44.9 |
| 4 | P-Less Norm (t=0.7) **←** | Ours | 44.9 |
| 5 | P-Less Norm (t=1.0) **←** | Ours | 44.6 |
| 6 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 44.5 |
| 7 | greedy (t=1.0) **←** | Ours | 44.2 |
| 8 | P-Less (t=1.0) **←** | Ours | 44.1 |
| 9 | Top-p (paper, t=0.2) | Paper | 44.0 |
| 10 | P-Less Norm (t=0.6) **←** | Ours | 43.9 |
| 11 | P-Less (t=0.6) **←** | Ours | 43.9 |
| 12 | Temperature (t=0.7) **←** | Ours | 43.1 |
| 13 | Top-p p=0.9 (t=1.0) **←** | Ours | 42.8 |


## Extended Metrics (Our Methods Only)

### Qwen2.5-Coder-3B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 60.4 | 61.0 | 60.4 | 61.0 | 60.4 | 61.0 | 60.4 |
| beam8 (t=1.0) | 59.2 | 59.2 | 59.2 | 59.2 | 59.2 | 56.0 | 59.2 | 56.0 | 59.2 | 56.0 | 59.2 | 56.0 |
| greedy (t=1.0) | 60.2 | 60.2 | 60.2 | 60.2 | 60.2 | 59.8 | 60.2 | 59.8 | 60.2 | 59.8 | 60.2 | 59.8 |
| P-Less (t=0.6) | 59.3 | 63.4 | 64.8 | 66.2 | 66.2 | 65.4 | 63.4 | 11.6 | 60.0 | 1.8 | 57.2 | 0.0 |
| P-Less (t=0.7) | 58.6 | 64.5 | 66.2 | 67.4 | 67.4 | 67.2 | 64.4 | 17.4 | 60.0 | 3.4 | 55.4 | 0.4 |
| P-Less (t=1.0) | 56.5 | 66.4 | 69.4 | 72.2 | 72.2 | 72.2 | 65.8 | 36.6 | 59.4 | 12.2 | 51.6 | 3.8 |
| P-Less Norm (t=0.6) | 59.4 | 63.7 | 65.2 | 66.6 | 66.6 | 66.2 | 63.4 | 10.8 | 60.0 | 2.2 | 57.2 | 0.2 |
| P-Less Norm (t=0.7) | 59.0 | 65.0 | 66.7 | 68.0 | 68.0 | 67.8 | 65.0 | 16.6 | 61.2 | 4.6 | 55.2 | 0.6 |
| P-Less Norm (t=1.0) | 57.6 | 66.9 | 69.8 | 72.4 | 72.4 | 72.2 | 66.8 | 37.4 | 59.4 | 13.2 | 53.4 | 3.0 |
| Temperature (t=0.7) | 42.6 | 63.9 | 70.6 | 77.6 | 77.6 | 77.4 | 61.6 | 59.2 | 48.2 | 40.2 | 34.2 | 17.0 |
| Top-p p=0.95 (paper replication) (t=0.2) | 58.2 | 66.6 | 69.2 | 71.6 | 71.6 | 71.6 | 65.6 | 30.8 | 61.8 | 9.6 | 54.4 | 2.2 |
| Top-p p=0.9 (t=1.0) | 34.7 | 58.4 | 67.3 | 77.0 | 77.0 | 76.8 | 53.6 | 52.4 | 37.4 | 33.2 | 20.6 | 13.8 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### Qwen2.5-Coder-1.5B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 53.6 | 54.4 | 53.6 | 54.4 | 53.6 | 54.4 | 53.6 |
| beam8 (t=1.0) | 53.8 | 53.8 | 53.8 | 53.8 | 53.8 | 50.6 | 53.8 | 50.6 | 53.8 | 50.6 | 53.8 | 50.6 |
| greedy (t=1.0) | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 | 54.4 |
| P-Less (t=0.6) | 53.1 | 57.7 | 59.2 | 60.8 | 60.8 | 60.8 | 57.4 | 10.4 | 53.8 | 1.4 | 51.0 | 0.0 |
| P-Less (t=0.7) | 53.0 | 58.6 | 60.5 | 62.2 | 62.2 | 62.2 | 57.8 | 13.2 | 53.8 | 2.8 | 50.2 | 0.6 |
| P-Less (t=1.0) | 51.3 | 61.2 | 64.7 | 68.8 | 68.8 | 68.6 | 60.2 | 35.4 | 52.6 | 15.8 | 47.2 | 4.0 |
| P-Less Norm (t=0.6) | 52.8 | 58.0 | 59.8 | 61.4 | 61.4 | 61.4 | 57.8 | 11.4 | 53.4 | 1.4 | 49.8 | 0.0 |
| P-Less Norm (t=0.7) | 52.9 | 58.3 | 60.0 | 61.6 | 61.6 | 61.6 | 57.8 | 14.6 | 54.6 | 3.4 | 50.4 | 0.2 |
| P-Less Norm (t=1.0) | 51.9 | 61.4 | 64.5 | 67.8 | 67.8 | 67.8 | 60.2 | 34.2 | 54.4 | 15.4 | 48.4 | 4.2 |
| Temperature (t=0.7) | 38.1 | 58.0 | 64.7 | 71.0 | 71.0 | 71.0 | 56.0 | 53.8 | 41.4 | 36.0 | 29.0 | 16.2 |
| Top-p p=0.95 (paper replication) (t=0.2) | 52.5 | 61.2 | 64.4 | 68.6 | 68.6 | 68.4 | 59.4 | 31.4 | 54.8 | 10.8 | 48.8 | 2.6 |
| Top-p p=0.9 (t=1.0) | 32.7 | 53.4 | 61.4 | 70.2 | 70.2 | 70.2 | 49.2 | 48.6 | 34.2 | 32.0 | 20.4 | 15.6 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

### OCI-DS-1.3B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| beam4 (t=1.0) | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 | 46.4 |
| beam8 (t=1.0) | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 | 45.4 |
| greedy (t=1.0) | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 |
| P-Less (t=0.6) | 43.9 | 47.1 | 48.2 | 49.0 | 49.0 | 49.0 | 47.0 | 4.0 | 44.2 | 1.0 | 41.8 | 0.0 |
| P-Less (t=0.7) | 44.9 | 48.4 | 49.6 | 50.6 | 50.6 | 50.6 | 48.0 | 3.0 | 45.4 | 0.2 | 42.6 | 0.0 |
| P-Less (t=1.0) | 44.1 | 48.2 | 49.7 | 51.2 | 51.2 | 51.2 | 47.2 | 6.2 | 45.0 | 1.0 | 41.0 | 0.0 |
| P-Less Norm (t=0.6) | 43.9 | 47.3 | 48.4 | 49.4 | 49.4 | 49.4 | 47.0 | 4.2 | 44.4 | 0.6 | 41.6 | 0.2 |
| P-Less Norm (t=0.7) | 44.9 | 48.3 | 49.4 | 50.2 | 50.2 | 50.2 | 48.6 | 3.6 | 45.4 | 0.0 | 43.0 | 0.0 |
| P-Less Norm (t=1.0) | 44.6 | 48.6 | 49.8 | 51.0 | 51.0 | 51.0 | 48.4 | 6.2 | 45.6 | 1.0 | 42.8 | 0.2 |
| Temperature (t=0.7) | 43.1 | 54.0 | 58.0 | 62.4 | 62.4 | 62.4 | 51.4 | 33.6 | 44.0 | 13.8 | 38.0 | 4.8 |
| Top-p p=0.95 (paper replication) (t=0.2) | 44.5 | 48.8 | 50.1 | 51.8 | 51.8 | 51.8 | 47.8 | 7.8 | 46.0 | 1.4 | 42.4 | 0.0 |
| Top-p p=0.9 (t=1.0) | 42.8 | 54.6 | 58.8 | 64.0 | 64.0 | 64.0 | 52.4 | 39.8 | 45.2 | 16.2 | 37.2 | 4.2 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### Qwen2.5-Coder-3B

- **beam4 (t=1.0)**: rank 1/13
- **greedy (t=1.0)**: rank 2/13
- **P-Less Norm (t=0.6)**: rank 3/13
- **P-Less (t=0.6)**: rank 4/13
- **beam8 (t=1.0)**: rank 5/13
- **P-Less Norm (t=0.7)**: rank 6/13
- **P-Less (t=0.7)**: rank 7/13
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 8/13
- **P-Less Norm (t=1.0)**: rank 9/13
- **P-Less (t=1.0)**: rank 11/13
- **Temperature (t=0.7)**: rank 12/13
- **Top-p p=0.9 (t=1.0)**: rank 13/13

### Qwen2.5-Coder-1.5B

- **beam4 (t=1.0)**: rank 1/13
- **greedy (t=1.0)**: rank 2/13
- **beam8 (t=1.0)**: rank 3/13
- **P-Less (t=0.6)**: rank 4/13
- **P-Less (t=0.7)**: rank 5/13
- **P-Less Norm (t=0.7)**: rank 6/13
- **P-Less Norm (t=0.6)**: rank 7/13
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 8/13
- **P-Less Norm (t=1.0)**: rank 9/13
- **P-Less (t=1.0)**: rank 10/13
- **Temperature (t=0.7)**: rank 12/13
- **Top-p p=0.9 (t=1.0)**: rank 13/13

### OCI-DS-1.3B

- **beam4 (t=1.0)**: rank 1/13
- **beam8 (t=1.0)**: rank 2/13
- **P-Less (t=0.7)**: rank 3/13
- **P-Less Norm (t=0.7)**: rank 4/13
- **P-Less Norm (t=1.0)**: rank 5/13
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 6/13
- **greedy (t=1.0)**: rank 7/13
- **P-Less (t=1.0)**: rank 8/13
- **P-Less Norm (t=0.6)**: rank 10/13
- **P-Less (t=0.6)**: rank 11/13
- **Temperature (t=0.7)**: rank 12/13
- **Top-p p=0.9 (t=1.0)**: rank 13/13

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
