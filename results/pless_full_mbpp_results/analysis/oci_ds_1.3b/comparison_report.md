# MBPP: P-Less vs arXiv 2507.03160 Baseline (OCI-DS-1.3B)

Comparison of p-less sampling against the top_p=0.95/temp=0.2 baseline from "Assessing Small Language Models for Code Generation" (arXiv:2507.03160). BigCode zero-shot docstring format, MBPP full (500 tasks), n=10, unbiased pass@k estimator. Models: OCI-DS-1.3B.

## pass@1 Comparison

### OCI-DS-1.3B

| Rank | Method | Source | pass@1 (%) |
| ---: | ------ | ------ | ---------: |
| 1 | P-Less Norm (t=1.0) **←** | Ours | 44.6 |
| 2 | Top-p p=0.95 (paper replication) (t=0.2) **←** | Ours | 44.5 |
| 3 | P-Less (t=1.0) **←** | Ours | 44.1 |
| 4 | Top-p (paper, t=0.2) | Paper | 44.0 |
| 5 | P-Less Norm (t=0.6) **←** | Ours | 43.9 |
| 6 | P-Less (t=0.6) **←** | Ours | 43.9 |
| 7 | Temperature (t=0.7) **←** | Ours | 43.1 |
| 8 | Top-p p=0.9 (t=1.0) **←** | Ours | 42.8 |


## Extended Metrics (Our Methods Only)

### OCI-DS-1.3B — Extended Metrics

| Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (dist) | cover@0.3 | cover@0.3 (dist) | cover@0.5 | cover@0.5 (dist) | cover@0.7 | cover@0.7 (dist) |
| --- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| P-Less (t=0.6) | 43.9 | 47.1 | 48.2 | 49.0 | 49.0 | 49.0 | 47.0 | 4.0 | 44.2 | 1.0 | 41.8 | 0.0 |
| P-Less (t=1.0) | 44.1 | 48.2 | 49.7 | 51.2 | 51.2 | 51.2 | 47.2 | 6.2 | 45.0 | 1.0 | 41.0 | 0.0 |
| P-Less Norm (t=0.6) | 43.9 | 47.3 | 48.4 | 49.4 | 49.4 | 49.4 | 47.0 | 4.2 | 44.4 | 0.6 | 41.6 | 0.2 |
| P-Less Norm (t=1.0) | 44.6 | 48.6 | 49.8 | 51.0 | 51.0 | 51.0 | 48.4 | 6.2 | 45.6 | 1.0 | 42.8 | 0.2 |
| Temperature (t=0.7) | 43.1 | 54.0 | 58.0 | 62.4 | 62.4 | 62.4 | 51.4 | 33.6 | 44.0 | 13.8 | 38.0 | 4.8 |
| Top-p p=0.95 (paper replication) (t=0.2) | 44.5 | 48.8 | 50.1 | 51.8 | 51.8 | 51.8 | 47.8 | 7.8 | 46.0 | 1.4 | 42.4 | 0.0 |
| Top-p p=0.9 (t=1.0) | 42.8 | 54.6 | 58.8 | 64.0 | 64.0 | 64.0 | 52.4 | 39.8 | 45.2 | 16.2 | 37.2 | 4.2 |

*pass@k as %; cover@t = % of tasks where ≥t fraction of samples are correct; (dist) = distinct correct samples only.*

## Analysis

### OCI-DS-1.3B

- **P-Less Norm (t=1.0)**: rank 1/8
- **Top-p p=0.95 (paper replication) (t=0.2)**: rank 2/8
- **P-Less (t=1.0)**: rank 3/8
- **P-Less Norm (t=0.6)**: rank 5/8
- **P-Less (t=0.6)**: rank 6/8
- **Temperature (t=0.7)**: rank 7/8
- **Top-p p=0.9 (t=1.0)**: rank 8/8

### Limitations

- We ran 6 configs (pless t=0.6/1.0, pless_norm t=0.6/1.0, temp 0.7, top_p p=0.9) vs the paper's methods. The comparison is partial.
- Our `temp_0.7` serves as an anchor to validate evaluation setup similarity; exact match is not expected due to differences in prompting, generation length, and MBPP subset.
- The paper reports single-sample pass@1; our pass@1 uses the unbiased estimator over 10 samples, which may differ slightly from greedy/beam-search single-shot accuracy.
