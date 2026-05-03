# Truncation Partition Analysis — Qwen3-8B

**Question:** Is the +10.6pp pass@1 gap (C → H1) a real effect of the temp/pless change, or an artefact of the 4096 → 8192 token budget?

**Method:** Partition the 500 MBPP tasks by how many of C's 10 samples truncated (hit the 4096 ceiling without closing `</think>`). Then compute pass@1 for each partition under each config.

## Bucket sizes (by C truncation count)

| Bucket | Tasks |
|--------|-------|
| none (0/10) | 314 |
| partial (1-9/10) | 123 |
| all (10/10) | 63 |

## pass@1 by partition

Per-task pass@1 is `num_correct / 10`, then averaged over the bucket.

| Bucket (n) | C (4096t) | F (4096t) | H1 (8192t) | H2 (8192t) | H3 (8192t) | Δ H1−C |
|------------|-----|-----|-----|-----|-----|--------|
| none (0/10) (314) | 0.9672 | 0.9592 | 0.9669 | 0.9646 | 0.9675 | -0.0pp |
| partial (1-9/10) (123) | 0.5309 | 0.5276 | 0.8309 | 0.8268 | 0.8268 | +30.0pp |
| all (10/10) (63) | 0.0000 | 0.0111 | 0.2540 | 0.2651 | 0.2603 | +25.4pp |

## Sanity check: overall pass@1

| Config | pass@1 (recomputed) |
|--------|---------------------|
| C | 0.7380 |
| F | 0.7336 |
| H1 | 0.8436 |
| H2 | 0.8426 |
| H3 | 0.8438 |

## Decomposition of the C → H1 gap

Contribution = (bucket size / 500) × (bucket pass@1 delta).

| Bucket | Tasks | C pass@1 | H1 pass@1 | Δ | Contribution to total Δ |
|--------|-------|----------|-----------|---|-------------------------|
| none (0/10) | 314 | 0.9672 | 0.9669 | -0.0003 | -0.02pp |
| partial (1-9/10) | 123 | 0.5309 | 0.8309 | +0.3000 | +7.38pp |
| all (10/10) | 63 | 0.0000 | 0.2540 | +0.2540 | +3.20pp |
| **Total** | 500 | — | — | — | **+10.56pp** |

## Did H1 (8192 tokens) actually rescue the all-trunc-in-C tasks?

- C all-trunc tasks: 63
- Of those, **still all-truncated in H1 (8192t):** 24
- Of those, **fully rescued (0 truncated samples in H1):** 3
- Avg truncated samples in H1 for these tasks: 6.97/10

## Interpretation

- On tasks where C had **zero truncation** (the cleanest comparison — token budget can't be helping H1 here), H1 vs C delta is **-0.0pp**.
  - This is essentially flat, suggesting the **temp/pless setting change has no effect** in the no-truncation regime — the +10.6pp overall gap is mostly a budget artefact.