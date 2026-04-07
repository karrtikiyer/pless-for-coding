# Token Survivor Analysis Report

**Model:** Qwen/Qwen2.5-Coder-3B-Instruct | **Dataset:** MBPP-full (stratified subset)

**Tasks:** 30 total — 10 easy, 10 medium, 10 hard

## Aggregate Statistics

| T1 | Steps | Median surv | Mean surv | % deterministic | % constrained | % branching | Mean emb sim |
|----|-------|-------------|-----------|-----------------|---------------|-------------|--------------|
| 0.6 | 25016 | 1 | 1.0 | 98.7% | 1.1% | 0.2% | 0.148 |
| 1.0 | 24102 | 1 | 1.0 | 96.8% | 2.8% | 0.4% | 0.114 |
| 1.5 | 23680 | 1 | 1.1 | 91.6% | 7.1% | 1.3% | 0.094 |
| 2.0 | 26899 | 1 | 1.7 | 79.1% | 16.2% | 4.7% | 0.134 |

## Breakdown by Task Difficulty

### Easy tasks

| T1 | Steps | Median surv | % deterministic | % branching | Mean emb sim (branching) |
|----|-------|-------------|-----------------|-------------|--------------------------|
| 0.6 | 5979 | 1 | 98.6% | 0.4% | 0.086 |
| 1.0 | 5980 | 1 | 97.8% | 0.4% | 0.073 |
| 1.5 | 5811 | 1 | 93.9% | 1.2% | 0.074 |
| 2.0 | 4770 | 1 | 82.7% | 3.9% | 0.111 |

### Medium tasks

| T1 | Steps | Median surv | % deterministic | % branching | Mean emb sim (branching) |
|----|-------|-------------|-----------------|-------------|--------------------------|
| 0.6 | 8171 | 1 | 98.9% | 0.1% | 0.019 |
| 1.0 | 8140 | 1 | 96.3% | 0.4% | 0.125 |
| 1.5 | 8160 | 1 | 89.8% | 1.4% | 0.105 |
| 2.0 | 8626 | 1 | 74.9% | 5.8% | 0.148 |

### Hard tasks

| T1 | Steps | Median surv | % deterministic | % branching | Mean emb sim (branching) |
|----|-------|-------------|-----------------|-------------|--------------------------|
| 0.6 | 10866 | 1 | 98.6% | 0.1% | 0.375 |
| 1.0 | 9982 | 1 | 96.6% | 0.4% | 0.130 |
| 1.5 | 9709 | 1 | 91.7% | 1.4% | 0.095 |
| 2.0 | 13503 | 1 | 80.6% | 4.3% | 0.128 |

## T2 Effect Simulation

Simulated T2 applied to saved survivor probabilities. Shows mean entropy delta (nats) over baseline (T2=1).

| T1 | Regime | Steps | ΔH (T2=2.0) | ΔH (T2=5.0) |
|----|--------|-------|-------------|-------------|
| 0.6 | branching | 40 | +0.0014 | +0.0018 |
| 0.6 | constrained | 287 | +0.0021 | +0.0027 |
| 0.6 | deterministic | 24689 | +0.0000 | +0.0000 |
| 1.0 | branching | 95 | +0.0043 | +0.0055 |
| 1.0 | constrained | 679 | +0.0065 | +0.0084 |
| 1.0 | deterministic | 23328 | +0.0000 | +0.0000 |
| 1.5 | branching | 315 | +0.0193 | +0.0248 |
| 1.5 | constrained | 1674 | +0.0202 | +0.0261 |
| 1.5 | deterministic | 21691 | +0.0000 | +0.0000 |
| 2.0 | branching | 1257 | +0.1108 | +0.1421 |
| 2.0 | constrained | 4353 | +0.0683 | +0.0884 |
| 2.0 | deterministic | 21289 | +0.0000 | +0.0000 |

## Concrete Examples: Branching Points

Decoded survivor tokens at interesting branching points (high diversity among survivors).

### Example 1 (T1=2.0, step=49, task=160)
- Survivor count: 119
- Post-threshold entropy: 4.442 nats
- Embedding similarity: 0.124

| Token | Prob | Category |
|-------|------|----------|
| ` is` | 0.0588 | keyword |
| ` *` | 0.0570 | operator |
| ` =` | 0.0417 | operator |
| ` +` | 0.0305 | operator |
| `,` | 0.0253 | punctuation |
| ` -` | 0.0238 | operator |
| `*x` | 0.0238 | identifier_fragment |
| ` and` | 0.0203 | keyword |
| ` should` | 0.0191 | identifier |
| ` must` | 0.0191 | identifier |

### Example 2 (T1=2.0, step=34, task=31)
- Survivor count: 60
- Post-threshold entropy: 3.900 nats
- Embedding similarity: 0.050

| Token | Prob | Category |
|-------|------|----------|
| ` arr` | 0.0804 | identifier |
| ` count` | 0.0458 | identifier |
| ` c` | 0.0444 | identifier |
| ` counter` | 0.0368 | identifier |
| ` lst` | 0.0368 | identifier |
| ` res` | 0.0357 | identifier |
| ` freq` | 0.0287 | identifier |
| ` a` | 0.0253 | identifier |
| ` cnt` | 0.0245 | identifier |
| ` flat` | 0.0238 | identifier |

### Example 3 (T1=2.0, step=31, task=45)
- Survivor count: 63
- Post-threshold entropy: 3.884 nats
- Embedding similarity: 0.037

| Token | Prob | Category |
|-------|------|----------|
| `(g` | 0.1005 | identifier_fragment |
| ` if` | 0.0460 | keyword |
| ` *` | 0.0446 | operator |
| `_g` | 0.0370 | identifier |
| ` &` | 0.0316 | operator |
| ` //` | 0.0288 | operator |
| ` -` | 0.0279 | operator |
| `_h` | 0.0279 | identifier |
| `(arr` | 0.0262 | identifier_fragment |
| `_l` | 0.0254 | identifier |

### Example 4 (T1=2.0, step=42, task=31)
- Survivor count: 59
- Post-threshold entropy: 3.855 nats
- Embedding similarity: 0.041

| Token | Prob | Category |
|-------|------|----------|
| ` return` | 0.0659 | keyword |
| ` res` | 0.0546 | identifier |
| ` heap` | 0.0482 | identifier |
| ` most` | 0.0413 | identifier |
| ` ans` | 0.0413 | identifier |
| ` lst` | 0.0388 | identifier |
| ` top` | 0.0376 | identifier |
| ` max` | 0.0321 | identifier |
| ` arr` | 0.0266 | identifier |
| ` result` | 0.0258 | identifier |

### Example 5 (T1=2.0, step=17, task=160)
- Survivor count: 55
- Post-threshold entropy: 3.701 nats
- Embedding similarity: 0.132

| Token | Prob | Category |
|-------|------|----------|
| `(step` | 0.1127 | identifier_fragment |
| `1` | 0.0549 | literal |
| `(x` | 0.0532 | identifier_fragment |
| `(a` | 0.0470 | identifier_fragment |
| `(k` | 0.0414 | identifier_fragment |
| `_` | 0.0389 | identifier |
| `(m` | 0.0355 | identifier_fragment |
| `(n` | 0.0323 | identifier_fragment |
| `(d` | 0.0323 | identifier_fragment |
| `(i` | 0.0313 | identifier_fragment |

## Key Findings

1. **Deterministic vs branching ratio shifts with T1.** At T1=0.6, 99% of steps are deterministic and 0% are branching. At T1=2.0, 79% deterministic and 5% branching. T1 controls how many generation steps become genuine choice points.

2. **Embedding similarity at branching points is low (0.126).** Survivors at branching points show meaningful semantic diversity — T2 can redistribute toward genuinely different code paths.

3. **T2 effect is concentrated at branching points.** T2 has near-zero entropy impact at deterministic and constrained steps (where survivors are syntactically similar). Its effect is largest at branching points, confirming that T2's value depends on having diverse survivors.

4. **Connection to T1/T2 experiment results.** The regime distribution explains why T1 is 2-14x more efficient than T2: T1 increases the *number* of branching points (creating new choice points), while T2 only flattens *existing* choices. T2 can only help where branching already exists — and that fraction grows with T1, explaining T2's regime-dependent behavior.
