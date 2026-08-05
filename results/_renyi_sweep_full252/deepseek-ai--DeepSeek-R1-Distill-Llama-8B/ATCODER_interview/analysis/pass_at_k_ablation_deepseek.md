# pass@k ablation — G_k vs p-less (α=2) baseline, DeepSeek-R1-Distill-Llama-8B (ATCODER-interview, n=10)

Baseline α=2 (=G_2): `results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl` (pass@1=0.392, of failing samples 68.8% are truncated loops).
Problem-level paired design (same tasks; samples are independent draws across configs, so individual-sample fates are NOT tracked — only per-problem pass rates).

## Summary across k

| k | pass@1 (Δ) | pass@10 (Δ) | win / lose / net | new-solve / lost-solve | McNemar p | Wilcoxon p | Δpass@1 95% CI | loop-escape share |
|---|---|---|---|---|---|---|---|---|
| 1.6 | 0.400 (+0.008) | 0.643 (+0.016) | 66/50/+16 | 14/10 | 0.54 | 0.28 | [-0.009,+0.024] | 62% |
| 0.8 | 0.435 (+0.042) | 0.687 (+0.060) | 83/44/+39 | 22/7 | 0.0081 | 1.3e-05 | [+0.023,+0.063] | 68% |
| 0.4 | 0.469 (+0.077) | 0.730 (+0.103) | 108/36/+72 | 28/2 | 8.7e-07 | 7.5e-10 | [+0.054,+0.101] | 77% |
| 0.2 | 0.463 (+0.071) | 0.730 (+0.103) | 100/36/+64 | 30/4 | 6.2e-06 | 2.7e-09 | [+0.049,+0.094] | 78% |
| 0.1 | 0.463 (+0.070) | 0.714 (+0.087) | 99/32/+67 | 26/4 | 5.9e-05 | 1.3e-08 | [+0.047,+0.094] | 72% |

## B+D. Difficulty strata — Δpass@1 / Δpass@10 (share of gross gain)

Buckets fixed by baseline pass@1; n constant across k. Cell = mean Δpass@1 / mean Δpass@10 (share%). Δpass@1 ≫ Δpass@10 within a bucket ⇒ reliability (fewer auto-fails), not new coverage.
| k | dead (0) n=94 | hard (0,0.3] n=44 | mid (0.3,0.7] n=43 | easy (0.7,1] n=71 |
|---|---|---|---|---|
| 1.6 | +0.026 / +0.149 (22%) | +0.045 / -0.227 (18%) | -0.021 / +0.000 (-8%) | -0.023 / +0.000 (-15%) |
| 0.8 | +0.047 / +0.234 (24%) | +0.141 / -0.159 (34%) | +0.067 / +0.000 (16%) | -0.039 / +0.000 (-16%) |
| 0.4 | +0.073 / +0.298 (27%) | +0.175 / -0.045 (30%) | +0.144 / +0.000 (24%) | -0.020 / +0.000 (-5%) |
| 0.2 | +0.072 / +0.319 (29%) | +0.193 / -0.091 (36%) | +0.098 / +0.000 (18%) | -0.023 / +0.000 (-7%) |
| 0.1 | +0.067 / +0.277 (26%) | +0.195 / -0.091 (36%) | +0.107 / +0.000 (19%) | -0.025 / +0.000 (-8%) |

## B. Migration matrix (baseline stratum → k stratum, problem counts)

Rows = where a problem sat at α=2; columns = where it sits at that k. Mass above the diagonal (toward *easy*) is upward migration; below is regression. The **dead→(non-dead)** cells are exactly the *newly-solvable* problems of decomposition E; **(non-dead)→dead** are solves lost.

**k = 1.6**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 80 | 14 | 0 | 0 |
| **hard (0,0.3]** | 10 | 21 | 13 | 0 |
| **mid (0.3,0.7]** | 0 | 7 | 28 | 8 |
| **easy (0.7,1]** | 0 | 0 | 8 | 63 |

**k = 0.8**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 72 | 20 | 1 | 1 |
| **hard (0,0.3]** | 7 | 14 | 23 | 0 |
| **mid (0.3,0.7]** | 0 | 6 | 22 | 15 |
| **easy (0.7,1]** | 0 | 1 | 9 | 61 |

**k = 0.4**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 66 | 24 | 2 | 2 |
| **hard (0,0.3]** | 2 | 25 | 13 | 4 |
| **mid (0.3,0.7]** | 0 | 4 | 17 | 22 |
| **easy (0.7,1]** | 0 | 0 | 11 | 60 |

**k = 0.2**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 64 | 23 | 6 | 1 |
| **hard (0,0.3]** | 4 | 20 | 15 | 5 |
| **mid (0.3,0.7]** | 0 | 2 | 26 | 15 |
| **easy (0.7,1]** | 0 | 0 | 10 | 61 |

**k = 0.1**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 68 | 22 | 2 | 2 |
| **hard (0,0.3]** | 4 | 16 | 19 | 5 |
| **mid (0.3,0.7]** | 0 | 4 | 21 | 18 |
| **easy (0.7,1]** | 0 | 0 | 11 | 60 |

## C+E. Attribution & decomposition

| k | Δpass@1 | loopy(n≥0.3 trunc) Δ | clean Δ | ρ(Δp1, base-trunc) | gain: loop-escape / reasoning | gain: newly-solvable / partial |
|---|---|---|---|---|---|---|
| 1.6 | +0.008 | +0.017 (n=144) | -0.005 | +0.08 | 62% / 38% | 22% / -5% |
| 0.8 | +0.042 | +0.066 (n=144) | +0.011 | +0.17 | 68% / 32% | 24% / 35% |
| 0.4 | +0.077 | +0.117 (n=144) | +0.024 | +0.18 | 77% / 23% | 27% / 49% |
| 0.2 | +0.071 | +0.117 (n=144) | +0.009 | +0.27 | 78% / 22% | 29% / 48% |
| 0.1 | +0.070 | +0.108 (n=144) | +0.019 | +0.22 | 72% / 28% | 26% / 48% |

## C×B. Is each stratum's gain due to loops? (baseline truncation + loop-escape share of gain)

Per baseline stratum, cell = **Δtrunc** (α=2 trunc% → k trunc%; how much looping actually fell) · **esc%** (share of that stratum's positive Δpass@1 from truncation-dominated-failure problems). **Δtrunc is the hard bound: loop-escape can explain at most |Δtrunc| of the pass rate.** If truncation barely fell in a stratum, its gain is NOT loops. esc% is a coarser problem-level heuristic (rounds a whole problem's gain to loop-escape when ≥50% of its α=2 failures were truncations).
| k | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| 1.6 | 75→70% (Δ-5) · esc 79% | 50→45% (Δ-5) · esc 65% | 24→25% (Δ+1) · esc 52% | 3→5% (Δ+2) · esc 50% |
| 0.8 | 75→52% (Δ-23) · esc 70% | 50→22% (Δ-28) · esc 74% | 24→11% (Δ-13) · esc 66% | 3→3% (Δ-0) · esc 38% |
| 0.4 | 75→0% (Δ-75) · esc 80% | 50→0% (Δ-50) · esc 80% | 24→0% (Δ-24) · esc 76% | 3→0% (Δ-3) · esc 59% |
| 0.2 | 75→0% (Δ-75) · esc 84% | 50→0% (Δ-50) · esc 76% | 24→0% (Δ-24) · esc 77% | 3→0% (Δ-3) · esc 67% |
| 0.1 | 75→0% (Δ-75) · esc 78% | 50→0% (Δ-50) · esc 74% | 24→0% (Δ-24) · esc 69% | 3→0% (Δ-3) · esc 58% |

## How to read (A–E)

- **A (summary win/lose + McNemar/Wilcoxon/CI)**: are there real losers, and is the paired net shift significant.
- **B (strata + migration)**: if gain concentrates in *dead/hard/mid* → loop-escape/coverage; in *easy* → Matthew (H4). The migration matrix shows which buckets move up (mid→easy) vs stay (dead→dead).
- **C (loop-escape share / ρ(Δp1, base-trunc))**: if Δpass@1 tracks how much a problem truncated at α=2, and most gain is from truncation-dominated problems, the win is loops escaping the token cap, not new reasoning (H1).
- **D (Δpass@1 ≫ Δpass@10, incl. per-stratum)**: reliability (fewer auto-fail draws), not coverage (new solutions) (H2).
- **E (newly-solvable vs partial→more)**: newly-solvable ⇔ dead→non-dead migration; partial→more ⇔ within/among the non-dead buckets. Dominant partial→more with tiny newly-solvable ⇒ consolidating borderline problems, not unlocking new ones.

Generated by `scripts/pass_at_k_ablation.py`. Every number recomputed live from metrics JSON + jsonl.
