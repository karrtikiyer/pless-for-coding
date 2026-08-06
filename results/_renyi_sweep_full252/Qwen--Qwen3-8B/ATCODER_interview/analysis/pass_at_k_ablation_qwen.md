# pass@k ablation — G_k vs p-less (α=2) baseline, Qwen3-8B (ATCODER-interview, n=10)

Baseline α=2 (=G_2): `results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl` (pass@1=0.625, of failing samples 38.8% are truncated loops).
Problem-level paired design (same tasks; samples are independent draws across configs, so individual-sample fates are NOT tracked — only per-problem pass rates).

## Summary across k

Column notes: **cov-McNemar p** tests only the *coverage-status* change (solve-at-least-once: new-solve vs lost-solve counts) — it does NOT test the win/lose ledger beside it. The significance of the net per-problem **pass@1** shift is the **Wilcoxon p** column. **loop-escape (esc%)** is the coarse problem-level heuristic (whole-problem gain → loops if ≥50% of its α=2 failures were truncations); the rigorous upper bound is Δtrunc in the C×B section. 

Caveats: Wilcoxon and the bootstrap CI treat each problem's pass@1 as a noiseless point estimate (they resample the 252 problems, not the 10 within-problem draws), so p-values / CIs are mildly *anticonservative* — immaterial for the p≈1e-9…1e-13 arms, relevant near k=1.6. No multiple-comparisons correction is applied across the 6 k arms (the surviving effects are orders of magnitude below any correction threshold). Arms are unpaired, so differences *between* k arms are not significance-tested.

| k | pass@1 (Δ) | pass@10 (Δ) | win / lose / net | new-solve / lost-solve | cov-McNemar p | Wilcoxon p | Δpass@1 95% CI | loop-escape (esc%) |
|---|---|---|---|---|---|---|---|---|
| 1.6 | 0.627 (+0.001) | 0.813 (-0.012) | 62/53/+9 | 4/7 | 0.55 | 0.67 | [-0.015,+0.018] | 43% |
| 0.8 | 0.669 (+0.044) | 0.837 (+0.012) | 83/37/+46 | 7/4 | 0.55 | 2.6e-06 | [+0.027,+0.062] | 51% |
| 0.4 | 0.696 (+0.070) | 0.833 (+0.008) | 89/30/+59 | 7/5 | 0.77 | 1.6e-09 | [+0.050,+0.092] | 53% |
| 0.2 | 0.701 (+0.076) | 0.833 (+0.008) | 95/25/+70 | 7/5 | 0.77 | 2.1e-09 | [+0.053,+0.100] | 58% |
| 0.1 | 0.719 (+0.094) | 0.845 (+0.020) | 105/19/+86 | 8/3 | 0.23 | 1.6e-13 | [+0.071,+0.117] | 49% |
| 0.05 | 0.717 (+0.091) | 0.833 (+0.008) | 99/22/+77 | 6/4 | 0.75 | 2.7e-13 | [+0.069,+0.115] | 50% |

## B+D. Difficulty strata — Δpass@1 / Δpass@10 (net Δ contribution)

Buckets fixed by baseline pass@1; n constant across k. Cell = mean Δpass@1 / mean Δpass@10 (contrib%). **contrib%** = that stratum's *net* Δpass@1 as a fraction of the *gross winner* gain (Σ of positive per-problem Δ) — so a net-losing stratum shows a **negative** contrib%, and the columns sum to <100% by the total loss fraction (not an error). Δpass@1 ≫ Δpass@10 within a bucket ⇒ reliability (fewer auto-fails), not new coverage.
| k | dead (0) n=44 | hard (0,0.3] n=28 | mid (0.3,0.7] n=47 | easy (0.7,1] n=133 |
|---|---|---|---|---|
| 1.6 | +0.011 / +0.091 (5%) | +0.032 / -0.214 (9%) | +0.026 / -0.021 (12%) | -0.017 / +0.000 (-23%) |
| 0.8 | +0.030 / +0.159 (8%) | +0.118 / -0.143 (20%) | +0.113 / +0.000 (33%) | +0.009 / +0.000 (7%) |
| 0.4 | +0.032 / +0.159 (6%) | +0.186 / -0.179 (23%) | +0.204 / +0.000 (43%) | +0.011 / +0.000 (7%) |
| 0.2 | +0.025 / +0.159 (5%) | +0.204 / -0.179 (24%) | +0.236 / +0.000 (46%) | +0.009 / +0.000 (5%) |
| 0.1 | +0.043 / +0.182 (7%) | +0.221 / -0.107 (23%) | +0.257 / +0.000 (45%) | +0.026 / +0.000 (13%) |
| 0.05 | +0.041 / +0.136 (7%) | +0.236 / -0.143 (25%) | +0.255 / +0.000 (46%) | +0.020 / +0.000 (10%) |

## B. Migration matrix (baseline stratum → k stratum, problem counts)

Rows = where a problem sat at α=2; columns = where it sits at that k. Mass above the diagonal (toward *easy*) is upward migration; below is regression. The **dead→(non-dead)** cells are exactly the *newly-solvable* problems of decomposition E; **(non-dead)→dead** are solves lost.

**k = 1.6**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 40 | 4 | 0 | 0 |
| **hard (0,0.3]** | 6 | 17 | 5 | 0 |
| **mid (0.3,0.7]** | 1 | 7 | 25 | 14 |
| **easy (0.7,1]** | 0 | 1 | 9 | 123 |

**k = 0.8**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 37 | 6 | 1 | 0 |
| **hard (0,0.3]** | 4 | 15 | 7 | 2 |
| **mid (0.3,0.7]** | 0 | 3 | 24 | 20 |
| **easy (0.7,1]** | 0 | 0 | 5 | 128 |

**k = 0.4**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 37 | 6 | 1 | 0 |
| **hard (0,0.3]** | 5 | 10 | 9 | 4 |
| **mid (0.3,0.7]** | 0 | 4 | 14 | 29 |
| **easy (0.7,1]** | 0 | 0 | 6 | 127 |

**k = 0.2**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 37 | 7 | 0 | 0 |
| **hard (0,0.3]** | 5 | 12 | 5 | 6 |
| **mid (0.3,0.7]** | 0 | 2 | 15 | 30 |
| **easy (0.7,1]** | 0 | 0 | 7 | 126 |

**k = 0.1**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 36 | 7 | 0 | 1 |
| **hard (0,0.3]** | 3 | 10 | 10 | 5 |
| **mid (0.3,0.7]** | 0 | 4 | 12 | 31 |
| **easy (0.7,1]** | 0 | 0 | 2 | 131 |

**k = 0.05**

| baseline ↓ / k → | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| **dead (0)** | 38 | 4 | 1 | 1 |
| **hard (0,0.3]** | 4 | 9 | 10 | 5 |
| **mid (0.3,0.7]** | 0 | 1 | 13 | 33 |
| **easy (0.7,1]** | 0 | 0 | 3 | 130 |

## C+E. Attribution & decomposition

| k | Δpass@1 | loopy(n≥0.3 trunc) Δ | clean Δ | ρ(Δp1, base-trunc) | gain: loop-escape / reasoning | gain: newly-solvable / partial |
|---|---|---|---|---|---|---|
| 1.6 | +0.001 | +0.026 (n=53) | -0.006 | +0.16 | 43% / 57% | 5% / -2% |
| 0.8 | +0.044 | +0.108 (n=53) | +0.027 | +0.33 | 51% / 49% | 8% / 60% |
| 0.4 | +0.070 | +0.168 (n=53) | +0.044 | +0.43 | 53% / 47% | 6% / 73% |
| 0.2 | +0.076 | +0.192 (n=53) | +0.045 | +0.44 | 58% / 42% | 5% / 75% |
| 0.1 | +0.094 | +0.194 (n=53) | +0.067 | +0.39 | 49% / 51% | 7% / 81% |
| 0.05 | +0.091 | +0.196 (n=53) | +0.063 | +0.39 | 50% / 50% | 7% / 82% |

## C×B. Is each stratum's gain due to loops? (baseline truncation + loop-escape share of gain)

Per baseline stratum, cell = **Δtrunc** (α=2 trunc% → k trunc%; how much looping actually fell) · **esc%** (share of that stratum's positive Δpass@1 from truncation-dominated-failure problems). **Δtrunc bounds the loop contribution: loop-escape can lift pass@1 by at most |Δtrunc|** — under the premise that a truncated sample always fails (true: an unclosed thinking phase yields no gradable answer) and given truncation only *falls* with looser k here (Δtrunc ≤ 0 in every stratum, so the aggregate rate change equals the max rescuable mass). If truncation barely fell in a stratum, its gain is NOT loops. esc% is a coarser problem-level heuristic (rounds a whole problem's gain to loop-escape when ≥50% of its α=2 failures were truncations) and tends to *over*-credit loops.
| k | dead (0) | hard (0,0.3] | mid (0.3,0.7] | easy (0.7,1] |
|---|---|---|---|---|
| 1.6 | 36→35% (Δ-2) · esc 80% | 29→28% (Δ-1) · esc 5% | 20→19% (Δ-1) · esc 44% | 2→2% (Δ-0) · esc 60% |
| 0.8 | 36→22% (Δ-15) · esc 38% | 29→17% (Δ-12) · esc 24% | 20→10% (Δ-9) · esc 59% | 2→1% (Δ-2) · esc 69% |
| 0.4 | 36→0% (Δ-36) · esc 50% | 29→2% (Δ-27) · esc 40% | 20→0% (Δ-19) · esc 55% | 2→0% (Δ-2) · esc 71% |
| 0.2 | 36→0% (Δ-36) · esc 55% | 29→0% (Δ-29) · esc 43% | 20→0% (Δ-19) · esc 61% | 2→0% (Δ-2) · esc 73% |
| 0.1 | 36→0% (Δ-36) · esc 32% | 29→0% (Δ-29) · esc 39% | 20→0% (Δ-20) · esc 52% | 2→0% (Δ-2) · esc 62% |
| 0.05 | 36→0% (Δ-36) · esc 39% | 29→0% (Δ-29) · esc 36% | 20→0% (Δ-19) · esc 52% | 2→0% (Δ-2) · esc 68% |

## How to read (A–E)

- **A (summary)**: win/lose/net shows how many problems improved vs regressed; the **net pass@1 shift's significance is Wilcoxon** (not cov-McNemar, which only tests the new-solve vs lost-solve coverage change). Bootstrap CI is over problems (not draws) — see the summary caveats.
- **B (strata + migration)**: if gain concentrates in *dead/hard/mid* → loop-escape/coverage; in *easy* → Matthew (H4). The migration matrix shows which buckets move up (mid→easy) vs stay (dead→dead).
- **C (loop-escape share / ρ(Δp1, base-trunc))**: if Δpass@1 tracks how much a problem truncated at α=2, and most gain is from truncation-dominated problems, the win is loops escaping the token cap, not new reasoning (H1).
- **D (Δpass@1 ≫ Δpass@10, incl. per-stratum)**: reliability (fewer auto-fail draws), not coverage (new solutions) (H2).
- **E (newly-solvable vs partial→more)**: newly-solvable ⇔ dead→non-dead migration; partial→more ⇔ within/among the non-dead buckets. Dominant partial→more with tiny newly-solvable ⇒ consolidating borderline problems, not unlocking new ones.

Generated by `scripts/pass_at_k_ablation.py`. Every number recomputed live from metrics JSON + jsonl.
