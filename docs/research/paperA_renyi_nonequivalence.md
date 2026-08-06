# Paper A — the Rényi-order non-equivalence (self-contained, computed 2026-07-30)

**Purpose.** A manuscript-ready, self-contained observation (NOT a theorem, NOT a novelty claim) that
our raw power-sum family and the origin paper's rooted Rényi form are *distinct* filters for order > 2.
All numbers below were computed live (see the reproducing snippet at the end); none are from memory.

## Setup
Both are hyperparameter-free truncation filters: compute a threshold `T` from the full next-token
distribution `p`, admit tokens with `pᵢ ≥ T`, renormalize the survivors, sample.

- **Ours (τ_α):** `T = τ_α(p) = Σᵢ pᵢ^α` (raw α-th frequency moment). α = 2 is exactly p-less
  (collision entropy `Σpᵢ²`).
- **Origin paper (G_k), App. B.5 of Tan, Wu & Howard (2025), arXiv:2509.23234:**
  `T = G_k(p) = exp(−H_k(p)) = (Σᵢ pᵢ^k)^{1/(k−1)}`, the *rooted* Rényi form. The paper proposes this
  generalization but runs **no experiments at k ≠ 2** (verified by full-text fetch).

## Observation 1 — they coincide at order 2 (any distribution)
At order 2, `G_2 = (Σpᵢ²)^{1/1} = Σpᵢ² = τ_2` — both reduce to the p-less collision-entropy
threshold. Verified for `[.5,.3,.2]`, `[.9,.05,.05]`, `[.25,.25,.25,.25]` (all `τ_2 == G_2`).

## Observation 2 — opposite monotonicity in the order (peaked p = [0.7, 0.2, 0.1])
| order | τ_α (ours) | G_k (origin) |
|---:|---:|---:|
| 2 | 0.5400 | 0.5400 |
| 3 | 0.3520 | 0.5933 |
| 4 | 0.2418 | 0.6230 |
| 5 | 0.1684 | 0.6406 |

As the order rises, **τ_α decreases** (threshold drops → survivor set *loosens*, admits more tail
tokens → higher diversity) while **G_k increases toward `max pᵢ = 0.70`** (threshold rises → survivor
set *tightens* toward the mode → lower diversity). They move in **opposite directions**; they are not
order-preserving reparameterizations of one another.

## Observation 3 — filter non-equivalence (no order-matching aligns the admitted sets)
Distribution `p = [0.45, 0.25, 0.15, 0.10, 0.05]`, sweeping each family's order 2→8:

| family | admitted-set trajectory as order 2→8 | # distinct filters reachable |
|---|---|---|
| **τ_α (ours)** | {0} → {0,1} → {0,1,2} → {0,1,2,3} → {0,1,2,3,4} | **5** (mode-only up to full support) |
| **G_k (origin)** | {0} at *every* order | **1** (mode-only, always) |

Increasing the order lets τ_α admit progressively more of the tail (reaching full support by α≈4),
whereas G_k stays locked on the argmax at every order. The two families do not even generate the same
*set* of reachable filters on this distribution, so there is no monotone map α ↔ k that makes their
admitted sets agree for all `p`. **Non-equivalent for order > 2.**

## How to state it in the paper (framing guardrails)

> **SUPERSEDED for Paper B (2026-08-06).** After running the origin paper's own rooted `G_k` form on
> APPS (12-arm sweep, both reasoning models), Paper B (`paper/paperB`) was **repositioned to lead with
> G_k and drop τ_α entirely** (user decision). Paper B now claims the **first empirical evaluation** of
> the origin's `G_k` (App B.5 is theoretical-only — no k≠2 experiments; re-verified independently, one
> automated read fabricated a table, so confirm against the PDF before submission). The guardrails
> below reflect the earlier τ_α-primary framing and are retained for history / for any τ_α-based writeup;
> they do **not** govern Paper B. The non-equivalence result itself (below) is still correct and is why
> the two families are distinct — it is simply no longer featured in Paper B.

- Present as: *"At α = 2 our τ_α coincides with the collision-entropy p-less filter and with the
  origin paper's Rényi form; for α > 2 the two diverge — ours loosens, theirs tightens — so our
  α-sweep explores a filter family the origin paper's proposal does not reach."*
- **Do NOT** call it a theorem, and **do NOT** write "we are first to generalize p-less to higher
  Rényi orders" (the origin paper already proposed a Rényi generalization; our contribution is the
  *specific raw-moment form* and the *first empirical α > 2 sweep*, not the idea of generalizing).
- This resolves the internal contradiction previously in
  `docs/research/openreview_higher_moment_pless_comparison.md` (which wrongly called the two forms
  "order-preserving / same direction"). Corrected there.

## Reproduce
`uv run python` with: `tau(p,a)=Σp^a`, `G(p,k)=(Σp^k)**(1/(k-1))`, `admitted(p,T)={i: p_i≥T}`.
Full snippet in the session log (2026-07-30).
