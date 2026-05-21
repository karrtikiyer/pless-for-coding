# ICLR Diversity-Guarantee Theory: Majorization Theorem for α-Truncation

**Status (2026-05-21):** *Phase 0 — prior-art investigation complete; Phase 1
(proof attempt) not yet started; Phase 0.5 (brute-force counterexample
search) pending.*

This document captures the investigation, verified prior art, target theorem
statement, technical risks, and Phase-1 plan for a potential ICLR
theoretical contribution distinct from the C7 pass@k work. C7 yielded a
descriptive (not predictive) result; this is a separate, sharper question.

## Why a new direction

C7 work summary:
- C7 v1 naive iid factorization — **rejected** (54 pp error, see
  `c7_verdict.md`).
- C7 v3 beta-binomial framework — **succeeded as description**, but
  Step 5 (predicting `(a_α, b_α)` from per-position entropy features)
  was inconclusive (see `c7_step5_verdict.md`). And the headline ν(α)
  monotonicity, while real on 3 non-thinking models, was falsified
  on Qwen3-thinking — the regularity is narrower than first claimed.

C7 gives us a descriptive empirical framework, not a *predictive* or
*provable* claim. For ICLR theory-track work we need something
sharper.

**Pivot:** instead of trying to predict pass@k from sampler
hyperparameters, ask whether the α-sweep has a *provable diversity
guarantee*. Our empirical signal is strong: struct_div, cb_div, and
NAUADC are all monotone in α across all 4 models tested. If we can
prove this from first principles for the *sampler distribution*
itself, every downstream diversity functional follows as a corollary.

## The question

Can we prove that Rényi-α p-less truncation has a *universal*
diversity-monotonicity guarantee, holding for any source distribution
`p`, across every Schur-concave diversity functional simultaneously?

## Background — sampler definition

Hard-threshold operator `Trunc_α : p ↦ q_α`, where for a probability
distribution `p` over a finite alphabet `V`:

- threshold `T_α = Σ_i p_i^α` (the unnormalized α-norm of `p`)
- kept set `K_α = {i : p_i ≥ T_α}`
- `q_α(i) = p_i / Σ_{j ∈ K_α} p_j` if `i ∈ K_α`, else `0`

α=2 reproduces the original p-less sampler of Tan, Wu, Howard
([arXiv:2509.23234](https://arxiv.org/abs/2509.23234)).

## Target theorem (the thing we want to prove)

> **Theorem (target).** For any probability vector `p` over a finite
> alphabet `V` and any `1 ≤ α < α'`, the α'-truncated renormalized
> distribution `q_α'` (lifted to `V` with zeros on discarded tokens)
> is **majorized by** `q_α`:
>
> $$q_{\alpha'} \prec q_\alpha.$$
>
> i.e. higher α produces the **more spread out** (more diverse)
> distribution; the smaller-α distribution majorizes the larger-α one.
>
> Consequently, every Schur-concave diversity functional —
> Shannon entropy, Rényi-β entropy for any β ≥ 0, Gini index,
> effective support cardinality, collision probability — is
> non-decreasing in α along the α-truncation trajectory.

**Sign-convention note (added 2026-05-21):** the brute-force checker
in Phase 0.5 caught a direction error in an earlier draft of this
doc which had `q_α ≺ q_α'` instead of `q_α' ≺ q_α`. Standard
majorization convention is `x ≺ y` ⟺ `y` is more peaked (larger
sorted-descending partial sums) ⟺ `x` has more entropy. Worked
example: `p=[0.5, 0.3, 0.2]` gives `q_2=[1, 0, 0]` (partial sums
`[1,1,1]`) and `q_5=[0.5, 0.3, 0.2]` (partial sums `[0.5, 0.8, 1.0]`).
Since `q_5`'s partial sums are pointwise ≤ `q_2`'s, `q_5 ≺ q_2`,
confirming the corrected direction.

**Why this would be ICLR-grade:** *one* universal theorem implying
*all* the empirical monotonicity we've measured (struct_div, NAUADC,
cb_div across 4 models) as direct corollaries — not ad-hoc
observations. The proof technique (majorization on lifted vectors) is
standard machinery, but applying it to a *truncate-and-renormalize*
operator parameterized by α appears to be novel given prior art (see
"Phase 0 verification" below).

## Phase 0 verification — what's been checked

Two independent research agents ran prior-art surveys, with the second
agent specifically tasked to find any published result that would
*imply* or *refute* the target theorem. Findings below; every paper
citation has been verified by the agents via WebSearch/WebFetch.

### Phase 0a — Candidate landscape (first agent)

Evaluated six candidate theorem shapes for an ICLR diversity
contribution:

| Candidate | Shape | Verdict |
|---|---|---|
| 1 — Entropy lower bound on `q_α` via escort framework | `H_β(q_α) ≥ f(α, p)` for some β | 1-week theorem, fallback if majorization fails |
| 2 — Coverage guarantee on kept-set size | `\|K_α\| ≥ g(α, H_α(p))` | Trivial corollary; folds in, not standalone |
| 3 — Diversity monotonic in α | `H_2(q_α)` non-decreasing in α | Strongest candidate; ~60% confidence true |
| 4 — Anti-concentration cover@t bound | `P(all k samples within ε) ≤ g(α, k, ε)` | **Drop** — no traction in literature |
| 5 — Rate-distortion / variational characterization | `q_α = argmin` of some regularized objective | Largely subsumed by Ji et al. 2026 ([arXiv:2602.18292](https://arxiv.org/abs/2602.18292)) |
| 6 — Majorization framework (Hardy-Littlewood-Polya) | `q_α ≺ q_α'` (the universal version of #3) | Strongest possible form; combine with #3 |

**First agent recommendation:** pursue candidate 3+6 jointly. ~50% chance
of headline success (60% true × 80% provable in 1 week).

Key prior-art references found:

- **Ji, Tutunov, Zimmer, Bou-Ammar 2026** ([arXiv:2602.18292](https://arxiv.org/abs/2602.18292)) — verified verbatim: *"This single template recovers greedy decoding, Softmax sampling, Top-K, Top-P, and Sparsemax-style sparsity as special cases"*. **Does not cover α-p-less.** Sets the variational-characterization prior-art boundary.
- **Bercher 2009** ([arXiv:1109.3385](https://arxiv.org/abs/1109.3385)) — escort distributions and Rényi entropy bounds. Uses Karamata-on-escort technique.
- **Linden, Mosonyi, Winter** ([arXiv:1212.0248](https://arxiv.org/abs/1212.0248)) — structure of Rényi entropic inequalities.
- **Wu, Yu, Guo** ([arXiv:2312.01819](https://arxiv.org/abs/2312.01819)) — complete monotonicity of Rényi entropy.
- **Tan, Wu, Howard 2025** ([arXiv:2509.23234](https://arxiv.org/abs/2509.23234)) — upstream p-less paper. App. B.5 mentions α-generalization but **no majorization theorem**.

### Phase 0b — Deep-dive on escort & truncation literature (second agent)

Specifically asked: has any paper proved or refuted the target?

**Section A — Existing results that would IMPLY our target (would kill novelty):** **None found.**

- Yadav & Shkel ([arXiv:2605.09655](https://arxiv.org/abs/2605.09655)) studies the *lattice* of majorization at *fixed* α, not a parametric family `q_α`. Does not imply our target.
- Sason & Verdú ([arXiv:1812.03324](https://arxiv.org/abs/1812.03324)) bounds Rényi entropy gap for fixed `p`. Does not transport.
- The well-known fact `H_α(p)` is anti-monotone in α holds for *fixed* `p`; does not transport to a moving distribution.

**Section B — Existing results that REFUTE our target (would kill viability):** **None found.**

- No counterexample, no impossibility result in any of the escort, Tsallis, Rényi-monotonicity, top-K/top-P, or smoothing-inequality literatures.
- Lesche-stability counterexample ([arXiv:0903.4169](https://arxiv.org/abs/0903.4169)) is unrelated (continuous Boltzmann).

**Section C — Adjacent results (close but not equivalent):**

1. **Bercher 2009** — works with the *soft escort* `e_α(i) = p_i^α / Σ p_j^α`, not a hard-threshold operator. The normalizer is `Σ p_j^α` — coincidentally equal to our threshold `T_α`, a suggestive structural link. Closable in 1 week if a proof bridges hard truncation to the escort.
2. **Top-H Decoding** ([arXiv:2509.02510](https://arxiv.org/abs/2509.02510)) Theorem 3, verified: *"the entropy of `q` increases strictly at each selection step and is maximized only when all tokens are selected."* Adjacent stepwise-entropy lemma; different operator (ECMM, not α-threshold). Probably not directly adaptable.
3. **Power majorization** ([arXiv:1210.6630](https://arxiv.org/abs/1210.6630), Brandão & Plenio) — definition involves `x^α - y^α` integrals. Suggestive but no direct theorem found.

**Section D — Methodology hints:**

- **Karamata's inequality / Schur-Ostrowski criterion** — standard route to prove `q_α ≺ q_α'`: show sorted partial sums `S_k(q_α) = Σ_{i=1..k} q_α^↓(i)` are non-increasing in α for all `k`.
- **Two-regime decomposition** (the genuine technical novelty — **not in any paper**):
  1. **Smooth regions:** α-intervals where `K_α` is locally constant. Here `q_α(i) ∝ p_i` (constants of proportionality fixed); the renormalizer `Σ_{j∈K_α} p_j` is independent of α within the region. Partial-sum monotonicity reduces to elementary calculus.
  2. **Drop events:** specific α values where a new token enters `K_α` (its `p_i` becomes ≥ the decreasing `T_α`). The set discontinuously grows; the renormalizer jumps. **This is where standard Karamata proofs typically break.**
- **Top-H stepwise lemma** as template for handling drop events.

**Section E — Honest verdict (second agent):**

> "Given (a) no novelty kill, (b) no viability kill, (c) two adjacent-but-not-identical proof techniques exist, and (d) the discrete drop-event structure of `Trunc_α` is the genuine technical novelty that no prior work has handled — the prior agent's ~50% estimate looks roughly right. I would not raise it above ~55%: the drop events are exactly where Karamata-style proofs typically break, and no source I found has handled an analogous discontinuity."
>
> **proof attempt YES**

## The technical risk in one sentence

**Drop events** — discontinuous expansions of the kept set `K_α` as α
grows — are the genuine technical novelty and the genuine technical
risk. Karamata-style proofs on smooth deformations are well-trodden;
proofs through a discontinuity in the operator's structure are not.
If the drop-event step admits a clean handling, the proof is ≤ 2
pages and could be done in a day. If not, the proof either becomes a
multi-week problem or is genuinely false.

## Phase 1 plan — 5 working days, hard cap

| Day | Task | Output | Acceptance criterion |
|---|---|---|---|
| **0.5** (pre) | Brute-force counterexample checker on small `V` (3–8 tokens) × 10⁴ random `p` × dense α grid | Pass/fail per random `p`; if any fail, the specific `p` + α pair | If ANY counterexample → theorem dead, drop to Candidate 1 (escort entropy bound). If none after 10⁵ trials → proceed to Day 1. |
| 1 (morning) | Smooth-region argument — locally constant `K_α`, partial-sum monotonicity via elementary calculus on the renormalizer | Subsection draft (1–2 pages) | Proof must hold for any locally-constant region; verify with synthetic example. |
| 1 (afternoon) — 3 | Drop-event argument — handle discontinuities. Two routes to try in parallel:<br>• Karamata + careful boundary book-keeping<br>• Stepwise lemma in spirit of Top-H Theorem 3 | Either clean proof or identification of why both routes fail | Drop-event step must preserve partial-sum inequalities. |
| 4 | If proof works: write up theorem + corollaries (Shannon, Rényi-β, Gini, effective support, collision probability, struct_div, NAUADC) | Proof note (~3 pages) | Corollaries follow by Schur-concavity from theorem; spot-check at least 3 functionals. |
| 5 | If proof works: empirical verification on existing α=2 entropy sidecars (CodeLlama + Qwen2.5 MBPP, 295k+ positions each). Compute `H_β(q_α)` for β ∈ {1, 2, ∞} and confirm monotonicity at every position. | Verification figure + counterexample-rate (should be 0) | If ≥ 1 position violates monotonicity, the proof has a bug — return to Day 2. |

**Hard cap: 5 working days.** If by end of Day 3 no proof and no
counterexample, stop. Fall back to Candidate 1 (escort entropy bound)
as a workshop-grade contribution.

## Acceptance criteria summary

| Outcome | Decision |
|---|---|
| Brute-force counterexample found (Day 0.5) | Drop majorization; pursue Candidate 1 (escort entropy bound) as workshop fallback |
| Brute-force clean (no counterexample on 10⁵ trials) | Phase 1 commit; prior on theorem rises to ~80%+ |
| Proof complete by Day 5 | ICLR theory-track candidate. Combine with C7 v3 empirical ν(α) (where applicable) and bimodal entropy as supporting evidence. |
| Proof stuck at drop-event step by Day 3 | Cap and pivot to Candidate 1 |
| Proof complete but empirical verification fails (Day 5) | Treat as proof bug; debug or pivot |

## What this would enable for the paper

If theorem proves:

1. **Universal corollary** for every Schur-concave diversity functional:
   our empirical struct_div / cb_div / NAUADC monotonicity claims become
   theoretical predictions, not just data.
2. **Pareto-style framing:** decoding choice between α values is on a
   provable diversity-monotone curve; no diversity is lost by going to
   higher α (subject to the saturation reading where mean p effects
   dominate, per the Qwen3-thinking observation in
   `results/pless_alpha_full/Qwen--Qwen3-8B/full_sweep_summary.md`).
3. **Distinction from related work:**
   - Ji et al. ([arXiv:2602.18292](https://arxiv.org/abs/2602.18292)) provides a *variational characterization* of decoding rules; we provide a *parametric monotonicity guarantee* — orthogonal directions.
   - Top-H decoding ([arXiv:2509.02510](https://arxiv.org/abs/2509.02510)) has a stepwise entropy increase; we have a continuous-parameter majorization.
   - Tan et al. p-less ([arXiv:2509.23234](https://arxiv.org/abs/2509.23234)) provides α=2 (collision probability); we provide the full α-parametric guarantee.

## Open questions parked for now

- Whether the empirical Qwen3-thinking regime (mean p climbs, ν flat
  or shrinks) interacts with this theorem. Hypothesis: the theorem is
  about the *sampler-induced distribution* `q_α`, not the per-task
  pass-rate distribution. The mean-p effect on thinking models is a
  *downstream-of-sampling* phenomenon. So the theorem should still
  hold; what changes for thinking models is the mapping from
  sampler-diversity to pass@k-distribution-shape. Worth checking after
  the proof.
- Whether tensor-parallel vLLM affects sampling determinism enough to
  void our empirical verification on real sidecars. Probably not —
  our empirical claims are about distributions over random draws, not
  bit-identical sequences.

## Status log

| Date | Step | Outcome |
|---|---|---|
| 2026-05-21 | Phase 0a — candidate landscape survey | Complete; majorization (Candidate 3+6) recommended at ~50% confidence |
| 2026-05-21 | Phase 0b — deep-dive escort/truncation literature | Complete; no kill-shot found, confidence revised to ~55% |
| 2026-05-21 | Phase 0.5 — brute-force counterexample checker (`bench/eval/check_majorization_trunc_alpha.py`) | **Sign error caught + corrected**. With the corrected statement `q_α' ≺ q_α` for α<α': **0 counterexamples in ~1.235M non-degenerate trials** over V ∈ {3..15} × Dirichlet(1,..,1) random `p` × 28 α-pairs from {1.1, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 50.0}. Min-slack quantiles indicate the inequality is not tight in most cases. Confidence revised **upward to ~80–85%**. |
| 2026-05-21 | Phase 0.6 — uniqueness check (`bench/eval/check_majorization_topp_topk.py`) | **Top-p and top-k ALSO satisfy the same monotonicity.** 0 counterexamples in 3.15M top-p trials + 600k top-k trials (slacks within ±8e-16 FP noise). The Phase-1 theorem for α is NOT uniquely α's — it's a property shared by the class of "truncate-and-renormalize" operators. Paper positioning must shift from "uniquely principled" to "first proved member of a conjectured class." See "Paper-positioning update" below. |
| Pending | Phase 1 — proof attempt or counterexample documentation | Ready to launch; budget 5 working days. |

### Phase 0.5 detail

The agent caught a sign error in the originally-drafted theorem statement
(literal `q_α ≺ q_α'` was reverse of the intended semantic claim).
After correcting to `q_α' ≺ q_α`:

| Direction | Outcome |
|---|---|
| `q_α ≺ q_α'` for α<α' (**original draft — wrong direction**) | 911,596 counterexamples (theorem dies) |
| `q_α' ≺ q_α` for α<α' (**corrected**) | **0 counterexamples in 1,235,063 non-degenerate trials**; remaining "violations" all within ±9e-16 FP noise |

Clean counterexample for the wrong direction: `p=[0.4, 0.4, 0.1, 0.1]`,
`α=2.0`, `α'=5.0` gives `q_2=[0.5, 0.5, 0, 0]` and `q_5=[0.4, 0.4, 0.1, 0.1]`.
At k=0, sorted partial sums: 0.5 (for q_2) > 0.4 (for q_5). So q_2 is
more peaked; **q_5 ≺ q_2**, not q_2 ≺ q_5.

This is the meta-lesson: even when stating a "well-known" framework like
majorization, the direction convention matters and is easy to flip in
writing. The Phase 0.5 checker did exactly its job — catch the bug at
the cheap-test level before the week-long proof investment.

### Acceptance verdict

The Phase-0.5 acceptance criterion from the plan was:

> If ANY counterexample → theorem dead, drop to Candidate 1 (escort
> entropy bound). If none after 10⁵ trials → proceed to Day 1.

With the corrected statement: **PHASE-1 PROCEED**. The literal target
holds across V ∈ {3..15} and a wide α-grid with effectively zero
counterexamples (only FP-noise edge cases). Prior on the theorem
being true has risen from ~60% (pre-checker) to **~80–85%** (post-checker).

### Phase 0.6 — uniqueness check (added 2026-05-21)

After Phase 0.5 succeeded, asked: is the conjectured monotonicity
UNIQUE to α-truncation, or do top-p and top-k also satisfy it?
This determines whether the paper's theoretical contribution is
"unique guarantee" or "first proved member of a class."

Brute-force checker at `bench/eval/check_majorization_topp_topk.py`:

| Operator | Trials | Counterexamples | Min slack (passing) |
|---|---:|---:|---|
| α-truncation (`Trunc_α`) | 1,235,063 | 0 | within ±FP noise |
| Top-p (`Trunc_topP`) | 3,150,000 | **0** | within ±8e-16 FP noise |
| Top-k (`Trunc_topK`) | 600,000 | **0** | within ±8e-16 FP noise |

**All three operators empirically satisfy the same majorization
monotonicity.** The Phase-1 theorem, if proved for α, would not be
uniquely α's — it would be the first *proved* instance of a property
that empirically holds for the whole "truncate-and-renormalize" class.

### Paper-positioning update (2026-05-21)

What the contribution **can** claim:

1. **α-truncation has a data-dependent threshold** (`T_α = Σpᵢ^α`,
   intrinsic to the input distribution). Top-p uses an external
   cumulative-mass target `p`; top-k uses an external count `k`.
   This is a genuine qualitative difference.
2. **α=2 is a hyperparameter-free starting point** — the original
   p-less sampler of Tan et al. ([arXiv:2509.23234](https://arxiv.org/abs/2509.23234))
   based on collision probability. No analogous canonical "zero" exists
   for top-p or top-k.
3. **Phase-1 theorem (if proved):** the first proved majorization
   monotonicity for any truncate-and-renormalize sampler. Top-p and
   top-k satisfy it empirically but no theorem is published; this paper
   would be the first proved instance.

What the contribution **cannot** claim:

- "α is uniquely principled" — empirically false; top-p and top-k
  satisfy the same monotonicity.
- "α reaches operating points other samplers can't" — empirically
  false; the `bench/eval/sampler_comparison.py` output shows existing
  stochastic samplers (especially `pless@T=2.0`, `top_p0.9`, `top_k5`)
  reach Pareto-equivalent or Pareto-dominating points on
  (pass@10, cb_div) across the 3 instruct models × 2 datasets.

The honest empirical+theoretical paper claim:

> The α-truncation family parameterizes a diversity-quality trade-off
> curve with a **provable monotonic-diversity guarantee** via the
> Rényi-α majorization theorem (Phase-1). Empirically, the same
> property appears to hold for top-p and top-k truncation
> (3.15M + 600k trials, zero counterexamples), suggesting a broader
> class of *monotone-diversity truncation operators*. α-truncation
> is the **first proved member** of this conjectured class and the
> only one with a *data-dependent* threshold (intrinsic to the input
> distribution), corresponding at α=2 to the hyperparameter-free
> p-less baseline.

Empirically, on 4 models × 2 datasets, α-arms reach a Pareto
frontier comparable to (and on MBPP, slightly *within*) the frontier
achievable by tuned top-p / top-k / pless@high-T samplers. The
contribution is principled-parameterization, not new operating
points.

## References

Verified via WebSearch/WebFetch by the two prior-art agents:

- [arXiv:2509.23234 — p-less Sampling (Tan, Wu, Howard 2025)](https://arxiv.org/abs/2509.23234) — upstream paper, α=2 only.
- [arXiv:2602.18292 — Decoding as Optimisation on the Probability Simplex (Ji et al. 2026)](https://arxiv.org/abs/2602.18292) — variational characterization of top-K, top-P, sparsemax, Best-of-K; does not cover α-p-less.
- [arXiv:1109.3385 — Source Coding with Escort Distributions (Bercher 2009)](https://arxiv.org/abs/1109.3385) — escort framework.
- [arXiv:1212.0248 — Structure of Rényi Entropic Inequalities (Linden, Mosonyi, Winter)](https://arxiv.org/abs/1212.0248)
- [arXiv:2312.01819 — Complete Monotonicity of Rényi Entropy (Wu, Yu, Guo)](https://arxiv.org/abs/2312.01819)
- [arXiv:2605.09655 — Geometry of Rényi Entropy on Majorization Lattice (Yadav & Shkel)](https://arxiv.org/abs/2605.09655)
- [arXiv:1812.03324 — Tight Bounds on Rényi Entropy via Majorization (Sason & Verdú)](https://arxiv.org/abs/1812.03324)
- [arXiv:1812.02004 — Differential-Escort Transformations and LMC-Rényi Monotonicity](https://arxiv.org/abs/1812.02004)
- [arXiv:2509.02510 — Top-H Decoding](https://arxiv.org/abs/2509.02510)
- [arXiv:1605.00019 — Sharp Bounds Between Two Rényi Entropies of Distinct Orders (Sason)](https://arxiv.org/abs/1605.00019)
- [arXiv:1210.6630 — Trumping and Power Majorization](https://arxiv.org/abs/1210.6630)
- [arXiv:1206.5127 — Escort Distributions and Tsallis Entropy](https://arxiv.org/abs/1206.5127)
- [arXiv:2502.10295 — Fenchel-Young Variational Inference](https://arxiv.org/abs/2502.10295)

**Marshall-Olkin, *Inequalities: Theory of Majorization and Its
Applications* (2011, 2nd ed.)** — textbook reference for majorization
machinery. Not on arXiv; standard library reference.
