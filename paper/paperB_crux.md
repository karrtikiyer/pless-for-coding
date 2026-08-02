# Paper 2 — crux, abstract, claims (for sign-off before we build)

Status: DRAFT for review. Numbers grounded in `docs/research/paperA_master_numbers.md` +
`paperA_loop_positioning.md` (this session). Framing constraints agreed with user:
α is a **power-sum / frequency-moment** lever (τ_α=Σpᵢ^α), **not** a Rényi-entropy generalization;
mechanism is **decoder-side** (model-side/RL cause of peakedness is cited, not claimed);
α **matches, does not beat**, tuned temperature; reproducibility is a methods appendix, not a claim.

## Title — placeholder (until greedy-decoder results land)
**When a Hyperparameter-Free Decoder Fails: Diagnosing and Repairing p-less Sampling Loops in
Reasoning-Model Code Generation**
(The earlier "Collapse to Greedy" hook was rejected: we have not benchmarked greedy on these models,
so we cannot headline a greedy-collapse claim. Body reframed to the provable "survivor set collapses
to the single most-probable token." Revisit the catchier title once a greedy baseline exists.)

## Crux (one sentence)
The hyperparameter-free p-less threshold **silently degenerates on long chain-of-thought** in
reasoning models — a collision-probability threshold collapses to greedy on peaked distributions,
driving 15–42% of traces into (largely paraphrastic) loops — and raising the α-power-sum exponent
removes the loop; α is a **diagnostic and repair lever, not a SOTA sampler** (it only matches tuned
temperature).

## Abstract (draft)
Hyperparameter-free decoders such as p-less sampling promise to remove temperature and top-p tuning by
deriving a truncation threshold — the collision probability Σᵢpᵢ² — directly from the token
distribution. We show this promise fails silently on reasoning models generating code. On APPS
competitive-programming problems with two 8B reasoning models (DeepSeek-R1-Distill-Llama, Qwen3),
default p-less drives 15–42% of chain-of-thought traces into non-terminating loops and yields the
worst pass@1 of any decoder we test. We trace this to a decoder-side mechanism: on the peaked
next-token distributions typical of long reasoning, the collision threshold approaches the top token's
probability, so the survivor set collapses to the mode and the model re-derives the same step — a
truncation sampler *amplifying* rather than curing repetition. A taxonomy of the loops shows roughly
half are paraphrastic (semantically redundant but lexically distinct; 41–47% of truncations) — these
lack the sustained periodicity that verbatim-loop and hidden-state-precursor detectors rely on, and a
published precursor does not transfer to this code setting. We then study a one-parameter family that
raises the power-sum exponent, τ_α(p)=Σᵢpᵢ^α (α=2 recovers p-less; a frequency-moment threshold,
*not* a Rényi-entropy generalization): raising α flattens the survivor set, monotonically eliminates
looping (truncation 42%→0.3%), recovers most lost accuracy, and roughly halves wasted tokens. The best
α **matches or beats each model's official recommended sampling settings** with zero tuning, and is
outperformed only by a hand-swept temperature configuration — the very tuning p-less exists to avoid.
α is thus a diagnostic and repair lever, not a new state-of-the-art sampler. Our contribution is a
mechanism for when and why hyperparameter-free decoding silently breaks on reasoning-model code
generation, a code-specific loop taxonomy, and the first empirical study of the α-power-sum lever.

## Claims (each falsifiable + grounded)
- **CL1 — Silent failure.** Default p-less (α=2) and p-less-norm loop catastrophically on
  reasoning-model CoT code: **41.8%** (DeepSeek) / **14.5%** (Qwen3) of traces never terminate →
  worst pass@1 on DeepSeek (**0.392** vs best temp 0.480), near-worst on Qwen. `[Certain]`
- **CL2 — Decoder-side mechanism.** Peakedness × hard threshold: on near-deterministic steps
  Σpᵢ²→max pᵢ, so only the mode survives → the model re-derives the same step. p-less amplifies
  repetition exactly where the model is confident — effectively decoding *near-greedily* on peaked
  steps, and Qwen's own docs warn that greedy decoding on Qwen3 causes "endless repetitions." (Why the
  model is peaked — RL-trained reasoning — is cited to Pipis et al., not claimed.) `[verifiable]`
- **CL3 — Code loop taxonomy (novel) — CORRECTED per review.** ~half of loops are **paraphrastic**
  (semantic drift): 41.3% (Qwen) / 46.8% (DeepSeek) vs 40.7% / 49.8% verbatim. **We do NOT claim
  n-gram detection is useless** — it is the basis of our own working rescue for the verbatim half, and
  in fact it *over*-fires (68 Qwen / 54 DeepSeek *completed-correct* traces also trip a 30-gram repeat
  but recover). The precise claim: paraphrastic loops lack the **sustained periodicity** that
  distinguishes a terminal loop from benign transient repetition, so they are missed by
  verbatim-*period* and hidden-state-*precursor* methods; the published precursor (Duan et al. 2026)
  does not transfer to code (~17–20% early detection vs 0.64–0.76 on synthetic verbatim loops;
  directional). Prior loop work is math/QA and detection-only. `[fractions Certain; precursor directional]`
- **CL4 — The α-power-sum lever, on three axes.** Raising τ_α=Σpᵢ^α (α=2 = p-less) monotonically
  (i) removes the loop (trunc 41.8→0.3% DeepSeek / 14.5→0.6% Qwen), (ii) recovers pass@1
  (0.392→0.483 / 0.625→0.696 @α4), (iii) roughly halves wasted thinking tokens (DeepSeek 17.3k→9.4k;
  Qwen 13.5k→11.1k), and (iv) partly restores diversity (cb\_div DeepSeek 0.489→0.553; Qwen
  0.453→0.474). No per-task tuning; prevention (high α up front) ≥ reactive detect-and-chop rescue. `[Certain]`
- **CL4b — Token-efficiency vs. other decoders, tracks the loop rate.** On the **high-loop model
  (DeepSeek)**, high-α pless is the **single most token-efficient decoder tested**: α4/α5 (9.2k/9.4k
  thinking tokens) use fewer tokens than *every* temperature/top-p/top-k config (9.6k–10.1k) at
  matched-or-better pass@1 — not a truncation artifact (α5 trunc 0.3%). On the **low-loop model
  (Qwen)** it is middle-of-the-pack, **tied** with temp/top-p/top-k (~11.0–11.3k). The efficiency edge
  is therefore loop-rate-dependent: the savings come from not spending budget on loops, so the more a
  model loops at baseline, the more high-α pless dominates other decoders on tokens. **Do NOT claim α
  is universally more token-efficient — Qwen is a tie.** `[Certain]`
- **CL5 — Honest ceiling + framing — STRENGTHENED per review.** The best pless-α **matches or beats
  each model's official recommended settings** with zero tuning: Qwen3 α=4 **0.696** > recommended
  T0.6/p0.95/k20 **0.680**; DeepSeek α=5 **0.483** > recommended T0.6/p0.95 **0.475**. It is
  outperformed only by a *hand-swept* temperature config (temp T1.0/p0.95: Qwen 0.705, DeepSeek 0.480
  ≈ tie) — the tuning p-less exists to avoid. Temperature retains a slight diversity edge (cb\_div
  ~0.556 vs α 0.553 DeepSeek). So α is diagnosis + repair, competitive out-of-the-box, not a decoding
  win; and τ_α is a **power-sum/frequency-moment** threshold that coincides with the origin paper's
  Rényi-entropy form only at α=2 — not a Rényi generalization. `[Certain]`

### Model-recommended settings (verified 2026-07-30) + the three-way comparison for CL5
- Qwen3-8B (thinking): **T=0.6, top_p=0.95, top_k=20** (Qwen HF card / docs; "do not use greedy →
  endless repetitions"). Our matching arm `temp p+k @T0.6` = **0.680**.
- DeepSeek-R1: **T=0.6, top_p=0.95** (DeepSeek API docs). Our matching arm `temp t0.6 (p0.95)` = **0.475**.
- Comparison per model: **pless-α (no tuning) ≥ recommended-temp (no tuning) < best-swept-temp (needs a sweep)**.

## What we explicitly do NOT claim (honesty guardrails)
- α does not beat temperature (CL5).
- α is not uniquely principled — top-p/top-k share the same diversity monotonicity; no optimality claim.
- We do not explain *why* RL reasoning models are peaked (cited, not ours).
- We do not solve loop detection/escape — paraphrastic loops remain hard (CL3).
- Not framed as Rényi entropy anywhere.

## Open questions for the user (decide before building)
1. Is the crux the right target claim, or should the **taxonomy (CL3)** be the headline instead of the
   failure→lever arc?
2. Two reasoning models + one benchmark (APPS) — enough scope, or add a second benchmark
   (LiveCodeBench / a math-CoT set) to show the mechanism generalizes? (Would need new runs.)
3. Keep CL4's prevention-vs-rescue inside this paper, or is it a distraction from the diagnosis?
4. Venue posture: this is an **analysis/diagnosis** paper (negative-result flavored, α ties temp).
   Comfortable with that as the ceiling, or do we need a "win" we don't currently have?
