# Rambling / looping in reasoning models: the decoding-sampler class + related work

Synthesis of the conceptual + literature findings from the June 2026 discussions. The
*experimental* results live elsewhere (`docs/decoder_comparison_cot_apps_qwen3.md`,
`docs/loopforce_w1200_comparison_apps_qwen3.md`, `docs/loopforce_deepseek_smoke_findings.md`,
todos A31–A37). This note records (1) the mechanism, (2) which decoding methods share the
failure and why, and (3) where our work sits in the literature.

**Citation discipline:** every arXiv claim below is tagged `[verified]` (fetched or
search-confirmed this session) or `[unverified]` (surfaced but not independently read —
do NOT cite as fact without fetching). Two of our own earlier inferences were wrong and are
marked as corrected.

---

## 1. The mechanism (grounded in our experiments)

The think-phase loop is a **peakedness × hard-threshold interaction**, not a temperature
problem per se:

- **Controlled A/B (same T=0.6, only the pless mask differs):** plain temperature truncates
  **0.4%**, pless truncates **19%** (Qwen3-8B, APPS ATCODER-interview 252). Source:
  `docs/loopforce_w1200_comparison_apps_qwen3.md` peakedness section.
- **Monotonic in temperature:** pless truncation T0.6 19% → T1.0 14.5% → T1.5 8% → T2.0 0.2%
  (todos A34). Low T sharpens the distribution → collision entropy `Σpᵢ²` rises → pless
  keeps ~1–2 tokens → near-greedy → locks into a **low-entropy loop attractor**; plain temp
  keeps the tail as an escape route.

So a reasoning loop is a *confident* (low-entropy) state, and any truncator that **tightens
as the model gets confident** removes the escape route exactly when it's needed.

---

## 2. The decoding-sampler class — who loops and why

**Discriminator:** does the truncation threshold *collapse onto the mode at a peaked
position* (→ loops) or is it *capped / temperature-invariant / mode-avoiding* (→ resists)?

| Method | Threshold | Behaviour at a peaked loop | Class |
|---|---|---|---|
| greedy / beam | argmax | always the mode | **loops** (canonical) `[verified: Holtzman]` |
| **pless** | keep `p ≥ Σpᵢ²` | `Σpᵢ²→p_max` → collapses to mode **earliest** | **loops (worst case)** |
| **min-p** | keep `p ≥ p_base·p_max` | rises with `p_max`, but ~7× gentler than pless | **loops (milder)** `[verified: 2407.01082]` |
| Rényi `G_k` | keep `p ≥ exp(−H_k)` | `G_2 = Σpᵢ²` *is* pless; tightens with peak | **loops (same family)** |
| **η-sampling** | keep `p ≥ min(ε, α·e^−H)` | threshold **capped at ε** → keeps the tail | **resists** `[verified: 2210.15191]` |
| **top-nσ** | keep `logit ≥ M−n·σ` | **temperature-invariant** nucleus | **resists (T-stable)** `[verified: 2411.07641]` |
| top-p | smallest set with mass ≥ p | mass-anchored, not peak-scaled | resists (mostly) |
| temperature (T≳1) | none (reweight only) | whole tail kept | resists |
| **top-H** | bound entropy of kept set | "tightens when confident" → *maybe* collapses | **UNRESOLVED** `[verified def 2509.02510; loop-class not verified]` |
| typical | target the typical set | may *exclude* the over-confident mode | **UNRESOLVED** |

**The min-p arithmetic (why pless is the worst case):** for a top-heavy distribution
`Σpᵢ² ≈ p_max²`, while min-p's threshold is `p_base·p_max` (`p_base`~0.05–0.1). At
`p_max=0.7`: pless threshold ≈ **0.49** (cuts the 2nd token → greedy); min-p ≈ **0.07**
(keeps the tail). Ratio ≈ `p_max/p_base` ≈ 7×. So pless goes near-greedy as soon as the
model is *moderately* confident, while min-p only collapses at much deeper peaks — pless's
`Σpᵢ²` (quadratic in the probabilities) is the **most aggressive** peak-sensitive threshold,
which is precisely why it loops the most.

**Two corrections (we inferred wrong, then read the definitions):**
- **η-sampling** — first lumped with pless; WRONG. The `min(ε, ·)` *cap* keeps the threshold
  ≤ a small ε, so it preserves the escape route; the paper reports it is "better at breaking
  out of repetition." η **resists**.
- **typical sampling** — retracted from the "loops" list; its mechanism can *avoid* the mode,
  so it may resist. Unresolved without reading it.

---

## 3. Related work — the looping literature (math/QA, not coding)

**The phenomenon itself (closest to ours):**
- **Why Do Reasoning Models Loop?** ([2512.12895](https://arxiv.org/abs/2512.12895)) `[verified, fetched]`
  — "they often loop … **at low temperatures or with greedy decoding**"; root cause = risk
  aversion (easy cyclic action) + Transformer temporally-correlated-error bias;
  **"temperature is a stopgap rather than a holistic solution."** Corroborates our
  peakedness/low-T finding *and* "temp↑ removes the loop but not the correctness."
- **Circular Reasoning** ([2601.05693](https://arxiv.org/abs/2601.05693)) `[verified, abstract]`
  — "a **self-reinforcing trap** where generated content acts as a logical premise for its
  own recurrence"; "state collapse," V-shaped attention; **"semantic repetition precedes
  textual repetition"**; CUSUM early detection; LoopBench. The "semantic precedes textual"
  point is the sharpest critique of our surface n-gram detector (it only catches the textual
  phase — the ~1.5% it misses).
- **The Curious Case of Neural Text Degeneration** ([1904.09751](https://arxiv.org/abs/1904.09751))
  `[verified]` — origin: maximization decoding (greedy/beam) → repetitive loops.

**Adjacent failure modes:**
- **Overthinking** ([2412.21187](https://arxiv.org/abs/2412.21187)) `[verified]` — o1-like burn
  1,953% more tokens on trivial problems (13 solutions for "2+3"); *converges* but wastes
  compute. Inefficiency, not non-termination.
- **Underthinking** ([2501.18585](https://arxiv.org/abs/2501.18585)) `[verified]` — frequent
  thought-switching → never commits → *more* tokens and wrong (AIME). A non-convergence sibling.

**Intervention (mirrors our force-`</think>`):**
- **s1: Simple Test-Time Scaling** ([2501.19393](https://arxiv.org/abs/2501.19393)) `[verified]`
  — "budget forcing": forcefully terminate thinking (or extend with "Wait"). The established
  analog of our loop-force action.

**Decoding-method framing (the key positioning):** the standard literature treats
maximization (greedy/beam/low-T) as the *cause* and **truncation samplers (top-p/top-k/min-p/
Mirostat/contrastive search) as the *cure*** for tail-driven repetition. A recurring
(secondary-source) observation — *"top-p/top-k prevent **initial** repetition but cannot
**recover** once it occurs"* — is the qualitative version of our **prevention ≫ rescue**;
the quantified, reasoning/code version appears to be ours.

---

## 4. Where our work sits (positioning / novelty)

Prior work is **complementary, not pre-empting**:
1. **Domain:** the looping literature is math/QA; ours is **code** (APPS competitive
   programming), which has different per-token entropy structure.
2. **The inversion:** the field treats adaptive truncation as the *antidote* to repetition.
   We show the **dual** — a *peak-sensitive hard-truncation* sampler (pless; and by the same
   logic min-p / `G_k`) **amplifies** looping at confident positions, because its threshold
   rises with confidence and a loop is a confident state. No surveyed paper frames the
   adaptive-threshold class as a looping *cause*.
3. **Mechanism + fix:** the lit calls temperature a "stopgap" and goes to *training-time*
   fixes; we show a **training-free decoding fix** (α↑/T↑ — prevention) that resolves the
   truncation, while confirming the lit's point that it does not manufacture correctness
   ("solution-existence is the disease").
4. **Quantified prevention ≫ rescue** and the **closed-no-code diagnostic** (cross-model:
   Qwen3 0.4% vs DeepSeek 34–41%).

**One-line framing for a paper:** *"The decoding literature treats truncation sampling as
the remedy for greedy-decoding repetition; we show the dual — a peak-sensitive hard-truncation
sampler re-introduces the failure at confident positions, and the fix is to flatten (α↑/T↑),
not to detect-and-rescue."*

---

## 5. Open / to verify

- **top-H** floor behaviour: does its entropy bound shrink to ~0 at near-zero model entropy
  (→ collapses, loops) or floor (→ resists)? Needs the ECMM algorithm read or a test.
- **["Repetitions are not all alike"](https://arxiv.org/abs/2504.01100)** `[unverified]` —
  most relevant primary source on *repetition mechanisms*; read before claiming the
  peak-sensitive-causes-loops point is novel.
- **["Balancing Diversity and Risk in LLM Sampling"](https://arxiv.org/abs/2408.13586)**
  `[unverified]`, **production-repetition study ([2512.04419](https://arxiv.org/abs/2512.04419))**
  `[unverified]`, **LZ-Penalty ([2504.20131](https://arxiv.org/abs/2504.20131))** `[unverified]`
  — check whether any already states our inversion.
- **Testable prediction:** run min-p, top-nσ, and top-H on the same APPS-interview CoT setup;
  the mechanism predicts min-p loops (milder than pless), top-nσ resists (T-invariant), top-H
  unknown. The argument yields a ranked hypothesis; only the experiment settles membership.
