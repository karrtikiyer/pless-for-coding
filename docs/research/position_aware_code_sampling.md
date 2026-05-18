# Position-Aware Decoding for Code: Tail-Promotion and Related Ideas

Parked research direction. Revisit after the CODEFORCES NAUADC story
lands and the workshop paper has a first draft.

## Motivating question

Code generation has lower surface diversity than general writing — most
token positions in code are near-deterministic (whitespace,
indentation, closing brackets, syntactic keywords) while a few
positions are high-entropy (function name choice, operator choice,
algorithm-defining branch, base-case-vs-recursive-step). Current
samplers (temperature, top-p, p-less) all reweight a sorted-by-probability
distribution and end up sampling proportional to probability inside
the kept set. **Can a sampler that explicitly *promotes* lower-probability
tokens at high-entropy positions buy diversity that pless / temp / top-p
cannot?**

## First-principles framing

Code is **bimodally low-entropy**:

- Most positions: top token has ≥95% mass. Temperature wastes its
  noise budget here, producing syntax errors.
- A few "high-leverage" positions: 3–5 tokens are genuinely plausible.
  This is where algorithmic divergence lives.

Pless handles the *cutoff* adaptively (collision-entropy threshold
`p = Σ probsᵢ²`), but within the kept set it still samples proportional
to probability — so the second-best token at a semantic position, which
is where algorithmic diversity lives, still loses to the top token most
of the time.

The clean idea: detect high-entropy positions and at those positions
**bias toward the tail of the plausible set** rather than just widening
the cutoff or flattening proportions.

## What already exists

### Closest analogs (entropy / position-aware temperature)

- **AdapT** ([arXiv:2309.02772](https://arxiv.org/abs/2309.02772),
  AAAI 2024) — classifies code tokens as "challenging" (first token
  of a code block, high loss) vs "confident"; applies T_high to
  challenging and T_low to confident. Tested MBPP/HumanEval, jointly
  improves pass@5/10/15.
- **EDT** ([arXiv:2403.14541](https://arxiv.org/abs/2403.14541)) —
  per-token T as monotone function of local entropy `H(p_t)`. General
  NLG, not code-first.
- **AdaDec** ([arXiv:2506.08980](https://arxiv.org/abs/2506.08980),
  2025) — at high-uncertainty positions, *re-ranks* top-3 via
  5-token lookahead average log-prob. Up to +20.9 pp pass@1 on
  HumanEval+ / MBPP+ / DevEval. Key empirical claim: **at uncertain
  positions the correct token is often *in* top-3 but mis-ranked.**
- **DecoRTL** ([arXiv:2507.02226](https://arxiv.org/abs/2507.02226),
  2025) — syntax-aware T for Verilog. Low-T for syntax tokens,
  high-T for "exploratory" tokens.
- **Entropix** (xjdr-alt, 2024,
  [repo](https://github.com/xjdr-alt/entropix)) — joint entropy +
  varentropy thresholds; routes to greedy/resample/branch/CoT.
  Community project, no peer review.

### Closest tail-promotion analog

- **XTC** ("Exclude Top Choices", community sampler, p-e-w 2024,
  [PR #6335](https://github.com/oobabooga/text-generation-webui/pull/6335))
  — with probability `p_xtc`, removes all tokens with `p > threshold`
  *except* the least probable one above threshold. Used in
  oobabooga / llama.cpp for creative writing. **Never evaluated on code.
  Fires unconditionally, not entropy-gated.**

### Other context

- **Locally Typical Sampling**
  ([arXiv:2202.00666](https://arxiv.org/abs/2202.00666)) — picks
  tokens whose surprisal `−log p(y)` is closest to `H[Y|x]`.
  Information-theoretic middle, neither head nor tail.
- **Min-p** ([arXiv:2407.01082](https://arxiv.org/abs/2407.01082)) —
  floor at `p_floor × max_prob`. **Cautionary tale:**
  [critical reanalysis arXiv:2506.13681](https://arxiv.org/html/2506.13681v2)
  finds reported diversity gains do not replicate.
- **Mirostat** ([arXiv:2007.14966](https://arxiv.org/abs/2007.14966))
  — controls perplexity. **Collapses on MBPP** (7.8% vs FSD-d 21.2%
  per [Wei et al. arXiv:2402.06925](https://arxiv.org/abs/2402.06925)).
- **DoLa** ([arXiv:2309.03883](https://arxiv.org/abs/2309.03883)) —
  cross-layer logit contrast. Quality-focused (TruthfulQA), not
  diversity. Not tested on code.
- **Contrastive Decoding** ([arXiv:2210.15097](https://arxiv.org/abs/2210.15097),
  reasoning extension
  [arXiv:2309.09117](https://arxiv.org/abs/2309.09117)) —
  amateur-model suppression. Quality-focused.
- **FSD-d** ([arXiv:2305.12675](https://arxiv.org/abs/2305.12675)) —
  anti-LM penalty against already-emitted prefix. Tops MBPP/HumanEval
  in [Wei 2402.06925](https://arxiv.org/abs/2402.06925). Anti-repetition,
  not entropy/position-aware.

### Bimodal-entropy claim

No code-generation paper explicitly states "code decoding should be
position-aware *because of* entropy bimodality." AdapT operationalizes
the partition without naming the bimodality. DAPO / RLVR-line work
reports bimodal token-entropy in reasoning traces (peaks ~0.2 and
~1.3 nats), but for reasoning, not code generation.

## Novelty pocket

A genuinely new contribution would need to combine, at minimum, two of:

1. **Entropy-gated** firing (XTC fires unconditionally with probability
   `p_xtc`; the proposal would fire only when `H_t > τ`).
2. **Rigorous code evaluation** with pass@k and structural / algorithmic
   diversity (this repo's existing zss + algosim pipeline).
3. **Principled head-mass-removal** (e.g. tied to collision entropy
   `Σ p²` as in pless rather than a magic threshold).
4. **Ablations vs AdapT / EDT / min-p / AdaDec** to isolate the
   tail-bias effect from temperature scaling and lookahead reranking.

Clean framing: **"entropy-gated XTC for code"** or
**"anti-greedy at bimodal forks"**. Neither phrase appears in the
code-gen literature.

## Expected effect size

Estimated outcome distribution on MBPP-50 / HumanEval-50 smoke test:

| Outcome                                   | Estimated probability |
|-------------------------------------------|----------------------:|
| pass@10 up +1 to +3 pp, struct_div up     |                  ~30% |
| pass@10 flat, struct_div up               |                  ~40% |
| pass@10 down, struct_div up               |                  ~20% |
| pass@10 up clearly above noise (+5 pp+)   |                  ~10% |

The fattest probability mass is on the "different-looking wrong code"
failure mode: structural diversity rises, pass@k doesn't. The
AdaDec finding ("correct token in top-3 but mis-ranked") is the
mechanism behind this risk — tail-promotion at mis-ranked positions
pushes toward genuinely wrong tokens.

## Counter-arguments

1. **AdaDec's finding cuts against tail-promotion directly.** If the
   correct token is in top-3 but mis-ranked, reranking is the right
   move, not down-weighting the head.
2. **min-p reanalysis cautionary tale.** The most-hyped 2024
   "creativity sampler" did not survive scrutiny.
3. **Mirostat collapsed on MBPP.** Concrete prior of a plausible-looking
   sampler hurting code pass rates.
4. **Tail-promotion may just produce syntactically valid but
   semantically wrong code.** AST-fingerprint diversity rises while
   pass@k stays flat — easy to fool oneself.

## Extension: Rényi-α generalization of p-less

The p-less threshold `Σ p_i²` is the **collision probability**, equal
to `exp(−H₂(p))` where `H₂` is Rényi entropy of order 2. The natural
generalization is to use any α ≥ 1:

$$
\text{threshold}_\alpha = \sum_i p_i^\alpha
$$

— prune tokens below this, renormalize, sample. α = 2 is current
p-less. For α ≥ 1 and non-uniform distributions, `Σ p_i^α` is
monotonically decreasing in α, so:

- α < 2 → threshold higher → keeps *fewer* tokens (tighter, more greedy)
- α = 2 → current p-less
- α > 2 → threshold lower → keeps *more* tokens (looser, more diverse)
- α → ∞ → threshold → 0 → no pruning (sample from full distribution)

### The asymmetric-by-distribution-shape property

The key observation for code: **α > 2 selectively loosens at high-entropy
positions while preserving tightness at low-entropy ones**, because the
threshold scales with the input distribution's shape rather than being
applied as a global rescaling (the way temperature is).

**Syntactic position** (peaked: `[0.92, 0.05, 0.02, 0.01]`):

| α   | threshold | tokens kept |
|----:|----------:|---|
| 1.5 | 0.913     | {0.92}  |
| 2   | 0.849     | {0.92}  |
| 2.5 | 0.813     | {0.92}  |
| 3   | 0.779     | {0.92}  |
| 5   | 0.659     | {0.92}  |

Every reasonable α keeps the same single token. Higher α does not
break syntax.

**Semantic position** (flat: `[0.30, 0.25, 0.20, 0.15, 0.10]`):

| α   | threshold | tokens kept |
|----:|----------:|---|
| 1.5 | 0.469     | {} — fails  |
| 2   | 0.225     | {0.30, 0.25} |
| 2.5 | 0.110     | {0.30, 0.25, 0.20} |
| 3   | 0.055     | {0.30, 0.25, 0.20, 0.15} |
| 5   | tiny      | all 5 |

Higher α opens up the support exactly where diversity is wanted, with
no explicit detector required — the distribution shape itself triggers
the looseness.

### What it is *not*

- **Not temperature.** Temperature rescales the whole distribution
  (`p_i^{1/T}` then renormalize). p-less-α prunes binary, then
  renormalizes survivors. Different operation.
- **Not min-p.** Min-p thresholds at `p_floor × max_p` (linear in
  max_p). p-less-α uses the α-th power sum — different curvature with
  respect to distribution shape.
- **Not p-less-norm.** Pless-norm relaxes via `(v · Σp² − 1) / (v − 1)`
  — support-size normalization. Different relaxation axis.

### Trade-off

Breaks p-less's "hyperparameter-free" sales pitch. The contribution
becomes "for code, α=2 is empirically suboptimal — α ∈ [2.5, 3]
dominates on Pareto," which is weaker than the original paper's claim
but is a real, falsifiable empirical claim.

### Implementation cost

~5 lines in `bench/sampler_bridge.py`. Compute
`threshold = (probs ** alpha).sum()`, prune, renormalize. Guard the
α-too-low "no token survives" edge case (fall back to argmax, or
adaptively raise α). Works identically across HF and vLLM backends —
no logits-processor plumbing required.

### Expected effect distribution (MBPP-50 sweep over
`α ∈ {1.5, 2, 2.5, 3, 5}`)

| Outcome                                                              | Probability |
|----------------------------------------------------------------------|------------:|
| α ∈ [2.5, 3] Pareto-dominates α=2 across models                      |        ~25% |
| Small monotone diversity gain as α↑, indistinguishable from temp       |        ~45% |
| No meaningful movement (the head-token dominates regardless of α)   |        ~25% |
| Syntax breaks at high α (>5)                                         |         ~5% |

### Critical control to run

A genuine novelty claim requires showing α=2.5 beats `α=2 + temperature`
where temperature is tuned to roughly match α=2.5's expected support
size. If they match within noise, the α parameterization is just a
re-parameterization of temperature and the contribution collapses.

This is *more* experiment-worthy than tail-promotion because:

| Axis | Tail-promotion | Rényi-α |
|---|---|---|
| Implementation cost | Medium (new sampler in vLLM/HF) | **Trivial (5 lines)** |
| Novelty | Small pocket (entropy-gated XTC) | **Real parametric generalization** |
| On-thesis | Yes (workshop) | **Yes (workshop)** |
| Risk of "different-looking wrong code" | High | **Lower** (preserves pless's adaptive structure) |
| Theory grounding | Empirical | **Rényi-entropy family** |

## Extension: AST-aware adaptive p-less

A more aggressive variant: instead of letting distribution shape alone
trigger looseness (Rényi-α) or using a learned loss proxy (AdapT),
use **AST node type as the discriminator** for which α (or threshold)
to apply at each position. Hypothesis: different sections of code
benefit from different thresholds — function-name positions want
diversity, syntactic keywords want tightness.

### Mechanisms (in order of cost)

1. **Token-pattern lookback (cheap).** Regex over the last ~20 tokens
   to detect coarse context ("we are just after `def`" → loosen for
   identifier; "we are inside an expression" → default). No real
   parser. Coarse but cheap, implementable in a day as a
   `LogitsProcessor`.
2. **Tree-sitter reactive parsing (medium).** At each token-generation
   step, parse the prefix-so-far with tree-sitter (which has
   error-recovery for incomplete code), identify the deepest AST node
   currently being constructed, look up α from a per-node-type table.
3. **Grammar-state tracking (heavy).** Build on
   Outlines/XGrammar's grammar state machine — instead of *constraining*
   the next token to grammar-valid choices (their current use), use
   the grammar state as the signal for α selection.

### Three reasons this is harder to justify than Rényi-α

1. **The probability distribution already encodes most of what AST
   would tell you.** Pless / any entropy-adaptive sampler keeps few
   tokens at syntactic positions (because they're sharply peaked) and
   many at semantic positions (because they're flat). The empirical
   question is how much variance in "where the model is uncertain" AST
   node type explains *over and above* entropy alone. The bimodal
   entropy measurement (step 1 in the ranked plan) is what would
   settle this.

2. **AdapT is the direct competitor and uses a simpler signal.**
   AdapT ([arXiv:2309.02772](https://arxiv.org/abs/2309.02772))
   classifies tokens as "challenging" vs "confident" by **loss**, not
   AST node type. Joint pass@5/10/15 gains on MBPP/HumanEval. The
   honest comparison for AST-aware-α is "did we beat AdapT with the
   added complexity?" — and if the answer is "by <1 pp," the AST
   classifier is wash.

3. **Implementation cost vs existing infrastructure.** HF
   `LogitsProcessor` + tree-sitter is feasible but introduces a
   per-token parse cost. vLLM's logits-processor API is more
   constrained; AST integration there is hard. Means experiments would
   likely run in HF mode only (slower, doesn't scale to 11-model
   sweep). Rényi-α has none of these problems.

### Where AST-aware *would* be worth doing

There's a non-trivial hypothesis the AST signal might capture and
entropy can't: **the difference between "many tokens are plausible and
all are reasonable"** (function-name choice — open semantic space)
**vs "many tokens are plausible but most are wrong"** (operator
choice — discrete sparse correct set). Both look like high-entropy
positions to a generic detector. AST node type might allow treating
them differently — be loose at the first, tight at the second.

This is the version that has a chance of clearing AdapT as a baseline.
Articulate this hypothesis *with concrete examples* from existing
CODEFORCES generation traces before implementing — if compelling
examples are scarce, the idea probably doesn't have the discriminating
power needed.

### When this becomes worth the complexity

Only if **both**:

- Bimodal entropy measurement shows entropy alone is too coarse to
  separate "where diversity helps" from "where diversity hurts"
- Rényi-α plateaus, and the gap to "what an oracle would do" is large
  enough to justify a parser-aware sampler

If either condition fails, AST-aware is overengineered for the marginal
gain it can deliver.

## Ranked next steps (when revisiting)

1. **Bimodal-entropy characterization (cheap, on-thesis, prerequisite
   for everything else).** Regenerate ~50 MBPP/HumanEval tasks while
   logging per-token logits. Plot per-position entropy distribution.
   If bimodal in code and unimodal in matched prose generation, that's
   a publishable measurement in its own right and justifies every
   position-aware method below. Half a day of GPU. **Do this first
   regardless of which of the variants below gets pursued — it gates
   the framing of all three.**

2. **Rényi-α p-less (cheapest extension, highest expected ROI).**
   One-line sampler change. Smoke test on MBPP-50 with
   `α ∈ {1.5, 2, 2.5, 3, 5}` × `T ∈ {0.7, 1.0}`. Must include the
   "is it just temperature?" control. If a clear winner emerges,
   sweep on MBPP-500 across 2-3 models (one base, one instruct, one
   reasoning) and write up as a workshop-paper section: *"Beyond
   Collision Entropy: A Rényi-α Family of Hyperparameter-Free Decoding
   Rules."*

3. **AdaDec-style lookahead reranking (separate scope).** The +20.9 pp
   pass@1 claim is far larger than any of the diversity-focused
   variants. Pless at confident positions + AdaDec-style lookahead at
   uncertain ones is a clean compound technique. Likely its own paper
   if pursued, not the workshop submission.

4. **Entropy-gated XTC (tail-promotion) as a 1-day spike.** Only after
   the bimodal-entropy measurement and Rényi-α results land. Kill
   criteria:
   - pass@10 not lower than pless@1.0 baseline (Mirostat-collapse test)
   - pass@10 must be numerically higher than pless@1.0 AND structural
     diversity must be higher
   - the pass@10 lift must NOT be replicable by raising temperature
     on pless alone (control arm: pless@1.0 with `temp_after=1.2`)
   - the pass@10 lift must NOT be subsumed by Rényi-α@α=2.5 (if that
     experiment landed first)
   - If any fail, drop it. No full MBPP-500 run, no paper inclusion.

5. **AST-aware adaptive p-less (last-resort, complex, only if 1–4
   plateau).** Cheapest version first: token-pattern lookback only,
   no tree-sitter. Required baseline comparison: AdapT loss-based
   classification. If AST-aware does not beat AdapT by ≥2 pp pass@10
   on MBPP-100, drop. Tree-sitter and grammar-state-tracking versions
   are out of scope without strong evidence that the cheap version
   already shows signal.

6. **What not to do**: do not add tail-promotion or AST-aware as a
   header contribution to the Qwen3-8B paper alongside the existing
   6-config story. The workshop paper is the right venue because it's
   already the "11 models × code decoding" frame. Rényi-α is the only
   variant cheap enough to slot in as a small workshop-paper section
   without scope creep.

## Sources

- [AdapT — arXiv:2309.02772](https://arxiv.org/abs/2309.02772)
- [EDT — arXiv:2403.14541](https://arxiv.org/abs/2403.14541)
- [AdaDec — arXiv:2506.08980](https://arxiv.org/abs/2506.08980)
- [DecoRTL — arXiv:2507.02226](https://arxiv.org/abs/2507.02226)
- [Entropix repo](https://github.com/xjdr-alt/entropix)
- [XTC PR (oobabooga)](https://github.com/oobabooga/text-generation-webui/pull/6335)
- [Locally Typical Sampling — arXiv:2202.00666](https://arxiv.org/abs/2202.00666)
- [Min-p — arXiv:2407.01082](https://arxiv.org/abs/2407.01082)
- [Min-p critical reanalysis — arXiv:2506.13681](https://arxiv.org/html/2506.13681v2)
- [η/ε-sampling — arXiv:2210.15191](https://arxiv.org/abs/2210.15191)
- [Mirostat — arXiv:2007.14966](https://arxiv.org/abs/2007.14966)
- [DoLa — arXiv:2309.03883](https://arxiv.org/abs/2309.03883)
- [Contrastive Decoding — arXiv:2210.15097](https://arxiv.org/abs/2210.15097)
- [Contrastive Decoding for Reasoning — arXiv:2309.09117](https://arxiv.org/abs/2309.09117)
- [FSD/FSD-d — arXiv:2305.12675](https://arxiv.org/abs/2305.12675)
- [Wei et al., A Thorough Examination of Decoding Methods —
  arXiv:2402.06925](https://arxiv.org/abs/2402.06925)
- [Diverse Beam Search — arXiv:1610.02424](https://arxiv.org/abs/1610.02424)
- [Coder-Reviewer — arXiv:2211.16490](https://arxiv.org/abs/2211.16490)
- [AlphaCode — arXiv:2203.07814](https://arxiv.org/abs/2203.07814)
- [PG-TD — OpenReview](https://openreview.net/forum?id=LM1Nt6YHRwe)
- [Tan, Wu, Howard — original p-less paper, arXiv:2509.23234](https://arxiv.org/abs/2509.23234)
- [Outlines — guided generation](https://github.com/dottxt-ai/outlines)
- [XGrammar — arXiv:2411.15100](https://arxiv.org/abs/2411.15100)
- [tree-sitter](https://tree-sitter.github.io/)
