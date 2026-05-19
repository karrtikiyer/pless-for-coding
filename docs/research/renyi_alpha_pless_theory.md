# Rényi-α p-less: Theoretical Foundations

A reading-friendly write-up of the math + literature framing for the
workshop paper section. All equations rendered both in LaTeX (for
GitHub / VS Code preview) and in plain Unicode (so the file is readable
in any terminal viewer without a math renderer).

---

## 1. The fundamental object: Rényi entropy of order α

The quantity in our threshold isn't ad-hoc; it's part of a 60-year-old
information-theoretic family.

**Rényi entropy of order α** (Rényi, 1961, *On Measures of Entropy and
Information*):

$$
H_\alpha(p) \;=\; \frac{1}{1-\alpha} \log \left( \sum_i p_i^\alpha \right)
$$

Plain text:

```
H_α(p)  =  (1 / (1 − α)) · log( Σᵢ pᵢ^α )
```

defined for α > 0, α ≠ 1, with the following limiting cases:

| α       | Name             | Closed form               | Captures               |
|--------:|------------------|---------------------------|------------------------|
| α → 0   | Hartley          | `log |support|`           | support size           |
| α → 1   | Shannon          | `−Σ pᵢ log pᵢ`            | average information    |
| α = 2   | Collision        | `−log Σ pᵢ²`              | self-match probability |
| α → ∞   | Min-entropy      | `−log max(pᵢ)`            | best-case predictability |

Two facts from the literature the paper needs:

1. **`H_α(p)` is monotonically non-increasing in α** (Rényi, 1961; see
   also Cover & Thomas, *Elements of Information Theory*, ch. 17).
   Equality iff `p` is uniform.
2. **Our threshold is exactly the inverse-exponential of `H_α`:**

$$
T_\alpha(p) \;=\; \sum_i p_i^\alpha \;=\; \exp\!\bigl(-(\alpha - 1)\, H_\alpha(p)\bigr)
$$

Plain text:

```
T_α(p)  =  Σᵢ pᵢ^α  =  exp( −(α − 1) · H_α(p) )
```

So **our generalized p-less sampler is parameterized by the Rényi
entropy order**. The α=2 case isn't a magic number — it's one specific
point on this family, the one the original paper (Tan, Wu, Howard,
arXiv:2509.23234) chose for a specific reason.

---

## 2. Why the original paper chose α = 2

The threshold `Σpᵢ²` has three independent interpretations that
motivated Tan et al.'s choice:

### 2.1 Collision probability

In cryptography / information theory: the probability that two i.i.d.
samples from `p` give the same token.

$$
\Pr[X_1 = X_2 \mid X_1, X_2 \stackrel{\text{iid}}{\sim} p] = \sum_i p_i^2
$$

```
P(X₁ = X₂  |  X₁, X₂ iid ∼ p)  =  Σᵢ pᵢ²
```

Pruning below it means: *keep tokens more likely than a random
self-match.*

### 2.2 Expected next-token probability

$$
\mathbb{E}_{X \sim p}[p(X)] \;=\; \sum_i p_i \cdot p_i \;=\; \sum_i p_i^2
$$

```
E_{X ∼ p}[ p(X) ]  =  Σᵢ pᵢ · pᵢ  =  Σᵢ pᵢ²
```

So the rule is: *keep tokens whose probability is above the
distribution's own self-expectation.*

### 2.3 A bound tight enough to guarantee at least one token survives

By a Cauchy-Schwarz-style argument:

$$
\sum_i p_i^2 \;\leq\; \max_i(p_i) \cdot \sum_i p_i \;=\; \max_i(p_i) \cdot 1 \;=\; \max_i(p_i)
$$

```
Σᵢ pᵢ²  ≤  max(pᵢ) · Σᵢ pᵢ  =  max(pᵢ) · 1  =  max(pᵢ)
```

This is the *existential validity property* the upstream paper claims:
the threshold can never exceed the max probability, so the top token
always survives the prune. No edge cases.

All three interpretations are α=2-specific. Generalizing to other α
**breaks each of them**, which is why "hyperparameter-free" loses its
purity once you introduce α as a knob. What we gain in exchange is a
sharper version of the adaptive-by-shape property.

---

## 3. What changes mathematically as α grows

Pull the threshold apart algebraically:

$$
T_\alpha(p) \;=\; \sum_i p_i^\alpha \;=\; \sum_i p_i \cdot p_i^{\alpha-1} \;=\; \mathbb{E}_{X\sim p}\bigl[p(X)^{\alpha-1}\bigr]
$$

```
T_α(p)  =  Σᵢ pᵢ^α
        =  Σᵢ pᵢ · pᵢ^(α−1)
        =  E_{X ∼ p}[ p(X)^(α−1) ]
```

So **the threshold is the expectation, under `p` itself, of the
`(α − 1)`-th power of next-token probability**:

| α    | Threshold interpretation                |
|-----:|-----------------------------------------|
| α = 2 | `E[p(X)]`     — the self-expectation    |
| α = 3 | `E[p(X)²]`    — the second moment       |
| α = 5 | `E[p(X)⁴]`    — the fourth moment       |
| α → ∞ | dominated by `max(pᵢ)^α`                |

Since `p(X) ≤ 1`, raising to higher powers shrinks values. Higher
moments of `p(X)` are progressively smaller. So **`T_α` decreases
monotonically in α** (for non-uniform distributions, which all
language-model outputs are in practice).

That's the raw fact: higher α → lower threshold → more permissive
pruning. The *interesting* question is how the threshold relates to
the distribution's shape.

---

## 4. The adaptive-permissiveness theorem (the load-bearing lemma)

This is what the paper needs to state formally. Let
`m := max(pᵢ)` and `V := |support|`.

### Theorem (bounds on the threshold)

For α ≥ 1:

$$
V^{1-\alpha} \;\leq\; T_\alpha(p) \;\leq\; m^{\alpha-1}
$$

```
V^(1−α)  ≤  T_α(p)  ≤  m^(α−1)
```

- **Upper bound:**
  $\sum_i p_i^\alpha = \sum_i p_i \cdot p_i^{\alpha-1} \leq m^{\alpha-1} \sum_i p_i = m^{\alpha-1}$.
  Achieved at point mass.
- **Lower bound:** by the power-mean inequality.
  Achieved at uniform.

### Corollary 1 — *Max always survives at α ≥ 2*

For α ≥ 2:

$$
T_\alpha(p) \;\leq\; m^{\alpha-1} \;\leq\; m \quad (\text{since } m \leq 1 \text{ and } \alpha - 1 \geq 1)
$$

```
T_α(p)  ≤  m^(α−1)  ≤  m       (since m ≤ 1 and α − 1 ≥ 1)
```

So the max-probability token always satisfies `pᵢ ≥ T_α` and is never
pruned. The existential-validity property of upstream p-less
generalizes to all α ≥ 2.

At α < 2 this breaks: `m^(α-1) > m` when `m < 1`, so the threshold
*can* exceed the mode. That's exactly why our implementation needs the
argmax fallback for α < 2 — to handle the case where the prune would
zero everything out.

### Corollary 2 — *Asymmetric behavior by shape*

The ratio `T_α / m` measures how far the threshold sits below the mode.
Two limiting cases:

| Shape                                  | `T_α / m` as α grows           |
|----------------------------------------|--------------------------------|
| **Point-mass** (only the mode has any prob) | exactly 1, independent of α   |
| **Uniform** (all V tokens have prob 1/V)    | `V^(1−α) / (1/V) = V^(2−α) → 0` |

So:

- For **peaked distributions**, the threshold stays near the mode at
  every α (only the mode survives — no matter how high α gets).
- For **flat distributions**, the threshold collapses far below the
  mode as α grows (many tokens survive).

This is the **adaptive-permissiveness** property. It's a *structural
consequence* of the Rényi entropy ordering, not a tuning trick. We get
it for free as a corollary of using `H_α` as the entropy measure.

The two corollaries above are the only theorems the paper needs to
prove. Each is two lines of algebra.

---

## 5. Position in the decoding literature

The α-sweep slots cleanly into the existing landscape of *adaptive
truncation* samplers — and unifies several of them.

| Method            | Threshold formula                             | Shape-adaptive via |
|-------------------|-----------------------------------------------|--------------------|
| Top-k             | rank-k probability                            | no (fixed support) |
| Top-p (nucleus)   | smallest cumulative ≥ p                       | cumulative mass    |
| Locally typical   | `\|−log p(x) − H[Y\|x]\| ≤ τ`                  | Shannon entropy `H₁` |
| η-sampling        | `max(ε, √ε · exp(−H[Y\|x]))`                  | Shannon entropy `H₁` |
| Min-p             | `p_floor · max(pᵢ)`                           | min-entropy `H_∞`  |
| p-less (α=2)      | `Σ pᵢ²`                                       | Rényi entropy `H₂` |
| **p-less-α (ours)** | **`Σ pᵢ^α`**                                | **Rényi entropy `H_α`, parametric in α** |

References:

- **Top-k**: Fan et al., 2018, *Hierarchical Neural Story Generation*.
- **Top-p (nucleus)**: Holtzman et al., ICLR 2020 (arXiv:1904.09751).
- **Locally typical**: Meister et al., EMNLP 2022 (arXiv:2202.00666).
- **η-sampling / Truncation Sampling as Desmoothing**: Hewitt et al.,
  EMNLP 2022 (arXiv:2210.15191).
- **Min-p**: Nguyen et al., 2024 (arXiv:2407.01082). Note: critical
  reanalysis arXiv:2506.13681 found min-p's reported diversity gains
  don't replicate; cite as cautionary context.
- **p-less (α=2)**: Tan, Wu, Howard, 2026 (arXiv:2509.23234).

### Framing for the paper

> Existing adaptive truncation samplers each correspond to one
> particular Rényi entropy order. p-less uses `H₂`. Locally typical
> and η-sampling use `H₁` (Shannon). Min-p uses `H_∞`. We show that
> for code generation, the family parameterized by α — with α=2 as the
> special case of Tan et al. — produces a Pareto-superior frontier
> when α ∈ (2, 5].

This positions the work as **unifying** rather than just adding another
sampler. Rényi-α p-less is the parametric family that makes the
existing methods comparable along one axis.

---

## 6. Why this works for code specifically — the bimodal-entropy hypothesis

The empirical observation (across 3 models × 2 benchmarks): the
α-sweep extends the Pareto frontier without breaking syntax. The
theoretical mechanism:

### Claim (paper section)

> *Code tokens lie on a bimodal entropy distribution. Position-
> conditional next-token Rényi entropy `H_α(p_t | x_<t)` has two
> modes:*
>
> 1. *a low-entropy mode* — syntactic positions (indentation, closing
>    brackets, keywords like `def` / `return`, punctuation), and
> 2. *a higher-entropy mode* — semantic decision points (function-name
>    choice, algorithm-choice, operator selection).
>
> *The α-sweep adapts to this bimodality automatically:*
>
> - *at low-entropy positions*, all α ≥ 2 keep only the mode (by
>   Corollary 1);
> - *at high-entropy positions*, raising α shrinks the threshold
>   proportionally faster than `m`, admitting more candidate tokens
>   (by Corollary 2).

This is closely related to **AdapT** (Zhu et al., AAAI 2024,
arXiv:2309.02772), which classified tokens as "challenging" vs
"confident" by *loss* and adapted temperature accordingly. p-less-α
achieves the same selective-loosening **without an explicit position
classifier** — the distribution shape itself triggers the right
behavior.

### Caveat about evidence

The bimodal-entropy claim is **theoretical-by-analogy** in this work;
we haven't yet measured per-position `H_α(p_t | x_<t)` on our task
distributions. AdapT empirically partitions code tokens as
"challenging vs confident" by *loss*, not Rényi-α entropy. To make the
bimodal-entropy claim load-bearing, we'd need a half-day GPU run that
logs logits at every position on ~50 problems per model and shows the
distribution of `H_α(p_t)` is bimodal.

Until that's measured, the paper section should phrase the claim
carefully as the *plausible explanation* for the asymmetric-by-shape
pattern, supported by:

- AdapT's loss-based finding (Zhu et al.) of "challenging vs confident"
  position classes.
- Our empirical observation that α-sweep doesn't break syntax
  (corollary 1 holds in practice — pass@1 cost is bounded at 1.4–3.0
  pp across all 6 cells).

Empirical verification then becomes the natural follow-up.

---

## 7. Quality-diversity Pareto interpretation

The empirical result — pass@1 monotonically ↓ and pass@10 monotonically
↑ with α — is the **information-theoretic quality-diversity trade-off**,
formalized:

- **Quality** ≈ `Σᵢ pᵢ · 𝟙[correct(i)]` (probability of sampling a
  correct token in one shot)
- **Diversity** at the Rényi-α level ≈ `H_α(p)` over the kept set
  after pruning

For finite α, narrowing the kept set (lower α) concentrates probability
mass on high-`p` tokens — raising quality, lowering diversity.
Widening it (higher α) does the reverse.

The trade-off itself is not new:

- **Caccia et al., ICLR 2020** (arXiv:1811.02549), *Language GANs
  Falling Short*: framed the quality-diversity Pareto for sequence GANs.
- **Holtzman et al., ICLR 2020**: framed it for nucleus sampling.
- **Self-consistency** (Wang et al., 2022): more diverse samples →
  better majority-vote answer at fixed budget.

### What's actually new in our claim

The quality-diversity trade-off curve is **not the same shape** for
the temperature axis as for the Rényi-α axis on code.

Concretely:

- The temperature curve and the α curve **intersect at α = 2 and T = 1**
  (the same baseline).
- They diverge as you move away.
- The α curve sits **above-and-left** of the temperature curve in
  (pass@1, pass@10) space.

This means: **at any chosen pass@10, the α-sweep gives higher pass@1
than temperature alone can achieve.**

The mathematical reason is structural:

| Knob        | Operation                                                  |
|-------------|------------------------------------------------------------|
| Temperature | `pᵢ → pᵢ^(1/T) / Z` — reshapes the **whole distribution** |
| Rényi-α     | Hard threshold on `pᵢ ≥ T_α`, then renormalize survivors  |

Temperature disturbs peaked positions (it widens what should be
deterministic). The α-sweep is a *threshold* operation that leaves
the survivor probabilities **proportional to the original** — it only
shrinks or expands the support, never re-weights the head against the
body.

This is the cleanest framing for the paper's central claim.

---

## 8. Recommended paper section structure

| § | Title                                                | Contents |
|--:|------------------------------------------------------|----------|
| 1 | Introduction                                          | Pareto-extension claim; quality-diversity framing; cross-model robustness. |
| 2 | Background: Rényi entropy & decoding                  | The `H_α` family; existing samplers as one-point selections (table in §5). |
| 3 | p-less-α                                              | Definition; threshold `Σ pᵢ^α`; recovery of upstream at α=2; the adaptive-permissiveness theorem with both corollaries. |
| 4 | The bimodal-entropy hypothesis for code               | Measure per-position Rényi-α entropy on MBPP; show bimodality; argue why this makes Rényi-α the natural sampler family for code. (Conditional on running the measurement experiment.) |
| 5 | Experiments                                           | The 3×2×4 matrix; pass@k, struct_div, cb_div; NAUADC for 2 models. |
| 6 | Pareto-frontier analysis                              | Temperature curve vs α curve in (pass@1, pass@10); statistical significance; cross-model invariance. |
| 7 | Why temperature fails where α works                   | The threshold-vs-reshape distinction; connection to syntactic/semantic position asymmetry. |
| 8 | Discussion / limitations                              | The bimodal-entropy claim as hypothesis-to-be-verified; missing T-baseline on CodeLlama; one-temperature ablation. |
| 9 | Conclusion                                            | The minimal-modification result: one parameter, ~5-line code change, replicates across 3 models × 2 benchmarks. |

---

## 9. Key claim hierarchy for the workshop submission

| Layer | Claim | Evidence we have today |
|-------|-------|------------------------|
| **Theoretical** | Rényi-α p-less generalizes upstream p-less via the `H_α` family; α=2 is the special case. | Two-line algebra (§3). |
| **Theoretical** | At α ≥ 2, the max-prob token always survives the prune (Corollary 1). | Two-line algebra. |
| **Theoretical** | The threshold/mode ratio `T_α / m` depends on distribution shape — peaked dists stay tight, flat dists open up (Corollary 2). | Two-line algebra. |
| **Empirical** | The α-sweep produces a monotone pass@k curve and monotone diversity curves across 3 models × 2 benchmarks. | The 6-cell matrix in `cross_model_cross_dataset_summary.md`. |
| **Empirical** | NAUADC confirms algorithmic-diversity gain — rules out the "different-looking same algorithm" failure mode. | Claude-judge run on 2 models, NAUADC monotonic in α. |
| **Empirical** | The α curve Pareto-dominates the temperature curve on the same model. | Qwen MBPP T-envelope check holds clean. **Missing pless@T=1.5 on CodeLlama and m-a-p** — would need ~1h GPU each to close. |
| **Hypothesis (testable)** | Code-token next-token entropy `H_α(p_t)` is bimodal across positions. | AdapT-by-analogy + indirect empirical signal that α doesn't break syntax. **Needs a per-position entropy measurement experiment** (~half day GPU). |

---

## 10. References (full list for the paper)

In addition to the upstream p-less paper and Holtzman nucleus, the
paper should cite:

- **Rényi, A. (1961)**, *On Measures of Entropy and Information.*
  *Proceedings of the Fourth Berkeley Symposium on Mathematical
  Statistics and Probability, Volume 1: Contributions to the Theory of
  Statistics*, pp. 547–561. — Primary reference for `H_α`.
- **Cover, T., Thomas, J. (2006)**, *Elements of Information Theory.*
  Wiley. — Standard graduate text; cite for `H_α` monotonicity in α
  and the power-mean inequality used in Corollary 2.
- **Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y. (2020)**,
  *The Curious Case of Neural Text Degeneration.* ICLR.
  ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)) — Nucleus
  sampling; the "before" picture for adaptive truncation.
- **Meister, C., Pimentel, T., Wiher, G., Cotterell, R. (2022)**,
  *Locally Typical Sampling.* EMNLP.
  ([arXiv:2202.00666](https://arxiv.org/abs/2202.00666)) — Closest
  info-theoretic decoder; uses Shannon entropy `H₁`.
- **Hewitt, J., Manning, C. D., Liang, P. (2022)**, *Truncation
  Sampling as Language Model Desmoothing.* EMNLP.
  ([arXiv:2210.15191](https://arxiv.org/abs/2210.15191)) — η-sampling.
- **Nguyen, M., Pinedo, J., Baumgartner, M., Goyal, K., Ardalani, N.,
  Larkin, J. (2024)**, *Min P Sampling: Balancing Creativity and
  Coherence at High Temperature.*
  ([arXiv:2407.01082](https://arxiv.org/abs/2407.01082)). Plus
  cautionary follow-up ([arXiv:2506.13681](https://arxiv.org/html/2506.13681v2)).
- **Caccia, M., Caccia, L., Fedus, W., Larochelle, H., Pineau, J.,
  Charlin, L. (2020)**, *Language GANs Falling Short.* ICLR.
  ([arXiv:1811.02549](https://arxiv.org/abs/1811.02549)) — Formalizes
  the quality-diversity trade-off for sequence generation.
- **Wei, S. et al. (2024)**, *A Thorough Examination of Decoding
  Methods in the Era of LLMs.*
  ([arXiv:2402.06925](https://arxiv.org/abs/2402.06925)) — Empirical
  context for decoding methods on MBPP/HumanEval.
- **Zhu, J. et al. (2024)**, *Hot or Cold? Adaptive Temperature
  Sampling for Code Generation with Large Language Models.* AAAI.
  ([arXiv:2309.02772](https://arxiv.org/abs/2309.02772)) — AdapT;
  closest prior to the bimodal-entropy claim.
- **Tan, J., Wu, T., Howard, J. (2026)**, *Hyperparameter-Free
  Sampling via Collision Probability.*
  ([arXiv:2509.23234](https://arxiv.org/abs/2509.23234)) — The
  original p-less paper.

---

## 11. Worked numerical examples (for intuition)

### Peaked distribution: `[0.92, 0.05, 0.02, 0.01]` (one heavy token)

| α   | T_α (computed) | Surviving tokens |
|----:|---------------:|------------------|
| 2.0 |          0.846 | only 0.92        |
| 2.5 |          0.813 | only 0.92        |
| 3.0 |          0.779 | only 0.92        |
| 5.0 |          0.659 | only 0.92        |

Even at α=5, threshold (0.659) is bigger than the second-best (0.05).
**Corollary 1 in action**: the mode always survives at α ≥ 2.

### Flat distribution: `[0.30, 0.25, 0.20, 0.15, 0.10]`

| α   | T_α    | Kept tokens                        | # kept |
|----:|-------:|------------------------------------|-------:|
| 2.0 | 0.225  | {0.30, 0.25}                       |      2 |
| 2.5 | 0.110  | {0.30, 0.25, 0.20}                 |      3 |
| 3.0 | 0.055  | {0.30, 0.25, 0.20, 0.15}           |      4 |
| 5.0 | 0.004  | {0.30, 0.25, 0.20, 0.15, 0.10}     |      5 (all) |

**Corollary 2 in action**: at flat positions, the threshold collapses
much faster than the mode, so the kept set grows.

### α = 1.5 on `[0.5, 0.5]` (the fallback case)

T_α = 0.5^1.5 + 0.5^1.5 = 2 · 0.3536 = **0.707**.

But max(pᵢ) = 0.5 < 0.707. So `mask.all(dim=-1) = True` — every token
would be pruned. The argmax fallback in our implementation un-masks
position 0 (the first occurrence of the max), so the sampler returns
token 0 deterministically. This is **degenerate to greedy** — and it's
why α<2 collapses to greedy in practice on code (every step hits this
case).

---

## 12. One claim worth being careful about (limitations)

**The bimodal entropy story is currently theoretical-by-analogy, not
empirically verified in this work.** AdapT empirically partitions code
tokens as "challenging vs confident" by *loss*, not Rényi-α entropy.

To make the bimodal-entropy claim load-bearing in the paper, we need
to actually measure per-position `H_α(p_t | x_<t)` across MBPP /
HumanEval positions and show the histogram is bimodal. That's a
half-day GPU run logging logits at every position on ~50 problems per
model. Cheap, but not yet done.

If we don't run it, the paper section should phrase the bimodal-entropy
claim more carefully — as the *plausible* explanation for the
asymmetric-by-shape pattern, supported by AdapT's loss-based finding
and consistent with our empirical observation that the α-sweep doesn't
break syntax. Empirical verification then becomes the natural follow-up
(and is cheap enough to do for a journal extension).

---

## Appendix A — Two-line proof of Corollary 1 (max-survival at α ≥ 2)

For α ≥ 1, by factoring out the largest term:

$$
\sum_i p_i^\alpha \;=\; \sum_i p_i \cdot p_i^{\alpha-1} \;\leq\; \max_i\!\bigl(p_i^{\alpha-1}\bigr) \cdot \sum_i p_i \;=\; m^{\alpha-1}
$$

```
Σᵢ pᵢ^α  =  Σᵢ pᵢ · pᵢ^(α−1)  ≤  max(pᵢ^(α−1)) · Σᵢ pᵢ  =  m^(α−1)
```

For α ≥ 2: since `0 ≤ m ≤ 1` and `α − 1 ≥ 1`, raising `m` to a positive
power doesn't increase it:

$$
m^{\alpha-1} \;\leq\; m
$$

```
m^(α−1)  ≤  m       (since m ≤ 1 and α − 1 ≥ 1)
```

Combining: `T_α ≤ m`. So `p_{argmax} ≥ T_α`, and the mode is never
pruned. □

---

## Appendix B — Why the lower bound `V^(1−α)` matters

For uniform distributions over `V` tokens (`pᵢ = 1/V` for all `i`):

$$
T_\alpha(\text{uniform}) \;=\; V \cdot (1/V)^\alpha \;=\; V^{1-\alpha}
$$

```
T_α(uniform)  =  V · (1/V)^α  =  V^(1−α)
```

For language-model vocabularies (V ≈ 32,000 to 150,000), this is
absurdly small at α = 5:

- V = 32,000: T_α(uniform) = `32_000^(−4) ≈ 9.5 × 10^−19`
- Every token at uniform has probability 1/V ≈ `3.1 × 10^−5`
- Ratio: every token's probability is `~3 × 10^13` times bigger than
  the threshold

So at α = 5 on a uniform distribution, **every single token survives**.
The sampler effectively becomes plain multinomial sampling. This is
why α → ∞ recovers full multinomial sampling without truncation.

The implication for the paper: the α-sweep traces a smooth curve from
**greedy (α → 1)** through **standard p-less (α = 2)** to **plain
multinomial (α → ∞)**. The "sweet spot" α ∈ (2, 5] for code lives on
this curve — a single hyperparameter selects where on the
quality-diversity frontier the sampler operates.

---

## Appendix C — Summary of the theorems in one sentence

> The threshold `T_α = Σ pᵢ^α = exp(−(α−1) H_α(p))` is bounded above
> by `max(pᵢ)^(α−1)` and below by `V^(1−α)`, so for α ≥ 2 the
> max-probability token always survives the prune, while the relative
> "tightness" of the threshold versus the mode shrinks at exactly the
> rate of `H_α(p)` itself — automatically tight at peaked positions
> and loose at flat ones.

That sentence is the mathematical heart of the paper. Everything else
in the theoretical sections is unpacking it.
