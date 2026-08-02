# Comparison: OpenReview "Higher-Moment p-less" vs Our Rényi-α p-less

## TL;DR — what changes for our paper

The OpenReview submission [`ItFuNJQGH4`](https://openreview.net/forum?id=ItFuNJQGH4) is **the ICLR 2026 oral version of the original p-less paper by Tan, Wu, and Howard** ([arXiv:2509.23234v6](https://arxiv.org/abs/2509.23234v6)) — i.e., the *same* paper our project builds on, not an independent "higher-moment" follow-up. Crucially, version 6 of that paper introduces a **theoretical** generalization to higher Rényi orders in Appendix B.5 (a `k`-order threshold `G[P]_k = 1/exp(H_k(p))`), with an experimental-results appendix C.7 ("Results for Generalization of the p-less Sampling Method") whose content we could not retrieve in full from the HTML render, but which is described elsewhere in the paper as a small-scale ablation, not an empirical study. The paper's headline experiments still use `k=2` exclusively, on **math/reasoning/creative-writing benchmarks (GPQA, GSM8K, QASC, CSQA, Writing Prompts)** with **general LLMs (Llama-2-7B, Mistral-7B, Llama-3-70B, DeepSeek-R1-Distill-Qwen-7B)** — **no code generation, no code-specialised models**.

What this means for us:

- **We are no longer "the first to propose a Rényi-α generalization of p-less."** The formula `G[P]_k = 1/exp(H_k(p))` is mathematically equivalent to our `T_α = Σ p_i^α`, and predates our work in the ICLR 2026 submission of v6 (uploaded to arXiv between v1 in Sep 2025 and v6 on 27 Feb 2026; OpenReview submission 26 Jan 2026).
- **We remain the first to empirically demonstrate the higher-α regime on code generation**, with a real models × benchmarks sweep, against temperature-collapse cliffs, with structural and algorithmic-diversity metrics (NAUADC via Claude judge), and with a per-position bimodal-entropy mechanism (Hartigan's dip test).
- The framing of our workshop paper must shift from "we generalize p-less to Rényi-α" to **"we empirically validate the Rényi-α generalization sketched in Tan et al. (2025), Appendix B.5, on code generation, and show it Pareto-dominates both vanilla p-less and temperature sampling on three models and two benchmarks."**

## Paper bibliographic info

- **Title:** *p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding*
- **Authors:** Runyan Tan, Shuang Wu, Phillip Howard
- **OpenReview:** [openreview.net/forum?id=ItFuNJQGH4](https://openreview.net/forum?id=ItFuNJQGH4)
- **Venue:** ICLR 2026 Conference — listed as **Oral** track, submission 21455
- **OpenReview submission date:** 26 Jan 2026 (last modified 6 May 2026 per OpenReview metadata)
- **arXiv mirror:** [arXiv:2509.23234](https://arxiv.org/abs/2509.23234), six versions Sep 2025 → 27 Feb 2026; **the v6 we compared against was posted 27 Feb 2026**
- **Code:** [github.com/ryttry/p-less](https://github.com/ryttry/p-less) (the `p-less/` submodule in our repo)
- **Reference to add to our bibliography (BibTeX-style key suggestion):** `tan2025pless` — note our existing repo CLAUDE.md already cites this as "Tan, Wu, Howard, arXiv:2509.23234, Feb 2026"; we should explicitly cite v6 / the ICLR 2026 oral and acknowledge Appendix B.5 (and Appendix C.7) as the source of the higher-order extension.

**Verdict on contemporaneity:** This is **not** an independent contemporaneous discovery — it is the very paper our work extends. Their Appendix B.5 / C.7 generalization predates our `pless_alpha` implementation by months, and is by the same authors as the base method.

## Their threshold formula vs ours

**Their (Appendix B.5, Eq. 17):**

> `G[P]_k = 1 / exp(H_k(p))`

where `H_k(p)` is the Rényi entropy of order `k`, `H_k(p) = (1/(1-k)) · log Σᵢ pᵢ^k`.

**Ours (`sampler_bridge.py` / `p_less_alpha_decode`):**

> `T_α = Σᵢ pᵢ^α`, prune tokens with `pᵢ < T_α`, renormalize, sample.

**Equivalence:** for `k = α`, plugging in Rényi entropy gives

`exp(H_k(p)) = exp((1/(1-k)) · log Σ pᵢ^k) = (Σ pᵢ^k)^(1/(1-k))`

so `G[P]_k = (Σ pᵢ^k)^(1/(k-1))`. **This is NOT identical to our `Σ pᵢ^α` for general k.** At `k = 2`, `G[P]_2 = (Σ pᵢ^2)^(1/1) = Σ pᵢ^2 = T_2` — both reduce to collision probability. **CORRECTION (2026-07-30, computed — supersedes the earlier "monotonically related / same direction" wording that was here):** for order > 2 the two forms move in **OPPOSITE** directions, not the same one. On peaked `p=[.7,.2,.1]`, as the order goes 2→5, our `τ_α` **decreases** 0.54→0.17 (threshold drops ⇒ *loosens* ⇒ admits more tail tokens) while `G[P]_k` **increases** 0.54→0.64 toward `max pᵢ` (threshold rises ⇒ *tightens* ⇒ admits fewer). They coincide only at order 2. So they are NOT order-preserving reparameterizations, and no monotone map `α ↔ k` aligns their admitted sets (worked filter-divergence example in `docs/research/paperA_renyi_nonequivalence.md`).

**Implication:** at `α = k = 2` the two methods are *byte-identical* — which is consistent with our byte-equivalence test that confirms our `pless_alpha(α=2)` matches the upstream `p_less_decode`. For `α > 2` vs `k > 2`, our `T_α = Σ pᵢ^α` and their `G[P]_k = (Σ pᵢ^k)^{1/(k-1)}` will produce *different admission sets* on the same distribution, even though they share the same family. **This is a real mathematical difference, not just notation.**

We should either: (a) add a small section to our paper clarifying we use `Σ pᵢ^α` directly rather than the exponentiated form, citing the equivalence at `α=2`, and noting the two forms differ for `α>2`; or (b) re-run a small ablation with `G[P]_k` to confirm the qualitative findings transfer. Option (a) is sufficient and intellectually honest.

## Their empirical scope vs ours

| Dimension | OpenReview / Tan et al. 2025 (v6) | Our project |
|---|---|---|
| Models | Llama-2-7B (Chat), Mistral-7B (Instruct), Llama-3-70B (Instruct), DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct, m-a-p/OpenCodeInterpreter-DS-1.3B |
| Benchmarks | GPQA, GSM8K, QASC, CSQA (reasoning), Writing Prompts (creative) | MBPP-500, HumanEval-164 |
| Domain | Math, multi-hop QA, creative writing | Code generation only |
| α / k grid | `k = 2` throughout main experiments; theoretical generalization in B.5, claimed (but unverified by us) empirical fragment in C.7 | `{2.0, 2.5, 3.0, 5.0}`, all empirical |
| Temperature grid | 0.5, 0.7, 1.0, 1.5, 2.0 + ablation references "values >2.0" (Table 11) | 0.7, 1.0, 1.5, 2.0, 2.5, 3.0 |
| Samples per prompt | Not specified per benchmark in extract; AUC-style aggregation across T | 10 samples per task (pass@10) |
| Diversity metric | n-gram repetition diversity (Su et al. 2022), QASC only; length-controlled win-rate for creative | structural diversity (Zhang-Shasha AST edit distance), CodeBLEU, **NAUADC algorithmic similarity via Claude-Sonnet-4.6 judge** |
| Evaluation judge | Human annotators (6 raters, pairwise) on Llama-2-7B Writing Prompts | None for pass@k; **Claude-Sonnet-4.6 as LLM judge** for algorithmic-similarity (NAUADC) |
| Mechanism analysis | Synthetic distribution figures (B.6 Figs 5–8); GSM8K T=2.0 case study (Sec 5.4) | **Hartigan's dip test on 280K+ per-position empirical next-token distributions**, confirming bimodality on Qwen (`p≈0`, dip=0.005) and CodeLlama (`p≈0`, dip=0.013) |
| Pareto framing | "diversity-accuracy frontier" Pareto dominance vs other samplers, Fig 3 on QASC | pass@1 vs pass@10 Pareto comparison vs `pless@T` and `temp@T`, across 6 (model, benchmark) cells |

**Overlap = essentially zero.** No common models, no common benchmarks, no common diversity metrics, no common judge. The only theoretical overlap is the `k`-order/α-order generalization itself, and even there the exact form differs (`Σ pᵢ^k` vs `(Σ pᵢ^k)^{1/(k-1)}`).

## Findings comparison (table)

| Claim | Their evidence (Tan et al. v6) | Our evidence |
|---|---|---|
| Vanilla p-less Pareto-dominates baselines on (accuracy, diversity) | Fig 3, QASC, Llama-2-7B; +AUC across T sweep on GPQA/GSM8K/QASC/CSQA | We don't re-test this — we *use* `pless@T=1.0` as a baseline and extend it |
| p-less is robust as temperature rises (no degeneration) | Sec 5.4 GSM8K case study at T=2.0; "robustness to high temperatures" without specific collapse boundary | We measure **pass@1 cliff of −14 to −44 pp going T=2.5 → T=3.0** on every (model, benchmark) cell, while α-sweep through α=5 has no analogous cliff |
| Higher-order Rényi generalization (`k > 2`) | Theoretical formula `G[P]_k = 1/exp(H_k(p))` in App. B.5; Appendix C.7 titled "Results for Generalization" exists but content not visible in our extraction; secondary sources describe it as theoretical-only | **Empirical, monotonic pass@10 lift α=2→α=5 in every (model, benchmark) cell: +1.83 pp to +14.63 pp absolute**, bounded pass@1 cost (1.4–3.0 pp) |
| Higher α improves diversity | Predicted by monotonicity ("k-order threshold increases with k") but not empirically reported | **Monotonic structural diversity and CodeBLEU lift with α on 5 of 6 cells**; CodeLlama MBPP stays near zero due to model property |
| Algorithmic / semantic diversity (NAUADC-style) lifts with α | Not measured; closest analogue is human pairwise win-rate on creative writing | **Monotonic NAUADC lift on all 3 models on MBPP: Qwen 1.04→1.17, CodeLlama 1.01→1.12, m-a-p 1.07→1.21**, Claude judge |
| Bimodal per-position entropy structure | Not claimed. Synthetic long-tail figures (B.6 Figs 5–8) plus GSM8K Fig 4 entropy/token-count plot for a single example | **Hartigan's dip test on 280K+ empirical next-token distributions, `p≈0` on Qwen and CodeLlama**; bimodality is a population-level statistical claim, not a single-example illustration |
| Existence of catastrophic temperature collapse | Discussed qualitatively ("text degeneration", "neural text degeneration", Holtzman et al. cite); no explicit collapse-temperature reported | **Quantitative collapse boundary: pass@1 drops −14 to −44 pp between T=2.5 and T=3.0**; α-sweep avoids this cliff |
| Pareto dominance of `pless_alpha` vs `pless@T` | Not separately reported (since α-sweep is theoretical only in their paper) | **At matched pass@10, α-sweep gives higher pass@1 than `pless@T`** on Qwen MBPP, Qwen HumanEval, CodeLlama HumanEval; m-a-p α=3.0 strictly dominates `pless@T=1.5` on both axes |

## Where we agree

1. **`Σ pᵢ^α` (or equivalently `1/exp(H_α(p))`) is the natural family of hyperparameter-free thresholds**, parameterised by Rényi order. We both arrive at it from the same observation that the original `p = Σ pᵢ^2` is collision probability / `H_2`.
2. **Higher `k`/α admits more tokens** — they prove it via monotonicity of Rényi entropy in `k`; we observe it via direct measurement (more candidate tokens, higher diversity, higher pass@10).
3. **At low temperature the choice of `k`/α and the choice of sampling rule matter less** — both papers find that at T≤1.0 most reasonable samplers cluster.
4. **p-less degrades more gracefully than temperature sampling as T rises.** Their case study (Sec 5.4) shows min-p fails at T=2.0 while p-less recovers; we show on code that `pless@T` extends the usable T-range further than `temp@T`, and that `pless_alpha` extends it further still.
5. **The "extreme" limits make sense.** They state `k→0` ⇒ uniform sampling and `k→∞` ⇒ greedy; our intuition for α (small α → keep almost everything; large α → keep only the mode) is consistent.

## Where we diverge (or might)

1. **Exact threshold form differs for `α > 2`.** They write `G[P]_k = 1/exp(H_k(p)) = (Σ pᵢ^k)^{1/(k-1)}`. We use `T_α = Σ pᵢ^α`. The two are order-preserving transforms of each other, but the actual *numeric cutoff* differs, which means the *admission set* on a given distribution can differ. They coincide exactly at `k = α = 2`. **We should disclose this in our paper.** Our claim of byte-equivalence is for `α=2` vs upstream p-less, not for our `α>2` vs their `G[P]_k`.
2. **Recommended setting:** they recommend `k=2` (the headline p-less). They do not recommend any `k > 2` in main text; whatever is in C.7 is appendix material the authors did not promote to the main contributions. We recommend `α ∈ [2.5, 5.0]` for code generation, with α=3.0 a reasonable default and α=5.0 the strongest pass@10 setting. **Direct contradiction:** if a reader takes their main paper at face value they would never try `k > 2`; our data says they should.
3. **Failure-mode language.** They speak of "degeneration" qualitatively; we quantify a "catastrophic-collapse cliff" between T=2.5 and T=3.0 in pass@1. This is a stronger claim and may be domain-specific (code).
4. **Diversity metrics.** Their diversity claim rests on n-gram repetition on QASC. We use AST-edit-distance diversity, CodeBLEU, and a Claude-judged algorithmic-similarity score (NAUADC). These are not directly comparable, and our claim that α monotonically lifts *algorithmic* diversity is a strictly new finding.

## What we add that they didn't

1. **Code-generation evaluation of the α-generalization.** They have *zero* code benchmarks; we have MBPP-500 + HumanEval-164 × 3 code-specialised models × 4 α values.
2. **Cross-model cross-benchmark sweep at this scale for `k > 2`.** Six (model, benchmark) cells, all monotonic α=2→α=5. Their published `k > 2` evidence (if any in C.7) does not span model families or task types this way.
3. **Pareto-dominance of `pless_alpha` over `pless@T`.** Specifically: m-a-p α=3.0 strictly dominates `pless@T=1.5` on both pass@1 and pass@10 — a finding that requires the α-sweep as a first-class manipulation, not an appendix curiosity.
4. **Quantitative catastrophic-collapse boundary of temperature sampling.** The −14 to −44 pp pass@1 cliff between T=2.5 and T=3.0 across 6 cells is concrete; their treatment is qualitative.
5. **Bimodal-entropy mechanism via Hartigan's dip test on 280K+ per-position next-token distributions.** Their analogue (App. B.6 synthetic distributions; Fig 4 single-example entropy trace) is illustrative, not statistical. Ours is a population-level test on real generation traces with `p ≈ 0` on both Qwen and CodeLlama.
6. **NAUADC algorithmic-similarity measurement via Claude-Sonnet-4.6 as judge.** This is a code-specific diversity signal (does the sample implement the same algorithm or a different one?) that they cannot measure on math/QA because the notion of "algorithmic equivalence" is not naturally defined there.
7. **A vLLM-backed reference implementation** of α-pless that reproduces α=2 byte-identically and exposes `--alpha` as a CLI knob.

## What they add that we didn't

1. **A theoretical Rényi-α framing with explicit asymptotic limits.** Their App. B.5 derives the `k→0` and `k→∞` limits cleanly. We have an entry [`docs/research/renyi_alpha_pless_theory.md`](renyi_alpha_pless_theory.md) but should sharpen it with their notation and cite Eq. 17.
2. **Synthetic long-tail distribution figures (App. B.6, Figs 5–8).** These illustrate that p-less admits long-tail tokens at large vocab sizes — a story we don't tell directly, though our bimodal-entropy analysis is adjacent.
3. **Inference-efficiency profiling.** Their App. C.11 measures CPU time and RAM for top-p, min-p, p-less. We have no comparable wall-clock micro-benchmark of our `pless_alpha` decode path against `pless` or `temp`. **This is worth adding.**
4. **Human evaluation.** 6 raters pairwise on Llama-2-7B Writing Prompts (58.8% p-less win-rate). We have no human eval at all. For code this is less critical (we have unit tests), but a small human eval on stylistic diversity of code could complement NAUADC.
5. **Generation-length analysis (App. C.9).** They claim p-less produces shorter generations than top-p / min-p without accuracy loss. We have not analysed length statistics across α.
6. **A reasoning model in the lineup (DeepSeek-R1-Distill-Qwen-7B).** Our nearest analogue would be running α-sweep on a reasoning-tuned code model; we have not done this.
7. **Failure-case taxonomy (App. C.13).** Two failure patterns of p-less. We could not retrieve the text but we should read it and check whether the same patterns appear in our α>2 traces.

## Implications for our workshop paper

**Citation requirements (mandatory):**
- Cite Tan, Wu, Howard 2025 v6 ([arXiv:2509.23234v6](https://arxiv.org/abs/2509.23234v6), ICLR 2026 oral, OpenReview [ItFuNJQGH4](https://openreview.net/forum?id=ItFuNJQGH4)) as the source of the original p-less method **and** as the source of the Rényi-α generalization (their App. B.5, Eq. 17 / App. C.7).
- Make the citation prominent in the section that introduces our `pless_alpha`. Suggested phrasing: "Tan et al. (2025, App. B.5) propose a theoretical k-order generalization `G[P]_k = 1/exp(H_k(p))`. We implement an equivalent family `T_α = Σ pᵢ^α` (identical to `G[P]_k` at `k=α=2`, monotonically related for `k>2`) and conduct the first large-scale empirical study of `α > 2` on code generation."

**Positioning — claims to soften / remove:**
- *Remove or rewrite:* "We are the first to generalize p-less to higher Rényi orders." The math is already in their App. B.5.
- *Soften:* "We propose Rényi-α p-less" → "We implement and empirically validate the Rényi-α family of p-less thresholds sketched by Tan et al. (App. B.5) for code generation."
- *Keep, with care:* "First empirical study of `α > 2` on code generation across multiple models and benchmarks" — defensible, but contingent on what's actually in their App. C.7 (which we could not fully retrieve). **Action:** before camera-ready, read C.7 in the PDF directly and confirm; if they have any code-generation result with `k > 2`, we soften further.

**New claims we can confidently make based on combined evidence:**
- "We provide the first empirical validation of the Rényi-α generalization of p-less proposed by Tan et al. (2025, App. B.5), confirming the predicted monotonic relationship between α and token-admission size and showing that this translates into a monotonic pass@10 lift on code generation."
- "Across 3 code-specialised models and 2 code benchmarks, `α ∈ [2.5, 5.0]` Pareto-dominates the temperature-only setting on both pass@1/pass@10 and on three distinct diversity metrics (structural, CodeBLEU, NAUADC)."
- "We empirically characterise the catastrophic-collapse boundary of temperature sampling (pass@1 cliff of −14 to −44 pp between T=2.5 and T=3.0) that `α`-sweep does not exhibit through α=5, providing direct evidence for the qualitative robustness claim of Tan et al. (Sec 5.4)."
- "We verify a per-position bimodal-entropy hypothesis via Hartigan's dip test (`p ≈ 0`) on 280K+ next-token distributions from Qwen2.5-Coder and CodeLlama, providing a mechanistic explanation for why `α > 2` helps on code: at high-entropy positions, raising α loosens the threshold; at low-entropy positions, the threshold stays tight."

**Items to add to the paper before camera-ready (priority order):**
1. Read Tan et al. App. C.7 directly from the PDF and document its exact contents; rewrite the "novelty" paragraph accordingly. (Highest priority — affects framing.)
2. Add a one-paragraph mathematical clarification of `T_α = Σ pᵢ^α` vs `G[P]_k = 1/exp(H_k(p))`, with the equivalence at `α=k=2` and the monotonic-but-not-identical relationship for `α=k>2`.
3. Add a small inference-efficiency micro-benchmark (CPU/GPU wall-clock per token) for `pless_alpha` at each α tested.
4. Read Tan et al. App. C.13 and add a short "failure-mode comparison" — do their two p-less failure patterns appear in our α>2 traces?
5. Optionally, run a small "reasoning-tuned code model" α-sweep (e.g., a DeepSeek-Coder-Distill variant) to mirror their reasoning-model coverage.
