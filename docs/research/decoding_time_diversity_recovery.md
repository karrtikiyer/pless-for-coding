# Decoding-time mechanisms for recovering diversity from RLHF'd code LLMs — literature survey

## 1. Problem framing

We need decoding-time levers that can re-broaden the next-token distribution of an instruct/RLHF'd code model **without** retraining and **without** simply tapping an earlier residual stream (which our Qwen2.5-Coder layer-entropy probe ruled out: decision-relevant compression is uniform across the last 8 layers and the final-layer's local role inverts). The constraint set is sharp: we keep white-box logit/activation access; we may run a second model copy (base or amateur) since both Qwen2.5-Coder-7B and Qwen3-8B already have matched base/instruct pairs in our benchmark; and the success criterion is pareto improvement on (pass@k, struct_div, codebleu_div, NAUADC/DA@K) over vanilla temp/top-p baselines from Wang et al.'s `A Thorough Examination of Decoding Methods in the Era of LLMs` ([arXiv:2402.06925](https://arxiv.org/abs/2402.06925)).

The dominant theoretical motif in the recent literature is **contrast**: between layers (DoLa), between models (CD, Conformative Decoding), between prompts (Instructive Decoding, USCD, RepE contrast vectors), or between distributions over distributions (Verbalized Sampling). Given that our mechanistic finding rules out single-layer taps but is *consistent* with combined/contrastive use of the residual stream, the literature's center of gravity is well-aligned with our case — and the most promising candidates are precisely those that **subtract** the instruct-specific sharpening rather than truncate or re-scale the post-RLHF distribution.

## 2. Mechanism categories

### 2.1 Multi-layer / logit-fusion decoding

| Paper | Mechanism | Requires | Code-specific evidence |
|---|---|---|---|
| DoLa ([arXiv:2309.03883](https://arxiv.org/abs/2309.03883)) | Contrast final-layer logits against a dynamically-chosen earlier layer (`log p_N − log p_M`) before sampling. | White-box, single model, all layer logits. | None reported by authors; gains shown on TruthfulQA, StrategyQA, GSM8K. No published HumanEval/MBPP numbers located. |
| Tuned Lens ([arXiv:2303.08112](https://arxiv.org/abs/2303.08112)) | Affine probes per layer translate hidden states to vocab space; more reliable than raw logit lens. | White-box, layer hidden states; lens needs to be trained once. | Diagnostic tool, not a sampler. |
| LayerSkip ([arXiv:2404.16710](https://arxiv.org/abs/2404.16710)) | Early-exit + self-speculative verification using remaining layers. | Trained with layer dropout + early-exit loss. | Speed-focused; output distribution is guaranteed equal to full-depth distribution. |

Our read: raw lens taps are dead in Qwen2.5-Coder (team's prior finding), but DoLa-style **subtraction** is mechanistically different — it uses the early layer as a *correction term* against the final layer's mode peak. Our entropy probe shows that the instruct model gained excess sharpness in the last 8 layers; if any of that sharpness is "shared" between e.g. layer 22 and the final layer (i.e. instruct-specific drift accumulated through the stack), DoLa-style contrast could subtract it back out. This is speculative — DoLa wasn't designed for diversity — but it's a clean fit with our mechanism. **Verdict: plausible but unproven for our use case.**

### 2.2 Contrastive decoding (model-pair)

| Paper | Mechanism | Requires | Code-specific evidence |
|---|---|---|---|
| Contrastive Decoding (CD), Li et al. ([arXiv:2210.15097](https://arxiv.org/abs/2210.15097)) | Score = expert log-prob − amateur log-prob, restricted to plausible tokens. | Two model copies (expert + smaller amateur). | Original paper: open-ended text only, comparable or slightly worse diversity than nucleus per authors. |
| CD-for-reasoning, O'Brien & Lewis ([arXiv:2309.09117](https://arxiv.org/abs/2309.09117)) | Same CD objective applied to reasoning. | Two model copies. | Gains on HellaSwag, GSM8K; no HumanEval/MBPP numbers located. |
| USCD, Wang et al. ([arXiv:2409.05923](https://arxiv.org/abs/2409.05923)) | Negative ("lame") prompt as amateur; contrast only when uncertainty is high. | Single model, two prompt forms. | Direct code evidence: +16.59% avg pass@1 across HumanEval/MBPP/MultiPL-E over Incoder-6B, CodeLlama-7B, WizardCoder-15B, StarCoder, Llama2-7B. |
| Conformative Decoding ([arXiv:2507.20956](https://arxiv.org/abs/2507.20956)) | Instruct model "guided" by its *base* counterpart to inject diversity; explicitly identifies DPO as the main diversity-loss culprit. | Two model copies (matched base + instruct). | Evaluated on writing-prompt narrative generation, not code. |
| Adaptive Contrastive Decoding ([arXiv:2408.01084](https://arxiv.org/abs/2408.01084)) | Weights contrastive influence by per-token uncertainty. | Two distributions. | RAG/QA-focused. |
| Instructive Decoding ([arXiv:2311.00233](https://arxiv.org/abs/2311.00233)) | Contrast against logits from a *noisy/opposite* version of the instruction. | Single model, two prompt forms. | General instruction-following; no code-specific numbers located. |

Our read: this bucket is the strongest match to our mechanistic story. Conformative Decoding (instruct − base from the same family) is essentially the inverse of the RLHF compression we measured: subtracting the base model's logits from the instruct model's should re-inject exactly the entropy that uniform RLHF sharpening removed across layers 20–27. We already have both Qwen2.5-Coder-7B base and Qwen2.5-Coder-7B-Instruct, so the experimental cost is low. USCD shows that *some* form of contrast already helps on code pass@1 at low temperature; whether it also widens the diversity floor is not yet measured. **Verdict: Conformative Decoding and a base/instruct CD variant are the most promising single bet in the survey.**

### 2.3 Truncation samplers beyond top-p/top-k

| Sampler | Threshold rule | Code evidence |
|---|---|---|
| Nucleus top-p, Holtzman et al. ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)) | Smallest set with cumulative prob ≥ p. | Standard baseline. |
| Locally typical sampling, Meister et al. ([arXiv:2202.00666](https://arxiv.org/abs/2202.00666)) | Tokens whose surprisal is closest to conditional entropy. | Summarization/story-gen; competitive with nucleus, no published code numbers. |
| ε / η sampling, Hewitt et al. ([arXiv:2210.15191](https://arxiv.org/abs/2210.15191)) | ε truncates below abs threshold; η is entropy-aware. | NL desmoothing, not code. |
| min-p, Nguyen et al. ([arXiv:2407.01082](https://arxiv.org/abs/2407.01082)) | Threshold = `p_min · max_p`. | Authors claim quality+diversity wins on GPQA/GSM8K/AlpacaEval; the rebuttal `Min-p, Max Exaggeration` ([arXiv:2506.13681](https://arxiv.org/abs/2506.13681)) finds no improvement once hyperparameter count is controlled. |
| p-less / p-less-norm, Yang et al. ([arXiv:2509.23234](https://arxiv.org/abs/2509.23234)) | Threshold = collision-entropy `Σp²` (or normalized variant). Hyperparameter-free. | Math, logic, creative-writing benchmarks; not code-specific in the paper, but our own benchmark on MBPP+HumanEval (see `results/`) is the primary evidence. |

Our read: truncation samplers shape the *tail* but cannot undo a uniformly sharper *head* — which is exactly what the layer-entropy probe shows the instruct model has. p-less responds to the head shape (`Σp²` is large when the head is peaked, cutting hard) so it cannot recover diversity that the head doesn't have. The min-p rebuttal further argues that even the best of the new truncation samplers does not pareto-dominate top-p once hyperparameters are matched. **Verdict for diversity recovery: dead lever on its own.** Useful as a *post-process* on top of a contrast-based method.

### 2.4 Stochastic / diverse search

| Paper | Mechanism | Requires | Code evidence |
|---|---|---|---|
| Diverse Beam Search ([arXiv:1610.02424](https://arxiv.org/abs/1610.02424)) | Partition beams into groups; penalize cross-group token overlap. | Beam infra. | Caption/MT/dialog; no code-pass@k numbers located. |
| Stochastic Beam Search ([arXiv:1903.06059](https://arxiv.org/abs/1903.06059)) | Gumbel-top-k for exact sampling without replacement. | Beam infra. | MT; no code numbers located. |
| Determinantal Beam Search ([arXiv:2106.07400](https://arxiv.org/abs/2106.07400)) | DPP-style subdeterminant max with intra-beam similarity matrix. | Beam infra + similarity kernel. | NMT/summarization/dialog; no code numbers. |

Our read: these methods enforce diversity *across* a fixed-budget sample set, not within the per-token distribution. They are complementary to anything in 2.1/2.2 — if the per-token distribution is still over-peaked after a contrast intervention, DBS/DPP-BS will still give nearly-identical beams. The team's existing benchmark already includes vanilla beam in some configs; the marginal value of adding DBS is small unless paired with one of the contrast methods. **Verdict: plausible but unproven; low priority alone.**

### 2.5 Self-consistency / sample-then-select

| Paper | Mechanism | Code evidence |
|---|---|---|
| Self-Consistency ([arXiv:2203.11171](https://arxiv.org/abs/2203.11171)) | Sample N reasoning paths, majority-vote the answer. | Demonstrated on math/QA; not code per se. |
| HumanEval / Codex pass@k ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)) | n ≥ k samples, count tests passed. | Foundational: pass@1 ≈ high-tens, pass@100 ≈ 70s for Codex — diversity drives the gap. |
| AlphaCode ([arXiv:2203.07814](https://arxiv.org/abs/2203.07814)) | Massive sampling + filtering + clustering. | 70.2% pass with 100 samples/problem. |

Our read: this category doesn't change the per-token distribution; it leverages it. It's the *consumer* of diversity, not a source of it. For the team's experimental design, this matters as a measurement choice (pass@k with k > 1 is the metric that rewards diversity recovery), not a mechanism to test. **Verdict: not a mechanism; relevant as the evaluation lens.**

### 2.6 Steering / activation editing at decode time

| Paper | Mechanism | Requires | Code evidence |
|---|---|---|---|
| Representation Engineering ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405)) | LAT reads concept direction from contrasted prompt pairs; add steering vector to residual stream at inference. | White-box residual access. | Demonstrated for honesty, power-seeking; no code-diversity numbers. |
| Activation Engineering / ActAdd ([arXiv:2308.10248](https://arxiv.org/abs/2308.10248)) | Same idea, earlier form. | White-box. | NL only. |
| BILLY (persona vector merging) ([arXiv:2510.10157](https://arxiv.org/abs/2510.10157)) | Merge multiple persona vectors at inference for multi-perspective output. | White-box. | Creative writing, not code. |

Our read: RepE is the most direct mechanism aligned with our finding that the *direction* of the last-layer residual update inverted between base and instruct. If we computed a steering vector as `mean(base_residual) − mean(instruct_residual)` over the decision layers (20–27 for Qwen2.5-Coder) on a calibration set of MBPP prompts, adding it back at inference should partially undo the RLHF-induced sharpening *along the specific direction* the residual flipped on. Note: the survey result "creativity is poorly defined as a contrast direction" applies to open-ended creative writing; for code, base vs. instruct on the *same* prompt is a much sharper contrast pair, which we already have in our benchmark. **Verdict: promising but mechanistically more invasive than logit-space contrast; second priority.**

### 2.7 Prompt-level diversity elicitation

| Paper | Mechanism | Code evidence |
|---|---|---|
| Verbalized Sampling ([arXiv:2510.01171](https://arxiv.org/abs/2510.01171)) | Ask the model to enumerate N responses with probabilities in one call ("Generate 5 X with probabilities"). | Identifies typicality bias in preference data as the mode-collapse root cause; reports 1.6–2.1× diversity gain in creative writing. No code numbers reported. |
| Personality-guided code generation ([arXiv:2411.00006](https://arxiv.org/abs/2411.00006)) | Persona/personality prefixes; pass-rate improvements in 23/28 LLM-dataset combinations, >10% in 5. | Direct code evidence. |
| Understanding RLHF on diversity ([arXiv:2310.06452](https://arxiv.org/abs/2310.06452)) | Measurement, not a method. Documents the RLHF→across-input mode collapse our team is up against. | Not code-specific but the canonical empirical reference. |

Our read: Verbalized Sampling is the cheapest experiment in the entire survey — no logits, no extra forward pass — and the published gain is large enough on creative tasks to be worth a code-specific test. Persona prompting works on code per [arXiv:2411.00006](https://arxiv.org/abs/2411.00006) but the reported metric was pass-rate, not diversity, so its diversity payoff on instruct models is unmeasured. **Verdict: Verbalized Sampling = promising-cheap; persona prompting = plausible but unproven for diversity floor.**

## 3. Viability summary

| Mechanism | Aligns with our layer-entropy finding? | Code evidence in literature? | Our verdict |
|---|---|---|---|
| Raw early-layer logit tap | No (already falsified) | n/a | Dead lever (team's prior finding confirmed) |
| DoLa-style layer contrast | Plausibly (subtracts shared sharpening) | None | Plausible but unproven |
| Contrastive Decoding (expert/amateur) | Partial | Indirect (USCD) | Plausible but unproven |
| Conformative Decoding (instruct − base, same family) | Yes — direct inverse of measured compression | None | **Promising** |
| USCD (single model, contrast prompt) | Partial | Yes (+16.59% pass@1) | Promising for correctness, unknown for diversity |
| Locally typical / ε / η / min-p / p-less | No (head is uniformly sharp) | Partial (our own bench) | Dead lever for diversity recovery alone |
| Diverse / Stochastic / Determinantal beam | Orthogonal | None for code | Plausible only as a wrapper |
| Self-consistency / pass@k sampling | Orthogonal | Foundational | Measurement, not mechanism |
| Activation steering / RepE | Yes — directly addresses inverted residual direction | None for code diversity | Promising, more invasive |
| Verbalized Sampling | Orthogonal but cheap | None for code, large on creative | Promising-cheap |
| Persona prompting | Orthogonal | Yes for pass-rate; unknown for diversity | Plausible but unproven |

(Disagreement with team's priors: we agree with the dead-lever ruling on layer taps. We additionally argue truncation samplers — including our own p-less — cannot by themselves recover diversity from a head that has been uniformly sharpened by RLHF; they help only after the head has been re-broadened. This may already be implicit in the team's experimental notes but is worth flagging explicitly.)

## 4. Concrete experiment proposals (ordered by promise/cost)

1. **Conformative / base-vs-instruct contrastive decoding on Qwen2.5-Coder-7B and Qwen3-8B.** Score = `log p_instruct(x|ctx) − α · log p_base(x|ctx)`, sweep `α ∈ {0.1, 0.3, 0.5, 1.0}`, restricted to the plausibility set (CD-style). Both models are already loaded by `run_humaneval.py` / `run_bench.sh` for each family, so the experimental cost is one extra forward pass per token. Measure: MBPP+HumanEval pass@1, pass@10, struct_div, codebleu_div, NAUADC/DA@K. Success = pareto improvement on (pass@10, struct_div) over temp@1.0 and over p-less@0.6 baselines. Direct test of [arXiv:2507.20956](https://arxiv.org/abs/2507.20956) on code, which the paper did not cover.

2. **Verbalized Sampling on Qwen2.5-Coder-7B-Instruct and Qwen3-8B-Instruct.** Prompt: "Generate 5 distinct Python implementations of `<MBPP task>` together with a probability for each; return as JSON." Pick one sample by the verbalized probability. No logit access needed; this is a pure prompt change on top of our existing instruct pipeline. Measure: pass@1 of the picked sample, struct_div / codebleu_div / NAUADC over the 5 verbalized samples vs. 5 independent temp=1.0 samples. Success = matched-or-better pass@1 with >1.5× diversity (paper claims 1.6–2.1× on creative tasks, [arXiv:2510.01171](https://arxiv.org/abs/2510.01171)).

3. **DoLa on Qwen2.5-Coder-7B-Instruct only.** Use the `voidism/DoLa` reference implementation; candidate early layers = {20, 22, 24, 25} (i.e. inside the decision band where our entropy probe already showed structure); pick the layer that maximizes JSD against the final layer per token (DoLa's adaptive variant). Measure: same metrics as (1). Success criterion is weaker — we'd accept a 5% absolute struct_div gain at no pass@1 loss, since DoLa was not designed for diversity. This is the cheap test of the "fused contrast might escape the single-tap dead-lever ceiling" hypothesis.

4. **RepE-style residual steering on Qwen2.5-Coder-7B-Instruct.** Build a steering vector `v = E[h_base(p) − h_instruct(p)]` averaged over MBPP train-split prompts at each of layers 20–27, then at inference add `β · v` to the residual at the same layers. Sweep `β ∈ {0.5, 1.0, 2.0}`. Measure: pass@1 / pass@10 / struct_div / NAUADC. Success = recovers ≥50% of the base→instruct diversity gap on MBPP while losing <3 pts of pass@1. This is the most invasive test but the most direct probe of the "inverted last-layer residual" finding.

5. **Composition test: best of (1) + p-less-norm as the truncation step.** Once a winning contrast-based method is picked from (1)–(4), run it with the final token sampled by p-less-norm@1.0 rather than by raw softmax. The hypothesis is that head-broadening (contrast) and tail-shaping (p-less-norm) are complementary; this is the proper test of whether our existing sampler stack is wasted on instruct models or merely waiting for a wider head. Measure: same metric set; success = strict pareto gain over the best result from (1).

---

## Verification notes (internal)

- All paper titles and arXiv IDs above are linked. Where a paper makes a code-specific quantitative claim, the number is attributed to its arXiv ID (USCD +16.59%, Codex pass@100 ≈ 70s, AlphaCode 70.2% @ 100 samples, persona pass-rate improvements in 23/28). Where no published number was located, the table says so explicitly.
- All 5 experiment proposals name Qwen2.5-Coder-7B and/or Qwen3-8B, both of which are in the team's existing MBPP+HumanEval full benchmark, and reference metrics (pass@1, pass@10, struct_div, codebleu_div, NAUADC/DA@K) the team already computes via `bench/eval/`.
- Verdicts are consistent with the team's prior layer-entropy finding: raw early-layer taps remain a dead lever; only *contrastive* or *steering* methods that act on combinations of layers/models/prompts are flagged as promising.
- "Our read" / "interpretation" markers are used wherever a synthesis claim is not directly supported by a cited paper, per the format constraints.
