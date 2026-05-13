# Where Does Diversity Live in Code Language Models? Two Probes

**Status.** Phase-1 results. Algosim run on Qwen3-8B is 10 of 14 configs
complete (4 pending). Layer-entropy probe on the Qwen2.5-Coder-7B base/instruct
pair is complete. No decoding-time intervention has been built yet — these are
diagnostic experiments designed to tell us whether intervention is worth
building.

**Headline.** Both probes are *consistent* with the working hypothesis that
RLHF-style instruction tuning produces a localized "commit harder" effect in
the final 1–5 transformer layers, but the magnitude of the recoverable diversity
is **small** in absolute terms. The probes refine, rather than confirm, the
intuition. See **Limits** at the end before quoting any single number.

---

## 1. Context and the question

The decoding literature divides cleanly into work on **correctness**
(temperature, top-p, top-k, beam search, etc.) and work on **diversity** —
which has historically been a creative-writing concern. For *code* generation,
diversity matters because of the agentic generate-many-pick-best pattern: a
model that produces 10 textually different but algorithmically identical
solutions gives the chooser nothing to choose from.

Two known facts motivate the experiments:

1. **RLHF / preference-tuning reduces output diversity.** Documented in [Kirk
   et al. (ICLR 2024)](https://arxiv.org/abs/2310.06452) and reframed as
   "typicality bias" in [Verbalized Sampling (Zhang et al. 2025)](https://arxiv.org/abs/2510.01171).
2. **Intermediate transformer layers carry meaningfully different
   next-token distributions** from the final layer. Demonstrated by the
   [Tuned Lens (Belrose et al. 2023)](https://arxiv.org/abs/2303.08112) and
   exploited for *factuality* by [DoLa (Chuang et al. ICLR 2024)](https://arxiv.org/abs/2309.03883),
   which contrasts late- and early-layer logits.

This raises an obvious question that, as far as we can find, no published
work has answered for code:

> Does an instruction-tuned code model carry "more diverse" next-token
> beliefs in its penultimate (or earlier) layers than in its final layer,
> in a way that could be sampled from to recover algorithmic diversity
> without sacrificing correctness?

The two experiments below approach this from two ends:

- **Experiment A (algosim on Qwen3-8B split decoding)** asks what the *output*
  diversity landscape looks like across different sampling strategies — including
  Qwen3's thinking mode and per-phase sampler swapping. This is "diversity from
  the outside" — we look at the generations.
- **Experiment B (layer-entropy probe on Qwen2.5-Coder-7B base vs instruct)**
  asks where in the network the diversity loss is *located*. This is
  "diversity from the inside" — we look at the per-layer next-token distributions.

---

## 2. Experiment A — Algorithmic diversity across split-decoding strategies

### 2.1 Setup

**Model.** Qwen3-8B with reasoning toggle on, generating both `<think>…</think>`
and the code answer.

**Split-decoding mechanism.** We control the sampler and temperature
**independently for the two phases**. While the model is in the `<think>`
phase, sampler-`A` at temperature `T_A` is active; once `</think>` is emitted,
sampler-`B` at `T_B` takes over for the code phase. See
`bench/generator.py:generate_samples_split` (lines 497–650).

**Samplers in scope:**
- `temp_pure`: pure temperature scaling, no top-p/top-k filter
- `pless`: collision-entropy thresholding, the project's main method
- (`temp_standard` — top_p=0.95 + top_k=20 nucleus — deliberately excluded
  from the algosim run after pilot, to keep the comparison free of confounds)

**Configurations.** 14 configs span four families:
- **Baselines**: A (temp 0.7, no thinking), B (pless 0.7, no thinking),
  C (temp_think 0.6), D (pless_think 0.6), E (pless_norm_think 0.6)
- **Uniform high-temp thinking**: T15N (native HF temp 1.5), P15 (uniform pless 1.5)
- **Pure split baseline**: T15P (pure 1.5 → pure 1.5, no pless anywhere)
- **Pure split + pless on code**: H7P, H8P, H9P, H10P at code pless ∈ {1.0, 1.5, 2.0, 3.0}
- **Pure split stress tests**: H11P, H12P (think temperature 2.0, 2.5; code pless 3.0)

Naming map: `bench/eval/split_decoding_analysis.py:CONFIGS`.

**Benchmark.** Full MBPP-500 (500 tasks × 10 samples per config = 5,000
generations per config).

**Diversity metrics.**
- *Surface* (existing): `struct_div` (1 − mean self-AST-fingerprint similarity)
  and `codebleu_div` (1 − mean self-CodeBLEU). See `bench/eval/metrics.py`.
- *Algorithmic* (new): **NAUADC** (Normalised Area Under the Algorithmic
  Diversity Curve), **EA** (effective number of algorithms), **DA@K**
  (number of distinct algorithms within K samples). Defined in [Lee et al.,
  EMNLP 2025 Findings (arXiv:2503.00691)](https://arxiv.org/abs/2503.00691)
  and computed via their public repo [`sh0416/algosim`](https://github.com/sh0416/algosim).
  The clustering is performed by Llama-3.1-8B-Instruct as a judge model.

Algosim is run **only on functionally-correct samples** (per-task `pass_results`
from our existing metrics). Tasks with zero correct samples are dropped. This
matches the paper's protocol and isolates algorithmic-vs-syntactic diversity
on outputs that actually solve the problem.

Code reuse for the export step: `bench/eval/algosim_export.py` builds the
parquet requests; `bench/eval/algosim_report.py` recomputes per-config
NAUADC/EA/DA from response parquets (algosim's own `compute_metrics.py` pools
by problem-ID prefix and would mix our 10 configs together, so we replace it).

### 2.2 Results (10 of 14 configs evaluated)

The full table lives at
`results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis/algosim_full_comparison.md`.
The configs *with* NAUADC, sorted to expose the cleanest comparisons:

| | Config | label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC |
|---|---|---|---:|---:|---:|---:|---:|
| **No-think baseline** | A | temp 0.7 | 0.662 | 0.734 | 0.057 | 0.137 | 1.096 |
| **Think baseline** | C | temp_think 0.6 | 0.738 | 0.834 | 0.167 | 0.354 | 1.234 |
| **Uniform pless think** | P15 | uniform pless 1.5 | 0.824 | 0.898 | 0.159 | 0.296 | 1.222 |
| **Pure split, no pless** | T15P | pure 1.5 → pure 1.5 | 0.801 | 0.882 | 0.208 | 0.390 | 1.303 |
| **Pure split + pless code (sweet spot)** | **H8P** | pure 1.5 → pless 1.5 | 0.811 | 0.898 | 0.206 | 0.384 | **1.322** |
| Pure split + pless code | H9P | pure 1.5 → pless 2.0 | 0.807 | 0.908 | 0.217 | 0.401 | 1.319 |
| Pure split + pless code | H7P | pure 1.5 → pless 1.0 | 0.805 | 0.910 | 0.214 | 0.398 | 1.312 |
| Pure split + hot pless code | H10P | pure 1.5 → pless 3.0 | 0.803 | 0.906 | 0.202 | 0.389 | 1.283 |
| Stress: hot thinking | H11P | pure 2.0 → pless 3.0 | 0.458 | 0.802 | 0.201 | 0.370 | 1.198 |
| Stress: very hot thinking | H12P | pure 2.5 → pless 3.0 | 0.242 | 0.736 | 0.201 | 0.327 | 1.145 |

### 2.3 What we found

**(a) Thinking adds ~0.14 NAUADC over no-thinking** (A 1.096 → C 1.234,
holding model and benchmark constant). This is by far the largest single
intervention in the table — bigger than any sampler choice within the
thinking-enabled regime.

**(b) Within the pure-temp split family with thinking fixed at temp 1.5,
NAUADC is *not monotonic* in code-phase pless temperature.** H7P/H8P/H9P
cluster at 1.312–1.322 (Δ ≤ 0.01, plausibly within run-to-run noise) and H10P
drops to 1.283 — about a 0.04-nat drop, ~4× the intra-cluster spread.

**(c) Pless on the code phase appears to add a small algorithmic-diversity
bump over plain temperature on the code phase**, holding the thinking
configuration fixed. H8P (pless code 1.5) vs T15P (pure temp code 1.5):
**NAUADC 1.322 vs 1.303 (+0.019).** The same comparison on pass@10:
**0.898 vs 0.882 (+0.016 absolute, +1.8% relative).** Direction is consistent
across H7P/H8P/H9P; magnitude is small.

**(d) The cross-metric divergence is most visible at the hot code-pless end.**
H10P's struct_div (0.202) sits just below the H7P/H8P/H9P band (0.206–0.217),
but its NAUADC drop is sharper than the surface metric would predict. The LLM
judge appears to detect algorithmic convergence on tasks where the syntactic
form is still varied. We have not yet inspected per-task examples to localize
this effect.

**(e) The "high temperature keeps diversity even when pass@1 collapses"
claim does *not* hold in the stress regime.** H11P and H12P show that as
thinking temperature is raised past 1.5 (with code pless held at 3.0), pass@1
and NAUADC fall together: 0.458/1.198 and 0.242/1.145 respectively. Surface
metrics (struct_div ≈ 0.20) stay roughly flat, but both pass@k and
algorithmic-cluster count drop. This is a meaningful pushback on the more
optimistic readings of Lee et al. — at least for code at extreme thinking
temperatures, surface diversity is preserved while algorithmic diversity is not.

### 2.4 Limits

- **Single model, single benchmark.** Qwen3-8B on MBPP-500 only. The strong
  thinking-vs-no-thinking effect (a) is well-supported; the smaller pless-on-code
  effect (c) is on the order of plausible cross-run noise and should be
  replicated before any strong claim.
- **Llama-3.1-8B-Instruct as the judge.** Different judge model would give
  different absolute NAUADC. Relative ordering should be more robust but we
  haven't validated.
- **4 configs pending** (B, D, E, T15N). B in particular would quantify the
  "no-thinking pless mode-collapse floor" that surface metrics (struct_div
  0.007 for B vs 0.057 for A) already strongly suggest.
- **NAUADC is sensitive to cluster-count noise.** Mean clusters per task are
  small (1.30–1.42 across the entire table). Differences of 0.01 NAUADC
  correspond to roughly 0.02 extra distinct algorithms per task averaged
  over 25 K-values — small absolute movements.

---

## 3. Experiment B — Layer-entropy probe on Qwen2.5-Coder-7B (base vs instruct)

### 3.1 Setup

**Models.** Qwen2.5-Coder-7B and Qwen2.5-Coder-7B-Instruct. Same architecture
(28 transformer blocks), same tokenizer, same pretraining — the only
difference is the post-training (instruction tuning + RLHF). This is the
clean experimental contrast.

**Data.** 164 HumanEval prompts, 1 correct sample per (task, model) drawn
from our existing temperature-0.7 generation results
(`results/pless_human_eval_results/temprature_results/`).

**Method.** Teacher-force the (prompt + correct code) pair through each model
in a single forward pass, with forward hooks on every transformer block
capturing the residual stream output. For each captured residual, project
through the model's final RMSNorm + LM head (raw logit lens — no learned
adapter; see the [Tuned Lens paper](https://arxiv.org/abs/2303.08112) for
the warning that raw logit lens can be unfaithful for some architectures —
Qwen2.5 turned out to be readable enough, see §3.4).

For each *code-token position* at each layer, we record entropy, KL divergence
to the final layer's distribution, and whether the layer's argmax matches the
final's argmax. Implementation: `bench/eval/layer_entropy_probe.py`.

**Aggregation.** Per-layer overall and per-AST-phase (signature / body /
docstring / operator), using the same tree-sitter phase classifier as
`bench/eval/phase_entropy_probe.py`.

### 3.2 The layer-by-layer picture

```
        instruct H   base H    inst top-1 agreement with final
layer 22   2.98       4.21        0.38
layer 23   1.85       3.07        0.54
layer 24   1.12       2.29        0.67
layer 25   0.44       0.94        0.86
layer 26   0.094      0.189       0.979      ← penultimate
layer 27   0.073      0.209       1.000      ← final
```

The full curve (all 28 layers, all four AST phases) is in
`results/layer_entropy_probe/Qwen2.5-Coder-7B-Instruct/layer_entropy_curve.png`
and the base-vs-instruct overlay in
`results/layer_entropy_probe/compare_instruct_vs_base/compare_entropy.png`.

### 3.3 What we found

**(a) The instruct model's final layer is dramatically more peaked than the
base model's.** Final-layer entropy on code tokens: instruct **0.073 nats**, base
**0.209 nats** — instruct is **~3× more confident**. This is the largest single
effect in the experiment.

**(b) The sign of the penult-vs-final gradient flips between models.**

```
base:     penult 0.189 → final 0.209  (entropy rises by +0.020 at the final layer)
instruct: penult 0.094 → final 0.073  (entropy falls by  -0.021 at the final layer)
```

The base model continues to "open up" slightly at its final layer. The instruct
model does the opposite — its **final transformer block specifically commits**,
collapsing entropy by ~22%. This is the cleanest "RLHF localised in the last
layer" signature we found.

**Δ_gap = (penult−final, instruct) − (penult−final, base) = +0.041 nats.** That
is the headline number for the "RLHF compresses the final 1–2 layers" hypothesis.

**(c) But the sharpening is also distributed across the last 4–5 layers.**
The instruct model's entropy is *uniformly* lower than the base's from layer 22
onwards, by a roughly stable ~2× factor. The instruct model "commits earlier"
across the entire late stack, not only in the last block. The hypothesis "RLHF
damage lives entirely in the last 1–2 layers" is half-right; the *cleanest
isolatable step* is the last layer, but the *total* compression is broader.

**(d) Penultimate-layer sampling is feasible but the diversity reservoir is
small.** At layer 26 (instruct), 97.9% of code-token positions have the same
argmax as the final layer — sampling from layer 26 produces the same answer
~98% of the time. The 2.1% that differ are where any recoverable diversity
would live. Per AST phase, the disagreement rate at layer 26 is:

| Phase | inst top-1 disagreement w/ final |
|---|---:|
| signature | 2.9% |
| operator  | 2.2% |
| body      | 1.9% |
| docstring | 1.3% |

**(e) Layer 25 is a more interesting candidate.** At layer 25 (instruct),
top-1 agreement falls to 86%, but the per-phase disagreement rates are
**much larger**: 17.9% on signatures, 14.1% on body tokens. That's roughly
7× the body-token lever vs layer 26. This is the layer most likely to give
*alternative algorithms* if any layer can — but it also risks producing
sequences far enough from the model's greedy answer to hurt pass@k.

### 3.4 Sanity checks

- **Raw vs tuned logit lens.** We used the raw lens (final RMSNorm + LM head
  applied to every layer's residual). This is known to be unfaithful on
  some architectures; for Qwen2.5 the curves are smooth and the top-1
  agreement is monotonic in layer index — the signal we report does not
  obviously depend on lens fidelity, but a tuned-lens replication is the
  natural next robustness check.
- **Single sample per (task, model).** The aggregates (~30 K code-token
  positions per model) are large enough that per-token entropy means are
  stable. We have not run with multiple samples per task; this would let
  us separate "where the model's *belief* is uncertain" from "where the
  model is sampling around its own belief."

### 3.5 Limits

- **One model pair, one benchmark.** Qwen2.5-Coder-7B on HumanEval. The
  effect direction is unambiguous; the *magnitude* should not be quoted
  across model families without replication.
- **Absolute entropies are small.** 0.094 nats at layer 26 is still
  extremely peaked — almost all probability on the top token. The
  "diversity recovered" by sampling from layer 26 is per-token small.
  Whether it compounds across a 200-token solution into *algorithmically*
  different code is the open question — surface counts are not enough.
- **We have not verified that the layer-26 disagreements are meaningful.**
  An obvious failure mode is that the disagreements concentrate on
  semantically-equivalent token variants (whitespace, alternative
  spellings, equivalent operator orderings) rather than alternative
  algorithmic choices. A small disagreement audit
  (`top1_id_at_layer_26 ≠ top1_id_at_layer_27`, on body-phase positions
  in instruct) is the next sanity check before designing an intervention.

---

## 4. Putting the two experiments together

The algosim experiment (A) measures the *consequence* of post-training:
the instruct-style Qwen3-8B, even with thinking enabled and aggressive
samplers, only achieves NAUADC 1.32 — about 0.8 distinct algorithms over
the model's typical single-cluster output (since most tasks come back as
1–2 clusters). The interventions in the sampler space are small.

The layer-entropy experiment (B) tries to localize *where* the compression
happens. The answer is "concentrated in the last 1–2 layers, but distributed
across the last 4–5." That distributed picture matters: it means a
penultimate-only intervention (sampling from layer 26 instead of 27) is
a very small lever — the *total* sharpening between the model's "uncertain"
state at layer 22 and its peaked state at layer 27 happens over five
layers, not one.

The natural decoding-time intervention is therefore a **layer-mix** sampler
rather than a layer-swap:

> `logits' = (1 − α) · logits_final + α · logits_layer25`

with `α` swept on a held-out task set. At α = 0 we recover vanilla sampling;
at α = 1 we sample from layer 25's distribution. Layer 25 (rather than 26)
because the per-phase disagreement rates are large enough to plausibly
yield alternative algorithms rather than alternative tokens.

We have **not** built this yet. The right next steps before doing so:

1. **Disagreement audit on layer 26 / 25.** Are the disagreements semantically
   meaningful or just trivial variants? Cheap inspection of the
   `per_token_data.csv` outputs.
2. **Tuned-lens replication.** Fit a single-layer adapter at layer 25 and
   verify the disagreement structure survives. If yes, raw lens was fine.
3. **Cross-model replication of (B).** Re-run the probe on Qwen3-8B
   (with thinking disabled to match the architecture) and CodeLlama-7b
   (base + instruct) to test whether the "last-layer commit" signature
   is a property of RLHF in general or of Qwen2.5 specifically.

After those checks, the layer-mix sampler is straightforward to wire as an
additional sampler in `bench/generator.py` alongside `temp_pure` / `pless`,
and the existing MBPP / HumanEval / algosim pipeline evaluates it directly.

---

## 5. Reproducibility

All code and configuration for both experiments is in the project repository:

- **Experiment A (algosim)**:
  - Export: `bench/eval/algosim_export.py`
  - Per-config metrics: `bench/eval/algosim_report.py`
  - Joined comparison table + bar chart: `bench/eval/algosim_full_comparison.py`
  - GPU runner: `run_algosim_judge_qwen3.sh`
  - Source JSONLs: `results/pless_full_mbpp_results/Qwen--Qwen3-8B/*.jsonl`
  - Output: `results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis/algosim_*`
- **Experiment B (layer entropy)**:
  - Probe: `bench/eval/layer_entropy_probe.py`
  - Compare: `bench/eval/layer_entropy_compare.py`
  - Source JSONLs: `results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B*/humaneval/temp_t0.7.jsonl`
  - Output: `results/layer_entropy_probe/{Qwen2.5-Coder-7B,Qwen2.5-Coder-7B-Instruct,compare_instruct_vs_base}/`

## 6. References

- Kirk et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR 2024. [arXiv:2310.06452](https://arxiv.org/abs/2310.06452).
- Zhang et al. *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity.* 2025. [arXiv:2510.01171](https://arxiv.org/abs/2510.01171).
- Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* 2023. [arXiv:2303.08112](https://arxiv.org/abs/2303.08112).
- Chuang et al. *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models.* ICLR 2024. [arXiv:2309.03883](https://arxiv.org/abs/2309.03883).
- Lee et al. *How Diversely Can Language Models Solve Problems? Exploring the Algorithmic Diversity of Model-Generated Code.* EMNLP 2025 Findings. [arXiv:2503.00691](https://arxiv.org/abs/2503.00691).
