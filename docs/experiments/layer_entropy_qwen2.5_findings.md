# Layer-Entropy Probe: Locating RLHF-Induced Diversity Compression in Qwen2.5-Coder-7B

**One-sentence summary.** The instruct-tuned Qwen2.5-Coder-7B's final
transformer block is dramatically more peaked on code tokens than the
base model's (entropy 0.073 vs 0.209 nats); the sign of the
penultimate-to-final entropy gradient *flips* between the two models
(base entropy still rising into the final layer; instruct entropy falling),
producing a +0.041 nat "RLHF layer signature." But the total sharpening
is **distributed across the last 4–5 layers**, not just the final one, and
the per-token disagreement between penultimate and final is small enough
(~2%) and dominated by trivial variants enough that a "sample from
penultimate to recover diversity" intervention has a real but modest
diversity reservoir to work with.

**Status.** Probe complete on both Qwen2.5-Coder-7B (base) and
Qwen2.5-Coder-7B-Instruct. No decoding-time intervention built yet.

## 1. Question

Two well-supported observations from the literature motivate the probe:

1. **RLHF / preference-tuning reduces output diversity.** Demonstrated for
   conversational models in
   [Kirk et al. (ICLR 2024)](https://arxiv.org/abs/2310.06452); attributed
   to "typicality bias" in human preference data by
   [Verbalized Sampling (Zhang et al. 2025)](https://arxiv.org/abs/2510.01171).
2. **Different transformer layers carry meaningfully different next-token
   beliefs.** Demonstrated by the
   [Tuned Lens (Belrose et al. 2023)](https://arxiv.org/abs/2303.08112) and
   exploited for *factuality* by
   [DoLa (Chuang et al. ICLR 2024)](https://arxiv.org/abs/2309.03883), which
   contrasts late- and early-layer logits.

If RLHF damage to diversity is **localised to the final 1–2 transformer
blocks**, then sampling from a slightly earlier layer's distribution could in
principle recover that diversity, since the model's earlier-layer beliefs
would still resemble the pre-RLHF base.

This probe is the prerequisite: before designing such a sampler, we want to
*see* whether the diversity loss is actually layer-localized for at least one
RLHF-tuned code model.

## 2. Setup

**Model pair.** Qwen2.5-Coder-7B and Qwen2.5-Coder-7B-Instruct. Identical
architecture (28 transformer blocks, hidden 3584, vocab 152,064), identical
tokenizer, identical pretraining — the only difference is the post-training.
This is the cleanest controlled comparison available.

**Data.** 164 HumanEval prompts, one correct sample per (task, model) drawn
from our existing temperature-0.7 generations at
`results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B*/humaneval/temp_t0.7.jsonl`.
Functional correctness is re-verified per sample with `check_sample` (timeout
5 s) before use.

**Method.** Teacher-force the (prompt + correct code) pair through each model
in a single forward pass with hooks on every transformer block
(`bench/eval/layer_entropy_probe.py:teacher_forced_per_layer`). For each
captured residual we apply the model's final RMSNorm followed by its LM head
(raw logit lens — no learned adapter). At every code-token *prediction
position* we record per-layer entropy, KL divergence to the final layer's
distribution, the layer's argmax, and whether that argmax matches the final
layer's argmax.

**Aggregation.** Overall and per AST phase (signature / body / docstring /
operator), using the same tree-sitter classifier as
`bench/eval/phase_entropy_probe.py`. About 30 K code-token prediction
positions per model after filtering.

## 3. Results — the layer-by-layer picture

| Layer | inst H | base H | inst top-1 agrees w/ final | base top-1 agrees w/ final |
|---:|---:|---:|---:|---:|
| 22 | 2.98 | 4.21 | 0.38 | 0.39 |
| 23 | 1.85 | 3.07 | 0.54 | 0.54 |
| 24 | 1.12 | 2.29 | 0.67 | 0.67 |
| **25** | **0.44** | **0.94** | **0.86** | **0.82** |
| **26** | **0.094** | **0.189** | **0.979** | **0.954** |
| 27 | 0.073 | 0.209 | 1.000 | 1.000 |

Companion overlay plot:
`results/layer_entropy_probe/compare_instruct_vs_base/compare_entropy.png`.

## 4. Findings

### 4.1 The final layer of the instruct model is 3× more peaked than the base's

Final-layer entropy on code tokens averaged over ~30 K positions: **0.073
nats** (instruct) vs **0.209 nats** (base). Both are extreme — 0.073 nats
corresponds to ~99% probability mass on a single token. But the *3× ratio*
is the cleanest single-number summary of "RLHF makes the model commit
harder on code."

### 4.2 The penultimate-to-final entropy gradient flips sign between models

```
base:     penult 0.189 → final 0.209   (entropy rises by +0.020 at the final layer)
instruct: penult 0.094 → final 0.073   (entropy falls by  -0.021 at the final layer)
```

The base model continues to *open up* very slightly at its final layer.
The instruct model does the opposite — its **final transformer block
specifically commits**, collapsing entropy by ~22%. The sign-flip is the
cleanest evidence that *something specific* happens at the very last block
in the instruct model.

The headline statistic from `compare_summary.json`:

```
Δ_gap = (penult−final, instruct) − (penult−final, base) = +0.041 nats
```

### 4.3 But the sharpening is also distributed across layers 23–27

Plotting the two models' entropy curves shows they are roughly parallel
through layer 22, then diverge — the instruct model compresses entropy
about twice as fast as the base from layer 23 onwards:

| Layer | inst H | base H | ratio |
|---:|---:|---:|---:|
| 23 | 1.85 | 3.07 | 0.60 |
| 24 | 1.12 | 2.29 | 0.49 |
| 25 | 0.44 | 0.94 | 0.47 |
| 26 | 0.094 | 0.189 | 0.50 |
| 27 | 0.073 | 0.209 | 0.35 |

The *cleanest isolatable step* is layer 26 → 27 (where the sign flips), but
the *total* compression effect spans the last five layers. The "RLHF damage
lives only in the last 1–2 layers" framing is half-right; a useful
intervention may need to act on more than just the penultimate.

### 4.4 Penultimate-layer sampling has a small recoverable-diversity reservoir

At layer 26 (instruct), 97.9% of code-token positions have the same argmax
as the final layer. The 2.05% that differ are where any recoverable
diversity would live. Per AST phase the disagreement rate is:

| Phase | inst top-1 disagreement (layer 26 vs 27) |
|---|---:|
| signature | 2.9% |
| operator  | 2.2% |
| body      | 1.9% |
| docstring | 1.3% |

### 4.5 Most layer-26 disagreements are trivial variants, with a minority of real algorithmic alternatives

We pulled the 112 body-phase layer-26 / layer-27 disagreements from the
instruct probe and decoded the alternative tokens. A representative slice:

```
context tok       final picks    penult picks      character
" '"               " '"          " '':\n"         whitespace / newline
'           '      '       '     '           '    whitespace count
' num'             ' number'     ' num'           identifier suffix
' float'           ' numbers'    ' float'         variable name
' if'              ' if'         ' result'        keyword vs identifier ★
'[n'               '[n'          '.get'           subscript vs .get() ★
' True'            ' start'      ' True'          variable name
',\n'              ','           ',\n'            newline placement
```

The two starred rows (★) are the kind we hoped to find: meaningfully
different next-token choices that would drive the generation toward an
alternative algorithm path (a `.get` lookup instead of a subscript, or
branching on a temporary `result` instead of inlining the conditional).
Most of the disagreements, however, are whitespace, formatting, or
identifier-spelling variants that would not produce algorithmically
different code.

We did not run a quantitative classification of these disagreements yet;
on eyeball, perhaps 1 in 5 of the body-phase disagreements looks
algorithmically meaningful. Of the ~30 K body-token positions, that
suggests ~120 positions across the 164-task corpus where layer-26
sampling would route to a real alternative — about **0.7 such positions
per task** if uniformly distributed.

### 4.6 Layer 25 has a larger lever but with more risk

At layer 25 (instruct), top-1 agreement falls to 86%; per-phase disagreement
rates are 17.9% (signature), 14.1% (body), 9.3% (docstring), 6.2% (operator)
— roughly **7× the body-token rate of layer 26**. With more positions in
play, the proportion of *meaningful* alternative tokens is also likely
higher (because the per-token entropy is enough — 0.41 nats on body — to
have real competing top-K choices).

Layer 25 is the more interesting target for a future sampling intervention,
but at 14% off-greedy on body it carries a real risk of dropping pass@k.
The right exploration is a knob that *mixes* layer 25 (or 26) logits into
the final layer's distribution rather than a hard swap.

## 5. Sanity checks

- **Raw vs tuned logit lens.** We used the raw lens (apply the model's
  final RMSNorm + LM head to every layer's residual). The Tuned Lens paper
  notes that raw logit lens is unfaithful on some architectures (BLOOM,
  GPT-Neo). For Qwen2.5 the entropy curves are smooth and top-1 agreement
  is monotonic in layer index, which is consistent with the raw lens being
  faithful enough on this architecture. A tuned-lens replication at one
  or two layers (e.g., 25 and 26) would tighten this; we have not done it.
- **Single sample per (task, model).** With ~30 K code-token positions per
  model, the per-layer means are stable. We have not run multiple samples
  per task, which would let us distinguish "where the model's *belief* is
  uncertain" from "where the model is sampling around its own belief."

## 6. Limits

- **One model pair, one benchmark.** Qwen2.5-Coder-7B on HumanEval only.
  The +0.041 nat "RLHF layer signature" should not be assumed to generalise
  to other model families (Llama-3, CodeLlama, Mistral-Coder, Qwen3 with
  thinking on) without replication.
- **Absolute entropies are very small in the late layers.** Final-layer
  entropy 0.073 nats means ≥ 99% mass on a single token at the typical
  code-prediction position; the entire "diversity reservoir" at layer 26
  amounts to ~2% of token positions having an alternative top-1.
- **Most layer-26 disagreements are trivial token variants.** We did not
  quantify this rigorously; an eyeball read on body-phase disagreements
  suggests roughly 1 in 5 are algorithmically meaningful, ~120 positions
  in the 164-task corpus.
- **The probe measures *model belief*, not what would emerge from
  *autoregressive sampling*** from a hypothetical layer-mix sampler.
  Whether sampling from layer 26 actually changes the algorithm produced
  by token N+1 (and subsequent tokens) requires building and running
  such a sampler.

## 7. What this enables

The numbers above are **necessary but not sufficient** to commit to a
layer-mix decoding intervention:

1. The "RLHF compresses the late layers" hypothesis is *directionally*
   supported on this model pair — the sign-flip at the final layer and
   the consistent 2× entropy ratio across layers 23–27 are clean.
2. The recoverable diversity at the penultimate is small but non-zero,
   and at least some of the disagreements look algorithmically meaningful.

Open prerequisites before building a layer-mix sampler:

- **Quantitative classification of layer-26 disagreements.** Sample ~100
  body-phase layer-26 disagreements, classify each as "trivial variant"
  vs "alternative algorithmic choice," report the proportion.
- **Cross-model replication.** Re-run the probe on at least one other
  base/instruct pair (e.g., CodeLlama-7b vs CodeLlama-7b-Instruct-hf) to
  test whether the sign-flip generalises or is a Qwen2.5-specific artifact.
- **Tuned-lens replication at layer 25.** Confirm that the layer-25
  disagreement structure survives a learned-adapter projection.

## 8. Reproducibility

| Step | Path |
|---|---|
| Probe module | `bench/eval/layer_entropy_probe.py` |
| Base/instruct overlay module | `bench/eval/layer_entropy_compare.py` |
| Source generations | `results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B{,-Instruct}/humaneval/temp_t0.7.jsonl` |
| Per-token CSV (one row per task × sample × code-token × layer) | `results/layer_entropy_probe/Qwen2.5-Coder-7B{,-Instruct}/per_token_data.csv` |
| Per-layer summary statistics | `results/layer_entropy_probe/Qwen2.5-Coder-7B{,-Instruct}/layer_entropy_stats.json` |
| Compare summary + plots | `results/layer_entropy_probe/compare_instruct_vs_base/` |

Run sequence on a CUDA GPU (≥ 24 GB VRAM):

```bash
uv run python -m bench.eval.layer_entropy_probe \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --results-file results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/temp_t0.7.jsonl \
  --output-dir results/layer_entropy_probe/Qwen2.5-Coder-7B-Instruct \
  --max-tasks 164 --samples-per-task 1

uv run python -m bench.eval.layer_entropy_probe \
  --model Qwen/Qwen2.5-Coder-7B \
  --results-file results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B/humaneval/temp_t0.7.jsonl \
  --output-dir results/layer_entropy_probe/Qwen2.5-Coder-7B \
  --max-tasks 164 --samples-per-task 1

uv run python -m bench.eval.layer_entropy_compare \
  --run-a results/layer_entropy_probe/Qwen2.5-Coder-7B-Instruct \
  --run-b results/layer_entropy_probe/Qwen2.5-Coder-7B \
  --label-a "Qwen2.5-Coder-7B-Instruct" \
  --label-b "Qwen2.5-Coder-7B (base)" \
  --output-dir results/layer_entropy_probe/compare_instruct_vs_base
```

## 9. References

- Belrose et al. *Eliciting Latent Predictions from Transformers with the
  Tuned Lens.* 2023. [arXiv:2303.08112](https://arxiv.org/abs/2303.08112).
- Chuang et al. *DoLa: Decoding by Contrasting Layers Improves Factuality
  in Large Language Models.* ICLR 2024.
  [arXiv:2309.03883](https://arxiv.org/abs/2309.03883).
- Kirk et al. *Understanding the Effects of RLHF on LLM Generalisation and
  Diversity.* ICLR 2024.
  [arXiv:2310.06452](https://arxiv.org/abs/2310.06452).
- Zhang et al. *Verbalized Sampling.* 2025.
  [arXiv:2510.01171](https://arxiv.org/abs/2510.01171).
