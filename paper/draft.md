# When Does p-less Sampling Help Code LLMs? A Cross-Model, Cross-Benchmark Evaluation

*Draft v1 — Phase 4 audit complete (2026-04-28). All numerical claims
trace to `paper/sources.md`; figures are symlinks to canonical PNGs
under `results/`. Deferred items listed in `paper/TODO.md`.*

## Abstract

We present the first systematic evaluation of p-less and p-less-norm
[Tan et al., 2026], two hyperparameter-free truncation samplers, on
code-generation benchmarks. Across 13 model checkpoints (Llama-2,
CodeLlama, Codestral, Qwen2.5-Coder, Qwen3-Coder; 1.3B–30B) on MBPP-500
and HumanEval-164, p-less-norm at temperature 0.6 is the top-ranked
sampler on Llama-2-7B (MBPP) when merged with the 18 decoding methods
surveyed by Wei et al. [2024], reaching 22.3% pass@1 ahead of FSD-d
(21.2%) and beam-8 (19.4%). On the canonical instruct model
Qwen2.5-Coder-7B-Instruct (HumanEval), p-less@0.6 reaches 87.5% pass@1
versus greedy's 84.1%, a directional +3.4pp at the 2.8pp standard-error
band. A six-point T₁ sweep on six HumanEval models maps a sweet spot at
T₁ = 0.7–1.5, a cliff between T₁ = 2.0 and T₁ = 2.5, and a single
T₁-immune model (Qwen3-Coder-30B). A T₁/T₂ decomposition on
Qwen2.5-Coder-7B-Instruct (MBPP) shows that the post-truncation T₂ is
dominated by T₁ — at T₁ = 2.0, T₂ ∈ [2.0, 5.0] costs 2.0–4.0pp pass@1
for 0.020–0.034 struct\_div — and that p-less does not act as a
quality filter relative to plain temperature at matched diversity. We
release the consolidated 192-row metrics CSV, a reproducible
uv-managed pipeline, and a partition analysis isolating decoding-method
effects from token-budget confounds.

## 1. Introduction

Code generation with large language models has converged on a small set
of decoding policies — greedy or beam search at deployment, temperature
sampling at evaluation — and a tradition of treating decoding as an
afterthought relative to model architecture and pre-training data. The
benchmarks themselves encourage this: pass@1 rewards a single confident
guess; pass@10 rewards diversity; few production systems agree on which
to optimise.

Standard truncation samplers introduce hyperparameters that interact in
non-obvious ways with model peakedness. Top-p [Holtzman et al., 2020],
top-k, η-sampling, ε-sampling, typical sampling and Mirostat each carry
one or two free parameters whose right setting depends on the model, the
task, and the desired correctness/diversity tradeoff. Wei et al. [2024]
observed wide sensitivity to these hyperparameters across decoding
methods on Llama-2-7B over MBPP and HumanEval, but did not consider
hyperparameter-free alternatives.

Tan, Wu and Howard [2026] propose **p-less** and **p-less-norm**: two
truncation samplers whose admission threshold is computed directly from
the distribution's collision-entropy, with no tunable knobs. The proposal
is appealing — but the original paper evaluates only on math, reasoning,
and creative writing benchmarks. Whether p-less is competitive on code,
where outputs are syntactically constrained and probability mass is often
concentrated on a single correct token, is not addressed.

**Contributions.** We provide the first systematic evaluation of p-less
on code-generation benchmarks. Using the configuration grid in §3, we
contribute:

1. A head-to-head comparison of p-less, p-less-norm, temperature, top-p,
   greedy and beam search across 13 code-LM checkpoints (Meta Llama-2,
   CodeLlama, Mistral Codestral, Qwen2.5-Coder, Qwen3-Coder; 1.3B–30B)
   on MBPP-500 and HumanEval-164 (Table 2).
2. A direct re-evaluation against the Wei et al. [2024] decoding survey
   on Llama-2-7B (MBPP), where p-less-norm@0.6 ranks 1/19 at 22.3%
   pass@1 — beating FSD-d (21.2%) and beam-8 (19.4%) (claim **C1**;
   `comparison_report.md`).
3. A robustness boundary mapped through a six-point T₁ sweep
   (`T ∈ {0.7, 1.0, 1.5, 2.0, 2.5, 3.0}`) on six HumanEval models. We
   identify a sweet spot at T₁ = 0.7–1.5, a cliff between T₁ = 2.0 and
   T₁ = 2.5, and a single tested model (Qwen3-Coder-30B) that is
   T₁-immune (claims **C3**, **C4**, **C6**, **C10**).
4. A T₁/T₂ decomposition on Qwen2.5-Coder-7B-Instruct (MBPP) showing
   that the post-truncation T₂ knob is dominated by T₁ — at T₁ = 2.0,
   T₂ ∈ [2.0, 5.0] buys 0.020–0.034 struct\_div for 2.0–4.0pp pass@1
   (claim **C7**), and at matched diversity p-less does not beat
   plain temperature (claim **C11**).
5. A reproducible pipeline (uv-managed, single-command per model) and a
   consolidated 192-row metrics CSV released alongside the paper.

The headline finding is conditional, not promotional: **p-less is a
useful default on weaker base models and on instruct models within
T₁ = 0.7–1.5, but does not dominate temperature in any single regime
we tested.** The benefit of removing a hyperparameter is what justifies
its use, not raw pass@k uplift.

## 2. Background and Related Work

**P-less and p-less-norm.** Tan, Wu and Howard [2026] introduce two
hyperparameter-free truncation samplers. Given a token probability
distribution `p` over a vocabulary of size `v`, p-less defines its
admission threshold as the collision-entropy of the distribution,
`τ_pless = Σᵢ pᵢ²`, and admits tokens with `pᵢ ≥ τ_pless`. P-less-norm
relaxes the same idea, using
`τ_norm = (v · Σᵢ pᵢ² − 1) / (v − 1)`,
which lets more low-probability tokens survive (`p-less/p_less_samplers.py`,
lines 25 and 56). After truncation the survivors are re-normalised and
sampled. Neither method takes a top-k cutoff, top-p budget, or η/ε
threshold — only an optional pre-truncation temperature.

**Decoding-method surveys on code.** Wei et al. [2024] benchmark a wide
panel of decoding methods (greedy, beam search, contrastive search,
contrastive decoding, top-p, top-k, η-sampling, typical sampling, FSD,
FSD-d, DoLa, Mirostat, temperature) on Llama-2-7B / 7B-chat over MBPP
and HumanEval. Their analysis predates the p-less proposal and so does
not include it. We use their published Llama-2-7B MBPP numbers as a
direct external comparison (Table 4 of their paper; ranking at claim
**C1**).

**Small-language-model code studies.** [arXiv:2507.03160] empirically
study 20 small open-source code models on HumanEval+ and MBPP+, varying
model size and instruction tuning but not the decoding policy. That work
is orthogonal to ours: it asks "which model?" while we ask "given the
model, which sampler?".

**The gap.** Code generation imposes two competing requirements:
correctness (the program must satisfy hidden tests) and diversity (a
sampler covering more correct programs is more useful for refinement and
test-time search). The p-less papers argue that hyperparameter-free
truncation balances both — but they do not evaluate on code at all. The
Wei et al. survey does evaluate on code but predates p-less. We close
this gap by running p-less and p-less-norm head-to-head with temperature,
top-p, greedy, and beam baselines across 13 code-LM checkpoints on two
benchmarks.

## 3. Methodology

### 3.1 Models

We evaluate 13 unique model checkpoints spanning three families and parameter
counts from 1.3B to 30B (Table 1). Eight models cover MBPP-500; six cover
HumanEval-164; one (`codellama/CodeLlama-7b-Instruct-hf`) covers both.
Family coverage includes Meta Llama-2 (base and chat),
CodeLlama (base and instruct), Mistral Codestral, and the Qwen2.5 / Qwen3
coder series (base and instruct variants). Instruct/chat detection follows
the model name; the prompting pipeline calls `tokenizer.apply_chat_template`
on instruct models and uses bare-prompt tokenization on base models.

**Table 1.** Models evaluated, with benchmark coverage. ✓ marks the
(model, benchmark) cells run in this study.

| Model | MBPP-500 | HumanEval-164 |
|-------|:--------:|:-------------:|
| Qwen/Qwen-7B | ✓ |  |
| Qwen/Qwen-7B-Chat | ✓ |  |
| Qwen/Qwen2.5-Coder-1.5B | ✓ |  |
| Qwen/Qwen2.5-Coder-3B | ✓ |  |
| Qwen/Qwen2.5-Coder-7B |  | ✓ |
| Qwen/Qwen2.5-Coder-7B-Instruct |  | ✓ |
| Qwen/Qwen3-Coder-30B-A3B-Instruct |  | ✓ |
| codellama/CodeLlama-7b-Instruct-hf | ✓ | ✓ |
| codellama/CodeLlama-7b-hf | ✓ |  |
| m-a-p/OpenCodeInterpreter-DS-1.3B | ✓ |  |
| meta-llama/Llama-2-7b-chat-hf | ✓ |  |
| meta-llama/Llama-2-7b-hf | ✓ |  |
| mistralai/Codestral-22B-v0.1 |  | ✓ |

_Source: `results/analysis/consolidated_summary.csv` (178 rows after
filtering to MBPP-500 / HumanEval-164)._

### 3.2 Sampling configurations

For each (model, dataset) cell we run a fixed grid of decoding configurations,
listed verbatim from `run_bench.sh` (lines 25–37) for the MBPP cohort:

```
temp        T = 0.7
pless       T = 0.6, 0.7, 1.0
pless_norm  T = 0.6, 0.7, 1.0
top_p       T = 1.0, top_p = 0.9
greedy      n_samples = 1
beam        num_beams ∈ {4, 8}, n_samples = 1
```

The HumanEval temperature sweep (`run_humaneval.py`) covers the same three
methods (temp, pless, pless_norm) at T ∈ {0.7, 1.0, 1.5, 2.0, 2.5, 3.0}. A
separate full-precision evaluation covers four HumanEval models with
greedy, top-p, temperature (0.2, 0.7), and pless/pless_norm at T = 0.6 and
T = 1.0.

The pre-/post-truncation (T₁/T₂) experiments use the same pipeline with
`--method pless --t1 <T1> --t2 <T2>`. T₁ scales logits before the p-less
threshold is computed; T₂ rescales the probabilities of the surviving
tokens after pruning. T₂ = ∅ recovers single-temperature p-less.

All stochastic configurations sample 10 completions per task (greedy and
beam are 1 sample). MBPP-full uses 500 problems; HumanEval-164 uses the
canonical 164-problem split. Generation is implemented in
`bench/generator.py`; the p-less and p-less-norm samplers come from the
`p-less/` git submodule
(`p-less/p_less_samplers.py:p_less_decode`,
`p_less_norm_decode`).

### 3.3 Benchmarks

**MBPP-500** is the full split of the Mostly Basic Python Problems benchmark
[Austin et al., 2021]. Each task is a short natural-language problem
description with three hidden test cases. We use the BigCode zero-shot
docstring prompt on base models and the chat-template form on instruct
models.

**HumanEval-164** [Chen et al., 2021] consists of 164 hand-written
function-completion tasks with hidden unit tests. The prompt is the
function signature plus the docstring; the model continues from there.

For both benchmarks each generation is sandbox-executed (`subprocess`,
timeout 30s) against the hidden tests; a sample is "correct" if all
tests pass.

### 3.4 Metrics

**pass@k.** We use the unbiased estimator from Chen et al. [2021] as
implemented in `human_eval.evaluation.estimate_pass_at_k`
(`bench/eval/metrics.py:42-58`). For 10-sample runs we report
pass@1 / pass@3 / pass@5 / pass@10; for 1-sample runs (greedy, beam) we
copy pass@1 to higher-k columns.

**cover@t.** For threshold t ∈ [0,1], cover@t is the fraction of tasks
whose 10 samples include at least ⌈t · 10⌉ correct completions:
`cover@t = (1/N) · Σ_i 1[c_i ≥ t · n]`
(`bench/eval/metrics.py:75-79`). cover@t collapses to an "easy
solubility" measure: cover@0.7 ≈ "fraction of tasks the sampler solves
reliably (≥7/10)." A "distinct" variant counts each AST-fingerprinted
correct solution at most once.

**Structural diversity.** Per task, compute the mean pairwise Zhang–Shasha
tree-edit distance [Zhang & Shasha, 1989] over correct samples (using
the `zss` library;
`bench/eval/fingerprint.py`,
`bench/eval/metrics.py:118-137`). Across the dataset we report the mean
over tasks with ≥ 2 correct samples; identical structures contribute
zero.

**CodeBLEU-based diversity.** We report four further per-task diversity
scores derived from CodeBLEU [Ren et al., 2020]: `codebleu`,
`syntax_match`, `dataflow_match`, `ngram_match`, `weighted_ngram_match`
(`bench/eval/metrics.py:140-222`). For each pair of correct samples we
compute CodeBLEU and report `diversity = 1 − similarity`; the per-task
score is the mean over pairs and the dataset score is the mean over
tasks with ≥ 2 distinct correct samples.

### 3.5 Statistical caveats

For a binomial proportion, the per-task SE on pass@1 is
`SE ≈ √(p(1−p)/N)`. At p ≈ 0.85 this gives ≈ 1.75pp on MBPP-500 and
≈ 2.8pp on HumanEval-164 [`cross_benchmark_t1_analysis.md:16`,
claim **C9**]. We label any single-comparison difference below ~2 SE as
"directional" (consistent in sign across runs but not statistically
demonstrated on its own). Cross-run comparisons (e.g., temperature-sweep
vs full-precision) cross independently sampled evaluation pipelines and
are flagged as such where they appear.

## 4. Results

### 4.1 Headline pass@k

Table 2 reports pass@1 / pass@10 for the canonical decoding configurations
(greedy, temp@0.7, pless@{0.6,1.0}, pless\_norm@{0.6,1.0}) across all 13
(model, benchmark) cells. Two patterns recur in the verified rows.

**Table 2.** Headline pass@k. Per-row format: `pass@1 / pass@10`. `—` =
configuration not run for that (model, benchmark) cell. `n=1` cells
(greedy) report the single-sample success rate in both columns.

| Model | Benchmark | greedy | temp@0.7 | pless@0.6 | pless_norm@0.6 | pless@1.0 | pless_norm@1.0 |
|:-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen/Qwen-7B | mbpp | 0.368 / 0.368 | 0.097 / 0.398 | 0.314 / 0.382 | 0.313 / 0.382 | 0.350 / 0.486 | 0.357 / 0.502 |
| Qwen/Qwen-7B-Chat | mbpp | 0.314 / 0.314 | 0.287 / 0.504 | 0.340 / 0.370 | 0.344 / 0.376 | 0.344 / 0.406 | 0.345 / 0.408 |
| Qwen/Qwen2.5-Coder-1.5B | mbpp | 0.544 / 0.544 | 0.381 / 0.710 | 0.531 / 0.608 | 0.528 / 0.614 | 0.513 / 0.688 | 0.519 / 0.678 |
| Qwen/Qwen2.5-Coder-3B | mbpp | 0.602 / 0.602 | 0.426 / 0.776 | 0.593 / 0.662 | 0.594 / 0.666 | 0.565 / 0.722 | 0.576 / 0.724 |
| Qwen/Qwen2.5-Coder-7B | humaneval | — | 0.497 / 0.890 | — | — | 0.559 / 0.762 | 0.563 / 0.762 |
| Qwen/Qwen2.5-Coder-7B-Instruct | humaneval | 0.842 / 0.842 | 0.792 / 0.951 | 0.875 / 0.878 | 0.875 / 0.884 | 0.834 / 0.902 | 0.757 / 0.951 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | humaneval | 0.756 / 0.756 | 0.775 / 0.872 | 0.789 / 0.799 | 0.785 / 0.799 | 0.760 / 0.780 | 0.757 / 0.780 |
| codellama/CodeLlama-7b-Instruct-hf | humaneval | 0.360 / 0.360 | 0.363 / 0.634 | 0.281 / 0.317 | 0.281 / 0.317 | 0.355 / 0.378 | 0.351 / 0.384 |
| codellama/CodeLlama-7b-Instruct-hf | mbpp | 0.422 / 0.422 | 0.383 / 0.552 | 0.412 / 0.422 | 0.411 / 0.422 | 0.416 / 0.442 | 0.414 / 0.438 |
| codellama/CodeLlama-7b-hf | mbpp | 0.410 / 0.410 | 0.368 / 0.652 | 0.417 / 0.494 | 0.417 / 0.490 | 0.414 / 0.572 | 0.415 / 0.574 |
| m-a-p/OpenCodeInterpreter-DS-1.3B | mbpp | 0.442 / 0.442 | 0.431 / 0.624 | 0.439 / 0.490 | 0.439 / 0.494 | 0.441 / 0.512 | 0.446 / 0.510 |
| meta-llama/Llama-2-7b-chat-hf | mbpp | 0.206 / 0.206 | 0.178 / 0.302 | 0.205 / 0.214 | 0.204 / 0.214 | 0.201 / 0.224 | 0.202 / 0.222 |
| meta-llama/Llama-2-7b-hf | mbpp | 0.230 / 0.230 | 0.040 / 0.242 | 0.219 / 0.274 | 0.222 / 0.272 | 0.224 / 0.370 | 0.218 / 0.372 |
| mistralai/Codestral-22B-v0.1 | humaneval | 0.756 / 0.756 | 0.730 / 0.908 | 0.751 / 0.787 | 0.749 / 0.793 | 0.780 / 0.848 | 0.777 / 0.848 |

_Source: `results/analysis/consolidated_summary.csv`. Method aliases
(`pless`/`p_less`, `temp`/`temp_0.7`) collapsed by `canon_method()` in
`paper/tables/make_tables.py`._

First, on the **Llama-2-7B (base)** MBPP-500 cell, p-less-norm@0.6 is the
top-ranked sampler when our results are merged with the 18 Wei et al.
[2024] decoding methods evaluated on the same model and benchmark
(claim **C1**). Its 22.3% pass@1 sits ahead of FSD-d (21.2%), p-less@0.6
(22.2%), and beam-8 (19.4%); plain temperature@0.7 ranks 15/19 at 13.2%
(`comparison_report.md:9-29`). The Llama-2-7B-Chat cell is less
favourable: p-less variants rank 3rd–6th, behind beam search (21.6%)
and diverse beam search (21.2%) (`comparison_report.md:31-53`).

Second, on **Qwen2.5-Coder-7B-Instruct** (HumanEval-164), p-less@0.6 and
p-less-norm@0.6 both reach 87.5% pass@1, ahead of greedy at 84.1%
(claim **C2**; `report.md:23,26-27`). The Δ of +3.4pp is larger than
the 2.8pp HumanEval-164 SE band (claim **C9**) but rests on a single
within-run comparison; we report it as directional, not significant on
its own. The same model on MBPP-500 gives a flatter picture — at T₁ = 1.0,
p-less reaches 77.2% pass@1 vs greedy's 77.6%
(`instruct_t1_comparison_report.md:16,19`).

![**Supplementary Figure A.** Headline pass@1 (%) heatmap across all
(model, sampler) cells we ran on MBPP-500. Rows are grouped by model
family (Llama-2 → CodeLlama → Qwen-7B → Qwen2.5-Coder →
OpenCodeInterpreter), columns are ordered left-to-right by mean pass@1
across models — so the leftmost column is the globally-strongest sampler.
Each cell is annotated with its pass@1 percentage; "—" indicates the
(model, sampler) combination was not run. The per-row best sampler is
shown in bold with a black border, making the per-model winner readable
at a glance. Source:
`results/pless_full_mbpp_results/analysis/figures/pass_at_1_comparison.png`,
generated by the same `python -m bench.eval.plots …` invocation as
Figure 1c (unfiltered, all configs).
](figures/fig3_pass_at_1_comparison.png){#fig:headline width=90%}

A general observation across Table 2 is that the **0.6-temperature
p-less configurations stay close to greedy** on instruct models
(deltas in the −0.5pp to +3.4pp range across the four
(model, benchmark) cells where both are reported) while delivering
≈ 0.03 structural diversity at zero hyperparameter cost. On base
models the p-less@1.0 configurations typically dominate temperature@0.7
in pass@1 (e.g., Llama-2-7B base: 19.8% vs 13.2%, claim **C1** row;
`comparison_report.md:14,25`). The pass@10 trade is mixed: on Llama-2-7B
base pless@1.0 essentially matches temperature@0.7 (40.0 vs 39.0;
`comparison_report.md:64,66`; claim **C12**), while temperature@0.7
beats pless@1.0 by 5–8pp on CodeLlama-7B base (65.2 vs 57.2) and
Qwen2.5-Coder-3B base (77.6 vs 72.2;
`results/analysis/consolidated_summary.csv` rows 65, 69, 94, 96; claim
**C12**).

### 4.2 Pareto correctness × structural diversity

We plot pass@1 against structural diversity for our two paper-comparison
cohorts on MBPP-500. Figure 1a uses the six base/chat models from Zhu
et al. (2024) [arXiv:2402.06925], with each model's exact per-model
sampler settings as the baselines (different temperatures, top-p, and
top-k per model — see legend). Figure 1b uses the three coder models
from the BigCode coder paper [arXiv:2507.03160], with `top_p=0.9` and
`top_p=0.95` baselines under the BigCode prompt format. Both figures
restrict the sampler set to the p-less family (t ∈ {0.6, 1.0}), greedy,
beam search (4 and 8), and the paper-specific baselines for that
cohort; per-model trajectory lines are suppressed for readability.

![**Figure 1a.** Pareto pass@1 vs structural diversity on MBPP-500,
six base/chat models compared against Zhu et al. (2024)
[arXiv:2402.06925]. Per-model paper baselines: Llama-2-7B base/chat
and CodeLlama-7B-Instruct → temp=0.3, top-k=5, top-p=0.8; CodeLlama-7B
base → temp=0.6, top-k=5, top-p=0.8; Qwen-7B → temp=0.1, top-k=5,
top-p=0.85; Qwen-7B-Chat → temp=0.2, top-k=50, top-p=0.85. Color
encodes sampler *family* (one color each for temp, top-p, top-k, beam,
p-less, and p-less-norm); the two same-color p-less / p-less-norm dots
per model trace the within-family T-trajectory (t=0.6 sits at higher
pass@1 + lower diversity, t=1.0 at lower pass@1 + higher diversity).
The legend collapses to ~7 entries instead of one per (method, T, top-p,
top-k) tuple. Source:
`results/pless_full_mbpp_results/analysis/figures/pareto_zhu_2402_06925.png`,
generated by `python -m bench.eval.plots --metrics
results/pless_full_mbpp_results/analysis/consolidated_metrics/*/*_metrics.json
--output-dir results/pless_full_mbpp_results/analysis/figures
--dataset MBPP --pareto-only --pareto-output-name
pareto_zhu_2402_06925.png --models meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-chat-hf codellama/CodeLlama-7b-hf
codellama/CodeLlama-7b-Instruct-hf Qwen/Qwen-7B Qwen/Qwen-7B-Chat
--config-keys pless_t0.6 pless_t1.0 pless_norm_t0.6 pless_norm_t1.0
greedy_t1.0 beam4_t1.0 beam8_t1.0 temp_t0.1 temp_t0.2 temp_t0.3
temp_t0.6 top_k_k5_t1.0 top_k_k50_t1.0 top_p_p0.8_t1.0
top_p_p0.85_t1.0 --no-trajectories --family-palette`.
](figures/fig1a_pareto_zhu.png){#fig:paretoZhu width=90%}

![**Figure 1b.** Pareto pass@1 vs structural diversity on MBPP-500,
three coder models compared against the BigCode coder paper
[arXiv:2507.03160] under the BigCode prompt format. Paper baselines:
`top_p=0.9 @ t=1.0` and `top_p=0.95 @ t=0.2` (both collapsed under one
"top-p" color), plus `temp=0.7` for context. Same family-palette
encoding as Figure 1a: color = sampler family, marker = model; the two
same-color p-less / p-less-norm dots per model trace the within-family
T-trajectory. Legend collapses to ~6 entries. Source:
`results/pless_full_mbpp_results/analysis/figures/pareto_bigcode_2507_03160.png`,
generated by the same `bench.eval.plots` invocation as Figure 1a but
with `--models Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-Coder-3B
m-a-p/OpenCodeInterpreter-DS-1.3B --pareto-output-name
pareto_bigcode_2507_03160.png --config-keys pless_t0.6 pless_t1.0
pless_norm_t0.6 pless_norm_t1.0 greedy_t1.0 beam4_t1.0 beam8_t1.0
temp_t0.7 top_p_p0.9_t1.0 top_p_p0.95_t0.2 --no-trajectories
--family-palette`.
](figures/fig1b_pareto_bigcode.png){#fig:paretoBigcode width=90%}

Across both cohorts, p-less and p-less-norm sit on or near each model's
frontier but do not extend it — temperature and top-p at moderate-to-high
settings reach structural diversities (0.4–0.7) that no p-less
configuration reproduces in the tested range, because the p-less
threshold prunes more aggressively than temperature once probability
mass concentrates.

![**Figure 1c.** Companion view: structural diversity by sampler,
grouped by model on MBPP-500. Same underlying data as Figures 1a / 1b,
projected onto the diversity axis only — easier to read than the
Pareto scatter when the question is "how much diversity does each
sampler give up vs temperature?". Includes the full sampler sweep
(not restricted to paper baselines) so the diversity envelope per model
is visible. Source:
`results/pless_full_mbpp_results/analysis/figures/structural_diversity_bars.png`,
generated by the unfiltered `python -m bench.eval.plots …` invocation
that writes the sub-component appendix figures.
](figures/fig1c_structural_diversity_bars.png){#fig:divbars width=85%}

The frontier shape is therefore: **p-less variants own the high-pass@1,
low-diversity end; temperature owns the high-diversity end**; top-p falls
between them. The instruct models (Qwen2.5-Coder-7B-Instruct,
CodeLlama-7B-Instruct) compress this entire frontier into a narrower
range because their distributions are more peaked.

### 4.3 Robustness boundary on the T₁ sweep

Figure 2 plots pass@1 vs T₁ for six HumanEval models. Three regimes are
apparent (Table 4 summarises them, claim **C10**):

![**Figure 2.** Pass@1 vs T₁ on HumanEval-164 for six models, p-less
sampler. T₁ ∈ {0.7, 1.0, 1.5, 2.0, 2.5, 3.0}. Source:
`results/pless_human_eval_results/temprature_results/analysis/figures/pass_at_1_vs_temperature.png`,
generated by `bench/eval/report_temperature_sweep.py` from the per-model
JSONLs under `results/pless_human_eval_results/temprature_results/`.
](figures/fig2_temp_sweep.png){#fig:t1sweep width=85%}

1. **T₁ = 0.7–1.5: sweet spot.** On every T₁-sensitive model in the
   panel, pass@1 is flat within ≈ 1pp across this range
   (cross\_benchmark\_t1\_analysis.md:64; claim **C6**).
2. **T₁ = 2.0: cliff.** Strong instruct models begin to drop. Qwen2.5-
   Coder-7B-Instruct loses 2.4pp (84.8% → 82.4%; claim **C4** note;
   `temperature_sweep_report.md:168`); the base 7B loses 19.9pp
   (56.3% → 36.4%; claim **C4**; `temperature_sweep_report.md:166`) —
   a model-dependent margin. Codestral-22B is the outlier: it
   *gains* 9.7pp over the same range (5.7% → 15.4%), starting from a
   regime where the model's top-1 token is rarely correct (claim
   **C3**; `temperature_sweep_report.md:164`). At low absolute pass
   rates this pattern is hard to distinguish from sample noise on 164
   tasks, and we report it as such.
3. **T₁ ≥ 2.5: catastrophe.** Every T₁-sensitive model in the panel
   collapses by T₁ = 3.0 (`cross_benchmark_t1_analysis.md:171-180`).
   The cliff is at T₁ = 2.5 on HumanEval and between T₁ = 2.0 and
   T₁ = 3.0 on MBPP (T₁ = 2.5 was not run on MBPP).

Qwen3-Coder-30B is the single outlier: pass@1 stays in 75.1–76.2% across
the full T₁ ∈ [0.7, 3.0] grid — a 1.0pp spread well below the 2.8pp
HumanEval SE band (`temperature_sweep_report.md:82-95`). The model's
output distribution is sufficiently peaked that p-less effectively reduces
to greedy at every tested T₁ (claim **C5**, downgraded; the original
"~0.6pp gain" wording is replaced with "essentially flat with a 1.0pp
spread"). Practical implication: on extremely peaked models, the
hyperparameter-free property of p-less buys nothing because the threshold
admits only the top token.

### 4.4 T₁ / T₂ decomposition (Qwen2.5-Coder-7B-Instruct, MBPP)

Table 3 reports the T₁/T₂ grid on the strongest instruct model in our
cohort. Two findings.

**Table 3.** T₁ / T₂ grid on Qwen2.5-Coder-7B-Instruct (MBPP-500).
T₁ scales logits before the collision-entropy threshold; T₂ flattens the
survivor distribution after pruning. Four sub-tables: T₁ sweep, T₂ at
T₁=1.0, T₂ at T₁=2.0, and the matched-diversity comparison vs plain
temperature.

*T₁ sweep (T₂ = ∅):*

| T₁ | pass@1 (%) | pass@10 (%) | struct_div | codebleu_div | cover@0.7 (%) |
|:---|----------:|-----------:|-----------:|-------------:|--------------:|
| 0.6 | 77.2 | 79.8 | 0.0305 | 0.0808 | 76.0 |
| 1.0 | 77.2 | 82.2 | 0.0586 | 0.1359 | 75.0 |
| 1.5 | 76.7 | 85.8 | 0.1262 | 0.2792 | 74.0 |
| 2.0 | 72.5 | 89.6 | 0.3082 | 0.5587 | 70.4 |
| 3.0 |  2.7 | 18.8 | 0.2645 | 0.4128 |  0.0 |

*T₂ sweep at T₁ = 1.0 (baseline pass@1 = 77.2%, struct_div = 0.0586):*

| T₂ | pass@1 (%) | Δ pass@1 | struct_div | Δ struct_div |
|:---|----------:|---------:|-----------:|-------------:|
| 2.0 | 77.9 | +0.7 | 0.0572 | -0.0014 |
| 3.0 | 77.8 | +0.6 | 0.0581 | -0.0005 |
| 4.0 | 77.9 | +0.7 | 0.0555 | -0.0031 |
| 5.0 | 77.6 | +0.4 | 0.0562 | -0.0024 |

*T₂ sweep at T₁ = 2.0 (baseline pass@1 = 72.5%, struct_div = 0.3082):*

| T₂ | pass@1 (%) | Δ pass@1 | struct_div | Δ struct_div |
|:---|----------:|---------:|-----------:|-------------:|
| 2.0 | 70.1 | -2.4 | 0.3278 | +0.0196 |
| 3.0 | 70.0 | -2.5 | 0.3422 | +0.0340 |
| 4.0 | 70.5 | -2.0 | 0.3384 | +0.0302 |
| 5.0 | 68.5 | -4.0 | 0.3388 | +0.0306 |

*Matched-diversity comparison (P-less vs plain temperature):*

| P-less config | pless pass@1 | pless struct_div | nearest temp | temp pass@1 | temp struct_div | Δ pass@1 |
|:--------------|:------------:|:----------------:|:-------------|:-----------:|:---------------:|:--------:|
| pless T₁=1.5  | 76.7% | 0.1262 | temp t=0.2 | 76.8% | 0.0982 | -0.1pp |
| pless T₁=2.0  | 72.5% | 0.3082 | temp t=0.8 | 72.0% | 0.2779 | +0.4pp |
| pless T₁=3.0  |  2.7% | 0.2645 | temp t=1.5 | 12.3% | 0.2741 | -9.5pp |

_Source: `results/full_mbpp_pre_post_temp_pless/analysis/Qwen--Qwen2.5-Coder-7B-Instruct/instruct_t1_comparison_report.md` (lines 37–43, 49–53, 60–64, 69–75, 91)._

![**Figure 3.** T₁ sweep on Qwen2.5-Coder-7B-Instruct (MBPP-500). pass@1
falls 4.7pp from T₁=1.5 to T₁=2.0 and collapses 69.8pp from T₁=2.0 to
T₁=3.0. Source:
`results/full_mbpp_pre_post_temp_pless/analysis/Qwen--Qwen2.5-Coder-7B-Instruct/figures/t1_sweep.png`,
generated by `bench/eval/report_t1_t2.py` from the
`pless_t1_*_metrics.json` files under the same model directory.
](figures/fig4_t1_sweep.png){#fig:t1instruct width=70%}

![**Figure 4.** T₂ effect at T₁=1.0 on Qwen2.5-Coder-7B-Instruct (MBPP-
500). T₂ ∈ {2,3,4,5} barely moves pass@1 (Δ ≤ +0.7pp) and reduces
struct\_div slightly. Source:
`results/full_mbpp_pre_post_temp_pless/analysis/Qwen--Qwen2.5-Coder-7B-Instruct/figures/t2_effect_at_t1_1.0.png`,
generated by `bench/eval/report_t1_t2.py`.
](figures/fig5_t2_effect.png){#fig:t2effect width=70%}

**T₂ at T₁ = 1.0 is mostly inert.** Pass@1 stays within +0.7pp of the
T₂ = ∅ baseline (77.2%) for every tested T₂ ∈ {2, 3, 4, 5}, and
struct\_div changes by ≤ 0.003 — within the noise of the 500-task
estimate. T₂ neither helps nor hurts in this regime
(`instruct_t1_comparison_report.md:60-64`).

**T₂ at T₁ = 2.0 is dominated.** Adding T₂ ∈ {2, 3, 4, 5} costs
2.0–4.0pp pass@1 in exchange for 0.020–0.034 additional struct\_div
(claim **C7**; `instruct_t1_comparison_report.md:69-75`). The same
diversity is reached more cheaply by raising T₁ further within the
sweet spot. We therefore conclude that **the second temperature is a
dominated knob on instruct models**.

A matched-diversity comparison against plain temperature (Table 3,
bottom) gives Δ pass@1 ∈ [-9.5pp, +0.4pp] across three operating points
— so even in the regime where T₂ is least costly, p-less does not act
as a quality filter that beats temperature at equal diversity (claim
**C11**; `instruct_t1_comparison_report.md:51-53,91`).

### 4.5 Cross-benchmark replication (MBPP ↔ HumanEval)

Table 4 compares the five MBPP-derived conclusions against fresh
HumanEval evaluations across six models
(`cross_benchmark_t1_analysis.md` §5). The sweet-spot, catastrophe,
T₁-as-dominant-diversity-knob, and instruct-peakedness findings all
replicate (HIGH confidence in the source report). The "p-less ≥ greedy
on instruct models" finding is directionally positive on HumanEval
(+1.2pp to +3.3pp across two independent runs) but within the 2.8pp SE
band; we mark it as MEDIUM confidence and do not over-claim.

**Table 4.** Cross-benchmark replication. Five conclusions derived from
MBPP-500 are tested against freshly computed HumanEval-164 metrics.
Confidence labels copy the source report; SE bands of 1.75pp (MBPP) and
2.8pp (HumanEval) apply throughout.

| # | Conclusion | MBPP-500 | HumanEval-164 | Replicated? | Confidence |
|---|------------|----------|---------------|-------------|------------|
| 1 | Sweet spot T₁=0.7–1.5 (≤1pp pass@1 cost) | yes | yes (within SE) | ✓ | HIGH |
| 2 | Catastrophe between T₁=2.0 and T₁=3.0    | yes (cliff at T₁=2.0→3.0) | yes (cliff at T₁=2.0→2.5) | ✓ | HIGH |
| 3 | T₁ is the dominant diversity knob (vs T₂) | yes (3–17× efficiency) | tested via T₁ only; behaves as expected | ✓ | HIGH for T₁; MBPP-only for T₂ |
| 4 | P-less ≥ greedy on instruct models       | within noise (≈0pp) | directionally above (+1.2 to +3.3pp across two runs) | partial | MEDIUM |
| 5 | Instruct models are more peaked than base | yes (5–10× lower struct_div) | yes (10× lower struct_div at matched T₁) | ✓ | HIGH |

*Companion: peakedness comparison on Qwen2.5-Coder-7B-Instruct.*

| T₁ | MBPP struct_div | HumanEval struct_div | HE / MBPP ratio |
|:---|----------------:|---------------------:|----------------:|
| 0.7 | ~0.030 | 0.009 | 0.30× |
| 1.0 | 0.059  | 0.016 | 0.27× |
| 1.5 | 0.126  | 0.049 | 0.39× |
| 2.0 | 0.308  | 0.161 | 0.52× |

_Source: `results/analysis/cross_benchmark_t1_analysis.md` §5 (lines
247–270) and lines 197–202 (peakedness companion)._

Quantitatively, **HumanEval is 2–4× more peaked than MBPP** at every
T₁ tested on Qwen2.5-Coder-7B-Instruct (Table 4 companion;
`cross_benchmark_t1_analysis.md:197-202`). Practical implication:
recommendations tuned on MBPP transfer in shape but not in numerical
threshold — the same struct\_div level requires a higher T₁ on
HumanEval, and the catastrophe boundary moves accordingly (T₁ = 2.5 on
HumanEval vs T₁ = 3.0 on MBPP for the same model).

## 5. Discussion

**Where p-less helps.** Two regimes emerge from §4. On weaker base
models (Llama-2-7B, CodeLlama-7B base, Qwen-7B), where the model's
most-likely-next-token probability is rarely close to 1.0 (the model
is uncertain at most decoding steps), p-less truncation passes a
tractable survivor
set whose probability mass concentrates on the few syntactically
plausible continuations; the resulting pass@1 is competitive with or
ahead of beam-8 and FSD-d (claim **C1**). On strong instruct models in
the T₁ = 0.6–1.0 regime, p-less reproduces greedy's pass@1 while
delivering small but non-zero structural diversity at no hyperparameter
cost.

**Where p-less does not help.** On extremely peaked models
(Qwen3-Coder-30B in our cohort), the threshold `Σᵢ pᵢ²` is so close to
the top probability that only the top token survives at every tested T₁
— p-less collapses to greedy and provides no diversity (`§4.3`,
`cross_benchmark_t1_analysis.md:226-231`). On strong instruct models
under high-diversity demand, plain temperature reaches diversity levels
p-less cannot match within the safe T₁ range (`§4.2`).

**The Codestral outlier (speculation).** Codestral-22B is the only
T₁-sensitive model in our cohort whose pass@1 *rises* monotonically
through T₁ ≤ 2.0. We speculate — and label this as speculation — that
the model's low absolute pass rate at T₁ = 0.7 (5.7%) reflects a
correctly-calibrated but rarely-correct top-1 token; raising T₁
broadens the truncation set enough that the correct token, when it
exists, is more often included. We have not run a second seed or a
confirmatory full-precision evaluation, so the magnitude (+9.7pp) is
suggestive rather than established. A confirmatory run is in
`paper/TODO.md` for v2.

**Limitations.** (i) 10 samples per task; bootstrap CIs would tighten
the SE bands and are deferred. (ii) Two benchmarks (MBPP-500,
HumanEval-164); we do not test BigCodeBench, LiveCodeBench, or
multi-turn agentic settings. (iii) The T₁/T₂ grid is run on a single
strong instruct model (Qwen2.5-Coder-7B-Instruct) plus a single base
model (Qwen2.5-Coder-3B); generalisation to other architectures is
untested. (iv) All comparisons against the Wei et al. [2024] survey
cross independent evaluation pipelines and are subject to extraction
and execution-environment differences (claim **C1** sanity-check on
temperature@0.7 shows a 4.0pp pipeline gap on Llama-2-7B — see
`comparison_report.md:91`).

**Practical recommendation.** For a code-LM whose peakedness is
unknown, start with pless@T = 1.0 and compare pass@1 to greedy on a
small calibration set. If they match, retain p-less for its
hyperparameter-free property; if pass@1 is materially worse, fall back
to temperature in the 0.7–1.0 range (`cross_benchmark_t1_analysis.md:281`).

## 6. Conclusion

P-less and p-less-norm are **competitive but not dominant** decoding
samplers for code generation. They reach the Pareto frontier of
correctness × structural diversity on every base/chat model we tested,
and they outperform or match the methods surveyed by Wei et al. [2024]
on Llama-2-7B (MBPP). The pre-truncation temperature T₁ remains the
operative diversity knob; the post-truncation T₂ is dominated by T₁ on
the strong instruct model we tested. P-less is most useful where its
hyperparameter-free property is the decisive feature — a calibration-
free default that reproduces greedy on instruct models and tracks
beam-8 on weaker base models, with no top-p, top-k, η, or ε to tune.
Future work covers larger sample budgets, agentic and multi-turn
settings, and characterising the peakedness threshold beyond which
p-less collapses to greedy.

## References

See `refs.bib`.
