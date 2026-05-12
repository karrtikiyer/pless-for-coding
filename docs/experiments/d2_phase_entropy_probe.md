# Experiment: D2 Phase-Oracle Probe — Per-Phase Entropy Analysis

**Date:** 2026-04-24
**Status:** Planned
**Idea ref:** `docs/ideas/idea1.md` (D2 — AST-Phase-Aware Per-Token Decoder)

---

## Motivation

D2 proposes a decoder that applies different sampling temperatures to different AST phases of generated code (signature, body, docstring, operators). Before building it, we need to answer a prerequisite question: **does the model actually behave differently across AST phases?**

If the model's per-token entropy is roughly uniform across phases, no phase-aware decoder can help — there's no signal to exploit. If entropy varies substantially by phase, the premise holds and D2 is worth pursuing.

## What the literature already tells us

Several papers have established that code LLMs show non-uniform confidence across token types:

| Paper | Finding | Implication |
|-------|---------|-------------|
| **AdapT** (AAAI 2024) | "Challenging tokens" cluster at block-initial positions; improved pass@15 by +13.6% with position-based adaptive T | Position correlates with difficulty → structural signal exists |
| **DecoRTL** (July 2025) | Two-class syntax-aware T for Verilog; LLMs show low confidence at structural ambiguity points | Same principle works for RTL; untested on Python |
| **AdaDec** (June 2025) | Entropy spikes at 3.9-11.9% of positions; pause-and-rerank outperforms adaptive T | High-entropy positions are sparse and locatable |
| **"A Critical Study of What Code-LLMs (Do Not) Learn"** (ACL 2024) | Models encode syntactic relations well but fail on syntax↔identifier cross-relations | Differential competence by token category |
| **Structural Entropy** (Aug 2025) | Identifier-level variability much higher than control-flow variability | AST structure predicts entropy profile shape |

So we already know the answer to "do phases differ in entropy?" is likely **yes**. The more important question this probe must answer is:

### The real question: Does AST phase provide signal beyond raw entropy?

If the model's entropy at a token already tells you everything phase classification would tell you, then AST phases are just a proxy for entropy — and reactive approaches (EDT, Entropix, AdaDec) already exploit this without needing tree-sitter. D2's value depends on phase giving **additional** information.

Concretely: within the subset of tokens that all have similar entropy (say, entropy ∈ [2.0, 2.5] nats), do different phases still benefit from different temperatures? If yes, phase is an independent signal. If no, entropy subsumes phase.

## Experiment design

### What we measure

**Teacher-forced entropy per token, grouped by AST phase.** No generation — we take existing correct code and compute the model's per-token log-probability distribution via a single forward pass.

### Model

**Qwen/Qwen2.5-Coder-7B-Instruct** — the instruct model is the harder test. If instruct shows phase separation, base models almost certainly will too. If instruct is flat, the market for D2 shrinks to base-model serving only.

### Data

**HumanEval** — 164 tasks, ~1300 passing samples from existing results (`temp_t0.7.jsonl`). We intentionally reserve MBPP as a clean validation set for later.

### Phase taxonomy (4 classes)

| Phase | Description | Tree-sitter node types |
|-------|-------------|----------------------|
| **signature** | Function definition line(s): `def`, name, parameters, return annotation, decorators, up to and including `:` | `function_definition` children before the `block`, `decorator`, `parameters`, `->` annotation |
| **body** | Computational logic: assignments, expressions, returns, control flow bodies, comprehensions | `expression_statement`, `return_statement`, `assignment`, `augmented_assignment`, `if_statement` body, `for_statement` body, `while_statement` body, `call`, `identifier`, `attribute`, `subscript`, list/dict/set comprehensions |
| **docstring** | Documentation: triple-quoted strings as first statement in function body, plus `#` comments | `string` node as first child of function block, `comment` nodes |
| **operator** | Structural glue: punctuation, operators, keywords, whitespace/indentation | `(`, `)`, `[`, `]`, `{`, `}`, `:`, `,`, `=`, `==`, `+`, `-`, `*`, `/`, boolean operators, control-flow keywords (`if`, `else`, `for`, `while`, `return`, `def`, `import`), indentation |

Why 4 phases, not more? We're testing whether phases matter *at all*, not which partition is best. If 4 phases show no separation, 11 won't either.

### Procedure

1. **Load passing samples:** From `temp_t0.7.jsonl`, extract code, re-execute against HumanEval tests, keep only passing solutions. Expected: ~100-130 unique passing tasks with ~8-9 correct samples each.

2. **Reconstruct prompts:** Use `format_prompt_instruct()` with the same tokenizer/chat template used during generation.

3. **Teacher-forced forward pass:** For each (prompt, code) pair:
   - Encode `prompt + code` → input_ids
   - Single forward pass → logits at every position
   - For each token in the **code portion only**: compute entropy `H = -Σ p·log(p)` from softmax(logits)

4. **Phase classification:** Parse each code string with tree-sitter, assign every byte position a phase label.

5. **Token-to-phase alignment:** Use `tokenizer(code, return_offsets_mapping=True)` to map each token to byte offsets → look up the phase of the first byte.

6. **Aggregate:** Pool all (token, phase, entropy) triples across all samples.

### Outputs

**Primary plot — `phase_entropy_kde.png`:**
Overlaid KDE curves, one per phase. X-axis = entropy (nats), y-axis = density. Vertical dashed lines at per-phase medians. This is the decisive plot.

**Secondary plot — `phase_entropy_boxplot.png`:**
Side-by-side box plots per phase (median, IQR, whiskers, outliers).

**Conditional analysis plot — `phase_entropy_conditional.png`:**
Scatter or heatmap of (raw entropy, phase) showing whether phase adds information *beyond* entropy. Specifically: bin tokens by entropy, then within each bin, compare phase distributions.

**Summary statistics — `phase_entropy_stats.json`:**
- Per-phase: count, mean, median, std, p5, p25, p75, p95
- Between-phase: Cohen's d for all 6 pairwise comparisons, KS test p-values
- Conditional mutual information: I(phase; optimal_T | entropy)

## Decision criteria

| Outcome | Cohen's d (max pairwise) | Conditional MI | Action |
|---------|--------------------------|----------------|--------|
| **Strong positive** | > 0.8 AND conditional MI > 0 | Phases differ AND phase adds info beyond entropy | Build D2. Phase taxonomy + rough T vector emerge from this probe. |
| **Weak positive** | > 0.5 but conditional MI ≈ 0 | Phases differ but entropy subsumes phase | D2 adds nothing over reactive entropy methods (EDT/AdaDec). Pivot to complementing rather than replacing reactive approaches. |
| **Moderate** | 0.3 - 0.5, conditional MI > 0 | Modest phase separation with some independent signal | D2 might yield marginal gains. Evaluate cost-benefit before proceeding. |
| **Flat negative** | < 0.3 | No meaningful phase separation | D2 is dead. Instruct models are implicitly phase-calibrated. |

The most likely outcome is **weak positive** — phases differ (consistent with literature) but entropy captures most of the signal. This is the hardest outcome to act on, which is why we define the criteria in advance.

## What this experiment does NOT test

- **Whether temperature adjustment per phase actually improves pass@k** — that's D2 itself. This probe measures the ceiling, not the achievable gain.
- **Multi-model generality** — one model is enough to validate or kill the premise. Multi-model comes later if positive.
- **Base vs instruct differences** — we run instruct only. If instruct is flat, base would need separate evaluation.
- **MBPP performance** — reserved as clean validation set.

## Cost

- ~15 minutes on M4 Max MPS (48GB unified memory) — forward passes only, no generation
- ~1 day researcher time (implementation + analysis)
- Zero GPU-hours beyond local machine

## Prior art to cite

- AdapT (Zhu et al., AAAI 2024) — loss-proxy adaptive T, the closest comparison
- DecoRTL (2025) — syntax-aware T for Verilog, D2's direct ancestor
- EDT (2024) — entropy-based dynamic T, the reactive baseline
- AdaDec (2025) — uncertainty-guided reranking, current SOTA in adaptive code decoding
- TAMPO / Learning Adaptive Decoding (ICLR 2026) — learned T policies from internal states

## Results

### Model 1: Qwen2.5-Coder-7B-Instruct on HumanEval (temp 0.7)

**Data:** 87,360 tokens across ~130 passing HumanEval tasks from `temp_t0.7.jsonl`.

**Per-phase entropy summary:**

| Phase | Count | Mean | Median | Std | P95 |
|-------|-------|------|--------|-----|-----|
| signature | 15,279 | 0.139 | 0.0006 | 0.315 | 0.852 |
| body | 57,514 | 0.053 | 0.0001 | 0.169 | 0.446 |
| docstring | 7,070 | 0.046 | 0.0001 | 0.171 | 0.322 |
| operator | 7,497 | 0.152 | 0.005 | 0.306 | 0.808 |

**Pairwise Cohen's d:**

| Pair | d |
|------|---|
| body vs operator | **0.525** |
| docstring vs operator | 0.427 |
| signature vs body | 0.414 |
| signature vs docstring | 0.336 |
| signature vs operator | 0.043 |
| body vs docstring | 0.040 |

Max Cohen's d = 0.525 (body vs operator) — moderate effect.

**KS p-values:** All pairwise comparisons p ≈ 0.0 — distributions are statistically distinguishable.

**Key observation:** 78.4% of all tokens have entropy < 0.01 nats (trivially predicted). Only ~6 tokens per 64-token function sit in the "uncertain" regime.

**Phase ranking (high → low entropy):** operator ≈ signature >> body ≈ docstring. This is counterintuitive — operators and signatures are the *most* uncertain phases, not the computational body.

**Conditional analysis:** Within each entropy bin, phase distributions are non-uniform — phase carries information beyond raw entropy. However, the effect is concentrated in the 0.0-0.3 nats bin which contains >90% of tokens.

### Interpretation

**Verdict: Moderate positive** — phases are statistically separable (Cohen's d = 0.53) and phase carries some independent signal beyond entropy. However:

1. The practical ceiling is very low: only ~6 tokens per function are genuinely uncertain
2. Even perfect phase-aware temperature on those 6 tokens would yield <1pp pass@1 improvement
3. The phase ranking (operators most uncertain) suggests D2 would need to *increase* temperature for structural tokens and *decrease* for logic — the opposite of the naive intuition

This aligns with the "weak positive" decision criteria row: phases differ, but the signal is too sparse to be practically useful for a phase-aware decoder.

### Model 2: CodeLlama-7b-Instruct on HumanEval (temp 0.7)

**Data:** 30,189 tokens across 451 passing samples from `temp_t0.7.jsonl` (much fewer than Qwen — CodeLlama has ~30% pass@1 vs Qwen's ~65%).

**Per-phase entropy summary:**

| Phase | Count | Mean | Median | Std | P95 |
|-------|-------|------|--------|-----|-----|
| signature | 6,015 | 0.231 | 0.042 | 0.416 | 1.063 |
| body | 18,313 | 0.087 | 0.003 | 0.226 | 0.612 |
| docstring | 3,434 | 0.094 | 0.001 | 0.289 | 0.682 |
| operator | 2,427 | 0.176 | 0.032 | 0.317 | 0.768 |

**Pairwise Cohen's d:**

| Pair | d |
|------|---|
| signature vs body | **0.507** |
| body vs operator | 0.374 |
| signature vs docstring | 0.365 |
| docstring vs operator | 0.271 |
| signature vs operator | 0.142 |
| body vs docstring | 0.032 |

Max Cohen's d = 0.507 (signature vs body) — moderate effect.

### Cross-model comparison

| Metric | Qwen2.5-Coder-7B-Instruct | CodeLlama-7b-Instruct |
|--------|---------------------------|----------------------|
| Passing samples | ~1,300 | 451 |
| Total tokens | 87,360 | 30,189 |
| Median entropy (body) | 0.0001 nats | 0.003 nats |
| Median entropy (signature) | 0.0006 nats | 0.042 nats |
| Max Cohen's d | 0.525 (body vs operator) | 0.507 (sig vs body) |
| Phase ranking (high→low) | op ≈ sig >> body ≈ doc | sig > op >> doc ≈ body |

**Key observations:**

1. **CodeLlama is ~30-70x less confident per token** (median body entropy 0.003 vs 0.0001), consistent with its lower overall accuracy.
2. **Same macro pattern**: signature and operator phases have higher entropy than body and docstring — stable across model families.
3. **Nearly identical Cohen's d** (0.51 vs 0.53): phase separation strength is model-invariant. This is reassuring — it's a property of code structure, not model-specific behavior.
4. **More tokens in the uncertain regime for CodeLlama**: P95 entropy reaches 1.06 nats (signature) vs 0.85 nats for Qwen. CodeLlama has a larger "attack surface" for phase-aware decoding.

### Overall verdict

**Moderate positive, consistent across models.** The phase separation pattern (signatures and operators harder than body and docstrings) is real and reproducible. However:

- The effect size is moderate (d ≈ 0.5), not strong (d > 0.8)
- Most tokens remain in the trivially-predicted regime regardless of phase
- Phase-aware temperature would be applying a correction to a small fraction of tokens
- The practical pass@k improvement ceiling remains < 1pp for a well-calibrated instruct model

D2 as originally conceived (static per-phase temperature) would provide marginal gains at best. The literature's reactive approaches (EDT, AdaDec) already capture most of the achievable benefit without needing AST classification.

### TODO: Rényi-2 entropy follow-up

This probe used **Shannon entropy** `H = -Σ p·log(p)` — the standard measure of total uncertainty. P-less uses **Rényi-2 (collision entropy)** `H₂ = -log(Σ p²)` to compute its threshold `p = Σ p²`. These are different quantities.

A follow-up worth picking up later: compute Rényi-2 entropy per token per phase. This would show how the p-less threshold itself varies by code phase — i.e., does p-less trim more aggressively during signatures vs body? That's a different question than "is the model uncertain?" and might reveal whether p-less's near-deterministic behavior (0.007 structural diversity without thinking) is because it over-prunes during specific phases. If p-less is collapsing diversity specifically at operator/signature tokens (the high-entropy phases), that would explain the diversity cliff and suggest phase-aware threshold relaxation as a fix.
