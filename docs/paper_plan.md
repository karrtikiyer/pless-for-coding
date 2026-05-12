# Workshop Paper Plan — p-less Sampling for Code Generation

## Context

The original p-less paper (Tan, Wu, Howard, **arXiv:2509.23234**, Feb 2026) introduced a hyperparameter-free decoding rule that prunes tokens below the collision-entropy threshold `p = Σ probsᵢ²`. Their evaluation covered **3 LLMs × 5 datasets in math, logical reasoning, and creative writing — no code generation**. The companion "Thorough Examination" survey (Wei et al., **arXiv:2402.06925**) benchmarked many decoding methods on HumanEval/MBPP across Llama-2 family models, but **predates p-less**. The recent SLM-for-code study (**arXiv:2507.03160**) compares 20 small code models but does not vary decoding methods.

The intersection — **p-less × code generation × diverse model families × full diversity analysis** — is unoccupied. Our experiments under `results/` cover this space at non-trivial breadth: 11 models from 1.3B → 30B, both bases and instruct variants, MBPP-full and HumanEval, with five diversity metrics and a temperature × p-less interaction sweep that is not in the origin paper. This plan organizes those results into a workshop submission.

Out of scope (per user): all Qwen3-8B "thinking"/split-decoding experiments. Those become a separate paper.

## Target venue

**Primary: NeurIPS 2026 DL4C workshop** (Deep Learning for Code) — the natural home. DL4C ran at NeurIPS 2025 ("DL4C in the Agentic Era") and will likely return; expect CFP in **Jul–Aug 2026** with deadlines in **Aug–Sep 2026** following the 2025 cadence (deadline was 27 Aug 2025).

**Secondary: NeurIPS 2026 Evaluations & Datasets Track** — if reframed as a benchmarking contribution. Main-conference deadlines: abstract 4 May 2026, full paper 6 May 2026 (already past).

**Backup: COLM 2026** (Conference on Language Modeling) — broader LLM venue; check `colmweb.org/cfp.html` for current deadlines.

**Risk:** DL4C's 2026 edition is not yet announced. Track `dl4c.github.io` from June 2026 onward. If it doesn't recur, fall back to COLM or a GenAI/Foundation-Model workshop at NeurIPS 2026.

## Positioning — what's new vs prior work

| Question | Tan et al. 2026 (p-less origin) | Wei et al. 2024 (decoding survey) | **This work** |
|---|---|---|---|
| Code generation evaluated? | No (math/reasoning/writing) | Yes, but no p-less | **Yes, with p-less** |
| HumanEval + MBPP? | No | HumanEval, MBPP | HumanEval-164 + **MBPP-full (500)** |
| Number of code models | 0 | ~3 Llama-2 sizes | **11** (Llama-2, CodeLlama, Qwen-7B, Qwen2.5-Coder 1.5/3/7B, OCI-DS-1.3B, Codestral-22B, Qwen3-Coder-30B; bases + instructs) |
| Instruct/chat models on code | Not addressed | Llama-2-Chat only | **6 instruct models** |
| Diversity beyond pass@k | Generic "diversity assessments" | pass@1 only for code | **5 metrics**: structural AST (zss tree-edit), CodeBLEU, syntax-match, dataflow-match, n-gram |
| Cover@t (repeatability) | No | No | **Yes** (cover@{0.1,0.3,0.5,0.7}) |
| Temperature × p-less interaction | T sweep only | No interaction study | **T1/T2 split** (pre- vs post-threshold) |
| Catastrophic-collapse boundary | Not characterized | Not for code | **All models collapse at T1∈[2,3]**, cross-dataset |

## Title candidates

1. **"Hyperparameter-Free Decoding for Code: An Empirical Study of p-less Sampling Across 11 Models and Two Benchmarks"** (descriptive, accurate)
2. **"When Does p-less Sampling Help Code LLMs? A Cross-Model, Cross-Benchmark Evaluation"** (question-driven)
3. **"Beyond Top-p: Robustness and Diversity of Hyperparameter-Free Decoding for Code Generation"** (positioning vs top-p)

Recommend **#2** for a workshop — it signals empirical breadth and admits negative findings.

## Claimed contributions (final pitch)

1. **First systematic evaluation of p-less and p-less-norm on HumanEval and MBPP** across 11 model families/sizes from 1.3B to 30B (both base and instruct).
2. **Diversity-aware Pareto analysis** showing where p-less buys correctness × diversity over temperature, top-p, top-k, FSD, DoLa, contrastive search, mirostat, and others — using 5 code-aware diversity metrics including AST tree-edit distance.
3. **Operating envelope characterization**: the catastrophic-collapse boundary for p-less on code is sharp and consistent — sweet spot at T1 ∈ [0.7, 1.5], cliff at T1=2.0, catastrophe at T1=3.0; cross-model heterogeneity (Codestral-22B is the outlier — pass@1 *improves* +9.7pp from T1=0.7→2.0).
4. **The pre/post-temperature decomposition (T1/T2)**: applying temperature *after* p-less pruning (T2) reshapes the survivor distribution. We show T2 costs 2–4pp pass@1 for ~0.03 struct_div gain — empirically not worth it.
5. **A reproducible benchmarking pipeline** (uv-managed, JSONL streaming, AST-fingerprint diversity via `zss`) released at our repo, with consolidated CSV across all 192 configs.

## Paper structure (target 8–9 pages, double-column workshop format)

### Title + Abstract (1 paragraph, ~200 words)
Highlight: 11 models, 2 benchmarks, 5 diversity metrics, T1/T2 finding, robustness boundary.

### 1. Introduction (~1 page)
- Why decoding matters for code (correctness vs diversity, repeatability, sample efficiency for self-consistency / reranking).
- Hyperparameter sensitivity is a real cost — cite Wei et al. 2024.
- p-less promises hyperparameter-free decoding; we test that promise on code.
- 4-bullet contributions list (above).

### 2. Background and Related Work (~0.5 page)
- p-less in 2 sentences + the threshold formula `p = Σ probsᵢ²`. p-less-norm relaxation `(v·Σ probsᵢ² − 1) / (v − 1)`.
- Prior code-decoding studies: Wei et al. 2024 (survey), Chen et al. (HumanEval), Austin et al. (MBPP), Roziere et al. (CodeLlama).
- Note 2507.03160 (SLM-for-code) for the small-model context.

### 3. Methodology (~1 page)
- **Models** (Table 1): 11 models, sizes, base vs instruct, sourced from `pyproject.toml` + run scripts.
- **Sampling configurations**: temp@{0.2, 0.7}, top-p@{0.9, 0.95}, p-less@{0.6, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}, p-less-norm@{0.6, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}, greedy.
- **Benchmarks**: MBPP-full (500), HumanEval (164). 10 samples per task.
- **Metrics**:
  - Correctness: pass@{1,3,5,10}
  - Repeatability: cover@t — fraction of tasks with ≥t·n correct samples
  - Diversity: structural (zss tree-edit on AST), CodeBLEU, syntax-match, dataflow-match, n-gram
- **Statistical caveats**: SE on pass@1 ~ 1.75pp on MBPP-500, ~2.8pp on HumanEval-164; differences <2 SE are directional only. Lift verbatim from `cross_benchmark_t1_analysis.md`.

### 4. Results (~3 pages, the meat)

#### 4.1 Headline pass@k on the canonical configs
- **Table 2**: pass@1 for all 11 models × {greedy, temp@0.7, top-p@0.95, p-less@0.6, p-less-norm@0.6} on MBPP and HumanEval.
- Key finding (from `pless_full_mbpp_results/analysis`): on Llama-2-7B base, **p-less-norm@0.6 ranks 1/19 (22.3% pass@1)** beating FSD-d (21.2%) and Beam-8 (19.4%) — replicates and extends Wei et al.'s ranking.
- Key finding: on Qwen2.5-Coder-7B-Instruct (HumanEval), **p-less@0.6 reaches 87.5% pass@1 vs greedy 84.1%** (+3.4pp).

#### 4.2 Pareto: correctness × diversity (the diversity story)
- **Figure 1**: 4-panel Pareto frontier (struct_div, codebleu_div, ngram_div, dataflow_div on y; pass@1 on x). Reuse plots from `pless_full_mbpp_results/analysis/figures/pareto_*.png`.
- Story: p-less and p-less-norm sit on or above the Pareto frontier of all 12 baselines for most models. Top-p is dominated.

#### 4.3 The catastrophic-collapse boundary (the robustness story)
- **Figure 2**: pass@1 vs T1 for all 6 HumanEval models, with shaded "safe" and "danger" regions.
- T1 ∈ [0.7, 1.5] is the universal sweet spot.
- Cliff at T1=2.0 (-5pp typical); catastrophe at T1=3.0 (>70pp drops).
- **Cross-model heterogeneity**: Qwen2.5-7B drops -19.9pp from T1=0.7→2.0; Codestral-22B *improves* +9.7pp. Hypothesis: training data temperature priors differ.

#### 4.4 The T1/T2 interaction (the methodology story)
- T1 = pre-pless logit temperature; T2 = post-pless reshape `prob^(1/T2)`.
- **Table 3**: pass@1 / struct_div for T1 ∈ {1.0, 2.0} × T2 ∈ {1.0, 2.0, 3.0, 4.0, 5.0} on Qwen2.5-Coder-7B-Instruct (from `full_mbpp_pre_post_temp_pless/analysis/`).
- **Verdict**: T2 costs 2–4pp pass@1 for ~0.03 struct_div gain. Stick with T2=1.

#### 4.5 Cross-benchmark agreement
- **Table 4**: 5 conclusions from MBPP, replicated on HumanEval (already in `cross_benchmark_t1_analysis.md`).

### 5. Discussion (~0.5 page)
- p-less is not a free lunch — on strong instruct models (Qwen3-Coder-30B), gains are marginal (~0.6pp). It shines on weaker / base models.
- T1 still matters; "hyperparameter-free" is a strong claim that depends on what counts as a hyperparameter.
- Negative result: T2 is dominated.
- Limitations: 10 samples/task is modest; no agentic / multi-turn evaluation; no large code datasets (BigCodeBench, LiveCodeBench).

### 6. Conclusion (~0.25 page)
- p-less is a competitive baseline for code-LM decoding, especially when sample diversity matters and operator simplicity is prized.

### Appendix (workshop usually allows)
- Per-model full tables, full Pareto plots for all 5 diversity metrics.
- Reproducibility: random-seed protocol, exact CLI invocations, hardware (RTX 4090).
- Curated qualitative examples (already in `pless_human_eval_results/full_precision_results/analysis/curated_examples.md`).

## Figures & tables — what exists vs what to make

| Asset | Status | Source |
|---|---|---|
| Table 1: Model lineup | **Make** | from `pyproject.toml`, run scripts |
| Table 2: Headline pass@k matrix | **Stitch** | from `results/analysis/consolidated_summary.csv` |
| Figure 1: Pareto frontiers | **Reuse** | `pless_full_mbpp_results/analysis/figures/pareto_*.png` (5 versions exist) |
| Figure 2: pass@1 vs T1 sweep | **Reuse** | `pless_human_eval_results/temprature_results/analysis/figures/` |
| Table 3: T1/T2 interaction | **Reuse** | `full_mbpp_pre_post_temp_pless/analysis/t1_t2_comparison_report.md` |
| Table 4: Cross-benchmark agreement | **Reuse** | `results/analysis/cross_benchmark_t1_analysis.md` |
| Figure 3 (optional): cover@t bar chart | **Make** | from consolidated CSV |
| Curated qualitative examples | **Reuse** | `curated_examples.md` |

Most assets already exist. Estimated new figure work: **2–4 plots** to harmonize style across MBPP and HumanEval panels.

## Open questions to settle before writing

1. **Re-run scope**: do we need to re-run any configs to fill table cells, or is the consolidated CSV complete enough? (Likely complete — verify against the contribution list.)
2. **Significance bars**: add bootstrap CIs to all pass@1 numbers? Recommended for reviewer credibility.
3. **Chat-template asymmetry**: instruct models use `apply_chat_template`; bases don't. Spell this out so the comparison isn't apples-to-oranges.
4. **The "Codestral improves at T=2" finding** — is it a real effect or an artefact of generation length / EOS handling? Worth a quick confirmatory run.
5. **Including the BigCode-2507 small models cohort** — do they belong in the main paper or appendix?
6. **Author list / institutional affiliation** — pending.

## Timeline (assuming DL4C @ NeurIPS 2026, deadline ~late Aug 2026)

| Week | Milestone |
|---|---|
| Now (late Apr 2026) | Plan finalized; outline approved; freeze "no thinking" scope |
| May 2026 | Draft §1–§3 (intro, related work, methodology); finalize Table 1 + Table 2 |
| Jun 2026 | Draft §4 (results); harmonize figures; bootstrap CIs |
| Jul 2026 | Draft §5–§6 (discussion, conclusion); appendix; full pass |
| Aug 2026 | Internal review; reviewer-friendly polish; submission |
| Sep 2026 | DL4C decisions (per 2025 cadence) |

## Risks & mitigations

- **DL4C 2026 doesn't recur** → COLM 2026 (rolling submissions) or NeurIPS Foundation Model workshop as fallback.
- **"p-less is marginally better than top-p, why is this a paper?"** → emphasize the *robustness* story (catastrophic boundary) and the *diversity-Pareto* story, not just headline pass@1.
- **Reviewer asks about agentic / large-code benchmarks** → acknowledge in Limitations; cite as future work.
- **Threshold-formula re-derivation already in origin paper** → keep §2 background tight; don't re-prove.

## Files to create / write

- `paper/main.tex` (or markdown draft `paper/draft.md` first; convert when settled)
- `paper/figures/` — symlinks or copies of the reused plots
- `paper/tables.tex` — all 4 tables generated from `results/analysis/consolidated_summary.csv` via a small `make_tables.py` script
- `paper/refs.bib` — at minimum: Tan et al. 2026 (2509.23234), Wei et al. 2024 (2402.06925), 2507.03160, Chen et al. (HumanEval), Austin et al. (MBPP), Roziere et al. (CodeLlama), Wang et al. (CodeBLEU), Zhang & Shasha (zss tree-edit)

## Verification

A reviewer-friendly self-check before submission:
1. Every claim in §4 traces to a row/column in `consolidated_summary.csv` or a per-model report — no orphan numbers.
2. Every figure's source script is in the repo (no hand-edits in image editors).
3. Re-running `bench.eval.consolidated_eval` reproduces every number to 4 decimals.
4. The contributions list in §1 maps 1-to-1 to subsections in §4 — no over-claiming.

## References (anchor)

- Tan, R., Wu, S., Howard, P. (2026). *p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding.* arXiv:2509.23234.
- Wei et al. (2024). *A Thorough Examination of Decoding Methods in the Era of LLMs.* arXiv:2402.06925.
- (2507.03160) SLM-for-code empirical study — orthogonal scope, cite for context.
- DL4C @ NeurIPS 2025 site: dl4c.github.io.
