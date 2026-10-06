# Section skeleton — LLM4Code paper

Structure and evidence-source-per-section. Page/word budget deferred until the CFP posts.

Every claim listed under **Evidence:** is anchored to a specific file the paper must cite. Numbers come from `scripts/paper/make_llm4code_tables.py` — no hand-typing.

## 1. Introduction (~0.5 pg)

**Claims to make**:
- Sampler hyperparameters (T, top-p, top-k, beam width) are per-model, per-task, and per-benchmark. Tuning consumes practitioner effort and evaluation compute.
- p-less (Tan et al., arXiv:2509.23234) is a recent proposal for a hyperparameter-free sampler based on the collision-entropy threshold `p = Σ pᵢ²`.
- Origin paper evaluates on math / reasoning / writing — not code. We fill that gap.
- Contribution: (i) systematic p-less evaluation on MBPP + HumanEval across 4 code models; (ii) demonstration that default p-less fails on RL-trained reasoning models via CoT non-termination; (iii) empirical validation of the origin paper's Appendix-B.5 G_k generalization as the fix, on APPS-252.

**Evidence**: none — introductory framing only. Cite Tan et al. 2509.23234 (origin, verified), Shi et al. 2402.06925 (decoding-methods survey, related work).

## 2. Background & related work (~0.75 pg)

**Sub-sections**:
- **2.1 Decoding for code generation** — greedy, beam, temperature, top-p, top-k, and one-line mentions of the samplers we did NOT run (mirostat, typical, η-sampling, min-p, contrastive search, FSD) with the Shi et al. Table 26 citation. Do NOT compare cell-by-cell.
- **2.2 p-less sampling** — definition, `p_less_decode` and `p_less_norm_decode` from `p-less/p_less_samplers.py`. State that the p-less threshold is exactly the collision entropy `H₂ = −ln Σ pᵢ²` (equivalently `Σ pᵢ²`). Reference the Appendix-B.5 rooted-Rényi form `G_k = (Σ pᵢ^k)^{1/(k−1)}` and note k=2 recovers p-less.
- **2.3 Small-model code eval** — Hasan et al. arXiv:2507.03160 sets a single-decoder baseline (top_p=0.95, T=0.2) on 20 SLMs; we use it for a targeted comparison on 3 shared models.
- **2.4 Reasoning-model failure modes** — mention CoT non-termination, loop collapse, RL-tuned peakedness. Cite the loop-collapse and CoT-efficiency literature the repo has already surveyed (`docs/cot_early_stop_detection_survey.md`, `docs/loop_collapse_internal_state_findings.md` for the internal-state replication).

**Evidence**: literature-only. All arXiv IDs must be verified via WebFetch before citing.

## 3. p-less on standard code benchmarks (~1.5 pg)

**Setup**:
- Datasets: MBPP-full (500 tasks) + HumanEval (164 tasks).
- Models: 4 (CodeLlama-7b-hf, CodeLlama-7b-Instruct-hf, m-a-p/OCI-DS-1.3B, Qwen2.5-Coder-7B-Instruct). See `MODEL_ROSTER.md` for coverage caveats.
- Metrics (paper subset, defined in `scripts/paper/make_llm4code_tables.py::PAPER_COLUMNS`): pass@{1,3,5,10}, cover@{0.3, 0.7}, `cb_div` (CodeBLEU-based diversity). `struct_div` (zss AST edit distance) dropped from the paper tables as of 2026-10-06: 0.93 Pearson-correlated with `cb_div` across 106 cells, zero prose use, and APPS asymmetry (not computed on APPS CoT runs). Reasoning in `docs/theory/todos.md` version history + current session notes.
- Baselines (self-run only): greedy, beam4/8, temperature (multiple T), top-p (0.8, 0.9, 0.95), top-k (5).
- Under test: p-less at T ∈ {0.6, 0.7, 1.0}, p-less-norm at same T.
- Sample count: n=10 per task.

**Result 1 — pass@k tables**:
- Table 3A: MBPP pass@{1,3,5,10} for the 4 models × {best baseline, p-less, p-less-norm}.
- Table 3B: HumanEval same shape.
- Prose: p-less/p-less-norm within [X, Y] pp of best baseline on Z of 4 models on MBPP; similar summary on HE. **Numbers to be pulled by the extractor script — do not draft with placeholders like "competitive" until values are in.**

**Result 2 — pass@k curves + lift bars**:
- Fig 3a (`fig3a_passk_curves_mbpp.png`): pass@k curves per sampler, averaged across 7 MBPP models.
- Fig 3b (`fig3b_lift_bars_mbpp.png`): pass@10 − pass@1 lift per sampler per model.
- Prose: p-less tracks greedy on pass@1 within ±1.6pp but has near-zero pass@10 lift — stability-for-diversity trade. (The earlier "struct_div vs pass@10 scatter / Pareto view" plan was dropped when struct_div was removed from the paper.)

**Result 3 — small-model callout vs Hasan et al.**:
- Table 3C: Qwen2.5-Coder-{1.5B, 3B}, OCI-DS-1.3B on MBPP; Hasan et al. top_p=0.95/T=0.2 vs our p-less/p-less-norm.
- Source: `results/pless_full_mbpp_results/analysis/small_models_2507/comparison_report.md` (already computed).

**Evidence sources**:
- `results/analysis/consolidated_summary.csv` (starting point, but incomplete — audit against per-model dirs)
- `results/pless_full_mbpp_results/<model>/metrics/` — per-config metric JSONs
- `results/pless_human_eval_results/{full_precision_results,temprature_results}/<model>/` — HE metric JSONs
- `results/pless_full_mbpp_results/analysis/small_models_2507/comparison_report.md` — 2507 comparison

## 4. Default p-less collapses on reasoning models (~1.25 pg)

**Setup**:
- Reasoning models: Qwen3-8B (thinking-ON), DeepSeek-R1-Distill-Llama-8B (thinking-ON, tokenizer-fix asymmetry noted in footnote).
- Benchmark: APPS-252 (ATCODER-interview split), n=10, max_new_tokens=32768, vLLM, `--enable-thinking`.
- Metrics: pass@{1,10}, truncation rate (fraction of samples that hit max_new_tokens without emitting `</think>` and code).

**Result 4A — the truncation problem**:
- Table 4: pass@1, pass@10, truncation% for Qwen3-8B and DeepSeek at p-less (α=2 collision-entropy baseline) vs. tuned temperature baselines.
- Grounded (verified numbers from `docs/decoder_comparison_cot_apps_qwen3.md` + `docs/decoder_comparison_cot_apps_deepseek.md`): Qwen3-8B p-less 14.5% trunc / 0.625 pass@1; DeepSeek p-less 41.8% trunc / 0.392 pass@1.

**Result 4B — loop-collapse mechanism**:
- Reference internal-state replication in `docs/loop_collapse_internal_state_findings.md`: clean ≥6-cycle statement loops are ~100% truncation-bound on DeepSeek (734/739 = 99.3%) and Qwen3-8B (138/138 = 100%). Layer-wise cosine deepens across loop cycles on DeepSeek (R1 0.795 → R5 0.997); Qwen is already rigid on entry (~0.999). Caveat: verbatim-loop detection covers ≤41% (Qwen) / ~50% (DeepSeek) of all truncations — the rest are paraphrastic drift.
- Prose: loops are entered from a peaked distribution; the collision-entropy threshold `p ≈ 1` collapses the sampler to argmax; RL-tuned reasoners are structurally more prone to peaked local minima.

**Result 4C — instruct-model CoT control (truncation is reasoning-model-specific)**:
- Qwen2.5-Coder-7B/3B with induced-CoT prefill on APPS-252 show **≈0% truncation** across all decoders including p-less. Source: `docs/cot_instruct_decoder_comparison_apps.md`.
- Prose (careful framing): the *truncation / rambling failure* is reasoning-model-training-specific, not CoT-prompting-general. **Do NOT overclaim** — the same doc also finds that on instruct models under induced-CoT, "p-less trades coverage for stability" (i.e. worst pass@10 in the panel). The instruct-CoT control ONLY defends the reasoning-model narrative for truncation; it does NOT rescue the broader diversity story. Reviewers who read the source doc will find this — surface it in §4 or §6 (limitations).

**Framing constraint (from plan)**: RL-training attribution stays as **hypothesis with citations**, not our own demonstration. Cite established RL-post-training literature on peaked policy entropy.

## 5. G_k Rényi recovers on APPS-252 (~1 pg)

**Setup**:
- Same reasoning models, same benchmark, same sampling config.
- Under test: `pless_renyi` (implementation of Tan et al. Appendix B.5) at multiple k values.
- Qwen3-8B k-sweep: {0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.8, 1.6}
- DeepSeek k-sweep: {0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.8, 1.6}
- Baseline (α=2 collision entropy): matches p-less by construction.

**Result 5A — best-k per model**:

| Model | Baseline (α=2) | Best G_k | Δ pass@1 | Truncation |
|---|---|---|---|---|
| Qwen3-8B | pass@1=0.625, trunc 14.5% | k=0.1: pass@1=0.719, pass@10=0.845 | +9.4pp | 14.5% → 0% |
| DeepSeek-R1-Distill-Llama-8B | pass@1=0.392, trunc 41.8% | k=0.35: pass@1=0.485, pass@10=0.742 | +9.3pp | 41.8% → 0.1% |

Fig 5A: pass@1 (Qwen) and pass@10 (both) as functions of k, x-axis log-spaced.

**Result 5B — mechanism check**:
- Prose: lowering k below 2 raises the threshold from `Σpᵢ²` toward a weaker constraint (more tokens survive), which opens the sampler on peaked positions and prevents the loop-attractor from becoming absorbing.
- Framing: "first empirical evaluation of the origin paper's Appendix-B.5 generalization on reasoning-code."
- **DeepSeek framing**: "recovers the loss the α=2 collision-entropy threshold imposed on the DeepSeek p-less baseline" — NOT "best-in-class on DeepSeek" (0.485 doesn't back it).

**Evidence sources**:
- Results dirs: `results/_renyi_sweep_full252/{Qwen--Qwen3-8B, deepseek-ai--DeepSeek-R1-Distill-Llama-8B}/ATCODER_interview/`
- Baselines (per memory `project_apps_apple_to_apple_baselines.md`): `results/pless_recovery_full252/…` (Qwen), `results/_deepseek_fixed_full252/…` (DeepSeek)
- Sweep tables: `docs/decoder_comparison_cot_apps_{qwen3, deepseek}.md`

## 6. Discussion, limitations, future work (~0.5 pg)

- p-less as a strong default only where distributions are not RL-peaked; G_k with k<2 as the reasoning-model-safe drop-in.
- Limitations: two reasoning models; one benchmark family for reasoning-model rescue (APPS-252); base-model coverage on HE is limited to one model; no instruct-model G_k sweep.
- Future: G_k on instruct-CoT-prompted models; broader benchmark coverage; per-token k schedules.

## 7. Reproducibility appendix

- Repo, commit hashes, exact CLI flags per result cell.
- Reference `results/analysis/consolidated_summary.csv` for the MBPP/HE cells; per-model JSONs for the primary source.
- `p-less/` submodule commit hash + `scripts/paper/make_llm4code_tables.py` invocation.
