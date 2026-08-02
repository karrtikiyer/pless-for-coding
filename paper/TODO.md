# Deferred / Unresolved Items

## Audit-pending claims

(list grows from sources.md as items are downgraded/removed during Phase 1)

## Phase-1 claim audit for Narrative A (2026-07-30)

Draft `paper/draft.md` (v1 "When Does p-less Sampling Help Code LLMs?") audited against the
live-pulled master numbers (`docs/research/paperA_master_numbers.md`). Verdict per block:

**REUSE (numbers verified consistent with clean per-model reports — this is Narrative A's
"where it helps" + "T1 boundary" spine):**
- Abstract, §1 Intro, §2 Background, §3 Methodology.
- §4.1 Table 2 headline pass@k — spot-checks match `report.md`/`comparison_report.md`:
  Qwen2.5-Coder-7B-Instruct pless@0.6 0.875 / greedy 0.842; Codestral pless@1.0 0.780→0.848 /
  temp@0.7 0.730→0.908; CodeLlama-7b-Instruct pless@1.0 0.355→0.378 / temp@0.7 0.363→0.634. ✓
- §4.1 Llama-2-7B base: pless_norm@0.6 22.3% rank 1/19 (beats FSD-d 21.2, beam-8 19.4). ✓
- §4.2 Pareto (pless owns high-pass@1/low-diversity end), §4.3 T1 sweep, §4.4 T1/T2, §4.5
  cross-benchmark, §5 Discussion, §6 Conclusion ("competitive but not dominant"). All reusable.

**FIX — citation bugs (verified via arXiv fetch, both wrong in the draft):**
- arXiv:2402.06925 is **Shi et al. [2024]** (Chufan Shi et al., EMNLP 2024) — the draft calls it
  "Wei et al. [2024]" (text) and "Zhu et al. (2024)" (Fig 1a caption). BOTH WRONG. Note: CLAUDE.md's
  "Yi et al." is ALSO wrong. Global replace → **Shi et al. [2024]** everywhere (draft + CLAUDE.md).
- Origin paper arXiv:2509.23234 is **Tan, Wu, Howard, Sept 2025** — draft says "Tan et al., **2026**"
  (wrong year). Fix to 2025.

**RECHECK — data-source hygiene (resolved, but note for the manuscript):**
- Draft Table 1/2 cite `consolidated_summary.csv`; its raw rows are duplicated/conflicting, but the
  draft's `paper/tables/make_tables.py:canon_method()` dedups and the resulting numbers match the
  clean per-model reports. → OK to keep, but do NOT cite `consolidated_report.md` aggregates
  elsewhere (see `docs/research/paperA_master_numbers.md` §Correction).
- MBPP Llama pless@1.0: draft uses 19.8 (canonical `pless_full_mbpp_results/`), NOT the stale 28.5
  from `pless_mbpp_results/`. Correct corpus. ✓ (RECHECK closed.)

**ADD — new content for Narrative A (not in v1 draft; the reasoning-CoT half + α knob):**
- "Where it breaks" — APPS Qwen3-8B / DeepSeek-R1-Distill CoT looping (α=2 truncation 42%/15%).
  Note: line 17 below records these Qwen3-8B split-decoding runs were *excluded from v1 by request*;
  Narrative A re-includes them as the core.
- The Rényi-α family τ_α=Σpᵢ^α + first empirical α-sweep (C5 novelty anchor; Phase 2).
- Prevention-vs-rescue section (absorbs Narrative B).
- Reproducibility post-mortem: #45488 tokenizer + smoothing + fp32 (absorbs Narrative C).

**Framing carry-over:** the draft's thesis ("competitive but not dominant; the removed
hyperparameter is the value") is already Narrative-A-compatible — extend, don't rewrite. Also fold in
the corrected HumanEval reading (pass@1-competitive / pass@10-dominated tradeoff), which the draft's
§4.2 frontier analysis already states qualitatively.

## Deferred to v2

- Bootstrap CIs on pass@k (currently using analytic SE from cross_benchmark_t1_analysis.md).
- Codestral T=2 confirmatory run (single-temperature outlier; would need re-run to rule out seed luck).
- BigCode-2507 cohort placement (main paper vs appendix).
- LaTeX conversion (deferred per Phase 0 of draft plan).
- Author list (needed before submission, not before v1).

## Open methodology decisions

- Whether to include the Qwen3-8B split-decoding experiments (excluded from v1 per user request — token-budget confound documented in `results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis/truncation_partition.md`).
- Whether to keep beam4/beam8/greedy single-sample baselines in headline tables or move them to appendix.
