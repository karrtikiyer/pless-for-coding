# Narrative A — manuscript skeleton (Phase-4)

**Working title:** *When Does Hyperparameter-Free Decoding Break? A Rényi-α Analysis of
Entropy-Threshold Sampling for Code and Reasoning*

**Target:** ICLR 2027 main track (Sep 24, 2026), framed as an empirical/analysis paper. ARR August
is a later cycle, not this assembly. **Confirm page limit in the ICLR 2027 author guidelines** (do
not assume); budget below assumes ~9 main pages + unlimited refs/appendix.

**One-line thesis.** Hyperparameter-free entropy-threshold sampling (p-less) is a high-accuracy /
low-diversity decoder: competitive-to-best at pass@1 on short base-model code, but it *silently
degenerates* on long chain-of-thought because peaked distributions collapse the threshold to greedy →
loops; the Rényi-α family is the lens that both explains and repairs this, yet still only ties
well-tuned temperature — so the contribution is a *characterization of when parameter-free decoding
helps vs breaks*, not a new SOTA sampler.

**Every number cites a grounded source. Do not paste any value not in one of:**
`docs/research/paperA_master_numbers.md` (APPS+MBPP+HE), `…/paperA_renyi_nonequivalence.md` (α),
`…/paperA_loop_positioning.md` (loops), `paper/draft.md` + its `sources.md` (MBPP/HE spine).

---

## Contributions (claim list — all defensible per Phase 0–3)
1. **A when-does-it-help boundary** for hyperparameter-free decoding on code, across 13 checkpoints
   (MBPP/HE) + 2 reasoning models (APPS): competitive at pass@1, dominated at pass@10, catastrophic
   on long CoT. [master_numbers §1–3]
2. **A mechanism**: peakedness × hard-threshold → survivor set collapses to the mode → repetition /
   non-termination. Unifies the pass@10 diversity gap (HE) and the CoT loop (APPS). [master_numbers
   §3; `docs/theory/looping_sampler_class_and_related_work.md`; dip-test in
   `docs/theory/entropy_mechanism_framework.md`]
3. **The first empirical α-sweep** of an entropy-threshold sampler (τ_α=Σpᵢ^α), with a self-contained
   non-equivalence to the origin paper's un-evaluated rooted form. [renyi_nonequivalence]
4. **Code-specific reasoning-loop findings** uncovered by prior work: a paraphrastic-vs-verbatim
   fraction on APPS, and a verified *non-transfer* of the published hidden-state precursor to code
   (EDR 0.76→0.17). [loop_positioning]
5. **A reproducibility post-mortem**: three silent bugs (tokenizer #45488, blanket smoothing, fp32
   all-pruned) shifted pass@1 up to 22pp and fabricated an "adaptive ≫ prevention" result.
   [`docs/pless_alpha_comparison_apps.md` §Correction; `docs/theory/todos.md` A41/E6-E7]

---

## Section map (purpose | reuse | source | figures)

### Abstract  — REWRITE (from plan draft + corrections)
Fold in: the pass@1-vs-pass@10 tradeoff (not "pless loses"), α as lens, "only ties temp,"
reproducibility. Base text: plan file `idempotent-fluttering-deer.md` §Narrative A draft abstract →
edit per master_numbers corrections.

### 1. Introduction — REUSE draft §1, RE-FRAME
- Keep draft.md:30–56 (decoding-as-afterthought, hyperparameter sensitivity, p-less proposal).
- ADD the boundary thesis + the CoT-breakage hook (new). Move from "competitive but not dominant"
  (draft's conclusion) to "competitive where distributions are flat, silently broken where peaked."
- Contribution list above (replaces draft's C1–C12-style list).

### 2. Background & Related Work — REUSE draft §2, EXTEND
- Reuse draft.md:88–125 (p-less/norm defs, Shi et al. 2024 survey, the gap).
- ADD three related-work paragraphs: (a) reasoning-loop detection — Circular Reasoning 2601.05693,
  Word Salad Chopper 2511.00536 (math/QA, detection-only); (b) "escape is hard" — 2506.10979;
  (c) Rényi/entropy samplers — min-p 2407.01082, top-nσ 2411.07641, η 2210.15191, Holtzman 1904.09751.
  Positioning per `loop_positioning.md` (our delta = code + fraction + sampler-prevention + boundary).

### 3. Methodology — REUSE draft §3 + ADD APPS
- Reuse draft.md:127–253 (models table, sampling grid, pass@k/cover/diversity, SE bands).
- ADD APPS-CoT setup: Qwen3-8B + DeepSeek-R1-Distill, 252×10, thinking on, α∈{2..5}, adaptive,
  temp variants; canonical dirs + `scripts/build_decoder_comparison_table.py`. [master_numbers §1]
- ADD the τ_α definition + α=2≡p-less (renyi_nonequivalence §Setup).

### 4. Where it helps — REUSE draft §4.1–4.2, TRIM
- MBPP Llama-2-7B rank 1/19 (pless_norm@0.6 22.3%), base-model wins; HE pass@1-competitive.
- Pareto: pless owns high-pass@1/low-diversity end; temp owns diversity end (draft §4.2, Figs 1a-c).
- Source: master_numbers §2–3; draft Table 2; `comparison_report.md`, `report.md`.
- **Correction to bake in:** frame HE as the pass@1-vs-pass@10 tradeoff (Codestral p_less 78.0 >
  temp 72.6 @1; temp 91.5 > pless 84.8 @10). Do NOT cite consolidated_report aggregates.

### 5. Where it breaks: peaked distributions & long CoT — NEW (core)
- The T₁ cliff (draft §4.3, Fig 2) as the bridge: raise temperature → pless collapses.
- APPS: α=2 loops (DeepSeek 41.8% / Qwen 14.5% trunc), worst pass@1. [master_numbers §1]
- Mechanism: peakedness × hard threshold (contribution 2 sources).
- Loop taxonomy on code: paraphrastic vs verbatim fraction; precursor non-transfer (EDR 0.76→0.17).
  [loop_positioning]

### 6. The α knob — NEW
- α-sweep removes the loop (trunc→~0-1%), recovers pass@1, monotone. [master_numbers §1 APPS tables]
- Non-equivalence to origin's G_k (coincide at 2, opposite for >2; filter example).
  [renyi_nonequivalence — reproduce the 2 tables + the reachable-set example]
- **The honest punchline:** best α only *ties* temp (DeepSeek α=5 0.483 ≈ temp 0.480; Qwen temp
  0.705 > α=5 0.686). α is a *diagnostic lens + repair*, not a new SOTA sampler.

### 7. Prevention vs rescue (absorbs Narrative B) — NEW, COMPACT
- Prevention (α up front) vs reactive chop-and-continue; α=5 re-entry suppression (0/162 vs 54/101).
  Source: `docs/pless_alpha_comparison_apps.md`, `docs/chop_rescue_summary_for_researchers.md`.
- Keep short — one subsection; both are matched by good temp (do not over-claim).

### 8. Reproducibility post-mortem (absorbs Narrative C) — NEW, COMPACT
- #45488 tokenizer mangling + blanket smoothing + fp32 all-pruned; the fabricated "adaptive ≫
  prevention" that reversed on fix. Source: `docs/pless_alpha_comparison_apps.md` §Correction;
  `docs/theory/todos.md` A41/E6-E7; commits abdc0dc/118292e/9b33fc8.

### 9. Discussion & Limitations — REUSE draft §5–6, EXTEND
- Reuse draft's "where it helps / doesn't / practical recommendation."
- ADD honest limitations: pless only ties temp; majorization conjecture NOT unique to α (top-p/top-k
  satisfy it too — do not imply α is uniquely principled); no proven theory; 2 reasoning models.

---

## Reuse map (draft.md → outline)
| draft.md section | fate | lands in |
|---|---|---|
| Abstract | rewrite | Abstract |
| §1 Intro | reuse+reframe | §1 |
| §2 Background | reuse+extend | §2 |
| §3 Methodology | reuse+extend | §3 |
| §4.1 headline pass@k | reuse (Shi-cite fixed) | §4 |
| §4.2 Pareto | reuse | §4 |
| §4.3 T₁ sweep | reuse as bridge | §5 |
| §4.4 T₁/T₂ | move to appendix (secondary) | App |
| §4.5 cross-benchmark | condense | §4/§9 |
| §5 Discussion, §6 Conclusion | reuse+extend | §9 |

## Length budget (≈9 pp target — VERIFY limit)
Intro 1 | Background 0.75 | Method 1 | §4 helps 1.25 | §5 breaks 1.75 | §6 α 1.5 | §7 prevention 0.5 |
§8 repro 0.5 | §9 discussion 0.75. Appendix: T₁/T₂ grid, full APPS tables, per-model MBPP/HE,
detector configs, majorization brute-force note.

## Not-yet-done (before a full draft)
- Regenerate figures for APPS (currently only tables); pick 1-2 headline figures for §5/§6.
- Write the abstract + intro prose (this file is structure only).
- Decide: keep §7/§8 as sections or fold into §5/§9 if page-tight.
- Verify ICLR 2027 page limit + format (author guidelines).
- LaTeX conversion (draft is markdown).
