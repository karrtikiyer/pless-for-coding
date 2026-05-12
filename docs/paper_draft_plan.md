# Draft Plan — Producing v1 of the Workshop Paper

Companion to `docs/paper_plan.md`. That document fixed the *what* (positioning, contributions, structure). This one fixes the *how* — a step-by-step procedure to produce a complete, source-grounded first version, with explicit guardrails against fabricated numbers.

## Ground rules (non-negotiable)

1. **Every numerical claim cites a source line** in `paper/sources.md` of the form `<claim ID> → <file>:<line range or table row> → <verbatim quote>`. If a claim has no source, it does not go in the draft.
2. **No findings beyond what's in the result files.** If the data don't support a claim, it goes in `paper/TODO.md` (deferred) or in the §5 Limitations section. We do not bluff.
3. **No invented baselines.** Only baselines that actually have rows in `consolidated_summary.csv` or per-model reports get tables.
4. **Statistical caveats stay attached to the numbers.** SE on pass@1 is ~1.75pp on MBPP-500 and ~2.8pp on HumanEval-164 (per `cross_benchmark_t1_analysis.md`). Differences <2 SE are reported as "directional", never as "shown" or "demonstrated".
5. **No language hyperbole** ("dramatic", "remarkable", "groundbreaking"). Use neutral, precise verbs ("improves by", "matches", "drops to").

## Format choice — markdown for v1

Use **Pandoc-flavored markdown** for the draft. Reasons: faster iteration, easier diffs, compatible with the existing markdown reports we'll quote from. Convert to LaTeX with the workshop's `.sty` file once content is locked (estimated 2–4 hours of mechanical work near the end).

Trade-off acknowledged: page count in markdown is approximate (~ 60–80 lines = 1 LaTeX page double-column). Track approximate page count in the TOC; do an early conversion test if word budget gets tight.

## Directory layout

```
paper/
  draft.md              # the v1 manuscript
  sources.md            # claim → source-file:line audit log
  TODO.md               # deferred / unresolved items
  tables/
    table1_models.md
    table2_headline_passk.md
    table3_t1_t2.md
    table4_cross_benchmark.md
    make_tables.py      # generates the above from consolidated_summary.csv
  figures/              # symlinks to PNGs in results/
    fig1_pareto.png  -> ../../results/pless_full_mbpp_results/analysis/figures/pareto_correctness_diversity.png
    fig2_temp_sweep.png -> ...
    fig3_cover_at_t.png -> ...
  refs.bib
```

## Phase 0 — Infrastructure (estimated 1–2 hours)

1. Create `paper/` directory + the empty subfiles above.
2. Write `paper/refs.bib` with **only papers we can ground**:
   - Tan, Wu, Howard 2026 — arXiv:2509.23234 (p-less origin)
   - Wei et al. 2024 — arXiv:2402.06925 (decoding survey)
   - Chen et al. 2021 — HumanEval (arXiv:2107.03374)
   - Austin et al. 2021 — MBPP (arXiv:2108.07732)
   - Roziere et al. 2023 — CodeLlama (arXiv:2308.12950)
   - Hui et al. 2024 — Qwen2.5-Coder (arXiv:2409.12186)
   - Ren et al. 2020 — CodeBLEU (arXiv:2009.10297)
   - Zhang & Shasha 1989 — tree edit distance (algorithmica)
   - 2507.03160 — SLM-for-code empirical study (orthogonal context only)
3. Write `paper/tables/make_tables.py` — a thin script that reads `results/analysis/consolidated_summary.csv` (193 rows) and `results/analysis/consolidated_report.md` and emits the markdown tables. **No hand-computed numbers.**
4. Symlink the figures (don't copy — keeps them in sync if regenerated).

**Acceptance**: `paper/` directory exists with empty files; `make_tables.py` runs and emits at least Table 1 + Table 2 from the CSV.

## Phase 1 — Source-of-truth audit (BEFORE any prose)

This is the most important phase. Goal: anchor every claim from `docs/paper_plan.md` to a real file:line, or downgrade/remove the claim.

For each candidate claim from the paper plan, locate it in the source files and record in `paper/sources.md` with this template:

```
[C1] "p-less-norm@0.6 ranks 1/19 on Llama-2-7B at 22.3% pass@1, beating FSD-d (21.2%) and Beam-8 (19.4%)"
  source: results/pless_full_mbpp_results/analysis/<llama-7B-comparison-report>.md:<line>
  verbatim: "<paste the exact line>"
  status: VERIFIED | DOWNGRADE | REMOVE
```

**Claims to verify in the audit pass** (from the `paper_plan.md` contributions):

| ID | Claim | Likely source |
|----|-------|---------------|
| C1 | p-less-norm@0.6 ranks 1/19 on Llama-2-7B at 22.3% | `pless_full_mbpp_results/analysis/<per-model report>.md` |
| C2 | Qwen2.5-Coder-7B-Instruct (HumanEval): p-less@0.6 = 87.5% pass@1 vs greedy 84.1% | `pless_human_eval_results/full_precision_results/analysis/report.md` |
| C3 | Codestral-22B *improves* +9.7pp from T1=0.7→2.0 (HumanEval) | `pless_human_eval_results/temprature_results/analysis/temperature_sweep_report.md` |
| C4 | Qwen2.5-Coder-7B drops -19.9pp from T1=0.7→2.0 (HumanEval) | same as C3 |
| C5 | Qwen3-Coder-30B marginal pless gain ~0.6pp | `pless_human_eval_results/full_precision_results/analysis/report.md` |
| C6 | T1 sweet spot 0.7–1.5; cliff at T1=2.0 (~5pp); catastrophe at T1=3.0 (>70pp) | `full_mbpp_pre_post_temp_pless/analysis/t1_t2_comparison_report.md` |
| C7 | T2 costs 2–4pp pass@1 for ~0.03 struct_div gain on Qwen2.5-Coder-Instruct | same as C6 |
| C8 | pless T1=0.8 on Qwen2.5-Coder-7B-Instruct: 58.7% pass@1, 0.167 struct_div | `full_mbpp_pre_post_temp_pless/analysis/<per-model>.md` |
| C9 | SE pass@1 ≈ 1.75pp (MBPP-500) and ≈ 2.8pp (HumanEval-164) | `cross_benchmark_t1_analysis.md` |
| C10 | All models collapse between T1=2.0 and T1=3.0 | `temperature_sweep_report.md` |
| C11 | "Quality-filter hypothesis still unconfirmed" | `t1_t2_comparison_report.md` |

**Process**: read each source file, locate the line, paste verbatim into `sources.md`, mark VERIFIED. Any claim that cannot be verified gets DOWNGRADED (e.g., "+9.7pp" → "improves at higher T") or REMOVED.

**Acceptance**: `paper/sources.md` has at least 11 verified entries before any prose is written.

## Phase 2 — Tables and figures (1–2 hours)

### Tables — generated, not hand-typed

| Table | Source | Notes |
|-------|--------|-------|
| **Table 1** Models | `pyproject.toml`, `run_bench.sh`, run scripts | 11 rows: name, family, size, instruct?, source URL |
| **Table 2** Headline pass@k | `consolidated_summary.csv` (193 rows) | One row per (model, dataset). Methods columns: greedy, temp@0.7, top-p@0.95, p-less@0.6, p-less-norm@0.6 |
| **Table 3** T1/T2 grid | `t1_t2_comparison_report.md` | Already in markdown — re-format only |
| **Table 4** Cross-benchmark replication | `cross_benchmark_t1_analysis.md` | 5 conclusions × {MBPP, HumanEval} × {confirmed/disconfirmed} |

`make_tables.py` reads CSV/md, emits markdown tables. Run it; verify rows. Do not hand-edit.

### Figures — reuse existing PNGs

| Figure | Source PNG | Action |
|--------|------------|--------|
| **Fig 1** Pareto correctness × structural diversity | `pless_full_mbpp_results/analysis/figures/pareto_correctness_diversity.png` | Symlink |
| **Fig 2** pass@1 vs T1 across 6 HumanEval models | `pless_human_eval_results/temprature_results/analysis/figures/<sweep>.png` | Symlink |
| **Fig 3** cover@t bar chart | `pless_full_mbpp_results/analysis/figures/<cover>.png` (if exists) or **make new** | Decide after audit |
| **Fig 4** (optional) per-model pass@k by method | per-model directories | Defer to appendix |

**Acceptance**: 4 tables and 2–3 figures selected; symlinks in place; first-pass captions drafted.

## Phase 3 — Section drafting (3–5 hours total, one section at a time)

Write in this order. After each section, **stop and re-audit** before moving to the next.

### §3 Methodology FIRST (1 hour)

Why first: it's the most factual, least-rhetorical section. Sets the vocabulary used in §4.

- Subsection 3.1 Models — refer to Table 1.
- Subsection 3.2 Sampling configurations — list verbatim from `run_bench.sh` and the run scripts; quote the CLI invocations.
- Subsection 3.3 Benchmarks — MBPP-full (500 problems, 10 samples), HumanEval (164 problems, 10 samples). Cite Austin et al. and Chen et al.
- Subsection 3.4 Metrics — pass@k (cite original Codex paper), cover@t (define from `bench/eval/metrics.py`), 5 diversity metrics (cite CodeBLEU and zss for tree-edit). Keep formulas short.
- Subsection 3.5 Statistical caveats — quote SE numbers from `cross_benchmark_t1_analysis.md`. State "differences <2 SE reported as directional".

**Don't write yet**: any sentence that interprets results.

### §2 Background and Related Work (45 min)

Write **after** Methodology so the vocabulary is settled.

- Sentence 1–2: define p-less and p-less-norm with the exact threshold formulas from `p-less/p_less_samplers.py`. Cite Tan et al. 2026.
- Sentence 3–4: prior code-decoding work — Wei et al. 2024 surveyed many methods on HumanEval/MBPP without p-less. Cite verbatim from Wei abstract.
- Sentence 5–6: orthogonal context — Llanes-Jurado et al. 2025 (arXiv:2507.03160) studied 20 SLMs on code without varying decoding methods.
- Sentence 7: gap statement — code generation with p-less is unevaluated.

**Length target**: 1/2 page. Don't over-cite; this is a workshop paper.

### §1 Introduction (1 hour)

Write **after** Background. Now you have the language to motivate.

- Paragraph 1: code-LM decoding matters. Two demands: correctness and diversity.
- Paragraph 2: hyperparameters are real costs (top-p, top-k, η, etc.). Cite Wei et al. for evidence of sensitivity.
- Paragraph 3: p-less is hyperparameter-free in principle but unevaluated on code. We test that promise.
- Paragraph 4: contributions list (verbatim copy from `docs/paper_plan.md`, refined to match what the audit verified).

**Don't write**: forward references to numbers we haven't sourced.

### §4 Results (3–4 hours, the hardest section)

Write each subsection only after locating its anchor numbers in `sources.md`.

- **§4.1 Headline pass@k** — Table 2 + 3 paragraphs interpreting the wins/losses on the canonical configs. Cite C1, C2.
- **§4.2 Pareto correctness × diversity** — Figure 1 + 2 paragraphs. Be precise: name which models p-less is on the frontier for, name which it isn't.
- **§4.3 Robustness boundary** — Figure 2 + 4 paragraphs. Cite C3, C4, C6, C10.
- **§4.4 T1/T2 interaction** — Table 3 + 2 paragraphs. Cite C7, C11. **Negative result**: T2 is dominated.
- **§4.5 Cross-benchmark replication** — Table 4 + 1 paragraph. Cite C9.

**Banned in §4**: any speculation about *why* (mechanisms). Mechanisms go in §5.

### §5 Discussion (45 min) and §6 Conclusion (15 min)

- §5: where p-less wins (weaker/base models), where it doesn't (strong instruct models with marginal gains, e.g., Qwen3-Coder-30B per C5). The Codestral outlier is interesting; offer 1 sentence of speculation, label it speculation. Limitations: 10 samples/task, no agentic / multi-turn, no BigCodeBench / LiveCodeBench.
- §6: 4–6 sentences. p-less is a competitive baseline; T1 still matters; T2 is dominated; future work = larger sample budgets and agentic settings.

### Abstract — write LAST (30 min)

200 words. Mirror the contributions list in §1 but tighter. No new claims in the abstract.

## Phase 4 — Reproducibility audit (1 hour)

Before declaring v1 complete:

1. Run `uv run python -m bench.eval.consolidated_eval` and confirm `consolidated_summary.csv` matches what's in `paper/tables/`. If anything drifts, fix the source — never the table.
2. Read `paper/sources.md` end-to-end. Every entry should be VERIFIED or explicitly DOWNGRADE.
3. Search the draft for digits not in `sources.md` — every such number must be added to `sources.md` or removed.
4. Spot-check 5 random claims by re-reading the source file.
5. Ensure every figure caption references the data file and config that produced it.

## Phase 5 — v1 acceptance criteria

The first version is complete when **all** are true:

- [ ] All 6 sections + abstract present, no `[TODO]` placeholders inline.
- [ ] `paper/sources.md` has ≥ 11 VERIFIED entries.
- [ ] Page-count estimate is within budget (target 8 pages excluding refs/appendix; warn at 9, hard cap 10).
- [ ] All 4 tables generate from `make_tables.py` (no hand-typed numbers).
- [ ] All figures are symlinks to PNGs in `results/` (no PNG editing).
- [ ] `paper/TODO.md` lists every deferred item (e.g., bootstrap CIs, Codestral confirmatory run, BigCode-2507 cohort placement) — these become v2 work, not v1 blockers.
- [ ] Bibliography compiles without warnings.
- [ ] An external reader (the user) can spot-check any claim by reading `sources.md` → the cited file.

## Estimated effort

| Phase | Hours |
|-------|-------|
| 0 Infrastructure | 1–2 |
| 1 Source audit | 2–3 |
| 2 Tables + figures | 1–2 |
| 3 Section drafting | 3–5 |
| 4 Reproducibility audit | 1 |
| 5 Polish | 1 |
| **Total** | **9–14 hours** |

Spread over **2–3 working days** if done sequentially. Could be parallelised: I can run audits + tables while the user reviews §1–§3 prose.

## Decisions to make before starting

1. **Target venue lock-in**: DL4C @ NeurIPS 2026 (primary) — confirm? If yes, watch dl4c.github.io for CFP from June 2026.
2. **LaTeX timing**: defer LaTeX conversion to after v1 lock (recommended), or start in LaTeX now? Recommend defer.
3. **Author list**: needed before submission, not before v1.
4. **Codestral T=2 confirmatory run**: do it now (adds 1 day) or note as Limitation? Recommend defer to v2.
5. **BigCode-2507 cohort**: include in main paper or appendix? Recommend appendix to keep main paper coherent at 11 models.

## Risks specific to v1 drafting

- **Audit may invalidate claims** — if a number doesn't match the source, downgrade or remove. This is fine; better to find now than at submission.
- **Page budget pressure** — 11 models × multiple methods × 2 benchmarks easily explodes. Tables go in appendix when they don't directly support a §4 paragraph.
- **Source files have inconsistencies** — older reports may use different metric definitions. The audit should flag any such inconsistency and pick a single canonical source (`consolidated_summary.csv` is canonical when conflicts arise).

## What this plan deliberately does *not* include

- Drafting prose right now. Phase 0 + Phase 1 must happen first.
- Inventing numbers. If a claim isn't in the result files, it's not in the paper.
- Speculative section reordering. The structure is fixed by `docs/paper_plan.md`.
- New experiments. v1 ships with what we have.
