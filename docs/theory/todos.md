# pless-for-coding — wholistic TODO list

**Living document.** Persists the open / proposed / deferred items across
sessions so we don't lose context. Update *every time* a TODO is added,
status changes, completed, dropped, or re-prioritized. Append a new
entry to the **Version history** section at the bottom describing the
change and the context (the conversation thread or commit that motivated
it).

## How to maintain this file

1. **When adding a new TODO**: assign next free ID in the relevant
   section (A/B/C/D/E/F/G). Use the status legend below. Add an entry
   to Version history with date + reason.
2. **When status changes** (e.g. PROPOSED → ACTIVE → DONE): update the
   status column inline AND append a Version history entry citing the
   commit / discussion that drove the change.
3. **When completing**: leave the row in place with status DONE — don't
   delete. The historical context matters for paper writing.
4. **When dropping**: status DROPPED + a 1-line "why" in the row's
   notes column. Version history entry citing where it was abandoned.
5. **Estimates are rough**: GPU-hr or API-$ where applicable, day-budget
   for theory work.

## Status legend

- **ACTIVE** — discussed, ready to launch / next-step candidate; should be picked up if we have time
- **PROPOSED** — discussed but not yet committed to; user-decision needed before launching
- **DEFERRED** — explicitly parked (by user or by recommendation); revisit later
- **DONE** — completed; left in place for historical context
- **DROPPED** — investigated and abandoned with documented reason
- **OPEN** — open question we encountered but didn't resolve; needs a decision

---

## A. Compute / experiments

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| A1 | ACTIVE | APPS sweep — 3 models × 6 (source × difficulty) buckets × 4 α via vLLM | ~12 hr / 4 GPU | Plan in `docs/theory/diversity_majorization_plan.md` lineage; scripts at `run_pless_alpha_apps_all_models.sh`. Never launched. |
| A2 | ACTIVE | T-sweep cleanups — m-a-p HE T-sweep + pless@T={2.5,3.0} on CodeLlama + m-a-p MBPP | ~3 hr / 1 GPU | Script `run_t_sweep_cleanups.sh` |
| A3 | PROPOSED | NAUADC on Qwen3-NoThink × 4 α arms | ~$30, ~3-4 hr API | Predicts larger NAUADC growth than Qwen3-Think (less saturated). Third data point for decisive test. |
| A4 | PROPOSED | NAUADC on non-α winners (pless_t2.0, top_p0.9, top_k5) | ~$30-50, ~3-4 hr API | For apples-to-apples NAUADC vs cb_div comparison; addresses the sampler-comparison gap. |
| A5 | DEFERRED | Qwen3-8B on APPS | — | User explicitly said skip in AskUserQuestion answer (2026-05-19). |
| A6 | DEFERRED | Extended α grid (α=4.0, α=7.0) | ~30 min/model | Beyond current {2.0, 2.5, 3.0, 5.0}. |
| A7 | DEFERRED | Qwen3 split decoding (`split_temp_pure_*_pless_alpha_*`) | — | Uniform α chosen instead per user. |
| A8 | DEFERRED | T1/T2 experiment on instruct models | — | In project memory (`project_instruct_t1t2_experiment.md`) from before current session. |
| A9 | DONE | Qwen3-8B α sweep MBPP + HE, thinking ON | — | Commit `d779042` (eval), `308cf24` (β-binomial extension). |
| A10 | DONE | NAUADC × Qwen3-Think × 4 α arms (MBPP) | ~$30 spent | `results/pless_alpha_full_mbpp/Qwen--Qwen3-8B/analysis/`. |
| A11 | DONE | Qwen3-8B α sweep MBPP + HE, thinking OFF (decisive test) | — | Commit `b9d30e8`. Mechanism B (thinking) wins. |

## B. Theory / proof work

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| B1 | ACTIVE | Phase 1 — majorization theorem proof for α-truncation (`q_α' ≺ q_α` for α<α'), with drop-event handling | 5 working days | Plan in `docs/theory/diversity_majorization_plan.md`. Confidence ~80-85% post-Phase-0.6. |
| B2 | PROPOSED | Class-level theorem — prove monotonicity for top-p + top-k too | +3-7 days after B1 | Step 0.6 showed top-p/top-k empirically satisfy the same monotonicity. Proving them would make the paper "first proved class" instead of "first proved instance". |
| B3 | DROPPED | C7 Step 5 — predict (a_α, b_α) from per-position entropy aggregates | — | R² gains 3-5 pp; coefficient signs flipped across models; no clean mechanism. See `docs/theory/c7_step5_verdict.md`. |
| B4 | DONE | C7 v3 β-binomial framework — `ν(α)` regularity across 3 non-thinking + Qwen3-NoThink | — | `bench/eval/validate_pass_at_k_beta_binomial.py`. |
| B5 | DONE | C7 v3 decisive test — Mechanism B (thinking) over Mechanism A (saturation) | — | Within-model dichotomy on Qwen3-8B; commit `b9d30e8`. |
| B6 | DONE | Phase 0.5 + 0.6 brute-force checks (α, top-p, top-k all empirically monotone) | — | Commits `7e0c174`, `b3f8fd7`. |

## C. Analysis / synthesis

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| C1 | PROPOSED | ν(α) trajectory analogue for non-α samplers (pless@T sweep, top_p sweep, top_k sweep) | ~1 hr CPU | Does the C7 v3 "ν grows, mean p flat" pattern extend to other operators? If yes, the dichotomy generalizes to the whole truncate-and-renormalize class. |
| C2 | PROPOSED | Pareto-frontier visualization combining α-arm + non-α data on (pass@10, cb_div) | ~1 hr | Visual answer to "do α arms unlock a region?" — complements the existing sampler_comparison_summary.md tables. |
| C3 | PROPOSED | Add `temprature_results` HumanEval back into scope for m-a-p HE only | ~15 min CPU | User excluded it from sampler-comparison scope earlier — may have been unintentional; currently no non-α HE data for m-a-p. |
| C4 | OPEN | Investigate `p_less_norm` Qwen HE config (pass@10=95.12%, missing cb_div) | ~30 min | Older `full_precision_results` schema. Worth re-evaluating or excluding from analysis? |
| C5 | DONE | Sampler comparison (α-arm vs other stochastic) with Pareto-dominance check | — | `bench/eval/sampler_comparison.py`; commit `6f9ad44`. α arms NOT uniquely best. |
| C6 | DONE | β-binomial fit extension to Qwen3-Think + Qwen3-NoThink | — | Commits `308cf24`, `b9d30e8`. |

## D. Writing / paper

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| D1 | PROPOSED | Workshop paper outline / draft | 1-2 days | Offered earlier; never written. Needs revised positioning per Step 0.6 (α not uniquely principled). |
| D2 | ACTIVE | Update cross-model summary `results/pless_alpha_full/cross_model_cross_dataset_summary.md` | ~1 hr | Currently missing Qwen3 results, decisive-test verdict, sampler-comparison finding. |
| D3 | ACTIVE | Update C7 v3 verdict doc `docs/theory/c7_verdict_v3_findings.md` | ~30 min | Narrow claim from "universal regularity" to "non-thinking-model regularity"; cite decisive test + sampler-comparison. |
| D4 | DONE | Qwen3 full_sweep_summary.md with NAUADC + decisive-test verdict | — | `results/pless_alpha_full_mbpp/Qwen--Qwen3-8B/full_sweep_summary.md`; commits `c049892`, `d779042`, `b9d30e8`. |
| D5 | DONE | Diversity-theorem plan doc with revised positioning (post-Step 0.6) | — | `docs/theory/diversity_majorization_plan.md`; commit `b3f8fd7`. |

## E. Code / infrastructure

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| E1 | DEFERRED | vLLM `--tensor-parallel-size` wiring | ~30 min | Skip unless empirically needed for long APPS sequences. |
| E2 | PROPOSED | Reorganize folder-layout doc | ~15 min | You moved Qwen3 MBPP from `pless_alpha_full/` to `pless_alpha_full_mbpp/`; other 3 models followed. CLAUDE.md "Results Directory Structure" may be stale. |
| E3 | DONE | Canonical headline_table helper (`bench/eval/headline_table.py`) | — | Commit `c049892`. 8-col format locked. |
| E4 | DONE | Sampler-comparison tool + Pareto-dominance check | — | `bench/eval/sampler_comparison.py`; commit `6f9ad44`. |
| E5 | DONE | Brute-force majorization checkers (α + top-p/top-k) | — | `bench/eval/check_majorization_*`. |

## F. Methodology / process

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| F1 | DONE | Global `~/.claude/CLAUDE.md` Scientific Rigor section | — | Applies cross-project; not in this repo's git. |
| F2 | DONE | Project memory `feedback_verify_before_asserting.md` | — | In `~/.claude/projects/-Users-.../memory/`; not in repo git. |
| F3 | DONE | Stop phrase established: "did you verify?" / "show me the evidence" | — | Active for this and future sessions. |

## G. Older / earlier-session leftovers

| # | Status | Item | Estimate | Notes |
|---:|---|---|---|---|
| G1 | OPEN | Pod cluster CODEFORCES parquets (task #9) | — | Task from start of multi-day session; never picked up. |
| G2 | OPEN | Pull responses + update report (task #10) | — | Same. |

---

## Recommended next step (as of 2026-05-21)

If picking **one** item for ~5-day investment: **B1** (majorization theorem proof for α-truncation). Only item that could meaningfully change the paper's theoretical claim from "empirical observation" to "first proved theorem in a conjectured class."

Cheap parallel items worth doing alongside:

- **C1** — extend β-binomial ν(α) analysis to top-p/top-k/pless@T sweeps (~1 hr CPU)
- **D3** + **D2** — update verdict docs while everything is fresh (~1.5 hr)
- **A3** — NAUADC on Qwen3-NoThink for final piece of decisive-test puzzle ($30, ~3 hr API)

---

## Version history

Format: `YYYY-MM-DD — { ADDED / STATUS-CHANGE / DROPPED / OTHER }: <one-line description>. Context: <commit or discussion link / brief why>.`

- **2026-05-21** — **CREATED**: initial wholistic TODO list. Captures everything discussed across the multi-day session. Context: user asked for a holistic list to ensure we look at things wholistically; agreed to maintain this as a living document going forward with version history.
- **2026-05-21** — **STATUS**: Phase 0.6 (B6 partial) DONE. Top-p and top-k brute-force checks complete; both empirically satisfy majorization-monotonicity. Context: commit `b3f8fd7`. Paper positioning shifted from "uniquely principled" to "first proved member of conjectured class".
- **2026-05-21** — **STATUS**: Sampler comparison (C5) DONE. α arms are Pareto-equivalent (and on MBPP slightly Pareto-dominated) by tuned non-α samplers. Context: commit `6f9ad44`. Verdict-relevant for any "α arms are unique" claim.
- **2026-05-21** — **STATUS**: Decisive test (B5, A11) DONE. Within-model Qwen3 thinking-on/off comparison: Mechanism B (thinking) wins. Context: commit `b9d30e8`.
- **2026-05-21** — **STATUS**: Diversity-theorem plan (D5) DONE. Documents the Phase-0 (literature) + Phase-0.5 (brute-force) work; ready for Phase 1 proof attempt. Context: commits `7e0c174`, `b3f8fd7`.
- **2026-05-19** — Multi-day session began with: C7 v3 β-binomial framework (B4) was the active theory work; APPS sweep (A1) was the planned compute item.
