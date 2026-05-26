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
| A12 | DONE | NAUADC for Qwen3-NoThink × HumanEval (closed the 8-cell gap) | — | Closed 2026-05-22 via task #73. Actual cost $13.06 (under $15-20 estimate); judging finished in ~50 min wallclock (much faster than 3-5 hr estimate, probably due to API queue conditions). All 8 cells now have NAUADC in the writeup. |
| A13 | DONE | GSM8K α-sweep on Qwen2.5-Coder-7B-Instruct (4 α arms, 500 problems × 10 samples) | ~30 GPU-min | Cross-domain test of α-knob mechanism on math reasoning. **Verdict A: monotonic α↑ → pass@10↑ matches code.** pass@10: 0.908→0.950 (+4.2pp), pass@1 mild dip (-1.7pp), self_bleu_div doubles (0.32→0.65). Same trade-off sign as MBPP+HE on the same model. Verdict + cross-domain table at `results/pless_alpha_full_gsm8k/Qwen--Qwen2.5-Coder-7B-Instruct/analysis/gsm8k_alpha_sweep.md`. |
| A14 | PROPOSED | Entropy probe Option C — per-α pless_alpha trajectories on 6 cells (2 models × 3 datasets × endpoints α=2.0, 5.0) | ~18-22 GPU-hr / 4 GPU | Wired up 2026-05-23 (`run_entropy_probe_pless_alpha.sh`, 19 entropy probe tests pass). Smoke test pending. Status **downgraded ACTIVE → PROPOSED 2026-05-23**: A17-A19 path (lit-aligned classifier on existing data, no new GPU) is cheaper and more directly answers the mechanism question. A14 is justified ONLY if A17-A19 produce ambiguous results. See `docs/theory/entropy_mechanism_framework.md` §6 for the decision logic. Middle α arms (2.5, 3.0) further deferred even if A14 fires. |
| A15 | ACTIVE | SQuAD v1.1 α-sweep — falsification test for "bimodal entropy → α-effect" hypothesis (H1 vs H2) | ~10-15 GPU-min × 4 arms = ~1 GPU-hr / single GPU | Wired up 2026-05-23. `bench/squad/` module + `run_pless_alpha_squad.sh`. **55 unit tests pass**. Cross-checked: `normalize_answer` + `f1_score` verbatim against official SQuAD v1.1 eval script (worksheets.codalab.org fetch). Pass@k uses `human_eval.estimate_pass_at_k` (Chen 2021), same as code+math. **Critical eval design**: dual-track EM reporting — `pass_at_k` (preamble-stripped; primary; fair α-comparison) and `pass_at_k_raw` (no stripping; comparable to official SQuAD numbers; audit signal). Preamble stripping needed because instruct models emit "The answer is X" ~10-20% despite explicit "no preamble" system instruction. Default scope: Qwen2.5-Coder-7B-Instruct, 4 α arms (2.0/2.5/3.0/5.0), 500 problems × 10 samples = 20,000 generations. Output: `results/pless_alpha_full_squad/<model>/`. **GATED on A16 smoke**: dataset chosen for methodological cleanliness, NOT verified empirical difficulty for 7B-Instruct scale — published pass@k/EM numbers for any 7-13B Instruct model on SQuAD v1.1 not found in 3+ targeted searches. |
| A16 | DEFERRED | SQuAD smoke + difficulty triage (50 problems × 1 sample) | ~5 GPU-min | Pre-flight smoke for A15. Triage table: pass@1 in [0.4, 0.85] → green-light full A15; pass@1 > 0.9 → ceiling, try DROP (EM=34.2 for Llama-70B per arXiv:2504.11972) or HotpotQA (EM=54.0 same source); pass@1 < 0.4 → switch model (Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct non-Coder). **Deferred 2026-05-23**: per A17-A19, the lit-aligned path forward (forking-token analysis on existing data) is cheaper and more directly tests the mechanism. Revisit A16 only if A17-A19 produce ambiguous results. |
| A17 | ACTIVE | Implement task-type-aware token classifier in `bench/entropy_probe/analysis.py` | ~2 hr code | Add `classify_tokens_cot(top_pct=0.20)` per 80/20 paper (arXiv:DAPO; rank-based) and `classify_tokens_code(method="q3")` per SWEET (arXiv:2305.15060; corpus-percentile-based). Test cases per `docs/theory/entropy_mechanism_framework.md` §6. Rationale: our prior dip-test + GMM methodology answered "is the distribution bimodal" (yes) but not the now-refined question "does the α-knob exploit the forking-token positions specifically?" The lit-aligned classifier methodology is more focused and grounds against AdaDec + SWEET for code, 80/20 + arXiv:2603.18940 for CoT. |
| A18 | ACTIVE | Apply CoT-side classifier to existing GSM8K α-sweep data (Option 3) | ~30 GPU-min teacher-force + ~1 hr analysis | Uses existing `results/pless_alpha_full_gsm8k/Qwen--Qwen2.5-Coder-7B-Instruct/pless_alpha_a{2.0,2.5,3.0,5.0}_t1.0.jsonl`. Teacher-force per-token entropy on completions, apply `classify_tokens_cot(top_pct=0.20)`, count forking-token visits per α arm, compare distributions. Key question: does α↑ → more forking-token positions visited per problem? If yes: strong mechanism evidence (the α-knob specifically exploits the 80/20-paper's forking tokens). Cost is dominated by teacher-forcing (model forward passes); we already have the trajectories, no new generation. |
| A19 | ACTIVE | Apply code-side classifier to existing MBPP entropy probe data | ~1 hr analysis (no GPU — per-token entropies already in CSV) | Uses existing `results/entropy_probe/Qwen--Qwen2.5-Coder-7B-Instruct/mbpp/per_token_entropy.csv` from the 2026-05-22 cross-domain entropy probe. Apply `classify_tokens_code(method="q3")` (SWEET's HE Q3=0.68 nats threshold is reference; compute our own Q3 on this corpus). Cross-check with our existing dip test result (verified bimodal). Compare with code-side α-sweep results to characterize code-side forking-token structure. **Code-side note**: this data was generated with multinomial T=1.0 sampler (not pless_alpha) — confirms code generation has bimodal structure but doesn't yet test α-arm differences in code. That's A14 (Option C) territory. |
| A20 | DONE | GSM8K α-sweep × 4 models cross-validation | ~30 GPU-min generation (already done) + ~15 min eval | **DONE 2026-05-23**. Generated and evaluated GSM8K α-sweep on 3 new models (Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct) joining the original Qwen2.5-Coder run. **Verdict: 4-of-4 monotonic α↑ → pass@10↑** (+4.0 to +5.8 pp), pass@1 dips uniformly (-1.7 to -6.6 pp, scales with weakness), self-BLEU diversity ~2× across all 4. Mistral non-monotonic at α=5 (peak at α=3). Cross-model robustness across 3 model families (Qwen, Llama, Mistral) and 31.5 pp baseline range (Mistral 0.55 → Qwen2.5-7B 0.86). Verdict + tables in `results/pless_alpha_full_gsm8k/cross_model_summary.md`. |
| A21 | PROPOSED | β-binomial / ν(α) fit on the 4 GSM8K models | ~1 hr code + analysis | Extend `bench/eval/validate_pass_at_k_beta_binomial.py` cell-discovery to GSM8K paths. Core math (`fit_beta_binomial_mom`, `pass_at_k_beta`) is dataset-agnostic — verified earlier. Test whether the C7 v3 ν(α) regularity (across code models on MBPP+HE) extends to math reasoning across 4 models. If yes: theory contribution generalizes. If no: dichotomy may be code-specific. Estimated mechanical work; no new GPU time. |
| A22 | PROPOSED | Code-side cross-model expansion — MBPP/HE α-sweep on Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B-Instruct | ~10-15 GPU-hr | Currently we have MBPP+HE α-sweep on 4 code-leaning models (Qwen2.5-Coder, CodeLlama, m-a-p OCI, Qwen3-8B-NoThink). Adding the 3 new GSM8K models would give a clean **4×3 cross-section** (each of 4 instruct models × 3 datasets {MBPP, HE, GSM8K}). Gated on A20 outcome (now positive) + decision about whether the paper claim needs full cross-section or current evidence suffices. |

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
| C7 | PROPOSED | Intersection-restricted NAUADC across α arms (apples-to-apples problem set) | ~1 hr code + analysis | Current per-bucket NAUADC is computed on the correct-only subset per arm, so the α=2 (n=52) and α=5 (n=80) numbers in the Qwen2.5-Coder APPS report (CODEFORCES_introductory) are not over the same problem set. α=5 solved 31 problems α=2 didn't; intersection across all 4 arms = 47 problems. Hypothesis: α=5's diversity edge inflates because the "extra" harder problems α=5 alone solves have more intrinsic algorithmic variety. Fix: re-aggregate NAUADC on the per-bucket intersection-of-correct-only problem sets. Affects all 4 buckets in the Qwen2.5-Coder APPS analysis (and any future α-arm NAUADC). Note: the source paper (arXiv:2503.00691 Table 2) has the same confounder across models and doesn't address it — we'd be one step stricter. Code: add per-arm task-id filter pass to `bench/eval/algosim_report_apps.py::_aggregate`; report intersection-NAUADC as a sibling column to the current per-arm NAUADC. |

## D. Writing / paper

| # | Status | Item | Estimate | Notes / artifacts |
|---:|---|---|---|---|
| D1 | PROPOSED | Workshop paper outline / draft | 1-2 days | Offered earlier; never written. Needs revised positioning per Step 0.6 (α not uniquely principled). |
| D6 | ACTIVE | Decoding-methods literature landscape doc | 1 day | Sub-agent + verification. Survey temperature, top-k, top-p, min-p, eta, locally-typical, mirostat, contrastive search/decoding, p-less + B.5; verbatim formulas + primary citations. Output: `docs/research/decoding_methods_landscape.md`. Driven by 2026-05-21 discussion on how to position our Σpᵢ^α form. |
| D7 | OPEN | Decide framing: "α-frequency-moment thresholding" vs "Rényi-α generalization" | — | Discovered 2026-05-21: our `Σpᵢ^α` ≠ paper B.5 rooted form `(Σpᵢ^α)^(1/(α-1))`; monotone in opposite directions. Need to commit to a name + positioning before D1 paper draft. |
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

- **2026-05-26** — **ADDED C7**: Intersection-restricted NAUADC across α arms (apples-to-apples problem set). Discovered while interpreting the just-completed Qwen2.5-Coder APPS NAUADC report: n_problems differs across α arms within each bucket (CODEFORCES_introductory: α=2 n=52, α=2.5 n=75, α=3 n=83, α=5 n=80) because the correct-only export filter retains only problems with ≥1 correct sample, and pass@10 varies with α (0.174 → 0.278 → 0.268 across α=2 → 3 → 5). Verified by exact match between metrics-JSON pass-result counts and per-cell n_problems. Overlap analysis: α=5 solved 31 problems α=2 didn't; intersection across all 4 arms = 47 problems; union = 98 problems. Hypothesis: α=5's NAUADC edge inflates because the 31 "extra" problems α=5 alone solves are likely harder and have more intrinsic algorithmic variety. The clean fix is to re-aggregate NAUADC on the per-bucket intersection-of-correct-only problem sets — same problems across arms, only sampling distribution varies. Documented as PROPOSED (not blocking, paper has same confounder without addressing). User deferred ("address later"). Context: NAUADC report regenerated this session with `--label` flag on `algosim_report_apps.py` (also patched to support nested layouts + multi-underscore entities); commit `bc527d6`.
- **2026-05-21** — **CREATED**: initial wholistic TODO list. Captures everything discussed across the multi-day session. Context: user asked for a holistic list to ensure we look at things wholistically; agreed to maintain this as a living document going forward with version history.
- **2026-05-21** — **STATUS**: Phase 0.6 (B6 partial) DONE. Top-p and top-k brute-force checks complete; both empirically satisfy majorization-monotonicity. Context: commit `b3f8fd7`. Paper positioning shifted from "uniquely principled" to "first proved member of conjectured class".
- **2026-05-21** — **STATUS**: Sampler comparison (C5) DONE. α arms are Pareto-equivalent (and on MBPP slightly Pareto-dominated) by tuned non-α samplers. Context: commit `6f9ad44`. Verdict-relevant for any "α arms are unique" claim.
- **2026-05-21** — **STATUS**: Decisive test (B5, A11) DONE. Within-model Qwen3 thinking-on/off comparison: Mechanism B (thinking) wins. Context: commit `b9d30e8`.
- **2026-05-21** — **STATUS**: Diversity-theorem plan (D5) DONE. Documents the Phase-0 (literature) + Phase-0.5 (brute-force) work; ready for Phase 1 proof attempt. Context: commits `7e0c174`, `b3f8fd7`.
- **2026-05-21** — **DISCOVERED + ADDED**: Our `Σpᵢ^α` is NOT the paper's B.5 Rényi-α form `(Σpᵢ^α)^(1/(α-1))` — opposite monotonicity in α. Added D6 (literature landscape) and D7 (framing decision); renamed research-group writeup title from "Rényi-α p-less sweep" → "α-collision threshold sweep". Context: user pasted B.5 verbatim; derivation showed our form coincides with B.5 only at α=2.
- **2026-05-23** — **DONE A20**: GSM8K α-sweep × 4 models cross-validation complete. Generated + evaluated on Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct (joining existing Qwen2.5-Coder). **4-of-4 models show monotonic α↑ → pass@10↑ direction**: +4.0 to +5.8 pp pass@10 gain, -1.7 to -6.6 pp pass@1 dip, ~2× self-BLEU diversity increase. Mistral peaks at α=3 then drops 1.8pp at α=5 (single non-monotonicity). Span of model families (Qwen, Llama, Mistral) + 31.5 pp baseline range (Mistral 0.55 → Qwen2.5-7B 0.86). Verdict in `results/pless_alpha_full_gsm8k/cross_model_summary.md`. **Substantially strengthens cross-model robustness claim**: α-effect is not a Qwen2.5-Coder-specific artifact, it's a property of bimodal-entropy CoT generation across instruct model families. Added A21 (β-binomial extension to 4 GSM8K models) + A22 (code-side cross-model expansion to complete 4×3 cross-section). A18 (forking-token classifier on GSM8K data) now has a 4-model dataset to work with — including the interesting Mistral outlier.
- **2026-05-23** — **RETRACTED + ADDED + FRAMEWORK**: Major reframing of the cross-domain mechanism work. User pushed back hard on multiple unverified claims with the rigor stop phrase. Three rounds of retraction and verification produced a now-grounded framework documented in `docs/theory/entropy_mechanism_framework.md`. **Key retractions**: (a) "Multiple instruct models report WMT" — FALSE; Llama-3, Qwen2.5, Gemma all confirmed NOT to evaluate on WMT in their tech reports (direct fetch). (b) "GSM8K-no-CoT ablation is the strongest mechanism test" — baseline collapses to ~15% (arXiv:2505.23701 verified), too noisy. (c) "80/20 CoT-entropy lit grounds our MBPP+HE findings" — conflation; CoT lit (80/20, arXiv:2603.18940) and code-gen lit (AdaDec arXiv:2506.08980, SWEET arXiv:2305.15060) are SEPARATE bodies that use different threshold methodologies for principled reasons. **Key verified findings**: (a) Code generation has bimodal-like entropy structure per AdaDec (drift median=1.26 nats, non-drift median=0.03 nats) and SWEET (HE Q3=0.68 nats). (b) Lit-aligned methodology is task-type-specific: top-20% rank cutoff for CoT (80/20 paper), corpus Q3 percentile for code (SWEET). (c) Our methodology (dip test + GMM) verified bimodality (useful when novel) but is now redundant with lit; missing is the forking-token classification + α-arm comparison that's the actual paper contribution. **Tasks added**: A17 (implement task-type-aware classifier), A18 (apply CoT classifier to existing GSM8K data — cheap, novel), A19 (apply code classifier to existing MBPP probe data — free). **A16 deferred** in favor of A17-A19 path. **Prompt-format gap surfaced**: our MBPP+HE use direct prompting (no CoT), GSM8K uses Wei 2022 8-shot CoT — so MBPP+HE findings cite code-gen lit, GSM8K cites CoT lit. Different citations per task type. Detailed framework + lit grounding + path forward in `docs/theory/entropy_mechanism_framework.md`.
- **2026-05-23** — **RETRACTED + ADDED**: A16 added as pre-flight smoke + difficulty triage for A15. Two unverified claims retracted from prior session text: (a) "SQuAD was designed for span-extraction models" — overstated; the dataset is architecture-agnostic per the original paper (verified via arXiv:1606.05250 abstract). (b) "Likely saturation at 7B+ instruct scale on SQuAD v1.1" — speculation, not verified; 3+ targeted searches turned up zero published pass@k/EM numbers for any 7-13B Instruct model on v1.1. Context: user invoked rigor stop phrase ("are you sure? Have evidence?") on both claims. Honest answer to "why did we pick v1.1": methodological cleanliness (clean pass@k semantics, all answerable, official EM/F1 verified) — NOT verified empirical difficulty for our model scale. A16 smoke is the empirical check that should have preceded full A15 wiring.
- **2026-05-23** — **ADDED**: A15 (SQuAD v1.1 α-sweep) — falsification test for the "bimodal entropy → α-effect" mechanism hypothesis. Triggered by literature search verifying: (a) pass@k applies cleanly to short-answer QA via EM signal (arXiv:2502.11027 uses EM@k in parallel with Pass@k); (b) no paper has characterized per-token entropy on SQuAD/NQ/TriviaQA — genuine gap. Hypothesis: extractive QA has different entropy structure than CoT-heavy code+math (most tokens copied from passage, few decision points). If pass@10 grows monotonically with α on SQuAD → H1 falsified, H2 ("α works on any verifiable multi-solution task") stands. If pass@10 flat/non-monotonic → H1 stands. `bench/squad/` module + `run_pless_alpha_squad.sh` wired and tested (55/55 unit tests, normalize_answer + f1_score verified verbatim against official SQuAD v1.1 eval script, pass@k uses Chen 2021 unbiased estimator). **Eval design choice**: dual-track EM (preamble-stripped primary + raw audit) — addresses instruct-model preamble noise without breaking comparability with published SQuAD numbers. Context: user asked "what domain do we test on to prove that this works for some domains but not transfer to other domains" — pushed back on MT/CommonGen recommendation as memory-from-not-verified; web-searched and verified short-answer QA with EM is the cleanest pass@k-compatible candidate.
- **2026-05-23** — **STATUS + ADDED**: GSM8K α-sweep on Qwen2.5-Coder DONE (A13). Cross-domain mechanism transfer confirmed — pass@10 grows monotonically (+4.2pp from α=2 → α=5), pass@1 dips mildly (-1.7pp), self-BLEU diversity doubles. Pattern matches MBPP+HE on the same model. Verdict + 3-dataset comparison at `results/pless_alpha_full_gsm8k/Qwen--Qwen2.5-Coder-7B-Instruct/analysis/gsm8k_alpha_sweep.md`. Added A14 (entropy probe Option C, ACTIVE) — wired up earlier today (commits pending), launch gated on this verdict, now downgraded from diagnostic→confirmatory and recommended as confirmatory not urgent. Smoke test pending on GPU. Context: user asked "plan carefully what we should do next" after GSM8K generations landed; chose quick-verdict path, gated Option C on result.
- **2026-05-22** — **STATUS**: Tasks #69 (NAUADC HE 3 instruct models) + #70 (NAUADC MBPP Qwen3-NoThink) DONE. Generated per-model reports via `bench.eval.algosim_report` against existing response parquets. Reports + plots committed under `results/pless_alpha_full_humaneval/<model>/humaneval/analysis/algosim_*_alpha_he_claude.*` and `results/pless_alpha_full_mbpp/Qwen--Qwen3-8B/no-think/analysis/algosim_*_alpha_claude.*`. Research-group writeup regenerated with NAUADC column added (7 of 8 cells; Qwen3-NoThink HE NAUADC was never queued — new TODO row A12 tracks the gap). Context: yesterday's session left judging done but reports/plots not generated. Verified all 4 cells show monotonic α↑ → NAUADC↑ pattern except CodeLlama HE which is non-monotonic but with α=5 still highest.
- **2026-05-21** — **ADDED + STATUS**: Research-group writeup (`docs/research_group_writeup_2026-05-21.md`) DONE. 8 tables (pass@1/3/5/10 + codebleu_div) + 16 plots across 4 model configs (Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct, m-a-p OCI-1.3B, Qwen3-8B-NoThink) × 2 datasets (MBPP, HumanEval) × 4 α arms {2.0, 2.5, 3.0, 5.0}. Generator: `bench/eval/research_group_writeup.py`. CodeBLEU only — NAUADC HE for 3 instruct models + NAUADC MBPP for Qwen3-NoThink running in background (tasks #69, #70). `bench/eval/algosim_export.py` patched to accept HumanEval string task_ids. Context: user asked for research-group presentation today; Qwen3-NoThink added per follow-up request "to this: research_group_writeup, also add Qwen38B non thinking results".
- **2026-05-19** — Multi-day session began with: C7 v3 β-binomial framework (B4) was the active theory work; APPS sweep (A1) was the planned compute item.
