# Table inventory — LLM4Code paper

Extractor: `scripts/paper/make_llm4code_tables.py` (run with `uv run python`).
Output: `paper/llm4code/tables/*.md`, one file per table. Regenerate on every metrics update.

## Tables the paper will show

| # | File | Section | Source dirs | Status |
|---|---|---|---|---|
| 3A | `tables/table_3a_mbpp.md` | §3 | `results/pless_full_mbpp_results/<model>/metrics/` + `results/pless_mbpp_results/…` | ✓ generated |
| 3B | `tables/table_3b_humaneval.md` | §3 | `results/pless_human_eval_results/{full_precision_results,temprature_results}/<model>/metrics/` | ✓ generated (OCI empty — see gaps) |
| 3C | `tables/table_3c_small_models_2507.md` | §3 sub-callout | `results/pless_full_mbpp_results/analysis/small_models_2507/comparison_report.md` | ⚠ **already-computed report exists — pull relevant rows or reference directly** |
| 4 | `tables/table_4_reasoning_truncation.md` | §4 | Same APPS-252 JSONs as Table 5A, plus truncation-rate extraction from raw JSONL | ⚠ **not yet generated** — need a truncation-rate extractor that reads the JSONL, not just the metrics JSON |
| 5A | `tables/table_5a_apps_renyi.md` | §5 | `results/_renyi_sweep_full252/…` + baselines | ✓ generated |

## Key numerical observations from the auto-generated tables

These land as claims in the paper prose. All values pulled from the tables — never re-typed.

### Section 3 (MBPP + HE competitiveness)

- **p-less at T=0.7 is near-deterministic** on the instruct models. On CodeLlama-7b-Instruct-hf, MBPP p-less@0.7 gives pass@1 = 42.16% and pass@10 = 43.00% (Δ = 0.84pp), vs. top-p@0.9 pass@1 = 38.26% / pass@10 = 59.00% (Δ = 20.74pp). p-less wins pass@1 but effectively loses the whole pass@10 axis.
- On CodeLlama-7b-hf (base) the same pattern is milder: p-less@0.7 41.04→51.20 (Δ 10pp); top-p@0.9 34.56→68.20 (Δ 34pp).
- **cb_div is near zero for p-less** on the instruct model (0.03 vs 0.37 for top-p) — the diversity story on instruct models under default p-less is weak.
- **Implication for framing**: the paper CANNOT claim "p-less matches temp-based samplers on diversity" — the numbers do not support it. Correct framing: "p-less matches greedy on pass@1 without a temperature knob; it does not compete with top-p on pass@10 diversity at α=2." This motivates Section 5's G_k generalization as the diversity-unlocking fix.

### Section 5 (G_k Rényi on APPS-252)

- Qwen3-8B: α=2 baseline pass@1 = **62.54%**, pass@10 = 82.54%. Best at **k=0.1: pass@1 = 71.90%, pass@10 = 84.52%** (+9.36pp pass@1). k=0.05 comes very close (71.67%). Monotone in k: pass@1 falls back to 62.66% at k=1.6.
- DeepSeek-R1-Distill-Llama-8B: α=2 baseline pass@1 = **39.25%**, pass@10 = 62.70%. Best at **k=0.35: pass@1 = 48.45%, pass@10 = 74.21%** (+9.20pp pass@1, +11.51pp pass@10).
- Both numbers reconcile with the source docs (`docs/decoder_comparison_cot_apps_qwen3.md`, `docs/decoder_comparison_cot_apps_deepseek.md`) to within rounding.
- **cb_div for APPS-252** is reported via the `self_codebleu` per-task cache (top-level `codebleu_diversity` is null; the headline_table fallback aggregates from per-task). `struct_div` was dropped from the paper entirely (2026-10-06) — see SKELETON.md for reasoning.

## Gaps identified while running the extractor

1. **Qwen2.5-Coder-7B-Instruct MBPP**: only `pless_t1.0` and `pless_norm_t1.0` present under `results/pless_mbpp_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/`. **No baselines**. The MBPP table for this model has 2 rows only. Either (a) run the missing baselines, or (b) drop this model from the MBPP table (keep for HE table).
2. **OpenCodeInterpreter-DS-1.3B HumanEval**: no vanilla p-less HE runs on disk (only α-arm data under `results/pless_alpha_full_humaneval/`). Table 3B for this model is empty. Either (a) run vanilla p-less HE for OCI, or (b) drop from HE table.
3. **APPS-252 diversity metrics**: top-level `structural_diversity` and `codebleu_diversity` are null in the reasoning-model JSONs because APPS runs went through with `--skip-diversity`. A separate pass computed per-task `self_codebleu` (used via the headline_table fallback to populate the cb_div column). `struct_div` was dropped from the paper (2026-10-06) rather than back-filled — no residual gap.
4. **Truncation rates**: Section 4 headline table needs a truncation-rate extractor that reads the raw JSONL (or a purpose-built cot_efficiency report) — not currently emitted by `headline_table`. The numbers ARE in `docs/decoder_comparison_cot_apps_*.md` (Qwen 14.5%, DeepSeek 41.8%) but the underlying JSONL is authoritative.

## Notes on the extractor

- Small script (~150 lines), one config dict per benchmark. When a JSON path is missing, the row is dropped and a `GAP: …` line is emitted to stderr. This keeps the tables honest: what's shown is what's on disk.
- To add a new model/config, edit the MBPP / HUMANEVAL / APPS_RENYI dicts in `scripts/paper/make_llm4code_tables.py`. No other code change needed.
- The extractor delegates all formatting to `bench.eval.headline_table.headline_table(...)` — canonical 8-column format, locked by `tests/test_headline_table.py`.
