# Model Roster — LLM4Code Paper

Reality-check of on-disk data (verified 2026-09-22 via `ls`/`find`) — not from CSV, which is stale.

## Coverage matrix

| Model | MBPP-full | HumanEval | APPS-252 | Notes |
|---|---|---|---|---|
| **codellama/CodeLlama-7b-hf** (BASE) | ✓ full methods (`pless_full_mbpp_results/`) | ✓ pless-family (`pless_human_eval_results/temprature_results/`) | — | Clean full/full base model |
| **codellama/CodeLlama-7b-Instruct-hf** (INSTRUCT) | ✓ full methods (`pless_full_mbpp_results/`) | ✓ full-precision + temperature dirs | — | Clean full/full instruct |
| **m-a-p/OpenCodeInterpreter-DS-1.3B** (INSTRUCT) | ✓ full methods | ⚠ p-less-only (via `pless_alpha_a2.0` ≡ p-less baseline; no non-p-less HE baselines) | — | HE cell is thinnest; see `MISSING_EXPERIMENTS.md` — recommend dropping OCI from Table 3B rather than running |
| **Qwen/Qwen2.5-Coder-7B-Instruct** (INSTRUCT) | ✓ full (greedy + 3 temps + top-p@0.95 + p-less family) in `full_mbpp_pre_post_temp_pless/` — missing beam/top-k/top-p@0.9 only | ✓ both HE dirs | — | MBPP is fine once extractor points at the right dir |
| **Qwen/Qwen2.5-Coder-1.5B** (BASE) | ✓ full methods | — | — | Small-model 2507 comparison model (MBPP only for now) |
| **Qwen/Qwen2.5-Coder-3B** (BASE) | ✓ full methods | — | — | Small-model 2507 comparison model (MBPP only for now) |
| meta-llama/Llama-2-7b-hf (BASE) | ✓ | — | — | MBPP-only |
| meta-llama/Llama-2-7b-chat-hf (INSTRUCT) | ✓ | — | — | MBPP-only |
| Qwen/Qwen-7B (BASE, legacy) | ✓ | — | — | MBPP-only, transformers<5 |
| Qwen/Qwen-7B-Chat (INSTRUCT, legacy) | ✓ | — | — | MBPP-only, transformers<5 |
| mistralai/Codestral-22B-v0.1 (INSTRUCT) | — | ✓ full-precision + temperature | — | HE-only |
| Qwen/Qwen2.5-Coder-7B (BASE) | — | ✓ temperature | — | HE-only |
| Qwen/Qwen3-Coder-30B-A3B-Instruct (INSTRUCT) | — | ✓ full-precision + temperature | — | HE-only |
| **Qwen/Qwen3-8B** (REASONING) | ✓ α-arm | ✓ α-arm (think + no-think) | ✓ Rényi sweep 9 k-values, τ_α baseline | Section 4/5 model |
| **deepseek-ai/DeepSeek-R1-Distill-Llama-8B** (REASONING) | — | — | ✓ Rényi sweep 12 k-values, τ_α baseline | Section 4/5 model |

## Shortlist decision — Section 3 (MBPP + HumanEval competitiveness)

**Locked as the paper's cross-benchmark model set (4 models, both MBPP + HE)**:

1. **CodeLlama-7b-hf** (BASE) — full/full
2. **CodeLlama-7b-Instruct-hf** (INSTRUCT) — full/full
3. **Qwen2.5-Coder-7B-Instruct** (INSTRUCT) — HE full, MBPP scattered (see missing-experiment audit)
4. **m-a-p/OpenCodeInterpreter-DS-1.3B** (INSTRUCT) — MBPP full, HE α-arm only

Ratio: 1 base + 3 instruct. **Base coverage on HE is limited to CodeLlama-7b-hf**. Handled by prose acknowledgment in §3 (Section 6 lists it as a limitation).

**Revised decision (2026-09-22)**: do NOT run additional experiments to backfill. Per-model coverage variance is standard in workshop papers; be transparent and move on. See `MISSING_EXPERIMENTS.md` for the full audit.

## Small-model callout — Section 3, sub-callout

- Qwen2.5-Coder-1.5B (BASE), Qwen2.5-Coder-3B (BASE), OCI-DS-1.3B (INSTRUCT) — MBPP-only, compared against Hasan et al. (arXiv:2507.03160) single-config baseline (top_p=0.95, T=0.2).
- Source of truth: `results/pless_full_mbpp_results/analysis/small_models_2507/comparison_report.md`
- Our p-less beats Hasan et al. baseline on all three (Q2.5-Coder-3B: 59.4 vs 57.0; Q2.5-Coder-1.5B: 53.1 vs 51.0; OCI: 44.9 vs 44.0).

## Reasoning-model set — Sections 4 + 5

- **Qwen3-8B** (thinking-ON) — APPS-252, ATCODER-interview split
- **DeepSeek-R1-Distill-Llama-8B** (thinking-ON) — APPS-252, ATCODER-interview split, tokenizer-fix asymmetry (footnote)

## Instruct-CoT control set — Section 4 (control panel only)

- Qwen2.5-Coder-7B-Instruct + Qwen2.5-Coder-3B-Instruct with induced-CoT prefill on APPS-252
- Source: `docs/cot_instruct_decoder_comparison_apps.md`, `results/pless_cot_efficiency_instruct/`

## Dropped from paper scope

- Llama-2-7b (base + chat), Qwen-7B (base + chat), Qwen2.5-7B — MBPP-only, no clear paper role given the merged narrative. Retain for completeness in the repo but do not surface.
- Codestral-22B, Qwen3-Coder-30B-A3B-Instruct — HE-only, MBPP not run. Could be added to Section 3 as HE-only supplement if column-sparse tables are acceptable in the template.
