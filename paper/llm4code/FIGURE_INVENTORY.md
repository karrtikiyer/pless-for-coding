# Figure inventory — LLM4Code paper

Existing PNGs surveyed 2026-09-22. Reuse first; regenerate only if the figure needs re-styling to a template.

## Section 3 (MBPP + HE competitiveness) — reuse candidates

| Paper fig | Existing PNG | Notes |
|---|---|---|
| **Fig 3A — Pareto (pass@10 × diversity)** | `results/pless_full_mbpp_results/analysis/figures/pareto_correctness_diversity.png` | MBPP only; check that shortlist models are present |
| **Fig 3B — Pareto with 2507 overlay** | `results/pless_full_mbpp_results/analysis/figures/pareto_bigcode_2507_03160.png` | Already includes Hasan et al. baseline overlay for the 3 small models |
| **Fig 3C — Pareto (pass@1 × CodeBLEU) with sampler families** | `results/pless_full_mbpp_results/analysis/figures/pareto_zhu_2402_06925.png` | Filename is misleading — **verified 2026-09-22 by opening the PNG**: it plots ONLY our own data (samplers by color × models by shape × pass@1 × CodeBLEU on MBPP), no cross-paper overlay. Safe for main body. **Rename before use** (e.g. `pareto_sampler_diversity_mbpp.png`). Observation: p-less clusters at high pass@1 / low CodeBLEU; top-p/top-k cluster at low pass@1 / high CodeBLEU — neither dominates, both on the Pareto front. Motivates the G_k Rényi story in §5. |
| Supplementary: `diversity_metrics_comparison.png`, `pass_at_1_comparison.png`, `metrics_overview.png` | analysis/figures/ | Available; may or may not fit a 6-8 pg workshop budget. (`structural_diversity_bars.png` dropped 2026-10-06 along with the struct_div metric — see SKELETON.md.) |
| HumanEval Pareto | `results/pless_alpha_full_humaneval/diversity_passk_scatter_humaneval.jpg` | α-arm dataset; may need re-render for the vanilla-p-less variant story |

## Section 4 (default p-less collapse on reasoning models) — reuse candidates

| Paper fig | Existing PNG | Notes |
|---|---|---|
| **Fig 4A — loop-collapse layer-wise cosine/L₂ across cycles (DeepSeek)** | `results/loop_collapse_replication/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/figures/fig4_aggregate.png` | Direct replication of Circular-Reasoning Fig 4 on our data |
| **Fig 4B — entropy/prob at loop onset (DeepSeek)** | `results/loop_collapse_replication/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/figures/fig3b_prob_entropy.png` | Complements 4A with the entropy trace |
| Fig 4C (optional) — same for Qwen3-8B | `results/loop_collapse_replication/Qwen--Qwen3-8B/figures/fig4_aggregate.png` | Qwen3 does NOT reproduce as cleanly (see `docs/loop_collapse_internal_state_findings.md`). Consider using as a contrast: "DeepSeek matches the paper, Qwen3 exhibits weaker cycle deepening — the failure mode is family-specific" |
| **Fig 4D — truncation-rate bar chart across decoders** | ⚠ **needs generation** | Bars for pless / pless_norm / temp / top_p / top_k on Qwen3 + DeepSeek APPS-252. Numbers exist in `docs/decoder_comparison_cot_apps_{qwen3,deepseek}.md`; no PNG on disk. Add to `scripts/paper/make_llm4code_tables.py` as a second output |

## Section 5 (G_k Rényi rescue) — needs generation

| Paper fig | Existing PNG | Notes |
|---|---|---|
| **Fig 5A — pass@1 and pass@10 vs k, log-x** | ⚠ **needs generation** | Two panels (Qwen3-8B, DeepSeek); x = k on log scale; y = pass@k. Baseline (α=2) shown as a horizontal dashed line. Data from `results/_renyi_sweep_full252/…` |
| **Fig 5B — truncation-rate vs k** | ⚠ **needs generation** | Companion to 5A; shows the mechanism (lower k → lower truncation) |

## Regeneration script — TBD

Add `scripts/paper/make_llm4code_figures.py` (a companion to `make_llm4code_tables.py`) once the paper LaTeX template is chosen and we know export DPI / column width. Blocked on CFP posting.

Figures 5A / 5B / 4D use the same JSONs as Tables 5A / 4 — the script that generates the truncation table will also emit the Fig 4D / 5B data, and matplotlib per figure.

## Notes on styling

- Do NOT touch chart colors until the CFP + template is known. Once known, invoke the `dataviz` skill for the palette.
- If we regenerate: matplotlib figures at fixed size (single-column vs double-column depends on template), colorblind-safe palette, monospace legend labels.
