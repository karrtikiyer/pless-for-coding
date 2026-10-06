# Missing-experiment audit — LLM4Code paper

**Bottom line (revised 2026-09-22 after user pushback + re-verification): we can write the paper WITHOUT running any new experiments.** Per-model coverage variance is standard in workshop papers; be transparent and footnote it. The GPU time is better spent on drafting.

Prior version of this doc overstated coverage gaps by looking at only one of several result directories. Below is the corrected, per-cell view.

## Actual per-cell baseline coverage (shortlist models)

Cells populated on disk with metric JSONs (verified via `ls` on all relevant dirs):

| Model × benchmark | greedy | temp | top-p | beam | top-k | p-less | p-less-norm | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| CodeLlama-7b-hf × MBPP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | complete |
| CodeLlama-7b-Instruct-hf × MBPP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | complete |
| OCI-DS-1.3B × MBPP | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | fine (no top-k) |
| Qwen2.5-Coder-7B-Instruct × MBPP | ✓ | ✓ (0.2/0.6/0.8/1.5) | ✓ (0.95) | — | — | ✓ | ✓ | acceptable — has temp + top-p; missing beam + top-k + top-p@0.9 |
| CodeLlama-7b-hf × HE | — | ✓ (0.7/1.0) | — | — | — | ✓ | ✓ | thin — pless-family vs. temp only; usable |
| CodeLlama-7b-Instruct-hf × HE | ✓ | ✓ (0.2/0.7) | ✓ (0.9/0.95) | — | — | ✓ | ✓ | fine |
| OCI-DS-1.3B × HE | — | — | — | — | — | ✓ (labeled `pless_alpha_a2.0`) | ✓ (via `pless_alpha_a2.0`, α-arm dir) | **thinnest cell — pless-family only, no baselines** |
| Qwen2.5-Coder-7B-Instruct × HE | ✓ | ✓ (0.2/0.7) | ✓ (0.9/0.95) | — | — | ✓ | ✓ | fine |

Notes:
- **p-less ≡ p-less-alpha at α=2** — both apply the `Σpᵢ² ≤ p` collision-entropy threshold. `results/pless_alpha_full_humaneval/m-a-p--OpenCodeInterpreter-DS-1.3B/humaneval/pless_alpha_a2.0_t1.0_metrics.json` IS the p-less baseline for OCI HE, just labeled by the α-family generator. Extractor should treat it as `p-less`.
- Qwen2.5-Coder-7B-Instruct MBPP baselines live in `results/full_mbpp_pre_post_temp_pless/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/`, NOT the more obvious `pless_mbpp_results/` dir. Extractor needs updating.

## What the paper actually needs

The paper's claim is "p-less is competitive with the standard decoding methods practitioners use." Each row above has at minimum temp + top-p to test against (except OCI HE — see below). That's enough.

**One remaining editorial decision on OCI × HE**: no non-p-less baselines. Options:
- (a) Add prose footnote: "HumanEval coverage for OCI-DS-1.3B is limited to the p-less family; head-to-head with baselines on this benchmark × model cell is future work."
- (b) Present OCI as MBPP-only in Section 3; drop the OCI HE cell from Table 3B.
- (c) Run temp + top-p on OCI HE (~30 min, 1 GPU). Only worth it if idle.

**Recommendation: (b)** — cleanest. OCI is a small 1.3B model; its main role in the paper is the Hasan et al. 2507 comparison on MBPP.

## Extractor fixes needed (immediate)

- Update `scripts/paper/make_llm4code_tables.py`:
  - Point Qwen2.5-Coder-7B-Instruct MBPP rows at `results/full_mbpp_pre_post_temp_pless/…/metrics/*.json` for the baselines.
  - Point OCI HE p-less at `results/pless_alpha_full_humaneval/m-a-p--OpenCodeInterpreter-DS-1.3B/humaneval/metrics/pless_alpha_a2.0_t1.0_metrics.json` (relabel to "p-less" in the table).

## Truncation-rate extractor for §4 (still applies)

`scripts/paper/make_llm4code_tables.py` currently reads only the metrics JSON, which does not include truncation counts. Section 4 needs a rate table for {p-less, p-less-norm, temp, top-p, top-k} × {Qwen3-8B, DeepSeek} on APPS-252. Extend the extractor to walk the JSONL and count samples where `</think>` never appears and generation hits `max_new_tokens`. ~1 hour of code.

## Genuinely out of scope

- mirostat / typical / η-sampling / min-p / contrastive / FSD / DoLa — cite Shi et al. 2402.06925 Table 26 in related work; do not compare cell-by-cell (plan decision).
- Full-baseline suite on missing (model, benchmark) cells — not needed if per-model coverage is footnoted.
