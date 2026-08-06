# Rényi G_k sweep on APPS — interim results (Paper B)

**Status: COMPLETE — 12 of 12 arms scored (as of 2026-08-06).** Full k grid
{1.6, 0.8, 0.4, 0.2, 0.1, 0.05} × {Qwen3-8B, DeepSeek} on 252 APPS-interview problems.
Next: folds into Paper B via `scripts/build_decoder_comparison_table.py` (merge the G_k arms
into the `qwen` / `deepseek_fixed` SETs alongside τ_α). Numbers here are the source of truth
until that table is built.

## What this is

The origin p-less paper (arXiv:2509.23234, App. B.5) proposes a *rooted* Rényi threshold
`G_k = (Σpᵢ^k)^{1/(k−1)} = exp(−H_k)` but runs **zero** experiments at k ≠ 2. This is our
empirical sweep of that form (`--method pless_renyi`), to compare against our power-sum
family `τ_α = Σpᵢ^α` — the two coincide only at order 2 and diverge otherwise (see
`docs/research/paperA_renyi_nonequivalence.md`). G_k **loosens as k decreases** below 2;
k = 2 is plain p-less (= G_2 = τ_2), not re-run.

Model set: `Qwen/Qwen3-8B` (QW), `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (DS).
Benchmark: APPS ATCODER-interview, 252 tasks × 10 samples.

## Results (partial)

Arms grouped by model, k descending (tight → loose):

| Arm | pass@1 | pass@10 | cb_div | mean think tok | non-term% |
|---|---|---|---|---|---|
| QW k1.6 | 0.627 | 0.813 | 0.4644 | 12,971 | 14.0% |
| QW k0.8 | 0.669 | 0.837 | 0.4606 | 11,788 | 8.0% |
| QW k0.4 | 0.696 | 0.833 | 0.4574 | 10,912 | 0.4% |
| QW k0.2 | 0.701 | 0.833 | 0.4924 | 11,062 | 0.0% |
| QW k0.1 | 0.719 | 0.845 | 0.4983 | 10,808 | 0.0% |
| QW k0.05 | 0.717 | 0.833 | 0.4971 | 10,767 | 0.1% |
| DS k1.6 | 0.400 | 0.643 | 0.4941 | 16,420 | 39.6% |
| DS k0.8 | 0.435 | 0.687 | 0.5175 | 12,925 | 25.8% |
| DS k0.4 | 0.469 | 0.730 | 0.5745 | 9,615 | 0.0% |
| DS k0.2 | 0.463 | 0.730 | 0.5829 | 9,879 | 0.0% |
| DS k0.1 | 0.463 | 0.714 | 0.5878 | 9,463 | 0.2% |
| DS k0.05 | 0.459 | 0.710 | 0.5864 | 9,457 | 0.0% |

**Pending:** none — all 12 arms scored.

### Reading

As k decreases (looser G_k filter): **non-term% collapses to ≈0%** — by k≤0.2 for Qwen
(14.0%→0.4% at k0.4→0.0% at k0.2) and by k≤0.4 for DeepSeek (39.6%→0.0%) — and **mean think
tokens drop** (DS 16,420→~9,500 as runaway loops stop burning the 32768 budget). (Non-term% is
mildly non-monotone at the ≈0 floor, e.g. QW k0.05 0.1% vs k0.1 0.04% — sampling noise, not a
trend.) The loop-escape mechanism Paper B documents for τ_α reproduces on the rooted Rényi form.

**pass@1 improves to a plateau, then slightly over-loosens — the plateau location is
model-dependent.** Qwen improves through a broad plateau around **k≈0.1–0.2** (0.696/0.701/0.719/0.717
at k0.4/0.2/0.1/0.05); DeepSeek plateaus **earlier, around k≈0.2–0.4** (0.469/0.463/0.463/0.459),
with pass@10 also easing after k0.4 (0.730→0.710). **Caveat:** adjacent loose arms differ by only
~1.5–6 problems out of 2520 — well inside the ±0.023 per-arm CIs, and arms are unpaired (no
across-arm significance test) — so read these as a *statistically indistinguishable plateau then a
slight dip*, not a pinpoint optimum. The qualitative split is robust: the more loop-prone model
(DeepSeek, 39.6% non-term at k1.6) plateaus earlier — once its loops are gone (non-term ≈0 by k0.4)
further loosening only injects noise. Practical guidance: **DeepSeek wants a milder loosening than
Qwen**; anything in k≈0.1–0.4 is near-optimal for both.

**Do not read cb_div as an accuracy/tradeoff signal.** cb_div is computed over each config's
*correct* samples only (the ≥2-correct subset), so it measures structural variety *among the
solutions that already pass* — not diversity of the output at large. Worse, the ≥2-correct
subset and the per-problem correct count both shift with k, so comparing cb_div *across* k
carries a composition confound (a smaller correct subset can look more spread out without any
real diversity change). It is reported for completeness and for the eventual τ_α comparison at
matched configs, but it does not support claims about the pass@1 trajectory.

## Provenance (every number pulled live, matching the alpha arms + Paper B)

- **pass@1/10** (and pass@3/5, cover@t, `per_task` pass results): `uv run python -m bench.eval
  --results-file <f> --dataset apps --workers 8 --skip-diversity`. `--skip-diversity` matches
  how all 6 τ_α arms were scored (`structural_diversity=null`; zss AST diversity is intractable
  on APPS-CoT and is omitted from Paper B). Metrics JSON written to each `metrics/` subdir,
  identical shape to the alpha arms (252 tasks, 10 samples/task, k=1,3,5,10, t=0.1,0.3,0.5,0.7).
- **cb_div** (self-CodeBLEU diversity): `bench.eval.metrics.add_self_codebleu` +
  `compute_self_codebleu_diversity` — 1 − mean pairwise CodeBLEU among each problem's *correct*
  samples (AST-deduped), averaged over problems with ≥2 correct; no execution. The exact path
  `scripts/build_decoder_comparison_table.py` uses for Paper B. (NOT computed by `bench.eval`;
  its `self_codebleu` field is always null.) **Caveat:** it is diversity *among passing
  solutions only*, and the ≥2-correct subset shifts with k — so cross-k cb_div comparisons are
  confounded and must not be used as an accuracy signal (see Reading).
- **non-term% (= trunc%)**: fraction of `samples_with_thinking` lacking `</think>`, from the
  jsonl directly — the table's own method.
- **mean think tok**: the table's `mean_tok_from_jsonl` (≈300-sample estimate; think phase =
  text before `</think>`; truncated samples counted at their cut length ≈ cap → biased UP for
  high-non-term arms — the rambling cost). DeepSeek carries a ~+3% byte-BPE re-tokenization
  inflation, but the DS τ_α arms re-tokenize identically, so it stays apple-to-apple.

## Apple-to-apple footing

**Verified from the jsonl records** (both models, baseline vs every G_k arm): identical
`temperature 1.0`, `top_p 1.0`, `top_k 0`, `backend vLLM`, `n=10`, `--enable-thinking`
(all carry `samples_with_thinking`), full 252 tasks — only `method` differs (pless vs pless_renyi).
**Shared token cap verified** by matched truncated-trace lengths (baseline vs arms cluster at the
same ceiling; QW re-tokenizes to exactly 32768). The specific value **32768** and
`VLLM_USE_FLASHINFER_SAMPLER=0` come from the run scripts, not the records (jsonl stores no
`max_new_tokens`) — the data proves *same cap*, and the number is the configured one.

**Residual assumption (not verifiable from the records):** DeepSeek's #45488 tokenizer fix is
auto-applied on the vLLM path, so both the `_deepseek_fixed` baseline and the G_k arms should have
it — but baseline and arms live in different result trees, so we cannot confirm from the jsonl that
both were generated post-fix. If the G_k arms predated the fix, G_k-vs-α=2 would confound with
fix-vs-no-fix (the same trap we avoid by not using `pless_recovery_full252/deepseek` as the baseline).
The clean sample content + zero all-pruned failures in the arms make a pre-fix run unlikely, but it
is an assumption, not a verification.

**τ_α comparison baselines** (the correct dirs to merge against):
- QW → `results/pless_recovery_full252/Qwen--Qwen3-8B/ATCODER_interview`
- DS → `results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview`
  (the post-#45488-fix rerun — NOT `pless_recovery_full252/deepseek`, which is pre-fix and would
  confound G_k-vs-τ_α with fix-vs-no-fix).

Results tree: `results/_renyi_sweep_full252/`.

## Next steps

1. Eval each remaining arm as it syncs in (same `bench.eval --skip-diversity` command).
2. Once all 12 arms are scored, add the renyi arms to the `qwen` and `deepseek_fixed` SETs in
   `scripts/build_decoder_comparison_table.py`, regenerate the per-model tables (G_k arms
   interleaved with τ_α arms), and overlay the G_k vs τ_α loosening curves.
3. Fold the finding into Paper B (§ on the τ_α-vs-G_k comparison) — reporting the *profile*
   difference (G_k loosens confident steps; τ_α loosens everywhere) as measured, not asserted.
