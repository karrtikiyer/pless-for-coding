# CoT Token-Efficiency vs Accuracy — deepseek-ai/DeepSeek-R1-Distill-Llama-8B / APPS

**Date:** 2026-06-05  
**Configs analyzed:** 6 (6 at the 24576-token budget used for the frontier)

Think length is measured in **tokens** (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`). Efficiency is decomposed per arXiv:2602.09805 into completion rate, conditional correctness (pass rate among completed samples), and think length.

## Column definitions

A *sample* is one generation; each problem has 10 samples. A sample is **completed** iff it has a closing `</think>` (the think phase finished, not truncated at the cap) AND code was extracted (for APPS, the executor's `extraction_success`); **truncated** iff it has no `</think>`.

- **budget** — generation cap (`--max-new-tokens`); all configs here share it.
- **compl%** (`completion_rate`) — % of samples that completed (`#completed / #samples`).
- **trunc%** (`truncation_rate`) — % of samples with no closing `</think>` (think ran into the token cap). Primary truncation signal is the missing `</think>`, not the token count.
- **cond-correctness** (`conditional_correctness`) — pass rate *among completed samples only*: `#(completed & correct) / #completed`. Answers "given it finished reasoning, did the code pass all hidden tests?" Equals `pass@1 / compl%`.
- **mean think tok** (`mean_think_tokens`) — mean think-block length over **ALL** samples, in tokens (think = text between `<think>` and `</think>`; for prompt-injected `<think>` models, text up to `</think>`). **Includes truncated samples**, which contribute their length-at-cut (≈ the cap) — so this is inflated toward the cap for configs that truncate, and NOT comparable across configs with different trunc%.
- **median (all)** (`median_think_tokens`) — median think length over **ALL** samples; the Pareto-frontier axis. Budget-insensitive (the truncated cap *value* doesn't move it) but **biased UPWARD by truncation rate**: truncated samples occupy the top ranks, so higher trunc% pushes the median to a higher percentile of the completed distribution. So a config sitting right may be there partly because it truncates more, not only because it reasons longer — read the trunc% (marker size) alongside. It is the least-misleading single length stat here (unlike median-done it won't falsely make a truncating config look short), but it is NOT fully decoupled from truncation.
- **median (done)** (`median_think_tokens_completed`) — median think length over **completed samples only**. Cap-robust but **censored**: a config's longest traces were truncated out, biasing its completed-median low. Don't read it as "who reasons shorter" across configs with different trunc%.
- **pass@1 / pass@10** — unbiased pass@k (human-eval estimator, `metrics.compute_pass_at_k`) over each problem's `(num_correct, n=10)`. pass@1 = overall fraction of single samples that pass; pass@10 (k=n=10) = fraction of problems solved by **≥1** of the 10 samples (coverage).
- **cov@0.3 / cov@0.5** (CSV) — % of problems with ≥30% / ≥50% of their samples correct (`num_correct ≥ t·n`).

**Coherence checks (should hold every run):**
- `pass@1 ≈ compl% × cond-correctness` per row, with residual = `#(passed but NOT completed) / n`. Passing requires extracted code that passes the tests, which *usually* implies a closed `</think>` — but a **truncated** trace can still contain a passing code block, and such samples count in pass@1 yet not in `completed`. So pass@1 can slightly EXCEED compl%×cond; the residual is audited per-config below (0 = identity holds exactly).
- `pass@10 ≥ pass@5 ≥ pass@1` per row (monotone in k).

**No single length stat is clean under differing truncation — each is biased a different way:** `mean think tok` = avg tokens *spent* (counts truncated at the cap → biased UP + budget-dependent); `median (all)` = typical length but biased UP by truncation rate (rank effect; budget-insensitive — the frontier axis, least-misleading); `median (done)` = typical *finished* length, biased DOWN (censored — drops the truncated long tail). For a truncating config: mean ≫ median(all) > median(done). So **read trunc% (marker size) alongside any length**; a clean cross-config length comparison needs a budget where all configs complete (no truncation).

## Per-config decomposition

| Config (think→code) | budget | compl% | trunc% | cond-correctness | mean think tok | median (all) | median (done) | pass@1 | pass@10 |
|---|---|---|---|---|---|---|---|---|---|
| temp 0.6 (unfiltered) | 24576 | 98.9 | 1.1 | 0.5450 | 5276 | 3991 | 3934 | 0.5390 | 0.8600 |
| temp 1.0 (top_k 20) | 24576 | 99.8 | 0.2 | 0.5331 | 5721 | 4790 | 4792 | 0.5340 | 0.8900 |
| temp 1.0 (top_p 0.95) | 24576 | 100.0 | 0 | 0.5240 | 5342 | 4296 | 4296 | 0.5240 | 0.8600 |
| temp 0.6 (top_p 0.95 + top_k 20) | 24576 | 95.4 | 4.6 | 0.5461 | 5832 | 3790 | 3577 | 0.5210 | 0.8000 |
| pless (t1.0) | 24576 | 63.0 | 37.0 | 0.6206 | 10376 | 4442 | 2232 | 0.3920 | 0.7400 |
| pless_norm (t1.0) | 24576 | 63.9 | 36.1 | 0.6103 | 10160 | 4088 | 2192 | 0.3910 | 0.7200 |

**Coherence audit** — configs where a passing sample was truncated (counted in pass@1 but not `completed`, so pass@1 > compl%×cond by that many samples / n): pless_norm (t1.0) (1), pless (t1.0) (1), temp 1.0 (top_k 20) (2). All other configs: residual 0 (identity exact).

## Pareto-dominant configs (24576-token budget)

Not dominated on (shorter median think tokens, pass@1 within 1pt); configs with >25% truncation excluded as context-limited failures. Length axis = `median (all)` — budget-insensitive but biased UP by truncation rate (so a config may rank longer partly because it truncates more); read trunc% alongside:

| Config | median (all) | trunc% | pass@1 | pass@10 | cond-correctness |
|---|---|---|---|---|---|
| temp 0.6 (top_p 0.95 + top_k 20) | 3790 | 4.6 | 0.5210 | 0.8000 | 0.5461 |
| temp 0.6 (unfiltered) | 3991 | 1.1 | 0.5390 | 0.8600 | 0.5450 |

## Limitations

- Single model / single difficulty; no cross-model generalization.
- Samplers compared are whatever was generated for this run; greedy is excluded by design (Qwen discourages it in thinking mode).
- Token counts are analysis-time estimates (tokenizer special-token handling may differ slightly from generation time).
- Truncation at the 24576 cap censors the upper tail: truncated samples are pinned near the cap (inflating the *mean* for configs that truncate) yet underestimate their true length. So length is NOT comparable across configs with different trunc% — see Column definitions.
- Stochastic samplers run at a fixed temperature, not matched effective entropy, so cross-config pass@1 differences mix sampler + operating point.
- Correlational across independently-generated configs (no paired seeds).
