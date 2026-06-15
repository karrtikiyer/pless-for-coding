# CoT Token-Efficiency vs Accuracy — Qwen/Qwen3-8B / APPS

**Date:** 2026-06-13  
**Configs analyzed:** 4 (4 at the 32768-token budget used for the frontier)

Think length is measured in **tokens** (`Qwen/Qwen3-8B`). Efficiency is decomposed per arXiv:2602.09805 into completion rate, conditional correctness (pass rate among completed samples), and think length.

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
| pless_alpha a4.0 (t1.0) | 32768 | 98.6 | 1.4 | 0.7058 | 11279 | 9820 | 9677 | 0.6960 | 0.8214 |
| pless (t2.0) | 32768 | 99.8 | 0.2 | 0.6954 | 11027 | 9796 | 9749 | 0.6940 | 0.8214 |
| pless_alpha a5.0 (t1.0) | 32768 | 99.4 | 0.6 | 0.6905 | 11124 | 9646 | 9600 | 0.6861 | 0.8333 |
| pless_alpha a3.0 (t1.0) | 32768 | 97.3 | 2.7 | 0.6945 | 11495 | 9757 | 9502 | 0.6758 | 0.8056 |

**Coherence audit** — `pass@1 == compl% × cond-correctness` exactly for every config (no passing sample was truncated).

## Pareto-dominant configs (32768-token budget)

Not dominated on (shorter median think tokens, pass@1 within 1pt); configs with >25% truncation excluded as context-limited failures. Length axis = `median (all)` — budget-insensitive but biased UP by truncation rate (so a config may rank longer partly because it truncates more); read trunc% alongside:

| Config | median (all) | trunc% | pass@1 | pass@10 | cond-correctness |
|---|---|---|---|---|---|
| pless_alpha a5.0 (t1.0) | 9646 | 0.6 | 0.6861 | 0.8333 | 0.6905 |

## Limitations

- Single model / single difficulty; no cross-model generalization.
- Samplers compared are whatever was generated for this run; greedy is excluded by design (Qwen discourages it in thinking mode).
- Token counts are analysis-time estimates (tokenizer special-token handling may differ slightly from generation time).
- Truncation at the 32768 cap censors the upper tail: truncated samples are pinned near the cap (inflating the *mean* for configs that truncate) yet underestimate their true length. So length is NOT comparable across configs with different trunc% — see Column definitions.
- Stochastic samplers run at a fixed temperature, not matched effective entropy, so cross-config pass@1 differences mix sampler + operating point.
- Correlational across independently-generated configs (no paired seeds).
