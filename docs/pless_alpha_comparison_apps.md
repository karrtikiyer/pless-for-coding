# Pless α-schedule comparison — APPS ATCODER-interview (Qwen3-8B & DeepSeek-R1-Distill)

**Date:** 2026-07-13
**Scope:** ATCODER-interview, 252 problems, n=10 samples/problem, per-token pless family.
Three conditions compared per model:
- **α=2 (default):** standard pless (`p = Σpᵢ²`).
- **α=5 (prevention):** `pless_alpha(5)` from the start — flatter survivor set, avoids the loop.
- **adaptive (α=2 → detect → chop → α=5):** run α=2; on live n-gram loop detection, chop the
  looping span and continue the same sample at α=5 (no nudge). Detector: Qwen 30-gram/k=6/
  window=1600; DeepSeek 30-gram/k=8/window=3000.

> **Status of the adaptive column: PROVISIONAL / UNRECONCILED.** Its numbers were produced by
> *inline* scoring during a *fresh HF* generation, and they fail two consistency checks (below).
> They are NOT comparable to the α=2/α=5 columns yet and must be re-scored through the standard
> pipeline before use. The α=2 and α=5 columns are standard-pipeline-scored and trusted.

## Qwen3-8B

| condition | pass@1 | pass@3 | pass@5 | pass@10 | no-code (≈trunc) |
|---|---|---|---|---|---|
| pless α=2 (default) | 0.625 | 0.757 | 0.792 | **0.825** | 10.5% |
| pless α=5 (prevention) | **0.686** | 0.781 | 0.803 | **0.833** | 0.4% |
| adaptive α=2→α=5 *(provisional)* | 0.673 | 0.762 | 0.782 | 0.798 | 3.7% |

## DeepSeek-R1-Distill-Llama-8B

| condition | pass@1 | pass@3 | pass@5 | pass@10 | no-code (≈trunc) |
|---|---|---|---|---|---|
| pless α=2 (default) | 0.174 | 0.301 | 0.368 | 0.464 | 49.3% |
| pless α=5 (prevention) | **0.295** | **0.442** | **0.500** | **0.560** | 0.8% |
| adaptive α=2→α=5 *(provisional)* | 0.447 | 0.594 | 0.642 | 0.690 | 6.0% |

(`no-code` = fraction of samples with no extractable code, i.e. the sample never closed
`</think>`/produced code — the truncation proxy. Computed from `per_task.extraction_success`
for α=2/α=5; from `closed_think` for adaptive.)

## Source files (provenance — all numbers pulled live, none from memory)
- **α=2:** `results/pless_cot_efficiency_vllm/{Qwen--Qwen3-8B/ATCODER_interview_all_252,
  deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview}/metrics/pless_think_t1.0_t1.0_metrics.json`
- **α=5:** `results/pless_recovery_full252/{…}/metrics/pless_alpha_think_t1.0_a5.0_t1.0_metrics.json`
- **adaptive:** `results/_live_adaptive/{qwen,deepseek}_full_n10.jsonl` (pass@k via the unbiased
  estimator from per-task correct counts; inline-scored).

## Findings

### 1. Prevention (α=5) is the clean, trusted winner
On the standard-scored columns, α=5 beats α=2 on pass@1 for both models — Qwen **0.625→0.686**,
DeepSeek **0.174→0.295** — and eliminates truncation (no-code **10.5%→0.4%** Qwen, **49.3%→0.8%**
DeepSeek). The DeepSeek gain is large because its α=2 truncation is severe (~half of samples
produce no code); flattening the distribution removes the loop. Prevention also holds or improves
pass@10 (Qwen 0.825→0.833; DeepSeek 0.464→0.560). This is the established result.

### 2. Qwen adaptive is plausible but trades pass@10 for pass@1
Adaptive pass@1 (0.673) sits **below** α=5 (0.686) — the expected ordering (prevention ≥ rescue) —
and above α=2 (0.625). But adaptive pass@10 (**0.798**) is **below both** α=2 (0.825) and α=5
(0.833): the rescue lifts per-draw accuracy but **loses coverage**, consistent with false-positive
chopping cutting some genuine long-reasoners (the A35 loop-force finding). Even here, treat as
provisional until re-scored.

### 3. DeepSeek adaptive is NOT believable as-is — two failed checks
- **Baseline mismatch:** the adaptive run's own plain-α=2 baseline (non-fired samples' pass rate,
  which should reproduce benchmark α=2) is **0.386 vs the benchmark α=2 of 0.174 — 2.2× too high.**
  (Qwen's equivalent is 0.587 vs 0.625 — close, tolerable.)
- **Sanity violation:** adaptive pass@1 **0.447 > α=5 prevention 0.295.** Rescue cannot beat
  prevention on the same generation — prevention (α=5 from the start) strictly dominates
  "α=2 then rescue only the loopers." So the number is logically backwards.

### 4. Why "unreconciled" — the gap we must close
The adaptive numbers were produced by **inline scoring** on a **fresh HF α=2 generation**; the
α=2/α=5 columns by the **standard eval pipeline** on the **base vLLM runs**. Two unseparated
differences: (a) scoring path (inline extraction+exec vs standard), and (b) generation (fresh HF
α=2 loops far less than base vLLM — DeepSeek fired 40.2% vs base no-code 49.3%, so it completes and
passes more). Until the adaptive outputs are re-scored through the same standard pipeline (and the
live α=2 baseline matches benchmark α=2, or we understand why not), the adaptive column is measured
on a different footing and the DeepSeek row proves that gap is biting.

## Next step (required before the adaptive column is usable)
Re-score both live runs (`*_full_n10.jsonl`, which store `samples`/`samples_with_thinking`) through
the standard eval pipeline; recompute the adaptive column; re-check that the live α=2 baseline
reproduces benchmark α=2 (0.625 Qwen / 0.174 DeepSeek) and that adaptive ≤ α=5. Only then report
the rescue gain.

## Deployable takeaway so far (trusted columns only)
**Prevention (α=5 from the start) is the recommended lever** — it beats default α=2 on pass@1 and
removes truncation, with no detector needed. The adaptive rescue's value over prevention is
**not established** (and cannot exceed prevention); pending reconciliation.
