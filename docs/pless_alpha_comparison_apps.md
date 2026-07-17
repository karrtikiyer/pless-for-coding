# Pless α-schedule comparison — APPS ATCODER-interview (Qwen3-8B & DeepSeek-R1-Distill)

**Date:** 2026-07-16 (supersedes the 2026-07-13 version — its DeepSeek numbers were confounded
by a vLLM prompt-tokenizer bug, and its adaptive column was cross-backend; see **Correction
history** below).

**Scope:** ATCODER-interview, 252 problems, n=10 samples/problem, per-token pless family.
All three conditions are now on **one footing — fixed-vLLM backend, `bench.eval` scoring** — so
the comparison is apple-to-apple (no HF↔vLLM crossing, no mangled prompts).

Conditions:
- **α=2 (default):** standard pless (`p = Σpᵢ²`).
- **α=5 (prevention):** `pless_alpha(5)` from the start — flatter survivor set, avoids the loop.
- **adaptive (α=2 → detect → chop → α=5):** run α=2; on live n-gram loop detection, chop the
  looping span back to onset and continue that sample at α=5 (no nudge). Detector: Qwen
  30-gram/k=6/window=1600; DeepSeek 30/k=8/window=3000. **Single-chop** (see caveats).

## Qwen3-8B

| condition | pass@1 | pass@3 | pass@5 | pass@10 | no-code (≈trunc) |
|---|---|---|---|---|---|
| pless α=2 (default) | 0.625 | 0.757 | 0.792 | 0.825 | 14.5% |
| adaptive α=2→α=5 | 0.682 | 0.787 | **0.818** | **0.845** | 2.7% |
| pless α=5 (prevention) | **0.686** | **0.781** | 0.803 | 0.833 | 0.6% |

## DeepSeek-R1-Distill-Llama-8B

| condition | pass@1 | pass@3 | pass@5 | pass@10 | no-code (≈trunc) |
|---|---|---|---|---|---|
| pless α=2 (default) | 0.392 | 0.527 | 0.574 | 0.627 | 41.7% |
| adaptive α=2→α=5 | 0.457 | 0.589 | 0.633 | 0.687 | 7.1% |
| pless α=5 (prevention) | **0.483** | **0.619** | **0.663** | **0.714** | 0.3% |

(`no-code` = fraction of samples that never closed `</think>` — the loop-truncation proxy.)

## Findings

### 1. Prevention (α=5) is the robust winner; adaptive is competitive only on the low-loop model
- **DeepSeek (40.8% of samples loop):** α=5 > adaptive > α=2 at **every** k — α=5 beats adaptive
  by **+2.6pp pass@1** (0.483 vs 0.457) and **+2.7pp pass@10** (0.714 vs 0.687). Prevention is
  clearly best.
- **Qwen (20.2% loop):** α=5 ≈ adaptive at pass@1 (0.686 vs 0.682, within noise), and adaptive
  **edges** α=5 at pass@3/5/10 (pass@10 **0.845 vs 0.833**). Essentially a tie, adaptive
  marginally ahead on coverage.
- **Interpretation:** the more a model loops, the more "α=5 everywhere" (prevention) beats
  surgical rescue. On a low-loop model, adaptive's "α=2 where the model is fine, α=5 only where
  it loops" captures the union of both regimes and ties/edges prevention on coverage; on a
  high-loop model, pervasive looping + the α=2-prefix cost + the single-chop limit let
  prevention pull ahead.

### 2. Adaptive strongly beats the α=2 default on both models (rescue works)
pass@1 **+5.7pp** Qwen (0.625→0.682), **+6.5pp** DeepSeek (0.392→0.457); truncation collapses
(Qwen 14.5%→2.7%, DeepSeek 41.7%→7.1%). So the chop→α=5 rescue is real — it just doesn't
outperform doing α=5 from the start (except marginally on Qwen).

### 3. The earlier "adaptive ≫ α=5" (DeepSeek) result was 100% artifact — now resolved
See Correction history. Short version: the old DeepSeek α=5 (0.295) was generated on
whitespace-mangled prompts, and adaptive (HF, un-mangled) was compared against it cross-backend.
Both defects removed, the ordering flips to the expected α=5 ≥ adaptive.

## Correction history (why the 2026-07-13 numbers were wrong)

1. **Tokenizer bug (DeepSeek only).** Every DeepSeek **vLLM** run — α=2, α=5, and the temp
   cross-method arms — was generated with WHITESPACE-MANGLED prompts: transformers-v5 routed
   DeepSeek's tokenizer through `LlamaTokenizer`, whose Metaspace override strips spaces/newlines
   (HF #45488), so the model saw `deff(a,b):` instead of `def f(a, b):`. Fixed in `abdc0dc`
   (`bench/generator_vllm.py:encode_prompt_for_vllm` pre-encodes with the safe tokenizer).
   Effect: **α=5 0.295 → 0.483 (+19pp); α=2 0.174 → 0.392 (+22pp)**. α=5 got the biggest
   correction because it was fully mangled — which is exactly what fabricated the paradox
   (a broken α=5 compared against a healthy adaptive). Qwen was immune (Qwen2Tokenizer).
2. **Cross-backend adaptive.** The chop rescue needs mid-stream KV rollback, which vLLM can't do,
   so the original adaptive was HF-native. The 2026-07-13 table compared **HF adaptive** to
   **vLLM α=5** — different backend *and* (for DeepSeek) a mangled α=5.
3. **Resolution.** (a) Fix + regenerate all vLLM DeepSeek configs; (b) reconstruct the adaptive
   on vLLM (`scripts/vllm_adaptive_reconstruct.py`): reuse the fixed vLLM α=2 as phase-1 (so the
   non-fired samples ARE the α=2 baseline — zero offset), detect the loop onset with the deployed
   `RepeatDetector`, and continue fired samples at α=5 via `TokensPrompt`. Now all three share the
   vLLM footing + `bench.eval`.
   - **Port check (DeepSeek):** recon adaptive **0.457 ≈ HF adaptive 0.447** (within ~1pp at every
     k) → the reconstruction is faithful.
   - **Qwen footing note:** the recon adaptive pass@10 (0.845) is ~4.7pp **above** the HF-inline
     adaptive (0.798). That gap is the HF-adaptive-run's pipeline offset (its own α=2 baseline read
     ~4pp below vLLM α=2), *not* a scoring artifact (HF bench.eval == HF inline == 0.798 here). The
     recon removes that offset, so **the recon is the apple-to-apple number** vs vLLM α=2/α=5.

## Provenance (all numbers pulled live from metrics, none from memory)
- **Qwen α=2 / α=5:** `results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json`;
  `results/pless_recovery_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_alpha_think_t1.0_a5.0_t1.0_metrics.json`.
- **DeepSeek α=2 / α=5 (fixed vLLM):** `results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/{pless_think_t1.0_t1.0, pless_alpha_think_t1.0_a5.0_t1.0}_metrics.json`.
- **adaptive (both):** `.../pless_adaptive_recon.jsonl` from `scripts/vllm_adaptive_reconstruct.py`
  (fixed vLLM α=2 phase-1 + α=5 continuation), scored `python -m bench.eval --dataset apps
  --skip-diversity`.
- HF-adaptive references (port check): `results/_live_adaptive/{qwen,deepseek}_full_n10.jsonl`.

## Caveats
- **Re-chop tested (DeepSeek) — does NOT close the gap.** The 1-chop reconstruction left ~10% of
  fired samples re-looping at α=5. Running MAX_CHOPS=3 (matching HF) raised closure 90.5%→94.6%
  and cut truncation 7.1%→5.4%, **but pass@1/@10 stayed flat** (0.452/0.687 vs 1-chop 0.457/0.687)
  — the re-rescued samples are the hardest ones (48 needed 2 chops, 70 needed 3); escaping the
  loop makes them *complete with wrong code*, not pass. So adaptive ≈ 0.45–0.46 on DeepSeek at any
  chop depth, and α=5 (0.483) still wins — the gap is real, not a single-chop artifact. (Qwen
  re-chop not yet run.)
- **Diversity axis not recomputed** here (`--skip-diversity` for speed); `struct_div`/`cb_div`
  can be added if the diversity comparison is needed.
- **Detector configs are per-model** (Qwen 30/6/1600, DeepSeek 30/8/3000). The DeepSeek window was
  re-validated on un-mangled traces with the deployed detector: 30/8/3000 → 92.3% catch / 2.5% FP;
  30/8/4000 → 96.6% / 2.5% (so 3000 is slightly conservative — it under-rescues, which biases
  *against* adaptive, not for it).

## Deployable takeaway
**Prevention (α=5 from the start) is the recommended lever** — it wins outright on the high-loop
model (DeepSeek) and ties the surgical adaptive on the low-loop model (Qwen), with no detector
required. Adaptive is worth its complexity only where looping is mild (Qwen), and even there the
gain over α=5 is marginal (coverage only). The chop rescue does clearly beat the α=2 default on
both models, so it remains the right move *if you are stuck on α=2*; but if you can choose the
sampler up front, α=5 is simpler and at least as good.
