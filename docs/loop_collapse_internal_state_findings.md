# Findings — the Circular-Reasoning "internal-state collapse" (Figs 3b & 4) reproduces on DeepSeek-R1-Distill but **not** on Qwen3-8B

**Date:** 2026-06-30 · **Status:** complete · **Paper:** Duan, Pang et al., *Circular Reasoning*
(arXiv:2601.05693), Fig 3b (statement-loop entropy/probability) + Fig 4 (layer-wise cosine/L₂
collapse across repetition cycles), case study on DS-Qwen-14B.
· **Code:** `scripts/loop_collapse_{screen,extract,plot,control}.py`
· **Data:** `results/loop_collapse_replication/<model>/{manifest.jsonl, vectors/*.npz, figures/*.png}`

## TL;DR

We replicated the paper's two case-study figures on **naturalistic** p-less ATCODER
code-reasoning loops (the paper used synthetic verbatim loops), for two 8B reasoning models,
on 5 hand-selected **reflective statement loops** each (period 50–275 / 46–221 tokens,
"Wait/perhaps/stuck/Let me think" impasse cycles). The result is a clean **cross-model split**:

- **DeepSeek-R1-Distill-Llama-8B reproduces the paper faithfully.** Layer-wise cosine across
  consecutive loop cycles deepens to ~1.0 (last-layer mean R1=**0.795** → R5=**0.997**), L₂
  vanishes (**74.4 → 9.5**), and entropy collapses at onset (pre **0.300** → post **0.134**
  nats; prob 0.923 → 0.966). The "Repeat-k" deepening and the post-RMSNorm L₂ last-layer spike
  match Fig 4 visually.
- **Qwen3-8B does NOT show the deepening dynamic.** Loop-cycle cosine is **saturated from the
  first cycle** (R1=**0.997** → R5=**0.999**) — the loop state is *directionally rigid on entry*,
  with no progressive collapse. L₂ tightens only mildly (**6.1 → 3.2**). Entropy barely moves at
  onset (pre **0.180** → post **0.123**; only 1 of 5 traces shows a clear Fig-3b collapse) — Qwen
  is already highly confident *before* the loop.

The boundary tracks model lineage: the paper's DS-Qwen-14B and our DeepSeek-R1-Distill are both
**R1 reasoning distillations**, and both show the collapse; **Qwen3-8B** (different training) does
not. This is consistent with our earlier output-distribution negative on Qwen (Σpᵢ²/entropy do
not separate Qwen loops, AUC≈0.5; `docs/pilot1_findings_hidden_state_detection.md`).

## Method (faithful + deviations)

- **Trace selection** (`loop_collapse_screen.py`, CPU): among truncated p-less samples, detect a
  verbatim **statement-level** period by token-stream autocorrelation on a loop-dominated
  post-onset region (period ≥10, match ≥0.85), pick an anchor token occurring **once per cycle**,
  and keep its per-cycle positions. From a full catalog we hand-picked 5 **reflective** loops per
  model spanning a range of periods (selection is transparent — `catalog.jsonl` + `--select`).
- **Extraction** (`loop_collapse_extract.py`, GPU, bf16): teacher-force prompt+think, store
  per-token top-1 probability + entropy (Fig 3b) and **all-layer** hidden states at the anchor's
  per-cycle positions (Fig 4). Causal-attention suffix truncation keeps the forward ≤~27k tokens
  (lossless for kept positions).
- **Faithful choices:** prob/entropy on the **raw** softmax (model's intrinsic determinism, not
  the p-less-filtered distribution); cosine/L₂ are **raw per layer** (no standardization) — cosine
  is scale-invariant and the last tuple entry is post-final-RMSNorm, matching the paper's plot.
- **Deviations:** anchor = once-per-period **rarest token** (not necessarily a phrase boundary like
  the paper's `\n\nBut`); only needs the *same* token at one-per-cycle cadence. Re-tokenized from
  decoded text (token ids were not stored) — same caveat as the pilot1 work.

## Result tables (verified, last layer; n=5 traces/model)

| metric | DeepSeek-R1-Distill-8B (L32) | Qwen3-8B (L36) |
|---|---|---|
| loop cosine R1 → R5 | 0.795 → 0.997 (**deepens**) | 0.997 → 0.999 (**saturated**) |
| loop L₂ R1 → R5 | 74.4 → 9.5 (**collapses**) | 6.1 → 3.2 (mild) |
| Normal-baseline cosine | 0.900 | 0.963 |
| entropy pre→post onset (nats) | 0.300 → 0.134 (**clear**) | 0.180 → 0.123 (weak) |
| prob pre→post onset | 0.923 → 0.966 | 0.940 → 0.960 |

Figures: `figures/fig3b_prob_entropy.png`, `fig4_per_trace.png`, `fig4_aggregate.png` per model.

## Control — is this loop signature specific to FAILURE? (`loop_collapse_control.py`)

A key control the paper lacks: do these verbatim loops appear in **completed** (closed `</think>`)
and **correct** traces, or only in truncated/failed ones? Cross-tab over all 2520 samples/model:

| | clean ≥6-cycle statement loop | any n-gram repeat (incl. transient) |
|---|---|---|
| **Qwen** completed+correct | **0** | 68 |
| Qwen completed+wrong | 0 | 49 |
| Qwen truncated (≈all wrong) | 138 | 336 |
| **DeepSeek** completed+correct | **3** | 54 |
| DeepSeek completed+wrong | 2 | 103 |
| DeepSeek truncated | 734 (733 wrong, 1 "correct") | 1599 |

- The **clean statement loop is overwhelmingly failure-associated**: Qwen 138/138 = 100% truncated;
  DeepSeek 734/739 = 99.3% truncated. It is **not** a benign pattern that also drives successful
  reasoning.
- But it is **not absolutely exclusive**: DeepSeek has 5 completed (3 correct) clean-loop traces
  (all task 1924, period ~28: *"Constraints… Wait, no, the problem statement says:"* repeated ~100×
  then escaped and solved). Qwen has **zero**.
- **Broader transient repetition is common and not fatal**: 68 (Qwen) / 54 (DeepSeek) completed
  **correct** traces tripped the n-gram detector on some 30-gram but recovered and solved. So
  "repetition" ≠ failure; the *sustained, terminal* statement loop is the failure signal.

Implication for Fig 4: a stronger control than the paper's non-repeating "Normal" baseline is a
**completed-recovered** statement loop. Qwen offers none (0); DeepSeek offers a small matched set
(the 3 completed+correct task-1924 loops) — an **untested extension**: if those collapse *less* /
recover, the collapse is diagnostic of being terminally stuck rather than a tautology of repetition.

## Caveats / scope

1. **5 traces/model**, single dataset (ATCODER-interview), single p-less config. The aggregate
   curves are means over 5; the cross-model split is large (cosine R1 0.80 vs 1.00) but n is small.
2. **Near-tautology for Qwen:** because Qwen's verbatim cycles are near-identical from cycle 1,
   its high cosine is partly mechanical; the *informative* contrast is that DeepSeek shows a true
   gradient while Qwen does not, and that Qwen's entropy never surges.
3. Re-tokenization from decoded text; anchor = once-per-period rarest token (operationalization).
4. Numbers here are pulled live from the npz / control JSONs, not from memory.

## Relation to prior threads

- **Detection negative** (`docs/pilot1_findings_hidden_state_detection.md`): the loop *precursor*
  (semantic circularity *before* onset) does not transfer to naturalistic Qwen loops. This work is
  the complementary *during-loop* signature — and it too is weak on Qwen, strong on DeepSeek.
- The Qwen entropy result here re-confirms the Σpᵢ²/entropy AUC≈0.5 negative: Qwen is confident
  in and out of loops, so the "determinism surge" has nothing to surge from.

## Reproduce

```
# Phase 0 (CPU): screen + select reflective statement loops
./run_loop_collapse_replication.sh screen           # or per-model --select (see manifests)
# Phase 1 (GPU pod, one model per GPU):
CUDA_VISIBLE_DEVICES=0 ... loop_collapse_extract.py --manifest <qwen> --model Qwen/Qwen3-8B ...
CUDA_VISIBLE_DEVICES=1 ... loop_collapse_extract.py --manifest <ds>   --model deepseek-ai/... ...
# Phase 2 (CPU): figures
./run_loop_collapse_replication.sh plot
# Control: completion x loop x correctness
uv run python scripts/loop_collapse_control.py --model <m> --jsonl <j> --metrics <metrics.json> --out <out.json>
```
