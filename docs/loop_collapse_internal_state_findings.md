# Findings — the Circular-Reasoning "internal-state collapse" (Figs 3b & 4): the rigid-loop **endpoint** reproduces on both 8B models; the collapse **transient** appears to differ (DeepSeek gradual, Qwen instant) but is **confounded** — controls pending

**Date:** 2026-06-30 · **Status:** endpoint solid; transient onset-confound **ruled out** (matched early-onset Qwen control), anchor-confound reduced → transient now read as a model property
· **Paper:** Duan, Pang et al., *Circular Reasoning* (arXiv:2601.05693), Fig 3b (statement-loop
entropy/probability) + Fig 4 (layer-wise cosine/L₂ collapse across cycles), case study on DS-Qwen-14B.
· **Code:** `scripts/loop_collapse_{screen,extract,plot,control}.py`
· **Data:** `results/loop_collapse_replication/<model>/{manifest.jsonl, vectors/*.npz, figures/*.png}`

## TL;DR

We replicated the paper's two case-study figures on **naturalistic** p-less ATCODER code-reasoning
loops (the paper used synthetic LoopBench loops), for two 8B reasoning models, on 5 hand-selected
**reflective statement loops** each (period 50–275 Qwen / 46–221 DeepSeek; "Wait/perhaps/stuck/Let
me think" impasse cycles). The careful reading separates **two questions**:

- **Endpoint — "is the loop a separable, rigid internal state?" (the paper's universal claim, and
  the basis of its detector). BOTH models pass.** Settled consecutive-cycle cosine reaches ~1.0 and
  sits clearly above the normal-recurring-token baseline: DeepSeek **0.997 vs 0.90**, Qwen **0.999
  vs 0.96**. So Qwen's loop *is* a distinct rigid state — consistent with the paper (incl. its
  Table-3 detection EDR 0.64 for Qwen3-8B on synthetic loops). An earlier draft's "Qwen does NOT
  show the collapse" was **wrong** — it conflated the transient (below) with the endpoint.

- **Transient — "how fast does it rigidify?" The two models *appear* to differ, but it is
  confounded.** DeepSeek ramps gradually (last-layer consecutive cosine R1=**0.795** → R5=**0.997**;
  L₂ 74.4 → 9.5; entropy collapse 0.300 → 0.134 nats, 5/5), matching Fig 4's "Repeat-k" deepening.
  Qwen is **instant** (R1≈**0.997** already; L₂ 6.1 → 3.2; entropy barely moves 0.180 → 0.123, 1/5).
  A matched **early-onset Qwen control** (mean onset 3822 ≈ DeepSeek's 4813) is *still* instant
  (R1=0.995) → the onset-depth confound is **ruled out**; with 10 Qwen traces across 10 anchors all
  instant, the transient is now best read as a **genuine model property** (see confounds section).

**Verified (no GPU): both models lock into *exact verbatim* repetition within ~1 cycle** (per-cycle
token-match to the steady block jumps 0→1.0 at the first captured cycle). So the activation-ramp
difference is **not** a paraphrastic-entry artifact and **not** a windowing artifact (we capture
entry for both). It is a genuine representational difference *in our data* — but see confounds.

**The paper's "robust across architectures" claim is about detector efficacy (endpoint
separability), NOT the Fig-4 ramp shape** — Fig 4 was only ever run on DS-Qwen-14B. So our
transient observation lives *underneath* the paper's resolution: it is neither predicted nor
contradicted by it.

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

## Results — endpoint vs transient (verified, last layer; n=5 traces/model)

| property | what it tests | DeepSeek-R1-Distill-8B (L32) | Qwen3-8B (L36) | confounded? |
|---|---|---|---|---|
| **Separable rigid loop state** (endpoint — *paper's universal claim / detector basis*) | settled loop cosine ≫ normal | **0.997 vs 0.90** ✅ | **0.999 vs 0.96** ✅ | no — solid |
| Collapse **transient** (cosine ramp R1→R5) | *how fast* it rigidifies | 0.795 → 0.997 (gradual) | 0.997 → 0.999 (instant) | **yes** |
| L₂ reduction (R1→R5) | magnitude tightening | 74.4 → 9.5 (8×) | 6.1 → 3.2 (2×) — raw scale not cross-comparable | yes (same transient) |
| entropy pre→post onset (nats) | determinism surge | 0.300 → 0.134 (5/5) | 0.180 → 0.123 (1/5) | partly |
| prob pre→post onset | probability surge | 0.923 → 0.966 | 0.940 → 0.960 | partly |

Figures: `figures/fig3b_prob_entropy.png`, `fig4_per_trace.png`, `fig4_aggregate.png` per model.

### Reconciliation with the paper's Table 3 (detection) and "robust across architectures"

The paper's Table 3 reports CUSUM-probe **detection** (EDR/FPR/lead-time) on **LoopBench**
(synthetic, greedy) — and lists **Qwen3-8B at EDR 0.64** (the *lowest* of the 8 models; DeepSeek/Phi
0.72–0.76). Detection is an *endpoint/separability* metric — it works iff loop-states are
classifiable from normal-states. Our endpoint row shows Qwen's loop is separable (0.999 ≫ 0.96), so
**we agree with the paper that Qwen3-8B loops are an internally detectable state** (this is the same
synthetic 0.64–0.76 number our detection-thread negative compared against — `pilot1_findings`).
"Robust across architectures" is this detector-efficacy claim; the **Fig-4 ramp shape was only run
on DS-Qwen-14B**, never per-model, so our transient observation neither confirms nor contradicts it.

### Confounds — one RULED OUT by a matched control, one reduced to a small residual

1. **Loop-onset depth — RULED OUT (control run).** Qwen's main 5 onset late (mean ≈10.8k) vs
   DeepSeek early (≈4.8k), so Qwen loops carry ~2× more preceding KV-cache context, which *could*
   pre-stabilize the representation. We ran the matched control: 5 **early-onset** Qwen reflective
   loops (`manifest_early.jsonl`; tasks 1085/1172/2505/2367/2522; mean onset **3822**, *matched to
   DeepSeek's 4813*; periods 46–190, same reflective bar). Result: early-onset Qwen is **just as
   instant** — last-layer cosine **R1=0.995** → R5=0.999 (vs late Qwen R1=0.997; DeepSeek R1=0.795).
   All 5 have R1≥0.989. ⇒ deep context was **not** the cause. (Bonus: early-onset Qwen has *higher*
   pre-onset entropy, 0.385 vs late 0.180, yet still instant — so Qwen's directional rigidity is
   **decoupled** from pre-loop confidence.)
2. **Anchor-token type — reduced to a small residual.** Anchor = "rarest once-per-period token"; the
   two tokenizers picked different *kinds* (Qwen word-initial/punctuation; DeepSeek some mid-word BPE
   fragments). With the early-onset set we now have **10 Qwen traces across 10 different anchors**
   (`follows`,`then`,`as`,`avoid`,`efficiently`,`No`,`solved`,`yes`,`=`,`stuck`) — **all instant**
   (R1≥0.989). Such consistency across varied anchors makes an anchor-artifact explanation unlikely.
   **Optional remaining hardening:** block-average cosine/L₂ over all P period positions (the paper's
   "activation vectors of identical tokens" read literally) would formally close it.

**Updated claim (confound-controlled):** the transient difference is now best read as a **genuine
model property** — Qwen3-8B's repeated-token representation is rigid from the first verbatim cycle
(robust to onset depth and anchor choice), while DeepSeek-R1-Distill's keeps relocating for ~4 cycles
(cos→0.58 from cycle 0) before locking. The **endpoint** (separable rigid loop) holds for both; only
the **route in** differs. Both are consistent with the paper, which measured only the endpoint
(detector) universally and the ramp on a single model (DS-Qwen-14B).

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

## Truncation taxonomy — does verbatim-loop detection cover the failures? (`loop_collapse_categorize.py`)

We studied **verbatim statement loops** (Fig 3b/4). But what fraction of *all* truncations are they?
Categorizing every truncated sample (`truncation_taxonomy.json` per model):

| failure mode | Qwen | DeepSeek | caught by verbatim n-gram / Fig-4 collapse? |
|---|---|---|---|
| **Verbatim statement loop** (periodic, P 10–800; the studied kind) | 40.7% (reflective 18.0%) | 49.8% (reflective 32.5%) | ✅ yes |
| **Paraphrastic / semantic drift** (n-gram fires, but aperiodic) | **41.3%** | **46.8%** | ❌ no |
| Short / degenerate (<10 tok: digit runs + short code fragments) | 9.9% | 1.3% | partial |
| No detected loop (n-gram never fires) | 8.2% | 2.3% | ❌ no |
| (long period >800) | 0% | 0% | — (empty; dropped) |

**The paraphrastic class is real, not a threshold artifact** (dissected): among "paraphrastic" traces
the *best* periodicity strength (max token self-match over any lag 10–800) is below 0.5 for **74%
(Qwen) / 73% (DeepSeek)**, median **0.23 / 0.26**. Reading them confirms the mechanism — the model
re-explores the *same idea in drifting words* (e.g. *"But how to compute this? One approach is BFS…
but that's impossible. So perhaps a different approach. Wait, perhaps…"* → next pass, same theme, new
words). No minimal repeating unit recurs, so there is no verbatim period and **no identical token
across cycles** for the Fig-4 collapse to be computed on.

**Answer: verbatim-loop detection covers ≤41% (Qwen) / ~50% (DeepSeek) of truncations.** A roughly
equal share (~41–47%) is genuine paraphrastic/semantic drift that *neither* the n-gram detector nor
the hidden-state collapse can catch, plus ~10–18% short/degenerate/no-loop. The "numerical loop" I
earlier guessed dominates is in fact only the **single-token digit runs = 3.3% (Qwen) / 0.4%
(DeepSeek)**.

### Reconciliation with LoopBench — paraphrastic drift is *outside the paper's loop definition*

The paper's loop definitions (Appendix B, p13) are both **unit-recurrence** criteria: *Numerical
Loop* = `k·l > 500` (minimal repeating unit length `l`, `k` consecutive repeats); *Statement Loop* =
sentence-unit recurs **`k > 3`**. Both require a **minimal repeating *unit* that recurs** — i.e.
verbatim/near-verbatim repetition. A purely paraphrastic case (same idea, new words, no recurring
unit) **cannot satisfy either** → it is not labeled a loop in LoopBench, and the Table-3 detection
set ("repetitive vs non-repetitive cases") is verbatim by construction. The paper *does* describe a
semantic-drift phase (Fig 5c, *"semantic circularity precedes textual repetition,"* sentences
*"lexically distinct"* but cluster-recurrent) — but strictly as the **precursor that ends in verbatim
repetition**, not a standalone failure mode. LoopBench's tasks (high-precision arithmetic with a
500-digit cap; recursive state tracking — Table 4) are engineered to induce exactly that verbatim
digit/state recurrence.

Consequences:
1. **Our paraphrastic class (~41–47%) is outside the paper's loop definition entirely** — those
   traces have no unit recurring `k>3` (median self-match 0.23–0.26), so the paper would not count
   them as loops. Its detector + intervention target only the verbatim half.
2. **This mechanistically explains the detection-transfer failure** (with `pilot1_findings`): on
   LoopBench every loop *by definition* reaches verbatim repetition with a semantic precursor, so
   "semantic circularity precedes textual repetition" always has a textual onset to precede; on our
   naturalistic data ~half the loops never converge to verbatim (they drift to the token cap), so the
   precursor mechanism has nothing to anticipate — hence their synthetic EDR 0.64 → our 0.17–0.20.
3. **So paraphrastic semantic drift is a loop class the paper's benchmark construction and `k>3`
   definition structurally exclude** — a novel-relative-to-the-paper failure mode, and the one most
   resistant to both verbatim and hidden-state detection.

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
