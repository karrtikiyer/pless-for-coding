# Findings — Hidden-state loop-precursor detection does **not** generalize from synthetic to naturalistic reasoning loops

**Date:** 2026-06-26 · **Status:** complete (negative result) · **Design & deviations:**
`docs/pilot1_circular_reasoning_replication.md` · **Code:** `scripts/pilot1_{segment,extract,analyze,replicate,periodicity}.py`
· **Data:** `results/pilot1_hidden/`

## TL;DR

We faithfully reimplemented the early-loop-detection mechanism of **Duan, Pang et al.,
"Circular Reasoning" (arXiv:2601.05693)** — *both* its deployed detector (last-layer
linear probe + CUSUM) *and* its mechanistic precursor (K-means cluster-trajectory
periodicity) — and tested it on **naturalistic, paraphrastic** code-reasoning loops
(Qwen3-8B, ATCODER-interview), the setting the paper did not study (it used synthetic,
verbatim LoopBench loops). **The mechanism does not transfer:**

- Their **detector** catches only **~17–20 %** of terminal loops before onset (lower for
  transient), vs the paper's ~0.64–0.76 on synthetic loops.
- Their **precursor** — "semantic circularity precedes textual repetition" — is **absent
  before onset** on our loops: pre-onset periodicity is at/below chance vs clean
  (AUC 0.56 / 0.39 by two methods), and when periodicity is detected it fires **at or
  after** the textual onset (negative lead), not before.

The likely cause: their loops repeat *exact* tokens, so the representation cycles cleanly
*before* the surface repeats; ours repeat the *idea* in drifting words, so surface and
semantics stay entangled and there is no "semantic-precedes-textual" gap to exploit.

## Setup

- **Model / data:** Qwen3-8B; 90 traces from `pless_cot_efficiency_vllm/.../ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl`,
  balanced 30 **terminal** (looped to context cap) / 30 **transient** (looped then
  recovered) / 30 **clean**. Hidden states teacher-forced (HF, `output_hidden_states`),
  per-sentence mean-pooled at layers 6/16/24/36.
- **Ground-truth onset:** our validated n-gram detector (n=30, k=6, w=1200). *Mandated
  deviation* from the paper's 3-repetition textual onset (our loops are paraphrastic) →
  EDR/ASE/ATE are **directional, not numerically comparable** to their Table 3.
- **What we replicated:** Section-4 detector (probe + CUSUM, grid-searched α∈[1,2],
  p∈{3,4,5}, metrics EDR/FPR/ASE/ATE) and Section-3.1 precursor (K-means K=200 →
  periodicity). No authors' code exists (checked); reimplemented from the verified text.

## Result 1 — separation exists, but is mostly a trace-identity confound

Per-sentence loop-vs-normal AUC (exploratory, `pilot1_results.json`):

| layer | probe terminal-vs-clean | probe **pre-vs-post (within-trace)** | cosine **pre-vs-post** | cosine terminal-vs-clean |
|---|---|---|---|---|
| 6  | 0.881 | 0.753 | 0.603 | 0.934 |
| 16 | 0.885 | 0.739 | 0.588 | 0.947 |
| 24 | 0.902 | 0.749 | 0.566 | 0.950 |
| 36 | 0.900 | 0.785 | 0.550 | 0.944 |

Hidden states *do* separate loop-prone from clean traces strongly (0.88–0.95). **But the
within-trace control (post-onset vs pre-onset of the *same* traces) collapses it:** the
unsupervised cosine drops to **0.55–0.60 (≈ chance)** — the strong 0.95 was the
loop-trace-vs-clean-trace confound, not an onset signal. The supervised probe retains a
**moderate** within-trace signal (0.74–0.79), roughly flat across layers (no semantic-layer
peak; layer 36 marginally best at 0.785).

## Result 2 — the paper's detector: low early-detection recall (`pilot1_replicate_results.json`)

Faithful probe + CUSUM, best EDR at FPR ≤ 0.35 (grid-searched; best config was always the
most permissive α=1.0, p=3 — tightening only lowers EDR):

| layer | group | EDR | FPR | ASE (sent) | ATE (tok) | early/total |
|---|---|---|---|---|---|---|
| 36 (faithful) | terminal | 0.17 | 0.20 | 122 | 2303 | 5/30 |
| 36 | transient | 0.07 | 0.20 | 134 | 1684 | 2/30 |
| 24 | terminal | 0.20 | 0.20 | 98 | 1846 | 6/30 |
| 6  | terminal | 0.20 | 0.27 | 229 | 3998 | 6/30 |

**EDR ~0.17–0.20 (terminal), lower for transient — ~3–4× below the paper's ~0.7** at
comparable FPR. Notably, for the ~20 % it *does* catch, lead time is large (ASE ~100–230
sentences, ATE ~1.8–4 k tokens), comparable to/exceeding the paper's ~40–50 sent / ~1.3–2 k
tok. So the detector isn't useless — it fires genuinely early on a minority — but **misses
~80 %** of naturalistic loops. Layer barely matters.

## Result 3 — the paper's precursor: does **not** precede onset (`pilot1_periodicity_results.json`, layer 36)

Two operationalizations of cluster-trajectory periodicity (both ours; paper underspecifies
the detection step):

**(i) Pre-onset periodicity strength vs clean** — does the precursor exist *before* onset?
- Method 1 (label-match autocorrelation on K-means K=200): AUC **0.564**
- Method 2 (mean-centred trajectory autocorrelation): AUC **0.386**

Near chance / below — the pre-onset region of loop traces is **not** more periodic than
clean traces.

**(ii) Semantic-onset lead vs textual (n-gram) onset** (θ on held-out clean; FPR 0.07):

| method | group | fired | before-onset | lead median (tok) |
|---|---|---|---|---|
| label-match | terminal | 10/30 | 3 | **−152** |
| label-match | transient | 2/30 | 0 | **−6706** |
| centered | terminal | 12/30 | 6 | **−41** |
| centered | transient | 5/30 | 1 | **−730** |

Median lead is **negative** on both methods — periodicity, when detected, fires **at or
after** textual onset, not before. The two independent methods agree, so the negative is
robust to our operationalization choice.

## Synthesis

3b-1 alone left two explanations for the low probe recall: (a) the signal is genuinely
weak, or (b) the signal exists but the linear probe is a poor readout. **3b-2 settles it
as (a):** the probe-free geometric precursor is *itself* absent before onset (near-chance
pre-onset strength, negative lead). So it is not a readout problem —

> **The "semantic circularity precedes textual repetition" early-warning mechanism is
> substantially specific to clean, verbatim loops and does not transfer to naturalistic,
> paraphrastic code-reasoning loops.**

This is consistent with our earlier null result that *output-distribution* signals
(Σpᵢ², Shannon entropy, fork-density) also fail to give early warning on these loops — the
information simply isn't there pre-onset, in the logits *or* the hidden-state geometry, for
paraphrastic loops.

## Caveats / limitations

1. **Not numerically comparable** to the paper's Table 3 — different onset definition
   (n-gram vs 3-rep) and synthetic-vs-real data. Claims are *directional*.
2. **Periodicity-onset detection is our operationalization** (the paper underspecifies it);
   mitigated by two agreeing methods and an operationalization-independent pre-onset-strength AUC.
3. **Reimplementation from text** (no public authors' code/data found 2026-06-26);
   cannot be bit-identical.
4. **Re-tokenization from decoded text** (original generation token ids were not stored) —
   tiny, same limitation as the Σpᵢ² probe.
5. Single model (Qwen3-8B), single dataset (ATCODER-interview), n=30/group. A second model
   (e.g. DeepSeek-R1-Distill, whose loops are *more* verbatim) could plausibly show the
   precursor — an untested boundary of this negative result.

## Implication

Detection of reasoning loops via hidden-state precursors does not give a reliable, early,
low-supervision signal on realistic code loops. This reinforces the project's primary
direction — **prevention** (high-α p-less sampling, which on Qwen cut truncation
14.5 %→0.6 % and lifted pass@1, and is under test on DeepSeek) — over detect-and-escape.
The one open boundary worth a cheap check: re-run 3b-2 on DeepSeek-R1-Distill loops, which
are closer to the paper's verbatim regime, to see whether the precursor reappears there.

## Provenance / reproduce

```
# Phase 1 (CPU): manifest        scripts/pilot1_segment.py
# Phase 2 (GPU): hidden states   scripts/pilot1_extract.py   --layers 6 16 24 36
# Phase 3  (CPU): exploratory     scripts/pilot1_analyze.py
# Phase 3b-1 (CPU): faithful detector   scripts/pilot1_replicate.py
# Phase 3b-2 (CPU): precursor            scripts/pilot1_periodicity.py
```
All inputs/outputs under `results/pilot1_hidden/` (manifest committed; vectors regenerable).
