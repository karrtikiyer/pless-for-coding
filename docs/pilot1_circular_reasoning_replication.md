# Pilot 1 — Faithful replication of the Circular-Reasoning hidden-state mechanism on *naturalistic* loops

**Status:** design locked (pre-registration), 2026-06-26. Code: `scripts/pilot1_segment.py`
(Phase 1), `scripts/pilot1_extract.py` (Phase 2), `scripts/pilot1_analyze.py`
(Phase 3, exploratory), `scripts/pilot1_replicate.py` (Phase 3b, faithful — to build).

## Goal

Test whether the early-loop-detection mechanism of **Duan, Pang et al., "Circular
Reasoning: Understanding Self-Reinforcing Loops in Large Reasoning Models"
(arXiv:2601.05693, Jan 2026)** generalises from their **synthetic, verbatim**
LoopBench loops to our **naturalistic, paraphrastic** ATCODER-interview code-reasoning
loops (Qwen3-8B). This is the generalisation test the paper did not run.

## Provenance — why we reimplement from the description

No public code or data exists for the paper (checked 2026-06-26): the arXiv page has
no code/data link; general, author-name, and "LoopBench" searches returned nothing; the
senior author's homepage (Liang Pang, GitHub `pl8787`) does not list the paper. So we
reimplement from the paper text + appendix, which we read and verified directly (HTML +
PDF appendix D). **This is reimplementation-from-description; it cannot be bit-identical
to the authors' code.** Every method detail below was taken verbatim from the paper.

## The paper's mechanism (verified from arXiv:2601.05693, incl. Appendix D)

### Part 1 — the evaluable detector (Section 4, Table 3)
- **Unit:** sentence. **Feature:** mean of the **last-layer** hidden states over the
  sentence's tokens (`h_i`).
- **Probe:** linear classifier, `x_i = wᵀh_i + b`, trained to separate repetitive vs
  non-repetitive sentences. Training set: "≥50 repetitive and ≥50 non-repetitive cases"
  per model (balanced).
- **CUSUM:** `S_i = max(0, S_{i-1} + (x_i − r))`, with `r` = mean score on
  non-repetitive cases, `h = α · S_max` calibrated on a held-out normal set, fire when
  `S_i > h` for `p` consecutive steps. Grid-searched **`p ∈ {3,4,5,…}`** and
  **`α ∈ [1.0, 2.0]`** per model (Appendix D.2).
- **Ground-truth onset (`t_onset`):** the start of explicit **textual** repetition;
  intervention breakpoints set at **3 repetitions** for statement loops, 20 for numerical
  (Appendix E).
- **Metrics (Appendix D.1):**
  - `EDR = N_early / N_loop` — fraction of loops whose alert fires **before** `t_onset`.
  - `FPR = N_fp / N_normal`.
  - `ASE = mean(S_onset − S_alert)` over **early-detected** samples (sentence units).
  - `ATE = mean(T_onset − T_alert)` over early-detected samples (token units).
- **Reported (their data):** EDR ≈ 0.64–0.76, FPR ≈ 0.24–0.34, ASLT ≈ 40–51 sentences,
  ATLT ≈ 1306–1980 tokens.

### Part 2 — the mechanistic precursor (Section 3.1, Fig 22)
- **K-means (K=200)** on per-sentence last-layer mean hidden states → cluster-label
  trajectory. **Periodic oscillation of cluster labels precedes textual onset**
  ("semantic circularity precedes explicit textual repetition"). Also: per-token
  activation cosine across repetition cycles saturates → ~1.0 ("state collapse").
- NOTE: the paper does **not** specify how periodicity onset is detected
  computationally — this step is underspecified (see D-periodicity).

## Our setting (fixed)

Qwen3-8B; 90 traces (30 terminal / 30 transient / 30 clean) from
`results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl`;
per-sentence hidden states at layers [6,16,24,36] already extracted
(`results/pilot1_hidden/`). Faithful replication uses **layer 36** (their last layer);
6/16/24 are reported as *our* extension.

## Fidelity table

| Component | Paper | Ours | Status |
|---|---|---|---|
| unit / feature | sentence / last-layer mean | same (layer 36) | identical |
| probe | linear | linear (logistic) | identical |
| CUSUM formula | `S=max(0,S+(x−r))`, fire p-consec > h | same | identical |
| CUSUM tuning | grid α∈[1,2], p∈{3,4,5} | **match** (was fixed α=1.5,p=3) | fidelity fix |
| K-means | K=200 | K=200 | identical |
| metrics | EDR/FPR/ASE/ATE | **add these** (were lead-time/fired) | fidelity fix |
| probe training | 50/50 balanced holdout | leave-trace-out grouped CV | **D-probe** |
| onset def | 3-rep textual | n-gram n=30/k=6/w=1200 | **D-onset (mandated)** |
| periodicity onset | unspecified | autocorrelation/FFT of cluster seq | **D-periodicity** |
| data | synthetic LoopBench | naturalistic ATCODER | mandated (the point) |
| token ids | (their pipeline) | re-tokenized from decoded text | inherent limitation |

## Deviations — decisions and justifications

### D-onset — **mandated by our data.** Keep the n-gram onset (n=30/k=6/w=1200).
Their "3 identical-sentence repetitions" assumes clean verbatim loops; our loops are
paraphrastic (same idea, drifting words) so identical-3-rep would rarely fire. We use the
n-gram detector already validated for our setting. **Consequence:** ASE/ATE/EDR are
measured relative to a *different* zero-point than their Table 3, so the **numbers are
not directly comparable**; only the qualitative claim ("does the signal fire before
onset, and how reliably") transfers.

### D-probe — **rigor improvement.** Keep leave-trace-out grouped CV as primary.
Their random 50/50 holdout risks same-trace sentences spanning train/test (leakage that
inflates AUC). Leave-trace-out `StratifiedGroupKFold` is strictly cleaner. We report CV
as primary; a 50/50-random secondary can be added if a closer numeric match is wanted.

### D-periodicity — **underspecified in paper; our operationalization (corrected).**
The paper shows cluster-label periodicity precedes onset (Fig 22c) but never specifies
*how* periodicity is detected. Our first draft ("autocorrelation of the integer
cluster-label sequence") was **wrong** — K-means labels are *nominal*, so numeric
autocorrelation is meaningless. Corrected method, run BOTH ways:

- **Method 1 (faithful — keeps K-means K=200):** label-**match** autocorrelation.
  K-means K=min(200, n_sent) on layer-36 sentence vectors → label sequence; for lag d,
  `τ(d) = mean_i [label_i == label_{i+d}]` (autocorrelation of the *match indicator*,
  not the integer values). Periodicity = a τ(d) peak well above the chance floor
  `Σ_k (n_k/N)²`. This is the correct categorical-sequence autocorrelation.
- **Method 2 (cross-check — K-means-free):** autocorrelation of the *mean-centred*
  hidden-state trajectory: subtract the trace's mean sentence vector, then
  `ρ(d) = mean_i⟨c_i,c_{i+d}⟩ / mean_i⟨c_i,c_i⟩`. Reveals periodic structure above the
  high baseline self-similarity. Tests the "trajectory cycles" claim directly; if it
  agrees with Method 1, the K-means step isn't an artefact.

For both: **semantic onset** = first sliding-window position where the windowed
periodicity metric stays above a threshold θ (calibrated on held-out clean traces) for
p consecutive steps; **lead = n-gram-onset − semantic-onset**. Clean false-periodicity
rate measured on held-out clean. Both flagged as OUR operationalization, not the authors'.

### Kept identical (no deviation)
per-sentence last-layer mean features; linear probe; CUSUM formula; K=200;
EDR/FPR/ASE/ATE definitions.

### Inherent limitations (cannot be removed)
1. Reimplementation-from-text (no authors' code) — not bit-identical.
2. Only decoded text was stored (not original generation token ids) → re-tokenization
   may differ slightly from emitted ids. Same limitation as the Σpᵢ² probe.
3. Synthetic-verbatim (theirs) vs naturalistic-paraphrastic (ours) loops — the
   substantive difference we are testing, not a flaw.

## Comparability statement (for any writeup)

Because of D-onset and the synthetic-vs-natural data difference, our EDR/ASE/ATE are
**not** head-to-head numeric comparisons with their Table 3. The claim we *can* make:
"Using the paper's own probe+CUSUM detector (and their K-means precursor analysis),
faithfully reimplemented, does an early hidden-state signal precede the loop on
naturalistic code loops — and how does it compare to the supervised upper bound and the
trace-identity confound control?"

## Plan

- **Phase 3b-1** (`pilot1_replicate.py`): faithful probe+CUSUM on **layer 36**,
  grid-searched α/p, reporting **EDR/FPR/ASE/ATE** per group (terminal/transient/clean),
  plus the within-trace (pre-vs-post) control. Layers 6/16/24 reported as an extension.
- **Phase 3b-2** (`pilot1_periodicity.py`): periodicity precursor, BOTH methods above
  (label-match autocorrelation on K-means K=200, and mean-centred trajectory
  autocorrelation). Reports: (i) periodicity-strength separation loop-vs-clean (does the
  precursor exist at all), (ii) semantic-onset lead vs n-gram onset per loop group,
  (iii) clean false-periodicity rate. Tests whether the signal exists in the geometry
  even where the linear probe (3b-1) is a poor readout.

Both run offline on the existing vectors (no GPU).

## Decision log

- **2026-06-26** — Design locked. D-onset = keep n-gram (mandated). D-probe = leave-trace-out
  CV primary (rigor). D-periodicity = autocorrelation/FFT operationalization (ours).
  No public code found → reimplement from verified paper text. Exploratory Phase 3
  (`pilot1_analyze.py`) already run: probe within-trace AUC 0.75–0.79 (layer 36 best),
  cosine within-trace ≈ chance (confound-controlled), poor earliness under fixed CUSUM —
  motivating this faithful pass before any conclusion about the paper's mechanism.
