# Paper A — loop-space positioning (Phase-3 kill-test, 2026-07-30)

Read end-to-end (full HTML, not abstracts): **Circular Reasoning** (arXiv:2601.05693) and
**Word Salad Chopper** (arXiv:2511.00536) — the two closest prior works. Verdict: **core survives.**

## What both prior papers do (and do NOT do)
| Axis | Circular Reasoning (2601.05693) | Word Salad Chopper (2511.00536) |
|---|---|---|
| Datasets | LoopBench (arithmetic/recursive puzzles), AIME2025, SuperGPQA — **math/QA only** | GSM8K, MATH-500, AIME25, GPQA-Diamond — **math/QA only** |
| Any CODE (HumanEval/MBPP/APPS)? | **No** | **No** |
| Verbatim-vs-paraphrastic **fraction**? | **No** — qualitative ("lexically distinct, semantically redundant"), no % | **No** — "both exist," no breakdown |
| Evaluates parameter-free/entropy samplers (p-less, min-p, α)? | **No** (only notes temperature "no complete cure") | **No** |
| Boundary/peakedness "when does a decoder break across regimes" study? | **No** — detect+intervene only | **No** — detect+intervene only |
| Detection | CUSUM on hidden-state classifier; EDR 0.64–0.76 (math) | linear probe on `\n\n` hidden states; 92–98 AUROC (math) |
| Action | (precursor prediction) | chop + regenerate; MATH-500 90.8→89.6 recovery |

## What is genuinely OURS (uncovered by the two closest papers)
1. **The CODE setting** — reasoning loops on APPS competitive-programming, not math/QA. Fully uncovered.
2. **A quantified verbatim-vs-paraphrastic fraction on code** (~40–47%, `docs/loop_collapse_internal_state_findings.md`, `docs/two_week_summary_2026-07-02.md`). Neither prior paper reports any such fraction → the "last C2 slice" the lit agent worried about **survives**.
3. **The sampler/prevention angle** — which decoder *causes* vs *prevents* loops (parameter-free vs temperature; the α knob). Both prior papers are detection-only and never touch the sampler.
4. **The boundary/peakedness characterization** across model/task regimes (parameter-free helps on short base-model code, breaks on long CoT). Neither prior paper does this.
5. **A precise negative-transfer result** (see correction below).

## Required correction to Narrative A (drop the falsified sub-claim)
- **DROP:** "hidden-state probes miss paraphrastic loops." **FALSE** — WSC gets 92–98 AUROC and
  Circular Reasoning 0.64–0.76 EDR catching exactly these semantic loops (on math).
- **REPLACE with the verified, sharper claim:** *the published hidden-state precursor
  (2601.05693's CUSUM) does not transfer to our code-CoT setting* — our replication
  (`docs/pilot1_findings_hidden_state_detection.md`, `docs/pilot1_circular_reasoning_replication.md`)
  measured EDR 0.17–0.20 vs the paper's 0.64–0.76; precursor absent before onset. This is a
  *transfer-failure on code*, not "probes categorically miss it" — and it is a genuine finding,
  because both prior detectors were only ever validated on math/QA.

## Net effect on the plan's hard-truth #3
Hard-truth #3 ("loop taxonomy partly scooped; salvageable only as a code fraction") was **too
pessimistic** — it rested on abstract-only reads. End-to-end, the code setting, the fraction, the
sampler/prevention angle, and the boundary study are all uncovered by the two nearest papers. The
loop material can be a genuine (if narrow) **contribution**, not merely motivation. Cite 2506.10979
for "escape is hard" and both papers for "semantic loops exist + are catchable on math"; our delta
is code + fraction + sampler-prevention + transfer-failure.
