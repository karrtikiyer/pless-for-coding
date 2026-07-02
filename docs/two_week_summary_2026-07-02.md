# Two-week research summary (2026-06-23 → 07-02) — reasoning-model loops: detection, prevention, and a missing loop class

**Audience:** fellow AI researchers. **Thesis:** early *detection* and p-less *prevention* of
reasoning-model loops both fail on **naturalistic** code-reasoning loops — and the mechanistic reason
is a **paraphrastic-drift loop class** (~41–47% of truncations) that prior loop benchmarks don't
measure.

**Common setup** (unless noted): Qwen3-8B and DeepSeek-R1-Distill-Llama-8B; APPS ATCODER-interview,
252 tasks × 10 samples; p-less "think" sampling at T=1.0; 32768-token budget; loop onset = validated
streaming n-gram detector (n=30, k=6). Every number below is sourced to a committed doc.

---

## 1. Output-distribution signals give no early warning `[Certain, exploratory]`
Per-token collision probability (Σpᵢ²) and Shannon entropy do **not** separate looping from clean
reasoning (AUC ≈ 0.5) — the model is already confident before it loops, so nothing "surges."
*Source:* `scripts/signal_diagnostic.py` (commits 2026-06-23/24).

## 2. The published hidden-state loop-*precursor* detector does not generalize `[Certain within scope]`
Faithful re-implementation of Circular-Reasoning (arXiv:2601.05693) — last-layer linear probe + CUSUM
**and** the K-means cluster-periodicity precursor — on 90 Qwen3-8B ATCODER traces (30 terminal / 30
transient / 30 clean):
- Detector catches only **EDR 0.17–0.20** of terminal loops early (lower for transient), vs the
  paper's **~0.64–0.76 on synthetic LoopBench** (3–4× worse).
- The precursor ("semantic circularity precedes textual repetition") is **absent before onset**:
  pre-onset periodicity AUC **0.56 / 0.39** (≈ chance), firing *at or after* the textual onset
  (negative lead).

*Source:* `docs/pilot1_findings_hidden_state_detection.md` (commits 2026-06-26). Numbers are
directional, not head-to-head with their Table 3 (different onset definition + data).

## 3. p-less α-tuning fixes DeepSeek looping but does not beat standard decoding — incl. diversity `[Certain, DeepSeek only]`
Full 9-config comparison, same 252×10:
- Hyperparameter-free **pless (α=2) / pless_norm loop catastrophically**: ~65% of samples truncate,
  pass@1 = 0.17.
- Raising Rényi-α fixes looping (truncation 64.9% → 4.5% @α4 → 1.2% @α5), recovering pass@1 to ~0.30–0.32.
- **`temp T1.0 top_p0.95` ≥ the best pless-α arm on all six axes at once**: pass@1 (0.328 vs 0.315),
  pass@5, pass@10, cov@0.3/0.5, **and both diversity metrics** (struct_div 0.576 vs 0.527; cb_div
  0.619 vs 0.583) — at 0.1% truncation.

So p-less's earlier "+14pp" was *recovery from its own pathology to parity-minus*, not an advantage;
its hyperparameter-free default is the worst of the nine. *Source:*
`docs/deepseek_alpha_crossmethod_findings.md`. Caveat: Qwen head-to-head still open (todos A39).

## 4. Circular-Reasoning "internal-state collapse" (Fig 3b/4): endpoint universal, transient is a model property `[Likely; n=5 case studies/model]`
Hand-selected reflective statement loops, GPU teacher-forced (all layers):
- **Endpoint (the paper's universal claim) holds for both models** — the loop is a separable rigid
  state: settled cross-cycle cosine ≫ normal baseline (DeepSeek 0.997 vs 0.90; Qwen 0.999 vs 0.96).
  Qwen loops *are* internally detectable, consistent with the paper's Table-3 EDR 0.64 for Qwen3-8B.
- **The transient differs and survives confound controls** — DeepSeek collapses *gradually*
  (last-layer cosine R1 0.795 → R5 0.997; L₂ 74.4 → 9.5), Qwen *instantly* (R1 ≈ 0.995). Onset-depth
  confound **ruled out** by a matched early-onset Qwen set (still instant, R1 0.995); anchor-token
  confound reduced (10 anchors, all instant). Both models lock to *exact verbatim* within ~1 cycle
  (verified from text) — so this is not a measurement artifact.

*Source:* `docs/loop_collapse_internal_state_findings.md` (commits 2026-06-30→07-02). Open hardening:
block-average over the full period; a DeepSeek completed-recovered-loop control.

## 5. Unifying result: ~half of real reasoning-loop truncations are *paraphrastic semantic drift* `[Certain, full 252×10]`
Categorizing **every** truncation (`loop_collapse_categorize.py`, `truncation_taxonomy.json`):

| failure mode | Qwen | DeepSeek | caught by verbatim / hidden-state collapse? |
|---|---|---|---|
| Verbatim statement loop (periodic) | 40.7% | 49.8% | ✅ |
| **Paraphrastic / semantic drift** (n-gram fires but *aperiodic*) | **41.3%** | **46.8%** | ❌ |
| Short/degenerate (incl. true digit-run "numerical" = 3.3% / 0.4%) | 9.9% | 1.3% | partial |
| No detected loop | 8.2% | 2.3% | ❌ |

The paraphrastic class is validated real — best periodicity self-match median **0.23 / 0.26** (74% /
73% below 0.5). The model re-explores the *same idea in drifting words*, so there is no recurring unit
and **no identical token across cycles** → invisible to verbatim n-gram detection *and* to the Fig-4
collapse (which requires identical tokens to compute).

**Why this reconciles everything:** LoopBench defines loops by *unit recurrence* (Numerical `k·l>500`;
Statement `k>3`) — verbatim by construction — so paraphrastic drift isn't labeled a loop there, and
its "semantic circularity" is only a *precursor* to verbatim repetition. On naturalistic loops ~half
**never converge to verbatim** (they drift to the token cap), so there is no textual onset for a
precursor to anticipate — mechanistically explaining finding #2 (their 0.64 → our 0.17–0.20).

**Control (full 252×10):** the verbatim statement loop is failure-specific — 138/138 (Qwen) /
734/739 (DeepSeek) truncated; **0 completed+correct Qwen** contain one (DeepSeek: 3). But *transient*
repetition is common and non-fatal (68/54 completed+correct traces tripped the n-gram detector and
still solved). The *sustained terminal* loop is the failure signal, not repetition per se.

*Source:* `docs/loop_collapse_internal_state_findings.md` (taxonomy + LoopBench reconciliation).

---

## Bottom line for the group
Across three independent signal families (output distribution, hidden-state precursor, realized-text
taxonomy), **early detection of naturalistic code-reasoning loops yields no actionable lead time**,
and **p-less prevention doesn't beat standard decoding**. The mechanistic culprit is a
**paraphrastic-drift loop class (~41–47% of truncations)** that is (a) invisible to verbatim and
hidden-state methods, and (b) structurally outside the current loop-benchmark literature (LoopBench =
verbatim; PRMBench = step-level reward-model eval; overthinking work = efficiency framing).

And the *escape* side is independently hard, per literature `[Certain, verified]`: once a loop locks
in, *"normal decoding cannot escape"* — LoopGuard (arXiv:2604.10044) shows escape needs active
KV-cache intervention (detect-onset → prune the repetitive tail); the Circular-Reasoning paper calls
loops an "inescapable" self-reinforcing V-shaped-attention cycle; self-reinforcement is long
established (RIRO, NeurIPS 2023). Our data agrees (clean loops escape <1% within budget). **But** all
these escape mechanisms assume *identical recent tokens* (attention/KV amplification of repeats), so
the one viable verbatim pipeline — detect-established-loop → KV-prune — **cannot touch the paraphrastic
half** (no repetitive tail). So detect-and-escape is *doubly* hard on real code loops: late detection
*and* an escape mechanism that only addresses the verbatim ~half.

## Scope / limitations (state these when presenting)
- Finding #4 = **n=5 case studies/model**; #2 = 90 traces; #3, #5, control = full 252×10.
- Two models, one dataset (ATCODER-interview), one sampling config (p-less T=1.0).
- Cross-method #3 is DeepSeek-only (Qwen A39 pending).
- The PRMBench "Non-Circular Logic = semantic" positioning claim is **not yet primary-source-confirmed**.

## Open questions / next steps
- **A39:** Qwen3-8B 9-config cross-method + diversity table (is the "p-less ≤ standard decoding"
  verdict general or DeepSeek-specific?).
- **Anchor-confound final close:** block-average Fig-4 cosine/L₂ over the full repeating period.
- **Recovered-loop control:** extract DeepSeek's 3 completed+correct statement loops; do they collapse
  *less* (→ collapse is diagnostic of being terminally stuck) or the same (→ tautology of repetition)?
- **Detection of paraphrastic drift:** the open problem — a *semantic* (embedding-space) circularity
  signal, since verbatim + hidden-state-collapse both miss it.
- **Confirm PRMBench NCL** from the primary source before citing.

## Provenance
Docs: `docs/{pilot1_findings_hidden_state_detection,deepseek_alpha_crossmethod_findings,loop_collapse_internal_state_findings}.md`.
Code: `scripts/{signal_diagnostic,pilot1_*,loop_collapse_{screen,extract,plot,control,categorize}}.py`.
Data: `results/{pilot1_hidden,loop_collapse_replication,pless_cot_efficiency_vllm,pless_recovery_full252}/`.
Commits: 2026-06-23 `fdd4160` → 2026-07-02 `a69b341` (branch `feat/vllm-backend`).
