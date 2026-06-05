# Detecting a rambling reasoning trace & forcing early termination — inference-time methods

**Date:** 2026-06-05
**Status:** literature survey (for [[A27]] / new A28). Motivated by the CoT-efficiency
finding that pless/pless_norm fail to terminate on hard problems — **37%/36% truncation on
DeepSeek-R1-Distill APPS introductory**, dragging pass@1 to ~0.39 (worst of 6 configs),
vs ~14% on Qwen3-8B interview. See
`results/pless_cot_efficiency_deepseek/.../ATCODER_introductory/analysis/cot_efficiency_apps_report.md`.

## Provenance (read before trusting any number)

Produced by the deep-research workflow run `wf_0bdc3580-c14` (2026-06-05): 5 search angles,
21 sources fetched, 102 claims extracted, **top 25 adversarially verified (3 skeptic voters
each; ≥2/3 refutations kill). Result: 25/25 confirmed, 0 killed** — 22 at 3-0, 3 at 2-1
(flagged below). The workflow's final synthesis stage emitted a schema stub, so this note was
**reconstructed from the verification transcripts** — each claim below carries its source and
the verbatim supporting quote the verifier checked.

**Verification level:** adversarially verified against a supporting quote from the source. These
quotes have **not** yet been independently re-fetched per the project's literature-rigor rule —
do that for any claim before it enters the paper. Several arXiv IDs are 2026 preprints (2601–2605).

## Question

At inference time (no retraining), can we detect *during* generation that a reasoning model's
think phase has stopped progressing / is looping / won't terminate, and force `</think>` early —
cutting a stuck trace while letting a hard-but-productive one run? (Adaptive per-generation
detector, **distinct from a fixed token cap** = A27.) Survey general reasoning LLMs; code-gen
noted separately.

## Headline

The relationship between think-length and accuracy is **non-monotonic** — both under- and
over-thinking hurt — so cutting at the right point can *improve* accuracy, not just trade it.
Multiple training-free detectors exist; **two were validated on DeepSeek-R1-Distill** (our model).

## Methods, ranked by evidence strength × relevance

### 1. DEER — confidence-threshold early exit `[3-0]` — arXiv:2504.15895 — tested on R1-Distill
- **Signal:** at reasoning-transition points (`Wait` tokens), induce a trial answer; confidence =
  mean of max-token-probability over trial-answer tokens.
- **Stop rule:** if confidence C > λ (~0.95–0.97), halt and emit the conclusion.
- **Result:** on DeepSeek-R1-Distill, **CoT 31–43% shorter AND accuracy +1.7–5.7%**.
  > "reducing the length of CoT sequences by an average of 31% to 43% while improving accuracy by 1.7% to 5.7%"

### 2. Entropy-after-`</think>` probe `[3-0]` — arXiv:2509.26522 — nearly free for us
- **Signal:** periodically append `</think>` and read the **next-token entropy**; it decreases and
  stabilizes when Pass@1 plateaus.
- **Stop rule:** threshold the **variance of that entropy** under an exponential moving average.
- **Result:** **12–22% token cut, no accuracy loss** (MATH500, AIME2025).
  > "appending a stop thinking token (</think>) and monitoring the entropy of the following token... decreases and stabilizes when Pass@1 plateaus"; "reduces token usage by 12 - 22% without harming accuracy"

### 3. CoDE-Stop — confidence-dynamics, explicitly catches looping — arXiv:2604.04930
- **Signal:** intermediate-answer confidence dynamics. Two triggers: a **confidence threshold**
  (reliable answer) **OR a "degeneration score" Dₖ** that accumulates under persistent confidence
  oscillation = *unproductive looping*. `[mechanism 3-0; "preserves accuracy" 2-1]`
- **Stop rule (verbatim):** stop at step k if `c_k ≥ r_k` (ramping threshold) or `D_k ≥ τ`;
  instability indicator `v_i = 1(2c_i − c_{i-1} < δ)`, δ=0.55, weighted `w_i = log(T_k/T_i)+1`
  (emphasize earlier steps). **Then re-prompt with the answer-generation prompt** to emit the answer.
- **Result:** 25–50% token cut, "more favorable accuracy–compute tradeoff than prior early-stopping."
  (preserve-accuracy claim was 2-1 → likely, not certain.)

### 4. SyncThink — attention-saturation `[mechanism 2-1; results 3-0]` — arXiv:2601.03649 — R1-Distill
- **Signal:** answer tokens attend weakly to early reasoning, concentrate on the `/think` token →
  "information bottleneck" read as saturation.
- **Result:** 3 R1-distilled models (GSM8K/MMLU/GPQA/BBH): **62.00% acc @656 tok vs 61.22% @2141 tok**
  full CoT (~69% fewer tokens, slight gain). **GPQA +8.1 absolute** by preventing over-thinking.
  > "62.00 percent average Top-1 accuracy using 656 generated tokens... compared to 61.22 percent, 2141 tokens... for full CoT"; "on long-horizon tasks such as GPQA... up to +8.1 absolute accuracy by preventing over-thinking"

### 5. Hidden-state probe `[3-0]` — arXiv:2504.05419
- **Signal:** a **linear probe** on hidden states verifies intermediate-answer correctness
  (calibrated); also encodes **future-answer** correctness (signal before the step completes).
- **Result:** probe-as-verifier early-exit → **24% fewer tokens, no accuracy loss.** (Needs a
  one-time probe fit — not quite "free.")
  > "use the probe as a verifier to decide whether to exit reasoning at intermediate answers... reducing the number of inference tokens by 24% without compromising performance"

### 6. EDIS — entropy-instability patterns `[3-0]` — arXiv:2602.01288
- **Signal:** `EDIS(H) = S(H)·(1+Var(H))` over the per-token entropy trajectory; flags two
  instabilities — sustained entropy rise (burst spikes), and entropy dropping to a local min then
  sharply rebounding ("false confidence followed by renewed uncertainty"). Tuned for the
  *stuck/oscillating* case.

### General confidence-awareness — arXiv:2510.08146
Shannon entropy from token logprobs as stop signal → **25–50% token savings maintaining accuracy**;
threshold is model-specific but **one-shot calibratable from a few examples**. `[mechanism 3-0;
"models know they're done early" framing 2-1]`

## The recovery half (force-stop + still get a usable answer)
- **s1 budget forcing — arXiv:2501.19393** is *bidirectional*: force-terminates when the model
  tries to continue, **and** extends (append `Wait`) — extension raised s1-32B AIME24 50→57%.
  The terminate direction is the lever; the extend direction is why a naive cut can hurt.
- Recovery that preserves answers (CoDE-Stop): after forcing `</think>`, **re-prompt with the
  answer-generation prompt** so the model emits a clean answer, not garbage.

## Honest gaps (what the survey does NOT establish)
1. **All validation is math/QA** (MATH500, AIME, GPQA, GSM8K, MMLU, BBH) — **none on code/APPS.**
   Code-gen has different entropy structure (cf. AdaDec/SWEET grounding in
   `docs/theory/entropy_mechanism_framework.md`), so transfer is unverified.
2. **Two detector flavors; our case is the harder one.** DEER / confidence-threshold catch
   "answer is *ready*." Our 37%-truncation R1-Distill failures are "never converges / rambles" —
   the **instability** case, which only **CoDE-Stop (degeneration score)** and **EDIS** explicitly
   target. Those two are best-matched, not the confidence-threshold methods.
3. **Repetition/loop detection (raw n-gram) came back thin** — 5 sources searched
   (arXiv:2511.00536, 2601.05693, 2508.17627, 2602.13935, 2504.12608) but none of their claims
   survived into the verified top-25 (verification budget went to entropy/confidence/probe). A gap,
   not a negative. The loop case surfaced via *confidence instability* (CoDE-Stop), not repetition.

## Map to our pipeline
- **pless already computes `Σpᵢ²` (collision entropy) per token** — exactly the raw material the
  entropy stoppers (2509.26522, 2510.08146, EDIS) want, so an entropy-variance / degeneration
  detector is nearly free to bolt on.
- **DEER + SyncThink were validated on R1-Distill** → strong prior that *something* works on our
  model. The unknown is gap #1: does it hold on APPS code.
- We **already have the traces** (incl. the 37%-truncation set) to prototype a
  CoDE-Stop-style degeneration detector **offline** before touching generation — see A28.

## Sources

| arXiv / URL | Method | Family | Verified claims |
|---|---|---|---|
| 2504.15895 | DEER | confidence early-exit | 3 (3-0) |
| 2509.26522 | entropy-after-`</think>` | entropy dynamics | 3 (3-0) |
| 2604.04930 | CoDE-Stop | confidence-dynamics / looping | 3 (mech 3-0, preserve 2-1) |
| 2601.03649 | SyncThink | attention-saturation | 3 (mech 2-1, results 3-0) |
| 2504.05419 | hidden-state probe | learned probe | 3 (3-0) |
| 2602.01288 | EDIS | entropy instability | 2 (3-0) |
| 2510.08146 | Shannon-entropy early stop | entropy dynamics | 3 (mech 3-0, "know" 2-1) |
| 2501.19393 | s1 budget forcing | recovery / force-stop | 3 (3-0) |
| 2511.10788, 2504.05419, github Awesome-Efficient-Reasoning-LLMs | surveys | broad | searched |
| 2511.00536, 2601.05693, 2508.17627, 2602.13935, 2504.12608 | repetition/loop | family 2 | searched, none in top-25 |
| 2506.02536, 2512.05325 | progress-stall / probes | families 3–4 | searched |

## Recommended next step
Prototype offline on existing DeepSeek truncated traces (A28): replay each trace, compute an
entropy-degeneration score per step, and check whether it would have fired *before* the cap on the
37% that truncated **without** firing on the hard-but-productive traces. Turns gap #1 (code-domain
transfer) into an experiment on data we already have — no new generation.
