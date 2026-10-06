# Prior review learnings — from Paper B collaborator review

Extracted from Claude Code session `6387d886-890c-473e-9f8d-8bc007436067` (2026-09-17). This is a **durable record of the 21 Paper B review comments and their resolutions**. The on-disk RTC template (`docs/paperB_response_to_comments.md`) was never populated; this file fills that gap.

**Why it matters for LLM4Code**: same p-less authors will review this paper via Overleaf. Two of the 21 comments came from a p-less author directly (#4, #10–11); the rest from another collaborator. Every pattern below is a landmine to preempt in v1 of the LLM4Code paper.

## The 21 comments (final state — all addressed)

| # | Anchor | Comment | Author | Resolution | Status |
|---|---|---|---|---|---|
| 1 | Abstract / §1 / §Setup — "first empirical evaluation at k≠2" | The original p-less paper evaluated a range of ks. | collab | Scoped to "first empirical evaluation on code generation with reasoning models"; dropped "origin reports no experiments at k≠2." | addressed |
| 2 | "calibration-free" (G_k) | What is the reason for specifically mentioning 'calibration-free'? | collab | Replaced 4 occurrences → "single-parameter" (G_k has tuned k). Kept "hyperparameter-free" only for default p-less. | addressed |
| 3 | Abstract L47 / §1 L76–77 — repair framing | Why attribute repair only to DeepSeek and not Qwen? | collab | Unified: single-parameter repair on both; also "beats swept temp" → tie. | addressed |
| 4 | §1 contributions — "A decoder-side mechanism" | We should link this to reasoning models. | **p-less author** | Bullet reframed as decoder×model interaction on reasoning-model CoT (RL-trained loop-propensity; decoder triggers, doesn't create). | addressed |
| 5 | §1 L73 / §ablation L289,301,412 — "borderline" | What are borderline problems? | collab | Replaced → "partially-solved (moderate-difficulty), 0<pass@1<1 at default"; defined via difficulty strata. | addressed |
| 6 | §Setup L120 — "our loop proxy" | omit/rephrase | collab | Deleted gloss; non-termination defined directly (truncation ≠ loop). | addressed |
| 7 | §Setup / cb_div (L137) — "configuration" | What is defined as a configuration? | collab | Defined: one decoder+settings combo = one Table 1 row. | addressed |
| 8 | §Setup L137 — "cross-configuration" | Cross-configuration is mentioned in several places; explain more. | collab | Rephrased: cb_div not directly comparable across configs (differing correct subsets) → reported per-config. | addressed |
| 9 | §Failure L175 — "slow decline into repetition" | Not clear; is this true or rephrase? | collab | Tightened to measured fact: loops without closing thinking → hits token cap. Dropped "repetition." | addressed |
| 10–11 | §Failure L172–173 — "pathology structural to p-less threshold" | Contradicts §mech; also p-less-norm also loops. | **p-less author** | Reworded to "triggered by the collision-probability threshold — arises for both p-less and p-less-norm but not temperature." | addressed |
| 12 | §Failure L174 — "secondary truncation" + generality | What is secondary truncation? This would also happen for min-p/greedy/beam? | collab | "secondary truncation" → "top-p/top-k truncation"; added hedged §mech sentence generalizing to mode-concentrating decoders (cite Holtzman for greedy/beam; min-p as prediction). | addressed |
| 13 | Table 1 caption — "cross-verified…" + "zero-tuning" | Cross-verified against stored values not clear; is 'zero-tuning' correct? | collab | Dropped internal QA phrase; dropped "zero-tuning" label (G_k has a tuned knob; best-G_k symmetric to swept-T). | addressed |
| 14 | Table 1 caption — verbosity + internal-tooling text | Too verbose; do 'regenerated live' etc. matter for submission? | collab | Cut internal QA phrases and "zero-tuning"; moved per-model claims to §Positioning; corrected swept-temp overclaims. | addressed |
| 15 | §mechanism L227 — "plain sampling" / Fig 1 bar | Which bar is plain sampling in Fig 1; is 'plain sampling' correct? | collab | "plain sampling" → "temperature with no top-p/top-k"; cite §Failure matched-T result (19.0% vs 0.4% @T0.6) instead of Fig 1. | addressed |
| 16 | §mechanism L238 / Conclusion L345 — "lets temperature escape" | What is the meaning of temperature escape? | collab | Reworded to "the sampling diversity that would otherwise let the model break out of the loop." | addressed |
| 17 | §7 L245 — "flattens the survivor set" | Is 'flattens' correct here? | collab | "flattens" → "widens" (lowering k lowers threshold → larger survivor set). | addressed |
| 18 | §7 L251–253 — cb_div + "no per-task tuning" | Explain why so? / does 'no per-task tuning' need clarifying? | collab | Dropped self-contradictory cb_div clause and "with no per-task tuning" (vacuous + re-implies tuning-freeness). | addressed |
| 19 | §7 — "swept temperatures" | What temperatures have we swept? | collab | Only T∈{0.6, 1.0} run; sweep was top-p. Rephrased "swept temperature" → "best-swept top-p setting"; † = best of top-p sweep. | addressed |
| 20 | §7 token line — "indistinguishable from temperature" | What do we mean by indistinguishable from temperature? | collab | Softened to "comparable … similar to the temperature configurations" (no significance test implied). | addressed |
| 21 | §7 token paragraph — "most token-efficient / Qwen tie" | Qwen G_k–temp token gap ~ DeepSeek's; is the framing right? | collab | "most token-efficient decoder we test" is false (top-p 0.8/0.9, α4/5 use fewer). Reframed: token saving vs looping default; vs tuned temp/top-p, comparable on both. | addressed |

## Patterns to preempt in the LLM4Code draft

**Author-sensitive (p-less author will push back)**:
- ❌ "p-less breaks / fails / has a pathology" → ✅ "the collision-probability threshold interacts with RL-trained peaked distributions"
- ❌ frame loops as intrinsic to the sampler → ✅ frame as decoder×model interaction (decoder triggers, model creates the propensity)
- Every decoder claim in the CoT sections must link to the reasoning-model behavior explicitly

**Unhedged superlatives (collaborator will call these out)**:
- ❌ "first empirical evaluation" without scope → ✅ "first evaluation on X, following the direction of Appendix B.5"
- ❌ "best-in-class" → ✅ "recovers the loss the α=2 threshold imposes"
- ❌ "beats swept X" → check the "swept" claim; if only 2 values were run, say so
- ❌ "most token-efficient" → check across ALL tested decoders including α arms + top-p sweep
- ❌ "indistinguishable from" (implies stat test) → "comparable to"

**Undefined jargon (call it out and define once, then use consistently)**:
- ❌ calibration-free, zero-tuning, plain sampling, temperature escape, secondary truncation, borderline, cross-configuration, loop proxy, flattens
- ✅ hyperparameter-free (default p-less only), single-parameter (G_k), partially-solved (moderate-difficulty), etc.

**Claims about untested decoders**:
- ❌ "min-p / greedy / beam would also fail here" as a result
- ✅ "min-p / greedy / beam would also fail here" as a **prediction**, cited to Holtzman for the mode-concentration mechanism

**Origin-paper attributions to verify** (do NOT trust memory):
- Whether the origin paper evaluated k≠2 (comment #1 was a factual error we made)
- Sampler-formula exactness (already checked in task #8 — clean)

**Sweep terminology**:
- Only say "swept X" if 3+ values were run. For 2 values say "T∈{0.6, 1.0}" explicitly.

## When drafting LLM4Code prose

Before writing any section, re-read the relevant preempt list above. Do a search-and-verify pass at the end of §3, §4, §5 for each of these patterns.

## Overleaf-comment tracking (this time)

Fill `RTC_TEMPLATE.md` **as comments arrive**, not after. Paper B's failure was letting comments live only in the Overleaf Review panel (which can't be exported cleanly) and hoping to reconstruct later.
