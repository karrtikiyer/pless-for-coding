# LLM4Code Paper — Response To Comments (RTC)

**Reviewers**: Runyan Tan, Shuang Wu, Phillip Howard (arXiv:2509.23234 authors, ongoing collaboration). Shared via Overleaf comment-only access.

**Purpose**: Durable, version-controlled record of every reviewer comment on the Overleaf draft and our response — the "working history" of comments ↔ responses. This file does NOT depend on Overleaf's comment anchors (which break on rewrites), so it survives regardless of how the text is migrated. Circulate alongside each new draft version.

**Format learned from**: `docs/paperB_response_to_comments.md` (Paper B review round; only ~2 rows filled — this time we fill it consistently as comments arrive).

## Draft version tracking

- **v1** — first draft shared: [FILL date]
- **v1.1** — [FILL summary of changes]
- **v2** — [FILL if a full-rewrite becomes necessary]

## Workflow (when comments arrive)

1. **Catalogue** every Overleaf comment into the table below — one row each. Overleaf's Review-panel copy is the source of truth; text goes in **verbatim**, no paraphrasing.
2. **Triage**: does the anchored text still exist in the current draft?
   - `stable` → section unchanged; address in place.
   - `rewritten` / `removed` → section restructured / dropped; still needs an explicit response.
3. **Respond**: fill "Our response" + "Addressed in (new §)". Ground every response in a source doc / metrics JSON / verified citation — never from memory.
4. **Sync**: in Overleaf, reply in-thread pointing to the RTC id ("addressed — RTC #N, §5") and resolve the comment.

## Status legend

`open` (not yet answered) · `addressed` (edited + responded) · `wontfix` (declined, with reason) · `discuss` (needs synchronous discussion) · `moot` (no longer applies after a rewrite; state why).

## Triage tally

- total comments: [FILL] · stable: [FILL] · rewritten/removed: [FILL] · addressed: [FILL] · open: [FILL]

## Anticipated comment categories (fill in as they arrive; use to pre-tune the draft)

Because the reviewers are the p-less authors themselves, expect scrutiny on:
- **Sampler definitions** — any deviation from their paper's formulas
- **G_k Rényi (Appendix B.5)** — is our implementation faithful to their proposal?
- **Attribution** — proper credit for both the vanilla sampler AND the B.5 generalization
- **"Breaks on reasoning models" framing** — reasonable request for softer language / conditions
- **Reproducibility** — configurations, seeds, tokenizer notes (esp. DeepSeek tokenizer-fix asymmetry)
- **Missing citations** — related decoding literature they know that we may have missed

## Comments

| # | Overleaf § / anchor (short quote) | author | comment (verbatim) | triage | our response | addressed in (new §) | status |
|---|---|---|---|---|---|---|---|
| 1 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | open |

<!-- Add one row per comment. Keep author + verbatim text; do not paraphrase. Update status as you resolve. -->

## Reviewer cover note (draft — polish before sending)

> Dear Runyan, Shuang, Phillip,
>
> Attached is our v1 draft for LLM4Code 2027, evaluating p-less on code generation. Three findings:
> (i) on MBPP + HumanEval, p-less matches greedy pass@1 without a temperature knob and sits on the pass@1 axis of the Pareto frontier;
> (ii) default p-less at α=2 saturates on RL-trained reasoning models (Qwen3-8B, DeepSeek-R1-Distill-Llama-8B) under CoT, driving 14-42% of traces into non-terminating loops;
> (iii) your Appendix B.5 G_k generalization directly controls this: k=0.1 on Qwen3-8B and k=0.35 on DeepSeek recover +9pp pass@1 with truncation eliminated. This is, to our knowledge, the first empirical evaluation of B.5.
>
> We would value your comments on: (a) sampler-definition fidelity, (b) framing of the reasoning-model finding, (c) any related work we may have missed.
>
> [FILL — timeline for feedback ask]
>
> Thank you,
> [FILL — authors]
