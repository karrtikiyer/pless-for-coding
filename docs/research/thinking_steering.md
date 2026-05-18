# Steering the `<think>` block of a reasoning model

How to force a reasoning model (Qwen3-8B in thinking mode, DeepSeek-R1,
QwQ, R1-Distill, etc.) to follow a specific *structure* inside its
`<think>...</think>` trace — e.g. emit pseudocode first, then translate
to Python.

## TL;DR

**Yes, this is possible, and prefix-injection inside `<think>` is by far
the cheapest thing to try first.** With Qwen3-8B served by vLLM you can
use the OpenAI-compatible `continue_final_message=true` parameter (or
`prefix=true` in the DeepSeek/Anthropic style) to literally seed the
assistant turn with `<think>\nPseudocode:\n` and let the model continue.
No fine-tuning, no logits hacking, no custom kernels. Empirically the
second-cheapest lever — putting an explicit *thinking instruction* in
the system prompt ("first write Python-like pseudocode in your reasoning,
then translate") — also works on Qwen3/DeepSeek-R1, though recent
CoT-controllability papers show RL-trained reasoners obey
thinking-style instructions much less reliably than they obey
final-answer instructions. The fancy stuff (grammar-constrained
thinking, steering vectors, process-reward training) is mostly
diminishing returns for the pseudocode-first goal.

## 1. Prefix-injection / partial-think prefilling

How it works: include an `assistant` turn whose content ends mid-think
(e.g. `"<think>\nPseudocode:\n"`) and tell the server "do not start a
new turn, continue this one." The model emits tokens that conditionally
extend the seed.

- **vLLM**: supports this directly via `continue_final_message=true` on
  `/v1/chat/completions`; cannot be combined with
  `add_generation_prompt`.
  ([vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))
- **DeepSeek API / LiteLLM**: `prefix=true` on the last assistant
  message — explicitly endorsed for prefilling thoughts on R1.
  ([LiteLLM prefix docs](https://docs.litellm.ai/docs/completion/prefix),
  [DeepSeek-R1 HF discussion #158](https://huggingface.co/deepseek-ai/DeepSeek-R1/discussions/158))
- **SGLang/TGI/llama.cpp**: all support continuation via raw
  `/completions` style; just construct the templated prompt yourself
  ending inside `<think>`.
- **Qwen3 specific**: default chat template auto-emits `<think>` after
  the generation prompt; you can override by templating yourself and
  ending the prompt with the prefix you want. Note Qwen3 needs both
  opening *and* closing `<think>` tokens handled correctly
  (vLLM bug [#27118](https://github.com/vllm-project/vllm/issues/27118)).

There's no formal academic name for the technique; community calls it
"prefilling," "prefix completion," or "assistant prefill."

**Maturity:** production-grade. **Applicable to local Qwen3-8B/vLLM:**
yes, trivially. **Works for pseudocode-first:** yes — this is the
right primary lever.

## 2. System / user prompt steering of thinking

How it works: instruct the model *about how to think* rather than what
to answer ("Before writing code, in your `<think>` block, draft
pseudocode using only `for`, `if`, `while`, and function calls"). The
Qwen3 docs explicitly support inline `/think` and `/no_think`
directives, suggesting the family is trained to follow thinking-mode
directives.
([Qwen quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html))

The sobering finding: **CoT controllability is much weaker than output
controllability.** "Reasoning Models Struggle to Control their Chains
of Thought" (arXiv [2603.05706](https://arxiv.org/html/2603.05706v1))
reports Claude Sonnet 4.5 controlled its CoT only 2.7% of the time vs.
61.9% for the final output, and finds controllability *decreases* with
RL pressure. "Effectively Controlling Reasoning Models through Thinking
Intervention" (Wu et al., arXiv [2503.24370](https://arxiv.org/abs/2503.24370))
is the strongest positive result: strategically *inserting* tokens
inside the thinking trace (a souped-up version of #1) materially raises
instruction-following on IFEval / SEP / SORRY-Bench. The takeaway is to
combine #1 and #2: prompt-only steering is unreliable, but a
system-prompt directive *plus* a `<think>` prefix is roughly the
Thinking Intervention recipe.

**Maturity:** paper + production. **Applicable:** yes. **Works for
pseudocode:** moderately, much better when paired with prefill.

## 3. Constrained decoding inside the think block

How it works: enforce a CFG / regex over generated tokens via XGrammar,
Outlines, Guidance, or lm-format-enforcer.

- XGrammar (arXiv [2411.15100](https://arxiv.org/abs/2411.15100)) is
  the default backend in vLLM / SGLang / TensorRT-LLM, ~40µs/token.
- Caveat consistently reported: **rigid grammars hurt reasoning-heavy
  tasks.** The vLLM team's own guide
  ([blog.vllm.ai 2025/01/14](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html))
  recommends a hybrid — free-form scratchpad + grammar only on the
  final answer.

**Maturity:** production. **Applicable to Qwen3-8B/vLLM:** yes
(`guided_grammar`/`guided_regex`). **Works for pseudocode:** not
recommended — pseudocode is fuzzy by definition; a strict grammar will
both be hard to write and degrade reasoning. Use a *loose* anchor at
most ("must start with `Pseudocode:` and contain `Python:` before
`</think>`") via a regex on the *opening* of the think block, not the
body.

## 4. Process-reward / verifier-driven thinking

- **rStar-Math** (arXiv [2501.04519](https://arxiv.org/abs/2501.04519)):
  MCTS with a process-preference model trained on per-step Q-values;
  lifts Qwen2.5-Math-7B from 58.8 → 90.0 on MATH. Code-augmented CoT
  data synthesis is core. Heavy infra; not appropriate for a quick
  experiment.
- **Quiet-STaR / STaR / Step-DPO**: train the *style* of thinking via
  SFT/DPO on curated traces.
- **s1: Simple test-time scaling** (arXiv [2501.19393](https://arxiv.org/abs/2501.19393)):
  "budget forcing" — append `Wait` to extend thinking, or inject
  `Final Answer:` to stop. A cousin of prefix injection that works at
  the *end* rather than the start; potentially useful to *stop* the
  pseudocode phase and start coding.
- **Steering vectors** ("Understanding Reasoning in Thinking Language
  Models via Steering Vectors," arXiv [2506.18167](https://arxiv.org/abs/2506.18167)):
  linear activation edits on DeepSeek-R1-Distill control backtracking
  / uncertainty / example-testing. No public "pseudocode direction"
  yet; you'd have to extract one.

**Maturity:** paper / research code. **Applicable to local Qwen3-8B:**
feasible but expensive. **Works for pseudocode-first specifically:**
overkill unless you plan to ship.

## 5. Plan-and-Solve / Pseudocode / Program-of-Thoughts lineage

PAL (arXiv [2211.10435](https://arxiv.org/pdf/2211.10435)),
Program-of-Thoughts ([OpenReview](https://openreview.net/forum?id=YfZ4ZPt8zd)),
Chain-of-Code ([arXiv 2312.04474](https://arxiv.org/html/2312.04474)),
Structured Chain-of-Thought (Li et al.,
arXiv [2305.06599](https://arxiv.org/abs/2305.06599)), and Mishra
et al.'s pseudocode-prompting (cf. "Training with Pseudo-Code for
Instruction Following," arXiv [2505.18011](https://arxiv.org/html/2505.18011v1))
all show that asking a *non-reasoning* model to write pseudocode /
program-skeletons before code yields large gains (CoC +12% over CoT on
BBH; SCoT +13.8% pass@1). Interestingly SCoT outperformed its
pseudocode variant (SCoT-P) by leaning on three control structures
rather than freeform pseudocode — relevant if you care about the
*shape* of the pseudocode.

How well it transfers to *modern* reasoning models: partly obsolete in
that R1-style models already do plan-first internally, but the
*prompting wording* from these papers is gold for crafting your
prefill — e.g. SCoT's "design with sequence/branch/loop" phrasing is a
ready-made `<think>` seed.

**Maturity:** production prompting recipes. **Applicable:** yes, free.
**Works for pseudocode-first:** yes — these are essentially the
templates you'll inject.

## 6. Recent (2024–2026) work on controllable thinking for R1-style models

- **Thinking Intervention**
  (arXiv [2503.24370](https://arxiv.org/abs/2503.24370)) — the central
  reference; explicit token insertion inside `<think>`.
- **D-CoT: Disciplined CoT**
  (arXiv [2602.21786](https://arxiv.org/pdf/2602.21786)) — trains
  "mode of thought" with control tags as scaffolding.
- **"Understanding and Steering the Cognitive Behaviors of Reasoning
  Models at Test-Time"**
  (arXiv [2512.24574](https://arxiv.org/html/2512.24574v2)) — CREST
  identifies cognitive attention heads for targeted edits.
- **"Base Models Know How to Reason, Thinking Models Learn When"**
  (arXiv [2510.07364](https://arxiv.org/html/2510.07364v1)) — argues
  most reasoning capability is already in the base model; the
  thinking-mode RL only learns *when* to deploy it. Implication:
  prefilling the think block can unlock plan-first behavior that's
  already latent.
- **Interactive Reasoning / Hippo**
  (arXiv [2506.23678](https://arxiv.org/html/2506.23678v1)) — UI-level
  steering of subtrees of the CoT.

## Recommended next steps for our Qwen3-8B split-decoding setup

1. **Prefix-injection (technique #1).** Add
   `continue_final_message=true` and seed `<think>\nPseudocode:\n` (or
   SCoT's three-structure phrasing) as the assistant prefix. Cheapest,
   most reversible, and exactly what Thinking Intervention reduces to
   in practice. Couples naturally with our split sampler: keep the
   high-temp `pless` sampler for the post-`Pseudocode:` body, switch
   to the code sampler when we detect `</think>` or a `Python:`
   anchor.
2. **System-prompt instruction-about-thinking (technique #2)** layered
   on top of #1. Borrow SCoT's wording
   (arXiv [2305.06599](https://arxiv.org/abs/2305.06599)).
3. **Budget forcing from s1 (technique #4 lite)** as an optional
   second injection point: when the model tries to close `</think>`
   before producing pseudocode, append a
   `Wait, write the pseudocode first:` continuation. Or inject
   `Python:\n\`\`\`python\n` to force the transition from pseudocode
   to code. Zero training cost, complementary to the split sampler.

Skip grammar-constrained thinking, PRM training, and steering-vector
extraction for v1 — none of them target the "pseudocode first" use
case better than the three above, and all are substantially more
expensive.

## Subtleties worth knowing

- **"Base Models Know How to Reason, Thinking Models Learn When"**
  (arXiv [2510.07364](https://arxiv.org/html/2510.07364v1)) argues
  plan-first reasoning is *latent* in the base — RL just teaches the
  model when to invoke it. So prefilling can unlock dormant
  pseudocode-first behavior that wasn't being triggered.
- **"Understanding and Steering Cognitive Behaviors at Test-Time"**
  (arXiv [2512.24574](https://arxiv.org/html/2512.24574v2)) identifies
  specific attention heads for cognitive modes — if simple prefilling
  works well, this would be the natural follow-up for a paper.

## Sources

- [vLLM OpenAI-Compatible Server (continue_final_message / add_generation_prompt)](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/)
- [vLLM issue #27118 — Qwen3-Thinking think-token parsing](https://github.com/vllm-project/vllm/issues/27118)
- [LiteLLM Pre-fix Assistant Messages](https://docs.litellm.ai/docs/completion/prefix)
- [DeepSeek-R1 HF discussion #158 — prefix complete](https://huggingface.co/deepseek-ai/DeepSeek-R1/discussions/158)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen quickstart — /think and /no_think directives](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Prompt Injection and Mode Drift in Qwen3 (security analysis)](https://blog.lukaszolejnik.com/prompt-injection-and-mode-drift-in-qwen3-a-security-analysis/)
- [Wu et al., "Effectively Controlling Reasoning Models through Thinking Intervention" (arXiv 2503.24370)](https://arxiv.org/abs/2503.24370)
- ["Reasoning Models Struggle to Control their Chains of Thought" (arXiv 2603.05706)](https://arxiv.org/html/2603.05706v1)
- [Arcuschin et al., "Understanding Reasoning in Thinking Language Models via Steering Vectors" (arXiv 2506.18167)](https://arxiv.org/abs/2506.18167)
- ["Understanding and Steering Cognitive Behaviors of Reasoning Models at Test-Time" (arXiv 2512.24574)](https://arxiv.org/html/2512.24574v2)
- ["Base Models Know How to Reason, Thinking Models Learn When" (arXiv 2510.07364)](https://arxiv.org/html/2510.07364v1)
- ["D-CoT: Disciplined Chain-of-Thought Learning" (arXiv 2602.21786)](https://arxiv.org/pdf/2602.21786)
- [Muennighoff et al., "s1: Simple test-time scaling" (arXiv 2501.19393)](https://arxiv.org/abs/2501.19393)
- [Guan et al., "rStar-Math" (arXiv 2501.04519)](https://arxiv.org/abs/2501.04519)
- [Dong et al., "XGrammar" (arXiv 2411.15100)](https://arxiv.org/abs/2411.15100)
- [vLLM Blog — Structured Decoding (Jan 2025)](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- [Gao et al., "PAL: Program-aided Language Models" (arXiv 2211.10435)](https://arxiv.org/pdf/2211.10435)
- [Chen et al., "Program of Thoughts Prompting"](https://openreview.net/forum?id=YfZ4ZPt8zd)
- [Li et al., "Chain of Code" (arXiv 2312.04474)](https://arxiv.org/html/2312.04474)
- [Li et al., "Structured Chain-of-Thought Prompting for Code Generation" (arXiv 2305.06599)](https://arxiv.org/abs/2305.06599)
- ["Training with Pseudo-Code for Instruction Following" (arXiv 2505.18011)](https://arxiv.org/html/2505.18011v1)
- [Interactive Reasoning / Hippo (arXiv 2506.23678)](https://arxiv.org/html/2506.23678v1)
