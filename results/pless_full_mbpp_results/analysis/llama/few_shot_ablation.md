# Few-Shot Ablation: Do Examples Help on Llama-2-7b-hf MBPP?

Model: `meta-llama/Llama-2-7b-hf` | Dataset: Full MBPP (500 tasks, 10 samples each)

## The Controlled Experiment

Two fixes landed simultaneously in the H100 run (`f1f4917` + `53cf1de`), making it
impossible to attribute any performance change to either alone. The ns0 ablation isolates
the few-shot effect by holding format and stop sequences constant:

| Config | few-shots | Prompt format | Stop | pless_t0.6 | temp_t0.7 |
|--------|-----------|--------------|------|-----------|-----------|
| H100 (Mar-23) | **3** | `[BEGIN]\n` | `\n[DONE]` | **22.5%** | **17.1%** |
| ns0 (Mar-24) | **0** | `[BEGIN]\n` | `\n[DONE]` | **13.8%** | **4.0%** |

**H100 vs ns0 = pure few-shot effect.** Same format, same stop, only the 3 in-context
examples differ.

Reference baseline with a different prompt format (not a controlled comparison):

| Config | few-shots | Prompt format | Stop | pless_t0.6 | temp_t0.7 |
|--------|-----------|--------------|------|-----------|-----------|
| Archive (Mar-6) | 0 | `def func_name(` scaffold | `\ndef` | 23.9% | 14.0% |

## Findings

### 1. Few-shot examples have large positive impact within the `[BEGIN]` format

| Method | 0-shot `[BEGIN]` | 3-shot `[BEGIN]` | Δ from examples |
|--------|-----------------|-----------------|-----------------|
| pless_t0.6 | 13.8% | 22.5% | **+8.7pp** |
| temp_t0.7 | 4.0% | 17.1% | **+13.1pp** |

The examples are critical when using the `[BEGIN]`/`[DONE]` delimiter format.

### 2. Why zero-shot `[BEGIN]` fails so badly

The `[BEGIN]`/`[DONE]` delimiters are not part of Llama-2-7b's fine-tuning — the model
has only seen them in generic template/documentation contexts in pretraining. Without
examples showing `[BEGIN]\n{Python code}\n[DONE]`, the model interprets `[BEGIN]` as a
generic template marker and generates:

- At **t=0.6** (pless): collapses to `[BEGIN]\n[END]\n[BEGIN]\n...` cycling — the most
  probable continuation of `[BEGIN]` in template contexts
- At **t=0.7** (temp): diverse outputs ranging from StackOverflow discussion text to
  Python REPL loops. Only ~4% of tasks pass (accidental valid extractions)

The 3 few-shot examples teach the model: "in this context, `[BEGIN]` = write Python here."

### 3. The `[BEGIN]` format itself is suboptimal for pless vs old zero-shot

Even with 3 examples, 3-shot `[BEGIN]` (22.5%) performs *worse* than the old zero-shot
format with `def func_name(` scaffold (23.9%). The old format was more effective for pless
because it:
- Directly scaffolded the function signature (model only had to generate the body)
- Eliminated ambiguity about what `[BEGIN]` means at low temperature

For temp, 3-shot `[BEGIN]` (17.1%) *outperforms* old zero-shot `def func(` (14.0%) because:
- Old `\ndef` stop sometimes prematurely terminated inside function bodies
- 3-shot examples improve format adherence for temp's higher-entropy sampling

### 4. Answering the original question

The user's observation "no impact of few-shot examples" was based on seeing pless flat
on CodeLlama-7b-hf (41.4% → 41.7%). CodeLlama is a code-specialized model that handles
the `[BEGIN]` format robustly even without examples. Llama-2-7b is a general-purpose model
that needs the examples to understand the format.

**Within the `[BEGIN]` format**: examples have massive impact (+8.7pp pless, +13.1pp temp).

**For the overall change from old zero-shot to current 3-shot setup**:
- pless: -1.4pp net (format change hurt more than examples helped)
- temp: +3.1pp net (examples + stop fix together improved)

## Summary

| Question | Answer |
|----------|--------|
| Do few-shot examples help within `[BEGIN]` format? | **Yes, substantially** (+8.7pp pless, +13.1pp temp) |
| Is the current 3-shot setup better than old zero-shot for pless? | **No** (22.5% < 23.9%) |
| Is the current 3-shot setup better than old zero-shot for temp? | **Yes** (17.1% > 14.0%) |
| Would CodeLlama show the same large few-shot effect? | **Unlikely** — code-specialized model is format-robust |
