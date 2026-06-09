# Induced-CoT decoder comparison on APPS (instruct models) — TODO A29

**Question (within-model):** on an *instruct* (non-reasoning) model placed under
**induced** chain-of-thought, do the hyperparameter-free **pless / pless_norm**
samplers give better pass@k **and** lower token consumption than well-tuned
standard stochastic decoders (temperature, top-p, top-k, and their provider
combination)? CoT is the fixed condition; the **decoder is the only variable**.

This is *not* a comparison against reasoning models (Qwen3 / DeepSeek), and α
arms are out of scope — see the version history in `docs/theory/todos.md`.

## Setup

- **Models:** Qwen2.5-Coder-7B-Instruct, Qwen2.5-Coder-3B-Instruct.
- **Benchmark:** APPS, ATCODER **interview**, all **252** problems, **n=10**
  samples, 8192-token budget, vLLM backend.
- **CoT induction:** `<think>` prefill (`--prompt-format cot-prefill`,
  `bench/apps/prompts.py::format_prompt_apps_cot_prefill`). The model has no
  native think phase; the prompt ends with `<think>\n` and instructs it to close
  with `</think>` then emit code.
- **Configs (all under induced CoT):**
  - `pless`, `pless_norm` @ t1.0 — hyperparameter-free, under test
  - **provider combo** — temp 0.7 + top_p 0.8 + top_k 20 + repetition_penalty
    (1.1 for 7B, 1.05 for 3B; verified from each model's shipped
    `generation_config.json`)
  - **temp 0.2 + top_p 0.95** — code-gen literature pass@1 setting
  - **temp 0.8** — diversity / coverage setting
- **Think-token measurement:** the code fence (` ``` `) delimits reasoning, not
  `</think>` — instruct models emit `</think>` only ~53% of the time, so the
  fence (~100% present) is the robust boundary. Validated against `</think>`
  where both exist: median 0-token difference. (`--think-delimiter fence`.)

## Metric verification (independently re-derived before reporting)

All headline metrics were re-derived from the raw per-sample data with a
*separate* implementation and matched the pipeline exactly:

- **pass@1/5/10** — re-implemented the unbiased estimator in closed form
  (`1 - comb(n-c,k)/comb(n,k)`) vs the pipeline's product form, driven from the
  raw per-sample pass booleans; confirmed the `pass@1 = mean(c/n)` and
  `pass@10 = frac(c≥1)` (n=k=10) identities; confirmed the CSV and metrics-JSON
  reporting paths agree. **Exact match on all 10 configs.**
- **Data integrity** — 252 tasks, 10 samples each, `num_correct == sum(pass_results)`,
  zero passed-but-unextracted samples. **All pass.**
- **think + total tokens** — re-implemented the fence cut independently (not
  importing the module) and re-tokenized; matched the report to <0.1 token;
  `think ≤ full` always. **Exact match.**
- **structural_diversity** — found to be confounded for cross-config comparison
  (see appendix); **excluded from the headline.**

## Results

### Qwen2.5-Coder-7B-Instruct
| config | pass@1 | pass@5 | pass@10 | think tok | total tok |
|---|---|---|---|---|---|
| temp 0.8 | 0.097 | 0.218 | 0.278 | 340 | 637 |
| provider combo | 0.094 | 0.207 | 0.254 | 322 | 629 |
| temp 0.2 + top_p 0.95 | 0.108 | 0.207 | 0.246 | 343 | 624 |
| pless_norm | 0.111 | 0.205 | 0.242 | 340 | 630 |
| pless | **0.117** | 0.203 | 0.238 | 341 | 633 |

### Qwen2.5-Coder-3B-Instruct
| config | pass@1 | pass@5 | pass@10 | think tok | total tok |
|---|---|---|---|---|---|
| provider combo | 0.059 | 0.132 | 0.171 | 334 | 690 |
| temp 0.8 | 0.043 | 0.115 | 0.155 | 331 | 710 |
| temp 0.2 + top_p 0.95 | **0.068** | 0.128 | 0.151 | 325 | 696 |
| pless_norm | 0.059 | 0.109 | 0.135 | 326 | 689 |
| pless | 0.055 | 0.114 | 0.135 | 340 | 703 |

## Findings

**The hypothesis ("better pass@k AND fewer tokens") is not supported.** pless is
a **probability concentrator**:

1. **pass@1:** pless wins on the capable model — 7B pless **0.117**, the best of
   the five (provider combo 0.094, lit-temp-0.2 0.108). On the weaker 3B it does
   *not* win (temp 0.2 leads at 0.068; pless 0.055). Concentrating onto the top
   token only helps when that token is often correct, which needs a strong
   enough model.
2. **pass@10 (coverage):** pless is **last on both models** (7B 0.238 vs temp 0.8
   0.278; 3B 0.135 vs provider 0.171). Concentration costs sample-to-sample
   coverage. The loosest decoder (temp 0.8) wins pass@10 on 7B.
3. **Token consumption: a flat null.** think tokens 322–343 and total tokens
   624–637 (7B) / 689–710 (3B) are essentially constant across *all* decoders;
   pless is among the **highest**, never the cheapest. Truncation ≈ 0 everywhere
   — induced CoT does not ramble (unlike RL-trained reasoners under pless), and
   the sampler does not change CoT or total length. There is no token-savings
   story.

**Verdict:** under induced CoT on an instruct model, **pless trades coverage for
single-shot accuracy at identical token cost** — useful if you optimize pass@1
on a capable model, strictly worse if you optimize pass@10. It does not
dominate tuned standard decoders.

### Caveats
- Absolute pass rates are modest (pass@1 ≈ 0.06–0.12, pass@10 ≈ 0.13–0.28 on 252
  problems) — ATCODER interview is hard for 3–7B coders. Pairwise gaps are not
  yet bootstrap-tested; the cross-k *ordering* (pless best@1 / worst@10 on 7B)
  is the robust signal, individual gaps should carry error bars before strong
  claims.
- The provider combo is a pass@1 outlier (lowest on 7B at 0.094 despite mid
  diversity) — likely top_p 0.8 + top_k 20 + repetition_penalty over-truncating
  single-shot quality.

## Appendix — why structural_diversity is not in the headline

`compute_structural_diversity` (metrics.py:124, 308) averages per-task mean
pairwise AST-edit distance **only over correct samples, only on tasks with ≥2
correct**. Each config solves a different set of tasks ≥2×, so the reported
numbers average over **non-comparable task subsets** (the NAUADC/C7 confound).
Recomputed on the **common task set** (≥2 correct in all 5 configs):

### Qwen2.5-Coder-7B-Instruct — structural diversity (n_common=27)
| config | reported (confounded) | common-set |
|---|---|---|
| temp 0.8 | 0.624 | 0.631 |
| provider combo | 0.478 | 0.444 |
| pless_norm | 0.392 | 0.400 |
| pless | 0.366 | 0.392 |
| temp 0.2 + top_p 0.95 | 0.423 | 0.382 |

### Qwen2.5-Coder-3B-Instruct — structural diversity (n_common=16)
| config | reported (confounded) | common-set |
|---|---|---|
| temp 0.8 | 0.569 | 0.591 |
| provider combo | 0.440 | 0.454 |
| pless_norm | 0.402 | 0.367 |
| pless | 0.429 | 0.355 |
| temp 0.2 + top_p 0.95 | 0.350 | 0.279 |

On the common set the clean "pass@10 ↔ diversity rank match" **does not hold**:
temp 0.2 becomes the *lowest*-diversity config yet keeps mid pass@10. Only the
extremes align (temp 0.8 = most diverse + best coverage; pless = low diversity +
low coverage); the middle is within noise (n_common = 27 / 16). pass@10 over all
252 tasks is the clean coverage measure and needs no diversity proxy.

## Reproduce
```
# generation (CUDA pod): run_cot_instruct_apps_decoders.sh per model, BACKEND=vllm
# eval:   python -m bench.eval --results-file <jsonl> --dataset apps --workers 32 --skip-diversity
# report: python -m bench.eval.cot_efficiency --results-dir <model>/ATCODER_interview \
#           --dataset apps --max-tokens 8192 --tokenizer <model> --think-delimiter fence
```
Per-model artifacts: `results/pless_cot_efficiency_instruct/<model>/ATCODER_interview/analysis/`.
