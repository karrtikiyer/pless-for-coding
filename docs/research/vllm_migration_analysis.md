# vLLM Migration Impact Analysis

**Date:** 2026-05-15
**Author:** internal analysis (Claude)
**Status:** decision-ready draft — *not a commitment to migrate*
**Audience:** the team running this benchmark stack on H100 80 GB

This document grounds the speedup, engineering, and risk tradeoffs of
replacing the current HuggingFace-based generation backend in
`bench/generator.py` with vLLM. It exists because earlier
chat-level estimates ("3–5×", "5–10×") were ungrounded; the user
asked for a written analysis before any code is touched.

---

## 1. Current backend in one diagram

```
                                     ┌──────────────────────────────────────────────────────┐
                                     │ bench/generator.py (HuggingFace backend, KEEP AS-IS) │
                                     │                                                      │
runner.py / apps/runner.py           │   load_model_and_tokenizer()  (line 85)              │
   ─── (CLI args, prompts, JSONL)──▶ │   generate_samples_standard() (line 176)             │ ──── list[str] samples ───▶ runner appends to JSONL
   ◀─── (model + tokenizer reused) ─ │   generate_samples()          (line 370)             │
                                     │   generate_samples_split()    (line 497)  ◄── HOT    │
                                     │                                                      │
                                     │   manual decode loop:                                │
                                     │     model(input_ids, past_kv) → logits               │
                                     │     temp / softmax                                   │
                                     │     pless or pless_norm sampler  (per-step Python)   │
                                     │     finished mask / inside_think mask                │
                                     │     stop-string scan (CPU decode per step)           │
                                     │     past_key_values = output.past_key_values         │
                                     └──────────────────────────────────────────────────────┘
```

**Contracts a migration must preserve:**

- Public surface: `generate_samples_*` returns `list[str]` keyed by sample index.
- Sampler semantics: `pless`, `pless_norm`, `temp_pure`, `temp_standard` with
  the *same* probability transformation (mask + renormalize for pless;
  top-p/top-k for temp_standard; no truncation for temp_pure).
- Split decoding: think-phase sampler until `</think>` (Qwen3 id 151668), then
  code-phase sampler thereafter, per sequence, one-way.
- JSONL output: `bench/checkpointing.py:33-37` append-per-task contract.
- Stop-string truncation post-processing (MBPP/HumanEval base models only).

**Contracts that may change:**

- KV-cache lifecycle (HF `past_key_values` vs vLLM's paged cache — internal).
- Per-step Python dispatching of the sampler (vLLM batches across requests).
- Tokenizer location (vLLM owns it inside the engine).

---

## 2. Measured baseline

The most recent CODEFORCES-competition H7P sweep
(`results/pless_apps_results/Qwen--Qwen3-8B/CODEFORCES_competition/split_temp_pure_t1.5_pless_t1.0_think_t1.0_t1.0.jsonl`)
covered **268 problems × 10 samples** end-to-end in **16.58 hours** on a
single H100 80 GB.

| Metric | Value |
|---|---:|
| Problems completed | 268 |
| Samples produced | 2,680 |
| Wall time | 59,702 s (16.58 h) |
| Per-problem wall time (mean / median / p90 / max) | 224 / 229 / 236 / 255 s |
| Aggregate throughput | ≈ **265 generated tok/s** (chars ÷ 4.5 ≈ tokens) |
| Per-problem throughput | ≈ 290 tok/s (assuming all 10 samples hit the 8192 cap, which the length analysis confirms is roughly true) |

Two things stand out:

1. **Wall time per problem is essentially constant** (mean 224 s, p90 236 s).
   The decode loop hits the 8192-token budget for almost every sample;
   EOS-based early termination is rare. This is consistent with the
   length-distribution analysis where p50 generated chars ≈ 27,800
   (≈ 6,200 tokens, ≈ 76 % of the cap).
2. **The current aggregate ≈ 265 tok/s** is the number any speedup is
   measured against. On 8B-parameter models on a single H100, vLLM's
   published peak under ideal conditions is **~12,500 tok/s** for
   prompt-heavy serving workloads ([databasemart H100 vLLM benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-h100),
   [morphllm 2026 throughput guide](https://www.morphllm.com/vllm-benchmarks)).
   The headline ratio between the published ceiling and our measured baseline
   is **~47×** — but that ceiling is for OpenAI-style serving with
   throughput-optimized request mixes, *not* a single-prompt × 10-samples ×
   custom-sampler workload like ours. Realistic gain estimate in §5.

**What we have not measured:** a `torch.profiler` step-time
attribution (forward vs sampler vs CPU stop-string scan). The 265 tok/s
number is the aggregate, not a decomposition. If the analysis turns
into a go decision, the prototype work in §7 should start with a 5-minute
profiler run to confirm where the time goes.

---

## 3. What vLLM provides for our case

### Custom `LogitsProcessor` API (v1)

Verified via [vLLM custom-logitsprocs docs](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/).
Subclass `vllm.v1.sample.logits_processor.LogitsProcessor`; required methods:

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `(vllm_config, device, is_pin_memory)` | engine wires it in |
| `apply` | `(logits: Tensor) -> Tensor` (shape `(num_requests, vocab)`) | **logit transform only — cannot return a chosen token** |
| `update_state` | `(batch_update: Optional[BatchUpdate]) -> None` | handles added/removed/moved requests + their emitted tokens |
| `is_argmax_invariant` | `() -> bool` | optimization hint |
| `validate_params` | `(sampling_params: SamplingParams)` | per-request param validation |

This is the integration point for our `p_less_decode` /
`p_less_norm_decode`. They currently do *mask + renormalize +
multinomial*. We hand vLLM the first two steps; vLLM's internal sampler
does the multinomial. Algebraically equivalent given the same RNG seed
modulo the kernel-precision caveats in §8.

### Continuous batching

vLLM's headline feature: when one request finishes, its slot is freed
and a new request is packed in mid-flight. Big win when requests have
*variable* lengths.

**For us:** §2 shows wall time per problem is essentially constant
because every sample runs to the 8192-token cap. Continuous batching
mostly buys us the ability to overlap *across problems* (run problem 2's
prefill while problem 1 is still decoding). That's a real but bounded
gain — see §5.

### Paged attention

Non-contiguous KV memory allocation. Eliminates the padding waste of
HF-style batched decoding. For our workload with variable prompt
lengths (200–3000 tokens on APPS) and uniform generation length (~8K)
this is a moderate win — ~10–20 % of the KV cache footprint reclaimed.

### Qwen3 reasoning parser

`vllm.reasoning.qwen3_reasoning_parser` is *not* a substitute for our
sampler switch. The parser splits the *output text* on `<think>`/`</think>`
after generation finishes; it does not change the distribution mid-decode.
Our split-decoding requirement (different sampling distribution before
vs after the `</think>` boundary) has to be implemented in our own
`LogitsProcessor`. The reasoning parser is irrelevant to that.

---

## 4. Integration design (rough)

```python
class PlessSplitProcessor(LogitsProcessor):
    """One per generation run; one engine step calls apply() once on
    the stacked batch of all in-flight sequences."""

    THINK_END_TOKEN_ID = 151668  # Qwen3 </think>

    def __init__(self, vllm_config, device, is_pin_memory):
        self.phase: dict[int, str] = {}   # req_id -> "think" | "code"
        self.t_think: dict[int, float] = {}
        self.t_code:  dict[int, float] = {}
        # sampler choices ("pless", "pless_norm", "temp_pure") per request

    def update_state(self, batch_update):
        # 1. handle add/remove/move
        # 2. for each existing request, inspect its newly-emitted tokens
        #    in batch_update; if THINK_END_TOKEN_ID appears, flip phase.

    def apply(self, logits):
        # logits: (num_requests, vocab)
        # for each row, apply the appropriate sampler's mask + renormalize
        # under the per-request temperature.
        # Group rows by (phase, sampler_type, temperature) to keep this
        # vectorized; avoid Python per-row work.
        return transformed_logits
```

**Runner integration** (`bench/runner_vllm.py` or
`generator_vllm.py`):

```python
def generate_samples_split_vllm(
    engine, tokenizer, prompt_text, sampler_fn_think, sampler_fn_code,
    n_samples, max_new_tokens, temperature_think, temperature_code,
    stop_strings=None,
) -> list[str]:
    sp = SamplingParams(
        n=n_samples, max_tokens=max_new_tokens,
        logits_processors=[PlessSplitProcessor(
            t_think=temperature_think, t_code=temperature_code,
            sampler_think=sampler_fn_think, sampler_code=sampler_fn_code,
        )],
    )
    outs = engine.generate([prompt_text], sp)
    return [c.text for c in outs[0].outputs]
```

The runner code (`bench/apps/runner.py:184`) changes one line — the
function call — and is otherwise unchanged. Same JSONL write, same
checkpointing, same `--max-new-tokens`. CLI flag added per §10b.

---

## 5. Throughput estimate

Three numbers, each grounded in a different assumption set:

| Scenario | Speedup | Required for | New tok/s | Hours for one CODEFORCES sweep |
|---|---:|---|---:|---:|
| Pessimistic | **1.5–2×** | LogitsProcessor Python dispatch is slow, paged attention only helps the prompt portion | 400–530 | 8–11 |
| Realistic | **3–5×** | LogitsProcessor overhead < 10% of step time, continuous batching nets a 1.5–2× cross-problem win, attention kernels 1.2× faster | 800–1,300 | 3–6 |
| Optimistic | **5–8×** | LogitsProcessor overhead < 5%, paged attention reclaims full 20% of cache, continuous batching dominates | 1,300–2,100 | 2–3 |

**The team should plan around the realistic band (3–5×).** The
optimistic case requires assumptions about vLLM's overhead that
nobody has measured for our specific custom-sampler workload, and the
benchmarks cited in §2 are for serving workloads not single-prompt
batched generation.

**Critical caveat:** these are reasoned estimates, not measurements.
No vLLM run against Qwen3-8B on this codebase has been benchmarked.
The realistic 3–5× could be wrong in either direction.

The published 8B-on-H100 peak of ~12,500 tok/s is not achievable for
us because: (a) our `LogitsProcessor` is in Python and runs on every
engine step; (b) we're running 10 samples per prompt, not 100s of
independent requests; (c) every sample hits the 8K cap so there is no
"swap finished out, pack new in" benefit *within* one problem.

---

## 6. Memory analysis on H100 80 GB

Qwen3-8B's verified architecture: 36 hidden layers, 32 attention heads,
8 KV heads (GQA), head_dim 128, vocab 151,936, max position 40,960.

**KV cache per token, per layer, bf16:**
`2 (K+V) × 8 KV-heads × 128 head_dim × 2 bytes = 4 KB`

**KV cache per token, all 36 layers:** `144 KB`

| Sequences in flight × tokens | KV cache | + Model weights (16 GB) | + ~10 GB activations/scratch | Fits 80 GB? |
|---:|---:|---:|---:|---|
| 10 × 8K (current single-problem) | 11.8 GB | 27.8 GB | 37.8 GB | ✓ comfortable |
| 20 × 8K (2 problems batched) | 23.6 GB | 39.6 GB | 49.6 GB | ✓ comfortable |
| 30 × 8K (3 problems batched) | 35.4 GB | 51.4 GB | 61.4 GB | ✓ workable |
| 40 × 8K (4 problems batched) | 47.2 GB | 63.2 GB | 73.2 GB | ⚠ on the edge |
| 50 × 8K | 59.0 GB | 75.0 GB | 85.0 GB | ✗ over |

Conclusion: **on H100 80 GB, KV memory is not the binding constraint
at the 3× batch level. Compute is.** That's why a vLLM migration's value
proposition is compute-throughput-driven (continuous batching +
kernel-level efficiency), not memory-driven.

This also means the cheap-lever alternative (§10) — batch-across-problems
without vLLM — has *memory headroom for ~3× batch* on H100, validating
the 1.5–2× speedup estimate from chat.

---

## 7. Engineering cost

| Task | File(s) touched | Days | Risk |
|---|---|---:|---|
| Wrap `p_less_decode` + `p_less_norm_decode` as `LogitsProcessor` subclass; cover `temp_pure` and `temp_standard` paths too | new `bench/generator_vllm.py` (only) | 0.5 | low (pure functions, well-tested) |
| Implement stateful think→code phase tracker in `LogitsProcessor.update_state` | new `bench/generator_vllm.py` | 1.0 | medium (per-sequence state, prone to off-by-one) |
| Wrap engine lifecycle (single `LLM` instance reused across problems) + match `generate_samples_split` signature | new `bench/generator_vllm.py` | 0.5 | low |
| Add `--backend {hf,vllm}` CLI flag + dispatch in each runner | `bench/runner.py`, `bench/apps/runner.py`, `bench/humaneval/runner.py` | 0.5 | low |
| Correctness regression test: same prompt + seed + thresholds, compare distribution metrics (NAUADC, struct_div, codebleu_div) over 10 problems × 10 samples on HF vs vLLM | new `tests/test_vllm_parity.py`, scratch script | 0.5 | high — silent drift is the failure mode here |
| Memory + throughput measurement: H7P-CODEFORCES, 10 problems on both backends | scratch script + plot | 0.5 | medium |
| Separate venv setup + dependency-pinning notes | new `pyproject-vllm.toml`, README addendum | 0.25 | low |
| Documentation + 1-hour code review | doc updates only | 0.5 | low |

**Total: 4.25 days of engineering + ~1 day of buffer for debugging
the correctness regression** (the buffer is generous if the parity test
passes, but balloons if it doesn't).

---

## 8. Failure modes and silent risks

1. **Silent correctness drift in the LogitsProcessor.** Per-sequence
   state machine, batched logit transforms, and grouping by phase are
   each places where a subtle bug shifts the sampling distribution
   without crashing or failing CI. *Mitigation:* the parity regression
   test in §7 (HF vs vLLM with same seeds on 10 problems × 10 samples
   per config; compare NAUADC, struct_div, codebleu_div within the
   metrics' measured noise floors). If parity fails, do not ship.

2. **vLLM API drift.** The v1 `LogitsProcessor` API documented today
   is relatively recent. *Mitigation:* pin vLLM exactly via the new
   `pyproject-vllm.toml`; date the analysis doc; treat any minor
   vLLM upgrade as a chance to re-run the parity test before deploying.

3. **CUDA / PyTorch version conflict.** vLLM pins specific CUDA and
   PyTorch versions that are not always compatible with our existing
   `uv.lock`. *Mitigation:* separate `.venv-vllm/` (§10b-C). The
   existing venv stays untouched so the algosim work and the layer-
   entropy probes keep running.

4. **Continuous batching tail-latency.** Our workload has long, fixed-
   length generations (every sample hits the 8K cap). vLLM's continuous-
   batching win shrinks when the in-flight sequences are all roughly
   the same length, because there's no short-sequence churn to pack.
   This is the **single biggest uncertainty in the 3–5× estimate**.
   *Mitigation:* benchmark before scaling; the §7 throughput measurement
   step is non-negotiable before any production switch.

5. **Determinism / reproducibility.** vLLM's attention kernel
   (FlashAttention-2 / xFormers) is not bit-identical to HF's SDPA.
   Even with the same seed, sampled tokens may differ by step 1 in
   ways that compound. *Mitigation:* the parity test compares
   *distributional* metrics (NAUADC, struct_div), not bit-identity.
   Cross-backend reports must label the backend per record (§10b-B).

6. **LogitsProcessor Python overhead.** Every engine step calls our
   `apply()` in Python (the *logit transformation* runs on GPU, but
   the *dispatch* is Python). For 8K decode steps × 10 sequences,
   even 100 µs per call adds 800 ms per request. Not catastrophic
   but real. *Mitigation:* group rows by phase before the transform
   so the GPU sees vectorized ops, not a Python loop.

7. **Long-running engine memory growth.** vLLM keeps a persistent
   `LLM` engine; KV pages get allocated/freed. Sustained runs (hours)
   sometimes accumulate fragmentation. *Mitigation:* periodic
   engine restart between configs; not just one engine per sweep.

8. **`torch.multinomial` differences.** vLLM's sampling stage uses
   its own multinomial implementation, which may differ from
   `torch.multinomial` in tie-breaking on degenerate distributions.
   For our use case (probs after pless masking) this is unlikely to
   matter, but it's worth flagging. *Mitigation:* the parity test
   covers this if it differs.

---

## 9. Decision criteria

**Vote vLLM-yes if:**

- The remaining APPS work plan includes ≥3 more difficulty buckets
  (introductory, interview, more CODEFORCES configs).
- The team plans to run the experiments from the decoding-time
  diversity research doc — Conformative Decoding, DoLa, Verbalized
  Sampling — which all require many new generation passes against
  the same models.
- The 4.25 days of engineering plus 1 day of buffer is acceptable
  against the remaining-work value.

**Vote vLLM-no (or "later") if:**

- The remaining sweeps fit inside the next 1–2 days of GPU time on
  the current backend (i.e., we'd burn the engineering budget while
  the GPU finishes anyway).
- The team is moving to a different focus area (e.g., training-time
  interventions) where the generation backend isn't the hot path.
- The parity regression test fails on the first try and unfreezes the
  schedule risk.

**Vote batch-across-problems (cheap alternative) if:**

- The team wants a 1.5–2× speedup *today*, with no new dependency,
  and accepts that vLLM's bigger gain is left on the table.

---

## 10. Alternative: batch-across-problems (cheap lever)

A reminder that the cheap lever exists. Modify
`generate_samples_split` to accept multiple prompts at once,
left-pad to the longest prompt in the batch, expand `past_key_values`
across all (K × N) sequences, and run one decode loop. Memory analysis
in §6 confirms 30 sequences in flight fit comfortably on H100 80 GB.

Expected speedup: **1.5–2× realistic, ~0.5 day work, zero new
dependency.** Decision matrix:

| If remaining work is… | …choose… |
|---|---|
| Small (< 2 days of GPU) | batch-across-problems or stay |
| Medium (a week of GPU) | batch-across-problems |
| Large (weeks of GPU + new experiments lined up) | vLLM |

---

## 10b. Zero-regression rollout + rollback

The whole migration has to be safe to try and easy to abandon.

### A. Side-by-side, never in-place

Create `bench/generator_vllm.py` parallel to `bench/generator.py`. The
HF backend stays at the same path with the same monkey-patches and the
same call signatures. **No edit to `bench/generator.py`** is required.

### B. CLI flag, not a hard cutover

Add `--backend {hf,vllm}` (default `hf`) to each of `bench/runner.py`,
`bench/apps/runner.py`, `bench/humaneval/runner.py`. Each runner
dispatches to the matching `generate_samples_*` function. One line of
extra code per runner.

Each JSONL record gains a `"backend": "hf" | "vllm"` field so that any
downstream metric / report can identify which backend produced any given
sample. Backward-compatible (the field is ignored by existing readers).

### C. Separate Python venv

vLLM has strict CUDA/PyTorch pinning. Use `.venv-vllm/` parallel to
`.venv/` (separate `pyproject-vllm.toml` or pinned `requirements.txt`).
The existing venv stays untouched, so the layer-entropy probes,
algosim_claude_judge, and all other in-flight work keep running on the
HF stack.

### D. Correctness gate before any production switch

Before any "real" config runs on vLLM:

1. **Distribution parity test:** identical prompt × identical RNG seed
   × identical pless thresholds. Sample 10 × 10 sequences on both
   backends; compute NAUADC, struct_div, codebleu_div. Pass criterion:
   each metric agrees within its measured noise floor (NAUADC ± 0.02,
   struct_div ± 0.01, codebleu_div ± 0.02 over 10 problems × 10
   samples).
2. **10-problem A/B run** on H7P-CODEFORCES competition. Side-by-side
   numbers reviewed before scaling.

If parity fails on either, do not ship vLLM. See §10b-G.

### E. Rollback in three forms

1. **Per-run rollback** (free, instant): switch `--backend vllm` back
   to `--backend hf`. Existing outputs unchanged.
2. **Per-config rollback** (free, instant): the `backend` field per
   record lets us pool, separate, or discard vLLM-generated data in
   any analysis.
3. **Full feature removal** (~1 hour): delete
   `bench/generator_vllm.py`, `.venv-vllm/`, `pyproject-vllm.toml`,
   and revert the `--backend` flag default. Repo returns to the
   pre-migration state modulo the now-ignored `backend` field in
   JSONLs.

### F. Things we deliberately do not touch

- `bench/generator.py` — HF backend, monkey-patches and all. Even
  the now-irrelevant Qwen-7B shims stay (still needed for the older
  models the project supports).
- `bench/checkpointing.py`, `bench/sampler_bridge.py`,
  `bench/prompts.py`, `bench/apps/prompts.py` — all unchanged.
- `bench/eval/` — every analysis script keeps working; the new
  JSONLs flow through unchanged.
- `results/*.jsonl` — existing files unchanged; vLLM run writes
  new files alongside.

### G. What "vLLM didn't prove useful" looks like

- **Speedup is below 1.5×:** use E.3 to remove. Keep the analysis
  doc as the record of "we considered this".
- **Correctness regression fails by more than the agreed tolerance:**
  E.3 to remove. Treat as "the migration story isn't clean", roll back.
- **Speedup is real but config-dependent:** keep the code but only
  use it for the heavy configs (APPS competition / interview).
  Leave MBPP on HF — simpler code paths, gains go where they matter.

---

## Appendix: sources

- vLLM custom logits processors: [docs.vllm.ai/en/latest/features/custom_logitsprocs](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/)
- vLLM v1 logits processors API: [docs.vllm.ai/en/latest/design/logits_processors](https://docs.vllm.ai/en/latest/design/logits_processors/)
- vLLM Qwen3 reasoning parser: [docs.vllm.ai/en/latest/api/vllm/reasoning/qwen3_reasoning_parser](https://docs.vllm.ai/en/latest/api/vllm/reasoning/qwen3_reasoning_parser/)
- vLLM H100 benchmarks: [databasemart H100 vLLM](https://www.databasemart.com/blog/vllm-gpu-benchmark-h100), [morphllm 2026 vLLM benchmarks](https://www.morphllm.com/vllm-benchmarks)
- Qwen3 reasoning controls in vLLM: [discuss.vllm.ai — Qwen3 hybrid thinking](https://discuss.vllm.ai/t/deployment-example-for-a-qwen3-model-with-hybrid-thinking/1462)
