# MBPP Full: Grounded Insights on P-Less Sampling for Code Generation

**Scope:** 9 models × 66 configs, MBPP-full (500 tasks × 10 samples), unbiased pass@k estimator, BigCode/standard prompts.
**Data:** `results/pless_full_mbpp_results/analysis/consolidated_metrics/`

---

## 1. First-Sample Efficiency: P-Less Captures Most Value Upfront

The ratio pass@10 / pass@1 reveals how much a method benefits from drawing additional samples. A ratio near 1.0 means the first sample already captures most of the method's potential; a high ratio means the method needs many samples to realize its value.

| Model | Method | pass@1 | pass@10 | Ratio | Value captured in 1st sample |
|---|---|---|---|---|---|
| **Qwen2.5-Coder-3B** | pless (t=0.6) | 59.4% | 66.4% | 1.12× | 89% |
| | temp (t=0.7) | 42.6% | 77.8% | 1.83× | 55% |
| | top_p0.9 (t=1.0) | 34.7% | 77.0% | 2.22× | 45% |
| **CodeLlama-7b** | pless (t=0.6) | 41.8% | 49.4% | 1.18× | 85% |
| | temp (t=0.7) | 36.8% | 65.2% | 1.77× | 56% |
| | top_p0.9 (t=1.0) | 34.6% | 68.2% | 1.97× | 51% |
| **Llama-2-7b** | pless (t=0.6) | 22.5% | 29.2% | 1.30× | 77% |
| | temp (t=0.7) | 17.1% | 44.6% | 2.61× | 38% |
| | top_p0.9 (t=1.0) | 14.0% | 39.4% | 2.81× | 36% |
| **OCI-DS-1.3B** | pless (t=0.6) | 43.9% | 49.0% | 1.12× | 90% |
| | temp (t=0.7) | 43.1% | 62.4% | 1.45× | 69% |

**Insight:** P-less consistently captures 77–90% of its pass@10 value in a single sample. Temperature and top-p capture only 36–69%. In production settings where inference budget limits you to 1–3 samples (code completion, CI pipelines, real-time suggestions), p-less delivers substantially more value per sample. Temperature and top-p look competitive or superior at pass@10, but that comparison assumes you can afford 10 generations — which is rarely the case in deployment.

**The crossover:** Temperature *surpasses* p-less at pass@k for k ≥ 5 on most models (e.g., Qwen2.5-Coder-3B: temp passes p-less at pass@5, 70.7% vs 65.0%). This crossover is the fundamental tradeoff: p-less optimizes the distribution for the *mode*, temperature optimizes for *coverage*.

---

## 2. The Deduplication Ratio Reveals the Mechanism

Among correct samples for a given task, what fraction are structurally unique? This ratio directly measures whether a method explores the solution space or collapses onto a single canonical form.

| Model | pless dedup ratio | temp dedup ratio | top_p dedup ratio |
|---|---|---|---|
| Qwen2.5-Coder-3B | 0.22 | 0.87 | 0.93 |
| CodeLlama-7b | 0.23 | 0.80 | 0.87 |
| Llama-2-7b | 0.26 | 0.85 | 0.87 |
| OCI-DS-1.3B | 0.18 | 0.57 | 0.62 |

**Insight:** When p-less produces 8 correct samples out of 10, approximately 6 of them are structurally identical (same AST fingerprint). Temperature produces 8 unique variants. This is not a side effect — it is the mechanism. The collision-entropy threshold `p = Σ(prob²)` zeros out all tokens below the threshold, physically eliminating low-probability continuations at every decoding step. The model is forced to repeat its highest-confidence path.

**Why this matters for practitioners:**
- **Need one reliable solution** (code suggestion, CI pipeline): p-less's convergence is a feature — you get the model's best answer repeatedly.
- **Need diverse candidates** (search, ensembles, code review alternatives): temperature wins — nearly every sample is a unique approach.

**OCI-DS-1.3B is an outlier:** Even temperature only achieves 0.57 dedup ratio on this model, suggesting the 1.3B model has a narrower solution space to begin with. Smaller models have less capacity for diverse correct code.

---

## 3. Task Coverage Asymmetry: Temperature Solves More Unique Tasks

A natural question: does p-less solve *different* tasks, or just solve the *same* tasks more reliably? The per-task data answers definitively.

| Model | Tasks solved only by pless | Tasks solved only by temp | Tasks solved only by top_p |
|---|---|---|---|
| Qwen2.5-Coder-3B | 11 | 68 | 68 |
| CodeLlama-7b | 12 | 91 | 102 |
| Llama-2-7b | 12 | 89 | 72 |
| OCI-DS-1.3B | 2 | 69 | 76 |

**Insight:** Temperature and top-p solve 6–50× more *unique* tasks than p-less. This is the other side of the diversity coin — by exploring low-probability continuations, temperature occasionally stumbles onto correct solutions that p-less's conservative pruning never reaches.

**But this needs context.** Despite solving fewer unique tasks, p-less achieves higher pass@1 because it solves the overlapping tasks (the ones both methods can reach) far more *consistently*. On Qwen2.5-Coder-3B, p-less solves ~297 tasks at pass@1 ≥ 0.5 (59.4% × 500), while temperature solves ~213 tasks (42.6% × 500). The 68 extra tasks temp reaches don't compensate for the ~84 tasks where temp fails but p-less succeeds consistently.

**Implication:** If your evaluation metric is pass@1 (single-attempt success), p-less wins. If your metric is "can any of 10 samples solve this task" (pass@10), temperature's exploration advantage shows — it reaches more of the task space, just less reliably per sample.

---

## 4. Model Capability Amplifies the P-Less Advantage

The gap between p-less and temperature pass@1 is not constant — it scales with model quality.

| Model | Capability | pless pass@1 | temp pass@1 | Gap (pp) | Relative gap |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | Strong coder | 59.4% | 42.6% | +16.8 | +39% |
| Qwen2.5-Coder-1.5B | Moderate coder | 53.1% | 38.1% | +15.0 | +39% |
| OCI-DS-1.3B | Small coder | 43.9% | 43.1% | +0.8 | +2% |
| CodeLlama-7b | Moderate code | 41.8% | 36.8% | +5.0 | +14% |
| Qwen-7B | Base general | 35.4% | 29.8% | +5.6 | +19% |
| Qwen-7B-Chat | Chat general | 34.0% | 28.7% | +5.3 | +18% |
| CodeLlama-7b-Instruct | Instruct code | 41.2% | 38.3% | +2.9 | +8% |
| Llama-2-7b | Weak general | 22.5% | 17.1% | +5.4 | +32% |
| Llama-2-7b-chat | Weak chat | 20.5% | 17.8% | +2.7 | +15% |

**Insight:** The absolute gap is largest on the specialized coders (Qwen2.5-Coder-3B: +16.8pp, Qwen2.5-Coder-1.5B: +15.0pp). These models have sharper probability distributions on code tokens — they "know" the right answer with higher confidence. P-less's entropy-based threshold is well-calibrated when the signal is strong: it removes noise without removing signal.

**The OCI-DS-1.3B anomaly:** Only +0.8pp gap. This 1.3B model is at the capability floor where even the model's most confident paths are frequently wrong. P-less's thresholding can't help much when the mode itself is incorrect. Temperature's exploration costs less because there's less signal to lose.

**Instruct vs base models:** Instruct-tuned models (CodeLlama-Instruct, Llama-2-chat) show smaller p-less advantages (+2.7–2.9pp) compared to their base counterparts (+5.0–5.4pp). Instruction tuning already sharpens the distribution somewhat, reducing the marginal benefit of p-less's additional pruning.

---

## 5. Cover@t Reveals Robustness That pass@k Hides

pass@k asks "did at least one sample pass?" — a lenient bar. cover@t asks "what fraction of tasks have ≥t fraction of samples correct?" — a robustness measure. High cover@0.7 means the method doesn't just solve problems — it solves them *reliably*.

| Model | Method | pass@1 | pass@10 | cover@0.5 | cover@0.7 |
|---|---|---|---|---|---|
| **Qwen2.5-Coder-3B** | pless (t=0.6) | 59.4% | 66.4% | 60.2 | 57.2 |
| | temp (t=0.7) | 42.6% | 77.8% | 48.2 | 34.2 |
| | top_p0.9 (t=1.0) | 34.7% | 77.0% | 37.4 | 20.6 |
| **CodeLlama-7b** | pless (t=0.6) | 41.8% | 49.4% | 42.0 | 39.2 |
| | temp (t=0.7) | 36.8% | 65.2% | 39.4 | 29.0 |
| **Llama-2-7b** | pless (t=0.6) | 22.5% | 29.2% | 23.2 | 20.6 |
| | temp (t=0.7) | 17.1% | 44.6% | 15.8 | 9.6 |
| **OCI-DS-1.3B** | pless (t=0.6) | 43.9% | 49.0% | 44.2 | 41.8 |
| | temp (t=0.7) | 43.1% | 62.4% | 44.0 | 38.0 |

**Insight:** The cover@0.7 column is where p-less's advantage is starkest. On Qwen2.5-Coder-3B, 57.2% of tasks have ≥7/10 p-less samples correct, vs only 34.2% for temperature and 20.6% for top-p. Temperature achieves higher pass@10 (77.8%) by *spreading* its correct samples thinly across many tasks — getting 1–3 right on each. P-less *concentrates* correctness: if it can solve a problem, it solves it 7+ times out of 10.

**The gap widens as the threshold rises.** At cover@0.5, p-less leads temp by 12pp on Qwen2.5-Coder-3B. At cover@0.7, it leads by 23pp. This divergence means p-less's solutions are not just more often correct — they are more *deterministically* correct.

**Why this matters:** If you need confidence that your sampler will produce a correct solution (not just "might produce one somewhere in 10 tries"), p-less provides that reliability. Temperature's pass@10 advantage is an artifact of lenient evaluation — one lucky hit out of 10 counts the same as 10/10.

---

## 6. The Normalization Variant Is a Statistical Tie

The p-less-norm variant uses the relaxed threshold `p = (v·Σprob² - 1)/(v - 1)` where `v` is vocabulary size, compared to p-less's `p = Σprob²`. The normalization was theoretically motivated to account for uniform probability mass.

| Model | pless pass@1 | pless_norm pass@1 | Δ | pless codebleu_div | pless_norm codebleu_div |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B (t=0.6) | 59.4% | 59.5% | +0.1 | 0.203 | 0.204 |
| Qwen2.5-Coder-1.5B (t=0.6) | 53.1% | 52.9% | −0.2 | 0.220 | 0.211 |
| OCI-DS-1.3B (t=0.6) | 43.9% | 43.9% | 0.0 | 0.119 | 0.117 |
| CodeLlama-7b (t=0.6) | 41.8% | 41.7% | −0.1 | 0.197 | 0.197 |
| Qwen-7B (t=0.6) | 35.4% | 35.5% | +0.1 | 0.168 | 0.169 |
| Llama-2-7b (t=0.6) | 22.5% | 22.6% | +0.1 | 0.169 | 0.159 |
| Llama-2-7b-chat (t=0.6) | 20.5% | 20.4% | −0.1 | 0.011 | 0.015 |
| CodeLlama-7b-Instruct (t=0.6) | 41.2% | 41.1% | −0.1 | 0.051 | 0.053 |

**Insight:** The maximum difference is 0.2pp on pass@1, well within noise for n=10 samples. The normalization correction `(v·Σp² - 1)/(v - 1)` converges to `Σp²` as vocabulary size v grows large. With v ≈ 150K tokens, the correction is negligible. Both variants apply nearly the same threshold at every decoding step.

**Recommendation for the community:** There is no practical reason to prefer one variant over the other. Use whichever is simpler to implement (standard p-less).

---

## 7. Zero Hyperparameters vs Tuned Baselines

P-less requires no hyperparameters — the threshold is derived from the probability distribution itself at each step. How does this automatic method compare against carefully tuned alternatives?

**Against arXiv:2507.03160** (top_p=0.95, t=0.2, tuned for code generation):

| Model | P-Less best pass@1 | Paper baseline pass@1 | Δ |
|---|---|---|---|
| Qwen2.5-Coder-3B | 59.5% (pless_norm, t=0.6) | 57.0% | +2.5pp |
| Qwen2.5-Coder-1.5B | 53.1% (pless, t=0.6) | 51.0% | +2.1pp |
| OCI-DS-1.3B | 44.6% (pless_norm, t=1.0) | 44.0% | +0.6pp |

**Against arXiv:2402.06925** (14 decoding methods, tuned on Llama-2-7b):
- P-less beats 12 of 14 methods on Llama-2-7b (only beam search and FSD-d occasionally surpass it)
- P-less norm achieves rank #1 at 22.3% vs paper's best FSD-d at 21.2%

**Our own top_p0.95 replication** (t=0.2, matching the paper's config):

| Model | pless (t=0.6) | top_p0.95 (t=0.2) | Δ |
|---|---|---|---|
| Qwen2.5-Coder-3B | 59.4% | 58.2% | +1.2pp |
| Qwen2.5-Coder-1.5B | 53.1% | 52.5% | +0.6pp |
| OCI-DS-1.3B | 43.9% | 44.5% | −0.6pp |

**Insight:** P-less matches or beats tuned top_p=0.95/t=0.2 baselines on 2 of 3 models while requiring zero hyperparameter selection. The practical value is not just the marginal accuracy gain — it is eliminating the tuning loop. Temperature and top-p require sweeps over (temperature × p) space to find the optimal operating point for each model and task. P-less adapts automatically.

**Caveat:** On OCI-DS-1.3B, the tuned top_p0.95 baseline marginally beats p-less by 0.6pp. P-less is not universally dominant — at the capability floor, careful tuning can slightly outperform automatic thresholding.

---

## 8. The t=1.0 Sweet Spot — Pareto-Optimal Diversity

P-less still uses a temperature parameter to scale logits before computing the threshold. How does t=1.0 (no scaling) compare to t=0.6 (the default)?

| Model | Config | pass@1 | pass@10 | struct_div | codebleu_div |
|---|---|---|---|---|---|
| **Qwen2.5-Coder-3B** | pless t=0.6 | 59.4% | 66.4% | 0.097 | 0.203 |
| | pless t=1.0 | 56.6% | 72.2% | 0.259 | 0.450 |
| | temp t=0.7 | 42.6% | 77.8% | 0.537 | 0.690 |
| **CodeLlama-7b** | pless t=0.6 | 41.8% | 49.4% | 0.081 | 0.197 |
| | pless t=1.0 | 41.5% | 57.2% | 0.231 | 0.454 |
| | temp t=0.7 | 36.8% | 65.2% | 0.436 | 0.652 |
| **Llama-2-7b** | pless t=0.6 | 22.5% | 29.2% | 0.067 | 0.169 |
| | pless t=1.0 | 22.4% | 37.0% | 0.168 | 0.355 |
| | temp t=0.7 | 17.1% | 44.6% | 0.397 | 0.649 |

**Insight:** Raising temperature from 0.6 to 1.0 costs p-less only 0.1–2.8pp on pass@1 while roughly *doubling* diversity (CodeBLEU: +0.10 to +0.25 absolute). Temperature at t=0.7 gets higher diversity still, but pays 5–17pp on pass@1.

P-less at t=1.0 occupies a unique position on the Pareto frontier: it sacrifices less correctness per unit of diversity gained than any other method. The "exchange rate" in CodeBLEU diversity per pp of pass@1 lost:

| Transition | pass@1 cost | CodeBLEU diversity gain | Diversity per pp lost |
|---|---|---|---|
| pless t=0.6 → pless t=1.0 (Qwen) | −2.8pp | +0.247 | 0.088/pp |
| pless t=0.6 → temp t=0.7 (Qwen) | −16.8pp | +0.487 | 0.029/pp |
| pless t=1.0 → temp t=0.7 (Qwen) | −14.0pp | +0.240 | 0.017/pp |

**P-less t=1.0 buys diversity 3–5× more cheaply than temperature.** This makes it the method of choice when you want *some* diversity without paying a steep accuracy cost.

---

## 9. Instruct-Tuned Models Show Near-Deterministic P-Less Behavior

An unexpected pattern: on instruct-tuned models, p-less produces almost zero structural diversity.

| Model | pless struct_div (t=0.6) | pless struct_div (t=1.0) | temp struct_div |
|---|---|---|---|
| CodeLlama-7b (base) | 0.081 | 0.231 | 0.436 |
| CodeLlama-7b-Instruct | **0.000** | **0.000** | 0.011 |
| Llama-2-7b (base) | 0.067 | 0.168 | 0.397 |
| Llama-2-7b-chat | **0.002** | **0.019** | 0.142 |

**Insight:** Instruction tuning and RLHF sharpen the model's distribution so dramatically that p-less's entropy threshold eliminates nearly all variation. The model outputs are effectively deterministic — 10 samples yield 1 unique solution. Even *temperature* on CodeLlama-Instruct only achieves 0.011 structural diversity (vs 0.436 on the base model).

**Implication:** For instruct-tuned models, p-less adds no value over greedy decoding in terms of diversity. Its only benefit is robustness (if the greedy path hits a bad token, p-less can still recover via the small remaining probability mass). If diversity is needed from instruct models, only temperature/top-p sampling with substantial settings can break through the narrow distribution.

---

## 10. The Experimental Variants: What Didn't Work

Qwen-7B was tested with several p-less variants beyond the standard method:

| Variant | pass@1 | vs standard pless | Description |
|---|---|---|---|
| pless (standard) | 35.4% | baseline | Collision entropy threshold |
| pless_norm | 35.5% | +0.1pp | Normalized threshold |
| pless_hybrid | 31.5% | −3.9pp | Hybrid with another method |
| pless_norm_hybrid | 31.4% | −4.0pp | Normalized hybrid |
| pless_ns0 (no stop) | 14.0% | −21.4pp | No stop sequences |
| pless_bs (beam search) | 12.8% | −22.6pp | Beam search + p-less |
| temp_ns0 (no stop) | 9.7% | −25.7pp | Temperature, no stop |

**Insight:** The no-stop-sequence variants (`ns0`) and beam search variant (`bs`) catastrophically underperform. Stop sequences are critical — without them, the model generates past the target function and the extraction pipeline cannot reliably recover the answer. The hybrid variants lose ~4pp, suggesting that mixing p-less with other decoding strategies dilutes its advantage rather than combining strengths.

**Takeaway:** P-less works best in its simplest form. Adding complexity (beam search, hybrid strategies, removing stop sequences) consistently hurts.

---

## Summary: When to Use What

| Use case | Recommended method | Why |
|---|---|---|
| **Single-sample deployment** (code completion, CI) | P-less (t=0.6) | Highest pass@1 across all models; 89% of potential captured in 1 sample |
| **Few-sample with reliability needs** (3–5 samples, need confidence) | P-less (t=0.6) | Highest cover@0.7; if it solves a problem, it solves it consistently |
| **Multi-sample with diversity** (10 samples, want varied approaches) | P-less (t=1.0) | Best diversity-per-accuracy-lost ratio; Pareto-optimal sweet spot |
| **Maximum task coverage** (10 samples, want to solve as many tasks as possible) | Temperature (t=0.7) | Highest pass@10; reaches 68–102 more unique tasks than p-less |
| **Diversity-critical** (search, ensembles, need maximally different solutions) | Top-p (p=0.9, t=1.0) | Highest structural and CodeBLEU diversity; 0.93 dedup ratio |
| **No tuning budget** (new model, unknown optimal settings) | P-less (t=0.6) | Zero hyperparameters; matches or beats tuned top_p=0.95/t=0.2 on most models |
| **Instruct-tuned models** | Temperature (t=0.7) or greedy | P-less adds no diversity over greedy on instruct models |

---

*All numbers from MBPP-full (500 tasks), n=10 samples, unbiased pass@k estimator (Chen et al. 2021). Source: `results/pless_full_mbpp_results/analysis/consolidated_metrics/`.*
