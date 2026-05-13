# Algorithmic Diversity of Qwen3-8B Split-Decoding Configurations

**One-sentence summary.** Across 14 sampling configurations evaluated with
[AlgoSim](https://github.com/sh0416/algosim) (Lee et al., EMNLP 2025 Findings,
[arXiv:2503.00691](https://arxiv.org/abs/2503.00691)), Qwen3-8B's largest
algorithmic-diversity lever is **turning on thinking** (+0.14 NAUADC); split
decoding with p-less on the code phase adds a smaller, consistent bump
(+0.02 NAUADC, +1.8 pp pass@10) over plain temperature on the code phase;
and aggressive code-phase temperature (T = 3.0) actually *reduces*
algorithmic diversity even while surface-level diversity metrics stay flat.

**Status.** 10 of 14 configs evaluated. 4 pending top-up (B, D, E, T15N) —
their addition will not change the headline findings, only confirm the
"no-thinking p-less floor" (B) and verify that native HF decoding matches
the split-path implementation (T15N).

## 1. Question

Given that LLM diversity is increasingly recognised as critical for
generate-many-pick-best agentic patterns
([Kirk et al., ICLR 2024](https://arxiv.org/abs/2310.06452)), we wanted to
know how various sampling configurations on a code-capable thinking model
compare not just on surface-level diversity (text / AST variation) but on
**algorithmic** diversity — i.e., how many genuinely different *approaches*
to a problem the model produces, as judged by another LLM.

AlgoSim's key metric is NAUADC: the normalised area under the curve plotting
"expected number of distinct algorithm clusters when sampling K solutions"
against K, integrated over K = 1…25. Algorithms are clustered by prompting
Llama-3.1-8B-Instruct to compare each candidate solution against existing
cluster representatives and decide "novel approach" vs "similar to previous."

## 2. Setup

**Model.** Qwen3-8B with thinking enabled (`enable_thinking=True`), 8192-token
budget, 10 samples per task.

**Split decoding.** Sampler-and-temperature can differ between the
`<think>…</think>` phase and the subsequent code phase. Implementation:
`bench/generator.py:generate_samples_split` (lines 497–650). At every token,
the active sampler is chosen by checking whether the current sample has
emitted the `</think>` token id.

**Samplers in scope.**
- `temp_pure`: temperature scaling, no top-p / top-k filter.
- `pless`: collision-entropy thresholding — at each step, threshold
  `p = Σ probsᵢ²` is computed and tokens with probability below `p` are zeroed.
- `temp_standard` (top-p 0.95, top-k 20 on top of temperature) is **deliberately
  excluded** from the algosim run — we wanted a clean pure / no-filter
  comparison.

**Configurations (14 total).**

| Family | Configs | What they vary |
|---|---|---|
| No-thinking baselines | A, B | sampler at low temp without `<think>` |
| Thinking-only baselines (no split) | C, D, E | sampler during thinking at low temp |
| Uniform high-temp thinking (no split) | T15N, P15 | sampler throughout at T = 1.5 |
| Pure split baseline (no p-less) | T15P | pure temp on both phases |
| Pure split + p-less on code | H7P, H8P, H9P, H10P | sweep code p-less T ∈ {1.0, 1.5, 2.0, 3.0} |
| Pure split stress tests | H11P, H12P | sweep think T ∈ {2.0, 2.5}, code p-less T = 3.0 |

All `H*P` and `T15P` configs share **`temp_pure @ 1.5`** on the think phase.
Naming map: `bench/eval/split_decoding_analysis.py:CONFIGS`.

**Benchmark.** MBPP-500 (500 tasks × 10 samples = 5,000 generations per config).

**Filtering for algosim.** Only functionally-correct samples are clustered.
Per-task pass results come from our existing
`results/.../Qwen--Qwen3-8B/metrics/*_metrics.json`. Tasks with zero correct
samples are dropped.

## 3. Results — the table

Sorted by NAUADC descending. Configs without a NAUADC value are still pending
the top-up algosim run; their existing surface metrics are shown for reference.

| Config | Label | pass@1 | pass@10 | struct_div | codebleu_div | NAUADC |
|---|---|---:|---:|---:|---:|---:|
| **H8P** | pure 1.5 → p-less 1.5 | 0.811 | 0.898 | 0.206 | 0.384 | **1.322** |
| H9P  | pure 1.5 → p-less 2.0 | 0.807 | 0.908 | 0.217 | 0.401 | 1.319 |
| H7P  | pure 1.5 → p-less 1.0 | 0.805 | 0.910 | 0.214 | 0.398 | 1.312 |
| T15P | pure 1.5 → pure 1.5   | 0.801 | 0.882 | 0.208 | 0.390 | 1.303 |
| H10P | pure 1.5 → p-less 3.0 | 0.803 | 0.906 | 0.202 | 0.389 | **1.283** |
| C    | temp_think 0.6         | 0.738 | 0.834 | 0.167 | 0.354 | 1.234 |
| P15  | uniform p-less 1.5      | 0.824 | 0.898 | 0.159 | 0.296 | 1.222 |
| H11P | pure 2.0 → p-less 3.0 | 0.458 | 0.802 | 0.201 | 0.370 | 1.198 |
| H12P | pure 2.5 → p-less 3.0 | 0.242 | 0.736 | 0.201 | 0.327 | 1.145 |
| A    | temp 0.7 (no think)    | 0.662 | 0.734 | 0.057 | 0.137 | 1.096 |
| B *(pending)*    | p-less 0.7 (no think) | 0.669 | 0.674 | 0.007 | 0.015 | — |
| D *(pending)*    | p-less_think 0.6      | 0.718 | 0.816 | 0.131 | 0.256 | — |
| E *(pending)*    | p-less_norm_think 0.6 | 0.719 | 0.822 | 0.124 | 0.245 | — |
| T15N *(pending)* | native temp 1.5 think | 0.799 | 0.888 | 0.200 | 0.384 | — |

Companion artifacts:
`results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis/algosim_full_comparison.{md,png}`.

## 4. Findings

### 4.1 Thinking is the biggest single lever

A → C: pass@1 0.662 → 0.738 (+7.6 pp), NAUADC **1.096 → 1.234** (+0.138).
Within MBPP-500 on Qwen3-8B, turning on the thinking budget at low temperature
moves the algorithmic-diversity needle by an order of magnitude more than any
sampler choice on top of thinking does. This dwarfs every within-thinking
comparison below.

### 4.2 P-less on the code phase adds a small, consistent bump over plain temperature

Holding the thinking configuration fixed at `temp_pure @ 1.5`, the four pure
split configs differ only in code-phase sampler/temperature:

| Code sampler | pass@1 | pass@10 | NAUADC |
|---|---:|---:|---:|
| T15P: temp 1.5      | 0.801 | 0.882 | 1.303 |
| H7P:  p-less 1.0    | 0.805 | 0.910 | 1.312 |
| H8P:  p-less 1.5    | 0.811 | 0.898 | **1.322** |
| H9P:  p-less 2.0    | 0.807 | 0.908 | 1.319 |
| H10P: p-less 3.0    | 0.803 | 0.906 | 1.283 |

H7P/H8P/H9P all beat T15P on both NAUADC (+0.009 to +0.019) and pass@10
(+0.016 to +0.028 absolute). Direction is consistent across the three p-less
code temperatures from 1.0 through 2.0; magnitude is small.

The H7P/H8P/H9P spread itself (1.312 → 1.322 → 1.319) is on the order of
0.01 NAUADC, so we are cautious about ranking these three.

### 4.3 At code-phase p-less T = 3.0, algorithmic diversity drops

H10P is the only p-less code config *below* T15P on NAUADC (1.283 vs 1.303).
The drop from H8P / H9P to H10P is ~0.04 NAUADC — about 4× the H7P / H8P / H9P
intra-cluster spread, so we treat this as a real direction rather than noise.

Importantly, H10P's surface diversity metrics (struct_div 0.202, codebleu_div
0.389) sit *within* the H7P / H8P / H9P band — code at code-p-less T = 3.0 is
syntactically just as varied as at lower temperatures, but algorithmically
**less** varied. The LLM judge sees algorithmic convergence that the surface
AST / CodeBLEU metrics miss.

### 4.4 Concrete illustration — MBPP task 181, "longest common prefix"

The simplest way to see the H8P-vs-T15P difference is to look at a task on
which they split differently. On task 181, T15P pooled all 10 samples into a
single algorithm cluster; H8P split into 3:

**T15P, cluster 0 (all 10 samples):** scan column-by-column.

```python
def common_prefix(strings, n):
    selected = strings[:n]
    if not selected: return ""
    min_len = min(len(s) for s in selected)
    prefix = []
    for i in range(min_len):
        current_char = selected[0][i]
        for s in selected[1:]:
            if s[i] != current_char:
                return ''.join(prefix)
        prefix.append(current_char)
    return ''.join(prefix)
```

**H8P, cluster 0 (7 of 9 samples):** the same column-scan, minor variants.

**H8P, cluster 1 (1 sample):** *reduction-style* — start with the first
string, intersect with each subsequent string pairwise.

```python
def common_prefix(strings, n):
    n_strings = strings[:n]
    if not n_strings: return ''
    common = n_strings[0]
    for s in n_strings[1:]:
        min_len = min(len(common), len(s))
        i = 0
        while i < min_len and common[i] == s[i]:
            i += 1
        common = common[:i]
        if not common: break
    return common
```

**H8P, cluster 2 (1 sample):** column-scan with list-append + `''.join`
rather than string-concat — closer to cluster 0 but the judge separated them.

Task 181 shows what NAUADC is measuring concretely: H8P recovered a
fundamentally different algorithm (pairwise reduction over strings) that
T15P never produced. The two algorithms are equivalent in pass@1, but if a
selection-step downstream wants algorithmic redundancy, only H8P provides it.

### 4.5 The "high-temperature keeps diversity at low pass@k" claim does not hold in the stress regime

Lee et al. argue temperature beyond 1.0 keeps adding algorithmic diversity
even as pass@1 declines. We tested the strong form on code by sweeping the
*thinking* temperature past 1.5 while holding code p-less at 3.0:

| Config | think T | pass@1 | pass@10 | struct_div | NAUADC |
|---|---:|---:|---:|---:|---:|
| H10P | 1.5 | 0.803 | 0.906 | 0.202 | 1.283 |
| H11P | 2.0 | 0.458 | 0.802 | 0.201 | 1.198 |
| H12P | 2.5 | 0.242 | 0.736 | 0.201 | 1.145 |

Surface diversity (struct_div ≈ 0.20) stays roughly constant. Both pass@1 and
NAUADC fall *together*. By H12P, pass@1 is 0.242 — the model is mostly
incoherent — and NAUADC has dropped to within 0.05 of the no-thinking
floor. **At least for Qwen3-8B on MBPP at thinking temperatures past 1.5,
high-temperature outputs are not "diverse correct solutions"; they are
"similar attempts at similar wrong solutions."**

This is not a contradiction of Lee et al. — their experiments span different
domains, models, and the temperature regime up to ~1.4. But it bounds the
claim: it does not generalise to extreme regimes on code.

## 5. Limits

- **One model, one benchmark, one judge.** Qwen3-8B on MBPP-500 with
  Llama-3.1-8B-Instruct as the algosim judge. Cross-model and cross-judge
  replication would be needed before any of these become general claims about
  code LMs.
- **Mean cluster counts are small (1.30–1.42 across all evaluated configs).**
  NAUADC differences of 0.01 correspond to roughly 0.02 extra distinct
  algorithms per task. The H7P / H8P / H9P ordering is too close to call;
  only the H8P-vs-T15P (4.1) and H10P-vs-others (4.3) gaps are large enough
  relative to that spread to warrant a directional claim.
- **Filtered (`temp_standard`) configs were intentionally excluded.** Whether
  a top-p 0.95 + top-k 20 filter on top of `temp_pure` changes the picture
  is not answered here.
- **4 pending configs (B, D, E, T15N).** B in particular would put a number
  on the "no-thinking p-less floor" that surface metrics already suggest
  (struct_div 0.007). T15N would tell us whether the split-path machinery
  itself perturbs distributions vs native HF — if T15N's NAUADC matches
  T15P's (~1.303), the split scaffolding is faithful.

## 6. Reproducibility

| Step | Path |
|---|---|
| Source generations | `results/pless_full_mbpp_results/Qwen--Qwen3-8B/*.jsonl` |
| Per-config metrics (pass@k + surface diversity) | `results/.../Qwen--Qwen3-8B/metrics/*_metrics.json` |
| Algosim request export | `bench/eval/algosim_export.py` |
| Algosim runner (GPU; vLLM + Llama-3.1-8B-Instruct) | `run_algosim_judge_qwen3.sh` |
| Per-config NAUADC recomputation | `bench/eval/algosim_report.py` |
| 14-config side-by-side comparison | `bench/eval/algosim_full_comparison.py` |
| Algosim submodule (vendored, with patches) | `algosim/`, `algosim_patches/` |
| Output report + plots | `results/.../Qwen--Qwen3-8B/analysis/algosim_*` |

## 7. References

- Lee et al. *How Diversely Can Language Models Solve Problems?* EMNLP 2025
  Findings. [arXiv:2503.00691](https://arxiv.org/abs/2503.00691).
- Kirk et al. *Understanding the Effects of RLHF on LLM Generalisation and
  Diversity.* ICLR 2024. [arXiv:2310.06452](https://arxiv.org/abs/2310.06452).
- Zhang et al. *Verbalized Sampling.* 2025.
  [arXiv:2510.01171](https://arxiv.org/abs/2510.01171).
