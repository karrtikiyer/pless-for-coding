# Rényi-α P-less Full Sweep — m-a-p/OpenCodeInterpreter-DS-1.3B

**Verdict: PATTERN REPLICATES on a small (1.3B) model from a third
training family. α=5 lifts pass@10 by +11.0 pp on MBPP-500 and +7.9 pp
on HumanEval-164 over the α=2 baseline, with the same small pass@1
cost (~1.5–3 pp) and the same monotonic diversity climb seen on
Qwen2.5-Coder and CodeLlama. m-a-p's structural-diversity metric is
informative (unlike CodeLlama's), so the cross-model story is
complete on three diversity axes (struct_div, cb_div, NAUADC pending).**

500 MBPP-full problems × 10 samples and 164 HumanEval problems × 10
samples each, T=1.0, HF backend. 4 α-arms compared against each
other (no T-baseline available for this model — see "Caveats").

## Headline tables

### MBPP-500

| Config         | pass@1 | pass@3 | pass@5 | pass@10 | cov@0.3 | cov@0.5 | struct_div | cb_div |
|----------------|-------:|-------:|-------:|--------:|--------:|--------:|-----------:|-------:|
| α=2.0 (sanity) | 47.68% | 52.61% | 54.25% |  55.40% |   52.6% |   48.6% |     0.0903 | 0.1749 |
| α=2.5          | 47.28% | 55.43% | 58.43% |  61.40% |   54.2% |   48.2% |     0.2018 | 0.3515 |
| α=3.0          | **48.00%** | 57.95% | 61.42% | 65.00% | 56.8% | **49.6%** |  0.2535 | 0.4371 |
| α=5.0          | 46.26% | **57.28%** | **61.40%** | **66.40%** | 55.0% | 47.8% | **0.3419** | **0.5414** |

### HumanEval-164

| Config         | pass@1 | pass@3 | pass@5 | pass@10 | cov@0.3 | cov@0.5 | struct_div | cb_div |
|----------------|-------:|-------:|-------:|--------:|--------:|--------:|-----------:|-------:|
| α=2.0 (sanity) | 58.60% | 67.98% | 71.59% |  75.61% |   65.9% |   60.4% |     0.1235 | 0.1360 |
| α=2.5          | 56.95% | 70.39% | 75.01% |  79.88% |   68.9% |   59.8% |     0.2062 | 0.1988 |
| α=3.0          | 55.85% | 69.87% | 74.37% |  78.66% | **69.5%** |   59.1% |  0.2250 | 0.1998 |
| α=5.0          | 55.61% | **72.20%** | **77.55%** | **83.54%** | 69.5% | **59.8%** | **0.3297** | **0.2665** |

## Δ vs α=2 baseline (the diversity lever moving with α)

### MBPP

| Arm        | Δpass@1 | Δpass@10 | Δstruct_div | Δcb_div |
|------------|--------:|---------:|------------:|--------:|
| α=2.5      | −0.40 pp | +6.00 pp | +0.1115 | +0.1766 |
| α=3.0      | +0.32 pp | +9.60 pp | +0.1632 | +0.2622 |
| **α=5.0**  | **−1.42 pp** | **+11.00 pp** | **+0.2516** | **+0.3665** |

### HumanEval

| Arm        | Δpass@1 | Δpass@10 | Δstruct_div | Δcb_div |
|------------|--------:|---------:|------------:|--------:|
| α=2.5      | −1.65 pp | +4.27 pp  | +0.0827 | +0.0628 |
| α=3.0      | −2.75 pp | +3.05 pp  | +0.1015 | +0.0638 |
| **α=5.0**  | **−2.99 pp** | **+7.93 pp** | **+0.2062** | **+0.1305** |

## Key findings

1. **The α-sweep pattern replicates on a 1.3B model from a third
   training family** (m-a-p OpenCodeInterpreter is based on DeepSeek-Coder
   architecture, distinct from both Qwen and Llama lineages). Same
   monotonic pass@10 lift, same modest pass@1 cost, same diversity
   climb. Cross-family generalization confirmed at small scale.

2. **The pass@10 lift is large in absolute terms** despite the small
   model size:
   - MBPP: 55.40 → 66.40 (+11.00 pp absolute, +19.9% relative)
   - HumanEval: 75.61 → 83.54 (+7.93 pp absolute, +10.5% relative)
   - Comparable to the lifts seen on the 7B models. The α-sweep
     doesn't degrade with model scale.

3. **A 1.3B model hitting 83.5% pass@10 on HumanEval** is remarkable.
   Compared to CodeLlama-7B-Instruct's 46.95% pass@10 at α=5 on the
   same benchmark, this 1.3B model is **substantially stronger** —
   suggesting OpenCodeInterpreter's instruct training is particularly
   well-targeted at code completion benchmarks.

4. **Pass@1 stays high through α=3.0** (peaks at α=3.0 with 48.00% on
   MBPP — actually higher than α=2.0!) before dropping at α=5.0.
   Unique to this model: α=3.0 is the Pareto sweet spot here on MBPP,
   while α=5.0 is the sweet spot on HumanEval. On Qwen and CodeLlama,
   α=2.5 was the typical Pareto-optimum operating point. The optimum
   may shift with model size / training mix.

5. **Structural diversity is informative on this model** (unlike on
   CodeLlama-7B-Instruct where struct_div ≈ 0 across all α). This
   is consistent with the "smaller weaker models have less canonical
   solution distributions" intuition — OpenCodeInterpreter-DS-1.3B
   gives algorithmically diverse correct answers visible at the
   AST-fingerprint level, while CodeLlama-7B clusters everything
   to a single template.

## NAUADC — algorithmic diversity (Claude-Sonnet-4.6 judge, MBPP)

Added 2026-05-19. Completes the 3-model NAUADC matrix on MBPP.

| Config | NAUADC | EA     | DA@10 | Δ NAUADC vs α=2 |
|--------|-------:|-------:|------:|----------------:|
| α=2.0  | 1.0727 | 1.0641 | 1.0794 |          — |
| α=2.5  | 1.1318 | 1.1132 | 1.1431 |     +5.51% |
| α=3.0  | 1.1652 | 1.1313 | 1.1817 |     +8.62% |
| α=5.0  | 1.2095 | 1.1692 | 1.2295 | **+12.75%** |

Monotonic in α — same pattern as Qwen and CodeLlama.

**m-a-p has the highest absolute NAUADC of the 3-model sweep at every
matched α.** Despite being the smallest model (1.3B vs 7B), it produces
the most algorithmically diverse correct solutions per problem. Two
candidate explanations (not mutually exclusive):

1. **Less RLHF flattening** — m-a-p's instruction-tuning is shorter
   than Qwen's, so the model's natural variation isn't squashed.
2. **DeepSeek-Coder lineage** — different training mix may favor
   broader algorithmic coverage of common patterns.

Combined with the absolute pass@k numbers (~55% pass@10 baseline,
67% at α=5), this is a remarkable small-model showing.

Cost: $32.09 for 8,535 Claude-Sonnet-4.6 calls.

**Cross-model NAUADC summary at α=5.0** (highest-diversity arm):

| Model | pass@1 | pass@10 | struct_div | NAUADC |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 75.3% | 88.0% | 0.210 | 1.167 |
| CodeLlama-7B-Instruct | 40.3% | 53.2% | 0.008 | 1.119 |
| **m-a-p OpenCodeInterpreter-DS-1.3B** | 46.3% | 66.4% | 0.342 | **1.209** |

m-a-p wins on NAUADC and is middle on pass@10. Best
algorithmic-diversity-per-parameter among the three.

## Is it just temperature? (resolved 2026-05-19)

The pless@T={1.0, 1.5, 2.0} MBPP baselines were added on 2026-05-19,
making this comparison rigorous for the first time on this model.

| Config       | pass@1   | pass@10  | struct_div | cb_div   |
|--------------|---------:|---------:|-----------:|---------:|
| **α=2.0**    | 47.68%   | 55.40%   | 0.0903     | 0.1749   |
| pless@T=1.0  | 47.74%   | 55.60%   | 0.0853     | 0.1687   |
| α=2.5        | 47.28%   | 61.40%   | 0.2018     | 0.3515   |
| pless@T=1.5  | 47.14%   | 61.60%   | 0.2530     | 0.4019   |
| **α=3.0**    | **48.00%** | **65.00%** | 0.2535 | 0.4371   |
| α=5.0        | 46.26%   | 66.40%   | 0.3419     | 0.5414   |
| pless@T=2.0  | 45.82%   | **67.20%** | **0.4885** | **0.6463** |

### Three findings worth flagging

1. **α=2.0 ≡ pless@T=1.0 sanity gate clears cleanly**: Δpass@1 −0.06 pp,
   Δpass@10 −0.20 pp, Δstruct_div +0.005. Within sampling noise. Third
   model's sanity gate done.

2. **α=3.0 strictly Pareto-dominates pless@T=1.5 on both axes**:
   - pass@1: 48.00% vs 47.14% → **+0.86 pp**
   - pass@10: 65.00% vs 61.60% → **+3.40 pp**
   - struct_div: 0.254 vs 0.253 → basically tied
   
   This is the **cleanest strict Pareto-dominance result in the entire
   3-model sweep**. On Qwen and CodeLlama the α wins required accepting
   a small pass@1 cost; on m-a-p the α arm wins BOTH pass@1 and pass@10
   over its T counterpart.

3. **α=5.0 vs pless@T=2.0**: nearly identical pass@10 (66.4 vs 67.2,
   −0.80 pp) with slightly higher pass@1 (+0.44 pp, 46.26 vs 45.82).
   Pareto-comparable points. **Temperature wins more raw diversity
   here** (T=2.0 sd=0.489, α=5.0 sd=0.342) — for AST diversity alone,
   pushing T past α gives more variation; for the (pass@1, pass@10,
   diversity) joint frontier the choice is more nuanced.

### Mild pass@1 cliff at T=2.0

Pass@1 drops 47.14 → 45.82 going T=1.5 → T=2.0, a 1.32 pp drop. Smaller
than the cliff on Qwen (−4.20 pp at T=1.5 → T=2.0) or CodeLlama
(−4.08 pp) — possibly because this smaller model is less RLHF-flattened
and tolerates higher temperatures more gracefully. T=2.5/3.0 not yet
measured here; expect the cliff to deepen there.

## Cross-model context

| Model | MBPP Δp@10 (α=2→α=5) | HumanEval Δp@10 (α=2→α=5) | MBPP pass@10 max | HumanEval pass@10 max |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | +6.00 pp | +1.83 pp (saturated) | 88.00% | 91.46% |
| CodeLlama-7B-Instruct | +9.00 pp | +14.63 pp | 53.20% | 46.95% |
| **m-a-p/OpenCodeInterpreter-DS-1.3B** | **+11.00 pp** | **+7.93 pp** | 66.40% | **83.54%** |

m-a-p sits in an interesting position: stronger than CodeLlama on
HumanEval, weaker than Qwen on MBPP, with the biggest absolute MBPP
Δpass@10 of any model in the sweep. The α-sweep produces real lift
across the full strength spectrum.

## Pareto frontier (pass@1 vs pass@10) — visualization

```
MBPP                                       HumanEval

pass@10                                    pass@10
   |                                          |
66 ┤  *α=5.0                                84 ┤  *α=5.0
65 ┤  *α=3.0                                80 ┤  *α=2.5
61 ┤  *α=2.5                                79 ┤  *α=3.0
55 ┤  *α=2.0                                76 ┤  *α=2.0
   |                                          |
   └──────────────────────────────→            └──────────────────────────────→
      46    47    48    49                       55    56    57    58    59
                    pass@1                                       pass@1
```

The α-sweep traces the Pareto frontier cleanly on both benchmarks.
Notice: pass@1 actually PEAKS at α=3.0 on MBPP (48.0% vs 47.7% at α=2.0)
— a rare configuration where α=3.0 Pareto-dominates the baseline on
both axes. On HumanEval, pass@1 declines monotonically as expected.

## Observations specific to this model

1. **The diversity gains are largest on this model on MBPP** (struct_div
   0.090 → 0.342, a 3.8× increase; cb_div 0.175 → 0.541, a 3.1×
   increase). Suggests the model's correct-solution distribution is
   wider than CodeLlama's (where struct_div ≈ 0 throughout).
2. **On HumanEval, the α=3.0 vs α=2.5 ordering inverts** for pass@10
   (α=2.5: 79.88, α=3.0: 78.66). Both higher than α=2.0 (75.61),
   but α=2.5 actually wins pass@10 here over α=3.0. Could be sample
   noise at N=164. α=5.0 still wins overall (83.54).
3. **cov@0.5 is essentially flat across α** on both benchmarks (47–50%
   on MBPP, 59–60% on HumanEval). Diversity primarily lives in the
   "more than 1 correct sample exists" regime (cov@0.1, cov@0.3),
   not in the "majority of samples correct" regime (cov@0.5, cov@0.7).

## Caveats

- ~~**No pless@T baseline on MBPP for this model**~~ — **RESOLVED 2026-05-19**.
  T=1.0, T=1.5, T=2.0 added; α=3.0 now strictly Pareto-dominates
  pless@T=1.5 on this model (cleanest such result in the 3-model
  sweep). See "Is it just temperature?" section above.
- **No HumanEval pless@T sweep in the existing `temprature_results/`
  dir** for this model either — only Qwen and CodeLlama are covered.
- ~~**NAUADC not measured** for this model~~ — **RESOLVED 2026-05-19.**
  See "NAUADC — algorithmic diversity" section above. m-a-p produces
  the highest absolute NAUADC of the 3-model sweep.
- **1.3B size — different model class** from the 7B Qwen and CodeLlama
  pairs. Some patterns (e.g., where the Pareto sweet spot lives in α)
  shift with scale.

## Recommended next steps for this model

1. ~~**Run pless@T={1.0, 1.5, 2.0} on MBPP**~~ — **DONE 2026-05-19.**
   Closed the gap. Pareto-dominance verified (α=3.0 strictly dominates
   T=1.5 on both axes).
2. **NAUADC on MBPP α-arms** (~$10–15 of Claude API spend, ~30–60 min
   wall-clock). Would complete the 3-model × NAUADC matrix.
3. **HumanEval pless@T sweep** to match Qwen and CodeLlama coverage
   (less critical — Qwen + CodeLlama HumanEval T-envelope already
   gives strong evidence; this would just round it out).

## Files

```
results/pless_alpha_full/m-a-p--OpenCodeInterpreter-DS-1.3B/
├── pless_alpha_a2.0_t1.0.jsonl          # 500 × 10 generations (4 files)
├── pless_alpha_a2.5_t1.0.jsonl
├── pless_alpha_a3.0_t1.0.jsonl
├── pless_alpha_a5.0_t1.0.jsonl
├── full_sweep_summary.md                # this file
└── metrics/
    ├── pless_alpha_a2.0_t1.0_metrics.json
    ├── pless_alpha_a2.5_t1.0_metrics.json
    ├── pless_alpha_a3.0_t1.0_metrics.json
    └── pless_alpha_a5.0_t1.0_metrics.json

results/pless_alpha_full_humaneval/m-a-p--OpenCodeInterpreter-DS-1.3B/humaneval/
├── pless_alpha_a*.jsonl                 # 164 × 10 (4 files)
└── metrics/pless_alpha_a*_metrics.json
```
