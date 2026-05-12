# Table 4 — Cross-benchmark replication (MBPP-500 vs HumanEval-164)

Five conclusions derived from MBPP-500 are tested against freshly computed
HumanEval-164 metrics. Confidence labels copy the source report; an SE band
of 1.75pp (MBPP) and 2.8pp (HumanEval) is assumed throughout.

| # | Conclusion | MBPP-500 | HumanEval-164 | Replicated? | Confidence |
|---|------------|----------|---------------|-------------|------------|
| 1 | Sweet spot T1=0.7–1.5 (≤1pp pass@1 cost) | yes | yes (within SE) | ✓ | HIGH |
| 2 | Catastrophe between T1=2.0 and T1=3.0    | yes (cliff at T1=2.0→3.0) | yes (cliff at T1=2.0→2.5) | ✓ | HIGH |
| 3 | T1 is the dominant diversity knob (vs T2) | yes (3–17× efficiency) | tested via T1 only; behaves as expected | ✓ | HIGH for T1; MBPP-only for T2 |
| 4 | P-less ≥ greedy on instruct models       | within noise (≈0pp) | directionally above (+1.2 to +3.3pp across two runs) | partial | MEDIUM |
| 5 | Instruct models are more peaked than base | yes (5–10× lower struct_div) | yes (10× lower struct_div at matched T1) | ✓ | HIGH |

_Source: `results/analysis/cross_benchmark_t1_analysis.md` §5 (lines 247–270)._

## Companion: peakedness comparison (Qwen2.5-Coder-7B-Instruct)

| T1 | MBPP struct_div | HumanEval struct_div | HE / MBPP ratio |
|:---|----------------:|---------------------:|----------------:|
| 0.7 | ~0.030 | 0.009 | 0.30× |
| 1.0 | 0.059  | 0.016 | 0.27× |
| 1.5 | 0.126  | 0.049 | 0.39× |
| 2.0 | 0.308  | 0.161 | 0.52× |

_Source: `cross_benchmark_t1_analysis.md:197-202`._

**Interpretation tag (for §4.5):** HumanEval is 2–4× more peaked than MBPP on
the same model — the same T1 produces less structural diversity. T1 must be
pushed further on HumanEval to obtain comparable diversity, but pass@1 also
degrades more slowly there, so the operating window is similar in shape if
shifted toward higher T1.
