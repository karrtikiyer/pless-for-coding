# Source Audit Log

Every numerical claim in `draft.md` must have an entry here of the form:

```
[CN] "<claim text>"
  source: <relative path>:<line range or table row>
  verbatim: "<exact quote from source>"
  status: VERIFIED | DOWNGRADE | REMOVE
  note: <optional clarification, e.g. SE band, confounds>
```

Source files cited in this audit:

- `cb`  = `results/analysis/cross_benchmark_t1_analysis.md`
- `t12` = `results/full_mbpp_pre_post_temp_pless/analysis/t1_t2_comparison_report.md` (Qwen2.5-Coder-**3B base**)
- `inst7b` = `results/full_mbpp_pre_post_temp_pless/analysis/Qwen--Qwen2.5-Coder-7B-Instruct/instruct_t1_comparison_report.md`
- `cmp` = `results/pless_full_mbpp_results/analysis/comparison_report.md` (Llama-2-7B vs paper baselines)
- `fp`  = `results/pless_human_eval_results/full_precision_results/analysis/report.md`
- `ts`  = `results/pless_human_eval_results/temprature_results/analysis/temperature_sweep_report.md`

## Verified entries

```
[C1] "P-less-norm@0.6 ranks 1/19 on Llama-2-7B (base) at 22.3% pass@1,
      ahead of FSD-d (21.2%) and Beam-8 (19.4%)."
  source: cmp:11,13,15,86
  verbatim:
    "| 1 | P-Less Norm (t=0.6) **←** | Ours | 22.3 |"
    "| 3 | FSD-d | Paper | 21.2 |"
    "| 5 | Beam Search | Paper | 19.4 |"
    "- **P-Less Norm (t=0.6)**: rank 1/19"
  status: VERIFIED
  note: Beam-8 is reported in the source as "Beam Search"; paper-side
        method labels are taken from arXiv:2402.06925.
```

```
[C2] "On Qwen2.5-Coder-7B-Instruct (HumanEval-164): pless@0.6 = 87.5% pass@1
      vs greedy = 84.1%."
  source: fp:23,27
  verbatim:
    "| Qwen2.5-Coder-7B-Instruct | greedy | 84.1 | 84.1 | 84.1 | 84.1 | …"
    "| Qwen2.5-Coder-7B-Instruct | pless (t=0.6) | 87.5 | 87.8 | 87.8 | 87.8 | …"
  status: VERIFIED
  note: Δ = +3.4pp; SE on HumanEval-164 ≈ 2.8pp (see C9), so directionally
        positive but borderline-significant on a single comparison.
```

```
[C3] "Codestral-22B pless improves +9.7pp on HumanEval as T1 rises 0.7 → 2.0
      (5.7% → 15.4% pass@1)."
  source: ts:40,43,164
  verbatim:
    "| Codestral-22B | pless | 0.7 | 5.7% | 12.2% | 14.0% | 3.7% | 0.2280 |"
    "| Codestral-22B | pless | 2.0 | 15.4% | 47.6% | 65.2% | 6.7% | 0.6230 |"
    "- Codestral-22B / pless: 5.7% → 15.4% (Δ=-9.7pp)"
  status: VERIFIED
  note: Source line 164 reports "Δ=-9.7pp" using the convention
        Δ=(T=0.7)−(T=2.0); we will phrase this as a +9.7pp gain in T1=2.0
        relative to T1=0.7. Cross_benchmark §4 (`cb:233-237`) flags that
        Codestral and CodeLlama-7b-Base operate at low absolute pass rates
        where 1-3pp shifts cannot be cleanly distinguished from noise on
        164 tasks; treat the magnitude as suggestive and report it with
        the SE-band caveat from C9.
```

```
[C4] "Qwen2.5-Coder-7B (base) pless drops -19.9pp on HumanEval as T1 rises
      0.7 → 2.0 (56.3% → 36.4% pass@1)."
  source: ts:54,57,166
  verbatim:
    "| Qwen2.5-Coder-7B | pless | 0.7 | 56.3% | 67.8% | 70.1% | 57.9% | 0.0970 |"
    "| Qwen2.5-Coder-7B | pless | 2.0 | 36.4% | 73.4% | 83.5% | 36.0% | 0.5651 |"
    "- Qwen2.5-Coder-7B / pless: 56.3% → 36.4% (Δ=+19.9pp)"
  status: VERIFIED
  note: This is the BASE Qwen2.5-Coder-7B (not Instruct); paper plan’s
        original wording was ambiguous. The Instruct variant in the same
        sweep (`ts:135`) drops only -2.4pp over the same range
        (84.8% → 82.4%) — that contrast is the actual robustness story.
```

```
[C5] "Qwen3-Coder-30B shows a marginal pless gain on HumanEval."
  source: fp:32,36 ; ts:82-95
  verbatim (fp:32,36):
    "| Qwen3-Coder-30B-A3B-Instruct | greedy | 75.6 | 75.6 | 75.6 | 75.6 | …"
    "| Qwen3-Coder-30B-A3B-Instruct | pless (t=0.6) | 78.9 | 79.6 | 79.8 | 79.9 | …"
  verbatim (ts:82-95): pless pass@1 across T1∈{0.7,1.0,1.5,2.0,2.5,3.0}
    is 75.2 / 75.5 / 75.3 / 76.2 / 75.4 / 75.3% — a 1.0pp spread.
  status: DOWNGRADE
  note: Two source runs disagree:
          full-precision (fp:36): pless@0.6 = 78.9% vs greedy 75.6% → +3.3pp
          temperature sweep (ts:82): pless@0.7 = 75.2% vs greedy 75.6% → −0.4pp
        Cross-run pipeline differences (cb:18) prevent direct comparison.
        The paper plan’s original "~0.6pp" figure is not directly cited in
        either source. Re-state in draft as: "P-less is essentially flat
        across T1 on Qwen3-Coder-30B (1pp spread), consistent with the
        peaked-distribution analysis in cb:226-231."
```

```
[C6] "On Qwen2.5-Coder-7B-Instruct (MBPP-500): T1 sweet spot 0.7–1.5 (≤0.5pp
      pass@1 cost vs greedy), cliff between T1=1.5 and T1=2.0 (~4pp), and
      catastrophe at T1=3.0 (-69.8pp from T1=2.0)."
  source: inst7b:39-43
  verbatim:
    "| 0.6 | 77.2% | 79.8% | 0.0305 | 0.0808 | 76.0 |"
    "| 1.0 | 77.2% | 82.2% | 0.0586 | 0.1359 | 75.0 |"
    "| 1.5 | 76.7% | 85.8% | 0.1262 | 0.2792 | 74.0 |"
    "| 2.0 | 72.5% | 89.6% | 0.3082 | 0.5587 | 70.4 |"
    "| 3.0 | 2.7% |  18.8% | 0.2645 | 0.4128 |  0.0 |"
  status: VERIFIED
  note: Cliff size adjusted from "~5pp" in paper plan to the verbatim
        -4.2pp at T1=1.5→2.0; the larger -19.9pp/-69.8pp jumps occur at
        T1=2.0→2.5/3.0 (cb:62-64 confirms the cliff is between T1=2.0 and
        T1=2.5 on HumanEval-Instruct).
```

```
[C7] "On Qwen2.5-Coder-7B-Instruct, T2 (post-truncation flattening) is
      dominated by T1: at T1=2.0, T2∈{2,3,4,5} costs 2.0–4.0pp pass@1 for
      0.020–0.034 struct_div gain; at T1=1.0 T2 changes pass@1 by ≤0.7pp
      and reduces struct_div slightly."
  source: inst7b:60-75
  verbatim:
    "| 2.0 | 70.1% | -2.4pp | 0.3278 | +0.0196 | 0.5713 | +0.0126 |"
    "| 5.0 | 68.5% | -4.0pp | 0.3388 | +0.0306 | 0.5818 | +0.0231 |"
    "| 2.0 | 77.9% | +0.7pp | 0.0572 | -0.0014 | 0.1345 | -0.0014 |"
    "| 5.0 | 77.6% | +0.4pp | 0.0562 | -0.0024 | 0.1351 | -0.0008 |"
  status: VERIFIED
  note: Cited range "2-4pp pass@1 for ~0.03 struct_div" from paper plan
        is correct for the T1=2.0 row; spell out the T1=1.0 finding too
        because it strengthens the "T2 dominated" claim.
```

```
[C8] "On Qwen2.5-Coder-3B (base, MBPP-500): pless T1=0.8 reaches 58.7%
      pass@1 with struct_div=0.167 — matching top_p0.95@0.2 (58.2%) at
      higher diversity."
  source: t12:18,87
  verbatim:
    "| 8 | pless t=0.8 **←** | 0.8 | — | 58.7 | 65.7 | 67.7 | 69.2 |
       54.4 | 0.1673 | 0.2980 |"
    "Best new config: pless T1=0.8 (no T2). At 58.7% pass@1, 0.167
       struct_div, 0.298 codebleu_div, it matches top_p0.95/t=0.2 (58.2%)
       while providing a useful diversity level — with zero hyperparameters
       beyond T1."
  status: VERIFIED with model correction
  note: Paper plan attributed these numbers to Qwen2.5-Coder-7B-Instruct;
        the actual report header (`t12:3`) says "Qwen2.5-Coder-3B (base)".
        Use base-3B label in draft. The 7B-Instruct equivalent (inst7b:18,
        pless t=0.6) is 77.2% pass@1, 0.0305 sdiv — a separate data point.
```

```
[C9] "Per-task SE on pass@1 ≈ 1.75pp on MBPP-500 and ≈ 2.8pp on
      HumanEval-164. Differences below ~2 SE are reported as directional."
  source: cb:16
  verbatim:
    "Statistical significance. HumanEval has 164 tasks. At p@1 ~85%,
     SE ≈ √(0.85×0.15/164) ≈ 2.8pp. Any difference below ~3pp is within
     noise. MBPP (500 tasks) has SE ≈ 1.75pp. Claims are flagged with
     confidence levels throughout."
  status: VERIFIED
  note: SE is binomial-proportion at p≈0.85; for lower p the SE is
        smaller. We will report 2.8pp as the conservative HE band and
        1.75pp as the conservative MBPP band.
```

```
[C10] "Every T1-sensitive model collapses between T1=2.0 and T1=3.0; the
       cliff is at T1=2.5 on HumanEval and T1=3.0 on MBPP. Qwen3-Coder-30B
       is the only tested model that does not collapse."
  source: cb:171-180,226-231 ; ts:135,82-87
  verbatim (cb:171-180):
    "| Qwen2.5-7B-Inst | 82.4% | 64.3% | 19.9% | T1=2.5 |"
    "| Qwen2.5-7B-Base | 36.4% |  1.8% |  0.0% | T1=2.5 |"
    "| Qwen3-30B       | 76.2% | 75.4% | 75.3% | **No collapse** |"
    "| CL-7b-Inst      | 27.0% | 19.1% |  4.9% | T1=2.5-3.0 |"
    "| CL-7b-Base      |  1.6% |  0.1% |  0.0% | T1=2.0 (already near zero) |"
    "| Codestral-22B   | 15.4% |  2.1% |  0.0% | T1=2.5 |"
  status: VERIFIED
```

```
[C11] "The quality-filter hypothesis (P-less acting as a quality filter
       at matched diversity) is NOT confirmed on Qwen2.5-Coder-7B-Instruct
       (MBPP). At matched struct_div, Δ pass@1 vs temperature ranges from
       -9.5pp to +0.4pp."
  source: inst7b:51-53,91
  verbatim:
    "| pless T1=1.5 | 76.7% | 0.1262 | temp t=0.2 | 76.8% | 0.0982 | -0.1pp |"
    "| pless T1=2.0 | 72.5% | 0.3082 | temp t=0.8 | 72.0% | 0.2779 | +0.4pp |"
    "| pless T1=3.0 |  2.7% | 0.2645 | temp t=1.5 | 12.3% | 0.2741 | -9.5pp |"
    "The quality-filter hypothesis is NOT confirmed. At matched diversity,
     P-less does NOT beat temperature — the Δ pass@1 ranges from -9.5pp
     to +0.4pp."
  status: VERIFIED
  note: This is a negative result and should be reported as such — it
        argues against using P-less as a substitute for temperature in
        the high-T1 regime on instruct models.
```

```
[C12] "On Llama-2-7B base pless@1.0 essentially matches temperature@0.7
       on pass@10 (40.0 vs 39.0). On CodeLlama-7B base and
       Qwen2.5-Coder-3B base, temperature@0.7 beats pless@1.0 on pass@10
       by 5–8pp."
  source: cmp:64,66 ; results/analysis/consolidated_summary.csv:65,69,94,96
  verbatim (cmp:64,66, Llama-2-7B base Extended Metrics):
    "| P-Less (t=1.0) | 19.8 | 31.3 | 35.5 | 40.0 | … |"
    "| Temperature (t=0.7) | 13.2 | 24.9 | 30.9 | 39.0 | … |"
  verbatim (consolidated_summary.csv:65,69, Qwen2.5-Coder-3B base):
    "Qwen/Qwen2.5-Coder-3B,pless,1.0,500,10,0.5654,0.6638,0.6936,0.722,…"
    "Qwen/Qwen2.5-Coder-3B,temp,0.7,500,10,0.4262,0.6388,0.7061,0.776,…"
  verbatim (consolidated_summary.csv:94,96, CodeLlama-7B base):
    "codellama/CodeLlama-7b-hf,pless,1.0,500,10,0.4142,0.5033,0.5356,0.572,…"
    "codellama/CodeLlama-7b-hf,temp,0.7,500,10,0.3684,0.529,0.5886,0.652,…"
  status: VERIFIED
  note: The earlier draft cited "37.0 vs 24.2" for Llama-2-7B base from
        the consolidated CSV. That row pair is unreliable: the CSV
        contains TWO duplicate temp@0.7 rows for `meta-llama/Llama-2-7b-hf`
        (lines 138 and 140) reporting pass@10 of 24.2 and 44.6
        respectively — pipeline duplication that the canonical paper-
        comparison report does not have. The comparison_report.md numbers
        used for claim C1 are the canonical reference for Llama-2-7B base;
        we cite them here for consistency. CodeLlama-7B base and
        Qwen2.5-Coder-3B base do not have this duplication issue in the
        CSV (single row per (method, t)).
```

## Pending / not yet sourced

(None as of the Phase 4 close-out. Any new claim added during drafting
must be appended here with VERIFIED status before it lands in `draft.md`.)
