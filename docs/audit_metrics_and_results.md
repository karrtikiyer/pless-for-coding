# Audit: Metrics, Parsing, and Results

**Date:** 2026-04-01
**Scope:** pass@k computation, code extraction/parsing, diversity metrics (fingerprint, structural, CodeBLEU), generated results and figures.

## Summary

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | Silent exception swallowing in `add_self_codebleu()` | **Medium** | Add warning counter and logging |
| 2 | AST fingerprint parameter-order sensitivity | Low | Document as known limitation |
| 3 | `_strip_after_function` indentation heuristic | Low | No action (mitigated by `_trim_to_compilable`) |
| 4 | Hardcoded temperature inference in `consolidated_eval.py` | Low | Document expected filename patterns |
| 5 | pass@k estimator | **Verified correct** | — |
| 6 | Results/figures spot-check | **Verified correct** | — |

---

## Actionable Findings

### Finding 1: Silent exception swallowing in `add_self_codebleu()`

**File:** `bench/eval/metrics.py:197-198`
**Severity:** Medium

**Issue:** The pairwise CodeBLEU loop uses a bare `except Exception: continue`:

```python
for i in range(len(unique_codes)):
    for j in range(i + 1, len(unique_codes)):
        try:
            res = calc_codebleu(...)
            bleu_scores.append(res["codebleu"])
            ...
        except Exception:
            continue
```

If `calc_codebleu()` fails systematically (e.g., empty code strings, parser errors on unusual syntax), those pairs are silently dropped. This reduces the denominator when averaging, which could **inflate** the reported CodeBLEU diversity — tasks where CodeBLEU fails are excluded entirely, and surviving tasks may skew toward higher or lower diversity in non-obvious ways.

**Self-critique:** The catch *is* necessary — `calc_codebleu` can throw on legitimately malformed code, and crashing the entire 79-config pipeline for one bad pair would be worse. The issue is not the catch itself but the **silence**. If 0 out of 6 pairs fail, this is fine. If 50% of pairs fail, the metric is unreliable — and we'd never know.

**Recommendation:** Add a warning counter. After the loop, log `"WARNING: {n_failed}/{n_total} CodeBLEU pairs failed for task {task_id}"` if `n_failed > 0`. This makes failure visible without changing behavior.

---

### Finding 2: AST fingerprint is parameter-order-sensitive

**File:** `bench/eval/fingerprint.py` — `_NameNormalizer`
**Severity:** Low

**Issue:** The normalizer renames variables to positional placeholders (`_v0`, `_v1`, ...) based on the order they appear as function parameters. Two functions with identical logic but swapped parameter names (`def f(a, b)` vs `def f(b, a)`) produce different fingerprints, counting as "distinct" solutions even though they are semantically identical up to renaming.

**Self-critique:** In practice, this is a non-issue for MBPP and HumanEval. Both benchmarks provide a fixed function signature in the prompt (e.g., `def is_palindrome(s):`), so all generated samples share the same parameter order. The model would have to hallucinate a completely different signature for this to matter. This finding is theoretically correct but practically irrelevant for our current benchmarks.

**Recommendation:** No code change. Note as a known limitation if extending to free-form code generation tasks.

---

### Finding 3: `_strip_after_function` indentation heuristic

**File:** `bench/eval/executor.py` — `_strip_after_function()`
**Severity:** Low

**Issue:** Uses indentation level to detect where the target function body ends. If a model generates valid code with unusual indentation patterns (e.g., a nested helper function whose body continues at a different indentation level), the heuristic could truncate prematurely.

**Self-critique:** This is a non-issue because the extraction pipeline has a two-stage safety net:

1. `_strip_after_function()` — fast heuristic, may over-truncate
2. `_trim_to_compilable()` — fallback that progressively removes trailing lines until `compile()` succeeds

If step 1 over-truncates, step 2 catches it. If step 1 under-truncates (leaves extra code), subsequent steps (`_strip_check_and_main`, `_strip_code_fences`) handle common patterns. The pipeline as a whole is robust.

**Recommendation:** No action needed.

---

### Finding 4: Hardcoded temperature inference in `consolidated_eval.py`

**File:** `bench/eval/consolidated_eval.py`
**Severity:** Low

**Issue:** Temperature values are inferred from filename patterns (e.g., `_t0.6` suffix → temperature 0.6, with a fallback default of 1.0). This works for all current experiments but would silently assign wrong temperatures to files using non-standard naming.

**Self-critique:** All 79 current configs follow the naming convention, so current results are correct. This is a maintenance/extensibility concern, not a correctness bug. The hardcoded defaults are clearly documented in the code's discovery functions.

**Recommendation:** Add a brief comment in the discovery functions listing the expected filename patterns, so future contributors know what conventions to follow.

---

## Verified Correct

### Finding 5: pass@k estimator

**File:** `bench/eval/metrics.py:33`

The `compute_pass_at_k()` function delegates to `estimate_pass_at_k` from the official `human_eval` library (Chen et al., "Evaluating Large Language Models Trained on Code", 2021). This implements the standard unbiased estimator:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

where `n` = total samples, `c` = correct samples, `k` = the k value. This is the gold-standard implementation used across the field. **No issues found.**

### Finding 6: Results and figures spot-check

Spot-checked a sample of metrics JSON files against generated plots and reports:

- **pass@1 values:** Consistent between `*_metrics.json` files and bar chart heights in comparison figures
- **Diversity scores:** CodeBLEU diversity, structural diversity, and dataflow diversity values in JSON match heatmap cells and Pareto scatter coordinates
- **cover@t curves:** JSON values match plotted line positions
- **Cross-model consistency:** Models with known higher capability (e.g., Qwen2.5-Coder-3B > OCI-DS-1.3B) show correspondingly higher pass@k, as expected

**No anomalies found.** Results are internally consistent.
