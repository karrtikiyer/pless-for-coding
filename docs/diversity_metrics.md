# Diversity Metrics for Code Generation

## 1. Overview

Pass@k measures whether a sampling method can produce *at least one* correct solution in k attempts. But it says nothing about **diversity** — are those correct solutions structurally varied, or are they all minor token permutations of the same approach?

This matters because diverse correct solutions indicate a method is exploring the solution space broadly, not collapsing onto a single mode. Our research question:

> **Given a sampling method (p-less, temperature, top-p, greedy, beam search), how varied are the correct solutions it produces?**

We measure diversity with three metrics of increasing richness: distinct count, structural diversity, and CodeBLEU diversity.

## 2. The Three Diversity Metrics

### 2a. Distinct Count (`num_distinct_correct`) — coarsest

**What it measures:** The number of correct samples with unique AST fingerprints.

**How it works:** Each correct solution is normalized (docstrings stripped, variables renamed to positional placeholders like `_v0`, `_v1`, builtin names preserved) and hashed via SHA-256. Solutions with the same hash are structurally identical. The count of unique hashes is `num_distinct_correct`.

**Example:** If 8/10 samples pass and 5 have distinct AST fingerprints, `num_distinct_correct = 5`.

**Limitation:** Binary granularity. Two solutions that differ by a single AST node are "distinct" — same as two solutions that use completely different algorithms. This metric cannot distinguish "slightly different" from "radically different."

**Implementation:** `add_distinct_counts()` in `bench/eval/metrics.py`; fingerprinting via `ast_fingerprint()` in `bench/eval/fingerprint.py`.

---

### 2b. Structural Diversity (`structural_diversity`) — moderate

**What it measures:** Mean pairwise normalized tree edit distance between ASTs of correct solutions.

**How it works:** Uses the Zhang-Shasha algorithm (via the `zss` library) to compute the minimum edit distance between two normalized AST trees. The edit distance is normalized by dividing by the maximum of the two tree sizes, yielding a score in [0, 1].

**Range:** 0 = all correct solutions have identical ASTs. 1 = maximally different tree structures.

**Limitation:** Operates only at the tree level. Two solutions that use the same control flow but different variable names, different library calls, or different token sequences will have identical structural diversity. It captures *shape* but not *content*.

**Implementation:** `add_structural_diversity()` in `bench/eval/metrics.py`; uses `zss.simple_distance()`.

---

### 2c. CodeBLEU Diversity (`codebleu_diversity`) — richest

**What it measures:** 1 minus the mean pairwise CodeBLEU similarity among correct solutions. Captures token-level, syntactic, and semantic differences in a single composite score.

**Literature:** CodeBLEU was introduced by Ren et al. (2020), *"CodeBLEU: a Method for Automatic Evaluation of Code Synthesis."* It was originally designed to evaluate code generation quality by comparing generated code against reference solutions.

**The CodeBLEU formula:**

```
CodeBLEU = 0.25 * BLEU + 0.25 * BLEU_weight + 0.25 * Match_AST + 0.25 * Match_DF
```

| Component | What it captures | How |
|-----------|-----------------|-----|
| **BLEU** | Token n-gram overlap | Standard BLEU precision over 1-to-4-grams |
| **BLEU_weight** | Keyword-aware n-gram overlap | Same as BLEU but upweights language keywords (`def`, `return`, `for`, etc.) |
| **Match_AST** (syntax match) | Syntactic structure similarity | Fraction of AST sub-trees in one solution that appear in the other |
| **Match_DF** (dataflow match) | Semantic data-flow similarity | Fraction of variable use-definition chains in one solution that match the other |

CodeBLEU returns a **similarity** score in [0, 1]: 1 = identical code, 0 = completely different.

**Our adaptation — Self-CodeBLEU:** We repurpose CodeBLEU from a "generated vs. reference" quality metric into a **pairwise diversity metric among a method's own correct outputs**. Instead of comparing against a gold reference, we compare every pair of correct solutions against each other.

**Implementation:** `add_self_codebleu()` in `bench/eval/metrics.py`; uses `calc_codebleu()` from the `codebleu` package (>=0.7.0).

## 3. How Self-CodeBLEU Diversity Is Computed

### Step-by-step with a concrete example

Consider HumanEval task #42 where p-less generates 10 samples. 6 are correct. After AST fingerprint deduplication, 4 are structurally unique: `S1, S2, S3, S4`.

**Step 1 — All-pairs CodeBLEU similarity** (C(4,2) = 6 pairs):

| Pair | CodeBLEU | syntax_match | dataflow_match |
|------|----------|-------------|---------------|
| (S1, S2) | 0.82 | 0.90 | 0.75 |
| (S1, S3) | 0.65 | 0.70 | 0.60 |
| (S1, S4) | 0.71 | 0.80 | 0.65 |
| (S2, S3) | 0.68 | 0.72 | 0.62 |
| (S2, S4) | 0.74 | 0.85 | 0.70 |
| (S3, S4) | 0.60 | 0.65 | 0.55 |

**Step 2 — Mean similarity per component:**
- mean CodeBLEU = (0.82 + 0.65 + 0.71 + 0.68 + 0.74 + 0.60) / 6 = **0.70**
- mean syntax_match = **0.77**
- mean dataflow_match = **0.645**

**Step 3 — Invert similarity to diversity** (per-task scores):
- `self_codebleu` = 1 - 0.70 = **0.30**
- `self_syntax_match` = 1 - 0.77 = **0.23**
- `self_dataflow_match` = 1 - 0.645 = **0.355**

These are stored in the `per_task` array of the metrics JSON for this task.

**Step 4 — Aggregate across all tasks:**

Repeat for all 164 HumanEval tasks. Only tasks with >=2 correct solutions contribute. Average the per-task values:

```
codebleu_diversity       = mean([0.30, 0.15, 0.42, ...]) = 0.182
syntax_match_diversity   = mean([0.23, 0.10, 0.35, ...]) = 0.145
dataflow_match_diversity = mean([0.355, 0.20, 0.50, ...]) = 0.213
```

These dataset-level numbers are what appear on the Y-axis of the Pareto scatter and in the heatmap cells.

## 4. Why 1 - Similarity?

CodeBLEU is a **similarity** metric (1 = identical, 0 = nothing in common). We need a **diversity** metric where higher = more diverse = better.

Without inversion:
- Greedy decoding (all outputs identical) would score **1.0** (maximum similarity)
- Temperature sampling (varied outputs) would score **0.6** (lower similarity)

This is backwards — greedy looks best even though it produces zero diversity. The inversion `1 - similarity` fixes the direction:
- Greedy: 1 - 1.0 = **0.0** (no diversity, correct)
- Temperature: 1 - 0.6 = **0.4** (some diversity, correct)

The inversion also makes CodeBLEU diversity **directionally consistent** with structural diversity (AST edit distance), which is already a distance metric where higher = more different. All diversity metrics now point the same way on the same axis.

## 5. Metric Comparison

| Metric | Captures tokens? | Captures structure? | Captures data flow? | Type | Range |
|--------|:---:|:---:|:---:|------|-------|
| `num_distinct_correct` | No | Binary (same/diff fingerprint) | No | Count | 0 to n |
| `structural_diversity` | No | Continuous (edit cost) | No | Continuous | [0, 1] |
| `codebleu_diversity` | Yes (n-gram + keyword) | Yes (AST sub-tree match) | Yes (use-def chains) | Continuous | [0, 1] |

**Illustrative example:** Two correct solutions for the same task — one uses a `for` loop, the other uses a list comprehension:

- **Distinct count:** Both are "distinct" (different AST fingerprints) — but so would be two `for` loops with a minor tweak. No granularity.
- **Structural diversity:** Moderate — the AST shapes differ (loop node vs. comprehension node), captured as normalized edit distance.
- **CodeBLEU diversity:** Higher — captures the structural difference (Match_AST) PLUS different token sequences (`for`/`in`/`append` vs. bracket notation) PLUS different data flow patterns (explicit accumulator variable vs. implicit).

**Principled choice:** Use the richest available metric. CodeBLEU diversity subsumes the information in structural diversity and adds token-level and dataflow-level signal. When `codebleu_diversity` is available, prefer it; fall back to `structural_diversity` when it isn't.

## 6. Naming Chain: Code to Plot Labels

| Code computation | Per-task JSON key | Aggregated JSON key | Plot label |
|-----------------|-------------------|--------------------|----|
| `calc_codebleu()` per pair | (intermediate) | — | — |
| `1 - mean(pairs)` per task | `self_codebleu` | — | — |
| Mean across tasks | — | `codebleu_diversity` | "Mean CodeBLEU Diversity" |
| `1 - mean(syntax pairs)` per task | `self_syntax_match` | `syntax_match_diversity` | (sub-component) |
| `1 - mean(dataflow pairs)` per task | `self_dataflow_match` | `dataflow_match_diversity` | "1 - Dataflow Match" |
| Zhang-Shasha edit distance | `mean_pairwise_distance` | `structural_diversity` | "Mean Structural Diversity" |
| AST fingerprint dedup count | `num_distinct_correct` | (per-task only) | "num_distinct_correct" |

**Key point:** Whenever you see "CodeBLEU Diversity" on a plot, it means **Self-CodeBLEU** — pairwise diversity among a method's own correct outputs. There is no external reference involved.

## 7. Implementation References

| Component | File | Key functions |
|-----------|------|---------------|
| AST fingerprinting | `bench/eval/fingerprint.py` | `ast_fingerprint()` |
| Distinct counts | `bench/eval/metrics.py` | `add_distinct_counts()` |
| Structural diversity | `bench/eval/metrics.py` | `add_structural_diversity()`, `compute_structural_diversity()` |
| Self-CodeBLEU | `bench/eval/metrics.py` | `add_self_codebleu()`, `compute_self_codebleu_diversity()` |
| Full pipeline | `bench/eval/metrics.py` | `build_metrics_output()` |
| Visualizations | `bench/eval/plots.py` | `plot_pareto_scatter()`, `plot_method_heatmaps()`, `plot_diversity_metrics_bars()` |
| CodeBLEU package | `pyproject.toml` | `codebleu>=0.7.0` |
| Tree edit distance | `pyproject.toml` | `zss` (Zhang-Shasha) |
