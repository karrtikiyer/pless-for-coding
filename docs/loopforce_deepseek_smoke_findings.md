# DeepSeek-R1-Distill-Llama-8B loop-force — smoke (25-task) findings

**Status:** smoke only (25 hardest tasks, 8/25 globally-unsolvable). **Full 252 pending.**
Run: `results/loop_forcethink_deepseek_w3000/` (n=30, k=6, window=3000). Numbers below are
reproduced by `scripts/completion_breakdown.py` and `scripts/detector_deepseek_nk_grid.py`.

## 1. The model-aware `</think>` fix (E8) works end-to-end

The smoke ran to completion with no crash — confirming `resolve_think_end_id` correctly
resolves DeepSeek's `</think>`=128014 (vs Qwen3's 151668; DeepSeek vocab=128000, so the old
hardcoded id would have `IndexError`-ed at `row[151668]`). Truncation collapsed: pless
74.8→1.2%, pless_norm 75.2→0.4% (matched on the same 25 tasks). The detector fired on ~46–50%
of samples. **The mechanism is validated; that was the smoke's job.**

## 2. But on these 25 hard tasks, loop-force bought no accuracy — it relabeled the failure

Matched on the same 25 task_ids (DeepSeek baseline vs w3000 loop-force):

| arm | trunc% | cond-corr | pass@1 | pass@10 |
|---|---|---|---|---|
| pless baseline (no-force) | 74.8 | 0.397 | 0.100 | 0.200 |
| pless loop-force w3000 | 1.2 | **0.081** | 0.080 | 0.200 |
| pnorm baseline (no-force) | 75.2 | 0.339 | 0.084 | 0.160 |
| pnorm loop-force w3000 | 0.4 | **0.116** | 0.116 | 0.200 |

pass@1 flat (noise), pass@10 flat; conditional-correctness **collapsed** (0.40→0.08). Forcing
`</think>` on a stuck loop converts a truncated-fail into a completed-but-wrong (or no-code) fail.

## 3. The clean diagnostic — `closed-but-no-code%` (`scripts/completion_breakdown.py`)

`cot_efficiency` splits samples three ways (they do NOT sum as compl+trunc): truncated (no
`</think>`) + closed-no-code (`</think>` but no extractable code) + completed (`</think>` + code).
The **closed-no-code** rate is the diagnostic for whether the model had a solution when it stopped:

| config (force-`</think>`) | trunc% | closed-no-code% | completed% |
|---|---|---|---|
| Qwen3 pless w1200 (full 252) | 1.5 | **0.4** | 98.1 |
| Qwen3 pless w1200 (same 25) | 1.6 | **0.4** | 98.0 |
| Qwen3 pless BASE (full 252) | 14.5 | 0.0 | 85.5 |
| DeepSeek pless w3000 (25 smoke) | 1.2 | **40.8** | 58.0 |
| DeepSeek pnorm w3000 (25 smoke) | 0.4 | **33.6** | 66.0 |
| DeepSeek pless BASE (full 252) | 64.9 | 0.0 | 35.1 |

Same tasks, same intervention, only the model differs: **Qwen3 produces extractable code 98% of
the time when forced; DeepSeek only 58–66%.** This is the project's "solution-existence is the
disease, not the loop" finding made quantitative — Qwen3's loops are recoverable overthinking;
DeepSeek's (on these hard tasks) are dead-ends with nothing to write.

## 4. Detector re-tune for DeepSeek: k=6 → k=8 (`scripts/detector_deepseek_nk_grid.py`)

n=30/k=6 was carried over from Qwen3 without a DeepSeek-specific check. The grid (window=3000):

| k | FP% (good reasoning fired on) | catch% (loops) | median fire pos |
|---|---|---|---|
| 6 (carried over) | **1.5** | 98.5 | 5,430 |
| **8** | **0.0** | 93.5 | 5,830 |
| 10 | 0.0 | 87.0 | 5,830 |
| 12 | 0.0 | 82.0 | 6,030 |

**Use k=8 for the DeepSeek full run** — it zeroes the false-positive on productive reasoning at
only ~5pp catch cost. Note the fire position barely moves with k (5.4K→6.4K even at k=16):
DeepSeek's loops onset early (~5K) and are dense, so the ~6K think length is intrinsic, not an
over-aggression artifact (forced and natural-close traces were the same ~5.4K length; productive
passing reasoning on these tasks is ~7K median, p25–p75 4.4K–9.4K).

## 5. What's pending / caveats

- **The 25-slice is the worst case** (8/25 unsolvable, baseline pass@1 0.10). On the **full 252**,
  DeepSeek pless cond-correctness is 0.4955 — the easier ~227 tasks have real solutions, where
  forcing should produce passing code (like Qwen3). The full run at **k=8** is the real verdict.
- Run: `MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B TOKENIZER=… LOOP_N=30 LOOP_K=8
  LOOP_WINDOW=3000 RESULTS_DIR=results/loop_forcethink_deepseek_w3000_k8 …
  ./run_loop_forcethink_apps_qwen3.sh` (fresh dir — k=8 ≠ the k=6 smoke, so no resume).
- Even on the full set, expect loop-force to keep relabeling more than rescuing where loops are
  dead-ends; the `closed-no-code%` column is the metric to watch.
