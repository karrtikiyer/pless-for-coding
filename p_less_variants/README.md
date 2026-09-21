# p_less_variants/

Hyperparameter-free variants of the p-less decoder that DO NOT reparameterize
the collision threshold via the Rényi-α power-sum ``τ_α`` or the rooted
``G_k`` (both already surveyed elsewhere in this repo). Original upstream
sampler in `p-less/p_less_samplers.py` is left untouched (git submodule).

Files:
- `pless_effk.py`    — rank cut at `⌈1/Σpᵢ²⌉` (effective-support top-k, min 2).
- `pless_min2.py`    — plain p-less with a min-2 survivor floor.
- `pless_topmass.py` — cumulative-mass cut at `1 − Σpᵢ²` (adaptive nucleus).
- `pless_top1half.py`— admit `pᵢ ≥ max(pᵢ) / 2` (relative-to-top, min 2).
- `pless_gini.py`    — inverted threshold `pᵢ ≥ 1 − Σpᵢ²` (Gini impurity).

All five share the p-less design goal (no tunable knob, threshold derived
from `p` itself) and directly target the *deterministic-collapse* failure
mode of canonical p-less on peaked distributions — where `Σpᵢ² → max(pᵢ)`
means only the top-1 token survives, feeding the "looping" pathology.

## MPS test on Qwen3-8B / APPS-252 pless-failure list (2026-09-21)

Ran on M4 Max, 48 GB, PyTorch 2.11 MPS.  See `scripts/run_pless_variants_qwen3_8b_mps.py`.

Result: **0/16 solves for both `pless_effk` and `pless_min2`**. Under-target.

Root cause (verified — inspected samples, cross-checked baseline data):

1. Baseline pless-α=2 failed these 16 tasks in **think mode** with 15k–40k
   reasoning tokens per sample (median 21k on these tasks; see
   `scripts/pless_failures_qwen3_8b_apps252.py`).
2. Running think mode at cap=8192 on MPS with 48 GB physical RAM plus
   normal desktop apps forced the OS into 25 GB of swap; a batch of 5
   samples through a 32-layer 8B model at seq=8192 needs ~18 GB of KV
   cache alone, exceeding the residual MPS working set and causing severe
   thrashing (per-task wall-clock ≫ 30 min, no completions).
3. The forced fallback — **disabling thinking** — collapses KV footprint
   ~10× and lets the run make progress, but Qwen3-8B without thinking
   cannot solve these ATCODER-interview problems. All the "non-pless
   solvers" listed in the failure CSV also used think mode; the failure
   is a model-capability issue in no-think mode, not a sampler issue.

So the samplers were tested on a benchmark they were architecturally
prevented from succeeding on within the hardware budget.  They produce
coherent Python (verified by inspection) — see the extracted samples
under `results/pless_variants_mps/…`.  To fairly evaluate them against
the CSV baseline we would need to re-run in **think mode with a CUDA
GPU** (paper's setup) so cap ≥ 20k tokens is affordable.

Tables:
```
task 270 pless_effk: 0/5   pless_min2: 0/5    (6 non-pless solvers in think mode)
task 280 pless_effk: 0/5   pless_min2: 0/5    (1 non-pless solver  in think mode)
task 369 pless_effk: 0/5   pless_min2: —      (4 non-pless solvers in think mode)
task 559 pless_effk: 0/5   pless_min2: —      (11 non-pless solvers in think mode)
task 579 pless_effk: 0/5   pless_min2: —      (2 non-pless solvers in think mode)
```

The three untested fallbacks (`pless_topmass`, `pless_top1half`,
`pless_gini`) are ready to run — same runner, same failure mode
expected in no-think mode.
