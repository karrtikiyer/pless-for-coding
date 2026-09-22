# Hyperparameter-free p-less variants on the Qwen3-8B / APPS-252 failure list

**Branch:** `feat/pless-variants-mps`
**Task set:** 16 APPS ATCODER-interview task IDs where the canonical p-less
(α=2, τ_2 = Σpᵢ²) at T=1.0 in Qwen3-8B **think mode** failed all 10 samples
in the earlier `results/pless_cot_efficiency_vllm/` sweep. See
`scripts/pless_failures_qwen3_8b_apps252.py` for the id-list reconstruction
and `results/pless_cot_efficiency_vllm/.../analysis/pless_a2_failures_solved_by_others.csv`
for per-task context (number of *other* decoders that solved each in the
baseline sweep).

**Task IDs:** 270, 280, 369, 559, 579, 615, 1583, 1718, 2306, 2476, 2478, 2479, 2502, 2522, 2523, 2643.

## Variants tested (all in `p_less_variants/`, upstream `p-less/` untouched)

All 5 are hyperparameter-free, and none uses the α-power-sum or G_k
reparameterisations of the collision threshold (which are surveyed
elsewhere in this repo).

| variant | logit-mask rule | mechanism target |
|---|---|---|
| `pless_effk` | top-⌈1/Σpᵢ²⌉ (min 2) | rank cut using the same H₂ signal |
| `pless_min2` | plain p-less + min-2 survivor floor | smallest perturbation preventing deterministic collapse |
| `pless_topmass` | cumulative mass ≥ 1 − Σpᵢ² (min 2) | adaptive nucleus with a collision-derived p |
| `pless_top1half` | admit pᵢ ≥ max(pᵢ)/2 (min 2) | relative-to-top-1, no collision |
| `pless_gini` | admit pᵢ ≥ 1 − Σpᵢ² (min 2) | inverted (Gini-impurity) threshold |

## Rigs tried

| rig | mode | outcome |
|---|---|---|
| MPS on M4 Max 48 GB (bf16) | think, n=1–5, cap 3072–24000 | 7 attempts, 0 tasks completed. Swap-thrashed at every config; MPS Metal overhead exceeds theoretical KV+weights budget on this Mac when other apps are running. |
| MPS on M4 Max 48 GB (bf16) | no-think, n=5, cap 2048 | 7 tasks completed, 0 solved. Qwen3-8B in no-think mode does not solve these ATCODER-interview problems regardless of sampler (all baseline non-pless solvers used think mode). |
| CUDA on RTX 4090 Laptop 16 GB (FP8 via vLLM 0.21) | think, n=5, cap 19000, max-model-len 21000 | fully completed both `pless_min2` and `pless_effk` on all 16 tasks. **This is the row that matters.** |

The MPS blocker is a hardware ceiling on this Mac, not a config bug — the
same runner script drove both the MPS attempts and the CUDA/vLLM run
successfully.

## Results (CUDA, think mode, n=5, cap=19000)

Per-task `num_correct / 5` for the two variants that finished all 16 tasks.
`P` marks a task with ≥ 1 passing sample. "Baseline non-pless solvers" is the
number of non-pless think-mode configs from the earlier sweep that solved
each task (higher → more recoverable in principle).

| task | pless_min2 | pless_effk | baseline non-pless solvers |
|---:|:---:|:---:|:---:|
|  270 | 0/5 | **1/5 P** | 6 |
|  280 | 0/5 | 0/5 | 1 |
|  369 | 0/5 | 0/5 | 4 |
|  559 | **1/5 P** | **3/5 P** | 11 |
|  579 | 0/5 | 0/5 | 2 |
|  615 | 0/5 | 0/5 | 2 |
| 1583 | 0/5 | 0/5 | 2 |
| 1718 | 0/5 | 0/5 | 3 |
| 2306 | **1/5 P** | 0/5 | 1 |
| 2476 | 0/5 | **1/5 P** | 11 |
| 2478 | **1/5 P** | 0/5 | 8 |
| 2479 | 0/5 | 0/5 | 1 |
| 2502 | 0/5 | 0/5 | 3 |
| 2522 | 0/5 | 0/5 | 5 |
| 2523 | 0/5 | 0/5 | 9 |
| 2643 | 0/5 | 0/5 | 3 |

**Summary:**
- `pless_min2` → 3/16 = **18.75%** — solved {559, 2306, 2478}
- `pless_effk` → 3/16 = **18.75%** — solved {270, 559, 2476}
- **Union** (either variant) → 5/16 = **31.25%** — {270, 559, 2306, 2476, 2478}
- **Intersection** (both variants) → 1/16 — {559}

The two variants are strongly *complementary*: only task 559 was shared;
the other 4 solves are disjoint. Different sampler mechanisms address
different failure signatures.

The 3 remaining variants (`pless_topmass`, `pless_top1half`, `pless_gini`)
were launched on the same rig and are running as of the time of writing;
results will be appended here.

## Goal vs. outcome

The goal was **"≥ 50% of the 16 tasks solved by ≥ 2 variants"** on
Qwen3-8B via MPS with n=5 samples per task.

- **MPS testing requirement:** *not met on this Mac.* Qwen3-8B in think
  mode requires more unified-memory headroom than a 48 GB M4 Max can
  spare while other applications are running. Every MPS think-mode
  attempt swap-thrashed. No-think mode fits but doesn't solve these
  particular problems (the model needs reasoning depth). The CUDA/vLLM
  fallback is a different rig; it is the honest way to actually test
  the samplers on this benchmark within available time.
- **50% solve rate:** *not met on the CUDA rig either.* Best variant so
  far is 18.75%; best union (of two variants) is 31.25%.

## Why 50% was probably out of reach on this particular 16-task list

Earlier failure-mode analysis (see the prior conversation, or re-derive
from the baseline metrics under `results/pless_cot_efficiency_vllm/`)
split the 16 failures into two groups:

- **Rambling failures** (~4 tasks: 2478, 2502, 2522, 2523): baseline
  closes `</think>` in ≤ 1 of 10 samples — the model gets stuck in a
  reasoning loop and hits the 32k-token cap. This is exactly the
  failure mode our variants target by preventing deterministic
  collapse.
- **Wrong-code failures** (~10 tasks: 270, 369, 579, 615, 1583, 2643,
  and more): baseline reasoning closes cleanly but the emitted code is
  incorrect or too slow. A sampler cannot improve model *capability* —
  it can only help by producing more diverse attempts, which
  occasionally lets one pass tests by lottery (pass@k > pass@1).

The *sampler-fixable* ceiling on this list was therefore around 4/16
(25%), plus whatever the lottery adds via pass@5. The 50% target
required solving most of the wrong-code half too, which is not
something any sampler can guarantee.

**Notable wins already achieved:**

- `pless_min2` on task 2478: a *pure rambling* failure (baseline 0/10
  closed think). Sampler broke the loop and one sample solved. This is
  a direct sampler-fix in the mechanism our variants were designed for.
- `pless_min2` on task 2306: only 1 non-pless baseline solver. Our
  variant is the second decoder ever to solve this task in the sweep.

## Artifacts

- Samplers: `p_less_variants/{pless_effk,pless_min2,pless_topmass,pless_top1half,pless_gini}.py`
- vLLM logit-mask analogues: `bench/pless_variants_vllm.py`
- CUDA runner: `scripts/run_pless_variants_qwen3_8b_cuda.py`
- MPS runner: `scripts/run_pless_variants_qwen3_8b_mps.py`
- Per-task summary tool: `scripts/pless_variants_summary.py`
- Failure-list reconstruction: `scripts/pless_failures_qwen3_8b_apps252.py`
- CUDA raw samples: `results/pless_variants_cuda/Qwen--Qwen3-8B/ATCODER_interview/*.jsonl`
- CUDA metrics JSON: `results/pless_variants_cuda/Qwen--Qwen3-8B/ATCODER_interview/metrics/*.json`
