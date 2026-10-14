# Laptop batch-0 pause checkpoint — 2026-10-14

Recorded immediately before intentionally stopping the Laptop production job
(PID 18438) to free its GPU for the Phase 3 multi-arm-vs-single-arm
equivalence pilot (`scripts/run_phase3_pilot.py`). This file exists so the
exact restart command and progress state survive even if the current chat
session ends before we get back to it.

## State at time of pause

- **Timestamp**: 2026-10-14 14:11:18 IST
- **Host**: Laptop (`ssh karrtik@100.94.82.96`)
- **PID**: 18438
- **Elapsed runtime so far**: 4 days, 3h 41m
- **Working directory**: `/home/karrtik/pless-for-coding`

## Exact restart command

Run this on Laptop, from `/home/karrtik/pless-for-coding`, to resume exactly
where it left off (the script's own resume logic will skip every
already-completed task automatically — no manual coordination needed):

```bash
cd /home/karrtik/pless-for-coding
PYTHONPATH=/home/karrtik/pless-for-coding \
PATH=/home/karrtik/pless-for-coding/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup .venv/bin/python scripts/run_pless_variants_ministral_3b_lcb_cuda.py \
  --model mistralai/Ministral-3-3B-Reasoning-2512 --dtype bfloat16 \
  --gpu-memory-utilization 0.92 --max-model-len 37500 \
  --n-samples 10 --max-new-tokens 35000 \
  --configs pless_default pless_effk pless_effk_nofloor pless_topmass top_p_0_95 top_nsigma_1_5 pless_gini \
  --full-dataset --batch 0 --budget-minutes inf --per-config-minutes inf --seed 42 \
  > logs/ministral_fulldataset_batch0.log 2>&1 < /dev/null & disown
```

(`VLLM_USE_FLASHINFER_SAMPLER=0` is required on Laptop specifically — it
lacks the `nvcc`/build toolchain FlashInfer's sampler needs to JIT-compile;
confirmed earlier this session, not needed on gpu2.)

## Progress at time of pause (batch-0, 7 configs × 2 platforms = 14 total passes)

| Config | AtCoder (/121) | LeetCode (/121) |
|---|---|---|
| `pless_default` | 121/121 ✅ complete | not started |
| `pless_effk` | 121/121 ✅ complete | not started |
| `pless_effk_nofloor` | **28/121** (in progress) | not started |
| `pless_topmass` | not started | not started |
| `top_p_0_95` | not started | not started |
| `top_nsigma_1_5` | not started | not started |
| `pless_gini` | not started | not started |

Config order matches `--configs` flag order above; loop structure is
`for platform in [atcoder, leetcode]: for config in configs:` — so AtCoder
runs to completion across all 7 configs before LeetCode starts at all
(confirmed from `scripts/run_pless_variants_ministral_3b_lcb_cuda.py`'s
`main()`).

Resuming will re-scan each config's existing output file, skip the 121+121
already-done tasks for `pless_default`/`pless_effk`, and continue
`pless_effk_nofloor`/atcoder from task 29 onward — no re-generation, no
data loss beyond whatever single task was mid-generation at kill time.

## Why this pause is safe (verified earlier this session, not re-asserted here)

- Each task's 10 samples are only written to disk *after* the full
  generation call returns — a kill mid-task cannot corrupt the output file,
  only loses that one in-flight task's (unwritten) partial work.
- `--seed 42` is applied per-request (not a global RNG seeded once at
  process start) — a task generated now vs. after a kill-restart produces
  an identical result either way; restart position does not affect the
  project's apples-to-apples comparability.
- Resume scans `done_ids` freshly from each config's own output file at the
  top of that config's loop — no risk of resuming into the wrong
  config/platform even if other files have partial data from this same run.
