# gpu2 batch1/batch2 pause checkpoint — 2026-10-14

Recorded immediately before intentionally stopping both gpu2 full-dataset
Ministral LCB jobs (batch1, batch2) to apply a concurrency-safety fix to
`bench/checkpointing.py:append_result()` (no file locking today; both batches
append to the same shared output files with multi-MB JSON lines — see
conversation for the full analysis. No corruption observed in 454 lines
checked, but the gap is real and most of the remaining work for both batches
is still ahead, not behind, so fixing now rather than after more exposure).

## State at time of pause

- **Host**: gpu2 (`ssh gpu2`, i.e. `karrtik@10.132.3.11`)
- **Working directory**: `/home/karrtik/pless-for-coding`
- **Timestamp**: 2026-10-14 12:00:18 UTC (17:30 IST)

| Batch | GPU | PID (python) | Parent bash PID | Started |
|---|---|---|---|---|
| batch1 | CUDA_VISIBLE_DEVICES=0 | 3116250 | 3116249 | 2026-10-12 20:20:28 |
| batch2 | CUDA_VISIBLE_DEVICES=1 | 881363 | 881361 | 2026-10-13 05:43:27 |

Both verified via `/proc/<pid>/cmdline` and `/proc/<pid>/environ` directly
(not reconstructed from `ps aux` text, which can wrap/truncate long command
lines).

## Exact restart commands

**Do not add `VLLM_USE_FLASHINFER_SAMPLER`** — confirmed via `/proc/<pid>/environ`
that neither running process has this set. It is a Laptop-specific workaround
(Laptop lacks the `nvcc` toolchain FlashInfer's sampler needs to JIT-compile);
gpu2 does not need it. Adding it here would be an unrequested parameter change.

### batch1 (GPU0)

```bash
cd /home/karrtik/pless-for-coding
nohup env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/karrtik/pless-for-coding \
  PATH=/home/karrtik/pless-for-coding/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
  .venv/bin/python scripts/run_pless_variants_ministral_3b_lcb_cuda.py \
  --model mistralai/Ministral-3-3B-Reasoning-2512 --dtype bfloat16 \
  --gpu-memory-utilization 0.92 --max-model-len 37500 \
  --n-samples 10 --max-new-tokens 35000 \
  --configs pless_default pless_effk pless_effk_nofloor pless_topmass top_p_0_95 top_nsigma_1_5 pless_gini \
  --full-dataset --batch 1 --budget-minutes inf --per-config-minutes inf --seed 42 \
  > logs/ministral_fulldataset_batch1.log 2>&1 < /dev/null & disown
```

### batch2 (GPU1)

```bash
cd /home/karrtik/pless-for-coding
nohup env CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/home/karrtik/pless-for-coding \
  PATH=/home/karrtik/pless-for-coding/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
  .venv/bin/python scripts/run_pless_variants_ministral_3b_lcb_cuda.py \
  --model mistralai/Ministral-3-3B-Reasoning-2512 --dtype bfloat16 \
  --gpu-memory-utilization 0.92 --max-model-len 37500 \
  --n-samples 10 --max-new-tokens 35000 \
  --configs pless_default pless_effk pless_effk_nofloor pless_topmass top_p_0_95 top_nsigma_1_5 pless_gini \
  --full-dataset --batch 2 --budget-minutes inf --per-config-minutes inf --seed 42 \
  > logs/ministral_fulldataset_batch2.log 2>&1 < /dev/null & disown
```

Only difference between the two: `CUDA_VISIBLE_DEVICES` (0 vs 1), `--batch`
(1 vs 2), and the log filename.

## Progress at time of pause (shared output files, both batches combined)

| Config (AtCoder) | Lines (task_ids) done |
|---|---|
| `pless_default` | 242 |
| `pless_effk` | 201 |
| `pless_effk_nofloor` | 11 |
| `pless_topmass` | not started |
| `top_p_0_95` | not started |
| `top_nsigma_1_5` | not started |
| `pless_gini` | not started |

LeetCode: not started for any config, either batch.

batch1's full task set: 121 AtCoder + 90 LeetCode = 211 tasks (all 7 configs).
batch2's full task set: 121 AtCoder + 88 LeetCode = 209 tasks (all 7 configs).
Verified via `results/lcb_v6_full1046/batch_assignment.csv` (1046 total tasks,
5 batches, 0-indexed; batches 0/3/4 are not running on gpu2 right now — batch0
is the one separately paused on Laptop, see
`docs/theory/laptop_batch0_pause_checkpoint.md`).

## Why this pause is safe

- Same write-after-generate + per-config fresh `load_completed_ids` resume
  logic as the Laptop job (see `laptop_batch0_pause_checkpoint.md` for the
  general argument) — a kill mid-task only loses that one in-flight task.
- batch1 and batch2 write disjoint task_id sets (per `batch_assignment.csv`)
  into the same shared output files — resuming reads the combined file and
  correctly skips whichever task_ids are already present, regardless of
  which batch wrote them.
- Checked directly: all 454 existing lines across the 3 configs with any
  progress are valid, non-corrupted JSON with zero duplicate task_ids.
