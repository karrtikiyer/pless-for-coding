# CF-748 Qwen3-8B G_k=0.25 — multi-machine generation and merge plan

Run: Qwen/Qwen3-8B, APPS CODEFORCES-interview 748-task subset (`data/cf_interview_748_ids.txt`), `pless_renyi` k=0.25,
T=1.0, n=10, thinking ON, cap 32768. Tracked in `docs/theory/todos.md` row **A69**. Written 2026-11-11.

The arm was generated on several machines. **Three files make up the complete arm. They must be merged before eval.**

## Files to merge (expected total 748)

| # | Machine | Path (relative to the repo on that machine) | Expected lines | Last task_id | Who wrote it |
|---|---|---|---|---|---|
| 1 | gpu2 (`~/pless-for-coding`) | `results/_cf_interview_748/Qwen--Qwen3-8B/CODEFORCES_interview/pless_renyi_think_t1.0_k0.25_t1.0.jsonl` | **561** | 2276 | lines 1-381: original pod; 382-453: gpu2 single-GPU run; 454-561: gpu2 GPU0 after the split |
| 2 | gpu2 | `results/_cf_interview_748_k025_gpu1split/Qwen--Qwen3-8B/CODEFORCES_interview/pless_renyi_think_t1.0_k0.25_t1.0.jsonl` | **120** | 2353 | gpu2 GPU1 |
| 3 | RunPod A100-SXM4-80GB | `results/_cf_interview_748_k025_a100split/Qwen--Qwen3-8B/CODEFORCES_interview/pless_renyi_think_t1.0_k0.25_t1.0.jsonl` | **67** | 2711 | A100 (tail hand-off) |

561 + 120 + 67 = 748. Line order does not matter for eval, which keys on `task_id`.

**Do NOT merge:**
- the local Mac copy in `results/_cf_interview_748/.../CODEFORCES_interview/` (451 lines; an exact prefix of file 1, snapshot 2026-11-10 12:35 IST);
- `.../CODEFORCES_interview/pre_gpu2_copy_20261110/` (the 381-line pod originals, also a prefix of file 1);
- temp copies in `/tmp/cf748_snap_20261110/` and `/tmp/smoke_vllm032/` on gpu2, and `/tmp/cf748_stage_20261110/` on the Mac;
- any G_k=0.35 file. That is a separate arm, and its canonical copy is on RunPod, not gpu2.

## How the work was split (reference ID lists, on gpu2)

1. **2026-11-10 12:57 IST, first split.** The single gpu2 run was stopped right after task 453. The remaining 295 ids were sorted and alternated:
   - GPU0 got 148 and kept writing to file 1 (`~/cf748_k025_split/gpu0_ids.txt`, 601 ids = 453 done + 148).
   - GPU1 got 147 (`~/cf748_k025_split/gpu1_ids.txt`) and wrote file 2.
   - Script: `~/cf748_k025_split/make_split.py`; manifest: `~/cf748_k025_split/manifest.json`.
2. **2026-11-11 09:31 IST, A100 tail hand-off.** At that point 539 + 96 were done and 113 remained.
   - GPU0 kept its next 22 (`a100/keep_gpu0.txt`) and GPU1 its next 24 (`a100/keep_gpu1.txt`).
   - The A100 got the 67 tail ids (`a100/a100_ids.txt`, sha256 prefix `6ad7fa3c`).
   - gpu2 was not restarted. The runner processes ids one at a time in ascending order, and the kept ids sit below every A100 id on the same GPU.
   - Watchers stop each gpu2 process right after its target line: `~/cf748_k025_split/stop_at_line.sh`; status in `~/cf748_k025_split/a100/stop_gpu{0,1}_status.txt`. On a clean stop each status file reads `status=OK` with last ids 2276 and 2353.
   - Script: `~/cf748_k025_split/make_a100_split.py`; manifest: `~/cf748_k025_split/a100/a100_manifest.json`. Local copies of `a100_ids.txt` and the manifest are in `/tmp/cf748_split/a100/` (the Mac's `/tmp` is wiped on reboot; the gpu2 copies are canonical).
3. **Independent checks.** data-wrangler checked both partitions: disjoint, covering all 748, keep ids before A100 ids. All checks passed.

## Merge procedure (data-wrangler; read-only on the sources)

1. Confirm all three runs have finished:
   - both gpu2 status files say `status=OK` (if a status says `MISMATCH` or `ERROR`, stop and investigate);
   - the A100 log ended without errors.
2. Copy files 1-3 to a local staging dir with scp. Check that the sha256 matches the source.
3. Per file:
   - line count equals the expected count;
   - every line parses;
   - 10 `samples` per record;
   - `method == pless_renyi`, `renyi_k == 0.25`, T 1.0;
   - no duplicate ids;
   - file 1 ends at 2276, file 2 at 2353;
   - file 3's ids equal `a100_ids.txt`.
4. Across files:
   - pairwise-disjoint task ids;
   - the union equals the 748 ids in `data/cf_interview_748_ids.txt`;
   - no `a100_ids.txt` id in files 1 or 2.
5. The first 451 lines of file 1 are byte-identical to the current local file. Back up the local file to `.../CODEFORCES_interview/pre_merge_<date>/`, then install the merged file (concatenate 1 + 2 + 3) in its place.
6. Re-verify the installed file: 748 unique tasks × 10 samples, plus its sha256.

## After the merge

1. Eval **locally on the Mac**, never on gpu2 (CLAUDE.md gotcha #9), with the A69 flags:
   `uv run python -m bench.eval --results-file <merged.jsonl> --dataset apps --workers 8 --skip-diversity --k 1,3,5,8,10 --t 0.1,0.3,0.5,0.7`
2. `SET=qwen_cf PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py` (writes the cb_div cache), then
   `uv run python scripts/paper/make_cf748_table.py --model qwen3`.
3. results-validator checks the numbers before they are reported.
4. Update A69 in `todos.md`.

## Engine / hardware caveat (must travel with any number from this arm)

This arm mixes engines and hardware:
- tasks from the original pod (engine versions not recorded);
- gpu2 2× RTX 4090: vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, flashinfer 0.7.0.post1, `GPU_MEM_UTIL=0.96`; freeze at `gpu2:~/vllm032_env/freeze_vllm032.txt`;
- RunPod A100-SXM4-80GB: vLLM version to be recorded.

All machines used the same sampler, flags (`VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=1`), prompts and cap. Runs are unseeded. Label the arm as mixed-engine.
