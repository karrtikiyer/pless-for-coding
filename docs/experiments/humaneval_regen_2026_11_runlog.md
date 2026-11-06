# HumanEval regeneration (2026-11): run log

Living log for D13 (`docs/theory/todos.md`). The spec is `docs/experiments/humaneval_regen_2026_11_plan.md`.
This file records **what has been done, what is running, and what comes next**, so the work can resume
after a context loss. Every entry carries an IST timestamp. Append to the log; don't rewrite earlier entries.

## How to resume (read this first)

1. Read **Status board** and **Next steps** below.
2. Check the MSI:
   ```bash
   ssh karrtik@100.94.82.96 'cd ~/pless-for-coding && git log --oneline -1 && ls logs/regen_small.* 2>/dev/null; \
     tail -c 400 logs/regen_small_driver.log; wc -l results/humaneval_regen_2026_11/*/*.jsonl; nvidia-smi'
   ```
   - `logs/regen_small.done` means the run finished; `logs/regen_small.failed` means it stopped on an error
     (read `logs/regen_small_driver.log`).
   - Neither file and the GPU is busy means the run is still going.
3. A stopped run resumes per task. Rerun the same mode: `small` already passes `--resume`. Resume refuses if the
   model, config, seed, n, max tokens, T, top-p, prompt style or the vLLM/transformers/torch versions differ.

## Fixed facts

| Item | Value |
|---|---|
| Branch | `feat/humaneval-regen` (local and MSI) |
| GPU box | MSI laptop, RTX 4090 Laptop 16 GB, cc 8.9: `ssh karrtik@100.94.82.96`, repo `~/pless-for-coding` |
| Generation env (MSI) | `.venv-vllm033` (from `envs/vllm-033/uv.lock`: vllm 0.33.0, torch 2.14.0, transformers 5.20.2) |
| Orchestration | `./run_regen_humaneval_8models.sh {smoke,smoke-rerun,fit MODEL,small,large [IDS]}` |
| Full-run output (MSI) | `results/humaneval_regen_2026_11/<model>/<config>.jsonl` + `MANIFEST.json`, `README.md` |
| Logs (MSI) | `logs/regen_humaneval_*.log` (per model × config), `logs/regen_small_driver.log` |
| Grading | **local only**: `uv run python -m bench.eval --dataset humaneval --humaneval-grader official --results-file …` |
| Official cross-check | `uv run --with human-eval==1.0.3 python scripts/humaneval_regen_crosscheck_official.py --files …` |
| Smoke checker | `uv run python scripts/humaneval_regen_smoke_checks.py --smoke-root … --rerun-root …` |

**Rules.**
- Code moves by `git push` / `git pull` only. Results move by `scp` and are never committed.
- Never read `.env` or token files. The user logs in to Hugging Face on the MSI.
- Status updates carry IST timestamps.
- No number goes into the paper before the results-validator has checked it. Follow the pre-registered rule in
  plan §9: no reruns to move a result.

## Status board (update in place)

| Phase (plan §8) | Status | Evidence / note |
|---|---|---|
| 0 Machine setup | DONE | `.venv-vllm033` built on the MSI; branch pulled |
| 1 Spec committed | DONE | `7a68bd9` |
| 2 Code + tests + review | DONE | `4e9ed17`, `5075ad8`, `ecf91fc`, `da44a39`; two code-review rounds addressed |
| 3 Smoke test | DONE | smoke3 + rerun: `HARD CHECKS: PASS`; official cross-check 330/330 and 600/600 agree |
| — Regression check of `generator_vllm.py` | DONE | `395ea36`: old (`eda858c`) vs new, both venvs, `compared 12 calls x 4 samples: IDENTICAL` |
| 4 Budget gate | DONE | user: "no need of any time budget"; estimate from smoke2 about 1h35 for the 3 small models |
| 5 Full runs, small models | **RUNNING** | started 2026-11-06 20:00:38 IST, pid 34159 |
| 6 7B fit test → full runs | TODO | needs the user's HF login on the MSI for both Llama-2 models; Qwen2.5-Coder-7B-Instruct last (15.2 GB bf16, likely does not fit 16 GB) |
| 7 Evaluate locally + results-validator | TODO | |
| 8 Write-up (Table II, paper text) | TODO | user edits `main.tex` on Overleaf; give exact text/lines |

## Next steps (in order)

1. Wait for `logs/regen_small.done` / `.failed`.
2. Inspect the finished run on the MSI:
   - line count 164 per file;
   - no errors in the logs;
   - copy `MANIFEST.json`.
3. `scp -r` `results/humaneval_regen_2026_11/` from the MSI to the same path locally.
4. Run the smoke checker on the full folder: `--smoke-root results/humaneval_regen_2026_11`, no `--rerun-root`.
   This covers sample count, indentation, BOS, stops and the samples-not-identical check.
5. Grade every file locally with `--humaneval-grader official`.
6. Run the official cross-check on a sample of each model's files.
7. Run the results-validator on the graded metrics.
8. Run the 7B fit tests (`fit MODEL` for each LARGE model), then `large`. Ask the user to log in to HF on the MSI first.
9. Rebuild Table II from `results/humaneval_regen_2026_11/` only, with paired bootstrap CIs (plan §8.8).
   Then draft the paper text changes and close D13.

## Open issues / watch items

- OpenCodeInterpreter-DS-1.3B:
  - vLLM's detokenizer mangles whitespace, so a safe tokenizer is installed and vLLM stop strings are turned off.
    The text is cut afterwards with `truncate_at_stop`.
  - As a result, most samples run to 512 tokens (2.8 s/task greedy). This is expected, not a bug.
  - It also means the stop-in-raw check does not apply to this model.
- `VLLM_BATCH_INVARIANT` is unavailable (cc 8.9). Seeded reproducibility was checked empirically in the smoke
  test (rerun identical) and is not guaranteed by vLLM.
- Two pre-existing local test failures, unrelated to this work: `test_metrics.py::test_pass_at_k_skips_too_large_k`,
  `test_think_end_id.py::test_multitoken_think_end_raises`.
- Out of scope, tracked separately: E28 (MBPP double BOS), E29 (MBPP stop list), D14, D15.

## Log (append-only, newest last)

- **2026-11-06 (day)**: Plan written and approved (`7a68bd9`). The generation env `envs/vllm-033` was locked
  (`9feccbc`) and built on the MSI.
- **2026-11-06**: Runner, official-composition grader, orchestration and smoke checks written test-first
  (`4e9ed17`). Two code reviews followed:
  - fixed the `[PYTHON]`-tag grading bug, multi-fence replies, indented imports/defs, the legacy-grader guard,
    `max_model_len` over all 164 prompts, the resume parameter check and the sandbox (`ecf91fc`);
  - kept regen free of the `bench.eval` import (zss is absent in vllm-033) (`5075ad8`).
- **2026-11-06**: Smoke 1 and smoke 3 (+ reruns) passed every hard check:
  - 100% of base bodies start with 4 spaces;
  - every prompt has exactly one BOS;
  - every config is identical between runs.

  Smoke 2's text-audit failures were false positives, because vLLM drops the stop string from `.text`; fixed in
  `da44a39`. The official human-eval 1.0.3 cross-check agreed on 330/330 and 600/600 programs.
- **2026-11-06 (before 20:00 IST)**: `generator_vllm.py` regression check (`395ea36`, `scripts/check_generator_vllm_regression.py`):
  - old vs new code was IDENTICAL in `.venv` (vLLM 0.21.0) and `.venv-vllm033` (0.33.0);
  - 48/48 samples were non-empty, with 38 and 34 distinct.

  The temporary worktree `/tmp/pless_old` was removed.
- **2026-11-06 20:00:38 IST**: Full small-model run started on the MSI:
  `setsid nohup bash -c "./run_regen_humaneval_8models.sh small && touch logs/regen_small.done || touch logs/regen_small.failed"`.
  At 20:03 IST it was on OCI greedy at 2.8 s/task, GPU 100%, 14.5 GB.
- **2026-11-06 20:11 IST**: OCI greedy finished, 164/164 lines. OCI temp_t0.7 was at 27/164, about 3.7 s/task.
- **2026-11-06 20:28 IST**: OCI greedy and temp_t0.7 at 164/164; top_p0.9_t1.0 at 124/164. No marker file yet.
  Zero `traceback`/`error` lines in the driver log.
