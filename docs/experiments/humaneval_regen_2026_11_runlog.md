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
- **Grading-load protocol (user request 2026-11-07, pre-declared before any stripped-run grading).** Timeouts are
  wall-clock, so they are sensitive to CPU load (`bench/eval/_cpu_lock.py` docstring: a 3x slowdown under
  12-core stress).
  1. Grade only via the `bench.eval` CLI, which holds `/tmp/pless_cpu_heavy.lock` and so serializes with other
     evals. Use the default 4 workers on this 16-core, 48 GB machine.
  2. Before each grading batch, log the load average, free memory and other heavy processes (`uptime`,
     `memory_pressure`, `pgrep -fl bench.eval`).
  3. **The reported numbers come from ONE final grading pass of all 8 models**, run when no other eval holds the
     lock and the 1-minute load average is below about 2. Earlier passes are provisional.
  4. Report timeout% per file. Any file above 1% gets its timed-out programs inspected before reporting.
  - Baseline at 2026-11-07 16:10 IST: load 5.6–5.8; memory 77% free, no swap; another session's Ministral
    LiveCodeBench `bench.eval` held the lock.
- **No edits to the paper until the user's final review (user, 2026-11-07).** That covers
  `paper/llm4code/tables/`, `paper/llm4code/figures/` and `../pless-for-coding-paper/` (main.tex, tables, refs).
  New tables and draft text go to `results/humaneval_regen_2026_11/analysis/` or `docs/experiments/`. Note that
  `make_llm4code_tables.py` writes into both paper locations, so don't run it.
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
| 5 Full runs, small models | DONE, then SUPERSEDED by A1 | Unstripped 2026-11-06 (kept, out of tables). Stripped: MSI 2026-11-07 16:57 → 19:07 IST |
| 5b Stripped full runs, 7B (GPU2) | DONE | 2026-11-07 18:19 → 20:47 IST; 31 files |
| 7 Final grading + validation | DONE | Final pass 21:20–21:45 IST; results-validator: safe for Table II |
| 8 Table II rebuild (staging) → user review | NEXT | No paper edits until the user's review |
| 6 7B fit test → full runs | TODO | needs the user's HF login on the MSI for both Llama-2 models; Qwen2.5-Coder-7B-Instruct last (15.2 GB bf16, likely does not fit 16 GB) |
| 7 Evaluate locally + results-validator | TODO | |
| 8 Write-up (Table II, paper text) | TODO | user edits `main.tex` on Overleaf; give exact text/lines |

## Next steps (in order, with the agent that does each)

Per the user (2026-11-06): each step goes to its specialist subagent (CLAUDE.md routing). The main session
coordinates, monitors the MSI, briefs the agents, checks their hand-backs and updates this log.

| # | Step | Agent |
|---|---|---|
| 1 | Wait for `logs/regen_small.done` / `.failed` (background marker check) | main session |
| 2 | `scp -r` `results/humaneval_regen_2026_11/` from the MSI to the same path locally; check the integrity of every file: line count 164, schema, one record per task, n, `MANIFEST.json` present. Then run the smoke checker on the full folder (`--smoke-root results/humaneval_regen_2026_11`, no `--rerun-root`). Report row counts before and after. | **data-wrangler** |
| 3 | Grade every file locally with `--humaneval-grader official` (3.0 s timeout); run the official human-eval 1.0.3 cross-check on a sample per model; aggregate the per-config metrics in the canonical column set (`headline_table`) | **experiment-harness** |
| 4 | Validate the graded numbers: recompute from the raw per-sample results, sanity-check against the plan's expectations, and check that the claims match the data | **results-validator** |
| 5 | 7B fit tests (`fit MODEL` for each LARGE model), then `large`. Ask the user to log in to HF on the MSI first. Confirm the run before launch. | main session (launch/monitor) |
| 6 | Rebuild Table II from `results/humaneval_regen_2026_11/` only, with paired bootstrap CIs (plan §8.8) | **research-coder**, then **code-reviewer** |
| 7 | Re-validate the Table II numbers | **results-validator** |
| 8 | Draft the paper text changes (retract −7.87 pp, disclosures) for the user to paste on Overleaf; review them against the data | main session drafts; **paper-validator** reviews |
| 9 | Final `refs.bib` check before upload (D15 item 7) | **citation-checker** |

## Open issues / watch items

- OpenCodeInterpreter-DS-1.3B:
  - vLLM's detokenizer mangles whitespace, so a safe tokenizer is installed and vLLM stop strings are turned off.
    The text is cut afterwards with `truncate_at_stop`.
  - As a result, most samples run to 512 tokens (2.8 s/task greedy). This is expected, not a bug.
  - It also means the stop-in-raw check does not apply to this model.
  - **Process note (2026-11-06).** This was knowable at plan time: the safe-tokenizer workaround has existed since
    `1ef6128` (2026-05-22). It was missing from the original plan (`7a68bd9`). Code review caught it, and it was
    added to the plan in `4e9ed17`, but the user was first told only after launch. Consequences:
    - For OCI, `finish_reason` = "length" does not mean the completion was truncated. Compute at-cap on the cut text.
    - `completion_num_tokens` is counted on the cut text (`regen.py:221`), so mean_tok is fine.
    - The post-hoc cut applies the same stop rule, but is not bit-identical to a run with stops. Without batch
      invariance, sequences ending earlier change the batch and can shift later tokens.
    - An equivalence check (Qwen2.5-Coder-1.5B, vLLM stops on vs off + post-cut) was proposed and **skipped by
      user decision (2026-11-06)**. It changes no reported number: OCI is cut with the same stop list and graded
      by the same grader. It also could not test OCI itself.
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
- **2026-11-06 21:03 IST**: OCI greedy, temp_t0.7, top_p0.9_t1.0, pless_t0.7 and pless_norm_t0.7 at 164/164.
  top_p0.95_t0.2 at 108/164; beam4 still to come. No marker file; 0 tracebacks.
- **2026-11-06 22:13:35 IST**: **small-model run DONE** (`logs/regen_small.done`): 21 files (3 models × 7 configs),
  each 164 lines, 3444 total. Next: step 2 (data-wrangler: scp + integrity).
- **2026-11-06 22:18 IST**: Step 2 **DONE** (data-wrangler):
  - 23 files scp'd to the local `results/humaneval_regen_2026_11/`; all sha256 match the remote.
  - Rows: remote 3444, local 3444. Every file has 164 unique task_ids and the correct n. Settings are constant
    within each file.
  - All 21 files: vllm 0.33.0 / transformers 5.20.2 / torch 2.14.0, git `395ea36`, git_dirty False.
    `backend_stop_strings` is False only for the OCI vLLM configs.
  - Full-folder checker: `HARD CHECKS: PASS`.
- **FINDING: p-less and p-less-norm produce the same samples** (main session re-checked):
  - Raw samples identical: Qwen2.5-Coder-1.5B 1640/1640, Qwen2.5-Coder-3B 1640/1640, OCI 1639/1640 (the
    hand-back said 1640/1640 for OCI; the re-check found 1639).
  - Cause (`generator_vllm.py:140`): the p-less-norm cutoff is Σp² − (1−Σp²)/(|V|−1). The gap is at most
    6.6e-6 for Qwen (|V| = 151,665) and 3.1e-5 for OCI (|V| = 32,022).
  - Both use the same per-request seed, so the survivor set and the draws coincide almost always.
  - OCI's single differing sample (larger gap there) is evidence that the norm cutoff is actually applied,
    i.e. dispatch is not broken. [Likely]
  - Paper consequence: the two rows in the HumanEval table will be (near-)identical. The paper text treats
    p-less-norm as a separate variant (main.tex l.39, 62, 82, 102, 121, 130, 149, 157, 187, 205). Decision pending with
    the user. Per plan §9: report as produced, no reseeding.
- **2026-11-06**: Step 3 (grading) handed to experiment-harness: official grader, `--skip-structural-diversity`
  (Table II reports cb_div, not struct_div), plus the official cross-check per model.
- **2026-11-06 ~22:50 IST**: Step 3 **DONE** (experiment-harness):
  - Metrics: `results/humaneval_regen_2026_11/<model>/metrics/<config>_metrics.json`. All 21 have num_tasks
    164, grader `humaneval-official-v1`, timeout 3.0 s. Max timeout share 0.6%.
  - Official cross-check, per model: 400/400 agree (×3).
  - Main session spot-checked pass@1 from 9 metrics files: matches.
  - pass@1 (%):

    | model | greedy | beam4 | T0.7 | top_p0.9 | top_p0.95 T0.2 | pless | pless_norm |
    |---|---|---|---|---|---|---|---|
    | OCI | 57.93 | 68.29 | 51.59 | 52.87 | 58.90 | 58.48 | 58.48 |
    | Q1.5B | 39.02 | 55.49 | 36.16 | 35.67 | 39.27 | 40.12 | 40.12 |
    | Q3B | 48.17 | 57.32 | 39.02 | 36.40 | 47.07 | 47.26 | 47.26 |

    These are **NOT VALIDATED yet**; do not report.
  - Issues raised:
    1. **cov@td = 0 everywhere: code defect.** `metrics.py::add_distinct_counts` fingerprints the body-only
       completion, which has no `def`, so `ast_fingerprint` returns None. cov@td is not in Table II, but it is
       in the canonical columns. Needs a fix (research-coder + code-reviewer); check whether MBPP base rows are
       affected.
    2. cb_div differs slightly between pless and pless_norm despite identical samples (non-determinism?).
    3. beam4 beats greedy by +9 to +16.5 pp.
    4. For n=1 configs, pass@3..10, cov and cb_div should read "n/a".
  - Sent to the results-validator (questions 1–5) and to paper-reader ×2 (Hasan et al. Table 4; Qwen2.5-Coder
    report base HumanEval) for external sanity numbers. The cov@td fix is held until the validator reports
    (does it touch MBPP?).
- **2026-11-06**: Qwen2.5-Coder report (arXiv:2409.12186v3, Table 5; main session re-checked the PDF text):
  - HE 43.9 (1.5B) and 52.4 (3B) for the base models. Note: `notes/qwen25coder_report_humaneval.md`.
  - The decoding setup is not stated in the paper; "greedy" never appears.
  - Ours: greedy 39.02 / 48.17 (≈ 4–5 pp below); beam4 55.49 / 57.32 (above).
  - Different, unstated harness and prompt, so this is a sanity range only, not a like-for-like comparison.
- **2026-11-06**: Hasan et al. (arXiv:2507.03160v4, Table 4, HumanEval pass@1; main session re-checked the PDF,
  l.403/405/411). Note: `notes/hasan2025slm_humaneval.md`. Same decoding as our `top_p0.95_t0.2`.

  | model | Hasan | ours top_p0.95_t0.2 |
  |---|---|---|
  | OCI | 0.59 (967/1640) | 58.90 (966/1640) |
  | Q1.5B | 0.44 | 39.27 |
  | Q3B | 0.51 | 47.07 |

  - **OCI matches to one sample. Both Qwen models are about 4–5 pp low**, against both references (Hasan and the
    Qwen report).
  - Known protocol difference: Hasan used the harness task `humaneval`, which passes `prompt.strip()`. We use
    `humaneval-unstripped` (prompt ends `"""\n`; plan decision). The Qwen tokenizer ends our prompt with the
    merged token `' """\n'`.
  - **Unverified hypothesis:** the trailing newline creates an unnatural token boundary for Qwen. That would
    hurt greedy and sampling, and beam may recover from it. This could also explain part of the beam > greedy gap.
  - To confirm: a diagnostic greedy run with stripped prompts (all 3 models, 164 tasks, a few GPU minutes) into a
    separate diag folder. **Pending user decision.** Switching the reported protocol would be a plan change,
    applied to all models, not a per-model choice.
- **2026-11-06 ~23:30 IST**: Step 4 (results-validator) on the small models:
  - **Q1 CONFIRMED.** All 21 pass@1/pass@10 recomputed. Independent re-execution: 50/50 ×3 random samples, and
    0 mismatches on all 984 greedy+beam programs.
  - **Q2 beam > greedy.**
    - Same prompt tokens on both backends.
    - The main cause is **greedy `pass` stubs** (`'    # Your code here\n    pass\n'`): Q1.5B 38/164 (main
      session count 37), Q3B 6, OCI 4; beam 3/0/0. At T=0.2, 360/1640 Q1.5B samples are stubs, so this is the
      model's mode under this prompt.
    - HF greedy on the 41 disputed Q1.5B tasks (Mac MPS) recovers about +5 tasks, so the backend explains about
      3 of the 16.5 pp [partial].
    - Paired CIs, beam − greedy: +16.46 [9.15, 23.78], +9.15 [2.44, 15.85], +10.37 [5.49, 15.24]. On MBPP (both
      HF) the gap is +0.0 / +0.8 / +2.2.
    - Validator: run HF greedy on the MSI before interpreting.
  - **Q3 CONFIRMED.** T0.2 ≈ greedy (CIs span 0).
  - **Q4 DISCREPANCY (metric code).** cb_div depends on PYTHONHASHSEED (CodeBLEU dataflow match): ±0.0005 in
    aggregate, per task up to 0.13 vs 0.49. This probably also affects MBPP. Fix: fixed PYTHONHASHSEED, reported.
  - **Q5.** cb_div is computed on prompt-excluded samples in both pipelines. But MBPP samples include the `def`
    signature (MBPP prompts have none) and HumanEval samples do not. Plan §5 "comparable" → "prompt-excluded".
  - Minor: for beam, `raw_samples` is already cut (`generator.py:363`). No number changes.
- **Combined proposal (pending user):** one greedy diagnostic on the MSI, 3 models × 164 tasks, into
  `results/humaneval_regen_2026_11_diag/`:
  - HF greedy on the unstripped prompt (backend effect);
  - vLLM greedy on the stripped prompt (prompt effect).

  It needs runner options for both (research-coder + code-reviewer). Code fixes queued for research-coder +
  code-reviewer: cov@td fingerprint, fixed PYTHONHASHSEED for cb_div, n=1 columns as n/a.
- **2026-11-07**: User approved the diagnostic and the three code fixes. Two research-coder agents in parallel:
  - (a) runner: `greedy_hf` config, `--prompt-variant {unstripped,stripped}`, a `diag` mode writing
    `results/humaneval_regen_2026_11_diag/`;
  - (b) eval: cov@td fingerprints the official program minus the test tail (legacy/MBPP unchanged);
    deterministic cb_div (fixed PYTHONHASHSEED, recorded); opt-in "—" for n=1 columns in `headline_table`.

  Next: code-reviewer on both → commit + push the branch → MSI `git pull` → run `diag` → data-wrangler (scp)
  → experiment-harness (grade) → results-validator.
- **2026-11-07 05:43:32 IST**: Runner change reviewed (code-reviewer: no blockers; 102 tests pass; main session
  re-ran 64 regen tests). Committed `ef6bef6`, pushed; MSI pulled; **`diag` mode started** (marker
  `logs/regen_diag.done/.failed`).
  - Reviewer notes for interpreting the run:
    - check by hand that `versions` in the diag files match the full-run `greedy.jsonl`;
    - for HF vs vLLM, read `sample_meta.text_matches_reference` and `num_output_tokens` (the HF and vLLM
      decode paths differ; OCI's vLLM runs have no backend stops);
    - stripped records keep config "greedy", so never merge the diag root with the main root.
- **2026-11-07**: Eval fixes done (research-coder; not committed):
  - `metrics.py`, `humaneval_official.py`, `__main__.py`, `headline_table.py`, new `_codebleu_worker.py`,
    `tests/test_eval_regen_fixes.py` (13 tests). 923 passed, 3 skipped.
  - Byte-identical checks: `build_program` on 28,764 regen samples, MBPP `add_distinct_counts` on 81,341 task
    results, and default `headline_table` on 186 metrics files.
  - Qwen-3B pless re-eval: cov@0.3d 0.0 → 9.76, cb_div 0.1534 → 0.1527 (deterministic across parent seeds).
  - **New finding: MBPP CodeLlama-7b-Instruct cov@td is broken the same way.** 25,256 of 26,737 correct samples
    don't fingerprint: they start with `'  def …'` (2-space indent). Other MBPP base models are minor (Qwen-7B
    458/22,978, OCI 710/27,549).
  - Also: `ast.parse` `MemoryError` on 2 Llama-2-7b samples. Not caught by `ast_fingerprint`; both failed, so no
    crash.
  - Code-reviewer running on the eval fixes.
- **2026-11-07**: Eval fixes reviewed: no blockers; main session re-ran 50 tests. Committed `5fd6b5b`, pushed.
  - Reviewer notes:
    - never re-grade with `--use-codebleu-cache`;
    - cb_div on base bodies still uses the raw completion, while cov@td uses the full program. Decide before the
      final table: keep body-only and disclose (default), or switch;
    - queued: a cache-mix warning and a more robust n=1 detection.
- **2026-11-07 06:15:14 IST**: **diag run DONE** (`logs/regen_diag.done`): 6 files × 164 lines. Copy and
  integrity check handed to data-wrangler.
- **2026-11-07**: Diag copy **DONE** (data-wrangler):
  - 8 files, all sha256 match; 164 rows before and after.
  - Versions identical to the full run apart from git_commit (`ef6bef6`). Stripped `task_prompt` == full-run
    prompt `.strip()` 164/164.
  - **Stub counts (no `return`, has `pass`):**

    | model | vLLM greedy | HF greedy | vLLM stripped | beam4 |
    |---|---|---|---|---|
    | Q1.5B | 37 | 37 | 12 | 2 |
    | Q3B | 6 | 5 | 3 | 0 |
    | OCI | 4 | 1 | 2 | 0 |

    The stubs follow the prompt, not the backend.
  - HF greedy vs vLLM greedy identical cut text: Q1.5B 125/164, Q3B 132, OCI 129.
  - Smoke checker on the diag folder: HARD FAIL on the 4-space check (stripped: Q1.5B 98.2%, Q3B 94.5%).
    Main session inspected it: real model behaviour, not an artefact. With the prompt ending in `"""` the model
    sometimes continues on the same line (`' count = 0'`, HumanEval/98, 90, 79). That gives `""" count = 0`,
    which becomes a SyntaxError at grading. So the stripped prompt has its own boundary failure.
  - OCI `text_matches_reference` False 164/164, the same as the full-run OCI greedy (known safe-tokenizer case).
  - Grading of the 6 diag files handed to experiment-harness.
- **2026-11-07**: Diag grading **DONE** (experiment-harness).
  - Official cross-check 984/984 agree. Main session re-checked all 6 pass@1 values from the metrics JSONs.
  - pass@1 % (correct tasks):

    | row | OCI | Q1.5B | Q3B |
    |---|---|---|---|
    | vLLM greedy (full) | 57.93 (95) | 39.02 (64) | 48.17 (79) |
    | beam4 (full) | 68.29 (112) | 55.49 (91) | 57.32 (94) |
    | HF greedy | 57.93 (95) | 40.24 (66) | 46.95 (77) |
    | vLLM greedy stripped | 59.76 (98) | 47.56 (78) | 50.61 (83) |

  - Paired 95% CIs, HF − vLLM: OCI +0.00 [−3.66, 3.05], Q1.5B +1.22 [−2.44, 4.88], Q3B −1.22 [−3.05, 0.00].
  - Paired 95% CIs, stripped − vLLM: +1.83 [−2.44, 5.49], **+8.54 [0.61, 16.46]**, +2.44 [−3.05, 7.93].
  - Paired 95% CIs, beam4 − HF greedy: +10.37 [5.49, 15.24], +15.24 [7.93, 21.95], +10.37 [3.66, 17.07].
  - Stripped failures: continued-on-`"""`-line SyntaxErrors Q1.5B 1, Q3B 8. Every remaining stub failed.
  - NOT VALIDATED; results-validator running (recompute, check C1–C4, red-team a post-hoc protocol switch).
- **2026-11-07**: Diag **VALIDATED** (results-validator).
  - All 12 pass@1 and all differences reproduce. McNemar: HF−vLLM all p ≥ 0.5; stripped−vLLM Q1.5B 29/15,
    p=0.049 (about 0.15 Holm-corrected over 3 models), others p > 0.5; beam−HF greedy all p ≤ 0.006.
  - C1 (backend not the cause) confirmed: bounds of about ±3–5 pp, well below the beam gap.
  - C2 overstated: the Q1.5B gain is net (29 up, 15 down), and only 15 of the 29 gains were stub tasks.
  - C4: stubs explain a minority of beam's lead. Excluding greedy-stub tasks, beam still leads by
    +9.4 / +10.2 / +6.3 pp.
  - **Stub mode is uneven across decoders** (share of samples with `pass` and no `return`; main session
    reproduced it):

    | model | greedy | pless T0.7 | top_p0.95 T0.2 | temp T0.7 | top_p0.9 T1.0 | beam4 |
    |---|---|---|---|---|---|---|
    | Q1.5B | 22.6% | 20.4% | 20.4% | 10.4% | 8.0% | 1.2% |
    | Q3B | 3.7% | 2.6% | 3.2% | 3.1% | 3.3% | 0.0% |
    | OCI | 2.4% | 3.0% | 3.8% | 5.7% | 5.7% | 0.0% |

    So the prompt format is **not** decoder-neutral on Q1.5B. A switch would likely help p-less and the low-T
    configs more than T0.7/top-p T1.0 (that size is unverified; the greedy diagnostic cannot predict it).
  - Validator: switching outright after seeing the results is not supported. Defensible options:
    - (a) keep unstripped as primary and disclose the stub mode plus the diagnostic;
    - (b) run all configs on both prompts, pre-declare the primary now, and report whether the p-less
      comparisons change sign or significance.
  - **Decision pending with the user.**
- **2026-11-07 14:52:31 IST**: User asked for a Hasan et al. reproduction test: `top_p0.95_t0.2` with
  `--prompt-variant stripped` for Qwen2.5-Coder-1.5B and 3B into `results/humaneval_regen_2026_11_diag/`
  (output `top_p0.95_t0.2__prompt_stripped.jsonl`).
  - Run by calling the runner directly on the MSI (`ef6bef6`), with the same 4 env vars the orchestration script
    exports. No code change. Marker: `logs/regen_diag2.done/.failed`.
  - Question it answers: does the stripped prompt reproduce Hasan's 0.44 / 0.51? OCI already reproduces
    unstripped (966 vs 967 of 1640).
  - Separate from the protocol decision, which is still pending (proposed rule: stripped-prompt sensitivity
    analysis only for base models with a greedy stub rate > 10%).
- **2026-11-07 15:00:04 IST**: Hasan reproduction run **DONE** (`logs/regen_diag2.done`): 2 files × 164 lines.
  Copy and integrity check handed to data-wrangler, then grading.
- **2026-11-07**: Hasan reproduction **graded** (experiment-harness; official cross-check 400/400; main session
  recomputed pass@k from per_task).
  - Copy (data-wrangler): sha256 match; stripped empty samples Q1.5B 17, Q3B 16 (unstripped: 0); `"""`-line
    continuations 10 / 77 samples.
  - pass@1 / @2 / @5 / @10:

    | model | stripped (Hasan config) | unstripped (ours) | Hasan v4 Table 4 |
    |---|---|---|---|
    | Q1.5B | .4372 / .5064 / .5860 / .6341 | .3927 / .4560 / .5254 / .5610 | .44 / .51 / .58 / .61 |
    | Q3B | .5146 / .5755 / .6395 / .6768 | .4707 / .5249 / .5904 / .6402 | .51 / .57 / .63 / .66 |

  - **The stripped prompt reproduces Hasan's pass@1 exactly at 2 dp on both Qwen models.** pass@2–10 are within
    +0.00 to +0.02. Combined with OCI unstripped = Hasan (966 vs 967 of 1640), all 3 small models are now
    reproduced. The Qwen gap was the prompt format.
  - Paired pass@1, stripped − unstripped:
    - Q1.5B +4.45 pp [−0.30, +8.96] (45 up / 31 down);
    - **Q3B +4.39 pp [+0.30, +8.54]** (44 up / 25 down).
  - **This corrects an earlier claim:** Q3B is also prompt-sensitive under T=0.2 sampling, despite its low stub
    rate (3.2%). The proposed "greedy stub rate > 10%" trigger rule would have missed it, so drop it.
  - Results-validator asked to check.
- **2026-11-07**: User asked for OCI `top_p0.95_t0.2` with the stripped prompt (same diag folder, same direct
  invocation). Purpose: check whether stripped is harmful for OCI before possibly adopting stripped uniformly for
  base models.
  - Main-session caveats given to the user: a switch must be declared in the plan before the 7B runs, with the
    existing unstripped results kept as a reported sensitivity analysis. "Uniform" covers base models only, since
    chat prompts have no trailing newline.
  - Marker: `logs/regen_diag3.done/.failed`.
- **2026-11-07**: Hasan reproduction **validated** (results-validator). Verdict: OVERSTATED; the numbers are
  right.
  - Run-to-run SD of pass@1 is about 0.006, so a match at 2 dp is weak evidence by itself. The strong evidence is
    that unstripped sits about 7 SD below Hasan.
  - Say "consistent with Hasan; the unstripped run is not", not "reproduced exactly". "All three reproduced"
    mixes two prompt configs until OCI stripped is run (now running).
  - The 512-includes-prompt hypothesis was tested and does not explain the pass@5/10 residual (at most −0.006).
    Dropped.
  - Paired task bootstrap: Q1.5B +4.45 [−0.12, 9.09], p=0.135; Q3B +4.39 [0.24, 8.60], p=0.030 (Holm ≈ 0.06).
    Q3B significance is fragile.
  - Mechanism: not stubs. In Q3B's 44 gained tasks the unstripped failures are 66% wrong logic, 10% stubs. The
    prompt shifts the whole output distribution. Unstripped Q3B also has 77 at-cap samples (stripped 40) and
    14 `<|fim…` token leaks (stripped 0).
  - The prompt effect (about 4.4 pp) is larger than p-less vs top_p0.95_t0.2 (+0.85 / +0.19 / −0.42) and uneven
    across decoders.
  - **Validator recommends (b):** unstripped primary, plus all sampled configs on both prompts for base models,
    and a pre-declared robustness criterion. It advises against (c) "switch primary": that is forking paths, and
    stripped has its own failures (same-line `"""`, empty samples).
  - The validator's side note (pless vs pless_norm byte check) was already done: 1640/1640 identical on Qwen,
    1639/1640 on OCI.
- **2026-11-07**: OCI stripped `top_p0.95_t0.2` **DONE** on the MSI; copy + grade handed to data-wrangler.
- **2026-11-07**: The user noted that **OCI is instruction-tuned**. Verified:
  - HF card: "This model is based on deepseek-coder-1.3b-base"; its usage example uses `apply_chat_template`.
  - `tokenizer_config.json` has a chat_template with the system prompt "You are an AI programming assistant,
    utilizing the Deepseek Coder model…".
  - We (and Hasan, and our MBPP runs) prompt it raw. So: label it "instruction-tuned, prompted raw", not "base".
  - The prompt-protocol question applies to all raw-prompted models: OCI, Q1.5B, Q3B, CodeLlama-7b,
    Llama-2-7b.
  - Whether instruction tuning makes it insensitive to the prompt boundary is unverified; the OCI stripped run
    tests it.
  - TODO for the user's final review: list where the paper/tables call OCI a base model (MBPP and HumanEval). No
    edits.
  - Paper check (read-only): main.tex never calls OCI "base". l.99 says "eight base and instruct … models"; l.121
    says "base" only of the Qwen models. Suggested clarifying clause near l.99 for the user's review.
- **2026-11-07**: OCI stripped `top_p0.95_t0.2` graded (data-wrangler; sha256 match; cross-check 400/400;
  integrity OK; 0 empty, 0 same-line, 31 stubs).
  - pass@1/2/5/10:
    - stripped .6140 / .6749 / .7259 / .7500;
    - unstripped .5890 / .6454 / .6998 / .7195;
    - Hasan .59 / .65 / .70 / .72.
  - Paired pass@1, stripped − unstripped: +2.50 pp [−0.67, +5.85] (31 up / 24 down). Not detectably different;
    not harmful.
  - **Tension.** Main session verified in BigCode harness `tasks/humaneval.py` l.32/66–67 that the task
    `humaneval` = `doc["prompt"].strip()`. Hasan's OCI config uses `"tasks": "humaneval"`, so their OCI run
    was stripped.
  - Yet our OCI matches Hasan with **unstripped** (all four values at 2 dp), and our stripped OCI is about
    +2.5 pp (about 4 run-to-run SD) above it. The Qwen models match Hasan under stripped.
  - So no single protocol of ours reproduces all three Hasan numbers. The Hasan comparison cannot be used as
    external evidence for the stripped prompt. Report the numbers side by side with the protocol differences;
    do not claim "reproduced".
  - OCI-specific pipeline difference (unverified as a cause): vLLM without backend stops plus a safe-tokenizer
    re-decode, vs HF with stops in the harness.
- **2026-11-07**: Hasan's OCI protocol **verified at the source** by the main session (earlier it rested only on
  the paper-reader note).
  - Authors' repo `results/evaluations/OpenCodeInterpreter-DS-1.3B_humaneval.json`: `config.tasks='humaneval'`,
    `instruction_tokens=None`, `prefix=''`, T 0.2, top_p 0.95, top_k 0, n 10, `max_length_generation` 512,
    bf16; pass@1 0.58963, pass@10 0.71951.
  - The fork's `bigcode_eval/tasks/humaneval.py` differs from upstream only in the k list:
    `humaneval` → `doc["prompt"].strip()`.
  - So Hasan ran OCI **stripped, as a base model** (not `instruct-humaneval`, which is a different dataset,
    `codeparrot/instructhumaneval`, with an instruction+context prompt).
  - Our unstripped OCI pass@10 = 0.7195, identical to theirs; our stripped OCI = 0.7500. The tension stands,
    now fully verified.
  - The Qwen 1.5B/3B summaries key the score under task `humaneval`, and the sibling Qwen-0.5B config says
    `tasks: "humaneval"`. So Hasan used stripped for Qwen as well [Likely].
- **2026-11-07: DECISION (user): stripped prompt throughout, all configs regenerated from scratch.**
  - Recorded as plan **§11 Amendment A1**, covering raw-prompted models (OCI, Q1.5B, Q3B, CodeLlama-7b,
    Llama-2-7b). Chat models are unchanged, in the same new root.
  - New root: `results/humaneval_regen_2026_11_stripped/`. The unstripped run and the diag runs are kept
    unchanged, out of the tables.
  - Research-coder is adding the stripped variant to the orchestration modes (raw-prompted only), new roots, and
    a stripped-aware indentation check in the smoke checker. Then code-reviewer → commit/push → MSI pull →
    stripped smoke + rerun → GPU budget confirmation → small → fit → large.
  - This supersedes the (a)/(b) options discussion and the stub-rate trigger rule (dropped).
- **2026-11-07 ~15:55 IST**: User confirmed the 8-model list.
  - Models: OCI (stripped), Q1.5B, Q3B, CodeLlama-7b-hf, Llama-2-7b-hf and **Qwen2.5-Coder-7B base** (all
    stripped); CodeLlama-7b-Instruct and Llama-2-7b-chat (chat template).
  - Defaults stated, no decision needed: no prefix; the BigCode HumanEval stop list for all raw-prompted models;
    512 new tokens; OCI raw.
  - **No separate stripped smoke test.** The diag runs already exercised the stripped path for greedy and
    top_p0.95 on all 3 small models. The smoke checker runs on the full-run output instead, and the 7B fit run
    (5 tasks) is the 7B smoke test.
  - Next: research-coder → code-reviewer → commit/push/pull → launch `small` stripped (about 2 h 15 min, no cap,
    unless the user says hold).
  - MSI idle at 15:53 IST.
- **2026-11-07 16:00 IST: the 7B models move to GPU2 (user).**
  - Machine: `ssh gpu2` (10.132.3.11, user karrtik). 2× RTX 4090 24 GB, cc 8.9, driver 580.178.04 / CUDA 13.0;
    129 GB free disk; both GPUs idle.
  - The existing `~/pless-for-coding` (branch `feat/pless-variants-mps`, other work) is **left untouched**.
  - New clone: `~/pless-for-coding-he-regen` at branch `feat/humaneval-regen`, `5fd6b5b`, submodules initialised.
    Env `.venv-vllm033` via `uv sync --frozen --project envs/vllm-033` (started 16:02:57 IST).
  - Planned queues (one model at a time per GPU, Llama models last on both):
    - GPU0: Qwen2.5-Coder-7B (stripped) → CodeLlama-7b-hf (stripped) → Llama-2-7b-hf (stripped);
    - GPU1: CodeLlama-7b-Instruct (chat) → Llama-2-7b-chat (chat).
  - Each model: fit test (5 tasks; acts as its smoke test) → full run. Needs the stripped-variant runner change
    (research-coder, pending review) pulled first, and the user's HF login on GPU2 before the Llama models.
  - **Hardware disclosure:** the small models run on the MSI (RTX 4090 Laptop 16 GB), the 7B models on GPU2
    (RTX 4090 24 GB). Same vLLM 0.33.0 env lock. Every config of a model runs on one machine, so comparisons
    within a model are unaffected.
  - Per CLAUDE.md gotcha 9, `bench.eval` grading stays local, never on GPU2.
- **2026-11-07 16:07 IST: GPU2 ready.**
  - Env `.venv-vllm033` at the clone root: vllm 0.33.0, torch 2.14.0+cu130, transformers 5.20.2, 2 GPUs visible.
    `uv pip freeze` is **identical to the MSI's** (211 packages each, empty diff).
  - Gotcha: `UV_PROJECT_ENVIRONMENT=.venv-vllm033` (relative, as in the `envs/vllm-033/pyproject.toml` comment)
    resolves against `--project envs/vllm-033`. Use an absolute path (`$PWD/.venv-vllm033`). The misplaced env was
    removed. TODO: fix the comment in `envs/vllm-033/pyproject.toml`.
  - HF login on GPU2, at the user's request using the local `.env` `HF_TOKEN`: the value was piped over SSH stdin
    into `huggingface_hub.login()` and never printed. The `.env` file was not copied. Logged in as `karrtikiyer`;
    access OK to `meta-llama/Llama-2-7b-hf` and `-chat-hf`.
- **2026-11-07**: Stripped-protocol runner change reviewed (code-reviewer).
  - Verdict: safe for single-queue runs. Dry run confirmed:
    - every raw-prompted model gets `--prompt-variant stripped` on its vLLM and beam4 calls;
    - chat models get no flag;
    - Qwen-7B base gets top_p0.95_t0.2, last;
    - the new roots don't collide.
  - SHOULD-FIX before GPU2's parallel queues:
    - MANIFEST.json unlocked read-modify-write (reproduced: 290 of 600 entries kept, 33 exceptions);
    - no guard against the same model in both queues.

    Sent to research-coder.
  - NITs:
    - the stripped root mixes `<config>__prompt_stripped.jsonl` (base) with `<config>.jsonl` (chat), so the
      table builder must not glob only `<config>.jsonl`;
    - beam4 + stripped was never run in diag, so check it first on the full-run output.
  - Committed `c8994d0`, pushed.
- **2026-11-07 16:57:36 IST: STRIPPED SMALL RUN STARTED on the MSI** (`PROMPT_VARIANT=stripped
  ./run_regen_humaneval_8models.sh small`; marker `logs/regen_stripped_small.done/.failed`).
  - The first log line confirms `| prompt-variant stripped -> results/humaneval_regen_2026_11_stripped`.
  - Estimate about 2 h 15 min, no cap.
- **2026-11-07 17:32:34 IST: 7B FIT TESTS STARTED on GPU2** (GPU0 only, single queue, code `c8994d0`; safe per the
  review for single-queue runs).
  - Order: Qwen2.5-Coder-7B → CodeLlama-7b-hf → CodeLlama-7b-Instruct → Llama-2-7b-hf → Llama-2-7b-chat.
  - Each fit = greedy (vLLM) + beam4 (HF), 5 tasks, into `results/humaneval_regen_2026_11_stripped_fit/`.
  - Marker: `logs/regen_stripped_fit.done/.failed`. Disk free 117 GB before the model downloads (5 × ~13.5 GB).
  - The parallel full 7B queues wait for the MANIFEST/queue-lock fix (research-coder running).
- **2026-11-07 17:44 IST**: Status.
  - GPU2: Qwen2.5-Coder-7B fit done (greedy + beam4, 5 tasks each, about 9 min including the download); CodeLlama-7b-hf
    fit running; GPU0 13.5 GB of 24 GB; 0 tracebacks.
  - **First beam4 + stripped output, inspected by the main session:**
    - all 10 Qwen-7B fit samples start with `\n    ` and a 4-space body; finish_reason `stop`;
    - the stored prompt ends `    """`.

    The reviewer's "beam4 + stripped untested" gap is closed for Qwen-7B.
  - MSI: OCI greedy/temp/top_p0.9/pless done (164), pless_norm 72/164; 0 tracebacks.
  - The lock fix is done (research-coder: 5 new tests, all failing on HEAD; 87 targeted and 946 full tests pass).
    Code-reviewer running. Don't `git pull` on GPU2 until the fit loop finishes, since its `bash -c` loop
    re-invokes the script per model.
- **2026-11-07**: Lock fix reviewed (no blockers, no should-fix; nits only: file mode, `--configs` dedupe,
  dir fsync). Committed `8919d00`, pushed.
- **2026-11-07 18:18:11 IST: 7B fit tests DONE**, 5/5 models, 0 tracebacks.
  - Smoke checker on `results/humaneval_regen_2026_11_stripped_fit`: **HARD CHECKS: PASS**.
  - All raw-prompted 7B stripped starts are new-line with a 4-space body: 5/5 per file, 0 `"""`-line, 0 empty.
    CodeLlama-7b-hf is at 100%, so the old 3-space bug is not back.
  - s/task (greedy / beam4):

    | model | greedy | beam4 |
    |---|---|---|
    | Qwen-7B | 0.5 | 2.1 |
    | CodeLlama-7b-hf | 0.4 | 1.2 |
    | CodeLlama-7b-Instruct | 1.1 | 1.8 |
    | Llama-2-7b-hf | 4.2 (2/5 at cap) | 6.6 |
    | Llama-2-7b-chat | 0.8 | 1.6 |

- **2026-11-07 18:19:51 IST: FULL 7B QUEUES STARTED on GPU2** (code `8919d00`, `PROMPT_VARIANT=stripped`).
  - GPU0: Qwen2.5-Coder-7B → CodeLlama-7b-hf → Llama-2-7b-hf.
  - GPU1: CodeLlama-7b-Instruct → Llama-2-7b-chat.
  - Markers: `logs/regen_stripped_large_gpu{0,1}.done/.failed`. Disk free 52 GB after the model downloads.
  - Time estimate [guess]: GPU0 about 3 h (Llama-2-7b-hf dominates), GPU1 about 1 h.
    - Basis: fit s/task above × 164 tasks, with sampled n=10 configs assumed at 2–3× greedy s/task (the MSI
      small run showed about 2×).
    - Range 2–5 h.
- **2026-11-07 19:07:00 IST: STRIPPED SMALL RUN DONE (MSI)**: 21 files × 164 lines, 3444 total, 0 tracebacks;
  about 2 h 10 min.
  - Copy + integrity handed to data-wrangler. The MSI MANIFEST/README are saved locally as `MANIFEST.msi.json` /
    `README.msi.md`, because GPU2 writes a root with the same name.
  - Grading waits for the single final pass after GPU2 finishes (grading-load protocol).
- **2026-11-07**: Small stripped copy **DONE** (data-wrangler).
  - 23 sha256 match; 3444 rows before and after. All integrity checks pass: prompt_variant stripped everywhere,
    task_prompt == `.strip()` 164/164, versions identical across 21 files (`c8994d0`, not dirty).
  - **Main session re-ran the smoke checker: HARD CHECKS: PASS.** 4sp 100% everywhere, 0 nl-other.
    - `"""`-line continuations: Q1.5B 1–17 per file, Q3B 8–80, OCI 0.
    - Empty samples: up to 98/1640 (Q3B top_p0.9 T1.0), 66 (Q1.5B top_p0.9), 37 (OCI top_p0.9); more at higher T.
  - pless ≡ pless_norm raw samples: Q1.5B 164/164 tasks, Q3B 164/164, OCI 162/164.
  - Diag vs full run (same prompt and seed): greedy identical 164/164 ×3. top_p0.95_t0.2 is identical for OCI
    (1640/1640) but not for the Qwen models (1394/1640, 1371/1640).
    - Cause unverified. Candidates: engine state from the earlier configs in the same process (e.g. prefix
      caching) or batch-shape numerics (no batch invariance on cc 8.9).
    - Not a validity issue: same decoder and seed, different draws. But **do not claim bit-reproducibility across
      different invocations** for sampled configs. Only "the same invocation rerun is identical" was tested
      (smoke rerun).
- **2026-11-07 20:47:57 IST: BOTH 7B QUEUES DONE (GPU2)**: 31 files × 164 lines (5084 total in GPU2's root),
  0 tracebacks; about 2 h 28 min wall-clock (estimate was 2–5 h).
  - **All 8 models are now generated.**
  - Copy + integrity handed to data-wrangler. GPU2's MANIFEST/README are saved locally as `*.gpu2.*`.
  - Next: the final grading pass (all 8 models, quiet machine) → official cross-check → results-validator.
- **2026-11-07**: 7B copy **DONE** (data-wrangler).
  - 31 sha256 match; 5084 rows before and after. All integrity checks pass, including one BOS for the chat models.
    Versions are identical across the 31 files (`8919d00`, not dirty).
  - vs the MSI: identical except **Python 3.12.3 (GPU2) vs 3.12.14 (MSI)**. vllm/torch/transformers/datasets/cuda/env
    are identical. No effect on generation expected; disclose.
  - Smoke checker on the 5 models, run on GPU2 (the gated Llama-2 tokenizers need the HF login; the checker doesn't use
    bench.eval):
    - 4sp 100% on every raw-prompted file; 0 `"""`-line for CodeLlama/Llama-2;
    - Qwen-7B empty samples up to 177/1640 (top_p0.9 T1.0) and `"""`-line up to 41.
  - **Hard-check failure WAIVED (documented):** "vLLM text differs from reference" on 9 samples (CodeLlama-7b-hf pless/pless_norm
    1+1; Llama-2-7b-hf pless/pless_norm 3+3, temp 1).
    - Main session inspected all 9: finish_reason `length` at 512 tokens, **no stop sequence anywhere in the text** (cut == raw),
      text ending mid-CJK.
    - So vLLM's stop strings had nothing to act on. The mismatch is [Likely] a multi-byte character cut at the token cap,
      decoded differently by vLLM and HF. At most 1 sample per 1640 per file; runaway completions.
  - Also noted for the analysis: at-cap counts are high for p-less on Llama-2-7b-hf (253/1640) and CodeLlama-7b-hf (120/1640)
    vs temp (52, 34).
  - pless ≡ pless_norm: 1640/1640 on four models; Llama-2-7b-hf 1639/1640.
- **2026-11-07 20:58 IST**: Final grading pass **waiting for a quiet machine**: load 6.0; another session's Ministral
  LiveCodeBench `bench.eval` series holds `/tmp/pless_cpu_heavy.lock`. A background watcher fires after 3
  consecutive minutes with no `bench.eval` process and 1-min load < 2.
- **2026-11-07 21:20:02 IST**: Quiet (the user stopped the Ministral evals). Load 1.63 / 2.48 / 3.74.
  - **FINAL GRADING PASS started** (experiment-harness): 52 files, one at a time via the `bench.eval` CLI,
    official grader, `--skip-structural-diversity`, no cache.
  - Load logged per model; the pass pauses if foreign load is above 3. Then the cross-check per model.
  - Tables are staged at `results/humaneval_regen_2026_11_stripped/analysis/final_pass_tables.md` (not paper).
  - Correction: the earlier "59 files" mentioned to the user also counted diag files; the reported set is 52.
- **2026-11-07 21:45 IST: FINAL GRADING PASS DONE** (experiment-harness; commit `8919d00`).
  - 52/52 graded: num_tasks 164, grader `humaneval-official-v1`, timeout 3.0 s, cb_div hashseed 0.
  - Official cross-check 400/400 agree for each of the 8 models.
  - Load:
    - 2.43 at the start: slightly above the "about 2" target, below the pause threshold 3, no foreign `bench.eval`;
    - 4.3–5.8 during grading, all from this pass's own 4 workers (WindowServer about 42% of a core was the only
      foreign process).
  - Timeouts above 1% (4 files: CL-Instruct beam4 2/164, CL-hf top_p0.9 19/1640, L2-chat beam4 3/164, L2-hf top_p0.9
    26/1640). All 50 timed-out programs, re-run one at a time with a 20 s limit, time out again (13 have a literal
    `while True`). They are genuinely non-terminating, not load-induced.
  - Tables are staged at `results/humaneval_regen_2026_11_stripped/analysis/final_pass_tables.md`. Main session
    spot-checked 9 pass@1 values from the metrics JSONs: all match.
  - Hasan models, our top_p0.95_t0.2 stripped vs Hasan: OCI 61.40 vs .59; Q1.5B 43.54 vs .44; Q3B 51.34 vs .51; Q7B 61.16 vs .62.
  - pless ≡ pless_norm per-task pass_results in all 8 models.
  - **NOT VALIDATED yet**: results-validator running.
- **2026-11-07: FINAL RESULTS VALIDATED (results-validator): "safe to build Table II: yes".**
  - All 52 metrics recomputed (0 mismatches). Independent re-execution 315/320 agree; the 5 differences are
    CodeLlama-Instruct samples repaired by the pre-registered indent fix.
  - Paired comparisons + sign tests: `results/humaneval_regen_2026_11_stripped/analysis/validated_comparisons.md`
    (staging). Main session re-checked 2 cells.
  - p-less-norm was validated separately: per-task results identical to p-less in all 8 models.
  - Disclosures needed: OCI +2.4 pp vs Hasan (unexplained); the CodeLlama-Instruct indent repair is uneven across
    configs; p-less ≈ greedy on the chat models; some bootstrap-significant gains are not sign-test supported;
    hardware split; Python patch versions; Amendment A1.
  - Scripts were preserved from /tmp into `analysis/validator_scripts/` and `analysis/grading_scripts/` (results/,
    not committed).
  - **Next: Table II rebuild into STAGING** (research-coder + code-reviewer), then the user's final review. No paper
    edits until then.
- **2026-11-08**: User asked for two things.
  1. **Table II rebuild in staging.** A new script, `scripts/paper/stage_humaneval_regen_table.py`, writes only into
     `results/humaneval_regen_2026_11_stripped/analysis/table_ii_staging{,_nodash}.{md,tex}`.
     - It covers 8 models × 7 configs (p-less-norm as its own row) and the columns pass@1/3/5/10, cov@0.3/0.7, cb_div.
     - n=1 cells come in both renderings. The caption is a draft.
     - The existing paper scripts are not run or modified.
  2. **Extra comparison configs, NOT for the paper:** p-less and p-less-norm at T=1.0 and T=0.6, all 8 models.
     - Everything else is identical: stripped/chat prompt rules, seed 42, n=10, 512, stop list, the same machines (small → MSI,
       7B → GPU2 two queues, Llama last).
     - New mode `extra`, root `results/humaneval_regen_2026_11_stripped_extra/`.
     - Estimate [Likely]: about 65 min on the MSI, about 35–45 min on GPU2.

  Both code changes are with research-coders in parallel (separate files); code-reviewer next.
- **2026-11-08: Table II STAGED** (`results/humaneval_regen_2026_11_stripped/analysis/table_ii_staging{,_nodash}.{md,tex}`).
  - Code-reviewer: **0/1456 cell mismatches** vs independent recomputation; both .tex compile (tectonic, IEEEtran);
    guards OK.
  - Should-fix applied: the caption no longer claims n=10 for greedy/beam.
  - Notes for the user's review:
    - p-less-norm rows equal p-less in all 8 models; this needs one sentence in the caption or text;
    - the group headers use `\hspace{-1.5mm}` to match Table I;
    - nodash cb_div 0.0000 for n=1 reads as "zero diversity".
  - Script committed; outputs not committed (results/). **Awaiting the user's review; no paper edits.**
- **2026-11-08**: Extra-config change reviewed (safe; differs from T=0.7 only in temperature; defaults unchanged).
  Committed `7a6b852`.
  - **MSI `extra` (OCI, Q1.5B, Q3B) started 06:03:21 IST; DONE before 09:06 IST**: 12 files × 164. Copy handed to
    data-wrangler (`MANIFEST.msi.json`).
  - GPU2 launch failed at 06:03: the local 1Password SSH agent offered no identities (locked). Nothing changed on
    GPU2. **The user unlocked it; GPU2 `extra` started 09:06:17 IST** (GPU0: Qwen-7B → CodeLlama-7b → Llama-2-7b;
    GPU1: CodeLlama-7b-Instruct → Llama-2-7b-chat). Markers: `logs/regen_extra_gpu{0,1}.done/.failed`.
  - Note: GPU2 SSH goes through the 1Password agent (`IdentityAgent` in `~/.ssh/config`). If it's locked,
    connections fail with "signing failed … communication with agent failed".
- **2026-11-08**: MSI extra copy **DONE** (data-wrangler).
  - 14 sha256 match; 1968 rows before and after. Integrity OK (`7a6b852`). HARD CHECKS: PASS.
  - **p-less vs p-less-norm are NOT identical at T=1.0.** Byte-identical tasks:
    - Q1.5B: **100/164** at T=1.0 vs 164 at T=0.6 and T=0.7;
    - Q3B: **82/164** at T=1.0 vs 164 at T=0.6 and T=0.7;
    - OCI: 162/164 at all T.
  - [Likely] Mechanism: a flatter distribution at T=1.0 puts more tokens inside the ≤(1−L)/(|V|−1) gap between the two
    cutoffs, and one differing token changes the rest of the sample.
  - This confirms the user's rule: always treat p-less-norm as its own config
    ([[feedback_never_drop_configs_from_tables]]).
  - Empty samples: Q3B 23/27 (p-less / norm) at T=1.0, 9 at T=0.6; Q1.5B 15–16.
- **2026-11-08 10:43:03 IST: GPU2 extra DONE**: 20 files × 164 (3280 rows), 0 tracebacks; about 1 h 37 min (the estimate
  of 35–45 min was too low). Copy handed to data-wrangler (`MANIFEST.gpu2.json`). Then one grading pass for all
  32 extra files, on a quiet machine.
- **2026-11-08**: GPU2 extra copy **DONE** (data-wrangler).
  - 22 sha256 match; 3280 rows before and after. Integrity OK (`7a6b852`).
  - p-less vs p-less-norm byte-identical tasks (T=1.0 / T=0.6): Q7B 164/164, CL-hf 163/164, L2-hf 162/163, CL-Inst 164/164,
    L2-chat 164/164.
  - Smoke checker (on GPU2): FAIL only on 8 samples with the known waivable signature (512 cap, no stop sequence,
    cut == raw). **Waived**, as on 2026-11-07. 4sp 100%.
  - at-cap: L2-hf 212–272/1640, CL-hf 102–124. Empty: Q7B 10–14.
- **2026-11-08 10:47 IST**: Quiet (load 1.76, no bench.eval; the lock file only shows a stale label). **Extra grading
  pass started** (experiment-harness). Output: `results/humaneval_regen_2026_11_stripped_extra/analysis/extra_pass_tables.md`.
- **2026-11-08 11:11 IST: extra grading DONE.**
  - 32/32: 164 tasks, official grader, hashseed 0. Max timeout 0.85%. Cross-check 400/400 ×8.
  - Load 2.50 at the start (above the ~2 target, below 3, no foreign eval); 4.4–6.1 during, from our own workers.
  - Per-task pass_results identical p-less vs p-less-norm: 164/164 at T=0.6 and T=0.7 for all 8 models; at T=1.0 all
    except **Q1.5B 148** and **Q3B 142**.
  - Main session spot-checked 6 cells (e.g. Q1.5B T1.0 p-less 44.15 vs norm 44.27 pass@1, 60.98 vs 62.80 pass@10;
    Q7B T1.0 pass@10 77.44): all match.
  - NOT VALIDATED yet; results-validator running (recompute, paired T and p-less vs norm comparisons).
- **2026-11-08: extra results VALIDATED** ("trustworthy: yes"). Summary in
  `results/humaneval_regen_2026_11_stripped_extra/analysis/validated_temperature_comparisons.md`; scripts saved from /tmp.
  - T=1.0 vs T=0.7: pass@10 up on all 8 models (significant on most) and cb_div up on all 8, but **pass@1 not improved**
    (Q7B −1.8 pp, sign p=0.037).
  - p-less vs p-less-norm at T=1.0: no detectable difference anywhere.
  - The chat models stay near-greedy at every T.
  - 112 uncorrected tests, one seed: wording must be cautious.
  - **Extra-config work DONE. Not for the paper unless the user decides otherwise.**
- **2026-11-08: Table II reordered and put into the paper repo** (user: match the MBPP order, drop the Hasan config, update the paper
  once the review passes).
  - Staging script `e15b370`: MBPP model and config order and labels, Qwen2.5-Coder-7B last, 6 configs. Code-reviewer:
    0/1344 cell mismatches; .tex compiles.
  - Paper repo `tables/table_3b_humaneval.{tex,md}` ← the nodash variant (computed values for greedy/beam, as in MBPP
    Table I), with staging/draft comments stripped and the body byte-identical. Caption "Performance of eight code models on
    HumanEval (164 problems; n=10 samples for sampling decoders, one for greedy and beam)"; label unchanged.
  - A temp-copy compile with tectonic builds: 7 pages, references start p.6. Not committed yet.
  - **Stale (not edited):**
    - main.tex HumanEval text: l.39/64 ranges, l.99 "7B-Instruct", l.119 "ten … experiments", l.121
      p-less@0.6 / −7.87 pp;
    - the code repo's `paper/llm4code/tables/table_3b_humaneval.*`;
    - `make_llm4code_tables.py` would regenerate the OLD Table II if run.
- **2026-11-08 ~12:12 IST: INCIDENT.** The research-coder fixing `make_llm4code_tables.py` ran the unmodified script against
  the REAL output dirs, against its brief.
  - The paper repo `tables/table_3b_humaneval.*` reverted to the old committed Table II. `table_3a_mbpp.*` was rewritten
    but is byte-identical to HEAD.
  - The code repo `paper/llm4code/tables/*` was rewritten by the same (current, uncommitted) script version.
  - The agent was stopped. `make_llm4code_tables.py` itself is unedited (197-line uncommitted diff intact).
  - Leftover untracked files: `tests/fixtures/make_llm4code_tables_prechange.py` (copy of the script) and
    `tests/test_make_llm4code_tables_humaneval_mode.py` (unfinished).
  - **The make_llm4code fix is NOT done.**
- **2026-11-08: Table II font fix + restore.**
  - Cause: both tables use `\resizebox{\columnwidth}{!}`. Table II's widest label (23 chars) is shorter than MBPP's
    (`top-p@0.95 with T=0.2 (Hasan)`, 30), so it was scaled up: measured 5.573 pt vs 5.169 pt.
  - Fix in the staging script: the header's first cell is `\rlap{Decoder}\phantom{<MBPP widest label>}`. Test-first; 23 tests.
  - Measured after the fix: **both 5.169 pt** (pdfplumber, tectonic build in /tmp; still 7 pages). MBPP files untouched.
  - Paper repo Table II restored from the new staging file (data rows identical). Code-reviewer pending.
- **2026-11-08**: Font fix reviewed and committed (`c6da222`); paper repo Table II committed (`a9dab94`).
- **2026-11-08: Overleaf export `pless_for_coding_paper_8_Nov_2026.zip` (12:31 IST) merged into the paper repo**, file by file:
  - Taken from the zip: `main.tex`, `main.pdf` and the `.tex` tables 3a/3b/a1/a2. These are the user's Overleaf edits:
    - captions on 3a/3b/a1/a2; `\hspace` on the a1/a2 headers;
    - in main.tex: the Acknowledgment section, the Appendix A rewrite, the peakedness-figure appendix, and §II-D wording.
  - Kept from the repo: the `.md` tables (the zip's were stale), `refs.bib` (EOF newline only), `fig6b` (unused).
  - Table II data rows unchanged vs `a9dab94`.
  - **TODO:** the code-repo generators still carry the OLD captions, so a rerun would revert the user's caption edits:
    - staging `CAPTION` → "Performance of eight code models on HumanEval";
    - make_llm4code_tables 3a/a1/a2 captions.

    Fold this into the make_llm4code safeguard work.
- **2026-11-08**: Overleaf merge committed in the paper repo (`01bce0a`).
  - Merged paper compiles (tectonic): 7 pages; Table I and II both 5.169 pt; all 8 `\input` tables present and identical to the zip.
  - **make_llm4code safeguard + caption sync restarted** with ENFORCED protection: the 4 output dirs (both repos'
    tables/figures) are `chmod -R a-w`, checksum snapshot in `/tmp/paperdirs_snapshot.sha` (50 files); a write test was refused.
  - **Restore write permission (`chmod -R u+w`) after the work and its review.**
  - Then commit and push both repos (user request).
- **2026-11-08: make_llm4code_tables safeguard DONE** (research-coder; code-reviewer: safe; main session added the review's
  should-fix, a loud warning when `--mirror-only` overwrites an Overleaf-edited table).
  - Default `--humaneval-table regen` reproduces all 8 paper-repo files (3a/3b/a1/a2 .tex+.md) byte-identically.
  - `legacy` writes the old Table II only to `paper/llm4code/tables/legacy/` unless `--allow-legacy-paper-overwrite`.
  - The mirror skips the Overleaf-edited 5a/a3 by default.
  - 57 tests. Locked dirs verified 50/50 unchanged throughout, then unlocked.
  - D16 added (A2 on the old protocol).
- **2026-11-08: COMMITTED + PUSHED both repos** (user request after review).
  - Code `e62bcd1` → origin/feat/humaneval-regen; paper `01bce0a` → origin/main. Both in sync.
- **2026-11-08: two new analysis-only jobs (user; NOT for the paper; staging):**
  1. **MBPP (500 tasks, bigcode prompt, HF runner) for Qwen2.5-Coder-7B (base)** on GPU2 GPU0.
     - Exactly as the other 7 MBPP models (template `run_bigcode_mbpp_qwen25coder3b.sh`).
     - 7 configs: MBPP's 6 + top-p@0.95 T=0.2 (a Hasan model).
     - Root `results/mbpp_qwen25coder7b_2026_11/`.
  2. **HumanEval, Qwen2.5-Coder-7B-Instruct** on GPU2 GPU1.
     - **Our chat protocol** (user answer), the 6 Table II configs.
     - Root `results/humaneval_regen_2026_11_instruct7b/`, via a new `instruct7b` mode.
  - Research-coder preparing both; code-reviewer next; then GPU2 launch after a GPU-time estimate.
  3. **(added by the user) MBPP for Qwen2.5-Coder-7B-Instruct.**
     - On whichever GPU2 GPU frees first. Same way as the other instruct MBPP rows (CodeLlama-Instruct, Llama-2-chat).
     - The 6 MBPP configs (no Hasan). Root `results/mbpp_qwen25coder7binstruct_2026_11/`.
  - User approval: start once the review and smoke runs pass (no separate budget confirmation; report the measured estimate).
  - **MBPP stripped/unstripped check (user question), verified by the main session:** BigCode `tasks/mbpp.py` l.55 is
    `f'"""\n{description}\n{test_example}\n"""\n'`, with no strip option or variant. The harness only prepends an empty prefix
    (`utils.py` l.61). Hasan's fork's mbpp.py is identical to upstream.
    - Ours, `bench/prompts.py:83` `format_prompt_base_bigcode`, is the same string, so **no change for MBPP**.
    - Instruct models use the chat template, so stripping is not applicable.
    - The remaining known difference is the stop list (E29), kept for comparability with the 7 existing MBPP models.
- **2026-11-08**: Three GPU2 job setups reviewed (code-reviewer: safe to smoke-run; flags equal the original MBPP runs:
  HF, bf16, mbpp-config full, 512, n=10 or n=1 for greedy/beam).
  - Scripts: `run_bigcode_mbpp_qwen25coder7b.sh`, `run_chat_mbpp_qwen25coder7binstruct.sh`, `scripts/mbpp_staging_common.sh`; regen
    `instruct7b` mode.
  - GPU2 main env (`uv sync --frozen`): torch 2.11.0+cu130, transformers 5.3.0.
  - **Code/env drift vs the existing MBPP rows (disclose in any comparison):**
    - p-less sampler changed in `118292e` (2026-06-05): 1e-3 uniform smoothing → argmax fallback. The reference rows Qwen-3B
      pless (2026-04-01) and CodeLlama-Instruct pless (2026-03-31) used the smoothed version. The commit reports 0.0138% of
      positions differ (not re-verified).
    - The lock moved from torch 2.10.0 → 2.11.0 (transformers 5.3.0 both).
    - The Qwen-3B temp/top-p rows predate the BPE decode fix `b256b0c` (said not to affect Qwen; unverified).
    - No seed in the MBPP runner, as in the original runs, so not bit-reproducible.
- **2026-11-08 13:39–13:47 IST: GPU2 smokes** (code `ce09f99`).
  - HE instruct7b smoke (5 tasks × 6 configs): HARD CHECKS PASS; 1.3–2.7 s/task.
  - First MBPP smoke failed: `uv` not on the non-interactive PATH on GPU2. Rerun with `export PATH=$HOME/.local/bin:$PATH`.
  - MBPP smokes then passed: base 7 configs, instruct 6, 3 tasks each.
    - Sample counts: n=1 for greedy/beam, n=10 otherwise; base completions start `def …`, instruct replies start with a python fence.
    - Record keys match the old 3B rows.
- **FULL RUNS started on GPU2** (approved: start once review and smoke pass).
  - GPU0: MBPP Qwen-7B base, about 21 s/task over 7 configs → ~3 h (2.5–4.5).
  - GPU1: HE instruct7b (~0.5 h) → MBPP Qwen-7B-Instruct (~1.3 h).
  - Markers: `logs/full_gpu{0,1}.done/.failed`. The MBPP runs resume after the 3 smoke tasks (same settings, by design).
- **2026-11-08: user added 4 configs to all three running jobs:** p-less / p-less-norm at T=1.0 and T=0.6.
  - Research-coder is adding an opt-in `EXTRA_PLESS_TEMPS=1` to the two MBPP scripts (same roots, standard names) and an
    `instruct7b-extra` HE mode (root `results/humaneval_regen_2026_11_instruct7b_extra/`).
  - Then code-reviewer, then queued after the current runs on each GPU (pull only after they finish).
  - Estimate [Likely]: GPU0 about 40 min; GPU1 about 15 + 40 min.
  - Naming note: "Qwen2.5-Coder-7B(-Instruct)" throughout; plain "Qwen-7B" in this project means `Qwen/Qwen-7B` (Appendix A1).
- **2026-11-08 15:21 IST: GPU1 queue FAILED, by design.**
  - HE instruct7b DONE (6 configs × 164). MBPP-instruct greedy/beam4 DONE (500 each).
  - MBPP-instruct temp_t0.7: **CUDA OOM on task_id 493** (22.3 GB in use on 24 GB). The runner swallowed it → 499 rows. Our
    per-task-error check stopped the queue (the D14-style silent gap was avoided).
  - Extras change reviewed, committed and pushed (`06a5441`); pulled on GPU2. GPU0's running job is unaffected (default commands identical).
- **2026-11-08 15:23 IST: GPU1 queue restarted** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator only):
  - MBPP-instruct resume (task 493 + top_p0.9/pless/pless_norm) → HE `instruct7b-extra` → MBPP-instruct `EXTRA_PLESS_TEMPS=1`.
  - GPU0: an on-machine waiter starts MBPP-base `EXTRA_PLESS_TEMPS=1` after `logs/full_gpu0.done`. Markers `logs/q1.*`, `logs/q0x.*`.
  - If task 493 OOMs again: stop and ask the user. Don't drop it.
- **2026-11-08 16:34 IST**: GPU0 MBPP base main 7 configs DONE (500 each; the script verified 500 unique ids).
- **2026-11-08 17:38:38 IST: ALL GPU2 analysis runs DONE** (`q0x.done`, `q1.done`; 0 OOM in the restarted logs).
  - MBPP base: 11 files × 500 (7 main + 4 extras).
  - MBPP instruct: 10 × 500. Task 493 regenerated OK with `expandable_segments`.
  - HE instruct7b: 6 × 164; HE instruct7b_extra: 4 × 164.
  - Copy + integrity handed to data-wrangler. Then: local grading (MBPP `bench.eval --dataset mbpp`; HE official grader) on a
    quiet machine → results-validator → staging tables.
- **2026-11-08**: Qwen2.5-Coder-7B copy **DONE** (data-wrangler).
  - 81 files sha256 match (31 JSONL + manifests + 46 GPU2 logs in `results/mbpp_qwen25coder7b_2026_11/gpu2_logs/`).
  - MBPP: 21 × 500, prompts identical to the 3B rows, 0 empty. Task 493 present (regenerated 15:24 IST).
  - HE: 10 × 164, chat prompts identical across files; smoke checker PASS ×2.
  - p-less ≡ p-less-norm byte-identical tasks:
    - HE instruct: 164/164 at every T (seeded vLLM);
    - MBPP base: 167 / 114 / 28 (T0.6 / 0.7 / 1.0); MBPP instruct: 363 / 344 / 257 (unseeded HF runner, so identity ≈ tasks that
      collapsed to one output).
- **2026-11-08 17:44 IST**: Grading started (experiment-harness; load 2.42; analysis-only, not the pre-declared paper pass).
  - MBPP via `bench.eval --dataset mbpp`, same settings as the old MBPP metrics; HE via the official grader; cb_div only.
  - Output: `results/mbpp_qwen25coder7b_2026_11/analysis/qwen25coder7b_tables.md`.
- **2026-11-08 18:11 IST: grading DONE** (experiment-harness).
  - 21 MBPP (500 each, `--dataset mbpp --skip-structural-diversity`, 5.0 s / 4 workers) + 10 HE (164, official grader). HE
    cross-check 400/400.
  - Load 2.88 at the start (target about 2, below 3), up to 6.9 from our own workers.
  - MBPP timeout% is null from the grader; a separate probe re-ran all failures (0 flipped; max 0.32%).
  - MBPP programs run under the system `python3` (Homebrew 3.14.4), as for the old rows.
  - Main session spot-checked 5 values (MBPP base greedy 64.00; base p-less-norm T0.7 64.28/71.80; instruct top-p 74.80/90.20;
    HE instruct greedy 87.20; HE p-less T1.0 87.87/89.63): all match.
  - pass@1 headline: MBPP base greedy 64.00, p-less T0.7 63.82, T@0.7 50.38; MBPP instruct greedy 78.00, p-less 77.58, T@0.7 74.92;
    HE instruct greedy 87.20, p-less 88.29, T@0.7 86.40. NOT VALIDATED; results-validator running.
- **2026-11-08: Qwen2.5-Coder-7B analysis VALIDATED** ("trustworthy: yes").
  - Summary + paired tests in `results/mbpp_qwen25coder7b_2026_11/analysis/validated_qwen25coder7b_comparisons.md`;
    scripts saved from /tmp.
  - The new rows fit the 9 reference MBPP models' pattern.
  - New caveats:
    - MBPP-instruct p-less gives a deterministic non-greedy output in 201/500 tasks (hypothesis: bf16 batch-size numerics in
      the custom loop vs `generate`; unverified);
    - instruct MBPP prompts show all 3 graded asserts;
    - the HE p@1 edge over greedy rests on 4 tasks.
  - **Analysis work DONE (staging only).**
- **2026-11-08: NEXT analysis (user): MBPP for ALL HE models with today's code, SEEDED.**
  - User decision: add a fixed seed to the MBPP runner and rerun the two Qwen2.5-Coder-7B MBPP jobs too, so all MBPP rows are on one
    protocol.
  - The 7 Table II models besides Qwen-7B will be regenerated (their April MBPP rows used older code).
  - **E28 double BOS CONFIRMED** with the runner's own code (`format_prompt_instruct` + `generator.py:214`
    `tokenizer.encode`), transformers 5.3.0: CodeLlama-7b-Instruct-hf and Llama-2-7b-chat-hf receive `[1, 1, …]` (2 BOS).
    Qwen2.5-Coder-7B-Instruct is unaffected (no BOS). The HE regeneration is single-BOS (smoke-checked).
  - **User: analyse whether it matters before fixing.** Plan: greedy on all 500 MBPP tasks, double vs single BOS, both
    chat models (about 30 min on GPU2) → changed outputs + paired pass@1 CI.
  - Research-coder is adding opt-in `--seed` and `--single-bos` (default byte-identical) + a BOS-check script.
- **2026-11-07**: AI-use acknowledgment (D15 item 6):
  - A revised draft was given to the user, with placement: `\section*{Acknowledgment}` after the Limitations
    subsection, before `\bibliographystyle`. No paper edits by Claude (user rule).
  - Evidence for the disclosure: 415/421 code-repo commits and 19/37 paper-repo commits carry a
    Co-Authored-By: Claude trailer (first on 2026-02-25).
