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
| 5 Full runs, small models | DONE | 20:00:38 → 22:13:35 IST 2026-11-06; 21 files × 164 lines |
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
- **2026-11-07**: AI-use acknowledgment (D15 item 6):
  - A revised draft was given to the user, with placement: `\section*{Acknowledgment}` after the Limitations
    subsection, before `\bibliographystyle`. No paper edits by Claude (user rule).
  - Evidence for the disclosure: 415/421 code-repo commits and 19/37 paper-repo commits carry a
    Co-Authored-By: Claude trailer (first on 2026-02-25).
