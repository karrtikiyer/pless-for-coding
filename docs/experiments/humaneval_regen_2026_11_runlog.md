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
