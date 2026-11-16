# Plan: a separate repo for measuring p-less variants on non-coding reasoning tasks

Status: DRAFT FOR REVIEW (2026-11-13). Nothing has been created or moved. Working name: `pless-reasoning-bench` (placeholder).
Evidence labels: **[verified]** = I read it in code/pages this session; **[reported]** = an agent reported it with file:line, I spot-checked only part of it; **[unverified]** = not checked; **[guess]** = my judgement.

## 0. Why, and the one-line scope

This repo was built for p-less on coding. It now carries 78 test files, ~80 `run_*.sh` scripts, a paper, LCB/APPS/MBPP/HumanEval pipelines and GPU2/MSI/Mac launch tooling [reported]. Adding non-coding datasets here mixes two purposes. The new repo should hold **only** what is needed to run p-less (authors' version), its variants, and standard baselines on reasoning datasets, and to measure accuracy and efficiency.

Scope boundary (needs your confirmation, decision D2): sampler code + the minimal measurement harness (engine, data adapters, grading, metrics). No results data, no paper, no coding benchmarks.

## 1. Facts the plan rests on

1. **Authors' p-less** = two PyTorch functions (`p_less_decode`, `p_less_norm_decode`, 61 lines), MIT, one commit `a681f23` (2025-10-02), identical to our submodule pin; remote head re-checked today [verified]. No vLLM/HF-generate integration, no tests, no temperature, no all-pruned guard [reported, matches repo README lines I read via fetch].
2. **vLLM upstream:** PR #45288 "Add the p-less Sampling Method" by `ryttry` (same account that owns the authors' repo), open since 2026-06-11, unmerged, repeated merge conflicts, no maintainer reply seen [verified via fetch; inline review comments not visible]. Boolean `p_less`, built-in logits processor + Triton kernel; **no p-less-norm** [verified via PR diff summary]. Cannot be installed on our pinned vLLM versions.
3. **Our vLLM port** uses the same thresholds and strict `<`, applies temperature first, masks logits, lets vLLM's sampler do the draw, and adds an argmax fallback when every token is pruned [reported]. Unverified: which logits width vLLM passes (sets `v` in p-less-norm).
4. **Parity tests today:** a CPU test compares our mask to the authors' function on surviving-token sets and empirical frequencies (within 0.02); the end-to-end GPU parity test is a skipped skeleton [reported].
5. **Variants exist twice** (prob-space `p_less_variants/` and logit-space `bench/pless_variants_vllm.py`) and no test compares them [reported].
6. **Coupling:** sampler/plumbing files import almost nothing from the coding code (`generator_vllm.py` imports only `bench.loop_detect`; `sampler_bridge.py`, `checkpointing.py`, `seeding.py`, `loop_detect.py` import nothing from the repo) [verified]. `bench/eval/metrics.py` is tangled: it imports `human_eval`, the code executor and `zss` at module level [verified]. `bench/generator.py` applies import-time monkeypatches to transformers [reported].
7. **Existing gaps to fix, not copy:** runner records carry no token counts/finish reasons; JSONL appends are unlocked; efficiency tooling hardcodes `</think>`; FlashInfer top-k needs `ninja` on PATH (the MSI failure); results-file naming assumes LCB platforms [verified/reported across this session].

## 2. Decisions needed from you (D1–D8)

- **D1 Name, owner, visibility, licence** of the new repo (private GitHub under `karrtikiyer`? our code licence MIT to match the authors'?). I will not create a remote without your OK.
- **D2 Scope reading:** "only code related to pless variants and default pless" = sampler code **plus** the minimal harness needed to measure it (recommended), or sampler code alone with the harness elsewhere?
- **D3 "Authors' version":** (a) the authors' reference functions run unmodified (pinned submodule) are the ground truth, and our vLLM processor is accepted only if it passes parity gates (recommended); (b) run the reference backend only (HF loop, very slow for 32k-token reasoning); (c) wait for vLLM PR #45288 (stalled; not recommended).
- **D4 Variants to carry:** pless, pless_norm, effk, effk_nofloor, min2, topmass(+nofloor), top1half, gini(+nofloor), Rényi G_k (`pless_renyi`) — recommended. Optional: `pless_alpha` (τ_α, your notes call it dropped from Paper B), top-nsigma (someone else's method), MLX ports (separate backend). Default recommendation: exclude τ_α, top-nsigma and MLX from v1; add later if needed.
- **D5 Backend:** vLLM only for measurement (pin 0.21.0); HF reference backend only for parity. mlx-lm excluded (not comparable).
- **D6 Dataset order:** GSM8K (pipeline check) → competition-math pool (AIME/HMMT/AMC) → GPQA Diamond → OlympiadBench for power; MATH-500 efficiency-only. (Final list is yours.)
- **D7 History:** fresh repo + `PROVENANCE.md` listing source files and source commit hashes (recommended) vs. `git filter-repo` to preserve history of moved files.
- **D8 This repo afterwards:** frozen as the coding-paper archive; one pointer line to the new repo; no further datasets here.

## 3. Target layout (proposal)

```
pless-reasoning-bench/
  pyproject.toml            # uv; python>=3.12; vLLM 0.21.0 lock (torch 2.11.0, transformers 5.8.1); extras for HF-reference and dev
  third_party/p-less/       # authors' repo, pinned submodule @ a681f23, unmodified; LICENSE kept
  src/plessbench/
    samplers/
      reference.py          # thin adapter around the authors' functions (HF backend, batch + all-pruned handling kept OUT of the reference path)
      variants.py           # ONE canonical logit-space definition per method (pless, pless_norm, effk, min2, topmass, top1half, gini, +nofloor ablations, renyi_g)
      registry.py           # name -> spec (family, params, native-vs-custom); baselines declared as vLLM-NATIVE only
    backends/
      vllm_backend.py       # logits-processor class registration, V1-runner env, ninja/PATH check, safe tokenizer, prompt encoding, metadata capture
      hf_reference.py       # token-by-token loop calling samplers/reference.py (parity runs only)
    data/                   # dataset adapters: id, prompt, gold, answer type, revision pin + content hash; gated sets read token from env, never log text
    grading/                # answer extraction + normalisation; MC letters; math equivalence; truncated/unparseable policy
    metrics/                # unbiased pass@k (local, no human_eval), maj@k, coverage, efficiency, paired bootstrap, table rendering
    runs/                   # run loop, seeding, locked append, resume, run manifest
    cli.py                  # plessbench run | grade | report | parity
  configs/                  # one YAML per experiment (model, datasets, methods, n, cap, seed, pins)
  tests/                    # TDD; sampler parity; grading fixtures; metrics; runner with fake engine
  docs/                     # how-to, PROVENANCE.md, pre-registration template
  results/                  # gitignored (data stays out of git); manifests + small summary tables only
```
Baselines (temperature, top-p, top-k, min-p) go **only** through vLLM native `SamplingParams` (project gotcha #8). No custom top-p/top-k code is carried over.

## 4. What moves, what is rewritten, what stays (file-level) [reported, sizes from `wc -l`]

| Source here | Action |
|---|---|
| `p-less/` submodule (pin `a681f23`) | re-add as `third_party/p-less` (unmodified); drop untracked `p_moment_decode.py` |
| `bench/generator_vllm.py` (1058) mask fns, processor class, engine loader, metadata capture | **extract**: keep mask functions + processor + loader + `_extract_completion_meta`; drop `"t_code"/"sampler_code"` split-decoding naming and the custom `_top_p_top_k_mask_logits` |
| `bench/pless_variants_vllm.py` (194), `bench/topnsigma_vllm.py` (85) | variants → `variants.py` (single logit-space definition); nsigma excluded (D4) |
| `p_less_variants/*.py` (244, prob-space) | **do not copy**; use as test oracle against the logit-space versions |
| `bench/sampler_bridge.py` (242) | reference-adapter idea only; rewrite without `sys.path` hacks |
| `bench/generator.py` (824, HF loop + import monkeypatches) | **do not copy**; write a clean minimal HF reference loop |
| `bench/loop_detect.py`, `seeding.py`, `checkpointing.py`, `eval/think_markers.py` | copy + generalise (locked append, per-model markers) |
| `bench/eval/metrics.py` | **do not copy whole**: re-implement unbiased pass@k, cover@t, termination/token metrics without `human_eval`, executor, zss |
| `scripts/ministral_lcb_leaderboard.py` bootstrap/rank/paired-delta | extract the generic bootstrap pieces; drop LCB platform pooling |
| `bench/gsm8k/`, `bench/squad/` | reference for adapters/evaluators only (HF-only runners, import coding metrics) — rewrite |
| tests: sampler parity, loop, think-end-id, metrics, headline table | port the sampler/plumbing/metrics tests; rewrite imports |
| everything coding (apps, LCB, MBPP, HumanEval, executors, codebleu, algosim, ~80 run_*.sh, mlx pools, GPU2/MSI launchers, paper/, docs/) | **stays here** |

## 5. "Authors' p-less" correctness — parity gates (must pass before any measurement)

1. **Unit parity (CPU):** for random, peaked and real-model logits, our logit-space `pless`/`pless_norm` surviving set == authors' `probs < p` set, exactly; boundary/tie cases; float16/bf16/float32 inputs; all-pruned fallback is the only intended difference (assert it never triggers on exact-arithmetic inputs).
2. **Vocab width for p-less-norm:** record the width vLLM passes to the processor for each model and test it against tokenizer length / `config.vocab_size` / padded size; decide and document which `v` the authors' formula implies (**open question, [unverified]**).
3. **Distribution parity:** empirical token frequencies vs the authors' sampler over ≥20k draws (chi-square / max abs diff), at T=1.0 and one T≠1.
4. **End-to-end:** small run (e.g. 200 GSM8K prompts) reference-HF vs vLLM at T=1.0: pass@1 and token-length distributions within sampling noise (tolerances fixed before running).
5. **vLLM PR #45288 cross-check (optional):** its `PLessLogitsProcessor.apply` math vs ours on the same logits; documents that p-less-norm has no upstream implementation.
6. **Variants:** logit-space vs the prob-space oracle for every variant (this test does not exist today).

## 6. Measurement design (from the research-reasoner review, my edits)

- **Performance:** pass@1 = mean of per-task c/n; unbiased pass@k (open-answer sets only; ≈1 by chance for 4-way MC); maj@10 (+ maj@k by subset averaging); unparseable = abstain in maj, wrong in pass; truncated = wrong, with trunc% and unparse% reported.
- **Efficiency:** native token counts and finish_reason **saved per sample at generation time**; tokens/sample (mean, median, p90); tokens per correct sample; accuracy-vs-budget as a lower bound; termination rate via per-model think-end marker; loop rate with a validated detector (hand-label ~100+100 samples, report precision/recall); wall-clock only as a caveat.
- **Statistics:** paired task-level bootstrap, seed 42, shared resample matrix, 2000 resamples (5000 for headline contrasts); pre-registered primary contrasts with Holm correction; k fixed from prior APPS results, not re-picked on new data. The ~0.65 pp paired SE is from code tasks at 420 tasks and must be re-estimated on a pilot; small sets (160 / 198 tasks) imply minimum detectable differences of about 2.6–2.9 pp **[guess]**.
- **Confounds controlled:** same cap/T/prompt/stops for all arms; thinking on and recorded; one backend+version per comparison (0.21.0, 0.33.0, mlx never mixed); fixed per-task seeds; tokenizer-dependent counts avoided by native counts; re-score a subset with alternative answer extraction.
- **Red-team checks built into reports:** tokens at matched accuracy (Pareto), tokens among terminated-and-correct only, efficiency gain recomputed after dropping looped/capped samples, baseline strength + ceiling (pass@10) shown beside every result.

## 7. Phases and acceptance criteria (sizes are guesses, S/M/L)

| Phase | Deliverable | Acceptance |
|---|---|---|
| P0 (S) | decisions D1–D8; repo created (private) | your sign-off |
| P1 (S) | skeleton, uv lock, CI (CPU tests), pinned submodule, PROVENANCE.md | `uv sync` clean; CI green; submodule at `a681f23` |
| P2 (M) | `samplers/` canonical variants + reference adapter | all CPU parity gates (§5.1, 5.3, 5.6) green; TDD log |
| P3 (M) | vLLM backend + engine loader + manifest + locked append | smoke on one GPU: gates §5.2, 5.4 pass; ninja/PATH and V1-runner env asserted in code, not shell; token counts present in every record |
| P4 (S–M) | metrics/stats | numbers reproduce the known pooled values from this repo's saved LCB per-task metrics (a regression test using a tiny fixture) |
| P5 (M) | data adapters + grading: GSM8K first, then math pool, then GPQA Diamond (hash-only logging) | grader fixtures pass; extraction-bias sensitivity check; GPQA loads from the official gated copy |
| P6 (S) | pilot: one model, few configs, small n | per-task SD, tokens/sample, time → power analysis; **you set the GPU budget** |
| P7 (L) | pre-registered experiments | independent validation (results-validator) of each result file before any claim |

Every code phase: tests first (red → green), research-coder writes, code-reviewer reviews, main session verifies hand-backs; results validated before they are reported.

## 8. Risks and open items

- **Authors' repo may change** (README promises inference-API contributions); the pin protects us, and PR #45288 could eventually merge — keep the backend swappable.
- **Version drift:** vLLM 0.21.0 vs 0.33.0 (this repo has an `envs/vllm-033` pyproject) are not comparable; the manifest records versions and the runner refuses mixed comparisons.
- **Custom logits processor depends on the V1 interface** (`VLLM_USE_V2_MODEL_RUNNER=0`) and FlashInfer behaviour; a vLLM upgrade can break it silently — covered by the §5.4 smoke test in CI-on-GPU (manual).
- **Gated data (GPQA):** official terms forbid revealing examples; the mirror `fingertap/GPQA-Diamond` was verified content-identical (198/198, different order); use official copy, log hashes only.
- **Remote machines:** GPU2/MSI changes need your `!` commands (classifier blocks remote state changes).
- **CLAUDE.md here is wrong** about the origin paper: it does report k≠2 experiments (DeepSeek-R1-Distill-Qwen-7B, Appendix C.7/Table 9). Fix before the new repo's README cites "first".
- **Unverified:** the vocab-width question (§5.2); whether any additional upstream integration (SGLang etc.) exists — the searches I ran found none for vLLM beyond PR #45288 and showed nothing relevant for SGLang, but I read only partial result pages.
