# HumanEval regeneration plan (2026-11)

Status: **PLAN — not started.** Written 2026-11-06. Nothing runs until the user approves this plan and the
GPU time budget (standing rule: no self-picked budgets). Tracked as TODO **D13** in `docs/theory/todos.md`.

## 1. Why

The paper's HumanEval table (Table II, `table_3b_humaneval`) cannot be trusted as it stands (D13):

- It mixes two pipelines. OLD rows (`full_precision_results/*_detailed.json`: greedy, T@0.2, T@0.7, top-p@0.95)
  were scored as prompt + body + test; the script that generated them is lost and its metadata names models
  only by alias (`codellama-7b`, `qwen25-7b`). NEW rows (`bench.eval`: p-less, p-less-norm, top-p@0.9) were
  scored as sample + test with **no task prompt**, which fails correct answers that rely on the prompt's
  imports. Example: CodeLlama-7b-Instruct, HumanEval/0 — the reply uses `List` without re-importing it →
  `NameError` under the NEW grader, pass under the official composition. Prepending the prompt flipped 119/1640
  CodeLlama-Instruct samples fail→pass and 0 pass→fail.
- The paper's "p-less is 7.87 pp below greedy on CodeLlama-Instruct" is mostly this grading artifact
  (p-less 28.11% → 35.61% with the prompt, vs greedy 35.98%).
- Base-model HumanEval rows dated 2026-03-02 (CodeLlama-7b-hf; Codestral in `temprature_results`) predate commit
  `d872d45` (2026-03-04), which fixed a decode bug that dropped one leading space (3-space indentation:
  1460/1640 CodeLlama-7b-hf T@0.7 samples).
- HumanEval covers 3 of the paper's 8 models; the instruct rows use p-less at T=0.6 while the rest of the paper uses T=0.7.

Everything is regenerated from scratch with one generator, one prompt specification and one grader, into a new folder.

## 2. Decisions (locked 2026-11-06, by the user)

| Item | Decision |
|---|---|
| Models | the paper's eight (§3) |
| Generation backend | vLLM **0.33.0** (latest on PyPI, 2026-11-02) for all configs except beam |
| Beam | **HF** `generate(num_beams=4)` — identical to MBPP's beam; vLLM's beam search has no stop strings |
| OpenCodeInterpreter-DS-1.3B | completion (raw) prompt, consistent with MBPP; labelled "instruction-tuned, completion prompt" |
| Stop sequences (base models) | BigCode harness HumanEval list, raw (unstripped) prompt — the harness's `humaneval-unstripped` setup |
| Grader timeout | 3.0 s (official `human_eval` and BigCode harness default) |
| Results folder | new: `results/humaneval_regen_2026_11/` |
| Machine / order | MSI laptop (RTX 4090 Laptop GPU); small models first, 7B models at the end |

## 3. Models

| # | HF model id | Prompt | bf16 weights | Run order |
|---|---|---|---|---|
| 1 | `m-a-p/OpenCodeInterpreter-DS-1.3B` | raw (base-style) | 2.7 GB | 1 |
| 2 | `Qwen/Qwen2.5-Coder-1.5B` | raw | 3.1 GB | 2 |
| 3 | `Qwen/Qwen2.5-Coder-3B` | raw | 6.2 GB | 3 |
| 4 | `codellama/CodeLlama-7b-hf` | raw | 13.5 GB | 4 |
| 5 | `codellama/CodeLlama-7b-Instruct-hf` | chat template | 13.5 GB | 5 |
| 6 | `meta-llama/Llama-2-7b-hf` | raw | ~13.5 GB (gated; same 6.7B architecture — not queried) | 6 |
| 7 | `meta-llama/Llama-2-7b-chat-hf` | chat template | ~13.5 GB (gated) | 7 |
| 8 | `Qwen/Qwen2.5-Coder-7B-Instruct` | chat template | 15.2 GB | 8 |

Sizes: parameter totals from the Hugging Face API (`safetensors.total` × 2 bytes). Base/instruct status for the
two small Qwen models checked on their model cards (both "base … pretraining"); OpenCodeInterpreter's card
describes an instruction/chat fine-tune of `deepseek-coder-1.3b-base`.

**Memory risk.** The RTX 4090 *Laptop* GPU has 16 GB (TechPowerUp). Models 1–3 fit easily. Models 4–7 leave
roughly 1 GiB for the KV cache at vLLM's default `gpu_memory_utilization=0.90` (estimate); model 8's weights
(15.2 GB) exceed that budget. A fit test runs before the 7B batch (§8, step 4); models that do not fit are
decided then (another GPU, or dropped) — not quantized, since that would make them different models from the MBPP rows.

## 4. Configs (per model)

| Config | Samples/task | Backend | Implementation |
|---|---|---|---|
| greedy | 1 | vLLM | native `SamplingParams(temperature=0)` (requires `n=1`) |
| beam@4 | 1 | HF | `bench/generator.py:generate_samples_beam`, best sequence only |
| T@0.7 | 10 | vLLM | native `SamplingParams(temperature=0.7)` |
| top-p@0.9 with T=1.0 | 10 | vLLM | native `SamplingParams(top_p=0.9, temperature=1.0)` (CLAUDE.md gotcha 8: never a custom top-p) |
| p-less with T=0.7 | 10 | vLLM | `PlessSplitLogitsProcessor` (temperature applied inside the processor, before the Σp² cutoff) |
| p-less-norm with T=0.7 | 10 | vLLM | same processor, `pless_norm` |
| top-p@0.95 with T=0.2 | 10 | vLLM | native; **models 1–3 only** (Hasan et al.'s config) |

Total: 51 runs — 35 sampled (35 × 164 × 10 = 57,400 generations) and 16 deterministic (16 × 164 = 2,624).
For n=1 configs, pass@3/5/10 copy pass@1 (`bench/eval/metrics.py:compute_pass_at_k`), as for MBPP.

## 5. Generator specification

- vLLM 0.33.0, exact pin, in a separate `uv` environment (new project file, e.g. `pyproject-vllm-033.toml`;
  0.33.0's build file pins `torch==2.14.0` — runtime requirement to confirm when building).
- `VLLM_USE_V2_MODEL_RUNNER=0`: our p-less processor uses the V1 logits-processor interface, which the V2 runner
  rejects (vLLM PR #56497; docs "custom_logitsprocs"). `VLLM_USE_FLASHINFER_SAMPLER=0` as in previous runs.
- `dtype=bfloat16`, `max_new_tokens=512` (new tokens, not total length).
  - Grounding (2026-11-06, existing HumanEval runs, re-tokenized): chat replies have median 61–72 tokens and
    0–1.04% reach 512; base completions, once cut at the BigCode stop list, have median 38–61 and p95 ≤ 310 tokens.
    The base samples that still reach 512 are mostly repetition loops (e.g. 63 of 81 long CodeLlama-7b-hf p-less
    completions repeat a line ≥5 times), which a larger cap would not fix.
  - The old base runs (2026-02-27..03-05) mostly ran to 512 because their stop sequences never fired (completions
    contain `\ndef check(`, `\nif __name__`, …); commit `b256b0c` (2026-03-27) fixed stop firing. The smoke test checks it.
  - Hasan et al.'s "512" is the BigCode harness `--max_length_generation`, whose help text says
    prompt + generation — a smaller new-token budget than ours; one more reason their numbers are approximate.
- `max_model_len` = longest prompt (over all 8 tokenizers, chat template included) + 512, rounded up to a
  multiple of 64, computed at setup; the runner asserts every prompt fits. Measured so far: longest chat prompt
  484 tokens (CodeLlama tokenizer), 432 (Qwen2.5) → about 1024. Smaller is better for fitting the 7B models.
- Seeds and reproducibility (vLLM docs, "Reproducibility"): vLLM is not reproducible by default; offline
  reproducibility needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` (deterministic scheduling) and holds only on the same
  hardware and vLLM version. Batch-invariant mode needs compute capability ≥ 9.0; the MSI GPU is 8.9 (nvidia-smi),
  so it is unavailable. Settings: engine `seed=42` (the V1 default is 0), per-request `SamplingParams(seed=42)`,
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`, one machine and one vLLM version for every run. HF beam is deterministic
  (no sampling); `torch.manual_seed(42)` is set anyway. All recorded in `MANIFEST.json`.
- Prompt encoding: built **once, as token ids**, and given to both backends (vLLM `TokensPrompt`; HF accepts
  `list[int]`). Base: `tokenizer(raw_prompt)` (one BOS added). Chat: `tokenizer.apply_chat_template(messages,
  tokenize=True, add_generation_prompt=True)`. Reason: the chat-template *string* already starts with `<s>`;
  encoding it again added a second BOS (CodeLlama-7b-Instruct tokenizer, transformers 5.3.0: ids `[1, 1, …]`).
- Text: base completions decoded from the full sequence with the prompt removed (the `d872d45` fix);
  stored sample = raw prompt + completion. Chat: the reply text.
- Stop sequences, base models only: `["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]`
  (BigCode harness `tasks/humaneval.py`); also applied as post-truncation. Chat models: none.
- Beam runs as a separate step per model (one model in GPU memory at a time).
- Each record stores: `task_id`, `model`, `config`, `backend`, `task_prompt` (raw), `prompt_text`,
  first prompt token ids, `samples`, `test`, `entry_point`, `seed`, `n`, `max_new_tokens`, `stop`,
  and versions (vllm, transformers, torch, CUDA, GPU name, git commit).

## 6. Prompt specification

- Dataset: `openai/openai_humaneval`, test split, 164 tasks — identical to the official `HumanEval.jsonl.gz`
  in prompt, test and entry point for all 164 tasks (checked 2026-11-06).
- Base models (1, 2, 3, 4, 6): the raw HumanEval prompt (imports + signature + docstring), zero-shot.
- Chat models (5, 7, 8): the model's own chat template with
  - system: `You are a helpful coding assistant. Write clean, correct Python code.`
  - user: `Complete the following Python function. Provide only the function implementation, no explanations.` + blank line + raw prompt
  (unchanged from `bench/humaneval/prompts.py`; this wording is ours, not a standard — state it in the paper).

## 7. Grader specification (official composition)

Program = prompt + completion + `"\n"` + test + `"\n"` + `check(entry_point)` (openai/human-eval `execution.py`).

| Sample | Program |
|---|---|
| base | sample (must start with the raw prompt — asserted) + test + check |
| chat reply containing `def <entry_point>` | raw prompt + extracted code + test + check (the reply's function replaces the stub; prompt imports stay) |
| chat reply without it (body fragment; 29/1640 CodeLlama-Instruct replies in the old run) | raw prompt + reply as written + test + check |

- `exit`, `quit` and `SystemExit` count as failures (official `reliability_guard` sets `exit`/`quit` to None and
  treats any exception as failure; ours previously passed on exit code 0).
- Interpreter: the venv's own Python (`sys.executable`), not `python3` on PATH. Run locally, never on the GPU box
  (CLAUDE.md gotcha 9).
- Timeout 3.0 s. Store per-sample pass flag, failure reason and a grader version string.
- MBPP grading is unchanged (regression test).

Validation before any number is used:
1. Canonical solutions pass 164/164.
2. Unit tests: empty body fails; missing `List` import passes with the prompt; the reply's own imports work;
   no double prepend; `sys.exit(0)` fails; a timeout fails; MBPP builder output unchanged.
3. Cross-check ~200 programs against the official `human_eval` package (run via `uv`, outside the project deps).

## 8. Phases and gates

0. **Machine setup (MSI).** `git pull` only (never copy files). `nvidia-smi` (memory, compute capability, any
   running job — E25 Ministral may still hold the GPU). Build the 0.33.0 env with `uv`; HF token with Llama-2
   access. Run the p-less parity tests (`tests/test_vllm_parity.py`) on the machine.
1. **Commit this spec** (done when this file is committed).
2. **Code, test-first** on a new branch: runner (greedy, beam via HF, seed, token-id prompts, record fields,
   new output folder with `MANIFEST.json` + `README.md`, refuse to overwrite unless resuming), grader (§7),
   orchestration script `run_regen_humaneval_8models.sh`. Then `code-reviewer`.
3. **Smoke test, small models.** 5 tasks × models 1–3 × {greedy, beam@4, T@0.7, p-less}, seeded. Pass criteria:
   ≥99% of base bodies indented 4 spaces; exactly one BOS for Llama-family chat prompts; greedy rerun identical;
   seeded rerun identical (vLLM's batch-invariant mode needs compute capability ≥9.0, so this is checked
   empirically); beam returns one sequence cut at a stop sequence; no base completion contains a stop sequence
   (stops fire); share of completions reaching 512 tokens reported (expected low, mostly loops); longest prompt
   fits in `max_model_len`.
   Measure seconds/task per model × config; grade locally; official-package cross-check.
4. **Budget gate.** Propose a time budget from the measurements; the user approves before any full run.
5. **Full runs, small models** (1 → 2 → 3): vLLM configs, then HF beam. Per-task resume; status updates carry IST timestamps.
6. **7B fit test** (engine start + 5 tasks, vLLM and HF beam) for models 4–8, then full runs for those that fit, model 8 last.
7. **Evaluate locally**, `results-validator` before any number is reported.
8. **Write-up.** Rebuild Table II from the new folder only, with paired bootstrap 95% CIs over the 164 tasks
   (5000 resamples). Retract the −7.87 pp claim. Disclose: MBPP ran on HF, HumanEval on vLLM (beam on HF in both);
   Hasan et al.'s HumanEval numbers (Table 4: 0.44 / 0.51 / 0.59; T=0.2, top-p 0.95, n=10, 512 tokens, BigCode
   harness, bf16) are a two-decimal, different-harness comparison. Close D13.

## 9. Pre-registered reporting rule

- Report every config's numbers as produced; no reruns or config changes to move a result.
- Comparisons (p-less vs greedy, vs T@0.7, vs top-p) as paired differences with 95% CIs; a difference whose CI
  spans zero is reported as "no detectable difference", not as a win or a loss.
- If a model cannot run on the available hardware, it is reported as not run, not substituted.

## 10. Out of scope (new TODOs)

- MBPP chat-model prompts (CodeLlama-Instruct, Llama-2-chat) probably had the same double BOS — effect unknown;
  needs one controlled GPU run to measure.
- MBPP "bigcode" stop list (`bench/runner.py:24`) differs from the BigCode harness's MBPP list (`\nif __name__`
  instead of `\nif`; lacks `\n<|/` and `` \n``` ``); code comment claims it matches Hasan et al.
- HumanEval appendix rows (Codestral-22B, Qwen3-Coder-30B, Qwen2.5-Coder-7B base) — decide later.

## Sources (fetched 2026-11-06)

- vLLM 0.33.0 on PyPI; `vllm/sampling_params.py`, `vllm/entrypoints/generate/beam_search/offline.py`,
  `vllm/envs.py` at tag v0.33.0; PR #56497 (Model Runner V2 custom logits processors); docs page
  "custom_logitsprocs".
- openai/human-eval `human_eval/execution.py` (program composition, `reliability_guard`) and `evaluation.py` (timeout 3.0).
- bigcode-evaluation-harness `bigcode_eval/tasks/humaneval.py` and `mbpp.py` (stop words, prompt stripping).
- Chen et al. 2021, arXiv:2107.03374 (Codex stop sequences).
- Hasan et al., arXiv:2507.03160, Table 4 and §2.4.
- TechPowerUp, GeForce RTX 4090 Mobile (16 GB GDDR6).
