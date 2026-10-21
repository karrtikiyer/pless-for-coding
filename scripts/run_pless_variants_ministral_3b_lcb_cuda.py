"""CUDA/vLLM runner: pless variants vs baselines on LiveCodeBench v6, Ministral-3-3B.

Sibling of ``scripts/run_pless_variants_qwen3_8b_lcb_cuda.py`` (same generation
machinery, same task-selection/output-file layout) — copied rather than
parameterized so the Qwen3 script carries zero regression risk from this
model's addition. Reuses the same imports:
  * ``bench.livecodebench.dataset.load_lcb`` for problems
  * ``bench.livecodebench.prompts.format_prompt_lcb_instruct`` for prompts
  * ``bench.generator_vllm.{load_engine,generate_samples_vllm,encode_prompt_for_vllm}``
  * ``bench.pless_variants_vllm.register()`` + ``bench.topnsigma_vllm.register()``

Differences from the Qwen3 script:
  * ``--model`` defaults to ``mistralai/Ministral-3-3B-Reasoning-2512``
    (``Mistral3ForConditionalGeneration``, 3.4B LM + 0.4B vision encoder,
    reasoning tags ``[THINK]``/``[/THINK]`` rather than Qwen3/DeepSeek's
    ``<think>``/``</think>``).
  * New ``--think-end-marker`` flag (default ``[/THINK]``), threaded through
    ``generate_samples_vllm(think_end_marker=...)`` and
    ``_strip_think_content(s, marker=...)``. Ministral's marker tokenizes to
    more than one id, so ``resolve_think_end_id`` resolves to ``None`` for it
    — harmless for every sampler here since none of these configs request
    genuine split/loop-force decoding (``sampler_think == sampler_code``
    always, ``loop_ngram_n``/``loop_ngram_k`` never passed) — see
    ``bench/generator_vllm.py``'s ``generate_samples_vllm`` docstring.
  * ``qwen_recipe`` (Qwen3's T=0.6/top_p=0.95/top_k=20 recommended recipe) is
    replaced with ``ministral_recipe`` (Mistral's own documented recipe for
    this model: T=0.7, top_p=0.95, no top_k truncation — cross-verified
    against the model card's vLLM usage example).
  * ``--max-model-len`` default is still 27000, same as the Qwen3 script —
    verified via a real load test on a 16GB card (weights 7.26 GiB + 5.07 GiB
    KV cache pool covering 51,056 tokens, comfortably above 27000).
  * Passes a custom ``system_prompt`` to ``format_prompt_lcb_instruct``
    (additive param — default ``None`` there preserves the Qwen3 script's
    ``_SYSTEM`` unchanged). Discovered via smoke test: Ministral-3 Reasoning
    models trigger ``[THINK]``/``[/THINK]`` via specific SYSTEM PROMPT
    CONTENT (Mistral publishes it as ``SYSTEM_PROMPT.txt`` on the model's HF
    repo), not a chat-template kwarg — ``enable_thinking=True`` is a Qwen3
    template variable that Mistral's own chat template doesn't consume at
    all, so reusing the Qwen3 script's coding-task ``_SYSTEM`` message
    verbatim silently produced zero reasoning output in the first smoke test
    (confirmed empirically: generation worked, but no [THINK] tags appeared).
    ``MINISTRAL_SYSTEM_PROMPT`` below combines Mistral's official
    reasoning-trigger text (quoted verbatim from the model repo) with our own
    coding-task framing.
  * Passes ``skip_special_tokens=False`` to ``generate_samples_vllm``.
    Second smoke-test finding: even with the correct system prompt, decoded
    output showed genuine chain-of-thought prose but never the literal
    ``[THINK]``/``[/THINK]`` markers — traced to those being registered
    ``special=True`` tokens (ids 34/35) that vLLM's default
    ``skip_special_tokens=True`` silently drops from the decoded text.
    Qwen3's ``<think>``/``</think>`` are registered ``special=False``
    (verified directly against both tokenizers), so this flag is a no-op for
    the Qwen3 script, which doesn't pass it (defaults to ``True`` there too).

**LCB eval constraint** (see ``bench/eval/__main__.py``'s livecodebench branch):
a results JSONL must be single-platform — ``records[0]["source"]`` picks which
test map loads. Since our stratified task set spans AtCoder + LeetCode, this
runner writes ONE FILE PER (config, platform) pair, not one file per config.

22 configs, mirroring the Qwen3 run's exact config list (11 base incl.
pless_min2 + 4 temp-before-pless sweep + 7-point top-p sweep), with
``qwen_recipe`` swapped for ``ministral_recipe`` — the only intended
difference from Qwen3's list:
  top_nsigma (n=1.0), top_nsigma_1.5, G_k=0.1, G_k=0.35, ministral_recipe
  (T=0.7/top_p=0.95), pless_default (canonical, "pless" sampler), pless_effk,
  pless_min2, pless_topmass, pless_gini, pless_top1half,
  pless_default_t1_5/t2_0/t2_5/t3_0, top_p_1_0/0_95/0_9/0_85/0_8/0_75/0_7.

Example::

    uv run --project pyproject-vllm.toml python scripts/run_pless_variants_ministral_3b_lcb_cuda.py \\
        --model mistralai/Ministral-3-3B-Reasoning-2512 --dtype bfloat16 \\
        --n-samples 10 --max-new-tokens 25000 --seed 42 \\
        --configs pless_gini top_nsigma
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Order matters: register() mutates the vLLM dispatch table before we import
# generate_samples_vllm's consumers below.
from bench.pless_variants_vllm import register as _register_variants
_register_variants()
from bench.topnsigma_vllm import register as _register_topnsigma
_register_topnsigma()

from bench.generator import _strip_think_content
from bench.generator_vllm import (
    encode_prompt_for_vllm,
    generate_samples_vllm,
    load_engine,
)
from bench.livecodebench.dataset import load_lcb
from bench.livecodebench.prompts import format_prompt_lcb_instruct

# Mistral's official reasoning-trigger system prompt (quoted verbatim from
# https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512/raw/main/SYSTEM_PROMPT.txt,
# fetched directly — a WebFetch summarization of this same file got visibly
# confused by its embedded [THINK]/[/THINK] example and returned garbled text,
# so this was re-verified via a raw curl fetch), combined with our own
# coding-task framing. Reasoning mode is triggered by this exact content, not
# by a chat-template kwarg — see the module docstring above.
_MINISTRAL_REASONING_INSTRUCTIONS = (
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at "
    "a response. Format your response using Markdown, and use LaTeX for any "
    "mathematical equations. Write both your thoughts and the response in "
    "the same language as the input.\n\n"
    "Your thinking process must follow the template below:[THINK]Your "
    "thoughts or/and draft, like working through an exercise on scratch "
    "paper. Be as casual and as long as you want until you are confident to "
    "generate the response to the user.[/THINK]Here, provide a "
    "self-contained response."
)
MINISTRAL_SYSTEM_PROMPT = (
    _MINISTRAL_REASONING_INSTRUCTIONS + "\n\n"
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

REPO = Path(__file__).resolve().parent.parent
TASK_SELECTION_CSV = REPO / "results/lcb_v6_stratified60/task_selection.csv"
OUT_ROOT_TEMPLATE = "results/pless_variants_cuda/{model_slug}/LCB_v6_stratified60"
BATCH_ASSIGNMENT_CSV = REPO / "results/lcb_v6_full1046/batch_assignment.csv"


def _load_batch_assignment() -> dict[str, int]:
    """{task_id -> batch_num} for all 1046 full-dataset tasks (see
    scripts/generate_lcb_full1046_batches.py). Includes the 60 stratified
    tasks too -- they're regenerated at this run's raised token budget
    (35000/37500 vs the stratified-60 run's 25000/27000), not reused, so
    the full 1046-task result is generated under one consistent condition
    (2026-10-09: reusing the old-budget 60 was considered and rejected --
    would silently mix two different generation conditions in one result).
    """
    with BATCH_ASSIGNMENT_CSV.open() as f:
        return {row["task_id"]: int(row["batch"]) for row in csv.DictReader(f)}

# (name, sampler_name, temperature, renyi_k_or_None)
CONFIGS: list[tuple[str, str, float, float | None]] = [
    ("top_nsigma",       "top_nsigma",       1.0, None),
    ("top_nsigma_1_5",   "top_nsigma_1_5",   1.0, None),
    ("gk_0_1",           "pless_renyi",      1.0, 0.1),
    ("gk_0_35",          "pless_renyi",      1.0, 0.35),
    # Ministral-only extra G_k points (not run for Qwen3-8B -- requested
    # 2026-10-05 as a Ministral-specific follow-up, breaks the usual
    # Qwen/Ministral config-list parity intentionally).
    ("gk_0_025",         "pless_renyi",      1.0, 0.025),
    ("gk_0_0325",        "pless_renyi",      1.0, 0.0325),
    ("gk_0_05",          "pless_renyi",      1.0, 0.05),
    ("gk_0_075",         "pless_renyi",      1.0, 0.075),
    ("gk_0_2",           "pless_renyi",      1.0, 0.2),
    ("gk_0_4",           "pless_renyi",      1.0, 0.4),
    ("gk_0_8",           "pless_renyi",      1.0, 0.8),
    ("gk_1_6",           "pless_renyi",      1.0, 1.6),
    ("ministral_recipe", "ministral_recipe", 0.7, None),
    ("pless_default",    "pless",            1.0, None),
    ("pless_effk",       "pless_effk",       1.0, None),
    # Isolates effk's rank-cut-vs-threshold selection rule from its min-2
    # floor (the missing 4th cell of the {threshold,rank-cut}x{floor,no-floor}
    # factorial -- see bench/pless_variants_vllm.py:pless_effk_nofloor_mask_logits
    # and the 2026-10-06 methodology review that found effk's floor-active set
    # is a strict SUBSET of min2's, not a superset).
    ("pless_effk_nofloor", "pless_effk_nofloor", 1.0, None),
    ("pless_min2",       "pless_min2",       1.0, None),
    ("pless_topmass",    "pless_topmass",    1.0, None),
    ("pless_topmass_nofloor", "pless_topmass_nofloor", 1.0, None),
    ("pless_gini",       "pless_gini",       1.0, None),
    ("pless_gini_nofloor", "pless_gini_nofloor", 1.0, None),
    ("pless_top1half",   "pless_top1half",   1.0, None),
    # T1-only sweep: canonical pless with the pre-truncation temperature raised
    # above 1.0 to flatten the (often very peaked, RL-sharpened) distribution
    # before the collision-entropy threshold is applied. Same "pless"
    # sampler_name as pless_default; temperature is what varies.
    ("pless_default_t1_5", "pless",          1.5, None),
    ("pless_default_t2_0", "pless",          2.0, None),
    ("pless_default_t2_5", "pless",          2.5, None),
    ("pless_default_t3_0", "pless",          3.0, None),
    # Standard nucleus (top-p only, top-k disabled, T=1.0) sweep — same
    # comparison-baseline mechanism used for the APPS-252 G_k benchmark and
    # mirrored verbatim from the Qwen3-8B LCB script's config list (temp=1.0,
    # top_k=0, top_p in {1.0,0.95,0.9,0.85,0.8,0.75,0.7}). p=1.0 reuses
    # "temp_pure" (identical output, no redundant sampler entry needed).
    ("top_p_1_0",  "temp_pure",   1.0, None),
    ("top_p_0_95", "top_p_0_95",  1.0, None),
    ("top_p_0_9",  "top_p_0_9",   1.0, None),
    ("top_p_0_85", "top_p_0_85",  1.0, None),
    ("top_p_0_8",  "top_p_0_8",   1.0, None),
    ("top_p_0_75", "top_p_0_75",  1.0, None),
    ("top_p_0_7",  "top_p_0_7",   1.0, None),
]
CONFIG_NAMES = [c[0] for c in CONFIGS]


def _load_task_selection() -> dict[str, list[dict]]:
    """Return {platform: [ {task_id, difficulty, contest_date}, ... ]}."""
    by_platform: dict[str, list[dict]] = {}
    with TASK_SELECTION_CSV.open() as f:
        for row in csv.DictReader(f):
            by_platform.setdefault(row["platform"], []).append(row)
    return by_platform


def _output_path(out_root: Path, config_name: str, platform: str,
                  temperature: float) -> Path:
    return out_root / f"{config_name}_{platform}_think_t{temperature}.jsonl"


def _run_config_on_platform(
    *, config_name: str, sampler_name: str, temperature: float,
    renyi_k: float | None, platform: str, problems: list, engine, tokenizer,
    n_samples: int, max_new_tokens: int, out_root: Path, model_id: str,
    seed: int | None, per_config_deadline: float, global_deadline: float,
    think_end_marker: str,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(out_root, config_name, platform, temperature)

    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(str(json.loads(line)["task_id"]))
            except Exception:
                pass
        if done_ids:
            print(f"[{config_name}/{platform}] resume: {len(done_ids)} done")
    print(f"[{config_name}/{platform}] writing -> {out_path}")

    for idx, problem in enumerate(problems, 1):
        if problem.task_id in done_ids:
            continue
        now = time.time()
        if now > per_config_deadline or now > global_deadline:
            print(f"[{config_name}/{platform}] budget exhausted after "
                  f"{idx-1}/{len(problems)} tasks")
            break
        t0 = time.time()
        prompt_text, code_prefix = format_prompt_lcb_instruct(
            problem, tokenizer, enable_thinking=True,
            system_prompt=MINISTRAL_SYSTEM_PROMPT,
        )
        prompt_text = encode_prompt_for_vllm(
            prompt_text, getattr(engine, "_safe_tokenizer", None)
        )
        kwargs = {}
        if renyi_k is not None:
            kwargs["renyi_k"] = renyi_k
        try:
            raw_samples = generate_samples_vllm(
                engine=engine, tokenizer=tokenizer, prompt_text=prompt_text,
                sampler_name=sampler_name, n_samples=n_samples,
                max_new_tokens=max_new_tokens, temperature=temperature,
                stop_strings=None, seed=seed, think_end_marker=think_end_marker,
                skip_special_tokens=False, **kwargs,
            )
        except Exception as exc:
            print(f"[{config_name}/{platform}] task {problem.task_id} crashed: {exc!r}")
            continue

        elapsed = time.time() - t0
        samples_with_think = [code_prefix + s for s in raw_samples]
        samples = [_strip_think_content(s, marker=think_end_marker) for s in samples_with_think]

        record = {
            "model": model_id, "backend": "vllm", "method": config_name,
            "sampler_name": sampler_name, "renyi_k": renyi_k,
            "temperature": temperature, "top_p": 1.0, "top_k": 0,
            "task_id": problem.task_id, "source": problem.platform,
            "difficulty": problem.difficulty, "contest_date": problem.contest_date,
            "prompt_text": problem.question,
            "samples": samples, "samples_with_thinking": samples_with_think,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enable_thinking": True,
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{config_name}/{platform}] task {problem.task_id}: {elapsed:.0f}s")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mistralai/Ministral-3-3B-Reasoning-2512")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization flag: 'awq', 'gptq', 'fp8' etc. "
                         "Required if model+KV won't fit unquantized.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=27000)
    ap.add_argument("--think-end-marker", default="[/THINK]",
                    help="Model's reasoning-close tag, threaded through "
                         "generate_samples_vllm + _strip_think_content. "
                         "Default matches Ministral-3's [THINK]/[/THINK] "
                         "convention (Qwen3/DeepSeek use </think>).")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=25000)
    ap.add_argument("--configs", nargs="+", default=CONFIG_NAMES, choices=CONFIG_NAMES)
    ap.add_argument("--platforms", nargs="+", default=["atcoder", "leetcode"],
                    choices=["atcoder", "leetcode"])
    ap.add_argument("--only-task-ids", nargs="+", default=None)
    ap.add_argument("--budget-minutes", type=float, default=3000.0)
    ap.add_argument("--per-config-minutes", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--full-dataset", action="store_true",
                    help="Run every AtCoder+LeetCode problem in LCB v6 (1046 "
                         "tasks) instead of the 60-task stratified subset. "
                         "Writes to a separate LCB_v6_full1046 output dir "
                         "(the existing 60-task data is seeded into it "
                         "separately, not regenerated -- see "
                         "scripts/generate_lcb_full1046_batches.py).")
    ap.add_argument("--batch", type=int, default=None,
                    help="Only with --full-dataset: restrict to one "
                         "stratified batch (0-indexed) from "
                         "results/lcb_v6_full1046/batch_assignment.csv, "
                         "instead of all 1046 tasks at once. Lets batches "
                         "run independently on whichever GPU is free, and "
                         "gives periodic results instead of waiting for "
                         "the full 1046 to complete.")
    args = ap.parse_args()
    if args.batch is not None and not args.full_dataset:
        # batch_assignment.csv is keyed on the full-1046 task set; silently
        # ignoring --batch here (as an earlier version of this script did)
        # would run the entire 60-task stratified set instead of erroring --
        # caught by review 2026-10-09.
        ap.error("--batch requires --full-dataset")

    global_start = time.time()
    global_deadline = global_start + args.budget_minutes * 60.0

    selection = _load_task_selection()
    model_slug = args.model.replace("/", "--")
    out_root_template = (
        "results/pless_variants_cuda/{model_slug}/LCB_v6_full1046"
        if args.full_dataset else OUT_ROOT_TEMPLATE
    )
    out_root = REPO / out_root_template.format(model_slug=model_slug)
    batch_assignment = _load_batch_assignment() if args.batch is not None else None

    load_kwargs = {"dtype": args.dtype,
                   "gpu_memory_utilization": args.gpu_memory_utilization,
                   "max_model_len": args.max_model_len}
    if args.quantization:
        load_kwargs["quantization"] = args.quantization
    print(f"Loading engine: {args.model} (kwargs={load_kwargs})")
    t0 = time.time()
    engine = load_engine(args.model, **load_kwargs)
    print(f"Engine ready in {time.time()-t0:.0f}s")
    tokenizer = getattr(engine, "_safe_tokenizer", None) or engine.get_tokenizer()
    # Third smoke-test finding: Mistral's own tokenizer library
    # (MistralCommonBackend) warns that apply_chat_template(tokenize=False)
    # is "unsafe" — confirmed empirically, the string round-trip through
    # format_prompt_lcb_instruct's default (tokenize=False) + encode_prompt_
    # for_vllm's re-encode silently corrupted the prompt enough that the
    # model never produced [THINK] at all. The tokenize=True path (direct
    # token ids, no re-encode) fixes it — reusing the existing
    # _qwen_direct_tokenize flag mechanism in format_prompt_lcb_instruct
    # (already generic despite the Qwen-specific name) rather than adding a
    # new branch, so bench/livecodebench/prompts.py needs no further changes.
    tokenizer._qwen_direct_tokenize = True

    config_by_name = {c[0]: c for c in CONFIGS}
    written = []
    for platform in args.platforms:
        if args.full_dataset:
            problems = list(load_lcb(platforms=(platform,), with_tests=False))
            if batch_assignment is not None:
                problems = [p for p in problems
                            if batch_assignment.get(p.task_id) == args.batch]
            if args.only_task_ids is not None:
                wanted = set(args.only_task_ids)
                problems = [p for p in problems if p.task_id in wanted]
            problems.sort(key=lambda p: p.task_id)
            print(f"Platform {platform}: full dataset"
                  + (f" batch {args.batch}" if batch_assignment is not None else "")
                  + f", {len(problems)} problems")
        else:
            rows = selection.get(platform, [])
            wanted_ids = {r["task_id"] for r in rows}
            if args.only_task_ids is not None:
                wanted_ids &= set(args.only_task_ids)
            problems = list(load_lcb(
                platforms=(platform,), task_ids=wanted_ids, with_tests=False,
            ))
            problems.sort(key=lambda p: p.task_id)
            print(f"Platform {platform}: matched {len(problems)}/{len(wanted_ids)} problems")

        for name in args.configs:
            _, sampler_name, temperature, renyi_k = config_by_name[name]
            per_config_deadline = min(
                time.time() + args.per_config_minutes * 60.0, global_deadline
            )
            p = _run_config_on_platform(
                config_name=name, sampler_name=sampler_name,
                temperature=temperature, renyi_k=renyi_k, platform=platform,
                problems=problems, engine=engine, tokenizer=tokenizer,
                n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                out_root=out_root, model_id=args.model, seed=args.seed,
                per_config_deadline=per_config_deadline,
                global_deadline=global_deadline,
                think_end_marker=args.think_end_marker,
            )
            written.append(p)

    total = time.time() - global_start
    print(f"\nDone in {total/60:.1f} min. Wrote:")
    for p in written:
        n = sum(1 for _ in p.open()) if p.exists() else 0
        print(f"  {p}   ({n} tasks)")


if __name__ == "__main__":
    main()
