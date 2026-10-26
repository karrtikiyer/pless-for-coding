"""CUDA/vLLM runner: pless_effk (both floor variants) + standard-decoding
baselines on LiveCodeBench v6, DeepSeek-R1-Distill-Llama-8B.

Experiment 3 (cross-model generality): pless_effk/pless_effk_nofloor clearly
recovered Ministral-3-3B-Reasoning-2512 from looping (termination% 63.2% ->
71.0%/71.3%, pass@1 56.67% -> 65.67%/66.33%) but only tied standard baselines
on the already-well-calibrated Qwen3-8B. DeepSeek-R1-Distill-Llama-8B is a
known loop-prone model on a DIFFERENT dataset (APPS: pless_default truncates
~65% at matched settings, pass@1 0.392 before Renyi-G_k repair to 0.469 at
k=0.4 -- docs/theory/todos.md A37/A41/A43) -- this is a real out-of-sample
test of "pless-variants help when the base model loops", not just a second
data point for its own sake.

Companion to scripts/run_pless_variants_qwen3_8b_lcb_cuda.py and
scripts/run_pless_variants_qwen25coder_7b_lcb_cuda.py -- same generation
machinery and output-file/report-script conventions. Unlike the
Qwen2.5-Coder script, this model needs NO cot-prefill: DeepSeek-R1-Distill's
own chat template natively appends "<think>\\n" via add_generation_prompt
(confirmed against its cached tokenizer_config.json), so this reuses
format_prompt_lcb_instruct exactly like the Qwen3-8B script does.
``enable_thinking=True`` is forwarded but is a template-level no-op for
DeepSeek (same mechanism/kwarg-path already exercised via
format_prompt_apps_instruct in bench/apps/runner.py for DeepSeek's APPS
runs -- read both, identical extra_kwargs forwarding, only _user_message/
_SYSTEM content differs).

Inherits the already-fixed vLLM whitespace-tokenizer bug (HF #45488,
docs/theory/todos.md A41) for free via bench.generator_vllm.load_engine's
_safe_tokenizer auto-detection + encode_prompt_for_vllm -- no new code
needed; this is the same path the Qwen3-8B/Qwen2.5-Coder scripts already
use, just now actually exercised by a model where the gate fires.

Token budget: max_new_tokens=32768 matches the established DeepSeek literal
used throughout A37/A41/A43. max_model_len=36000 (prompt+completion
COMBINED budget, not completion-only -- vLLM has no separate prompt
allowance) was sized against the real tokenized LCB v6 stratified-60 prompt
lengths for this model's own tokenizer (max 1240 tok, task abc342_e; median
480), leaving >3000 tok of margin over the 32768+1240=34008 worst case.

15 configs (name -> sampler_name/temperature/renyi_k):
  Core mechanism test:
    pless_default        -- canonical "pless" sampler, T=1.0
    pless_effk           -- rank-cut selection rule, min-2 floor, T=1.0
    pless_effk_nofloor   -- rank-cut selection rule, NO floor, T=1.0
  Supplementary (cross-model anchor + vendor baseline):
    gk_0_1    -- Renyi G_k, k=0.1 (shared with Qwen3/Qwen2.5-Coder for
                 cross-model comparability)
    gk_0_4    -- Renyi G_k, k=0.4 (DeepSeek's own empirically-best k per the
                 completed A43 sweep: pass@1 0.392 -> 0.469)
    deepseek_recipe -- temp_standard @ T=0.6 (DeepSeek's own HF-model-card
                 recommended temperature, verified this session)
  Standard-decoding baselines (does pless beat these, or just tie them like
  on Qwen3-8B?) -- same tuples already proven in the Qwen3-8B script:
    top_nsigma, top_nsigma_1_5
    top_p_1_0, top_p_0_95, top_p_0_9, top_p_0_85, top_p_0_8, top_p_0_75,
    top_p_0_7

Deliberately NOT included (not load-bearing for this experiment's question,
per 2026-10-08 plan review): pless_min2, pless_topmass(_nofloor),
pless_gini(_nofloor), pless_top1half, pless_default T1-sweep (t1_5..t3_0).

Example::

    uv run --project pyproject-vllm.toml python scripts/run_pless_variants_deepseek_8b_lcb_cuda.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dtype bfloat16 \\
        --n-samples 10 --max-new-tokens 32768 --seed 42 \\
        --configs pless_default pless_effk
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
    generate_samples_standard_vllm,
    generate_samples_vllm,
    load_engine,
)
from bench.livecodebench.dataset import load_lcb
from bench.livecodebench.prompts import format_prompt_lcb_instruct

REPO = Path(__file__).resolve().parent.parent
TASK_SELECTION_CSV = REPO / "results/lcb_v6_stratified60/task_selection.csv"
OUT_ROOT_TEMPLATE = "results/pless_variants_cuda/{model_slug}/LCB_v6_stratified60"

# (name, sampler_name, temperature, renyi_k_or_None)
CONFIGS: list[tuple[str, str, float, float | None]] = [
    ("pless_default",      "pless",              1.0, None),
    ("pless_effk",         "pless_effk",         1.0, None),
    ("pless_effk_nofloor", "pless_effk_nofloor", 1.0, None),
    ("gk_0_1",              "pless_renyi",        1.0, 0.1),
    ("gk_0_4",              "pless_renyi",        1.0, 0.4),
    ("deepseek_recipe",     "temp_standard",      0.6, None),
    ("top_nsigma",          "top_nsigma",         1.0, None),
    ("top_nsigma_1_5",      "top_nsigma_1_5",     1.0, None),
    ("top_p_1_0",  "temp_pure",   1.0, None),
    ("top_p_0_95", "top_p_0_95",  1.0, None),
    ("top_p_0_9",  "top_p_0_9",   1.0, None),
    ("top_p_0_85", "top_p_0_85",  1.0, None),
    ("top_p_0_8",  "top_p_0_8",   1.0, None),
    ("top_p_0_75", "top_p_0_75",  1.0, None),
    ("top_p_0_7",  "top_p_0_7",   1.0, None),
]
CONFIG_NAMES = [c[0] for c in CONFIGS]

# Sampler names that are genuinely standard (vLLM-native-equivalent) top-p/
# temperature baselines -- per CLAUDE.md Known Gotchas #8, these must NOT go
# through the custom PlessSplitLogitsProcessor (that path's reimplementation,
# _top_p_top_k_mask_logits, silently diverged from the standard nucleus-
# sampling definition for months -- fixed 2026-10-26, but new baselines
# should use vLLM's own native SamplingParams going forward regardless).
# Maps sampler_name -> (top_p, top_k); temperature is already carried
# per-CONFIGS-row. top_nsigma/top_nsigma_1_5/pless* are deliberately NOT
# here: top_nsigma has no vLLM-native equivalent (a genuinely custom
# algorithm, Tang et al. 2025, independently verified correct against the
# paper + authors' own code 2026-10-26), and pless* is this project's own
# method -- both correctly stay on the custom-processor path below.
_NATIVE_TOP_P_TOP_K: dict[str, tuple[float, int]] = {
    "temp_pure":     (1.0, 0),
    "temp_standard": (0.95, 20),  # deepseek_recipe
    "top_p_0_95":    (0.95, 0),
    "top_p_0_9":     (0.9, 0),
    "top_p_0_85":    (0.85, 0),
    "top_p_0_8":     (0.8, 0),
    "top_p_0_75":    (0.75, 0),
    "top_p_0_7":     (0.7, 0),
}


def _load_task_selection() -> dict[str, list[dict]]:
    """Return {platform: [ {task_id, difficulty, contest_date}, ... ]}."""
    by_platform: dict[str, list[dict]] = {}
    with TASK_SELECTION_CSV.open() as f:
        for row in csv.DictReader(f):
            by_platform.setdefault(row["platform"], []).append(row)
    return by_platform


def _output_path(out_root: Path, config_name: str, platform: str,
                  temperature: float) -> Path:
    # Same "_think_t{temperature}" suffix as the Qwen3-8B/Qwen2.5-Coder
    # scripts' convention -- report_lcb_v6_stratified60.py's default glob
    # discovers it with no script changes beyond the registry additions
    # (RECIPE_BY_MODEL_SUBSTR / THINK_END_MARKER_BY_MODEL_SUBSTR /
    # EXTRA_CONFIGS_BY_MODEL_SUBSTR) made alongside this script.
    return out_root / f"{config_name}_{platform}_think_t{temperature}.jsonl"


def _run_config_on_platform(
    *, config_name: str, sampler_name: str, temperature: float,
    renyi_k: float | None, platform: str, problems: list, engine, tokenizer,
    n_samples: int, max_new_tokens: int, out_root: Path, model_id: str,
    seed: int | None, per_config_deadline: float, global_deadline: float,
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
        )
        prompt_text = encode_prompt_for_vllm(
            prompt_text, getattr(engine, "_safe_tokenizer", None)
        )
        kwargs = {}
        if renyi_k is not None:
            kwargs["renyi_k"] = renyi_k
        try:
            if sampler_name in _NATIVE_TOP_P_TOP_K:
                top_p, top_k = _NATIVE_TOP_P_TOP_K[sampler_name]
                raw_samples = generate_samples_standard_vllm(
                    engine=engine, tokenizer=tokenizer, prompt_text=prompt_text,
                    n_samples=n_samples, max_new_tokens=max_new_tokens,
                    temperature=temperature, stop_strings=None,
                    top_p=top_p, top_k=top_k, seed=seed,
                )
            else:
                raw_samples = generate_samples_vllm(
                    engine=engine, tokenizer=tokenizer, prompt_text=prompt_text,
                    sampler_name=sampler_name, n_samples=n_samples,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    stop_strings=None, seed=seed, **kwargs,
                )
        except Exception as exc:
            print(f"[{config_name}/{platform}] task {problem.task_id} crashed: {exc!r}")
            continue

        elapsed = time.time() - t0
        samples_with_think = [code_prefix + s for s in raw_samples]
        samples = [_strip_think_content(s) for s in samples_with_think]

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
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization flag: 'awq', 'gptq', 'fp8' etc. "
                         "Required if model+KV won't fit unquantized.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=36000)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--configs", nargs="+", default=CONFIG_NAMES, choices=CONFIG_NAMES)
    ap.add_argument("--platforms", nargs="+", default=["atcoder", "leetcode"],
                    choices=["atcoder", "leetcode"])
    ap.add_argument("--only-task-ids", nargs="+", default=None)
    ap.add_argument("--budget-minutes", type=float, default=3000.0)
    ap.add_argument("--per-config-minutes", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    global_start = time.time()
    global_deadline = global_start + args.budget_minutes * 60.0

    selection = _load_task_selection()
    model_slug = args.model.replace("/", "--")
    out_root = REPO / OUT_ROOT_TEMPLATE.format(model_slug=model_slug)

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

    config_by_name = {c[0]: c for c in CONFIGS}
    written = []
    for platform in args.platforms:
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
            )
            written.append(p)

    total = time.time() - global_start
    print(f"\nDone in {total/60:.1f} min. Wrote:")
    for p in written:
        n = sum(1 for _ in p.open()) if p.exists() else 0
        print(f"  {p}   ({n} tasks)")


if __name__ == "__main__":
    main()
