"""CUDA/vLLM runner for the 5 new p_less_variants on the pless-α=2 failure list.

Same experiment as ``scripts/run_pless_variants_qwen3_8b_mps.py`` (see that file
for design rationale, task-id source, prompt formatting) but uses the vLLM
backend so it can run Qwen3-8B in *thinking* mode at cap ≥ 20k tokens on a
CUDA GPU (paged-attention, FlashAttention).

The 5 variants (from ``bench/pless_variants_vllm.py``) are spliced into
``bench.generator_vllm._SAMPLER_LOGIT_FN`` at startup, then routed via
``generate_samples_vllm(sampler_name=...)`` — no changes to the upstream
generator module needed.

Example (on 4090 16 GB — needs AWQ int4 or FP8 to fit)::

    uv run --project pyproject-vllm.toml python scripts/run_pless_variants_qwen3_8b_cuda.py \\
        --model Qwen/Qwen3-8B-AWQ --quantization awq \\
        --n-samples 5 --max-new-tokens 32768 \\
        --only-task-ids 2478 2502 2522 2523 \\
        --variants pless_min2 pless_effk pless_topmass pless_top1half pless_gini
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

from bench.apps.dataset import load_apps
from bench.apps.prompts import format_prompt_apps_instruct
from bench.generator import _strip_think_content
from bench.generator_vllm import (
    encode_prompt_for_vllm,
    generate_samples_vllm,
    load_engine,
)

VARIANT_NAMES = [
    "pless_effk", "pless_min2", "pless_topmass", "pless_top1half", "pless_gini",
]

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = (
    REPO / "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/"
    "ATCODER_interview_all_252/analysis/pless_a2_failures_solved_by_others.csv"
)
OUT_ROOT = REPO / "results/pless_variants_cuda/Qwen--Qwen3-8B/ATCODER_interview"


def _load_task_ids() -> list[int]:
    with CSV_PATH.open() as f:
        return [int(r["task_id"]) for r in csv.DictReader(f)]


def _run_variant(
    *, variant: str, engine, tokenizer, problems, n_samples: int,
    max_new_tokens: int, temperature: float, per_variant_deadline: float,
    global_deadline: float, enable_thinking: bool, model_id: str,
) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "think" if enable_thinking else "nothink"
    out_path = OUT_ROOT / f"{variant}_{mode}_t{temperature}.jsonl"
    if out_path.exists():
        out_path.unlink()
    print(f"[{variant}] writing → {out_path}")

    for idx, problem in enumerate(problems, 1):
        now = time.time()
        if now > per_variant_deadline or now > global_deadline:
            print(f"[{variant}] budget exhausted after {idx-1}/{len(problems)} tasks")
            break
        t0 = time.time()
        prompt_text, code_prefix = format_prompt_apps_instruct(
            problem, tokenizer, enable_thinking=enable_thinking,
        )
        prompt_text = encode_prompt_for_vllm(
            prompt_text, getattr(engine, "_safe_tokenizer", None)
        )
        try:
            raw_samples = generate_samples_vllm(
                engine=engine, tokenizer=tokenizer, prompt_text=prompt_text,
                sampler_name=variant, n_samples=n_samples,
                max_new_tokens=max_new_tokens, temperature=temperature,
                stop_strings=None,
            )
        except Exception as exc:
            print(f"[{variant}] task {problem.problem_id} crashed: {exc!r}")
            continue

        elapsed = time.time() - t0
        samples_with_think = [code_prefix + s for s in raw_samples]
        samples = ([_strip_think_content(s) for s in samples_with_think]
                   if enable_thinking else samples_with_think)

        record = {
            "model": model_id, "backend": "vllm", "method": variant,
            "temperature": temperature, "top_p": 1.0, "top_k": 0,
            "task_id": problem.problem_id, "source": problem.source,
            "difficulty": problem.difficulty, "prompt_text": problem.question,
            "samples": samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enable_thinking": enable_thinking,
        }
        if enable_thinking:
            record["samples_with_thinking"] = samples_with_think
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{variant}] task {problem.problem_id}: {elapsed:.0f}s")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization flag: 'awq', 'gptq', 'fp8' etc. "
                         "Required if model+KV won't fit in unquantized bf16 "
                         "(16 GB card cannot hold bf16 8B + KV — use awq/fp8).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="Fraction of GPU memory vLLM may claim.")
    ap.add_argument("--max-model-len", type=int, default=40000,
                    help="Max context length vLLM allocates KV for. Set to "
                         "prompt_len + max_new_tokens headroom.")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                    choices=VARIANT_NAMES)
    ap.add_argument("--only-task-ids", type=int, nargs="+", default=None)
    ap.add_argument("--budget-minutes", type=float, default=360.0)
    ap.add_argument("--per-variant-minutes", type=float, default=120.0)
    args = ap.parse_args()

    global_start = time.time()
    global_deadline = global_start + args.budget_minutes * 60.0

    task_ids = _load_task_ids()
    if args.only_task_ids is not None:
        wanted = set(args.only_task_ids)
        task_ids = [t for t in task_ids if t in wanted]
    print(f"Task ids ({len(task_ids)}): {task_ids}")

    problems = [p for p in load_apps(source="ATCODER", difficulty="interview")
                if p.problem_id in set(task_ids)]
    problems.sort(key=lambda p: p.problem_id)
    print(f"Matched {len(problems)}/{len(task_ids)} problems")

    load_kwargs = {"dtype": args.dtype, "gpu_memory_utilization":
                   args.gpu_memory_utilization,
                   "max_model_len": args.max_model_len}
    if args.quantization:
        load_kwargs["quantization"] = args.quantization
    print(f"Loading engine: {args.model} (kwargs={load_kwargs})")
    t0 = time.time()
    engine = load_engine(args.model, **load_kwargs)
    print(f"Engine ready in {time.time()-t0:.0f}s")
    tokenizer = (getattr(engine, "_safe_tokenizer", None)
                 or engine.get_tokenizer())

    written = []
    for variant in args.variants:
        per_variant_deadline = min(
            time.time() + args.per_variant_minutes * 60.0, global_deadline
        )
        p = _run_variant(
            variant=variant, engine=engine, tokenizer=tokenizer,
            problems=problems, n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            per_variant_deadline=per_variant_deadline,
            global_deadline=global_deadline,
            enable_thinking=not args.no_thinking, model_id=args.model,
        )
        written.append(p)

    total = time.time() - global_start
    print(f"\nDone in {total/60:.1f} min. Wrote:")
    for p in written:
        n = sum(1 for _ in p.open()) if p.exists() else 0
        print(f"  {p}   ({n} tasks)")
    print("\nEvaluate with:")
    for p in written:
        print(f"  uv run python -m bench.eval --dataset apps --results-file {p}")


if __name__ == "__main__":
    main()
