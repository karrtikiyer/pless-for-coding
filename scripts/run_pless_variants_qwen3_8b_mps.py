"""Drive the two new pless variants on the 16 Qwen3-8B / APPS-252 tasks that
canonical pless-α=2 failed on but at least one non-pless decoder solved.

Task-id list source: results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/
ATCODER_interview_all_252/analysis/pless_a2_failures_solved_by_others.csv.

Variants (in p_less_variants/, NOT modifying p-less/):
  - pless_effk : rank-cut at ⌈1/Σpᵢ²⌉ (min 2)
  - pless_min2 : plain pless with a min-2 survivor floor

Runs Qwen3-8B on MPS with thinking enabled, 5 samples per task per variant.
Batches all 5 samples for a task in a single prefill+decode pass.

Hard wall-clock stop: exits (leaves partial JSONL for eval) if the global
budget is exhausted or a per-variant cap is hit.

Output: one JSONL per variant under
  results/pless_variants_mps/Qwen--Qwen3-8B/ATCODER_interview/{variant}_...jsonl
Format matches bench.apps.runner records so `python -m bench.eval --dataset
apps --results-file <file>` evaluates them unchanged.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.apps.dataset import load_apps
from bench.apps.prompts import format_prompt_apps_instruct
from bench.generator import (
    _strip_think_content,
    generate_samples,
    load_model_and_tokenizer,
)
from p_less_variants import (
    p_less_effk_decode,
    p_less_gini_decode,
    p_less_min2_decode,
    p_less_top1half_decode,
    p_less_topmass_decode,
)

VARIANTS = {
    "pless_effk":     p_less_effk_decode,
    "pless_min2":     p_less_min2_decode,
    "pless_topmass":  p_less_topmass_decode,
    "pless_top1half": p_less_top1half_decode,
    "pless_gini":     p_less_gini_decode,
}

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = (
    REPO / "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/"
    "ATCODER_interview_all_252/analysis/pless_a2_failures_solved_by_others.csv"
)
OUT_ROOT = REPO / "results/pless_variants_mps/Qwen--Qwen3-8B/ATCODER_interview"


def _load_task_ids() -> list[int]:
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    return [int(r["task_id"]) for r in rows]


def _run_variant(
    *, variant: str, model, tokenizer, problems, n_samples: int,
    max_new_tokens: int, temperature: float, per_variant_deadline: float,
    global_deadline: float, enable_thinking: bool = True,
) -> Path:
    sampler_fn = VARIANTS[variant]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "think" if enable_thinking else "nothink"
    key = f"{variant}_{mode}_t{temperature}"
    out_path = OUT_ROOT / f"{key}.jsonl"
    # Fresh run per invocation (no resume) so partial timings are transparent.
    if out_path.exists():
        out_path.unlink()

    print(f"[{variant}] writing → {out_path}")
    print(f"[{variant}] deadline in {per_variant_deadline - time.time():.0f}s "
          f"(global: {global_deadline - time.time():.0f}s)")

    for idx, problem in enumerate(problems, 1):
        now = time.time()
        if now > per_variant_deadline or now > global_deadline:
            print(f"[{variant}] budget exhausted after {idx-1}/"
                  f"{len(problems)} tasks — stopping variant.")
            break
        remaining_tasks = len(problems) - (idx - 1)
        per_task_budget = min(
            (per_variant_deadline - now) / remaining_tasks,
            (global_deadline - now) / remaining_tasks,
        )
        t0 = time.time()
        prompt_text, code_prefix = format_prompt_apps_instruct(
            problem, tokenizer, enable_thinking=enable_thinking,
        )
        try:
            raw_samples = generate_samples(
                model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                sampler_fn=sampler_fn, n_samples=n_samples,
                max_new_tokens=max_new_tokens, temperature=temperature,
                stop_strings=None,
            )
        except Exception as exc:  # keep going — eval treats missing tasks as 0
            print(f"[{variant}] task {problem.problem_id}: gen crashed → {exc!r}")
            continue

        elapsed = time.time() - t0
        samples_with_think = [code_prefix + s for s in raw_samples]
        if enable_thinking:
            samples = [_strip_think_content(s) for s in samples_with_think]
        else:
            samples = samples_with_think

        record = {
            "model": "Qwen/Qwen3-8B",
            "backend": "hf",
            "method": variant,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "task_id": problem.problem_id,
            "source": problem.source,
            "difficulty": problem.difficulty,
            "prompt_text": problem.question,
            "samples": samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enable_thinking": enable_thinking,
        }
        if enable_thinking:
            record["samples_with_thinking"] = samples_with_think
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        print(f"[{variant}] task {problem.problem_id}: {elapsed:.0f}s "
              f"(budget/task ≈ {per_task_budget:.0f}s)")

        # Free MPS scratch between tasks (KV cache from prefill is the big one).
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--only-task-ids", type=int, nargs="+", default=None,
                    help="If set, restrict to this subset of the CSV's 16 ids. "
                         "Used to focus MPS budget on tasks whose baseline "
                         "think traces are short enough to fit our cap.")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Disable Qwen3 thinking mode (generate code directly). "
                         "Trades apples-to-apples comparability with the CSV's "
                         "think-mode baseline for a KV footprint that actually "
                         "fits a 48 GB M4 Max under other-app pressure.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                    choices=list(VARIANTS.keys()))
    ap.add_argument("--budget-minutes", type=float, default=90.0,
                    help="Hard global wall-clock budget across all variants.")
    ap.add_argument("--per-variant-minutes", type=float, default=45.0,
                    help="Per-variant cap so variant 1 can't starve variant 2.")
    args = ap.parse_args()

    global_start = time.time()
    global_deadline = global_start + args.budget_minutes * 60.0

    task_ids = _load_task_ids()
    if args.only_task_ids is not None:
        wanted = set(args.only_task_ids)
        task_ids = [t for t in task_ids if t in wanted]
    print(f"Task ids ({len(task_ids)}): {task_ids}")

    print("Loading APPS ATCODER/interview …")
    problems = [
        p for p in load_apps(source="ATCODER", difficulty="interview")
        if p.problem_id in set(task_ids)
    ]
    problems.sort(key=lambda p: p.problem_id)
    print(f"Matched {len(problems)}/{len(task_ids)} problems")

    print("Loading Qwen/Qwen3-8B on MPS (bfloat16, sdpa) …")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        "Qwen/Qwen3-8B", dtype="bfloat16", attn_impl="sdpa",
    )
    print(f"Model loaded in {time.time()-t0:.0f}s "
          f"(device={next(model.parameters()).device})")

    written = []
    for variant in args.variants:
        per_variant_deadline = min(
            time.time() + args.per_variant_minutes * 60.0,
            global_deadline,
        )
        out_path = _run_variant(
            variant=variant, model=model, tokenizer=tokenizer,
            problems=problems, n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            per_variant_deadline=per_variant_deadline,
            global_deadline=global_deadline,
            enable_thinking=not args.no_thinking,
        )
        written.append(out_path)

    total = time.time() - global_start
    print(f"\nDone in {total/60:.1f} min. Wrote:")
    for p in written:
        n_records = sum(1 for _ in p.open()) if p.exists() else 0
        print(f"  {p}   ({n_records} tasks)")
    print("\nEvaluate with:")
    for p in written:
        print(f"  uv run python -m bench.eval --dataset apps --results-file {p}")


if __name__ == "__main__":
    main()
