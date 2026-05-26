"""GSM8K generation runner — mirrors bench/runner.py for MBPP.

CLI:
    uv run python -m bench.gsm8k \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --method pless_alpha --alpha 5.0 --temperature 1.0 \\
        --n-samples 10 \\
        --n-problems 500 --seed 0

Output: one JSONL at
    results/pless_alpha_full_gsm8k/<model_slug>/pless_alpha_a{α}_t{T}.jsonl
with one record per problem:
    {"model", "method", "alpha", "temperature", "task_id",
     "raw_index", "question", "prompt_text", "samples", "gold_answer", "timestamp"}

Resumable: re-running with the same args skips problems already in the
output JSONL (via task_id).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bench.checkpointing import append_result, load_completed_ids
from bench.generator import (
    generate_samples,
    generate_samples_standard,
    load_model_and_tokenizer,
)
from bench.gsm8k.dataset import load_gsm8k_subset
from bench.gsm8k.prompts import STOP_STRINGS, format_prompt
from bench.sampler_bridge import SAMPLERS, make_pless_alpha_sampler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run α-sweep on GSM8K")
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument(
        "--method", required=True,
        choices=list(SAMPLERS.keys()) + ["temp", "top_p", "pless_alpha"],
    )
    p.add_argument("--alpha", type=float, default=None,
                   help="Rényi exponent for --method pless_alpha")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=400,
                   help="GSM8K CoT completions are typically 200-500 tokens; "
                        "400 caps degenerate runs without truncating typical answers.")
    p.add_argument("--n-problems", type=int, default=500,
                   help="Random subset size with --seed-based reproducibility. "
                        "Default 500 ≈ 38% of GSM8K test split (1,319); same "
                        "scale as our MBPP sweep for direct comparison.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for problem subsampling — determines which "
                        "subset of GSM8K test problems is used.")
    p.add_argument("--results-dir", default="results/pless_alpha_full_gsm8k")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--log-entropy", action="store_true",
                   help="Log per-position next-token entropy stats "
                        "(Σpᵢ², Σpᵢ³, Σpᵢ⁵, max(pᵢ), top-32) to a sidecar "
                        "JSONL at <out_path>.entropy.jsonl. Only works with "
                        "--method pless / pless_alpha (the generate_samples "
                        "path that supports the entropy_log hook). Mirrors "
                        "the MBPP runner's --log-entropy flag — used to "
                        "generate the survival-vs-entropy data for the "
                        "central figure (docs/theory/central_figure_plan.md).")
    args = p.parse_args(argv)
    if args.method == "pless_alpha" and args.alpha is None:
        p.error("--alpha is required when --method is pless_alpha")
    if args.alpha is not None and args.method != "pless_alpha":
        p.error("--alpha only applies to --method pless_alpha")
    if args.log_entropy and args.method not in SAMPLERS and args.method != "pless_alpha":
        p.error(
            "--log-entropy only works with --method pless / pless_norm / "
            "pless_alpha (the generate_samples path). For temp/top_p the "
            "raw softmax isn't captured in our HF generate path."
        )
    return args


def _method_key(args: argparse.Namespace) -> str:
    if args.method == "pless_alpha":
        return f"pless_alpha_a{args.alpha}"
    if args.method == "top_p":
        return f"top_p{args.top_p}"
    return args.method


def _output_path(results_dir: str, model: str, method_key: str,
                 temperature: float) -> Path:
    slug = model.replace("/", "--")
    p = Path(results_dir) / slug
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{method_key}_t{temperature}.jsonl"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    method_key = _method_key(args)
    out_path = _output_path(args.results_dir, args.model, method_key,
                            args.temperature)

    if args.no_resume and out_path.exists():
        out_path.unlink()
    completed = load_completed_ids(out_path) if out_path.exists() else set()
    if completed:
        print(f"Resuming: {len(completed)} problems already in {out_path}")

    print(f"Loading GSM8K test subset (n_problems={args.n_problems}, "
          f"seed={args.seed})...")
    problems = load_gsm8k_subset(n_problems=args.n_problems, seed=args.seed)
    print(f"Loaded {len(problems)} problems")

    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)

    if args.method == "pless_alpha":
        sampler_fn = make_pless_alpha_sampler(args.alpha)
    elif args.method in SAMPLERS:
        sampler_fn = SAMPLERS[args.method]
    else:
        sampler_fn = None

    progress = tqdm([p for p in problems if p.task_id not in completed],
                    desc=f"{method_key} t{args.temperature}")
    for problem in progress:
        prompt_text = format_prompt(problem.question)
        if args.method in ("temp", "top_p"):
            samples = generate_samples_standard(
                model=model, tokenizer=tokenizer,
                prompt_text=prompt_text,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop_strings=STOP_STRINGS,
                top_p=args.top_p if args.method == "top_p" else 1.0,
                top_k=0,
            )
        else:
            entropy_log = [] if args.log_entropy else None
            samples = generate_samples(
                model=model, tokenizer=tokenizer,
                prompt_text=prompt_text,
                sampler_fn=sampler_fn,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop_strings=STOP_STRINGS,
                entropy_log=entropy_log,
            )

        append_result(out_path, {
            "model": args.model,
            "method": args.method,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "task_id": problem.task_id,
            "raw_index": problem.raw_index,
            "question": problem.question,
            "prompt_text": prompt_text,
            "samples": samples,
            "gold_answer": problem.gold_answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Write entropy-log sidecar (one row per (sample, position))
        # — mirrors bench/runner.py:356-361 MBPP-side pattern exactly.
        if args.log_entropy and args.method not in ("temp", "top_p"):
            entropy_log_local = locals().get("entropy_log")
            if entropy_log_local is not None:
                entropy_sidecar = out_path.with_suffix(
                    out_path.suffix + ".entropy.jsonl"
                )
                with entropy_sidecar.open("a") as fh:
                    for rec in entropy_log_local:
                        rec_out = {"task_id": problem.task_id, **rec}
                        fh.write(json.dumps(rec_out) + "\n")

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
