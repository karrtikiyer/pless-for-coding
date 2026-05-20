"""APPS CLI runner — generates samples for a single (source, difficulty, config).

Mirrors ``bench/runner.py`` (MBPP) but:

* Loads APPS via :mod:`bench.apps.dataset` (filtered by source + difficulty).
* Formats prompts via :func:`bench.apps.prompts.format_prompt_apps_instruct`.
* Only supports instruct-model prompting; no few-shot base-model formats
  (APPS prompts are already 800-7400 chars).
* Does **not** emit test cases or attempt correctness scoring (APPS uses
  stdin/stdout I/O; out of scope for v1 — algosim measures diversity on
  the raw samples).
* JSONL output path embeds (source, difficulty) as a subdir so we can run
  buckets independently and keep them organized.

Usage (one invocation per config; see ``run_apps_qwen3_top_configs.sh`` for the
6-config sweep)::

    uv run python -m bench.apps \\
        --model Qwen/Qwen3-8B \\
        --source ATCODER --difficulty competition \\
        --method split \\
        --sampler-think temp_pure --temp-think 1.5 \\
        --sampler-code  pless     --temp-code  1.5 \\
        --enable-thinking \\
        --n-samples 10 --max-new-tokens 8192
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bench.apps.dataset import DIFFICULTIES, SOURCES, load_apps
from bench.apps.prompts import format_prompt_apps_instruct
from bench.checkpointing import append_result, load_completed_ids
from bench.generator import (
    _strip_think_content,
    generate_samples,
    generate_samples_split,
    generate_samples_standard,
    load_model_and_tokenizer,
)
from bench.prompts import is_instruct_model
from bench.sampler_bridge import (
    SAMPLERS,
    SPLIT_SAMPLERS,
    make_pless_alpha_sampler,
    make_pless_post_temp_sampler,
)


def _output_path(results_dir: str, model_id: str, source: str,
                 difficulty: str, method_key: str, temperature: float) -> Path:
    model_name = model_id.replace("/", "--")
    out_dir = Path(results_dir) / model_name / f"{source}_{difficulty}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{method_key}_t{temperature}.jsonl"


def _method_key(args: argparse.Namespace) -> str:
    if args.method == "split":
        key = (
            f"split_{args.sampler_think}_t{args.temp_think}_"
            f"{args.sampler_code}_t{args.temp_code}"
        )
    else:
        key = args.method
    if args.enable_thinking:
        key = f"{key}_think_t{args.temperature}"
    if args.post_temperature is not None:
        key = f"{key}_pt{args.post_temperature}"
    if args.method == "pless_alpha":
        key = f"{key}_a{args.alpha}"
    return key


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--source", required=True, choices=list(SOURCES),
                   help="APPS source bucket (ATCODER or CODEFORCES)")
    p.add_argument("--difficulty", required=True, choices=list(DIFFICULTIES),
                   help="APPS difficulty bucket")
    p.add_argument(
        "--method", required=True,
        choices=list(SAMPLERS.keys()) + ["temp", "split", "pless_alpha"],
        help="Sampling method",
    )
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Default 8192 to accommodate Qwen3 thinking.")
    p.add_argument("--results-dir", default="results/pless_apps_results")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="Generation backend. Default 'hf' (current behaviour, zero regression). "
                        "'vllm' routes through bench/generator_vllm.py and requires the "
                        ".venv-vllm environment (see pyproject-vllm.toml).")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap problems within the (source, difficulty) bucket (for smoke tests).")
    p.add_argument("--task-ids", type=int, nargs="+", default=None,
                   help="Only run these specific APPS problem_ids.")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None)
    p.add_argument("--enable-thinking", action="store_true")
    # split-decoding knobs
    p.add_argument("--temp-think", type=float, default=None)
    p.add_argument("--temp-code", type=float, default=None)
    p.add_argument("--sampler-think", choices=list(SPLIT_SAMPLERS.keys()), default=None)
    p.add_argument("--sampler-code", choices=list(SPLIT_SAMPLERS.keys()), default=None)
    p.add_argument("--post-temperature", type=float, default=None,
                   help="T₂ for p-less post-truncation flattening.")
    p.add_argument("--alpha", type=float, default=None,
                   help="Rényi exponent for --method pless_alpha. "
                        "Threshold = Σpᵢ^α. α=2 reproduces standard pless; "
                        "α>2 keeps more tokens at high-entropy positions.")
    args = p.parse_args()
    if args.method == "split":
        for name in ("temp_think", "temp_code", "sampler_think", "sampler_code"):
            if getattr(args, name) is None:
                p.error(f"--{name.replace('_', '-')} is required when --method is split")
    if args.post_temperature is not None and args.method not in SAMPLERS:
        p.error("--post-temperature only works with p-less methods")
    if args.method == "pless_alpha" and args.alpha is None:
        p.error("--alpha is required when --method is pless_alpha")
    if args.alpha is not None and args.method != "pless_alpha":
        p.error("--alpha only applies to --method pless_alpha")
    return args


def main():
    args = parse_args()

    if not is_instruct_model(args.model):
        raise SystemExit(
            f"APPS runner requires an instruct model — {args.model!r} looks "
            "like a base model. Add 'Instruct' to the model id or switch to MBPP."
        )

    out_path = _output_path(
        args.results_dir, args.model, args.source, args.difficulty,
        _method_key(args), args.temperature,
    )
    if args.no_resume and out_path.exists():
        out_path.unlink()

    completed_ids = load_completed_ids(out_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} problems already completed at {out_path}")

    print(f"Loading model: {args.model} (backend={args.backend})")
    if args.backend == "vllm":
        # Deferred import — only happens when --backend vllm is requested,
        # so the apps runner is still importable in the main .venv (no vLLM).
        from bench.generator_vllm import load_engine
        engine = load_engine(args.model, dtype=args.dtype)
        model = engine          # alias for downstream code that holds it
        tokenizer = engine.get_tokenizer()
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model, dtype=args.dtype, attn_impl=args.attn_impl,
        )

    # Resolve sampler(s) — only needed for the HF backend, since vLLM
    # selects its sampler by name inside the LogitsProcessor.
    sampler_fn = None
    sampler_fn_think = sampler_fn_code = None
    if args.backend == "hf":
        if args.method == "split":
            sampler_fn_think = SPLIT_SAMPLERS[args.sampler_think]
            sampler_fn_code = SPLIT_SAMPLERS[args.sampler_code]
        elif args.method == "pless_alpha":
            sampler_fn = make_pless_alpha_sampler(args.alpha)
        elif args.method != "temp":
            if args.post_temperature is not None:
                sampler_fn = make_pless_post_temp_sampler(args.post_temperature)
            else:
                sampler_fn = SAMPLERS[args.method]

    # Load APPS, filtered to the requested bucket.
    problems = list(load_apps(source=args.source, difficulty=args.difficulty))
    if args.task_ids is not None:
        wanted = set(args.task_ids)
        problems = [p for p in problems if p.problem_id in wanted]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    remaining = [p for p in problems if p.problem_id not in completed_ids]
    print(f"{args.source} / {args.difficulty}: {len(problems)} problems "
          f"({len(remaining)} remaining after resume)")

    bar = tqdm(remaining,
               desc=f"{_method_key(args)} on {args.source}/{args.difficulty}")
    for problem in bar:
        try:
            prompt_text, code_prefix = format_prompt_apps_instruct(
                problem, tokenizer, enable_thinking=args.enable_thinking,
            )

            if args.backend == "vllm":
                # vLLM dispatches by sampler name string, not callable.
                from bench.generator_vllm import (
                    generate_samples_split_vllm,
                    generate_samples_standard_vllm,
                    generate_samples_vllm,
                )
                if args.method == "temp":
                    raw_samples = generate_samples_standard_vllm(
                        engine=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, stop_strings=None,
                        top_p=1.0, top_k=0,
                    )
                elif args.method == "split":
                    raw_samples = generate_samples_split_vllm(
                        engine=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_fn_think=args.sampler_think,
                        sampler_fn_code=args.sampler_code,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        temperature_think=args.temp_think,
                        temperature_code=args.temp_code,
                        stop_strings=None,
                    )
                else:
                    raw_samples = generate_samples_vllm(
                        engine=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_name=args.method,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, stop_strings=None,
                        alpha=args.alpha,
                    )
            elif args.method == "temp":
                raw_samples = generate_samples_standard(
                    model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                    n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, stop_strings=None,
                    top_p=1.0, top_k=0,
                )
            elif args.method == "split":
                raw_samples = generate_samples_split(
                    model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                    sampler_fn_think=sampler_fn_think,
                    sampler_fn_code=sampler_fn_code,
                    n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                    temperature_think=args.temp_think,
                    temperature_code=args.temp_code,
                    stop_strings=None,
                )
            else:
                raw_samples = generate_samples(
                    model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                    sampler_fn=sampler_fn,
                    n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, stop_strings=None,
                )

            samples_with_think = [code_prefix + s for s in raw_samples]
            if args.enable_thinking:
                samples = [_strip_think_content(s) for s in samples_with_think]
            else:
                samples = samples_with_think

            record = {
                "model": args.model,
                "backend": args.backend,
                "method": args.method,
                "temperature": args.temperature,
                "task_id": problem.problem_id,
                "source": problem.source,
                "difficulty": problem.difficulty,
                "prompt_text": problem.question,
                "samples": samples,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if args.enable_thinking:
                record["samples_with_thinking"] = samples_with_think
            if args.method == "split":
                record["sampler_think"] = args.sampler_think
                record["sampler_code"] = args.sampler_code
                record["temp_think"] = args.temp_think
                record["temp_code"] = args.temp_code
            if args.post_temperature is not None:
                record["post_temperature"] = args.post_temperature

            append_result(out_path, record)
            tqdm.write(f"Completed problem_id={problem.problem_id}")
        except Exception as exc:
            tqdm.write(f"Error on problem_id={problem.problem_id}: {exc}")
            continue

    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
