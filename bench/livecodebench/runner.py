"""LiveCodeBench CLI runner — generates samples for a single (platform, config).

Mirrors bench/apps/runner.py but loads LiveCodeBench (bench.livecodebench.dataset)
and formats prompts via bench.livecodebench.prompts.format_prompt_lcb_instruct
(LCB's own template). Reuses the shared generation machinery
(bench.generator / generator_vllm / sampler_bridge / checkpointing) verbatim, and
the dataset-agnostic _method_key / _chunk_sizes from bench.apps.runner.

One platform per run (records carry source=platform; eval loads the matching test
map). Output: results/.../<model--slug>/LCB_<platform>/<method_key>_t<temp>.jsonl.

Usage (one invocation per config; see run_lcb_v6_apps_*.sh)::

    uv run python -m bench.livecodebench \\
        --model Qwen/Qwen3-8B --platform atcoder --backend vllm \\
        --method pless_renyi --renyi-k 0.1 --temperature 1.0 \\
        --enable-thinking --n-samples 10 --max-new-tokens 32768
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bench.apps.runner import _chunk_sizes, _method_key   # dataset-agnostic helpers
from bench.checkpointing import append_result, load_completed_ids
from bench.generator import (
    _strip_think_content,
    generate_samples,
    generate_samples_split,
    generate_samples_standard,
    load_model_and_tokenizer,
)
from bench.livecodebench.dataset import LCB_VERSION, PLATFORMS, load_lcb
from bench.livecodebench.prompts import format_prompt_lcb_instruct
from bench.prompts import is_instruct_model
from bench.sampler_bridge import (
    SAMPLERS,
    SPLIT_SAMPLERS,
    make_pless_alpha_sampler,
    make_pless_post_temp_sampler,
    make_pless_renyi_sampler,
)


def _output_path(results_dir: str, model_id: str, platform: str,
                 method_key: str, temperature: float) -> Path:
    model_name = model_id.replace("/", "--")
    out_dir = Path(results_dir) / model_name / f"LCB_{platform}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{method_key}_t{temperature}.jsonl"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--platform", required=True, choices=list(PLATFORMS),
                   help="LCB platform (one per run; records carry it as 'source').")
    p.add_argument("--version", default=LCB_VERSION, help="LCB release_tag (default v6).")
    p.add_argument("--window", default=None,
                   help="Date filter 'YYYY-MM..YYYY-MM' or 'YYYY-MM+'; default all dates.")
    p.add_argument(
        "--method", required=True,
        choices=list(SAMPLERS.keys()) + ["temp", "split", "pless_alpha", "pless_renyi"],
    )
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--results-dir", default="results/_lcb_v6")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--hf-batch-size", type=int, default=10)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap problems (after platform/window/task-id filters) for smoke tests.")
    p.add_argument("--task-ids", nargs="+", default=None,
                   help="Only run these LCB question_ids (strings).")
    p.add_argument("--task-ids-file", type=Path, default=None,
                   help="File of LCB question_ids (whitespace/newline-separated); "
                        "unioned with --task-ids.")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--temp-think", type=float, default=None)
    p.add_argument("--temp-code", type=float, default=None)
    p.add_argument("--sampler-think", choices=list(SPLIT_SAMPLERS.keys()), default=None)
    p.add_argument("--sampler-code", choices=list(SPLIT_SAMPLERS.keys()), default=None)
    p.add_argument("--post-temperature", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="Rényi exponent for --method pless_alpha (Σpᵢ^α).")
    p.add_argument("--renyi-k", type=float, default=None,
                   help="Rényi order k for --method pless_renyi ((Σpᵢ^k)^(1/(k-1))).")
    p.add_argument("--treat-as-instruct", action="store_true")
    return p


def parse_args() -> argparse.Namespace:
    args = _build_argparser().parse_args()
    if args.method == "split" and (args.sampler_think is None or args.sampler_code is None):
        raise SystemExit("--method split requires --sampler-think and --sampler-code")
    if args.method == "pless_alpha" and args.alpha is None:
        raise SystemExit("--method pless_alpha requires --alpha")
    if args.method == "pless_renyi" and args.renyi_k is None:
        raise SystemExit("--method pless_renyi requires --renyi-k")
    return args


def main():
    args = parse_args()

    if not is_instruct_model(args.model) and not args.treat_as_instruct:
        raise SystemExit(
            f"LCB runner requires an instruct model — {args.model!r} looks like a "
            "base model. Pass --treat-as-instruct if it is chat-tuned but not named so."
        )

    out_path = _output_path(args.results_dir, args.model, args.platform,
                            _method_key(args), args.temperature)
    if args.no_resume and out_path.exists():
        out_path.unlink()
    completed_ids = load_completed_ids(out_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} problems already completed at {out_path}")

    print(f"Loading model: {args.model} (backend={args.backend})")
    if args.backend == "vllm":
        from bench.generator_vllm import load_engine
        engine = load_engine(args.model, dtype=args.dtype)
        model = engine
        tokenizer = getattr(engine, "_safe_tokenizer", None) or engine.get_tokenizer()
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model, dtype=args.dtype, attn_impl=args.attn_impl)

    sampler_fn = sampler_fn_think = sampler_fn_code = None
    if args.backend == "hf":
        if args.method == "split":
            sampler_fn_think = SPLIT_SAMPLERS[args.sampler_think]
            sampler_fn_code = SPLIT_SAMPLERS[args.sampler_code]
        elif args.method == "pless_alpha":
            sampler_fn = make_pless_alpha_sampler(args.alpha)
        elif args.method == "pless_renyi":
            sampler_fn = make_pless_renyi_sampler(args.renyi_k)
        elif args.method != "temp":
            sampler_fn = (make_pless_post_temp_sampler(args.post_temperature)
                          if args.post_temperature is not None else SAMPLERS[args.method])

    # Load LCB (statement + starter only; tests not needed for generation).
    problems = list(load_lcb(version=args.version, platforms=(args.platform,),
                             window=args.window, with_tests=False))
    wanted: set[str] | None = None
    if args.task_ids is not None:
        wanted = {str(t) for t in args.task_ids}
    if args.task_ids_file is not None:
        file_ids = {tok for tok in args.task_ids_file.read_text().split()}
        wanted = file_ids if wanted is None else (wanted | file_ids)
    if wanted is not None:
        problems = [p for p in problems if p.task_id in wanted]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]

    remaining = [p for p in problems if p.task_id not in completed_ids]
    print(f"LCB {args.platform} (v={args.version}, window={args.window or 'all'}): "
          f"{len(problems)} problems ({len(remaining)} remaining after resume)")

    bar = tqdm(remaining, desc=f"{_method_key(args)} on LCB/{args.platform}")
    for problem in bar:
        try:
            prompt_text, code_prefix = format_prompt_lcb_instruct(
                problem, tokenizer, enable_thinking=args.enable_thinking)

            if args.backend == "vllm":
                from bench.generator_vllm import (
                    encode_prompt_for_vllm,
                    generate_samples_split_vllm,
                    generate_samples_standard_vllm,
                    generate_samples_vllm,
                )
                prompt_text = encode_prompt_for_vllm(
                    prompt_text, getattr(model, "_safe_tokenizer", None))
                if args.method == "temp":
                    raw_samples = generate_samples_standard_vllm(
                        engine=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, stop_strings=None,
                        top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                    )
                elif args.method == "split":
                    raw_samples = generate_samples_split_vllm(
                        engine=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_fn_think=args.sampler_think, sampler_fn_code=args.sampler_code,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        temperature_think=args.temp_think, temperature_code=args.temp_code,
                        stop_strings=None,
                    )
                else:
                    raw_samples = generate_samples_vllm(
                        engine=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_name=args.method,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, stop_strings=None,
                        alpha=args.alpha, renyi_k=args.renyi_k,
                        loop_ngram_n=None, loop_ngram_k=None, loop_window=1200,
                    )
            elif args.method == "temp":
                raw_samples = generate_samples_standard(
                    model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                    n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, stop_strings=None,
                    top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    hf_batch_size=args.hf_batch_size,
                )
            elif args.method == "split":
                raw_samples = []
                for b in _chunk_sizes(args.n_samples, args.hf_batch_size):
                    raw_samples.extend(generate_samples_split(
                        model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_fn_think=sampler_fn_think, sampler_fn_code=sampler_fn_code,
                        n_samples=b, max_new_tokens=args.max_new_tokens,
                        temperature_think=args.temp_think, temperature_code=args.temp_code,
                        stop_strings=None,
                    ))
            else:
                raw_samples = []
                for b in _chunk_sizes(args.n_samples, args.hf_batch_size):
                    raw_samples.extend(generate_samples(
                        model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_fn=sampler_fn,
                        n_samples=b, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, stop_strings=None,
                        entropy_log=None,
                    ))

            has_cot = args.enable_thinking
            samples_with_think = [code_prefix + s for s in raw_samples]
            samples = ([_strip_think_content(s) for s in samples_with_think]
                       if has_cot else samples_with_think)

            record = {
                "model": args.model,
                "backend": args.backend,
                "method": args.method,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "task_id": problem.task_id,
                "source": problem.platform,        # =platform (eval + cot grouping)
                "difficulty": problem.difficulty,
                "contest_date": problem.contest_date,   # for contamination windowing
                "prompt_text": problem.question,
                "samples": samples,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if args.alpha is not None:
                record["alpha"] = args.alpha
            if args.renyi_k is not None:
                record["renyi_k"] = args.renyi_k
            if args.repetition_penalty != 1.0:
                record["repetition_penalty"] = args.repetition_penalty
            if has_cot:
                record["samples_with_thinking"] = samples_with_think
            if args.method == "split":
                record["sampler_think"] = args.sampler_think
                record["sampler_code"] = args.sampler_code
                record["temp_think"] = args.temp_think
                record["temp_code"] = args.temp_code
            if args.post_temperature is not None:
                record["post_temperature"] = args.post_temperature

            append_result(out_path, record)
            tqdm.write(f"Completed task_id={problem.task_id}")
        except Exception as exc:
            tqdm.write(f"Error on task_id={problem.task_id}: {exc}")
            continue

    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
