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
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bench.apps.dataset import DIFFICULTIES, SOURCES, load_apps
from bench.apps.prompts import (
    format_prompt_apps_bigcode_chat,
    format_prompt_apps_bigcode_default,
    format_prompt_apps_cot_prefill,
    format_prompt_apps_instruct,
)
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


def _chunk_sizes(n_samples: int, hf_batch_size: int | None) -> list[int]:
    """Split n_samples into per-call batch sizes for HF generation.

    Returns a list of positive ints that sums to n_samples. ``hf_batch_size``
    None / 0 / negative / >= n_samples means a single chunk (preserve current
    behavior). Used by the pless / pless_alpha / split branches to bound
    KV cache memory on H100s — at n_samples=100 with Deepseek-6.7B and 1024
    max new tokens, a single-call batch needs ~100 GiB KV, exceeding the
    80 GiB capacity.
    """
    if hf_batch_size is None or hf_batch_size <= 0 or hf_batch_size >= n_samples:
        return [n_samples]
    chunks = [hf_batch_size] * (n_samples // hf_batch_size)
    rem = n_samples % hf_batch_size
    if rem:
        chunks.append(rem)
    return chunks


def _method_key(args: argparse.Namespace) -> str:
    if args.method == "split":
        key = (
            f"split_{args.sampler_think}_t{args.temp_think}_"
            f"{args.sampler_code}_t{args.temp_code}"
        )
    else:
        key = args.method
        # Encode temp-method filters so distinct (top_p, top_k) configs at the
        # same temperature don't collide on disk (e.g. temp+top_p0.95 vs
        # temp+top_k20 vs the combined config).
        if args.method == "temp":
            if args.top_p < 1.0:
                key = f"{key}_p{args.top_p}"
            if args.top_k > 0:
                key = f"{key}_k{args.top_k}"
            if args.repetition_penalty != 1.0:
                key = f"{key}_rp{args.repetition_penalty}"
    if args.enable_thinking:
        key = f"{key}_think_t{args.temperature}"
    if args.post_temperature is not None:
        key = f"{key}_pt{args.post_temperature}"
    if args.method == "pless_alpha":
        key = f"{key}_a{args.alpha}"
    return key


def _build_argparser() -> argparse.ArgumentParser:
    """Return the bare argparse.ArgumentParser without parsing sys.argv.

    Exposed so tests can construct + drive the parser without invoking
    main(). All validation lives in parse_args (called once main runs).
    """
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
    p.add_argument("--top-p", type=float, default=1.0,
                   help="Nucleus sampling cutoff (only applied when "
                        "--method temp). Default 1.0 disables nucleus. "
                        "Paper-replica uses 0.95.")
    p.add_argument("--top-k", type=int, default=0,
                   help="Top-k cutoff (only applied when --method temp). "
                        "Default 0 disables top-k. Qwen3 recommends 20. "
                        "Combine with --top-p 0.95 --temperature 0.6 for "
                        "Qwen3's full recommended generation config.")
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="Repetition penalty for --method temp (HF + vLLM "
                        "standard paths). Default 1.0 = no-op. Qwen2.5-Coder "
                        "ships 1.1 (7B) / 1.05 (3B) in its generation_config; "
                        "set it for the provider-faithful standard-decoder "
                        "baseline. pless/pless_norm ignore it (hyperparameter-"
                        "free by design).")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Default 8192 to accommodate Qwen3 thinking.")
    p.add_argument("--results-dir", default="results/pless_apps_results")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="Generation backend. Default 'hf' (current behaviour, zero regression). "
                        "'vllm' routes through bench/generator_vllm.py and requires the "
                        ".venv-vllm environment (see pyproject-vllm.toml).")
    p.add_argument("--hf-batch-size", type=int, default=10,
                   help="HF backend: split n_samples into chunks of this size to "
                        "avoid CUDA OOM at large n_samples (e.g. N=100 on Deepseek-6.7B "
                        "needs ~100 GiB KV cache vs 80 GiB H100 capacity). "
                        "Default 10 keeps peak VRAM ~23 GiB for 6.7B bf16. "
                        "No effect on --backend vllm (paged attention handles batching).")
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
    p.add_argument("--treat-as-instruct", action="store_true",
                   help="Force instruct-prompt formatting even when the model "
                        "id does not contain 'Instruct' or 'Chat'. Use for "
                        "models like m-a-p/OpenCodeInterpreter-DS-1.3B that "
                        "are chat-tuned but not named accordingly.")
    p.add_argument("--prompt-format",
                   choices=["auto", "bigcode-default", "bigcode-chat",
                            "cot-prefill"],
                   default="auto",
                   help="Which prompt formatter to use. 'auto' (default): "
                        "applies our chat-template-based formatter "
                        "(format_prompt_apps_instruct). 'bigcode-default': "
                        "emits bigcode-evaluation-harness's APPS prompt "
                        "verbatim (no chat template; 'QUESTION/Use Standard "
                        "Input format/ANSWER' bare-completion style). "
                        "'bigcode-chat': bigcode's bare prompt wrapped via "
                        "tokenizer.apply_chat_template() — the modification "
                        "paper authors most plausibly applied to use "
                        "bigcode-eval-harness on instruct models. Set "
                        "this when isolating backend effects (HF vs vLLM) "
                        "from prompt-format effects. Incompatible with "
                        "--paper-replica-model.")
    p.add_argument("--paper-replica-model", default=None,
                   help="OPTIONAL: HF model id whose paper-published prompts "
                        "should be used verbatim (loaded from "
                        "sh0416/outputs-apps via bench.apps.paper_replica). "
                        "When set, bypasses format_prompt_apps_instruct "
                        "entirely — the runner injects the paper's exact "
                        "prompt string for each matching problem_id. "
                        "Problems with no matching paper prompt are SKIPPED "
                        "(with a warning). Default behavior (this flag "
                        "unset) is unchanged.")
    p.add_argument("--paper-replica-cache-dir", type=Path, default=None,
                   help="Where to cache the dedup'd paper-prompt parquet. "
                        "Default: results/pless_alpha_apps/_paper_replica_cache/")
    p.add_argument("--log-entropy", action="store_true",
                   help="Log per-position next-token entropy stats "
                        "(Σpᵢ², Σpᵢ³, Σpᵢ⁵, max(pᵢ), top-32) to a sidecar "
                        "JSONL at <out_path>.entropy.jsonl. Only works with "
                        "--method pless / pless_norm / pless_alpha (the "
                        "generate_samples path that supports the entropy_log "
                        "hook). Mirrors the MBPP and GSM8K runners' "
                        "--log-entropy flag — used to extend the survival-"
                        "vs-entropy central figure to APPS for Deepseek.")
    return p


def parse_args():
    """Build the parser, parse sys.argv, and apply cross-flag validation.

    Validation centralized here so tests can build the parser without
    triggering the validation (via _build_argparser) or with it
    (via this function).
    """
    p = _build_argparser()
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
    if args.prompt_format != "auto" and args.paper_replica_model:
        p.error(f"--prompt-format {args.prompt_format} is incompatible with "
                "--paper-replica-model (both override the default formatter; "
                "pick one)")
    if args.log_entropy:
        # Only generate_samples (the manual token-by-token decode) exposes
        # the entropy_log hook. temp / split / vllm paths don't capture the
        # raw softmax. Refuse rather than silently skip the sidecar.
        if args.method == "temp":
            p.error("--log-entropy requires --method pless / pless_norm / "
                    "pless_alpha — temp routes through generate_samples_"
                    "standard (model.generate) which doesn't capture per-"
                    "position softmax.")
        if args.method == "split":
            p.error("--log-entropy not yet supported with --method split "
                    "(generate_samples_split doesn't expose entropy_log).")
        if args.backend == "vllm":
            p.error("--log-entropy not supported with --backend vllm "
                    "(no entropy_log hook in the vLLM generator).")
    return args


def main():
    args = parse_args()

    if not is_instruct_model(args.model) and not args.treat_as_instruct:
        raise SystemExit(
            f"APPS runner requires an instruct model — {args.model!r} looks "
            "like a base model. Add 'Instruct' to the model id, pass "
            "--treat-as-instruct if the model is chat-tuned but not named "
            "accordingly, or switch to MBPP."
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

    # Optional: load paper-replica prompts (Phase A Deepseek comparison).
    # When set, we filter problems to only those the paper has prompts for,
    # and the inner loop injects the paper's prompt string instead of calling
    # format_prompt_apps_instruct. Default (None) preserves existing behavior.
    paper_prompts: dict[int, str] | None = None
    if args.paper_replica_model is not None:
        from bench.apps.paper_replica import load_paper_prompts
        cache_dir = (args.paper_replica_cache_dir
                     or Path("results/pless_alpha_apps/_paper_replica_cache"))
        print(f"[paper-replica] loading prompts for "
              f"{args.paper_replica_model} on {args.source}/{args.difficulty}")
        paper_prompts = load_paper_prompts(
            model=args.paper_replica_model,
            source=args.source,
            difficulty=args.difficulty,
            cache_dir=cache_dir,
        )
        # Filter problems to those for which the paper has a prompt
        before = len(problems)
        problems = [p for p in problems if p.problem_id in paper_prompts]
        skipped = before - len(problems)
        print(f"[paper-replica] {len(problems)} problems matched paper prompts "
              f"({skipped} skipped — no paper prompt available)")
        if not problems:
            raise SystemExit(
                "[paper-replica] No overlap between requested bucket and "
                "paper's prompts. Check (model, source, difficulty)."
            )

    remaining = [p for p in problems if p.problem_id not in completed_ids]
    print(f"{args.source} / {args.difficulty}: {len(problems)} problems "
          f"({len(remaining)} remaining after resume)")

    bar = tqdm(remaining,
               desc=f"{_method_key(args)} on {args.source}/{args.difficulty}")
    for problem in bar:
        try:
            if paper_prompts is not None:
                # Inject paper's exact prompt string verbatim — bypass our
                # chat-template wrapper entirely for Phase A apples-to-apples.
                prompt_text = paper_prompts[problem.problem_id]
                code_prefix = ""
            elif args.prompt_format == "bigcode-default":
                # Reproduce bigcode-evaluation-harness's APPS get_prompt()
                # byte-for-byte. No chat template, no system prompt; treats
                # the model as bare-completion. Used to isolate backend
                # (HF vs vLLM) effects from prompt-format effects.
                prompt_text, code_prefix = format_prompt_apps_bigcode_default(problem)
            elif args.prompt_format == "cot-prefill":
                # Induce CoT from an instruct (non-reasoning) model via a
                # <think> prefill (DeepSeek-R1-Distill style). Prompt ends with
                # "<think>\n"; model is told to close with </think> then code.
                prompt_text, code_prefix = format_prompt_apps_cot_prefill(
                    problem, tokenizer,
                )
            elif args.prompt_format == "bigcode-chat":
                # bigcode's bare prompt wrapped in the model's chat template
                # — what paper authors most plausibly did to make
                # bigcode-eval-harness work on instruct models like
                # Deepseek-Coder-Instruct (otherwise bare bigcode → C++ /
                # off-topic output as seen in our smoke).
                prompt_text, code_prefix = format_prompt_apps_bigcode_chat(
                    problem, tokenizer,
                )
            else:
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
                        top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
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
                # generate_samples_standard handles chunking internally via
                # its own hf_batch_size kwarg.
                raw_samples = generate_samples_standard(
                    model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                    n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, stop_strings=None,
                    top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    hf_batch_size=args.hf_batch_size,
                )
            elif args.method == "split":
                # Loop chunks externally — keeps generate_samples_split
                # signature unchanged (also used by MBPP/HE).
                raw_samples = []
                for b in _chunk_sizes(args.n_samples, args.hf_batch_size):
                    raw_samples.extend(generate_samples_split(
                        model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_fn_think=sampler_fn_think,
                        sampler_fn_code=sampler_fn_code,
                        n_samples=b, max_new_tokens=args.max_new_tokens,
                        temperature_think=args.temp_think,
                        temperature_code=args.temp_code,
                        stop_strings=None,
                    ))
            else:
                # pless / pless_alpha / pless_norm — same chunk-at-call-site
                # pattern as split path. Each chunk re-runs prefill (small
                # cost, ~1% of total) but bounds KV memory.
                #
                # When --log-entropy is set, we accumulate per-position
                # entropy records across chunks here and renumber sample_ids
                # to give a flat 0..n_samples-1 sequence in the sidecar
                # (each generate_samples call emits sample_ids 0..b-1 for
                # its chunk; without renumbering the sidecar would have
                # duplicate ids across chunks).
                raw_samples = []
                entropy_log = [] if args.log_entropy else None
                sample_offset = 0
                for b in _chunk_sizes(args.n_samples, args.hf_batch_size):
                    chunk_log = [] if args.log_entropy else None
                    raw_samples.extend(generate_samples(
                        model=model, tokenizer=tokenizer, prompt_text=prompt_text,
                        sampler_fn=sampler_fn,
                        n_samples=b, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, stop_strings=None,
                        entropy_log=chunk_log,
                    ))
                    if chunk_log is not None and entropy_log is not None:
                        for rec in chunk_log:
                            rec2 = dict(rec)
                            rec2["sample_id"] = rec.get("sample_id", 0) + sample_offset
                            entropy_log.append(rec2)
                    sample_offset += b

            # Both native thinking (--enable-thinking) and induced CoT
            # (--prompt-format cot-prefill) emit a </think> we must strip so
            # `samples` holds the code that reaches eval; the raw trace is kept
            # in `samples_with_thinking`.
            has_cot = args.enable_thinking or args.prompt_format == "cot-prefill"
            samples_with_think = [code_prefix + s for s in raw_samples]
            if has_cot:
                samples = [_strip_think_content(s) for s in samples_with_think]
            else:
                samples = samples_with_think

            record = {
                "model": args.model,
                "backend": args.backend,
                "method": args.method,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "task_id": problem.problem_id,
                "source": problem.source,
                "difficulty": problem.difficulty,
                "prompt_text": problem.question,
                "samples": samples,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if args.alpha is not None:
                record["alpha"] = args.alpha
            if args.repetition_penalty != 1.0:
                record["repetition_penalty"] = args.repetition_penalty
            if args.paper_replica_model is not None:
                record["paper_replica_model"] = args.paper_replica_model
            if has_cot:
                record["samples_with_thinking"] = samples_with_think
            if args.prompt_format == "cot-prefill":
                record["prompt_format"] = args.prompt_format
            if args.method == "split":
                record["sampler_think"] = args.sampler_think
                record["sampler_code"] = args.sampler_code
                record["temp_think"] = args.temp_think
                record["temp_code"] = args.temp_code
            if args.post_temperature is not None:
                record["post_temperature"] = args.post_temperature

            append_result(out_path, record)

            # Entropy sidecar — mirrors bench/gsm8k/runner.py:174-183 and
            # bench/runner.py MBPP-side pattern. One row per (sample, position).
            # Only emitted when --log-entropy was set AND the method routes
            # through generate_samples (pless/pless_alpha/pless_norm); other
            # methods are rejected at parse time.
            if args.log_entropy and entropy_log:
                entropy_sidecar = out_path.with_suffix(
                    out_path.suffix + ".entropy.jsonl"
                )
                with entropy_sidecar.open("a") as fh:
                    for rec in entropy_log:
                        rec_out = {"task_id": problem.problem_id, **rec}
                        fh.write(json.dumps(rec_out) + "\n")

            tqdm.write(f"Completed problem_id={problem.problem_id}")
        except Exception as exc:
            tqdm.write(f"Error on problem_id={problem.problem_id}: {exc}")
            continue

    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
