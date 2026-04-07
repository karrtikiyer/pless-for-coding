import argparse
from datetime import datetime, timezone

from datasets import load_dataset
from tqdm import tqdm

from bench.checkpointing import append_result, get_output_path, load_completed_ids
from bench.generator import (generate_samples, generate_samples_standard,
                              generate_samples_greedy, generate_samples_beam,
                              load_model_and_tokenizer)
from bench.humaneval.prompts import HUMANEVAL_STOP_SEQUENCES
from bench.prompts import (format_prompt_base, format_prompt_base_hybrid,
                           format_prompt_base_begin_scaffold,
                           format_prompt_base_bigcode,
                           format_prompt_instruct, is_instruct_model)

# MBPP 3-shot prompt ends with "[BEGIN]\n", so the model starts generating "def func(...".
# HumanEval stop strings like "\ndef " would fire immediately (prompt trailing \n + generated "def ").
# Only stop on the few-shot delimiter; extraction handles anything after the function body.
MBPP_STOP_SEQUENCES = ["\n[DONE]"]
# BigCode/InCoder zero-shot format: no format delimiters, stop on code-level boundaries.
MBPP_BIGCODE_STOP_SEQUENCES = ["\nassert", "\nclass", "\nprint", '\n"""', "\nif __name__"]
from bench.sampler_bridge import SAMPLERS, make_pless_post_temp_sampler


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pless samplers on MBPP")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--method", required=True,
                        choices=list(SAMPLERS.keys()) + ["temp", "top_p", "greedy", "beam"],
                        help="Sampling method")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples per problem")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per sample")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for logits")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, delete existing results")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--no-stop", action="store_true", help="Disable stop sequences (for debugging)")
    parser.add_argument("--mbpp-config", choices=["sanitized", "full"], default="sanitized",
                        help="MBPP dataset config: 'sanitized' (257 problems) or 'full' (500 problems)")
    parser.add_argument("--top-p", type=float, default=None, help="top_p for nucleus sampling (method=top_p)")
    parser.add_argument("--num-beams", type=int, default=None, help="Beam width (method=beam)")
    parser.add_argument("--n-shots", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Few-shot examples in base model prompt (default 3, 0=zero-shot)")
    parser.add_argument("--prompt-style", choices=["paper", "hybrid", "begin_scaffold", "bigcode"],
                        default="paper",
                        help="Prompt format for base models: 'paper' = 3-shot [BEGIN]/[DONE] "
                             "(default), 'hybrid' = scaffold no [BEGIN], 'begin_scaffold' = [BEGIN]+scaffold, "
                             "'bigcode' = zero-shot InCoder docstring (matches arXiv 2507.03160)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Only run these specific task_ids (for targeted ablations)")
    parser.add_argument("--post-temperature", type=float, default=None,
                        help="Post-truncation temperature (T₂) for p-less variants. "
                             "Applied after p-less threshold pruning to flatten survivor distribution.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16",
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None,
                        help="Attention implementation (default: auto — sdpa, eager for old Qwen)")
    args = parser.parse_args()
    if args.method == "top_p" and args.top_p is None:
        parser.error("--top-p is required when --method is top_p")
    if args.method == "beam" and args.num_beams is None:
        parser.error("--num-beams is required when --method is beam")
    if args.post_temperature is not None and args.method not in SAMPLERS:
        parser.error("--post-temperature only works with p-less methods")
    return args


def main():
    args = parse_args()

    # Output path — encode n_shots in filename when not the default (3)
    if args.method == "top_p":
        method_key = f"top_p{args.top_p}"
    elif args.method == "beam":
        method_key = f"beam{args.num_beams}"
    else:
        method_key = args.method
    if not is_instruct_model(args.model) and args.n_shots != 3:
        method_key = f"{method_key}_ns{args.n_shots}"
    if not is_instruct_model(args.model) and args.prompt_style == "hybrid":
        method_key = f"{method_key}_hybrid"
    if not is_instruct_model(args.model) and args.prompt_style == "begin_scaffold":
        method_key = f"{method_key}_bs"
    if args.post_temperature is not None:
        method_key = f"{method_key}_pt{args.post_temperature}"
    if not is_instruct_model(args.model) and args.prompt_style == "bigcode":
        method_key = f"{method_key}_bigcode"
    if args.dtype != "bfloat16":
        method_key = f"{method_key}_{args.dtype}"
    if args.attn_impl is not None:
        method_key = f"{method_key}_{args.attn_impl}"
    out_path = get_output_path(args.results_dir, args.model, method_key, args.temperature)

    # Handle --no-resume
    if args.no_resume and out_path.exists():
        out_path.unlink()

    # Load checkpoint
    completed_ids = load_completed_ids(out_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} problems already completed")

    # Load dataset
    dataset = load_dataset("google-research-datasets/mbpp", args.mbpp_config, split="test")

    # Normalize column names: full config uses 'text' instead of 'prompt',
    # 'test_setup_code' instead of 'test_imports'
    if args.mbpp_config == "full":
        dataset = dataset.map(lambda task: {"prompt": task["text"]})

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype, attn_impl=args.attn_impl)

    if args.method not in ("temp", "top_p", "greedy", "beam"):
        if args.post_temperature is not None:
            sampler_fn = make_pless_post_temp_sampler(args.post_temperature)
        else:
            sampler_fn = SAMPLERS[args.method]
    instruct = is_instruct_model(args.model)

    # Stop sequences for base models only
    stop_strings = None
    if not instruct and not args.no_stop:
        if args.prompt_style == "bigcode":
            stop_strings = MBPP_BIGCODE_STOP_SEQUENCES
        else:
            stop_strings = MBPP_STOP_SEQUENCES

    # Filter to remaining problems
    remaining = [task for task in dataset if task["task_id"] not in completed_ids]
    if args.task_ids is not None:
        remaining = [task for task in remaining if task["task_id"] in set(args.task_ids)]
    if args.max_problems is not None:
        remaining = remaining[:args.max_problems]
    print(f"Problems remaining: {len(remaining)} / {len(dataset)}")

    for task in tqdm(remaining, desc=f"{args.method} @ {args.model.split('/')[-1]}"):
        task_id = task["task_id"]
        try:
            if instruct:
                prompt_text, code_prefix = format_prompt_instruct(task, tokenizer)
            elif args.prompt_style == "hybrid":
                prompt_text, code_prefix = format_prompt_base_hybrid(task)
            elif args.prompt_style == "begin_scaffold":
                prompt_text, code_prefix = format_prompt_base_begin_scaffold(task)
            elif args.prompt_style == "bigcode":
                prompt_text, code_prefix = format_prompt_base_bigcode(task)
            else:
                prompt_text, code_prefix = format_prompt_base(task, n_shots=args.n_shots)

            if args.method == "greedy":
                raw_samples = generate_samples_greedy(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    stop_strings=stop_strings,
                )
            elif args.method == "beam":
                raw_samples = generate_samples_beam(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    stop_strings=stop_strings,
                )
            elif args.method in ("temp", "top_p"):
                raw_samples = generate_samples_standard(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    n_samples=args.n_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop_strings=stop_strings,
                    top_p=args.top_p if args.method == "top_p" else 1.0,
                )
            else:
                raw_samples = generate_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    sampler_fn=sampler_fn,
                    n_samples=args.n_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop_strings=stop_strings,
                )

            # Prepend the code prefix so each sample is complete, executable code
            samples = [code_prefix + s for s in raw_samples]

            record = {
                "model": args.model,
                "method": args.method,
                "temperature": args.temperature,
                "task_id": task_id,
                "prompt_text": task["prompt"],
                "samples": samples,
                "test_list": task["test_list"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if args.method == "top_p":
                record["top_p"] = args.top_p
            if args.method == "beam":
                record["num_beams"] = args.num_beams
            if args.post_temperature is not None:
                record["post_temperature"] = args.post_temperature
            if task.get("test_setup_code"):
                record["test_setup_code"] = task["test_setup_code"]
            append_result(out_path, record)
            tqdm.write(f"Completed task_id={task_id}")

        except Exception as e:
            tqdm.write(f"Error on task_id={task_id}: {e}")
            continue

    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
