import argparse
from datetime import datetime, timezone

from datasets import load_dataset
from tqdm import tqdm

from bench.checkpointing import append_result, get_output_path, load_completed_ids
from bench.generator import generate_samples, generate_samples_standard, load_model_and_tokenizer
from bench.humaneval.prompts import HUMANEVAL_STOP_SEQUENCES
from bench.prompts import format_prompt_base, format_prompt_instruct, is_instruct_model

# MBPP uses standalone functions (same stop sequences as HumanEval)
MBPP_STOP_SEQUENCES = HUMANEVAL_STOP_SEQUENCES
from bench.sampler_bridge import SAMPLERS


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pless samplers on MBPP")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--method", required=True, choices=list(SAMPLERS.keys()) + ["temp", "top_p"], help="Sampling method")
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
    args = parser.parse_args()
    if args.method == "top_p" and args.top_p is None:
        parser.error("--top-p is required when --method is top_p")
    return args


def main():
    args = parse_args()

    # Output path
    method_key = f"top_p{args.top_p}" if args.method == "top_p" else args.method
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
    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.method not in ("temp", "top_p"):
        sampler_fn = SAMPLERS[args.method]
    instruct = is_instruct_model(args.model)

    # Stop sequences for base models only
    stop_strings = None
    if not instruct and not args.no_stop:
        stop_strings = MBPP_STOP_SEQUENCES

    # Filter to remaining problems
    remaining = [task for task in dataset if task["task_id"] not in completed_ids]
    if args.max_problems is not None:
        remaining = remaining[:args.max_problems]
    print(f"Problems remaining: {len(remaining)} / {len(dataset)}")

    for task in tqdm(remaining, desc=f"{args.method} @ {args.model.split('/')[-1]}"):
        task_id = task["task_id"]
        try:
            if instruct:
                prompt_text, code_prefix = format_prompt_instruct(task, tokenizer)
            else:
                prompt_text, code_prefix = format_prompt_base(task)

            if args.method in ("temp", "top_p"):
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
