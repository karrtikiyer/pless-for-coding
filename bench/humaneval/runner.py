import argparse
from datetime import datetime, timezone

from datasets import load_dataset
from tqdm import tqdm

from bench.checkpointing import append_result, get_output_path, load_completed_ids
from bench.generator import generate_samples, generate_samples_standard, load_model_and_tokenizer
from bench.humaneval.prompts import (
    HUMANEVAL_STOP_SEQUENCES,
    format_prompt_base,
    format_prompt_instruct,
    is_instruct_model,
)
from bench.sampler_bridge import SAMPLERS, make_pless_post_temp_sampler

METHODS = list(SAMPLERS.keys()) + ["temp", "top_p"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark samplers on HumanEval")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--method", required=True, choices=METHODS, help="Sampling method")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples per problem")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per sample")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, delete existing results")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--no-stop", action="store_true", help="Disable stop sequences (for debugging)")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Only run specific task IDs (e.g., HumanEval/74 HumanEval/59)")
    parser.add_argument("--top-p", type=float, default=None, help="top_p for nucleus sampling (method=top_p)")
    parser.add_argument("--post-temperature", type=float, default=None,
                        help="Post-truncation temperature (T₂) for p-less variants. "
                             "Applied after p-less threshold pruning to flatten survivor distribution.")
    args = parser.parse_args()
    if args.method == "top_p" and args.top_p is None:
        parser.error("--top-p is required when --method is top_p")
    if args.post_temperature is not None and args.method not in SAMPLERS:
        parser.error("--post-temperature only works with p-less methods")
    return args


def run_benchmark(
    model,
    tokenizer,
    model_id: str,
    method: str,
    temperature: float,
    n_samples: int = 10,
    max_new_tokens: int = 512,
    results_dir: str = "results",
    no_resume: bool = False,
    max_problems: int = None,
    no_stop: bool = False,
    task_ids: list[str] | None = None,
    top_p: float | None = None,
    post_temperature: float | None = None,
):
    """Run HumanEval benchmark for a single (method, temperature) config.

    Can be called directly (from orchestration script) with an already-loaded model,
    or via the CLI main() which loads the model itself.
    """
    method_key = f"top_p{top_p}" if method == "top_p" else method
    if post_temperature is not None:
        method_key = f"{method_key}_pt{post_temperature}"
    out_path = get_output_path(results_dir, model_id, method_key, temperature, benchmark="humaneval")

    if no_resume and out_path.exists():
        out_path.unlink()

    completed_ids = load_completed_ids(out_path)
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} problems already completed")

    dataset = load_dataset("openai/openai_humaneval", split="test")

    instruct = is_instruct_model(model_id)

    # Stop sequences for base models only (instruct models are constrained by chat template)
    stop_strings = None
    if not instruct and not no_stop:
        stop_strings = HUMANEVAL_STOP_SEQUENCES

    remaining = [task for task in dataset if task["task_id"] not in completed_ids]
    if task_ids is not None:
        remaining = [task for task in remaining if task["task_id"] in task_ids]
    if max_problems is not None:
        remaining = remaining[:max_problems]
    print(f"  Problems remaining: {len(remaining)} / {len(dataset)}")

    if method not in ("temp", "top_p"):
        if post_temperature is not None:
            sampler_fn = make_pless_post_temp_sampler(post_temperature)
        else:
            sampler_fn = SAMPLERS.get(method)
    else:
        sampler_fn = None

    for task in tqdm(remaining, desc=f"{method}_t{temperature}"):
        task_id = task["task_id"]
        try:
            if instruct:
                prompt_text, code_prefix = format_prompt_instruct(task, tokenizer)
            else:
                prompt_text, code_prefix = format_prompt_base(task)

            if method in ("temp", "top_p"):
                raw_samples = generate_samples_standard(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    n_samples=n_samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_strings=stop_strings,
                    top_p=top_p if method == "top_p" else 1.0,
                )
            else:
                raw_samples = generate_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    sampler_fn=sampler_fn,
                    n_samples=n_samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_strings=stop_strings,
                )

            samples = [code_prefix + s for s in raw_samples]

            record = {
                "model": model_id,
                "method": method,
                "temperature": temperature,
                "task_id": task_id,
                "prompt_text": prompt_text,
                "samples": samples,
                "test": task["test"],
                "entry_point": task["entry_point"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if method == "top_p":
                record["top_p"] = top_p
            if post_temperature is not None:
                record["post_temperature"] = post_temperature
            append_result(out_path, record)
            tqdm.write(f"  Completed task_id={task_id}")

        except Exception as e:
            tqdm.write(f"  Error on task_id={task_id}: {e}")
            continue

    print(f"  Done. Results saved to {out_path}")


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model,
        method=args.method,
        temperature=args.temperature,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        results_dir=args.results_dir,
        no_resume=args.no_resume,
        max_problems=args.max_problems,
        no_stop=args.no_stop,
        task_ids=args.task_ids,
        top_p=args.top_p,
        post_temperature=args.post_temperature,
    )


if __name__ == "__main__":
    main()
