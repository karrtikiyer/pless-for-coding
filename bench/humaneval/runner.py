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
from bench.sampler_bridge import SAMPLERS

METHODS = list(SAMPLERS.keys()) + ["temp"]


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
    return parser.parse_args()


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
):
    """Run HumanEval benchmark for a single (method, temperature) config.

    Can be called directly (from orchestration script) with an already-loaded model,
    or via the CLI main() which loads the model itself.
    """
    out_path = get_output_path(results_dir, model_id, method, temperature, benchmark="humaneval")

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

    sampler_fn = SAMPLERS.get(method) if method != "temp" else None

    for task in tqdm(remaining, desc=f"{method}_t{temperature}"):
        task_id = task["task_id"]
        try:
            if instruct:
                prompt_text, code_prefix = format_prompt_instruct(task, tokenizer)
            else:
                prompt_text, code_prefix = format_prompt_base(task)

            if method == "temp":
                raw_samples = generate_samples_standard(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    n_samples=n_samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_strings=stop_strings,
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
    )


if __name__ == "__main__":
    main()
