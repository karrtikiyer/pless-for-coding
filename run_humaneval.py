"""Orchestration script: load model once, run all 14 HumanEval configs."""

import argparse

from bench.generator import load_model_and_tokenizer
from bench.humaneval.runner import run_benchmark

CONFIGS = [
    ("temp", 0.7),
    ("temp", 1.0),
    ("pless", 0.7),
    ("pless", 1.0),
    ("pless", 1.5),
    ("pless", 2.0),
    ("pless", 2.5),
    ("pless", 3.0),
    ("pless_norm", 0.7),
    ("pless_norm", 1.0),
    ("pless_norm", 1.5),
    ("pless_norm", 2.0),
    ("pless_norm", 2.5),
    ("pless_norm", 3.0),
]


def main():
    parser = argparse.ArgumentParser(description="Run all HumanEval configs for a model")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples per problem")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per sample")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh for all configs")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems (for testing)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    for i, (method, temperature) in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Running {method} @ temperature={temperature}")
        run_benchmark(
            model=model,
            tokenizer=tokenizer,
            model_id=args.model,
            method=method,
            temperature=temperature,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            results_dir=args.results_dir,
            no_resume=args.no_resume,
            max_problems=args.max_problems,
        )

    print("\nAll configs complete!")


if __name__ == "__main__":
    main()
