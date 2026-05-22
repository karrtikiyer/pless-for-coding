"""Cross-domain entropy probe runner.

For each problem in a dataset:
  1. Format the prose-forcing prompt (or code prompt for MBPP).
  2. Greedy-generate one completion (temperature 0).
  3. Teacher-force the (prompt + completion) tokens, capturing per-token
     entropy of the model's predictive distribution at each position.
  4. Save: a CSV of per-token entropies, a JSONL of the full generations
     for later inspection, a dip-test summary, and a KDE plot.

This is a research-only utility: it does not modify any code-generation
runner, does not register any new sampling method, and lives in its own
module so it cannot regress the production pipelines for MBPP /
HumanEval / APPS.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from bench.entropy_probe.analysis import compute_dip_test, plot_entropy_kde
from bench.entropy_probe.datasets import DATASETS, EntropyProbeProblem
from bench.entropy_probe.entropy import teacher_forced_entropy
from bench.entropy_probe.prompts import format_prompt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--max-problems", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output-dir", default="results/entropy_probe")
    p.add_argument("--dtype", choices=["bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run even if outputs exist.")
    return p.parse_args(argv)


def _load_model(model_id: str, dtype: str):
    """Load via existing function — inherits the OCI byte-level BPE fix."""
    from bench.generator import load_model_and_tokenizer
    return load_model_and_tokenizer(model_id, dtype=dtype)


def _generate_greedy(model, tokenizer, prompt_text: str, max_new_tokens: int):
    """One greedy completion. Returns (full_input_ids, prompt_len, completion_text)."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        gen = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full_ids = gen[0:1]
    completion_text = tokenizer.decode(
        gen[0, prompt_len:], skip_special_tokens=True,
    )
    return full_ids, prompt_len, completion_text


def run_one_problem(
    model, tokenizer,
    problem: EntropyProbeProblem, dataset: str,
    max_new_tokens: int,
) -> dict:
    """Greedy decode + teacher-forced entropy for one problem.

    Returns a dict with: task_id, prompt, completion, entropies (list[float]).
    Raises if the prompt is empty or generation fails — caller is
    expected to catch and continue.
    """
    prompt_text = format_prompt(dataset, problem.problem, tokenizer)
    full_ids, prompt_len, completion_text = _generate_greedy(
        model, tokenizer, prompt_text, max_new_tokens,
    )
    entropies = teacher_forced_entropy(model, full_ids, prompt_len)
    return {
        "task_id": problem.task_id,
        "prompt": prompt_text,
        "completion": completion_text,
        "entropies_nats": entropies,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_slug = args.model.replace("/", "--")
    out_dir = Path(args.output_dir) / model_slug / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "per_token_entropy.csv"
    generations_path = out_dir / "generations.jsonl"
    summary_path = out_dir / "dip_test.json"
    kde_path = out_dir / "entropy_kde.png"

    if (not args.no_resume and csv_path.exists() and summary_path.exists()
            and generations_path.exists() and kde_path.exists()):
        print(f"[skip] All outputs exist for {model_slug}/{args.dataset}. "
              f"Pass --no-resume to force re-run.")
        return

    print(f"[entropy-probe] model={args.model}  dataset={args.dataset}  "
          f"max_problems={args.max_problems}")
    print(f"[entropy-probe] output_dir={out_dir}")

    problems = DATASETS[args.dataset](max_problems=args.max_problems)
    print(f"[entropy-probe] loaded {len(problems)} problems")

    print(f"[entropy-probe] loading model {args.model} (dtype={args.dtype}) ...")
    model, tokenizer = _load_model(args.model, args.dtype)
    print(f"[entropy-probe] model loaded on device {model.device}")

    all_entropies: list[float] = []
    n_failed = 0
    with csv_path.open("w", newline="") as csv_f, \
         generations_path.open("w") as gen_f:
        writer = csv.writer(csv_f)
        writer.writerow(["task_id", "position", "entropy_nats"])
        for problem in tqdm(problems, desc=f"{args.dataset} probe",
                            file=sys.stderr):
            try:
                rec = run_one_problem(
                    model, tokenizer, problem,
                    args.dataset, args.max_new_tokens,
                )
            except Exception as exc:
                n_failed += 1
                print(f"  [fail] {problem.task_id}: {exc!r}", file=sys.stderr)
                continue
            gen_f.write(json.dumps({
                "task_id": rec["task_id"],
                "prompt": rec["prompt"],
                "completion": rec["completion"],
                "n_completion_tokens": len(rec["entropies_nats"]),
            }) + "\n")
            for pos, ent in enumerate(rec["entropies_nats"]):
                writer.writerow([rec["task_id"], pos, f"{ent:.6f}"])
                all_entropies.append(ent)

    summary = compute_dip_test(all_entropies)
    summary["model"] = args.model
    summary["dataset"] = args.dataset
    summary["n_problems_loaded"] = len(problems)
    summary["n_problems_failed"] = n_failed
    summary["n_problems_succeeded"] = len(problems) - n_failed
    summary_path.write_text(json.dumps(summary, indent=2))

    title = (f"{args.model} on {args.dataset}\n"
             f"n_tokens={len(all_entropies)}  "
             f"dip_p={summary.get('p_value', 'NA')}  "
             f"{summary.get('interpretation', 'NA')}")
    plot_entropy_kde(all_entropies, kde_path, title=title)

    print(f"\n[entropy-probe] done.")
    print(f"  problems succeeded: {summary['n_problems_succeeded']}/{len(problems)}")
    print(f"  tokens collected:   {len(all_entropies)}")
    if "p_value" in summary:
        print(f"  Hartigan dip stat:  {summary['dip_statistic']:.4f}")
        print(f"  p-value:            {summary['p_value']:.4g}")
        print(f"  interpretation:     {summary['interpretation']}")
    print(f"  outputs in:         {out_dir}/")
