"""Cross-domain entropy probe runner.

For each problem in a dataset:
  1. Format the prose-forcing prompt (or code prompt for MBPP).
  2. Generate one or more completions per problem using the requested
     sampler (multinomial T=1.0, pless, or pless_alpha at a given α).
  3. Teacher-force the (prompt + completion) tokens, capturing per-token
     entropy of the model's predictive distribution at each position.
     **Important:** entropy is computed from the model's raw softmax
     output at each position, which is the **pre-truncation**
     distribution. So the logged entropy values are a property of the
     model's belief at the visited positions, not of the sampler's
     truncation behavior. The sampler only affects *which* positions
     get visited (trajectory).
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
    p.add_argument("--n-samples", type=int, default=1,
                   help="Completions per problem. With sampler=multinomial: "
                        "N=1 uses greedy decode (deterministic), N>1 uses "
                        "multinomial at T=1.0. With sampler=pless or "
                        "pless_alpha: always stochastic (the sampler itself "
                        "decides) — N controls trajectory count.")
    p.add_argument("--sampler", choices=["multinomial", "pless", "pless_alpha"],
                   default="multinomial",
                   help="Sampling strategy for trajectory generation. The "
                        "sampler only affects which positions get visited; "
                        "the entropy logged at each position comes from the "
                        "model's pre-truncation softmax output. Use "
                        "pless_alpha to match what the production α-knob "
                        "actually does at inference time.")
    p.add_argument("--alpha", type=float, default=None,
                   help="Required when --sampler=pless_alpha. The Rényi-α "
                        "exponent in the threshold Σpᵢ^α. α=2 reproduces "
                        "plain pless. Typical values: 2.0, 2.5, 3.0, 5.0.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (default 1.0). Only used "
                        "when --sampler != multinomial-greedy.")
    p.add_argument("--output-dir", default="results/entropy_probe")
    p.add_argument("--dtype", choices=["bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run even if outputs exist.")
    args = p.parse_args(argv)
    if args.sampler == "pless_alpha" and args.alpha is None:
        p.error("--alpha is required when --sampler=pless_alpha")
    if args.sampler != "pless_alpha" and args.alpha is not None:
        p.error("--alpha is only valid with --sampler=pless_alpha")
    return args


def _load_model(model_id: str, dtype: str):
    """Load via existing function — inherits the OCI byte-level BPE fix."""
    from bench.generator import load_model_and_tokenizer
    return load_model_and_tokenizer(model_id, dtype=dtype)


def _sampler_tag(args: argparse.Namespace) -> str:
    """Subdirectory tag identifying the sampler config."""
    if args.sampler == "multinomial":
        if args.n_samples == 1:
            return f"greedy_t{args.temperature}"
        return f"multinomial_t{args.temperature}"
    if args.sampler == "pless":
        return f"pless_t{args.temperature}"
    if args.sampler == "pless_alpha":
        return f"pless_alpha_a{args.alpha}_t{args.temperature}"
    raise ValueError(f"unknown sampler {args.sampler}")


def _generate_multinomial(
    model, tokenizer, prompt_text: str, max_new_tokens: int,
    *, do_sample: bool, temperature: float,
):
    """HF .generate() path: greedy (do_sample=False) or multinomial.

    Returns (full_input_ids, prompt_len, completion_text).
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        gen = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full_ids = gen[0:1]
    completion_text = tokenizer.decode(
        gen[0, prompt_len:], skip_special_tokens=True,
    )
    return full_ids, prompt_len, completion_text


def _generate_with_sampler_fn(
    model, tokenizer, prompt_text: str, max_new_tokens: int,
    *, sampler_fn, n_samples: int, temperature: float,
):
    """Custom-sampler path (pless / pless_alpha) via bench.generator.

    Yields ``n_samples`` tuples of (full_input_ids, prompt_len, completion_text)
    using the same code path the production α-sweep runs against. The
    returned ``full_input_ids`` is the exact token sequence the sampler
    emitted (truncated at first EOS), suitable for teacher-forcing.
    """
    from bench.generator import generate_samples

    samples_text, full_ids_list, prompt_len = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        sampler_fn=sampler_fn,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_strings=None,
        return_token_ids=True,
    )
    for text, full_1d in zip(samples_text, full_ids_list):
        # teacher_forced_entropy expects a 2-D (1, seq_len) tensor.
        full_2d = full_1d.unsqueeze(0)
        yield full_2d, prompt_len, text


def run_one_problem(
    model, tokenizer,
    problem: EntropyProbeProblem, dataset: str,
    max_new_tokens: int,
    n_samples: int = 1,
    sampler: str = "multinomial",
    alpha: float | None = None,
    temperature: float = 1.0,
) -> list[dict]:
    """Generate ``n_samples`` completions + teacher-forced entropy.

    Returns a list of ``n_samples`` dicts, each with: task_id,
    sample_idx, prompt, completion, entropies_nats.
    """
    prompt_text = format_prompt(dataset, problem.problem, tokenizer)
    out: list[dict] = []

    if sampler == "multinomial":
        # HF .generate() — greedy if N=1, multinomial T=temperature if N>1.
        use_sampling = (n_samples > 1)
        for sample_idx in range(n_samples):
            full_ids, prompt_len, completion_text = _generate_multinomial(
                model, tokenizer, prompt_text, max_new_tokens,
                do_sample=use_sampling, temperature=temperature,
            )
            entropies = teacher_forced_entropy(model, full_ids, prompt_len)
            out.append({
                "task_id": problem.task_id,
                "sample_idx": sample_idx,
                "prompt": prompt_text,
                "completion": completion_text,
                "entropies_nats": entropies,
            })
        return out

    # pless / pless_alpha — go through bench.generator with the same
    # sampler_fn the production runners use.
    from bench.sampler_bridge import (
        make_guarded_pless_sampler,
        make_pless_alpha_sampler,
    )
    if sampler == "pless":
        sampler_fn = make_guarded_pless_sampler()
    elif sampler == "pless_alpha":
        sampler_fn = make_pless_alpha_sampler(alpha)
    else:
        raise ValueError(f"unknown sampler {sampler}")

    for sample_idx, (full_ids, prompt_len, completion_text) in enumerate(
        _generate_with_sampler_fn(
            model, tokenizer, prompt_text, max_new_tokens,
            sampler_fn=sampler_fn, n_samples=n_samples, temperature=temperature,
        )
    ):
        entropies = teacher_forced_entropy(model, full_ids, prompt_len)
        out.append({
            "task_id": problem.task_id,
            "sample_idx": sample_idx,
            "prompt": prompt_text,
            "completion": completion_text,
            "entropies_nats": entropies,
        })
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_slug = args.model.replace("/", "--")
    sampler_tag = _sampler_tag(args)
    out_dir = Path(args.output_dir) / model_slug / args.dataset / sampler_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "per_token_entropy.csv"
    generations_path = out_dir / "generations.jsonl"
    summary_path = out_dir / "dip_test.json"
    kde_path = out_dir / "entropy_kde.png"

    if (not args.no_resume and csv_path.exists() and summary_path.exists()
            and generations_path.exists() and kde_path.exists()):
        print(f"[skip] All outputs exist for {model_slug}/{args.dataset}/"
              f"{sampler_tag}. Pass --no-resume to force re-run.")
        return

    print(f"[entropy-probe] model={args.model}  dataset={args.dataset}  "
          f"sampler={sampler_tag}  max_problems={args.max_problems}  "
          f"n_samples={args.n_samples}")
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
        writer.writerow(["task_id", "sample_idx", "position", "entropy_nats"])
        for problem in tqdm(problems, desc=f"{args.dataset} probe",
                            file=sys.stderr):
            try:
                recs = run_one_problem(
                    model, tokenizer, problem,
                    args.dataset, args.max_new_tokens,
                    n_samples=args.n_samples,
                    sampler=args.sampler,
                    alpha=args.alpha,
                    temperature=args.temperature,
                )
            except Exception as exc:
                n_failed += 1
                print(f"  [fail] {problem.task_id}: {exc!r}", file=sys.stderr)
                continue
            for rec in recs:
                gen_f.write(json.dumps({
                    "task_id": rec["task_id"],
                    "sample_idx": rec["sample_idx"],
                    "prompt": rec["prompt"],
                    "completion": rec["completion"],
                    "n_completion_tokens": len(rec["entropies_nats"]),
                }) + "\n")
                for pos, ent in enumerate(rec["entropies_nats"]):
                    writer.writerow([
                        rec["task_id"], rec["sample_idx"], pos, f"{ent:.6f}",
                    ])
                    all_entropies.append(ent)

    summary = compute_dip_test(all_entropies)
    summary["model"] = args.model
    summary["dataset"] = args.dataset
    summary["sampler"] = args.sampler
    summary["alpha"] = args.alpha
    summary["temperature"] = args.temperature
    summary["n_samples_per_problem"] = args.n_samples
    summary["n_problems_loaded"] = len(problems)
    summary["n_problems_failed"] = n_failed
    summary["n_problems_succeeded"] = len(problems) - n_failed
    summary_path.write_text(json.dumps(summary, indent=2))

    title = (f"{args.model} on {args.dataset} ({sampler_tag})\n"
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
