"""GSM8K evaluation CLI — reads a generations JSONL and computes pass@k +
pairwise BLEU-4 reasoning diversity.

Mirrors ``bench/eval/__main__.py`` for code, but with the GSM8K answer-
extraction and BLEU-on-reasoning instead of code execution and CodeBLEU.

CLI:
    uv run python -m bench.gsm8k.eval_runner \\
        --results-file results/pless_alpha_full_gsm8k/Qwen--Qwen2.5-Coder-7B-Instruct/pless_alpha_a5.0_t1.0.jsonl

Output: a sibling metrics JSON at
    <results_dir>/metrics/<results_file_stem>_metrics.json
with:
  * pass_at_k: {1, 3, 5, 10} (computed via the existing Chen 2021 estimator)
  * self_bleu_diversity: mean across tasks of pairwise-BLEU-4 diversity on
    correct-and-deduplicated samples
  * per_task: per-problem pass count + per-task diversity (when ≥2 unique
    correct samples)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.eval.metrics import compute_pass_at_k
from bench.gsm8k.diversity import (
    compute_aggregate_diversity,
    pairwise_bleu4_diversity,
)
from bench.gsm8k.evaluator import extract_predicted_answer, numeric_equals


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _evaluate_records(records: list[dict]) -> dict:
    per_task: list[dict] = []
    for rec in records:
        gold = rec.get("gold_answer", "")
        samples = rec.get("samples", [])
        pass_results: list[bool] = []
        correct_samples: list[str] = []
        for sample in samples:
            pred = extract_predicted_answer(sample)
            ok = numeric_equals(pred, gold)
            pass_results.append(bool(ok))
            if ok:
                correct_samples.append(sample)
        n_correct = sum(pass_results)
        diversity = pairwise_bleu4_diversity(correct_samples)
        per_task.append({
            "task_id": rec["task_id"],
            "num_correct": n_correct,
            "num_samples": len(samples),
            "pass_results": pass_results,
            "self_bleu_diversity": diversity,
        })

    pass_at_k = compute_pass_at_k(per_task, [1, 3, 5, 10])
    div_agg = compute_aggregate_diversity(per_task)
    return {
        "n_tasks": len(per_task),
        "samples_per_task": records[0]["samples"].__len__() if records else 0,
        "pass_at_k": pass_at_k,
        **div_agg,
        "per_task": per_task,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-file", type=Path, required=True,
                   help="JSONL file produced by bench.gsm8k")
    p.add_argument("--output", type=Path, default=None,
                   help="Destination for the metrics JSON (default: "
                        "sibling metrics/ under the results file's parent)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = _load_jsonl(args.results_file)
    if not records:
        raise SystemExit(f"No records in {args.results_file}")
    metrics = _evaluate_records(records)
    metrics["source"] = str(args.results_file)
    metrics["model"] = records[0].get("model")
    metrics["method"] = records[0].get("method")
    metrics["alpha"] = records[0].get("alpha")
    metrics["temperature"] = records[0].get("temperature")

    if args.output is None:
        out_dir = args.results_file.parent / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / (args.results_file.stem + "_metrics.json")
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {args.output}")
    print(f"  pass@1 = {metrics['pass_at_k'].get('1', 'n/a'):.4f}  "
          f"pass@10 = {metrics['pass_at_k'].get('10', 'n/a'):.4f}  "
          f"self_bleu_div = {metrics.get('self_bleu_diversity')}  "
          f"(over {metrics.get('n_tasks_with_diversity')} tasks with ≥2 unique correct)")


if __name__ == "__main__":
    main()
