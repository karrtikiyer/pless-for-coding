"""Curate interesting HumanEval examples where p-less sampling notably outperforms or underperforms baselines.

Loads full-precision HumanEval results (4 models × 6 methods), identifies tasks with
large p-less advantage/disadvantage, analyzes code-length and diversity patterns,
and produces a readable markdown report.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from bench.eval.parse_humaneval import parse_detailed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLESS_METHODS = ["p_less", "p_less_norm"]
BASELINE_METHODS = ["greedy", "temp_0.2", "temp_0.7", "top_p_0.95"]
ALL_METHODS = BASELINE_METHODS + PLESS_METHODS
NUM_TASKS = 164
NUM_SAMPLES = 10

MODEL_DIRS = [
    "CodeLlama-7B",
    "Codestral-22B",
    "Qwen2.5-Coder-7B",
    "Qwen3-Coder-30B",
]


# ---------------------------------------------------------------------------
# Step 1: Data loading
# ---------------------------------------------------------------------------


def load_all_data(results_dir: Path) -> dict:
    """Load detailed JSON and per-method metrics for every model.

    Returns:
        {model: {"detailed": dict, "metrics": {method: metrics_dict}}}
    """
    all_data = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        # Skip non-model dirs like 'figures'
        metrics_dir = model_dir / "metrics"
        if not metrics_dir.is_dir():
            continue

        model = model_dir.name

        # Load detailed JSON (there should be exactly one *_detailed.json)
        detailed_files = list(model_dir.glob("*_detailed.json"))
        if not detailed_files:
            continue
        with open(detailed_files[0]) as f:
            detailed = json.load(f)

        # Load metrics
        metrics = {}
        for mf in sorted(metrics_dir.glob("*_metrics.json")):
            method = mf.stem.replace("_metrics", "")
            with open(mf) as f:
                metrics[method] = json.load(f)

        all_data[model] = {"detailed": detailed, "metrics": metrics}

    return all_data


def build_task_matrix(all_data: dict) -> list[dict]:
    """Build a per-task matrix with scores, diversity, and p-less advantage.

    Returns a list of dicts, one per task_id, sorted by task index.
    """
    # Collect all task_ids from the first model's first method metrics
    first_model = next(iter(all_data.values()))
    first_method = next(iter(first_model["metrics"].values()))
    task_ids = [t["task_id"] for t in first_method["per_task"]]

    models = list(all_data.keys())

    rows = []
    for idx, task_id in enumerate(task_ids):
        scores = {}  # {model: {method: num_correct}}
        distinct = {}  # {model: {method: num_distinct_correct}}

        for model in models:
            scores[model] = {}
            distinct[model] = {}
            for method, mdata in all_data[model]["metrics"].items():
                per_task = mdata["per_task"]
                if idx < len(per_task):
                    task_entry = per_task[idx]
                    scores[model][method] = task_entry["num_correct"]
                    distinct[model][method] = task_entry.get("num_distinct_correct", 0)

        # Compute p-less advantage: mean across models of
        # max(p_less, p_less_norm) - max(baselines)
        advantages = []
        for model in models:
            ms = scores[model]
            pless_best = max(
                ms.get("p_less", 0),
                ms.get("p_less_norm", 0),
            )
            baseline_best = max(
                ms.get(b, 0) for b in BASELINE_METHODS if b in ms
            )
            advantages.append(pless_best - baseline_best)

        pless_advantage = mean(advantages) if advantages else 0.0

        # Difficulty based on mean best-baseline score across models
        baseline_bests = []
        for model in models:
            ms = scores[model]
            bb = max((ms.get(b, 0) for b in BASELINE_METHODS if b in ms), default=0)
            baseline_bests.append(bb)
        mean_baseline_best = mean(baseline_bests) if baseline_bests else 0

        if mean_baseline_best >= 8:
            difficulty = "easy"
        elif mean_baseline_best >= 3:
            difficulty = "medium"
        else:
            difficulty = "hard"

        rows.append({
            "task_id": task_id,
            "scores": scores,
            "distinct": distinct,
            "pless_advantage": pless_advantage,
            "difficulty": difficulty,
        })

    return rows


# ---------------------------------------------------------------------------
# Step 2: Selection
# ---------------------------------------------------------------------------


def select_examples(task_matrix: list[dict], num_examples: int = 7) -> dict:
    """Select top wins and losses, filtering out ceiling/floor effects.

    Returns {"wins": [...], "losses": [...]}.
    """
    filtered = []
    for row in task_matrix:
        # Check for ceiling (all 10/10) or floor (all 0/10) across all models & methods
        all_scores = []
        for model_scores in row["scores"].values():
            all_scores.extend(model_scores.values())

        if all_scores and all(s == NUM_SAMPLES for s in all_scores):
            continue  # ceiling
        if all_scores and all(s == 0 for s in all_scores):
            continue  # floor

        filtered.append(row)

    # Sort by pless_advantage
    sorted_by_adv = sorted(filtered, key=lambda r: r["pless_advantage"], reverse=True)

    wins = sorted_by_adv[:num_examples]
    losses = sorted_by_adv[-num_examples:]
    losses.reverse()  # most negative first

    return {"wins": wins, "losses": losses}


# ---------------------------------------------------------------------------
# Step 3: Analyses
# ---------------------------------------------------------------------------


def _partition_tasks(task_matrix: list[dict]) -> dict[str, list[dict]]:
    """Partition tasks into win/loss/tie based on pless_advantage."""
    partitions = {"win": [], "loss": [], "tie": []}
    for row in task_matrix:
        adv = row["pless_advantage"]
        if adv > 0.5:
            partitions["win"].append(row)
        elif adv < -0.5:
            partitions["loss"].append(row)
        else:
            partitions["tie"].append(row)
    return partitions


def analyze_code_length(task_matrix: list[dict], all_data: dict) -> dict:
    """Compute mean tokens_generated and code length per method for win/loss/tie partitions.

    Returns: {partition: {method: {"mean_tokens": float, "mean_code_len": float}}}
    """
    partitions = _partition_tasks(task_matrix)
    task_id_to_partition = {}
    for label, rows in partitions.items():
        for row in rows:
            task_id_to_partition[row["task_id"]] = label

    # Accumulate stats: {partition: {method: {"tokens": [], "code_len": []}}}
    stats = {
        label: {m: {"tokens": [], "code_len": []} for m in ALL_METHODS}
        for label in partitions
    }

    for model, mdata in all_data.items():
        detailed = mdata["detailed"]
        for method in ALL_METHODS:
            if method not in detailed:
                continue
            for task_entry in detailed[method]:
                tid = task_entry["task_id"]
                label = task_id_to_partition.get(tid)
                if label is None:
                    continue
                for sample in task_entry["samples"]:
                    tokens = sample.get("tokens_generated", 0)
                    code_len = len(sample.get("code", ""))
                    stats[label][method]["tokens"].append(tokens)
                    stats[label][method]["code_len"].append(code_len)

    result = {}
    for label in partitions:
        result[label] = {}
        for method in ALL_METHODS:
            toks = stats[label][method]["tokens"]
            lens = stats[label][method]["code_len"]
            result[label][method] = {
                "mean_tokens": mean(toks) if toks else 0.0,
                "mean_code_len": mean(lens) if lens else 0.0,
            }

    return result


def analyze_diversity(task_matrix: list[dict]) -> dict:
    """Compute diversity ratio (num_distinct_correct / num_correct) per method.

    Returns: {method: {"mean_diversity_ratio": float, "task_count": int}}
    """
    # Accumulate ratios per method
    ratios = {m: [] for m in ALL_METHODS}

    for row in task_matrix:
        for model, model_scores in row["scores"].items():
            model_distinct = row["distinct"].get(model, {})
            for method in ALL_METHODS:
                nc = model_scores.get(method, 0)
                nd = model_distinct.get(method, 0)
                if nc > 0:
                    ratios[method].append(nd / nc)

    result = {}
    for method in ALL_METHODS:
        vals = ratios[method]
        result[method] = {
            "mean_diversity_ratio": mean(vals) if vals else 0.0,
            "task_count": len(vals),
        }

    return result


def analyze_by_difficulty(task_matrix: list[dict]) -> dict:
    """Count win/loss/tie per difficulty bucket per model.

    Returns: {difficulty: {model: {"win": int, "loss": int, "tie": int}}}
    """
    models = set()
    for row in task_matrix:
        models.update(row["scores"].keys())
    models = sorted(models)

    result = {}
    for diff in ["easy", "medium", "hard"]:
        result[diff] = {m: {"win": 0, "loss": 0, "tie": 0} for m in models}

    for row in task_matrix:
        diff = row["difficulty"]
        for model in models:
            ms = row["scores"].get(model, {})
            pless_best = max(ms.get("p_less", 0), ms.get("p_less_norm", 0))
            baseline_best = max((ms.get(b, 0) for b in BASELINE_METHODS if b in ms), default=0)
            delta = pless_best - baseline_best
            if delta > 0:
                result[diff][model]["win"] += 1
            elif delta < 0:
                result[diff][model]["loss"] += 1
            else:
                result[diff][model]["tie"] += 1

    return result


def analyze_per_model(task_matrix: list[dict]) -> dict:
    """Flag per-model anomalies, e.g. p_less vs p_less_norm divergence.

    Returns: {model: {"p_less_mean": float, "p_less_norm_mean": float, "divergence": float, ...}}
    """
    models = set()
    for row in task_matrix:
        models.update(row["scores"].keys())
    models = sorted(models)

    result = {}
    for model in models:
        pless_scores = []
        pless_norm_scores = []
        for row in task_matrix:
            ms = row["scores"].get(model, {})
            pless_scores.append(ms.get("p_less", 0))
            pless_norm_scores.append(ms.get("p_less_norm", 0))

        pm = mean(pless_scores) if pless_scores else 0
        pnm = mean(pless_norm_scores) if pless_norm_scores else 0

        # Count per-model wins/losses
        wins = 0
        losses = 0
        for row in task_matrix:
            ms = row["scores"].get(model, {})
            pless_best = max(ms.get("p_less", 0), ms.get("p_less_norm", 0))
            baseline_best = max((ms.get(b, 0) for b in BASELINE_METHODS if b in ms), default=0)
            if pless_best > baseline_best:
                wins += 1
            elif pless_best < baseline_best:
                losses += 1

        result[model] = {
            "p_less_mean": pm,
            "p_less_norm_mean": pnm,
            "divergence": abs(pm - pnm),
            "wins": wins,
            "losses": losses,
        }

    return result


# ---------------------------------------------------------------------------
# Step 4: Report rendering
# ---------------------------------------------------------------------------


def _format_score_table(row: dict) -> str:
    """Render a correctness table (model × method → num_correct) for one task."""
    models = sorted(row["scores"].keys())
    methods = ALL_METHODS

    header = "| Model | " + " | ".join(methods) + " |"
    sep = "|" + "---|" * (len(methods) + 1)
    lines = [header, sep]
    for model in models:
        ms = row["scores"][model]
        cells = [str(ms.get(m, "-")) for m in methods]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pick_dramatic_model(row: dict) -> str:
    """Pick the model with the largest absolute p-less vs baseline delta."""
    best_model = None
    best_delta = -1
    for model, ms in row["scores"].items():
        pless_best = max(ms.get("p_less", 0), ms.get("p_less_norm", 0))
        baseline_best = max((ms.get(b, 0) for b in BASELINE_METHODS if b in ms), default=0)
        delta = abs(pless_best - baseline_best)
        if delta > best_delta:
            best_delta = delta
            best_model = model
    return best_model


def _get_sample_code(all_data: dict, model: str, method: str, task_id: str, passed: bool) -> str | None:
    """Get a code sample from a specific model/method/task, preferring passed or failed."""
    detailed = all_data.get(model, {}).get("detailed", {})
    if method not in detailed:
        return None
    for task_entry in detailed[method]:
        if task_entry["task_id"] == task_id:
            for sample in task_entry["samples"]:
                if sample["passed"] == passed:
                    return sample["code"]
            # Fall back to any sample
            if task_entry["samples"]:
                return task_entry["samples"][0]["code"]
    return None


def _render_example(row: dict, all_data: dict, is_win: bool) -> str:
    """Render a single curated example section."""
    lines = []
    task_id = row["task_id"]
    lines.append(f"### {task_id}")
    lines.append("")
    lines.append(f"**P-less advantage:** {row['pless_advantage']:+.2f} | **Difficulty:** {row['difficulty']}")
    lines.append("")
    lines.append(_format_score_table(row))
    lines.append("")

    model = _pick_dramatic_model(row)
    if model and all_data:
        ms = row["scores"][model]
        # Find best pless method and worst baseline (for wins) or vice versa
        if is_win:
            pless_method = max(PLESS_METHODS, key=lambda m: ms.get(m, 0))
            baseline_method = min(BASELINE_METHODS, key=lambda m: ms.get(m, NUM_SAMPLES))
            pless_code = _get_sample_code(all_data, model, pless_method, task_id, passed=True)
            baseline_code = _get_sample_code(all_data, model, baseline_method, task_id, passed=False)
            lines.append(f"**Most dramatic model:** {model} — "
                         f"`{pless_method}` ({ms.get(pless_method, '?')}/10) vs "
                         f"`{baseline_method}` ({ms.get(baseline_method, '?')}/10)")
        else:
            baseline_method = max(BASELINE_METHODS, key=lambda m: ms.get(m, 0))
            pless_method = min(PLESS_METHODS, key=lambda m: ms.get(m, NUM_SAMPLES))
            pless_code = _get_sample_code(all_data, model, pless_method, task_id, passed=False)
            baseline_code = _get_sample_code(all_data, model, baseline_method, task_id, passed=True)
            lines.append(f"**Most dramatic model:** {model} — "
                         f"`{baseline_method}` ({ms.get(baseline_method, '?')}/10) vs "
                         f"`{pless_method}` ({ms.get(pless_method, '?')}/10)")

        lines.append("")

        if pless_code:
            label = "P-less (passed)" if is_win else "P-less (failed)"
            lines.append(f"<details><summary>{label} — <code>{pless_method}</code></summary>")
            lines.append("")
            lines.append("```python")
            lines.append(pless_code.rstrip())
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        if baseline_code:
            label = "Baseline (failed)" if is_win else "Baseline (passed)"
            lines.append(f"<details><summary>{label} — <code>{baseline_method}</code></summary>")
            lines.append("")
            lines.append("```python")
            lines.append(baseline_code.rstrip())
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def render_report(
    task_matrix: list[dict],
    examples: dict,
    code_length: dict,
    diversity: dict,
    difficulty: dict,
    per_model: dict,
    all_data: dict,
) -> str:
    """Render the full markdown report."""
    lines = []

    # Count overall wins/losses/ties
    partitions = _partition_tasks(task_matrix)
    n_wins = len(partitions["win"])
    n_losses = len(partitions["loss"])
    n_ties = len(partitions["tie"])

    models = sorted(per_model.keys())

    # ---- Header ----
    lines.append("# P-Less Sampling: Curated HumanEval Examples")
    lines.append("")

    # ---- Executive Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **{n_wins}** tasks where p-less notably outperforms baselines (advantage > 0.5)")
    lines.append(f"- **{n_losses}** tasks where baselines notably outperform p-less (advantage < -0.5)")
    lines.append(f"- **{n_ties}** tasks with comparable performance (|advantage| ≤ 0.5)")
    lines.append(f"- Analyzed across **{len(models)} models**: {', '.join(models)}")

    # Key findings from diversity
    best_div_method = max(diversity, key=lambda m: diversity[m]["mean_diversity_ratio"])
    lines.append(f"- Highest mean diversity ratio: **{best_div_method}** "
                 f"({diversity[best_div_method]['mean_diversity_ratio']:.3f})")

    # Per-model summary
    for model in models:
        pm = per_model[model]
        lines.append(f"- **{model}**: {pm['wins']} wins, {pm['losses']} losses, "
                     f"p_less mean={pm['p_less_mean']:.2f}, p_less_norm mean={pm['p_less_norm_mean']:.2f}")
    lines.append("")

    # ---- Methodology ----
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Data**: 164 HumanEval tasks × 10 samples × 6 methods × 4 models")
    lines.append("- **P-less advantage**: mean across models of max(p_less, p_less_norm) − max(baselines)")
    lines.append("- **Baselines**: greedy, temp_0.2, temp_0.7, top_p_0.95")
    lines.append("- **Difficulty**: easy (best baseline ≥ 8/10), medium (3–7/10), hard (≤ 2/10)")
    lines.append("- Tasks where all methods score 10/10 or 0/10 are excluded from example selection")
    lines.append("")

    # ---- Aggregate Statistics ----
    lines.append("## Aggregate Statistics")
    lines.append("")

    lines.append("### P-less advantage distribution")
    lines.append("")
    lines.append("| Model | Wins | Losses | Ties |")
    lines.append("|---|---|---|---|")
    for model in models:
        pm = per_model[model]
        ties = NUM_TASKS - pm["wins"] - pm["losses"]
        lines.append(f"| {model} | {pm['wins']} | {pm['losses']} | {ties} |")
    lines.append("")

    lines.append("### By task difficulty")
    lines.append("")
    header = "| Difficulty | " + " | ".join(f"{m} W/L/T" for m in models) + " |"
    sep = "|" + "---|" * (len(models) + 1)
    lines.append(header)
    lines.append(sep)
    for diff in ["easy", "medium", "hard"]:
        cells = []
        for model in models:
            d = difficulty[diff][model]
            cells.append(f"{d['win']}/{d['loss']}/{d['tie']}")
        lines.append(f"| {diff.capitalize()} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Top P-Less Wins ----
    lines.append(f"## Top P-Less Wins ({len(examples['wins'])} examples)")
    lines.append("")
    for row in examples["wins"]:
        lines.append(_render_example(row, all_data, is_win=True))
        lines.append("")

    # ---- Top P-Less Losses ----
    lines.append(f"## Top P-Less Losses ({len(examples['losses'])} examples)")
    lines.append("")
    for row in examples["losses"]:
        lines.append(_render_example(row, all_data, is_win=False))
        lines.append("")

    # ---- Code Length Analysis ----
    lines.append("## Code Length Analysis")
    lines.append("")
    lines.append("Mean tokens generated per method, grouped by p-less outcome:")
    lines.append("")
    header = "| Partition | " + " | ".join(ALL_METHODS) + " |"
    sep = "|" + "---|" * (len(ALL_METHODS) + 1)
    lines.append(header)
    lines.append(sep)
    for label in ["win", "loss", "tie"]:
        if label in code_length:
            cells = [f"{code_length[label][m]['mean_tokens']:.1f}" for m in ALL_METHODS]
            lines.append(f"| {label.capitalize()} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean code length (characters):")
    lines.append("")
    lines.append(header.replace("Partition", "Partition (chars)"))
    lines.append(sep)
    for label in ["win", "loss", "tie"]:
        if label in code_length:
            cells = [f"{code_length[label][m]['mean_code_len']:.0f}" for m in ALL_METHODS]
            lines.append(f"| {label.capitalize()} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Diversity Analysis ----
    lines.append("## Diversity Analysis")
    lines.append("")
    lines.append("Mean diversity ratio (num_distinct_correct / num_correct) per method:")
    lines.append("")
    lines.append("| Method | Mean Diversity Ratio | Tasks with ≥1 correct |")
    lines.append("|---|---|---|")
    for method in ALL_METHODS:
        d = diversity[method]
        lines.append(f"| {method} | {d['mean_diversity_ratio']:.4f} | {d['task_count']} |")
    lines.append("")

    # ---- Per-Model Notes ----
    lines.append("## Per-Model Notes")
    lines.append("")
    for model in models:
        pm = per_model[model]
        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"- p_less mean score: {pm['p_less_mean']:.2f}/10")
        lines.append(f"- p_less_norm mean score: {pm['p_less_norm_mean']:.2f}/10")
        lines.append(f"- Divergence (|p_less − p_less_norm|): {pm['divergence']:.2f}")
        lines.append(f"- Wins: {pm['wins']}, Losses: {pm['losses']}")
        if pm["divergence"] > 1.0:
            lines.append(f"- ⚠ **Notable divergence** between p_less and p_less_norm")
        lines.append("")

    # ---- Appendix ----
    lines.append("## Appendix: Full Task Ranking")
    lines.append("")
    lines.append("All 164 tasks sorted by p-less advantage:")
    lines.append("")
    lines.append("| Rank | Task ID | P-less Advantage | Difficulty |")
    lines.append("|---|---|---|---|")
    sorted_tasks = sorted(task_matrix, key=lambda r: r["pless_advantage"], reverse=True)
    for i, row in enumerate(sorted_tasks, 1):
        lines.append(f"| {i} | {row['task_id']} | {row['pless_advantage']:+.2f} | {row['difficulty']} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5: CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curate interesting HumanEval examples: p-less vs baselines"
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Path to full_precision_results/ directory",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output markdown file (default: curated_examples.md in results-dir)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=7,
        help="Number of examples per category (default: 7)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = args.results_dir
    output = args.output or (results_dir / "curated_examples.md")
    num_examples = args.num_examples

    print(f"Loading data from {results_dir}")
    all_data = load_all_data(results_dir)
    print(f"Loaded {len(all_data)} models: {list(all_data.keys())}")

    print("Building task matrix...")
    task_matrix = build_task_matrix(all_data)
    print(f"Built matrix for {len(task_matrix)} tasks")

    print("Selecting examples...")
    examples = select_examples(task_matrix, num_examples=num_examples)
    print(f"Selected {len(examples['wins'])} wins, {len(examples['losses'])} losses")

    print("Analyzing code length...")
    code_length = analyze_code_length(task_matrix, all_data)

    print("Analyzing diversity...")
    diversity = analyze_diversity(task_matrix)

    print("Analyzing by difficulty...")
    difficulty_stats = analyze_by_difficulty(task_matrix)

    print("Analyzing per-model patterns...")
    per_model_stats = analyze_per_model(task_matrix)

    print("Rendering report...")
    report = render_report(
        task_matrix=task_matrix,
        examples=examples,
        code_length=code_length,
        diversity=diversity,
        difficulty=difficulty_stats,
        per_model=per_model_stats,
        all_data=all_data,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
