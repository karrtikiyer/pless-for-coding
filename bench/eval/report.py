"""Generate a markdown results table from metrics JSON files."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate markdown results table from metrics JSON files"
    )
    parser.add_argument(
        "metrics_files", nargs="+", type=Path,
        help="Paths to metrics JSON files",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output markdown file path (default: print to stdout)",
    )
    return parser.parse_args()


def load_metrics(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def generate_table(metrics_list: list[dict]) -> str:
    if not metrics_list:
        return ""

    # Collect k and t values across all files
    k_values = sorted({int(k) for m in metrics_list for k in m["pass_at_k"]})
    t_values = sorted({float(t) for m in metrics_list for t in m["cover_at_t"]})

    # Build header
    pass_cols = [f"pass@{k}" for k in k_values]
    cover_cols = []
    for t in t_values:
        cover_cols.append(f"cover@{t}")
        cover_cols.append(f"cover@{t} (distinct)")

    header = ["Model", "Method"] + pass_cols + cover_cols
    separator = ["---"] * len(header)

    # Build rows
    rows = []
    for m in metrics_list:
        model = m["model"].split("/")[-1] if "/" in m["model"] else m["model"]
        method = f"{m['method']} (t={m['temperature']})"
        row = [model, method]

        for k in k_values:
            val = m["pass_at_k"].get(str(k))
            row.append(f"{val * 100:.1f}" if val is not None else "-")

        for t in t_values:
            ts = str(t)
            val = m["cover_at_t"].get(ts)
            row.append(f"{val:.1f}" if val is not None else "-")
            val_d = m["cover_at_t_distinct"].get(ts)
            row.append(f"{val_d:.1f}" if val_d is not None else "-")

        rows.append(row)

    # Format as markdown
    lines = []
    dataset = metrics_list[0].get("dataset", "").upper()
    n_tasks = metrics_list[0].get("num_tasks", "?")
    n_samples = metrics_list[0].get("num_samples_per_task", "?")
    lines.append(f"## {dataset} Results ({n_tasks} tasks, {n_samples} samples/task)\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("*pass@k values are percentages. cover@t shows % of tasks where "
                 "the fraction of correct samples >= t.*")

    return "\n".join(lines)


def main():
    args = parse_args()
    metrics_list = load_metrics(args.metrics_files)
    table = generate_table(metrics_list)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Report written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
