"""Generate paper tables from results/analysis/consolidated_summary.csv.

No hand-computed numbers. Everything that ends up in the paper tables comes
from a CSV row or a per-model report. Run from repo root:

    uv run python paper/tables/make_tables.py

Outputs (overwrites):
- paper/tables/table1_models.md
- paper/tables/table2_headline_passk.md

Tables 3 and 4 (T1/T2 grid, cross-benchmark replication) are produced from
markdown reports in Phase 2 of the draft plan; this script does not touch them.
"""
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "results" / "analysis" / "consolidated_summary.csv"
OUT_DIR = REPO / "paper" / "tables"


def load_rows() -> list[dict]:
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def canon_model(name: str) -> str:
    """Treat HF '/' and directory '--' separators as the same model."""
    return name.replace("--", "/")


def canon_method(method: str, temperature: str) -> str | None:
    """Collapse alias spellings; return None for methods we exclude."""
    m = method.lower()
    if m in {"pless", "p_less"}:
        return f"pless@{temperature}"
    if m in {"pless_norm", "p_less_norm"}:
        return f"pless_norm@{temperature}"
    if m == "temp" or m.startswith("temp_"):
        return f"temp@{temperature}"
    if m in {"top_p", "top_p0.9", "top_p_0.95"}:
        return f"top_p@{temperature}"
    if m == "top_k":
        return f"top_k@{temperature}"
    if m == "greedy":
        return "greedy"
    if m == "beam4":
        return "beam4"
    if m == "beam8":
        return "beam8"
    return None


def filter_canonical(rows: list[dict]) -> list[dict]:
    """Keep only canonical task-count cohorts.

    Per the project memory rule (`feedback_full500_only.md`), MBPP-257 rows
    are excluded — only MBPP-full (500/499 tasks) is retained. HumanEval is
    164 tasks. Greedy/beam baselines have num_samples_per_task=1; stochastic
    methods have 10.
    """
    out = []
    for r in rows:
        n_tasks = int(r["num_tasks"])
        n_samples = int(r["num_samples_per_task"])
        if r["dataset"] == "mbpp":
            if n_tasks not in (499, 500):
                continue
        elif r["dataset"] == "humaneval":
            if n_tasks != 164:
                continue
        else:
            continue
        if n_samples not in (1, 10):
            continue
        out.append(r)
    return out


def write_table1_models(rows: list[dict]) -> str:
    """One row per unique (canonical) model, listing benchmarks it appears in."""
    models = defaultdict(set)
    for r in rows:
        models[canon_model(r["model"])].add(r["dataset"])

    lines = [
        "# Table 1 — Models",
        "",
        "Models evaluated in this paper. \"Benchmarks\" lists which benchmarks"
        " a model was run against in this study.",
        "",
        "| Model | MBPP-500 | HumanEval-164 |",
        "|-------|:--------:|:-------------:|",
    ]
    for m in sorted(models):
        lines.append(
            f"| {m} "
            f"| {'✓' if 'mbpp' in models[m] else ''} "
            f"| {'✓' if 'humaneval' in models[m] else ''} |"
        )
    lines.append("")
    lines.append(
        f"_Source: `results/analysis/consolidated_summary.csv` "
        f"({len(rows)} rows after filtering to MBPP-500 / HumanEval-164)._"
    )
    return "\n".join(lines) + "\n"


def write_table2_headline(rows: list[dict]) -> str:
    """Pivot pass@1, pass@10 by (model, dataset) × canonical-method.

    Methods reported as headline columns:
      greedy, temp@0.7, pless@0.6, pless_norm@0.6, pless@1.0, pless_norm@1.0
    These are the canonical comparison points; other temperatures land in
    the temperature-sweep table (Phase 2 Table 3).
    """
    headline_methods = [
        "greedy",
        "temp@0.7",
        "pless@0.6",
        "pless_norm@0.6",
        "pless@1.0",
        "pless_norm@1.0",
    ]

    # cell[(model, dataset)][method] = (pass@1, pass@10)
    cell: dict[tuple[str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    for r in rows:
        cm = canon_method(r["method"], r["temperature"])
        if cm not in headline_methods:
            continue
        key = (canon_model(r["model"]), r["dataset"])
        # Prefer 10-sample row over 1-sample row if both exist for the same
        # canonical method (e.g. greedy is always 1-sample, but pless is 10).
        if cm in cell[key]:
            existing_n = int(cell[key][cm][2])
            new_n = int(r["num_samples_per_task"])
            if new_n <= existing_n:
                continue
        cell[key][cm] = (r["pass@1"], r["pass@10"], r["num_samples_per_task"])

    keys = sorted(cell.keys())
    lines = [
        "# Table 2 — Headline pass@k",
        "",
        "Per-row format: `pass@1 / pass@10`. `—` = configuration not run for"
        " that (model, benchmark) cell. `n=1` cells (greedy, beam) report"
        " the single-sample success rate in both columns.",
        "",
    ]
    header_cols = ["Model", "Benchmark"] + headline_methods
    sep_cols = [":-----"] + [":---:"] * (len(header_cols) - 1)
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(sep_cols) + "|")
    for model, dataset in keys:
        row = [model, dataset]
        for m in headline_methods:
            v = cell[(model, dataset)].get(m)
            if v is None:
                row.append("—")
            else:
                p1, p10, _ = v
                row.append(f"{float(p1):.3f} / {float(p10):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "_Source: `results/analysis/consolidated_summary.csv`. Method aliases"
        " (`pless`/`p_less`, `temp`/`temp_0.7`) collapsed by"
        " `canon_method()` in this script._"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows_all = load_rows()
    rows = filter_canonical(rows_all)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t1 = OUT_DIR / "table1_models.md"
    t1.write_text(write_table1_models(rows))
    print(f"wrote {t1.relative_to(REPO)}")

    t2 = OUT_DIR / "table2_headline_passk.md"
    t2.write_text(write_table2_headline(rows))
    print(f"wrote {t2.relative_to(REPO)}")

    print(
        f"input rows: {len(rows_all)} "
        f"→ {len(rows)} after MBPP-500/HumanEval-164 filter"
    )


if __name__ == "__main__":
    main()
