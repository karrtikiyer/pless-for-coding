"""Canonical headline table builder for terminal reports and
full_sweep_summary.md files.

Locks the column set + formatting once so every report (terminal print
loops and markdown summaries alike) uses the SAME layout. Don't add or
reorder columns here without updating tests/test_headline_table.py and
noting it in CLAUDE.md.

The canonical columns (in display order) are:

  N                                     — tasks with data in this metrics file
  pass@1, pass@3, pass@5, pass@8, pass@10  — sample-efficiency (Chen et al. 2021)
  cov@0.3, cov@0.5, cov@0.7              — coverage at fractional threshold
  cov@0.3d, cov@0.5d, cov@0.7d           — same, distinct (AST-fingerprinted) solutions
  term%                                  — think-block termination rate
  mean_tok                               — mean token size (via the model's own tokenizer)
  timeout%                               — fraction of samples whose execution timed out
  struct_div                             — AST-edit-distance structural diversity
  cb_div                                 — CodeBLEU mean pairwise diversity

``struct_div``/``cb_div`` are CPU-heavy (pairwise AST edit distance / self-
CodeBLEU) and require a separate, non---skip-diversity eval pass — reserve
them for documentation (markdown summaries), not routine terminal reports.
``TERMINAL_COLUMNS`` (= ``CANONICAL_COLUMNS`` minus those two) is the set to
use for terminal/CLI reporting.

Reads metrics JSONs written by `bench.eval.__main__` (which dumps
``pass_at_k`` / ``cover_at_t`` / ``cover_at_t_distinct`` / ``termination_pct``
/ ``mean_token_size`` / ``timeout_pct`` / ``structural_diversity`` /
``codebleu_diversity``). Missing values display as "—".

Typical usage from a `full_sweep_summary.md`-generating script:

    from bench.eval.headline_table import headline_table

    rows = [
        ("results/.../pless_alpha_a2.0_t1.0_metrics.json", "α=2.0"),
        ("results/.../pless_alpha_a5.0_t1.0_metrics.json", "α=5.0"),
    ]
    print(headline_table(rows))                               # full (+ diversity)
    print(headline_table(rows, columns=TERMINAL_COLUMNS))     # terminal (no diversity)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


CANONICAL_COLUMNS = [
    "N",
    "pass@1",
    "pass@3",
    "pass@5",
    "pass@8",
    "pass@10",
    "cov@0.3",
    "cov@0.5",
    "cov@0.7",
    "cov@0.3d",
    "cov@0.5d",
    "cov@0.7d",
    "term%",
    "mean_tok",
    "timeout%",
    "struct_div",
    "cb_div",
]

# Terminal/CLI reports should not run the CPU-heavy diversity pass on every
# invocation -- reserve struct_div/cb_div for documentation (markdown
# summaries generated with a dedicated, non---skip-diversity eval pass).
TERMINAL_COLUMNS = [c for c in CANONICAL_COLUMNS if c not in ("struct_div", "cb_div")]

# One-line definition per column -- single source of truth so any report
# that renders CANONICAL_COLUMNS/TERMINAL_COLUMNS can also render a legend
# without hand-typing descriptions (which would drift from the real
# semantics the way the old hand-maintained progress_report.md did).
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "N": "Number of tasks with data in this metrics file (60 = fully complete for this benchmark; <60 = partial/in-progress).",
    "pass@1": "Probability at least 1 of 1 sampled solution is correct (unbiased estimator, Chen et al. 2021).",
    "pass@3": "Probability at least 1 of 3 independent samples is correct.",
    "pass@5": "Probability at least 1 of 5 independent samples is correct.",
    "pass@8": "Probability at least 1 of 8 independent samples is correct.",
    "pass@10": "Probability at least 1 of 10 independent samples is correct (= all samples drawn for this benchmark).",
    "cov@0.3": "% of tasks where at least 30% of the 10 samples were correct.",
    "cov@0.5": "% of tasks where at least 50% of the 10 samples were correct.",
    "cov@0.7": "% of tasks where at least 70% of the 10 samples were correct.",
    "cov@0.3d": "Same as cov@0.3, but counting only AST-fingerprint-DISTINCT correct solutions (penalizes a config that passes a task only by repeating the same solution).",
    "cov@0.5d": "Same as cov@0.5, but counting only AST-fingerprint-distinct correct solutions.",
    "cov@0.7d": "Same as cov@0.7, but counting only AST-fingerprint-distinct correct solutions.",
    "term%": "% of samples that closed their think block (</think> or [/THINK], model-specific) before hitting the token cap, vs. getting cut off mid-reasoning.",
    "mean_tok": "Mean token length of the raw generated sample (thinking + code combined).",
    "timeout%": "% of samples whose generated code timed out during test execution. Tracked for APPS/LiveCodeBench only — shows as — for datasets without per-sample failure-reason tracking, e.g. MBPP/HumanEval.",
    "struct_div": "Structural diversity: mean pairwise AST edit distance among a task's correct solutions (higher = more structurally different correct answers).",
    "cb_div": "CodeBLEU-based diversity: mean pairwise CodeBLEU dissimilarity among a task's correct solutions.",
}

_MISSING = "—"
_DISTINCT_T_SUFFIX = "d"  # "cov@0.3d" -> cover_at_t_distinct["0.3"]


def headline_row(metrics_path: Path | str, label: str) -> dict[str, str]:
    """Extract the canonical fields (label + metrics) from a metrics JSON.

    Returns a dict ready to be assembled into a markdown/terminal row. All
    values are pre-formatted strings (percentages or 4-decimal floats) to
    keep the table renderer simple.
    """
    p = Path(metrics_path)
    with p.open() as f:
        d = json.load(f)

    pk = d.get("pass_at_k", {})
    cov = d.get("cover_at_t", {})
    cov_distinct = d.get("cover_at_t_distinct", {})

    def _pct2(d: dict, key) -> str:
        """pass@k stored as fraction [0, 1] — display as percentage with 2 dp."""
        v = d.get(str(key))
        return f"{100 * v:.2f}%" if v is not None else _MISSING

    def _pct1(d: dict, key) -> str:
        """cov@t stored already as percentage — display as 1 dp."""
        v = d.get(str(key))
        return f"{v:.1f}%" if v is not None else _MISSING

    def _div(value) -> str:
        return f"{value:.4f}" if value is not None else _MISSING

    # Fallback: if top-level codebleu_diversity is None (e.g., APPS runs used
    # --skip-diversity), aggregate from per_task self_codebleu cache if present.
    cb_div = d.get("codebleu_diversity")
    if cb_div is None:
        per_task = d.get("per_task") or []
        cb_vals = [t.get("self_codebleu") for t in per_task if t.get("self_codebleu") is not None]
        if cb_vals:
            cb_div = sum(cb_vals) / len(cb_vals)

    num_tasks = d.get("num_tasks")

    return {
        "label": label,
        "N": str(num_tasks) if num_tasks is not None else _MISSING,
        "pass@1": _pct2(pk, 1),
        "pass@3": _pct2(pk, 3),
        "pass@5": _pct2(pk, 5),
        "pass@8": _pct2(pk, 8),
        "pass@10": _pct2(pk, 10),
        "cov@0.3": _pct1(cov, "0.3"),
        "cov@0.5": _pct1(cov, "0.5"),
        "cov@0.7": _pct1(cov, "0.7"),
        "cov@0.3d": _pct1(cov_distinct, "0.3"),
        "cov@0.5d": _pct1(cov_distinct, "0.5"),
        "cov@0.7d": _pct1(cov_distinct, "0.7"),
        "term%": _pct1({"v": d.get("termination_pct")}, "v"),
        "mean_tok": (
            f"{d['mean_token_size']:.0f}" if d.get("mean_token_size") is not None else _MISSING
        ),
        "timeout%": _pct1({"v": d.get("timeout_pct")}, "v"),
        "struct_div": _div(d.get("structural_diversity")),
        "cb_div": _div(cb_div),
    }


def headline_table(
    rows: Iterable[tuple[Path | str, str]],
    *,
    label_header: str = "Config",
    columns: list[str] | None = None,
) -> str:
    """Render the canonical markdown table.

    Args:
        rows: iterable of (metrics_json_path, label) pairs in display order.
        label_header: column header for the label column (default "Config";
            override e.g. "α" when the only varying dim is the Rényi exponent).
        columns: which columns to render, in order (default: CANONICAL_COLUMNS,
            the full documentation set including struct_div/cb_div). Pass
            ``TERMINAL_COLUMNS`` for routine terminal/CLI reports.

    Returns:
        Markdown string. Always emits the header even if rows is empty.
    """
    cols = columns if columns is not None else CANONICAL_COLUMNS
    header = f"| {label_header} | " + " | ".join(cols) + " |"
    # All metrics columns are right-aligned (numeric); label is left-aligned.
    sep_cells = [":---"] + ["---:" for _ in cols]
    separator = "|" + "|".join(sep_cells) + "|"

    body_lines = []
    for path, label in rows:
        row = headline_row(path, label)
        cells = [row["label"]] + [row[c] for c in cols]
        body_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + body_lines)
