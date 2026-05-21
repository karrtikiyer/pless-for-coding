"""Canonical 8-column headline table builder for full_sweep_summary.md files.

Locks the column set + formatting once so every per-model summary uses
the SAME table layout. Don't add or reorder columns here without
updating tests/test_headline_table.py and noting it in CLAUDE.md.

The canonical columns (in display order) are:

  pass@1, pass@3, pass@5, pass@10  — sample-efficiency metrics (Chen et al. 2021)
  cov@0.3, cov@0.5                 — coverage at distance threshold
  struct_div                       — AST-edit-distance structural diversity
  cb_div                           — CodeBLEU mean pairwise diversity

Reads metrics JSONs written by `bench.eval.__main__` (which dumps
``pass_at_k`` / ``cover_at_t`` / ``structural_diversity`` /
``codebleu_diversity``). Missing values display as "—".

Typical usage from a `full_sweep_summary.md`-generating script:

    from bench.eval.headline_table import headline_table

    rows = [
        ("results/.../pless_alpha_a2.0_t1.0_metrics.json", "α=2.0"),
        ("results/.../pless_alpha_a5.0_t1.0_metrics.json", "α=5.0"),
    ]
    print(headline_table(rows))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


CANONICAL_COLUMNS = [
    "pass@1",
    "pass@3",
    "pass@5",
    "pass@10",
    "cov@0.3",
    "cov@0.5",
    "struct_div",
    "cb_div",
]

_MISSING = "—"


def headline_row(metrics_path: Path | str, label: str) -> dict[str, str]:
    """Extract the canonical 9 fields (label + 8 metrics) from a metrics JSON.

    Returns a dict ready to be assembled into a markdown row. All values
    are pre-formatted strings (percentages or 4-decimal floats) to keep
    the table renderer simple.
    """
    p = Path(metrics_path)
    with p.open() as f:
        d = json.load(f)

    pk = d.get("pass_at_k", {})
    cov = d.get("cover_at_t", {})

    def _pct2(d: dict, key: str) -> str:
        """pass@k stored as fraction [0, 1] — display as percentage with 2 dp."""
        v = d.get(str(key))
        return f"{100 * v:.2f}%" if v is not None else _MISSING

    def _pct1(d: dict, key: str) -> str:
        """cov@t stored already as percentage — display as 1 dp."""
        v = d.get(str(key))
        return f"{v:.1f}%" if v is not None else _MISSING

    def _div(value) -> str:
        return f"{value:.4f}" if value is not None else _MISSING

    return {
        "label": label,
        "pass@1": _pct2(pk, 1),
        "pass@3": _pct2(pk, 3),
        "pass@5": _pct2(pk, 5),
        "pass@10": _pct2(pk, 10),
        "cov@0.3": _pct1(cov, "0.3"),
        "cov@0.5": _pct1(cov, "0.5"),
        "struct_div": _div(d.get("structural_diversity")),
        "cb_div": _div(d.get("codebleu_diversity")),
    }


def headline_table(
    rows: Iterable[tuple[Path | str, str]],
    *,
    label_header: str = "Config",
) -> str:
    """Render the canonical 8-column markdown table.

    Args:
        rows: iterable of (metrics_json_path, label) pairs in display order.
        label_header: column header for the label column (default "Config";
            override e.g. "α" when the only varying dim is the Rényi exponent).

    Returns:
        Markdown string. Always emits the header even if rows is empty.
    """
    header = f"| {label_header} | " + " | ".join(CANONICAL_COLUMNS) + " |"
    # All metrics columns are right-aligned (numeric); label is left-aligned.
    sep_cells = [":---"] + ["---:" for _ in CANONICAL_COLUMNS]
    separator = "|" + "|".join(sep_cells) + "|"

    body_lines = []
    for path, label in rows:
        row = headline_row(path, label)
        cells = [row["label"]] + [row[c] for c in CANONICAL_COLUMNS]
        body_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + body_lines)
