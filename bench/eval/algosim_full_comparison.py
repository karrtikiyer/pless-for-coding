"""Side-by-side comparison: our diversity metrics + NAUADC + pass@k.

Joins ``split_decoding_report_summary.json`` (which has pass@k +
``struct_div`` / ``codebleu_div``) with ``algosim_per_config.json`` (which
has NAUADC / EA / DA@10) for every Qwen3-8B config that does **not** use
``temp_standard`` (top_p=0.95, top_k=20). Configs are grouped by what they
test rather than sorted by any single metric so the reader can scan the
diversity-vs-pass@k trade-off in one view.

Outputs:

* ``algosim_full_comparison.md`` — single markdown table with section headers
* ``algosim_full_comparison.png`` — multi-panel bar chart (struct_div,
  codebleu_div, NAUADC, pass@1) across all kept configs

Usage::

    uv run python -m bench.eval.algosim_full_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path("results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis")
SUMMARY_PATH = ANALYSIS_DIR / "split_decoding_report_summary.json"
ALGOSIM_PATH = ANALYSIS_DIR / "algosim_per_config.json"

# Configs that use temp_standard (top_p=0.95, top_k=20) — excluded per
# user direction. Anything else with thinking + split + pless on at least
# one phase is kept.
EXCLUDE = {"F", "G", "T15",
           "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10",
           "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"}

# Ordered grouping: scan top-to-bottom is "less thinking" → "more thinking" →
# "pure-temp split sweep" → "stress tests".
GROUPS = [
    ("No-thinking baselines", ["A", "B"]),
    ("Thinking-only baselines (no split)", ["C", "D", "E"]),
    ("Uniform high-temp thinking (no split)", ["T15N", "P15"]),
    ("Pure-temp split baseline (no pless)", ["T15P"]),
    ("Pure-temp split + pless on code (think@1.5, code∈{1.0,1.5,2.0,3.0})",
     ["H7P", "H8P", "H9P", "H10P"]),
    ("Pure-temp split stress tests (think>1.5)", ["H11P", "H12P"]),
]


def _load() -> tuple[dict, dict]:
    summary = json.loads(SUMMARY_PATH.read_text())
    algosim = json.loads(ALGOSIM_PATH.read_text()) if ALGOSIM_PATH.exists() else {}
    algosim_by_cfg = {r["config"]: r for r in algosim.get("configs", [])}
    return summary, algosim_by_cfg


def _row_for(cfg: str, summary: dict, algosim_by_cfg: dict) -> dict | None:
    s = summary.get(cfg)
    if s is None:
        return None
    a = algosim_by_cfg.get(cfg)
    return {
        "config": cfg,
        "label": s["label"],
        "pass@1": s["pass_at_k"]["1"],
        "pass@3": s["pass_at_k"]["3"],
        "pass@5": s["pass_at_k"]["5"],
        "pass@10": s["pass_at_k"]["10"],
        "struct_div": s["struct_div"],
        "codebleu_div": s["codebleu_div"],
        "NAUADC": a["NAUADC"] if a else None,
    }


def write_markdown(rows_by_group: list[tuple[str, list[dict]]], out_path: Path) -> None:
    lines = [
        "# algosim Full Comparison — Qwen3-8B Configs (excl. temp_standard)",
        "",
        "Single view across our existing diversity metrics (`struct_div`, "
        "`codebleu_div`), the new algosim NAUADC, and pass@k. All configs that "
        "use `temp_standard` (top_p=0.95, top_k=20) on any phase are excluded; "
        "the remaining 14 configs span the pure-temp split-decoding family plus "
        "every non-split baseline.",
        "",
        "NAUADC values are shown only where algosim has been run; remaining "
        "configs are marked `—`.",
        "",
        "| Config | Label | pass@1 | pass@3 | pass@5 | pass@10 | struct_div | codebleu_div | NAUADC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_title, rows in rows_by_group:
        lines.append(f"| **{group_title}** |||||||||")
        for r in rows:
            nauadc = f"**{r['NAUADC']:.3f}**" if r["NAUADC"] is not None else "—"
            lines.append(
                f"| {r['config']} | {r['label']} | "
                f"{r['pass@1']:.3f} | {r['pass@3']:.3f} | "
                f"{r['pass@5']:.3f} | {r['pass@10']:.3f} | "
                f"{r['struct_div']:.3f} | {r['codebleu_div']:.3f} | {nauadc} |"
            )
    out_path.write_text("\n".join(lines) + "\n")


def plot_grid(rows: list[dict], out_path: Path) -> None:
    cfgs = [r["config"] for r in rows]
    xs = np.arange(len(cfgs))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    panels = [
        ("pass@1", [r["pass@1"] for r in rows], "#1565C0"),
        ("struct_div", [r["struct_div"] for r in rows], "#43A047"),
        ("codebleu_div", [r["codebleu_div"] for r in rows], "#FB8C00"),
        ("NAUADC", [r["NAUADC"] if r["NAUADC"] is not None else 0 for r in rows], "#E53935"),
    ]
    nauadc_missing = [r["NAUADC"] is None for r in rows]

    for ax, (title, ys, color) in zip(axes.flat, panels):
        bars = ax.bar(xs, ys, color=color, alpha=0.85)
        if title == "NAUADC":
            for i, missing in enumerate(nauadc_missing):
                if missing:
                    bars[i].set_alpha(0.2)
                    bars[i].set_hatch("//")
        ax.set_xticks(xs)
        ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        # Annotate values on top of bars
        for i, (b, y) in enumerate(zip(bars, ys)):
            if title == "NAUADC" and nauadc_missing[i]:
                ax.text(b.get_x() + b.get_width() / 2, 0.02,
                        "n/a", ha="center", va="bottom",
                        fontsize=8, color="gray")
            else:
                ax.text(b.get_x() + b.get_width() / 2, y,
                        f"{y:.3f}", ha="center", va="bottom",
                        fontsize=8)

    fig.suptitle("Qwen3-8B Diversity Metrics + pass@1 — pure-temp / baseline configs",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()

    summary, algosim_by_cfg = _load()

    rows_by_group: list[tuple[str, list[dict]]] = []
    flat_rows: list[dict] = []
    for title, cfgs in GROUPS:
        group_rows = []
        for cfg in cfgs:
            if cfg in EXCLUDE:
                continue
            row = _row_for(cfg, summary, algosim_by_cfg)
            if row is None:
                print(f"[warn] {cfg} not in summary.json; skipping")
                continue
            group_rows.append(row)
            flat_rows.append(row)
        if group_rows:
            rows_by_group.append((title, group_rows))

    md_path = args.analysis_dir / "algosim_full_comparison.md"
    write_markdown(rows_by_group, md_path)
    print(f"[full_comparison] wrote {md_path}")

    png_path = args.analysis_dir / "algosim_full_comparison.png"
    plot_grid(flat_rows, png_path)
    print(f"[full_comparison] wrote {png_path}")

    print(f"\n  total configs in view: {len(flat_rows)}")
    print(f"  with NAUADC:           {sum(1 for r in flat_rows if r['NAUADC'] is not None)}")
    print(f"  NAUADC missing:        {sum(1 for r in flat_rows if r['NAUADC'] is None)}")


if __name__ == "__main__":
    main()
