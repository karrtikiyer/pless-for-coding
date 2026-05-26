"""APPS algosim report — bucket NAUADC / EA / DA@K by (source, config/model, difficulty).

Reads a directory of algosim response parquets where each row's
``problem_id`` is encoded as ``<SOURCE>_<config_or_model_slug>_<difficulty>_<id>``
(produced by :mod:`bench.eval.algosim_export_apps` for our Qwen3-8B configs
and by :mod:`bench.eval.algosim_paper_replicate` for the paper's reference
models). For each (source, difficulty) cell we list every entity that has
problems in that cell, with its per-cell NAUADC / EA / DA@10.

Compares against paper-published numbers (Table 2 from arXiv:2503.00691) when
those exist for the bucket and model.

Usage::

    uv run python -m bench.eval.algosim_report_apps \\
        --responses-dir algosim_data/apps/responses \\
        --paper-baselines-dir algosim_data/apps_paper_baselines/responses \\
        --output-dir results/pless_apps_results/analysis
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy

K_VALUES = list(range(1, 26))

# Table 2 from Lee et al. 2025 (NAUADC values). Filled out per (model, source,
# difficulty). None where the paper does not publish a number. These let us
# show paper-reference rows alongside our re-clustered numbers in the report.
# Source: arXiv:2503.00691v2 Table 2. Numbers should be verified against the
# final published version once we have an exact copy.
PAPER_NAUADC_TABLE: dict[tuple[str, str, str], float] = {
    # (model, source, difficulty) -> NAUADC
    # AtCoder introductory (column from the abridged Table 2 we extracted)
    ("gpt-4o-2024-08-06",           "ATCODER",    "introductory"): 1.302,
    ("deepseek-coder-33b",          "ATCODER",    "introductory"): 1.780,
    # CodeForces introductory
    ("gpt-4o-2024-08-06",           "CODEFORCES", "introductory"): 1.507,
    ("deepseek-coder-33b",          "CODEFORCES", "introductory"): 1.952,
    # Add additional cells as we verify them from the paper's actual table.
}

# Reverse map from our slug to a display name + the paper's identifier (when
# different — e.g., we re-cluster TheBloke's AWQ build but the paper reports
# the un-quantised name).
SLUG_DISPLAY = {
    "ds6.7B-base":           "deepseek-coder-6.7b-base",
    "ds6.7B-instruct":       "deepseek-coder-6.7b-instruct",
    "ds33B-instruct-AWQ":    "deepseek-coder-33b-instruct (AWQ)",
    "gpt4o":                 "gpt-4o-2024-08-06",
    "gpt4o-mini":            "gpt-4o-mini-2024-07-18",
}


# ── algosim math (lifted from algosim_report.py) ─────────────────────────────


def _comb(n: int, k: int) -> float:
    if n < k:
        return 0.0
    return float(np.prod(np.arange(n, n - k, -1) / np.arange(k, 0, -1)))


def _da_at_k(group_sizes: np.ndarray, k: int) -> float:
    n, m = int(np.sum(group_sizes)), len(group_sizes)
    if k > n:
        return float(m)
    return m - float(np.sum([_comb(n - s, k) / _comb(n, k) for s in group_sizes]))


# ── problem_id parser ───────────────────────────────────────────────────────


_PID_RE = re.compile(
    r"^(?P<source>ATCODER|CODEFORCES)"
    # Entity may contain underscores (e.g. ``pless_alpha_a2.0_t1.0``), so we
    # match lazily and let the fixed difficulty alternation anchor the split.
    r"_(?P<entity>.+?)"
    r"_(?P<difficulty>introductory|interview|competition)"
    r"_(?P<id>.+)$"
)


def _parse_problem_id(pid: str) -> dict | None:
    m = _PID_RE.match(pid)
    if not m:
        return None
    return m.groupdict()


# ── Aggregation ──────────────────────────────────────────────────────────────


def _compute_bucket(df: pd.DataFrame) -> dict:
    """NAUADC / EA / DA@K for a single (entity, source, difficulty) bucket."""
    if df.empty:
        return {"n_problems": 0, "DA": [0.0] * len(K_VALUES),
                "NAUADC": 0.0, "EA": 0.0}
    group_indexes = df["group_index"].apply(
        lambda x: [] if x is None or (isinstance(x, float) and np.isnan(x))
        else list(x)
    )
    solution_groups = group_indexes.apply(
        lambda gi: [count for _, count in Counter(gi).most_common()]
    )
    keep = solution_groups.map(sum) >= 1
    solution_groups = solution_groups[keep]
    if solution_groups.empty:
        return {"n_problems": 0, "DA": [0.0] * len(K_VALUES),
                "NAUADC": 0.0, "EA": 0.0}

    da_curve = [
        float(solution_groups.apply(
            lambda g, k=k: _da_at_k(np.array(g), k)
        ).mean())
        for k in K_VALUES
    ]
    nauadc = float(np.trapezoid(da_curve, K_VALUES) / (K_VALUES[-1] - K_VALUES[0]))

    def _ea(counts):
        prob = np.array(counts) / np.sum(counts)
        return float(np.exp(-np.sum(xlogy(prob, prob))))
    ea = float(solution_groups.apply(_ea).mean())

    return {
        "n_problems": int(len(solution_groups)),
        "DA": da_curve,
        "NAUADC": nauadc,
        "EA": ea,
    }


def _load_responses(responses_dir: Path) -> pd.DataFrame:
    """Concatenate all response parquets and annotate parsed problem_id parts.

    Discovery is recursive — buckets may be nested as ``<bucket>/<arm>.parquet``
    (the layout the α-sweep judge produces). Bucketing into (source, difficulty)
    cells is driven by ``problem_id`` parsing, not by directory structure, so
    flat and nested layouts both work.
    """
    frames = []
    for p in sorted(responses_dir.rglob("*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception as exc:
            print(f"  [warn] skipping unreadable {p.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    parsed = df["problem_id"].map(_parse_problem_id)
    bad = parsed.isna().sum()
    if bad:
        print(f"  [warn] {bad}/{len(df)} rows had unparseable problem_id; dropping")
        df = df[parsed.notna()].copy()
        parsed = parsed.dropna()
    df["source"] = parsed.map(lambda d: d["source"])
    df["entity"] = parsed.map(lambda d: d["entity"])
    df["difficulty"] = parsed.map(lambda d: d["difficulty"])
    return df


def _aggregate(df: pd.DataFrame) -> list[dict]:
    """Compute NAUADC per (entity, source, difficulty) cell."""
    rows: list[dict] = []
    for (entity, source, difficulty), sub in df.groupby(
        ["entity", "source", "difficulty"]
    ):
        stats = _compute_bucket(sub)
        rows.append({
            "entity": entity,
            "display_name": SLUG_DISPLAY.get(entity, entity),
            "source": source,
            "difficulty": difficulty,
            **stats,
        })
    return rows


# ── Output ──────────────────────────────────────────────────────────────────


def _write_markdown(rows: list[dict], output_path: Path,
                    label: str = "our configs") -> None:
    cells = sorted({(r["source"], r["difficulty"]) for r in rows})
    n_entities = len({r["entity"] for r in rows})
    lines = [
        f"# AlgoSim APPS Report — {label} vs paper reference baselines",
        "",
        f"Per (source, difficulty) bucket: NAUADC / EA / DA@10 for each entity "
        f"(a {label} config OR a paper-baseline model re-clustered with our "
        f"pipeline). Paper-published Table 2 numbers are interleaved as `paper:` "
        f"rows where known.",
        "",
        "**Comparability caveats — read before quoting any number across blocks:**",
        "",
        f"1. **Sample-budget asymmetry.** Paper-baseline NAUADC is computed over "
        f"100 samples/problem; our {label} configs use 10/problem. DA@10 stays "
        f"directly comparable; NAUADC integrals span k=1..25 on different sample "
        f"budgets and should be read accordingly.",
        "",
        f"2. **Sample-filter asymmetry.** Paper baselines were clustered after "
        f"filtering to functionally-correct samples (`status == \"Passed\"`). Our "
        f"{label} configs are clustered without a correctness filter (we don't "
        f"run APPS execution at algosim-export time). On easy problems with high "
        f"pass rates this matters little; on competition difficulty, where most "
        f"{label} samples are broken-in-different-ways, the unfiltered NAUADC "
        f"inflates because the judge sees those broken samples as distinct "
        f"\"algorithms\". The **relative ordering across our {n_entities} configs** "
        f"remains informative; the **absolute comparison to paper baselines on "
        f"the same bucket** is only meaningful where pass rates are high enough "
        f"that filter vs no-filter would converge.",
        "",
    ]
    for source, difficulty in cells:
        bucket_rows = [r for r in rows
                       if r["source"] == source and r["difficulty"] == difficulty]
        lines.append(f"## {source} / {difficulty}")
        lines.append("")
        lines.append("| Entity (re-clustered) | NAUADC | EA | DA@10 | n_problems |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in sorted(bucket_rows, key=lambda r: -r["NAUADC"]):
            lines.append(
                f"| {r['display_name']} | **{r['NAUADC']:.3f}** | "
                f"{r['EA']:.3f} | {r['DA'][K_VALUES.index(10)]:.3f} | "
                f"{r['n_problems']} |"
            )
        # Paper-published rows for context
        paper_rows = [(model, naud) for (model, src, diff), naud
                      in PAPER_NAUADC_TABLE.items()
                      if src == source and diff == difficulty]
        if paper_rows:
            lines.append("")
            lines.append("| Paper Table 2 (their numbers, not re-clustered) | NAUADC |")
            lines.append("|---|---:|")
            for model, naud in sorted(paper_rows, key=lambda x: -x[1]):
                lines.append(f"| paper: {model} | {naud:.3f} |")
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")
    print(f"[apps_report] wrote {output_path}")


def _bar_chart(rows: list[dict], output_path: Path,
               label: str = "our configs") -> None:
    cells = sorted({(r["source"], r["difficulty"]) for r in rows})
    if not cells:
        return
    fig, axes = plt.subplots(
        len(cells), 1, figsize=(11, 4 * len(cells)), squeeze=False,
    )
    for ax, (source, difficulty) in zip(axes.flat, cells):
        bucket = sorted(
            [r for r in rows if r["source"] == source and r["difficulty"] == difficulty],
            key=lambda r: -r["NAUADC"],
        )
        entities = [r["display_name"] for r in bucket]
        nauadc = [r["NAUADC"] for r in bucket]
        xs = np.arange(len(entities))
        bars = ax.bar(xs, nauadc, color="#1565C0", alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(entities, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{source} / {difficulty}")
        ax.set_ylabel("NAUADC")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, nauadc):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        # Paper reference horizontal lines
        for (model, src, diff), naud in PAPER_NAUADC_TABLE.items():
            if src == source and diff == difficulty:
                ax.axhline(naud, color="red", linestyle="--", alpha=0.5,
                           label=f"paper: {model} = {naud:.3f}")
        if any(src == source and diff == difficulty
               for (_, src, diff), _ in PAPER_NAUADC_TABLE.items()):
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"APPS NAUADC by bucket — {label} configs + paper baselines",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[apps_report] wrote {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses-dir", type=Path, required=True,
                   help="Directory containing algosim response parquets (mixed entities OK).")
    p.add_argument("--paper-baselines-dir", type=Path, default=None,
                   help="Optional second responses directory (e.g. paper baselines). "
                        "If supplied, parquets from both dirs are pooled before bucketing.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default=None,
                   help="Display name for the entities being re-clustered "
                        "(e.g. 'Qwen2.5-Coder-7B-Instruct α-sweep'). Used in "
                        "the report title, prose, and chart suptitle. Defaults "
                        "to the responses-dir basename.")
    return p.parse_args()


def main():
    args = parse_args()
    df_ours = _load_responses(args.responses_dir)
    print(f"[apps_report] loaded {len(df_ours):>6} rows from {args.responses_dir}")
    if args.paper_baselines_dir is not None and args.paper_baselines_dir.exists():
        df_paper = _load_responses(args.paper_baselines_dir)
        print(f"[apps_report] loaded {len(df_paper):>6} rows from {args.paper_baselines_dir}")
        df = pd.concat([df_ours, df_paper], ignore_index=True)
    else:
        df = df_ours
    if df.empty:
        raise SystemExit("No response parquets to aggregate.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _aggregate(df)
    print(f"[apps_report] aggregated {len(rows)} (entity, source, difficulty) cells")

    raw_path = args.output_dir / "algosim_apps_per_cell.json"
    raw_path.write_text(json.dumps({"k_values": K_VALUES, "cells": rows}, indent=2))
    print(f"[apps_report] wrote {raw_path}")

    label = args.label or args.responses_dir.name
    _write_markdown(rows, args.output_dir / "algosim_apps_report.md", label=label)
    _bar_chart(rows, args.output_dir / "algosim_apps_bar.png", label=label)


if __name__ == "__main__":
    main()
