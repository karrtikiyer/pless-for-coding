"""Cross-sampler 'best config per metric' analysis for the 3-model × 2-dataset cell.

For each (model, dataset), report the argmax-per-metric config across:
  * Group A: α-arm samplers (pless_alpha_a{α}_t1.0) — the reference baseline
  * Group B: non-α stochastic samplers (pless, pless_norm, temp, top_p, top_k, split, etc.)

NON-stochastic samplers (greedy, beam) are excluded — they always produce
identical samples so diversity metrics are 0 and pass@k collapses to pass@1.

Metrics reported per argmax:
  * pass@1, pass@10 (Chen et al. unbiased)
  * codebleu_diversity (mean pairwise CodeBLEU distance)

Per the project's Scientific Rigor rules: every number in the output is
loaded live from a metrics JSON; nothing pulled from memory.

Result directories searched (set in the SOURCES list below):
  * results/pless_full_mbpp_results — main MBPP T-sweeps for CodeLlama + m-a-p
  * results/full_mbpp_pre_post_temp_pless — main MBPP T-sweeps for Qwen2.5-Coder
  * results/pless_human_eval_results/full_precision_results — full-precision HumanEval
  * results/pless_alpha_full_mbpp — α-arm MBPP (Group A)
  * results/pless_alpha_full_humaneval — α-arm HumanEval (Group A)

Directories with 'fix' in the path are excluded by request.

Outputs:
  * stdout: condensed best-config tables per (model, dataset) cell
  * results/analysis/sampler_comparison_summary.md — same as stdout, persisted
  * results/analysis/full_table_{model}_{dataset}.md — full per-config table for
    each cell (~50-100 rows; lets you verify the aggregation by eye)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

MODELS = {
    "Qwen2.5-Coder-7B-Instruct": "Qwen--Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct": "codellama--CodeLlama-7b-Instruct-hf",
    "OpenCodeInterpreter-DS-1.3B": "m-a-p--OpenCodeInterpreter-DS-1.3B",
}

# Directories to walk. Each entry: (root, dataset, glob_pattern).
# The glob pattern is relative to root and finds metric JSONs for ANY model
# in MODELS.values(). Each tuple describes one (root, dataset) pair to scan.
SOURCES = [
    # MBPP T-sweep + miscellaneous stochastic samplers
    ("results/pless_full_mbpp_results", "MBPP",
     "{slug}/metrics/*_metrics.json"),
    # Qwen2.5-Coder MBPP T-sweep + pre/post-temp variants
    ("results/full_mbpp_pre_post_temp_pless", "MBPP",
     "{slug}/metrics/*_metrics.json"),
    # HumanEval full-precision JSON (separate parsing — schema differs)
    ("results/pless_human_eval_results/full_precision_results", "HumanEval",
     "{slug}/metrics/*_metrics.json"),
    # α-arm MBPP (Group A reference)
    ("results/pless_alpha_full_mbpp", "MBPP",
     "{slug}/metrics/pless_alpha_a*_t1.0_metrics.json"),
    # α-arm HumanEval (Group A reference) — note two layouts:
    #   - {slug}/humaneval/metrics/  (original 3 non-thinking models)
    #   - {slug}/metrics/             (Qwen3-8B series; not in scope for this query)
    ("results/pless_alpha_full_humaneval", "HumanEval",
     "{slug}/humaneval/metrics/pless_alpha_a*_t1.0_metrics.json"),
]


# Non-stochastic samplers to exclude (always produce identical samples;
# diversity = 0 by construction).
NON_STOCHASTIC_PREFIXES = ("greedy", "beam")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class ConfigEntry:
    model: str          # short label, e.g. "Qwen2.5-Coder-7B-Instruct"
    dataset: str        # "MBPP" | "HumanEval"
    config: str         # e.g. "pless_t1.0", "pless_alpha_a5.0_t1.0", "temp_t0.7"
    metrics_path: Path
    pass_at_1: float | None
    pass_at_10: float | None
    codebleu_div: float | None
    is_alpha_arm: bool  # True if config starts with "pless_alpha_a"
    num_tasks: int | None
    is_stochastic: bool # filter result


def _config_name_from_path(p: Path) -> str:
    """Strip `_metrics.json` suffix."""
    return p.name.replace("_metrics.json", "")


def _is_stochastic(config_name: str) -> bool:
    for prefix in NON_STOCHASTIC_PREFIXES:
        if config_name.startswith(prefix):
            return False
    return True


def _load_entry(p: Path, model_short: str, dataset: str) -> ConfigEntry | None:
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    config = _config_name_from_path(p)
    pk = d.get("pass_at_k") or {}
    is_alpha = config.startswith("pless_alpha_a")
    return ConfigEntry(
        model=model_short,
        dataset=dataset,
        config=config,
        metrics_path=p,
        pass_at_1=pk.get("1"),
        pass_at_10=pk.get("10"),
        codebleu_div=d.get("codebleu_diversity"),
        is_alpha_arm=is_alpha,
        num_tasks=d.get("num_tasks"),
        is_stochastic=_is_stochastic(config),
    )


def collect_entries(repo_root: Path) -> list[ConfigEntry]:
    entries: list[ConfigEntry] = []
    for short, slug in MODELS.items():
        for root, dataset, pattern in SOURCES:
            base = repo_root / root
            if not base.exists():
                continue
            # Substitute {slug} and glob from base.
            for p in base.glob(pattern.format(slug=slug)):
                # Belt-and-suspenders: skip anything with 'fix' in path.
                if "fix" in str(p).lower():
                    continue
                e = _load_entry(p, short, dataset)
                if e is None:
                    continue
                entries.append(e)
    return entries


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _argmax(entries: list[ConfigEntry], key: str) -> ConfigEntry | None:
    """Pick the entry with the largest .{key}, skipping None values."""
    candidates = [e for e in entries if getattr(e, key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda e: getattr(e, key))


def _fmt_pct(v: float | None) -> str:
    return f"{100*v:.2f}%" if v is not None else "—"


def _fmt_div(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _pareto_dominates(a: ConfigEntry, b: ConfigEntry,
                      metrics: tuple[str, ...] = ("pass_at_10", "codebleu_div")) -> bool:
    """True iff `a` is ≥ b on every metric and strictly > on at least one.
    Missing values (None) treat that axis as 'incomparable' so a dominance
    claim requires both entries to have a value on every axis."""
    strict_anywhere = False
    for m in metrics:
        va, vb = getattr(a, m), getattr(b, m)
        if va is None or vb is None:
            return False
        if va < vb:
            return False
        if va > vb:
            strict_anywhere = True
    return strict_anywhere


def _best_config_table_md(entries: list[ConfigEntry]) -> str:
    """For each (model, dataset) cell, emit argmax-per-metric rows for
    Group A (α-arm) and Group B (other stochastic samplers), plus a
    Pareto-dominance check on (pass@10, codebleu_div)."""
    lines = []
    lines.append("# Sampler comparison — best config per metric")
    lines.append("")
    lines.append(
        "For each (model, dataset) cell, the config that maxes the named metric is "
        "shown, along with its other two metrics for context. Non-stochastic "
        "samplers (greedy, beam) excluded — those collapse the diversity metric "
        "to 0 by construction."
    )
    lines.append("")
    lines.append(
        "**Group A** = α-arm samplers (`pless_alpha_a{α}_t1.0`); these are the "
        "reference. **Group B** = all other stochastic samplers (pless, "
        "pless_norm, temp, top_p, top_k, split, pless_pt, etc.) "
        "— the comparison set."
    )
    lines.append("")
    lines.append(
        "**Scope gaps** to be aware of (limit the comparison):\n"
        "- Several non-α HumanEval configs come from the older "
        "`full_precision_results` format which lacks `codebleu_diversity`. "
        "Those cells show `—` in the cb_div column and are excluded from "
        "Pareto-dominance checks on the cb_div axis.\n"
        "- m-a-p OCI-DS-1.3B has no non-α HumanEval results in the included "
        "directories. Cell shows only the α-arm rows.\n"
        "- The `temprature_results` HumanEval directory is OUT of scope per "
        "user request; HumanEval T-sweep data for any model is not included."
    )
    lines.append("")

    cells = {}
    for e in entries:
        cells.setdefault((e.model, e.dataset), []).append(e)

    for (model, dataset), cell_entries in sorted(cells.items()):
        alpha = [e for e in cell_entries if e.is_alpha_arm and e.is_stochastic]
        other = [e for e in cell_entries if not e.is_alpha_arm and e.is_stochastic]

        lines.append(f"## {model} — {dataset}")
        lines.append("")
        lines.append(
            f"Config count: **{len(alpha)} α-arm** + **{len(other)} other "
            f"stochastic**. (Skipped {len([e for e in cell_entries if not e.is_stochastic])} "
            f"non-stochastic.)"
        )
        lines.append("")

        if not alpha and not other:
            lines.append("_No configs found._")
            lines.append("")
            continue

        lines.append("| Group | Best for | Config | pass@1 | pass@10 | codebleu_div | n_tasks |")
        lines.append("|---|---|---|---:|---:|---:|---:|")

        for label, group in [("A (α-arm)", alpha), ("B (other stochastic)", other)]:
            if not group:
                lines.append(f"| {label} | — | _no configs_ | | | | |")
                continue
            for metric in ("pass_at_1", "pass_at_10", "codebleu_div"):
                metric_label = {"pass_at_1": "pass@1", "pass_at_10": "pass@10",
                                "codebleu_div": "codebleu_div"}[metric]
                winner = _argmax(group, metric)
                if winner is None:
                    lines.append(f"| {label} | best {metric_label} | _missing_ | | | | |")
                    continue
                lines.append(
                    f"| {label} | best {metric_label} | `{winner.config}` | "
                    f"{_fmt_pct(winner.pass_at_1)} | {_fmt_pct(winner.pass_at_10)} | "
                    f"{_fmt_div(winner.codebleu_div)} | {winner.num_tasks or '—'} |"
                )
        lines.append("")

        # Pareto-dominance check on (pass@10, cb_div). Does any Group-B
        # config dominate the best Group-A α-arm config?
        alpha_pass10 = _argmax(alpha, "pass_at_10")
        alpha_cb = _argmax(alpha, "codebleu_div")
        if alpha_pass10 is not None and alpha_cb is not None:
            ref_configs = list({alpha_pass10.config: alpha_pass10,
                                alpha_cb.config: alpha_cb}.values())
            dominators = []
            for ref in ref_configs:
                for o in other:
                    if _pareto_dominates(o, ref, metrics=("pass_at_10", "codebleu_div")):
                        dominators.append((o.config, ref.config,
                                           o.pass_at_10, o.codebleu_div,
                                           ref.pass_at_10, ref.codebleu_div))
            if dominators:
                lines.append("**Pareto-dominance on (pass@10, cb_div):** the following "
                             "Group-B configs strictly dominate the best Group-A configs "
                             "(≥ on both, > on at least one). cb_div=`None` entries are "
                             "excluded from this check.")
                lines.append("")
                lines.append("| Group-B dominator | Group-A dominated | B pass@10 | B cb_div | A pass@10 | A cb_div |")
                lines.append("|---|---|---:|---:|---:|---:|")
                seen = set()
                for cfg_b, cfg_a, b10, bcb, a10, acb in dominators:
                    if (cfg_b, cfg_a) in seen:
                        continue
                    seen.add((cfg_b, cfg_a))
                    lines.append(
                        f"| `{cfg_b}` | `{cfg_a}` | {_fmt_pct(b10)} | {_fmt_div(bcb)} | "
                        f"{_fmt_pct(a10)} | {_fmt_div(acb)} |"
                    )
                lines.append("")
            else:
                lines.append("**Pareto-dominance on (pass@10, cb_div):** no "
                             "Group-B config strictly dominates the best α-arm "
                             "on both axes. (α-arm is on the Pareto frontier.)")
                lines.append("")
    return "\n".join(lines) + "\n"


def _full_table_md(entries: list[ConfigEntry], model: str, dataset: str) -> str:
    """Full per-config table for one (model, dataset) — for verification."""
    cell = [e for e in entries if e.model == model and e.dataset == dataset]
    lines = []
    lines.append(f"# Full per-config table — {model} / {dataset}")
    lines.append("")
    lines.append(
        f"All {len(cell)} configs found in the searched directories. Non-stochastic "
        "(greedy, beam) included here but flagged so you can see what was excluded "
        "from the 'best' tables."
    )
    lines.append("")
    lines.append("Sorted by pass@10 descending (NaNs last).")
    lines.append("")
    lines.append("| Config | Group | pass@1 | pass@10 | codebleu_div | n_tasks | stochastic? | metrics_path |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    sortable = sorted(
        cell,
        key=lambda e: (-(e.pass_at_10 if e.pass_at_10 is not None else -1.0)),
    )
    for e in sortable:
        group = "α" if e.is_alpha_arm else "other"
        stoch = "yes" if e.is_stochastic else "**no**"
        rel_path = str(e.metrics_path).replace(str(Path.cwd()) + "/", "")
        lines.append(
            f"| `{e.config}` | {group} | {_fmt_pct(e.pass_at_1)} | "
            f"{_fmt_pct(e.pass_at_10)} | {_fmt_div(e.codebleu_div)} | "
            f"{e.num_tasks or '—'} | {stoch} | `{rel_path}` |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    entries = collect_entries(repo_root)

    print(f"# Loaded {len(entries)} config-metric files across "
          f"{len(MODELS)} models × {len(set(e.dataset for e in entries))} datasets")
    by_cell = {}
    for e in entries:
        by_cell.setdefault((e.model, e.dataset), []).append(e)
    for (m, d), es in sorted(by_cell.items()):
        n_alpha = sum(1 for e in es if e.is_alpha_arm)
        n_stoch_other = sum(1 for e in es if not e.is_alpha_arm and e.is_stochastic)
        n_nonstoch = sum(1 for e in es if not e.is_stochastic)
        print(f"  {m:35} / {d:10}: α-arms={n_alpha:>3}  other-stochastic={n_stoch_other:>3}  non-stochastic={n_nonstoch:>3}")
    print()

    # Build outputs
    out_dir = repo_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_md = _best_config_table_md(entries)
    print(summary_md)
    (out_dir / "sampler_comparison_summary.md").write_text(summary_md)

    # Per-cell full tables
    cells = sorted({(e.model, e.dataset) for e in entries})
    for model, dataset in cells:
        slug_safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        ds_safe = dataset.lower()
        fname = f"full_table_{slug_safe}_{ds_safe}.md"
        (out_dir / fname).write_text(_full_table_md(entries, model, dataset))
        print(f"Wrote: {out_dir / fname}")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
