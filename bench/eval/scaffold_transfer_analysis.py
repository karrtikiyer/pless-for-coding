"""Analyse the external-scaffold transfer experiment: treatment vs control.

Both conditions are Qwen3-8B on the same never-solved APPS tasks with thinking
OFF; the only difference is whether the prompt carried a Claude-Opus algorithm
scaffold (treatment) or not (control/baseline). A task is "solved" by a
condition iff at least one sample passes (``num_correct > 0``). The headline
number is ``newly_recovered = treatment_solved - baseline_solved``.

CLI::

    uv run python -m bench.eval.scaffold_transfer_analysis \\
        --baseline-metrics <baseline _metrics.json> \\
        --treatment-metrics <treatment _metrics.json> \\
        --out results/scaffold_transfer/analysis/pilot_n5.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _per_task_num_correct(metrics: dict) -> dict[int, int]:
    return {int(e["task_id"]): int(e.get("num_correct", 0))
            for e in metrics.get("per_task", [])}


def compute_transfer(baseline: dict, treatment: dict) -> dict:
    """Reduce two metrics dicts to per-task and set-level transfer stats.

    Considers the union of task_ids present in either condition. pass@1 per
    task is ``num_correct / num_samples_per_task`` (per-condition sample count).
    """
    base_nc = _per_task_num_correct(baseline)
    treat_nc = _per_task_num_correct(treatment)
    base_n = int(baseline.get("num_samples_per_task", 0)) or 1
    treat_n = int(treatment.get("num_samples_per_task", 0)) or 1
    task_ids = sorted(set(base_nc) | set(treat_nc))

    baseline_solved = {t for t in task_ids if base_nc.get(t, 0) > 0}
    treatment_solved = {t for t in task_ids if treat_nc.get(t, 0) > 0}

    rows = []
    for t in task_ids:
        b = base_nc.get(t, 0)
        tr = treat_nc.get(t, 0)
        rows.append({
            "task_id": t,
            "baseline_num_correct": b,
            "treatment_num_correct": tr,
            "baseline_pass_at_1": b / base_n,
            "treatment_pass_at_1": tr / treat_n,
            "newly_recovered": (b == 0 and tr > 0),
            "regressed": (b > 0 and tr == 0),
        })

    return {
        "n_tasks": len(task_ids),
        "baseline_solved": baseline_solved,
        "treatment_solved": treatment_solved,
        "newly_recovered": treatment_solved - baseline_solved,
        "regressions": baseline_solved - treatment_solved,
        "baseline_pass_at_k": baseline.get("pass_at_k", {}),
        "treatment_pass_at_k": treatment.get("pass_at_k", {}),
        "rows": rows,
    }


def _render_markdown(r: dict, *, baseline_path: str, treatment_path: str) -> str:
    lines = [
        "# External-scaffold transfer — treatment vs control",
        "",
        f"- baseline (no scaffold, thinking off): `{baseline_path}`",
        f"- treatment (Claude-Opus scaffold, thinking off): `{treatment_path}`",
        "",
        f"- tasks: **{r['n_tasks']}**",
        f"- solved — baseline **{len(r['baseline_solved'])}** → "
        f"treatment **{len(r['treatment_solved'])}**",
        f"- **newly recovered** (0 → solved by scaffold): "
        f"**{len(r['newly_recovered'])}** {sorted(r['newly_recovered'])}",
        f"- regressions (baseline solved → treatment 0): "
        f"**{len(r['regressions'])}** {sorted(r['regressions'])}",
        f"- dataset pass@k — baseline {r['baseline_pass_at_k']} | "
        f"treatment {r['treatment_pass_at_k']}",
        "",
        "| task_id | base n_correct | treat n_correct | base pass@1 | treat pass@1 | recovered |",
        "|---|---|---|---|---|---|",
    ]
    for row in r["rows"]:
        lines.append(
            f"| {row['task_id']} | {row['baseline_num_correct']} | "
            f"{row['treatment_num_correct']} | {row['baseline_pass_at_1']:.2f} | "
            f"{row['treatment_pass_at_1']:.2f} | "
            f"{'✅' if row['newly_recovered'] else ''} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-metrics", type=Path, required=True)
    ap.add_argument("--treatment-metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    baseline = json.loads(args.baseline_metrics.read_text())
    treatment = json.loads(args.treatment_metrics.read_text())
    r = compute_transfer(baseline, treatment)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(
        r, baseline_path=str(args.baseline_metrics),
        treatment_path=str(args.treatment_metrics),
    )
    args.out.write_text(md)
    print(md)
    print(f"[scaffold-transfer] wrote {args.out}")


if __name__ == "__main__":
    main()
