"""
List APPS-252 (ATCODER_interview) tasks where pless α=2 (T=1.0) failed on all 10
samples on Qwen3-8B (think), but at least one non-pless decoder (temp / top-k /
top-p, any temperature in the 252-set) solved on ≥1 of 10 samples.

For each such task, emit boolean flags saying which pless-family variants
(pless_norm, pless@t0.6, pless@t2.0, τ_α α∈{3,4,5}, G_k k∈{0.05..1.6}) also
solved (num_correct ≥ 1). Output: CSV + markdown table under
results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/analysis/.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

REPO = Path("/Users/karrtikiyer/projects/airesearch/pless-for-coding")
Q3 = "Qwen--Qwen3-8B"
APPS = "ATCODER_interview"
ALL252 = "ATCODER_interview_all_252"

VLLM = REPO / "results/pless_cot_efficiency_vllm" / Q3 / ALL252 / "metrics"
TOPP = REPO / "results/_top_p_sweep_full252" / Q3 / APPS / "metrics"
RENYI = REPO / "results/_renyi_sweep_full252" / Q3 / APPS / "metrics"
RECOV = REPO / "results/pless_recovery_full252" / Q3 / APPS / "metrics"
DEC06 = REPO / "results/decoders_t0.6" / Q3 / APPS / "metrics"

OUT_DIR = REPO / "results/pless_cot_efficiency_vllm" / Q3 / ALL252 / "analysis"

BASELINE = VLLM / "pless_think_t1.0_t1.0_metrics.json"

NON_PLESS = {
    "temp@t1.0":        TOPP / "temp_think_t1.0_t1.0_metrics.json",
    "temp@t0.6":        VLLM / "temp_think_t0.6_t0.6_metrics.json",
    "temp_k20@t1.0":    VLLM / "temp_k20_think_t1.0_t1.0_metrics.json",
    "temp_k20@t0.6":    DEC06 / "temp_k20_think_t0.6_t0.6_metrics.json",
    "temp_p0.95@t1.0":  VLLM / "temp_p0.95_think_t1.0_t1.0_metrics.json",
    "temp_p0.95@t0.6":  DEC06 / "temp_p0.95_think_t0.6_t0.6_metrics.json",
    "temp_p0.95_k20@t0.6": VLLM / "temp_p0.95_k20_think_t0.6_t0.6_metrics.json",
    "top_p0.7@t1.0":    TOPP / "temp_p0.7_think_t1.0_t1.0_metrics.json",
    "top_p0.75@t1.0":   TOPP / "temp_p0.75_think_t1.0_t1.0_metrics.json",
    "top_p0.8@t1.0":    TOPP / "temp_p0.8_think_t1.0_t1.0_metrics.json",
    "top_p0.85@t1.0":   TOPP / "temp_p0.85_think_t1.0_t1.0_metrics.json",
    "top_p0.9@t1.0":    TOPP / "temp_p0.9_think_t1.0_t1.0_metrics.json",
}

PLESS_VARIANTS = {
    "pless_norm@t1.0":  VLLM / "pless_norm_think_t1.0_t1.0_metrics.json",
    "pless_norm@t0.6":  DEC06 / "pless_norm_think_t0.6_t0.6_metrics.json",
    "pless@t0.6":       DEC06 / "pless_think_t0.6_t0.6_metrics.json",
    "pless@t2.0":       RECOV / "pless_think_t2.0_t2.0_metrics.json",
    "tau_alpha_a3":     RECOV / "pless_alpha_think_t1.0_a3.0_t1.0_metrics.json",
    "tau_alpha_a4":     RECOV / "pless_alpha_think_t1.0_a4.0_t1.0_metrics.json",
    "tau_alpha_a5":     RECOV / "pless_alpha_think_t1.0_a5.0_t1.0_metrics.json",
    "G_k_0.05":         RENYI / "pless_renyi_think_t1.0_k0.05_t1.0_metrics.json",
    "G_k_0.1":          RENYI / "pless_renyi_think_t1.0_k0.1_t1.0_metrics.json",
    "G_k_0.2":          RENYI / "pless_renyi_think_t1.0_k0.2_t1.0_metrics.json",
    "G_k_0.25":         RENYI / "pless_renyi_think_t1.0_k0.25_t1.0_metrics.json",
    "G_k_0.3":          RENYI / "pless_renyi_think_t1.0_k0.3_t1.0_metrics.json",
    "G_k_0.35":         RENYI / "pless_renyi_think_t1.0_k0.35_t1.0_metrics.json",
    "G_k_0.4":          RENYI / "pless_renyi_think_t1.0_k0.4_t1.0_metrics.json",
    "G_k_0.8":          RENYI / "pless_renyi_think_t1.0_k0.8_t1.0_metrics.json",
    "G_k_1.6":          RENYI / "pless_renyi_think_t1.0_k1.6_t1.0_metrics.json",
}


def load_per_task(path: Path) -> dict[int, int]:
    """Return {task_id: num_correct} from a metrics JSON."""
    data = json.loads(path.read_text())
    return {row["task_id"]: row["num_correct"] for row in data["per_task"]}


def main() -> None:
    for p in [BASELINE, *NON_PLESS.values(), *PLESS_VARIANTS.values()]:
        if not p.exists():
            raise FileNotFoundError(p)

    base = load_per_task(BASELINE)
    non_pless = {name: load_per_task(p) for name, p in NON_PLESS.items()}
    variants = {name: load_per_task(p) for name, p in PLESS_VARIANTS.items()}

    pless_failed = [tid for tid, nc in base.items() if nc == 0]
    rows = []
    for tid in sorted(pless_failed):
        np_solvers = [n for n, m in non_pless.items() if m.get(tid, 0) >= 1]
        if not np_solvers:
            continue
        row = {
            "task_id": tid,
            "pless_a2_t1.0_num_correct": 0,
            "num_non_pless_solvers": len(np_solvers),
            "non_pless_solvers": ";".join(np_solvers),
        }
        for name, m in variants.items():
            nc = m.get(tid, 0)
            row[f"{name}_solved"] = int(nc >= 1)
            row[f"{name}_nc"] = nc
        row["any_pless_variant_solved"] = int(any(
            variants[name].get(tid, 0) >= 1 for name in variants
        ))
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pless_a2_failures_solved_by_others.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = OUT_DIR / "pless_a2_failures_solved_by_others.md"
    variant_names = list(variants.keys())
    lines = [
        "# APPS-252 tasks where pless α=2 (T=1.0) failed but another decoder solved",
        "",
        f"Model: **Qwen3-8B** (think). Dataset: **{APPS}** (full 252 subset).",
        "",
        "- **pless α=2 failed** = 0/10 samples pass at `pless@T=1.0`.",
        "- **Non-pless solved** = ≥1/10 samples pass at any of: "
        + ", ".join(NON_PLESS.keys()) + ".",
        f"- Variant flag columns show which pless-family variants also solved "
        f"(≥1/10). {len(variant_names)} variants tracked.",
        "",
        f"**Count: {len(rows)} tasks** meet the criterion. "
        f"Of these, "
        f"{sum(r['any_pless_variant_solved'] for r in rows)} are recovered "
        "by ≥1 pless-family variant.",
        "",
        "## Summary table",
        "",
        "| task_id | # non-pless solvers | any pless variant? | "
        + " | ".join(variant_names) + " |",
        "|---:|---:|:---:|" + "|".join([":---:"] * len(variant_names)) + "|",
    ]
    for r in rows:
        cells = [
            str(r["task_id"]),
            str(r["num_non_pless_solvers"]),
            "yes" if r["any_pless_variant_solved"] else "—",
        ]
        for name in variant_names:
            cells.append(str(r[f"{name}_nc"]) if r[f"{name}_solved"] else "·")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Cell values in variant columns = num_correct out of 10 samples "
        "(`·` = 0). See `pless_a2_failures_solved_by_others.csv` for the "
        "full per-task table incl. the specific non-pless solver names.",
    ])
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"pless-α=2-failed tasks: {len(pless_failed)}")
    print(f"…of which solved by ≥1 non-pless decoder: {len(rows)}")
    print(f"…of those, also solved by ≥1 pless-family variant: "
          f"{sum(r['any_pless_variant_solved'] for r in rows)}")


if __name__ == "__main__":
    main()
