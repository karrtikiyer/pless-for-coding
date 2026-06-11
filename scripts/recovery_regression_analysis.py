"""Paired per-problem regression analysis: do the recovery variants hurt problems
the baseline (pless T1.0) already handled?

For each variant arm vs the baseline, on the SAME 252 problems:
  - partition problems by baseline strength (strong: baseline num_correct>=8;
    mid: 3-7; weak: <=2 — the truncation-prone set the variants target)
  - per partition, mean per-problem pass@1 (num_correct/n) for baseline vs variant
  - count improved / unchanged / regressed, and the HARD regressions
    (solved-by-baseline >=1 → 0-by-variant) — the thing we actually worry about

Usage:
  python scripts/recovery_regression_analysis.py \
    --variant-dir <out>/metrics \
    --baseline-metrics <baseline>/metrics/pless_think_t1.0_t1.0_metrics.json \
    --out <out>/analysis/regression_vs_baseline.md
"""
import argparse
import glob
import json
import os


def load_per_task(path):
    m = json.load(open(path))
    return {t["task_id"]: sum(t["pass_results"]) for t in m["per_task"]}, \
           len(m["per_task"][0]["pass_results"]) if m["per_task"] else 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", required=True, help="metrics/ dir with variant *_metrics.json")
    ap.add_argument("--baseline-metrics", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, n = load_per_task(args.baseline_metrics)
    base_name = os.path.basename(args.baseline_metrics).replace("_metrics.json", "")

    variants = sorted(f for f in glob.glob(f"{args.variant_dir}/*_metrics.json"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lines = ["# Recovery variants vs baseline — paired per-problem regression\n"]
    lines.append(f"Baseline: `{base_name}` over {len(base)} problems (n={n}).  ")
    lines.append("Per-problem pass@1 = num_correct/n. Partitions by BASELINE strength: "
                 "**strong** (≥8/n), **mid** (3–7), **weak** (≤2, the truncation-prone set).\n")
    lines.append("**Regression** = variant num_correct < baseline. **HARD regression** = "
                 "baseline solved (≥1) → variant 0 (lost the problem entirely).\n")

    for vpath in variants:
        var, _ = load_per_task(vpath)
        vname = os.path.basename(vpath).replace("_metrics.json", "")
        common = sorted(set(base) & set(var))
        if not common:
            continue

        def part(pred):
            ids = [t for t in common if pred(base[t])]
            b = sum(base[t] for t in ids) / (len(ids) * n) if ids else 0
            v = sum(var[t] for t in ids) / (len(ids) * n) if ids else 0
            return ids, b, v

        strong = part(lambda c: c >= 8)
        mid = part(lambda c: 3 <= c <= 7)
        weak = part(lambda c: c <= 2)

        improved = [t for t in common if var[t] > base[t]]
        regressed = [t for t in common if var[t] < base[t]]
        hard_reg = sorted(t for t in common if base[t] >= 1 and var[t] == 0)
        gained = sorted(t for t in common if base[t] == 0 and var[t] >= 1)
        overall_b = sum(base[t] for t in common) / (len(common) * n)
        overall_v = sum(var[t] for t in common) / (len(common) * n)

        lines.append(f"\n## {vname}\n")
        lines.append(f"Overall pass@1: baseline **{overall_b:.3f}** → variant **{overall_v:.3f}** "
                     f"(Δ {overall_v - overall_b:+.3f}) over {len(common)} problems.\n")
        lines.append("| baseline partition | #probs | baseline pass@1 | variant pass@1 | Δ |")
        lines.append("|---|---|---|---|---|")
        for name, (ids, b, v) in [("strong (≥8)", strong), ("mid (3–7)", mid), ("weak (≤2)", weak)]:
            lines.append(f"| {name} | {len(ids)} | {b:.3f} | {v:.3f} | {v - b:+.3f} |")
        lines.append("")
        lines.append(f"- improved: **{len(improved)}**  | regressed: **{len(regressed)}**  | "
                     f"unchanged: {len(common) - len(improved) - len(regressed)}")
        lines.append(f"- **HARD regressions** (solved→lost): **{len(hard_reg)}** {hard_reg or ''}")
        lines.append(f"- gained (lost→solved): {len(gained)} {gained or ''}")
        # net effect on the STRONG set is the regression worry
        s_ids, s_b, s_v = strong
        s_reg = sorted(t for t in s_ids if var[t] < base[t])
        s_hard = sorted(t for t in s_ids if var[t] == 0)
        lines.append(f"- on STRONG problems: {len(s_reg)} regressed, {len(s_hard)} lost entirely "
                     f"({'CLEAN — no regression on easy problems' if not s_hard else 'CHECK these'})")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    # also echo a one-line verdict per arm to stdout
    for vpath in variants:
        var, _ = load_per_task(vpath)
        vname = os.path.basename(vpath).replace("_metrics.json", "")
        common = sorted(set(base) & set(var))
        hard = sum(1 for t in common if base[t] >= 1 and var[t] == 0)
        ov = sum(var[t] for t in common) / (len(common) * n) - sum(base[t] for t in common) / (len(common) * n)
        print(f"  {vname}: overall pass@1 Δ {ov:+.3f}, hard regressions {hard}")


if __name__ == "__main__":
    main()
