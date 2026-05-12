"""Partition tasks by truncation status to isolate token-budget vs temp/pless effects.

Each task in C (temp_think 0.6, 4096 tokens) is bucketed by how many of its
10 samples truncated. We then compute pass@1 for each bucket under several
configs (C, F at 4096; H1, H2, H3 at 8192). If the H1–C gap is concentrated
in tasks where C truncated heavily, the +10.6pp jump is mostly a budget
artefact. If H1 still beats C on tasks where C had zero truncation, the
temp/pless changes are genuinely helping.

Usage:
    uv run python -m bench.eval.truncation_partition_analysis \\
        --results-dir results/pless_full_mbpp_results/Qwen--Qwen3-8B \\
        --output results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis/truncation_partition.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


CONFIGS = {
    "C":  ("temp_think_t0.6.jsonl", 4096),
    "F":  ("split_temp_standard_t0.6_pless_t0.6_think_t1.0.jsonl", 4096),
    "H1": ("split_temp_standard_t0.7_pless_t1.0_think_t1.0.jsonl", 8192),
    "H2": ("split_temp_standard_t0.7_pless_t1.5_think_t1.0.jsonl", 8192),
    "H3": ("split_temp_standard_t0.7_pless_t2.0_think_t1.0.jsonl", 8192),
}


def trunc_per_task(jsonl_path: Path) -> dict[int, int]:
    """Count truncated samples per task: <think> opened but not closed."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            tid = r["task_id"]
            n_trunc = 0
            for s in r.get("samples_with_thinking", []):
                s = str(s)
                if "<think>" in s and "</think>" not in s:
                    n_trunc += 1
            out[tid] = n_trunc
    return out


def correct_per_task(metrics_path: Path) -> dict[int, int]:
    with open(metrics_path) as f:
        m = json.load(f)
    return {t["task_id"]: t["num_correct"] for t in m["per_task"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    metrics_dir = args.results_dir / "metrics"

    correct = {}
    for key, (fname, _) in CONFIGS.items():
        m_path = metrics_dir / fname.replace(".jsonl", "_metrics.json")
        correct[key] = correct_per_task(m_path)

    c_trunc = trunc_per_task(args.results_dir / CONFIGS["C"][0])
    h1_trunc = trunc_per_task(args.results_dir / CONFIGS["H1"][0])

    # Buckets by C's truncation count
    buckets = defaultdict(list)
    for tid, n in c_trunc.items():
        if n == 0:
            buckets["none (0/10)"].append(tid)
        elif n < 10:
            buckets["partial (1-9/10)"].append(tid)
        else:
            buckets["all (10/10)"].append(tid)

    bucket_order = ["none (0/10)", "partial (1-9/10)", "all (10/10)"]

    lines = []
    lines.append("# Truncation Partition Analysis — Qwen3-8B\n")
    lines.append("**Question:** Is the +10.6pp pass@1 gap (C → H1) a real effect "
                 "of the temp/pless change, or an artefact of the 4096 → 8192 token budget?\n")
    lines.append("**Method:** Partition the 500 MBPP tasks by how many of C's "
                 "10 samples truncated (hit the 4096 ceiling without closing `</think>`). "
                 "Then compute pass@1 for each partition under each config.\n")

    lines.append("## Bucket sizes (by C truncation count)\n")
    lines.append("| Bucket | Tasks |")
    lines.append("|--------|-------|")
    for b in bucket_order:
        lines.append(f"| {b} | {len(buckets[b])} |")
    lines.append("")

    lines.append("## pass@1 by partition\n")
    lines.append("Per-task pass@1 is `num_correct / 10`, then averaged over the bucket.\n")
    header = "| Bucket (n) | " + " | ".join(
        f"{k} ({CONFIGS[k][1]}t)" for k in CONFIGS) + " | Δ H1−C |"
    sep = "|------------|" + "|".join("-----" for _ in CONFIGS) + "|--------|"
    lines.append(header)
    lines.append(sep)

    bucket_pass1 = {}
    for b in bucket_order:
        tids = buckets[b]
        if not tids:
            continue
        row = [f"{b} ({len(tids)})"]
        cell_vals = {}
        for k in CONFIGS:
            vals = [correct[k][t] / 10.0 for t in tids if t in correct[k]]
            avg = sum(vals) / len(vals) if vals else 0
            cell_vals[k] = avg
            row.append(f"{avg:.4f}")
        delta = (cell_vals["H1"] - cell_vals["C"]) * 100
        sign = "+" if delta >= 0 else ""
        row.append(f"{sign}{delta:.1f}pp")
        bucket_pass1[b] = cell_vals
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Overall (re-derived as sanity check)
    lines.append("## Sanity check: overall pass@1\n")
    lines.append("| Config | pass@1 (recomputed) |")
    lines.append("|--------|---------------------|")
    for k in CONFIGS:
        vals = [v / 10.0 for v in correct[k].values()]
        lines.append(f"| {k} | {sum(vals)/len(vals):.4f} |")
    lines.append("")

    # Contribution analysis: how much of the C→H1 gap comes from each bucket?
    lines.append("## Decomposition of the C → H1 gap\n")
    lines.append("Contribution = (bucket size / 500) × (bucket pass@1 delta).\n")
    lines.append("| Bucket | Tasks | C pass@1 | H1 pass@1 | Δ | Contribution to total Δ |")
    lines.append("|--------|-------|----------|-----------|---|-------------------------|")
    total_contrib = 0.0
    for b in bucket_order:
        if b not in bucket_pass1:
            continue
        n = len(buckets[b])
        c_p = bucket_pass1[b]["C"]
        h_p = bucket_pass1[b]["H1"]
        d = h_p - c_p
        contrib = (n / 500) * d
        total_contrib += contrib
        lines.append(
            f"| {b} | {n} | {c_p:.4f} | {h_p:.4f} | {d:+.4f} | {contrib*100:+.2f}pp |"
        )
    lines.append(f"| **Total** | 500 | — | — | — | **{total_contrib*100:+.2f}pp** |\n")

    # Truncation in H1 itself (does H1 still truncate the C-all-trunc tasks?)
    lines.append("## Did H1 (8192 tokens) actually rescue the all-trunc-in-C tasks?\n")
    all_trunc_tids = buckets["all (10/10)"]
    if all_trunc_tids:
        h1_trunc_for_these = [h1_trunc.get(t, 0) for t in all_trunc_tids]
        avg = sum(h1_trunc_for_these) / len(h1_trunc_for_these)
        still_all_trunc = sum(1 for n in h1_trunc_for_these if n == 10)
        none_trunc = sum(1 for n in h1_trunc_for_these if n == 0)
        lines.append(f"- C all-trunc tasks: {len(all_trunc_tids)}")
        lines.append(f"- Of those, **still all-truncated in H1 (8192t):** {still_all_trunc}")
        lines.append(f"- Of those, **fully rescued (0 truncated samples in H1):** {none_trunc}")
        lines.append(f"- Avg truncated samples in H1 for these tasks: {avg:.2f}/10\n")

    # Interpretation
    lines.append("## Interpretation\n")
    if "none (0/10)" in bucket_pass1:
        d_clean = (bucket_pass1["none (0/10)"]["H1"]
                   - bucket_pass1["none (0/10)"]["C"]) * 100
        lines.append(
            f"- On tasks where C had **zero truncation** "
            f"(the cleanest comparison — token budget can't be helping H1 here), "
            f"H1 vs C delta is **{d_clean:+.1f}pp**."
        )
        if abs(d_clean) < 1.0:
            lines.append(
                f"  - This is essentially flat, suggesting the "
                f"**temp/pless setting change has no effect** in the no-truncation regime — "
                f"the +10.6pp overall gap is mostly a budget artefact."
            )
        elif d_clean > 0:
            lines.append(
                f"  - H1 is genuinely better even without budget help — "
                f"the temp/pless change contributes a real share of the gap."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
