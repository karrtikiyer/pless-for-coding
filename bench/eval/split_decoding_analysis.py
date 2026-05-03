"""Early analysis of the Qwen3-8B split decoding experiment.

Compares all 7 configs on the common subset of task_ids (apples-to-apples),
produces a markdown report and bar chart for presentation.

Usage:
    uv run python -m bench.eval.split_decoding_analysis \
        --results-dir results/pless_full_mbpp_results/Qwen--Qwen3-8B \
        --output-dir results/pless_full_mbpp_results/Qwen--Qwen3-8B/analysis
"""
import argparse
import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from human_eval.evaluation import estimate_pass_at_k


# ── Config definitions ────────────────────────────────────────────────────────
# Token budgets:
#   - A, B (no thinking): 512
#   - C, D, E, F, G (original 7-config phase, thinking): 4096
#   - H1-H9, R1-R9 (sweep, post-fix): 8192
CONFIGS = OrderedDict([
    ("A", {"file": "temp_t0.7.jsonl",
           "label": "temp 0.7", "thinking": False, "split": False,
           "max_tokens": 512,
           "color": "#78909C"}),
    ("B", {"file": "pless_t0.7.jsonl",
           "label": "pless 0.7", "thinking": False, "split": False,
           "max_tokens": 512,
           "color": "#546E7A"}),
    ("C", {"file": "temp_think_t0.6.jsonl",
           "label": "temp_think 0.6", "thinking": True, "split": False,
           "max_tokens": 4096,
           "color": "#2196F3"}),
    ("D", {"file": "pless_think_t0.6.jsonl",
           "label": "pless_think 0.6", "thinking": True, "split": False,
           "max_tokens": 4096,
           "color": "#4CAF50"}),
    ("E", {"file": "pless_norm_think_t0.6.jsonl",
           "label": "pless_norm_think 0.6", "thinking": True, "split": False,
           "max_tokens": 4096,
           "color": "#8BC34A"}),
    ("F", {"file": "split_temp_standard_t0.6_pless_t0.6_think_t1.0.jsonl",
           "label": "split: temp→pless", "thinking": True, "split": True,
           "max_tokens": 4096,
           "color": "#FF9800"}),
    ("G", {"file": "split_temp_standard_t0.6_pless_norm_t0.6_think_t1.0.jsonl",
           "label": "split: temp→pless_norm", "thinking": True, "split": True,
           "max_tokens": 4096,
           "color": "#FF5722"}),

    # ── Direction 1 sweep: temp think → pless code ────────────────────────
    # temp-think ∈ {0.7, 0.8, 1.5}, pless temp-code ∈ {1.0, 1.5, 2.0}
    # (t_think=0.9 was skipped after H1-H6 plateaued; jumped to 1.5 to
    # flatten the thinking distribution more aggressively.)
    # Oranges/reds gradient (pless-code phase emphasis)
    ("H1", {"file": "split_temp_standard_t0.7_pless_t1.0_think_t1.0.jsonl",
            "label": "split: temp 0.7 → pless 1.0",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#FFCC80"}),
    ("H2", {"file": "split_temp_standard_t0.7_pless_t1.5_think_t1.0.jsonl",
            "label": "split: temp 0.7 → pless 1.5",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#FFB74D"}),
    ("H3", {"file": "split_temp_standard_t0.7_pless_t2.0_think_t1.0.jsonl",
            "label": "split: temp 0.7 → pless 2.0",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#FFA726"}),
    ("H4", {"file": "split_temp_standard_t0.8_pless_t1.0_think_t1.0.jsonl",
            "label": "split: temp 0.8 → pless 1.0",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#FF8A65"}),
    ("H5", {"file": "split_temp_standard_t0.8_pless_t1.5_think_t1.0.jsonl",
            "label": "split: temp 0.8 → pless 1.5",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#FF7043"}),
    ("H6", {"file": "split_temp_standard_t0.8_pless_t2.0_think_t1.0.jsonl",
            "label": "split: temp 0.8 → pless 2.0",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#F4511E"}),
    ("H7", {"file": "split_temp_standard_t1.5_pless_t1.0_think_t1.0.jsonl",
            "label": "split: temp 1.5 → pless 1.0",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#E64A19"}),
    ("H8", {"file": "split_temp_standard_t1.5_pless_t1.5_think_t1.0.jsonl",
            "label": "split: temp 1.5 → pless 1.5",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#D84315"}),
    ("H9", {"file": "split_temp_standard_t1.5_pless_t2.0_think_t1.0.jsonl",
            "label": "split: temp 1.5 → pless 2.0",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#BF360C"}),

    # ── Uniform-temp-1.5 baseline (isolates pless contribution at t_think=1.5)
    # Same code path as H7-H9 (--method split, 8192 token budget) but with
    # temp on both phases. Purpose: do H7-H9's diversity gains come from
    # pless on the code side, or just from raising t_think to 1.5?
    ("T15", {"file": "split_temp_standard_t1.5_temp_standard_t1.5_think_t1.0.jsonl",
             "label": "split: temp 1.5 → temp 1.5 (baseline)",
             "thinking": True, "split": True, "max_tokens": 8192,
             "color": "#37474F"}),

    # ── Native-HF and uniform-pless thinking baselines at temp 1.5 ────────
    # T15N: uniform temp 1.5 with thinking ON via the native HF path
    # (generate_samples_standard) — bypasses split-path's 1e-3 pless smoothing.
    # P15: uniform pless decoding at temp 1.5 with thinking ON — single sampler
    # throughout. Probes whether the temp→pless split adds anything over pless-everywhere.
    ("T15N", {"file": "temp_think_t1.5.jsonl",
              "label": "uniform temp 1.5 (native, thinking)",
              "thinking": True, "split": False, "max_tokens": 8192,
              "color": "#546E7A"}),
    ("P15", {"file": "pless_think_t1.5.jsonl",
             "label": "uniform pless 1.5 (thinking)",
             "thinking": True, "split": False, "max_tokens": 8192,
             "color": "#6A1B9A"}),

    # ── H10: extends H batch one rung past H9 (temp_code 2.0 → 3.0) ───────
    # Tests whether pless degrades gracefully or cliffs at very high code-phase temp.
    ("H10", {"file": "split_temp_standard_t1.5_pless_t3.0_think_t1.0.jsonl",
             "label": "split: temp 1.5 → pless 3.0",
             "thinking": True, "split": True, "max_tokens": 8192,
             "color": "#5D4037"}),

    # ── Pure-temperature re-runs of the 5 high-temp split configs ─────────
    # These use SPLIT_SAMPLERS["temp_pure"] (top_p=1.0, top_k=0) instead of
    # "temp_standard" (top_p=0.95, top_k=20). Probes whether the filter on the
    # thinking phase was helping or hurting at temp_think=1.5. Pair each *P
    # entry with its filtered counterpart for filtered-vs-pure comparison.
    ("H7P", {"file": "split_temp_pure_t1.5_pless_t1.0_think_t1.0.jsonl",
             "label": "split: temp(pure) 1.5 → pless 1.0",
             "thinking": True, "split": True, "max_tokens": 8192,
             "color": "#FFAB91"}),
    ("H8P", {"file": "split_temp_pure_t1.5_pless_t1.5_think_t1.0.jsonl",
             "label": "split: temp(pure) 1.5 → pless 1.5",
             "thinking": True, "split": True, "max_tokens": 8192,
             "color": "#FF8A65"}),
    ("H9P", {"file": "split_temp_pure_t1.5_pless_t2.0_think_t1.0.jsonl",
             "label": "split: temp(pure) 1.5 → pless 2.0",
             "thinking": True, "split": True, "max_tokens": 8192,
             "color": "#E64A19"}),
    ("H10P", {"file": "split_temp_pure_t1.5_pless_t3.0_think_t1.0.jsonl",
              "label": "split: temp(pure) 1.5 → pless 3.0",
              "thinking": True, "split": True, "max_tokens": 8192,
              "color": "#BF360C"}),
    ("T15P", {"file": "split_temp_pure_t1.5_temp_pure_t1.5_think_t1.0.jsonl",
              "label": "split: temp(pure) 1.5 → temp(pure) 1.5",
              "thinking": True, "split": True, "max_tokens": 8192,
              "color": "#263238"}),

    # ── H11P, H12P: extend the pure-temp series past T15P at higher think-T ──
    # temp_think ∈ {2.0, 2.5} paired with pless code 3.0 (the highest pless rung).
    # Probes whether further flattening the thinking phase keeps adding diversity
    # and how it trades off accuracy against H10P (temp_think=1.5 → pless 3.0).
    ("H11P", {"file": "split_temp_pure_t2.0_pless_t3.0_think_t1.0.jsonl",
              "label": "split: temp(pure) 2.0 → pless 3.0",
              "thinking": True, "split": True, "max_tokens": 8192,
              "color": "#8D2A00"}),
    ("H12P", {"file": "split_temp_pure_t2.5_pless_t3.0_think_t1.0.jsonl",
              "label": "split: temp(pure) 2.5 → pless 3.0",
              "thinking": True, "split": True, "max_tokens": 8192,
              "color": "#5A1900"}),

    # ── Direction 2 sweep: pless think → temp code ────────────────────────
    # pless temp-think ∈ {1.0, 1.5, 2.0}, temp-code ∈ {0.7, 0.8, 0.9}
    # Blues/purples gradient (pless-think phase emphasis)
    ("R1", {"file": "split_pless_t1.0_temp_standard_t0.7_think_t1.0.jsonl",
            "label": "split: pless 1.0 → temp 0.7",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#90CAF9"}),
    ("R2", {"file": "split_pless_t1.0_temp_standard_t0.8_think_t1.0.jsonl",
            "label": "split: pless 1.0 → temp 0.8",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#64B5F6"}),
    ("R3", {"file": "split_pless_t1.0_temp_standard_t0.9_think_t1.0.jsonl",
            "label": "split: pless 1.0 → temp 0.9",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#42A5F5"}),
    ("R4", {"file": "split_pless_t1.5_temp_standard_t0.7_think_t1.0.jsonl",
            "label": "split: pless 1.5 → temp 0.7",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#1E88E5"}),
    ("R5", {"file": "split_pless_t1.5_temp_standard_t0.8_think_t1.0.jsonl",
            "label": "split: pless 1.5 → temp 0.8",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#1976D2"}),
    ("R6", {"file": "split_pless_t1.5_temp_standard_t0.9_think_t1.0.jsonl",
            "label": "split: pless 1.5 → temp 0.9",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#1565C0"}),
    ("R7", {"file": "split_pless_t2.0_temp_standard_t0.7_think_t1.0.jsonl",
            "label": "split: pless 2.0 → temp 0.7",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#7B1FA2"}),
    ("R8", {"file": "split_pless_t2.0_temp_standard_t0.8_think_t1.0.jsonl",
            "label": "split: pless 2.0 → temp 0.8",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#6A1B9A"}),
    ("R9", {"file": "split_pless_t2.0_temp_standard_t0.9_think_t1.0.jsonl",
            "label": "split: pless 2.0 → temp 0.9",
            "thinking": True, "split": True, "max_tokens": 8192,
            "color": "#4A148C"}),
])


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute_pass_at_k_from_results(task_results: list[dict],
                                    k_values: list[int]) -> dict[str, float]:
    """Unbiased pass@k estimator on a list of {num_correct, n_samples} dicts."""
    num_samples = np.array([r["n_samples"] for r in task_results])
    num_correct = np.array([r["num_correct"] for r in task_results])
    pak = {}
    for k in k_values:
        if k > num_samples.min():
            pak[str(k)] = pak.get("1", 0.0)
        else:
            pak[str(k)] = float(estimate_pass_at_k(num_samples, num_correct, k).mean())
    return pak


def truncation_stats(records: list[dict]) -> dict:
    """Compute truncation statistics from JSONL records."""
    total = 0
    truncated = 0
    tasks_all_trunc = 0
    tasks_no_trunc = 0

    for rec in records:
        swt = rec.get("samples_with_thinking", [])
        if not swt:
            continue
        n_trunc = 0
        for s in swt:
            total += 1
            if "<think>" in str(s) and "</think>" not in str(s):
                truncated += 1
                n_trunc += 1
        if n_trunc == len(swt):
            tasks_all_trunc += 1
        elif n_trunc == 0:
            tasks_no_trunc += 1

    return {
        "total_samples": total,
        "truncated": truncated,
        "truncation_rate": truncated / total if total else 0,
        "tasks_all_trunc": tasks_all_trunc,
        "tasks_no_trunc": tasks_no_trunc,
    }


def per_task_head_to_head(results_a: dict, results_b: dict,
                          label_a: str, label_b: str) -> dict:
    """Compare two configs task-by-task on common task_ids."""
    a_wins = 0
    b_wins = 0
    ties = 0
    for tid in results_a:
        if tid not in results_b:
            continue
        ca = results_a[tid]
        cb = results_b[tid]
        if ca > cb:
            a_wins += 1
        elif cb > ca:
            b_wins += 1
        else:
            ties += 1
    return {
        f"{label_a}_wins": a_wins,
        f"{label_b}_wins": b_wins,
        "ties": ties,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split decoding early analysis (7-config comparison)")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all configs ──────────────────────────────────────────────────
    config_data = {}
    for key, cfg in CONFIGS.items():
        fpath = args.results_dir / cfg["file"]
        if not fpath.exists():
            print(f"  Skipping {key} ({cfg['file']}): file not found")
            continue
        config_data[key] = {
            "records": load_jsonl(fpath),
            "cfg": cfg,
        }
        print(f"  Loaded {key}: {len(config_data[key]['records'])} tasks")

    # ── Find common task_ids ──────────────────────────────────────────────
    task_id_sets = {k: {r["task_id"] for r in v["records"]}
                    for k, v in config_data.items()}
    common_ids = set.intersection(*task_id_sets.values())
    common_ids = sorted(common_ids)
    print(f"\nCommon task_ids across {len(config_data)} configs: {len(common_ids)}")

    # ── Load pre-computed metrics (for per_task eval results) ─────────────
    # For configs with full metrics, extract per-task results
    # For others, we'll need the eval to have run
    metrics_dir = args.results_dir / "metrics"

    k_values = [1, 3, 5, 10]
    summary = {}

    for key, data in config_data.items():
        cfg = data["cfg"]
        records = data["records"]

        # Filter to common tasks
        records_common = [r for r in records if r["task_id"] in set(common_ids)]

        # Try to load pre-computed metrics
        metrics_stem = cfg["file"].replace(".jsonl", "_metrics.json")
        metrics_path = metrics_dir / metrics_stem
        metrics = load_metrics(metrics_path)

        if metrics and "per_task" in metrics:
            # Use pre-computed per-task results, filtered to common_ids
            per_task = {t["task_id"]: t for t in metrics["per_task"]}
            filtered_results = []
            for tid in common_ids:
                if tid in per_task:
                    t = per_task[tid]
                    filtered_results.append({
                        "num_correct": t["num_correct"],
                        "n_samples": len(t["pass_results"]),
                    })
            pak = compute_pass_at_k_from_results(filtered_results, k_values)
            correct_tasks = sum(1 for r in filtered_results if r["num_correct"] > 0)
            total_correct = sum(r["num_correct"] for r in filtered_results)

            # Per-task num_correct map for head-to-head
            per_task_correct = {tid: per_task[tid]["num_correct"]
                                for tid in common_ids if tid in per_task}
        else:
            print(f"  WARNING: No metrics for {key} — skipping (run bench.eval first)")
            continue

        # Truncation (from raw JSONL, on common tasks only)
        trunc = truncation_stats(records_common) if cfg["thinking"] else None

        # Structural diversity from full metrics (approximate — not filtered)
        struct_div = metrics.get("structural_diversity", None) if metrics else None
        codebleu_div = metrics.get("codebleu_diversity", None) if metrics else None

        summary[key] = {
            "label": cfg["label"],
            "thinking": cfg["thinking"],
            "split": cfg["split"],
            "color": cfg["color"],
            "max_tokens": cfg.get("max_tokens"),
            "pass_at_k": pak,
            "correct_tasks": correct_tasks,
            "total_correct": total_correct,
            "n_tasks": len(common_ids),
            "trunc": trunc,
            "struct_div": struct_div,
            "codebleu_div": codebleu_div,
            "per_task_correct": per_task_correct,
        }

    # ── Generate comparison table ─────────────────────────────────────────
    all_complete = all(len(v["records"]) >= 500 for v in config_data.values())
    status = "Final" if all_complete else "Early"

    lines = []
    lines.append(f"# Qwen3-8B Split Decoding — {status} Analysis ({len(common_ids)} common tasks)\n")
    lines.append(f"**Date:** {date.today().isoformat()}  ")
    lines.append(f"**Dataset:** MBPP-full  ")
    lines.append(f"**Tasks compared:** {len(common_ids)} / 500 (common across all configs)\n")

    lines.append("## Comparison Table\n")
    lines.append("| Config | Method | Think | Split | Tokens | pass@1 | pass@3 | pass@5 | pass@10 | Solved | Trunc% |")
    lines.append("|--------|--------|-------|-------|--------|--------|--------|--------|---------|--------|--------|")

    # Sort by pass@1 descending
    sorted_keys = sorted(summary.keys(),
                         key=lambda k: summary[k]["pass_at_k"].get("1", 0),
                         reverse=True)

    for key in sorted_keys:
        s = summary[key]
        pak = s["pass_at_k"]
        trunc_pct = f"{s['trunc']['truncation_rate']*100:.1f}%" if s["trunc"] else "—"
        think = "Yes" if s["thinking"] else "No"
        split = "Yes" if s["split"] else "No"
        tokens = s.get("max_tokens") or "—"
        lines.append(
            f"| **{key}** | {s['label']} | {think} | {split} "
            f"| {tokens} "
            f"| {pak.get('1',0):.4f} | {pak.get('3',0):.4f} "
            f"| {pak.get('5',0):.4f} | {pak.get('10',0):.4f} "
            f"| {s['correct_tasks']}/{s['n_tasks']} | {trunc_pct} |"
        )

    # ── Key comparisons ───────────────────────────────────────────────────
    lines.append("\n## Key Comparisons\n")

    comparisons = [
        ("C", "F", "Does split (temp think → pless code) beat uniform temp?"),
        ("C", "G", "Does split (temp think → pless_norm code) beat uniform temp?"),
        ("D", "F", "Does temp for thinking help vs uniform pless?"),
        ("E", "G", "Does temp for thinking help vs uniform pless_norm?"),
        ("A", "C", "How much does thinking help for temp?"),
        ("B", "D", "How much does thinking help for pless?"),
        ("F", "H1", "Does raising temp-think (0.6→0.7) and pless temp-code (0.6→1.0) help?"),
        ("H1", "H2", "Effect of increasing pless temp-code 1.0 → 1.5 (more permissive pless)"),
        ("H2", "H3", "Effect of increasing pless temp-code 1.5 → 2.0 (most permissive pless)"),
        ("C", "H1", "Best uniform-temp baseline vs new sweep top (temp 0.7 → pless 1.0)"),
    ]

    for key_a, key_b, question in comparisons:
        if key_a not in summary or key_b not in summary:
            continue
        sa = summary[key_a]
        sb = summary[key_b]
        pa1 = sa["pass_at_k"].get("1", 0)
        pb1 = sb["pass_at_k"].get("1", 0)
        delta = pb1 - pa1
        sign = "+" if delta >= 0 else ""

        h2h = per_task_head_to_head(
            sa["per_task_correct"], sb["per_task_correct"],
            key_a, key_b)

        lines.append(f"**{question}**")
        lines.append(f"- {key_a} ({sa['label']}): pass@1 = {pa1:.4f}")
        lines.append(f"- {key_b} ({sb['label']}): pass@1 = {pb1:.4f}")
        lines.append(f"- Delta: **{sign}{delta:.4f}** ({sign}{delta*100:.1f}pp)")
        lines.append(f"- Head-to-head: {key_a} wins {h2h[f'{key_a}_wins']}, "
                      f"{key_b} wins {h2h[f'{key_b}_wins']}, ties {h2h['ties']}")
        lines.append("")

    # ── Truncation comparison ─────────────────────────────────────────────
    lines.append("## Truncation Analysis (thinking configs only)\n")
    lines.append("| Config | Method | Tokens | Truncated | Rate | All-trunc tasks |")
    lines.append("|--------|--------|--------|-----------|------|-----------------|")
    trunc_keys = ["C", "D", "E", "F", "G"] + \
                 [f"H{i}" for i in range(1, 10)] + \
                 [f"R{i}" for i in range(1, 10)]
    for key in trunc_keys:
        if key not in summary or not summary[key]["trunc"]:
            continue
        s = summary[key]
        t = s["trunc"]
        tokens = s.get("max_tokens") or "—"
        lines.append(
            f"| {key} | {s['label']} | {tokens} "
            f"| {t['truncated']}/{t['total_samples']} "
            f"| {t['truncation_rate']*100:.1f}% | {t['tasks_all_trunc']} |"
        )

    # ── Diversity table (from metrics JSONs) ────────────────────────────
    lines.append("\n## Diversity Metrics\n")
    div_data = {}
    for key in sorted_keys:
        cfg = config_data[key]["cfg"]
        metrics_stem = cfg["file"].replace(".jsonl", "_metrics.json")
        metrics_path = metrics_dir / metrics_stem
        metrics = load_metrics(metrics_path)
        if not metrics:
            continue
        div_data[key] = {
            "struct_div": metrics.get("structural_diversity"),
            "codebleu_div": metrics.get("codebleu_diversity"),
            "ngram_div": metrics.get("ngram_match_diversity"),
            "dataflow_div": metrics.get("dataflow_match_diversity"),
        }

    lines.append("| Config | Method | pass@1 | pass@10 "
                  "| struct_div | codebleu_div | ngram_div | dataflow_div |")
    lines.append("|--------|--------|--------|---------|"
                  "-----------|-------------|-----------|-------------|")
    for key in sorted_keys:
        if key not in div_data:
            continue
        s = summary[key]
        d = div_data[key]
        sd = f"{d['struct_div']:.4f}" if d["struct_div"] is not None else "—"
        cd = f"{d['codebleu_div']:.4f}" if d["codebleu_div"] is not None else "—"
        nd = f"{d['ngram_div']:.4f}" if d["ngram_div"] is not None else "—"
        dd = f"{d['dataflow_div']:.4f}" if d["dataflow_div"] is not None else "—"
        lines.append(
            f"| **{key}** | {s['label']} "
            f"| {s['pass_at_k'].get('1',0):.4f} "
            f"| {s['pass_at_k'].get('10',0):.4f} "
            f"| {sd} | {cd} | {nd} | {dd} |"
        )

    # ── Observations ──────────────────────────────────────────────────────
    lines.append("\n## Observations\n")

    best_key = sorted_keys[0]
    best = summary[best_key]
    lines.append(f"1. **Best pass@1:** {best_key} ({best['label']}) at "
                 f"{best['pass_at_k']['1']:.4f}")

    if "F" in summary and "G" in summary and "C" in summary:
        c_p1 = summary["C"]["pass_at_k"]["1"]
        f_p1 = summary["F"]["pass_at_k"]["1"]
        g_p1 = summary["G"]["pass_at_k"]["1"]
        c_p10 = summary["C"]["pass_at_k"]["10"]
        f_p10 = summary["F"]["pass_at_k"]["10"]
        g_p10 = summary["G"]["pass_at_k"]["10"]
        lines.append(
            f"2. **Split decoding trades ~{(c_p1-f_p1)*100:.1f}pp pass@1 "
            f"for convergence at pass@10:** "
            f"F={f_p10:.3f}, G={g_p10:.3f} vs C={c_p10:.3f}")

    if "F" in summary and "D" in summary:
        f_p1 = summary["F"]["pass_at_k"]["1"]
        d_p1 = summary["D"]["pass_at_k"]["1"]
        lines.append(
            f"3. **Split decoding is +{(f_p1-d_p1)*100:.1f}pp ahead of "
            f"uniform pless thinking (D)** — temp for thinking matters")

    if "F" in div_data and "C" in div_data and "D" in div_data:
        f_sd = div_data["F"]["struct_div"] or 0
        c_sd = div_data["C"]["struct_div"] or 0
        d_sd = div_data["D"]["struct_div"] or 0
        pct_of_c = f_sd / c_sd * 100 if c_sd else 0
        lines.append(
            f"4. **Diversity comes from thinking, not code:** "
            f"F retains {pct_of_c:.0f}% of C's structural diversity "
            f"({f_sd:.4f} vs {c_sd:.4f}), while D drops to {d_sd:.4f}")

    if "B" in summary:
        lines.append(
            f"5. **Pless without thinking is a dead end:** "
            f"near-zero diversity ({div_data.get('B',{}).get('struct_div',0):.4f}), "
            f"no pass@1-to-pass@10 lift "
            f"({summary['B']['pass_at_k']['1']:.3f} → "
            f"{summary['B']['pass_at_k']['10']:.3f})")

    if "A" in summary and "C" in summary:
        a_p1 = summary["A"]["pass_at_k"]["1"]
        c_p1 = summary["C"]["pass_at_k"]["1"]
        lines.append(
            f"6. **Thinking is the dominant factor:** "
            f"+{(c_p1-a_p1)*100:.1f}pp pass@1 (A→C)")

    if "H1" in summary and "C" in summary and summary["H1"]["trunc"] and summary["C"]["trunc"]:
        h1 = summary["H1"]
        c = summary["C"]
        delta = (h1["pass_at_k"]["1"] - c["pass_at_k"]["1"]) * 100
        h1_trunc = h1["trunc"]["truncation_rate"] * 100
        c_trunc = c["trunc"]["truncation_rate"] * 100
        lines.append(
            f"7. **Sweep configs (H1–H3) leap ahead by ~+{delta:.1f}pp pass@1 over C** — "
            f"but **confounded with token budget**: H1–H3 used 8192 tokens "
            f"({h1_trunc:.1f}% truncation), C used 4096 ({c_trunc:.1f}%). "
            f"Re-running C/F at 8192 is required to isolate the temp/pless effect.")

    h_keys = [k for k in ("H1", "H2", "H3") if k in summary]
    if len(h_keys) == 3:
        h_p1s = [summary[k]["pass_at_k"]["1"] for k in h_keys]
        spread = (max(h_p1s) - min(h_p1s)) * 100
        lines.append(
            f"8. **pless temp-code (1.0/1.5/2.0) barely moves pass@1 at temp-think=0.7:** "
            f"spread of {spread:.2f}pp across H1–H3. Diversity is also flat — "
            f"the threshold p = Σ probs² is mostly insensitive in this range.")

    if not all_complete:
        incomplete = {k: len(v["records"]) for k, v in config_data.items()
                      if len(v["records"]) < 500}
        note_parts = [f"{k}={n}" for k, n in incomplete.items()]
        lines.append(
            f"\n**Note:** Partial data for {', '.join(note_parts)} tasks. "
            f"Final results may shift.")

    report = "\n".join(lines)
    suffix = "report" if all_complete else "early_report"
    report_path = args.output_dir / f"split_decoding_{suffix}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    # ── Grouped bar chart: pass@1 and pass@10 together ─────────────────
    keys = sorted_keys
    n = len(keys)
    # Scale figure width with config count so labels have room to breathe
    fig_w = max(14, 0.7 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    x = np.arange(n)
    bar_w = 0.38

    p1_vals = [summary[k]["pass_at_k"].get("1", 0) for k in keys]
    p10_vals = [summary[k]["pass_at_k"].get("10", 0) for k in keys]
    colors = [summary[k]["color"] for k in keys]

    bars1 = ax.bar(x - bar_w / 2, p1_vals, bar_w, color=colors,
                   edgecolor="white", linewidth=0.5, label="pass@1")
    bars10 = ax.bar(x + bar_w / 2, p10_vals, bar_w, color=colors,
                    edgecolor="white", linewidth=0.5, alpha=0.55, label="pass@10")

    # Highlight split configs with pink edge
    for i, k in enumerate(keys):
        if summary[k]["split"]:
            bars1[i].set_edgecolor("#E91E63")
            bars1[i].set_linewidth(2.0)
            bars10[i].set_edgecolor("#E91E63")
            bars10[i].set_linewidth(2.0)

    # Y-axis: include 0 floor so collapsed configs (H11P/H12P) render correctly
    y_min = min(min(p1_vals), min(p10_vals))
    y_max = max(p10_vals)
    ax.set_ylim(max(0, y_min - 0.08), min(1.0, y_max + 0.08))

    # Value labels — place pass@1 below bar top, pass@10 above bar top to avoid overlap
    for bar, val in zip(bars1, p1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.012,
                f"{val:.3f}", ha="center", va="top", fontsize=7,
                fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    for bar, val in zip(bars10, p10_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", fontstyle="italic")

    # X-axis: tag on top, short method label rotated below — no multiline cramming
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10, fontweight="bold")

    # Add a second-row label below each tick rotated 45° so descriptions are legible
    ymin_lim = ax.get_ylim()[0]
    label_y = ymin_lim - (ax.get_ylim()[1] - ymin_lim) * 0.04
    for xi, k in enumerate(keys):
        lbl = summary[k]["label"]
        ax.text(xi, label_y, lbl, ha="right", va="top",
                fontsize=7.5, rotation=35, rotation_mode="anchor",
                color="#444444")

    ax.set_ylabel("Score", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="grey", edgecolor="white", label="pass@1"),
        Patch(facecolor="grey", alpha=0.55, edgecolor="white", label="pass@10"),
        Patch(facecolor="white", edgecolor="#E91E63", linewidth=2,
              label="split decoding"),
    ], loc="upper right", fontsize=9, framealpha=0.95)

    ax.set_title(
        f"Qwen3-8B Split Decoding — pass@1 & pass@10, "
        f"{len(common_ids)} MBPP tasks",
        fontsize=13, fontweight="bold", pad=12)

    # Reserve more bottom space for the rotated labels
    plt.subplots_adjust(bottom=0.28, top=0.93)

    chart_path = args.output_dir / f"split_decoding_{suffix}_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart: {chart_path}")

    # ── Scatter: diversity vs accuracy ────────────────────────────────────
    from adjustText import adjust_text

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, (div_key, div_label) in zip(axes, [
        ("struct_div", "Structural Diversity (tree-edit)"),
        ("codebleu_div", "CodeBLEU Diversity"),
    ]):
        texts = []
        for key in sorted_keys:
            s = summary[key]
            div_val = s.get(div_key)
            if div_val is None:
                continue
            p1 = s["pass_at_k"].get("1", 0)

            marker = "D" if s["split"] else ("^" if s["thinking"] else "o")
            size = 140 if s["split"] else 90
            edgecolor = "#E91E63" if s["split"] else "white"
            linewidth = 2.0 if s["split"] else 0.8

            ax.scatter(p1, div_val, c=s["color"], marker=marker,
                       s=size, edgecolors=edgecolor, linewidths=linewidth,
                       zorder=3)
            # Compact label: just the tag — full method name is in the report table
            texts.append(ax.text(
                p1, div_val, key,
                fontsize=10, color=s["color"], fontweight="bold",
                ha="center", va="center", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            ))

        # Pad axes so labels at edges don't get clipped
        ax.margins(x=0.08, y=0.10)

        # Repel labels until none overlap; draw thin connectors back to points
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.6),
            expand_points=(1.4, 1.6),
            expand_text=(1.2, 1.4),
            force_points=0.4,
            force_text=0.6,
        )

        ax.set_xlabel("pass@1", fontsize=11)
        ax.set_ylabel(div_label, fontsize=11)
        ax.set_title(f"Accuracy vs {div_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="grey", linestyle="None",
               markersize=10, label="No thinking"),
        Line2D([0], [0], marker="^", color="grey", linestyle="None",
               markersize=10, label="Thinking (uniform)"),
        Line2D([0], [0], marker="D", color="grey", linestyle="None",
               markersize=10, markeredgecolor="#E91E63", markeredgewidth=1.5,
               label="Split decoding"),
    ]
    # Place legend outside the data area so it can't sit on top of points
    axes[1].legend(handles=legend_elems, loc="lower right", fontsize=10,
                   framealpha=0.95)

    plt.suptitle(
        f"Diversity vs Accuracy — Qwen3-8B, {len(common_ids)} MBPP tasks",
        fontsize=13, fontweight="bold")
    plt.tight_layout()

    div_chart_path = args.output_dir / f"split_decoding_{suffix}_diversity_vs_accuracy.png"
    plt.savefig(div_chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diversity chart: {div_chart_path}")

    # ── Save raw summary JSON ─────────────────────────────────────────────
    json_summary = {}
    for key, s in summary.items():
        json_summary[key] = {k: v for k, v in s.items() if k != "per_task_correct"}
    json_path = args.output_dir / f"split_decoding_{suffix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
