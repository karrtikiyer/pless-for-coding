"""Report pass@1/5/10, termination%, and mean token size for the LCB v6
stratified-60 pless-variants-vs-baselines comparison (Qwen3-8B).

Reuses ``bench.eval`` (via its CLI entrypoint, same pattern used all session
for the APPS tables) for pass@k / solved task_ids — no custom scoring logic.
Adds termination% (fraction of samples that closed ``</think>`` before the
token cap) and mean token size (via the model's own tokenizer, not a char-
length proxy) on top, then aggregates per-config across the two platform
files and reports a stratified breakdown by platform and by difficulty.

Usage::

    uv run python scripts/report_lcb_v6_stratified60.py \\
        --model Qwen/Qwen3-8B --configs pless_gini top_nsigma ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT_TEMPLATE = "results/pless_variants_cuda/{model_slug}/LCB_v6_stratified60"

ALL_CONFIGS = [
    "top_nsigma", "top_nsigma_1_5", "gk_0_1", "gk_0_35", "qwen_recipe",
    "pless_default", "pless_effk", "pless_topmass", "pless_gini", "pless_top1half",
    "pless_default_t1_5", "pless_default_t2_0",
]
PLATFORMS = ["atcoder", "leetcode"]


def _run_bench_eval(jsonl_path: Path) -> dict | None:
    """Invoke bench.eval on one file, return its metrics dict (or None if
    the file is missing / eval fails)."""
    if not jsonl_path.exists():
        return None
    metrics_path = jsonl_path.parent / "metrics" / f"{jsonl_path.stem}_metrics.json"
    subprocess.run(
        [sys.executable, "-m", "bench.eval", "--dataset", "livecodebench",
         "--results-file", str(jsonl_path), "--skip-diversity"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    if not metrics_path.exists():
        return None
    return json.load(metrics_path.open())


def _termination_and_tokens(jsonl_path: Path, tokenizer) -> dict[str, dict]:
    """Per-task_id: {closed_frac, mean_tokens, n_samples}."""
    out = {}
    if not jsonl_path.exists():
        return out
    for line in jsonl_path.open():
        r = json.loads(line)
        swt = r.get("samples_with_thinking", r.get("samples", []))
        if not swt:
            continue
        closed = sum("</think>" in s for s in swt)
        token_counts = [len(tokenizer.encode(s)) for s in swt]
        out[str(r["task_id"])] = {
            "closed_frac": closed / len(swt),
            "mean_tokens": sum(token_counts) / len(token_counts),
            "n_samples": len(swt),
            "difficulty": r.get("difficulty"),
            "platform": r.get("source"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    ap.add_argument("--platforms", nargs="+", default=PLATFORMS, choices=PLATFORMS)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.model} (CPU-only, no GPU needed)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_slug = args.model.replace("/", "--")
    out_root = REPO / OUT_ROOT_TEMPLATE.format(model_slug=model_slug)

    print(f"\n{'Config':<16} {'p@1':>7} {'p@5':>7} {'p@10':>7} {'Solved':>8} {'Term%':>7} {'MeanTok':>8}")
    print("-" * 70)

    all_rows = {}
    for config in args.configs:
        task_meta: dict[str, dict] = {}
        combined_pass_results: list[dict] = []
        for platform in args.platforms:
            jsonl_path = out_root / f"{config}_{platform}_think_t{{temp}}.jsonl"
            # temperature varies by config (qwen_recipe=0.6, rest=1.0) — glob for it.
            candidates = list(out_root.glob(f"{config}_{platform}_think_t*.jsonl"))
            if not candidates:
                continue
            jsonl_path = candidates[0]
            m = _run_bench_eval(jsonl_path)
            tm = _termination_and_tokens(jsonl_path, tokenizer)
            task_meta.update(tm)
            if m:
                for t in m.get("per_task", m.get("task_details", [])):
                    combined_pass_results.append(t)

        if not combined_pass_results:
            print(f"{config:<16} {'--':>7} {'--':>7} {'--':>7} {'pending':>8} {'--':>7} {'--':>8}")
            all_rows[config] = None
            continue

        from bench.eval.metrics import compute_pass_at_k
        pak = compute_pass_at_k(
            [{"task_id": t["task_id"], "num_correct": t.get("num_correct", 0),
              "pass_results": t.get("pass_results", [None] * t.get("num_samples", 10))}
             for t in combined_pass_results],
            [1, 5, 10],
        )
        solved = sum(1 for t in combined_pass_results if t.get("num_correct", 0) > 0)
        n_total = len(combined_pass_results)
        term_fracs = [v["closed_frac"] for v in task_meta.values()]
        tok_means = [v["mean_tokens"] for v in task_meta.values()]
        term_pct = 100 * sum(term_fracs) / max(len(term_fracs), 1)
        mean_tok = sum(tok_means) / max(len(tok_means), 1)

        print(f"{config:<16} {pak.get('1',0):>7.3f} {pak.get('5',0):>7.3f} "
              f"{pak.get('10',0):>7.3f} {solved:>4}/{n_total:<3} {term_pct:>6.1f}% {mean_tok:>8.0f}")

        all_rows[config] = {
            "pass_at_k": pak, "solved": solved, "n_total": n_total,
            "termination_pct": term_pct, "mean_tokens": mean_tok,
            "task_meta": task_meta,
        }

    # Stratified breakdown by (platform, difficulty)
    print(f"\n{'='*20} Stratified pass@10 (by platform x difficulty) {'='*20}")
    for config, row in all_rows.items():
        if row is None:
            continue
        print(f"\n{config}:")
        cells = {}
        for tid, meta in row["task_meta"].items():
            key = (meta["platform"], meta["difficulty"])
            cells.setdefault(key, []).append(tid)
        for plat in args.platforms:
            for diff in ["easy", "medium", "hard"]:
                ids = cells.get((plat, diff), [])
                print(f"  {plat:<10} {diff:<8} n={len(ids)}")

    out_json = out_root / "analysis" / "lcb_v6_stratified60_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_rows, out_json.open("w"), indent=2, default=str)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
