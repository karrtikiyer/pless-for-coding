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
# Script-mode invocation (`uv run python scripts/report_lcb_v6_stratified60.py`,
# as opposed to `-m`/`-c` mode) sets sys.path[0] to this file's own directory
# (scripts/), not the caller's cwd -- so `from bench.eval.metrics import ...`
# below fails with ModuleNotFoundError regardless of cwd. Reproduced directly
# 2026-09-29 when scripts/sync_and_eval_lcb_v6.sh (invoked via the auto-sync
# watcher) hit this for real. Other scripts in this repo work around the same
# root cause by requiring callers to export PYTHONPATH manually (see gpu2
# launch gotchas in docs/theory/todos.md A63) -- inserting REPO here instead
# makes this script self-sufficient rather than depending on every caller
# remembering an env var.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT_ROOT_TEMPLATE = "results/pless_variants_cuda/{model_slug}/LCB_v6_stratified60"

# Kept in sync by hand with CONFIG_NAMES in scripts/run_pless_variants_qwen3_8b_lcb_cuda.py
# and scripts/run_pless_variants_ministral_3b_lcb_cuda.py (both define the same 22-config
# set, differing only in the vendor-recipe entry). If those scripts' CONFIGS lists change,
# update COMMON_CONFIGS / RECIPE_BY_MODEL_SUBSTR here too.
COMMON_CONFIGS = [
    "top_nsigma", "top_nsigma_1_5", "gk_0_1", "gk_0_35",
    "pless_default", "pless_effk", "pless_min2", "pless_topmass", "pless_gini", "pless_top1half",
    "pless_default_t1_5", "pless_default_t2_0", "pless_default_t2_5", "pless_default_t3_0",
    "top_p_1_0", "top_p_0_95", "top_p_0_9", "top_p_0_85", "top_p_0_8", "top_p_0_75", "top_p_0_7",
]
RECIPE_BY_MODEL_SUBSTR = [
    ("qwen", "qwen_recipe"),
    ("ministral", "ministral_recipe"),
]
ALL_CONFIGS = COMMON_CONFIGS + [r for _, r in RECIPE_BY_MODEL_SUBSTR]
PLATFORMS = ["atcoder", "leetcode"]


def _default_configs_for_model(model: str) -> list[str]:
    model_lower = model.lower()
    for substr, recipe in RECIPE_BY_MODEL_SUBSTR:
        if substr in model_lower:
            return COMMON_CONFIGS + [recipe]
    raise ValueError(
        f"No recipe-config mapping for model {model!r} — add it to RECIPE_BY_MODEL_SUBSTR."
    )


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
    ap.add_argument("--configs", nargs="+", default=None, choices=ALL_CONFIGS,
                     help="Defaults to the full config set for --model's vendor recipe "
                          "(auto-detected from the model name).")
    ap.add_argument("--platforms", nargs="+", default=PLATFORMS, choices=PLATFORMS)
    args = ap.parse_args()
    if args.configs is None:
        args.configs = _default_configs_for_model(args.model)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.model} (CPU-only, no GPU needed)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_slug = args.model.replace("/", "--")
    out_root = REPO / OUT_ROOT_TEMPLATE.format(model_slug=model_slug)

    print(f"\n{'Config':<16} {'p@1':>7} {'p@5':>7} {'p@10':>7} {'Solved':>8} {'Term%':>7} "
          f"{'MeanTok':>8} {'Timeout%':>9}")
    print("-" * 80)

    all_rows = {}
    failed_configs: list[str] = []
    try:
        for config in args.configs:
            task_meta: dict[str, dict] = {}
            combined_pass_results: list[dict] = []
            timeout_counts: list[int] = []
            sample_counts: list[int] = []
            try:
                for platform in args.platforms:
                    # temperature varies by config (qwen_recipe=0.6, rest=1.0) — glob for it.
                    candidates = list(out_root.glob(f"{config}_{platform}_think_t*.jsonl"))
                    if not candidates:
                        continue
                    jsonl_path = candidates[0]
                    m = _run_bench_eval(jsonl_path)
                    tm = _termination_and_tokens(jsonl_path, tokenizer)
                    task_meta.update(tm)
                    if m:
                        for t in m.get("per_task", []):
                            combined_pass_results.append(t)
                        # Surface a signal that's already computed by
                        # apps_executor.py/lcb_executor.py (status_counts, incl.
                        # "Timeout" distinctly from "Failed"/"RuntimeError") but was
                        # never read anywhere before — not a new measurement, just
                        # exposing what execution_diagnostics.by_status already has.
                        by_status = m.get("execution_diagnostics", {}).get("by_status", {})
                        if by_status:
                            timeout_counts.append(by_status.get("Timeout", 0))
                            sample_counts.append(sum(by_status.values()))
            except subprocess.CalledProcessError as exc:
                print(f"{config:<16} EVAL FAILED (skipping this config, continuing): {exc}")
                if exc.stderr:
                    print(f"  stderr: {exc.stderr.strip()[:2000]}")
                failed_configs.append(config)
                all_rows[config] = None
                continue

            if not combined_pass_results:
                print(f"{config:<16} {'--':>7} {'--':>7} {'--':>7} {'pending':>8} {'--':>7} "
                      f"{'--':>8} {'--':>9}")
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
            timeout_pct = 100 * sum(timeout_counts) / max(sum(sample_counts), 1) if sample_counts else 0.0

            print(f"{config:<16} {pak.get('1',0):>7.3f} {pak.get('5',0):>7.3f} "
                  f"{pak.get('10',0):>7.3f} {solved:>4}/{n_total:<3} {term_pct:>6.1f}% "
                  f"{mean_tok:>8.0f} {timeout_pct:>8.1f}%")

            all_rows[config] = {
                "pass_at_k": pak, "solved": solved, "n_total": n_total,
                "termination_pct": term_pct, "mean_tokens": mean_tok,
                "timeout_pct": timeout_pct, "task_meta": task_meta,
            }
    finally:
        # Write whatever was collected even if a config eval above crashed
        # unexpectedly outside the per-config try/except (e.g. KeyboardInterrupt
        # is re-raised, but ordinary exceptions during the loop body should
        # already be caught above) — partial results beat no results when this
        # runs unattended.
        out_json = out_root / "analysis" / "lcb_v6_stratified60_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(all_rows, out_json.open("w"), indent=2, default=str)
        print(f"\nWrote {out_json}")
        if failed_configs:
            print(f"WARNING: {len(failed_configs)} config(s) failed eval and were skipped: {failed_configs}")

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


if __name__ == "__main__":
    main()
