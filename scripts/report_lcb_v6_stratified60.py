"""Report pass@1/5/10, termination%, and mean token size for the LCB v6
stratified-60 pless-variants-vs-baselines comparison (Qwen3-8B).

Reuses ``bench.eval`` (via its CLI entrypoint, same pattern used all session
for the APPS tables) for pass@k / solved task_ids — no custom scoring logic.
Adds termination% (fraction of samples that closed their think block before
the token cap — marker is model-specific, see THINK_END_MARKER_BY_MODEL_SUBSTR)
and mean token size (via the model's own tokenizer, not a char-length proxy)
on top, then aggregates per-config across the two platform files and reports
a stratified breakdown by platform and by difficulty.

Usage::

    uv run python scripts/report_lcb_v6_stratified60.py \\
        --model Qwen/Qwen3-8B --configs pless_gini top_nsigma ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
# Think-close marker varies by model family (Qwen3/DeepSeek use "</think>",
# Ministral uses "[/THINK]" — see bench/generator.py:_strip_think_content).
# Same substring-match convention as RECIPE_BY_MODEL_SUBSTR above.
THINK_END_MARKER_BY_MODEL_SUBSTR = [
    ("qwen", "</think>"),
    ("ministral", "[/THINK]"),
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


# Source files whose mtimes are folded into the cache fingerprint -- any edit
# to bench.eval's LCB-relevant scoring logic (timeout formula, extraction
# strategy, think-marker resolution, etc.) touches one of these and therefore
# auto-invalidates every cache entry, with no hand-maintained version
# constant to remember to bump (reviewed finding: a manual int living in this
# script, disconnected from bench/eval/, would silently serve stale cache
# after exactly the kind of fix made to lcb_executor.py earlier this session).
_EVAL_SOURCE_FILES = [
    REPO / "bench" / "eval" / "__main__.py",
    REPO / "bench" / "eval" / "metrics.py",
    REPO / "bench" / "eval" / "lcb_executor.py",
    REPO / "bench" / "eval" / "apps_executor.py",
    REPO / "bench" / "eval" / "apps_extractor.py",
    REPO / "bench" / "eval" / "think_markers.py",
]

# Metrics compared for the deviation check (scalar / small-dict fields only --
# per_task is intentionally excluded, it's large and any real scoring change
# already shows up in these aggregates).
_DEVIATION_CHECK_KEYS = [
    "pass_at_k", "cover_at_t", "cover_at_t_distinct",
    "termination_pct", "mean_token_size", "timeout_pct",
]
_DEVIATION_TOLERANCE = 1e-6


def _eval_source_fingerprint() -> list[int]:
    """mtime_ns of every file in _EVAL_SOURCE_FILES, in list order. Missing
    files (e.g. a future rename) contribute -1 rather than raising, so a
    stale file list degrades to "always looks changed" instead of crashing."""
    out = []
    for p in _EVAL_SOURCE_FILES:
        try:
            out.append(p.stat().st_mtime_ns)
        except OSError:
            out.append(-1)
    return out


def _fingerprint_jsonl(jsonl_path: Path) -> dict:
    st = jsonl_path.stat()
    return {
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "eval_source_fingerprint": _eval_source_fingerprint(),
    }


def _values_deviate(old_val, new_val, path: str) -> list[str]:
    """Recursively diff two JSON-shaped values (dict/float/None), returning
    human-readable mismatch descriptions. Only called when the jsonl's own
    content (mtime+size) is confirmed unchanged -- a mismatch here means
    either (a) bench.eval's scoring code changed since the cached file was
    written (expected, and not a bug -- e.g. a timeout-formula fix), or
    (b) genuine nondeterminism in execution (flaky timeout under CPU load).
    This function can't tell which; the caller's warning names both."""
    if isinstance(old_val, dict) and isinstance(new_val, dict):
        mismatches = []
        for key in set(old_val) | set(new_val):
            mismatches += _values_deviate(old_val.get(key), new_val.get(key), f"{path}.{key}")
        return mismatches
    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        if abs(old_val - new_val) > _DEVIATION_TOLERANCE:
            return [f"{path}: {old_val!r} -> {new_val!r}"]
        return []
    if old_val != new_val:
        return [f"{path}: {old_val!r} -> {new_val!r}"]
    return []


def _load_json_or_none(path: Path) -> dict | None:
    """json.load that treats a corrupt/unreadable file as "no cache" rather
    than crashing the whole script -- guards against a prior run's write
    having been interrupted (Ctrl-C, SIGKILL, disk full) leaving truncated
    JSON on disk."""
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _run_bench_eval(jsonl_path: Path, *, force_reeval: bool = False) -> dict | None:
    """Invoke bench.eval on one file, return its metrics dict (or None if
    the file is missing / eval fails).

    Caches on (jsonl mtime+size, bench.eval source mtimes): if the results
    file AND every file in _EVAL_SOURCE_FILES are unchanged since the last
    run that produced the existing metrics JSON, skip the subprocess call
    and return the cached metrics directly (this report re-evaluates every
    config on every invocation otherwise -- see docs/theory/todos.md A64's
    "known standing inefficiency").

    When the jsonl's content is confirmed unchanged (same mtime+size) but we
    re-run anyway -- ``force_reeval=True``, or an eval source file's mtime
    changed -- the freshly computed metrics are diffed against the cached
    ones and any deviation is printed as a warning naming both possible
    causes (an intentional scoring-code change, or genuine nondeterminism)
    rather than asserting either.
    """
    if not jsonl_path.exists():
        return None
    metrics_path = jsonl_path.parent / "metrics" / f"{jsonl_path.stem}_metrics.json"

    current_fp = _fingerprint_jsonl(jsonl_path)
    cached = None
    content_unchanged = False
    if metrics_path.exists():
        cached = _load_json_or_none(metrics_path)
        cached_fp = cached.get("_eval_cache_fingerprint") if cached is not None else None
        if cached_fp is not None:
            content_unchanged = (
                cached_fp.get("mtime_ns") == current_fp["mtime_ns"]
                and cached_fp.get("size") == current_fp["size"]
            )
            if not force_reeval and cached_fp == current_fp:
                return cached  # cache hit: jsonl + eval source both unchanged

    subprocess.run(
        [sys.executable, "-m", "bench.eval", "--dataset", "livecodebench",
         "--results-file", str(jsonl_path), "--skip-diversity"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    if not metrics_path.exists():
        return None
    fresh = json.load(metrics_path.open())

    if content_unchanged and cached is not None:
        mismatches = []
        for key in _DEVIATION_CHECK_KEYS:
            mismatches += _values_deviate(cached.get(key), fresh.get(key), key)
        if mismatches:
            print(f"WARNING: re-eval of unchanged {jsonl_path.name} produced different "
                  f"metrics. Since the input didn't change, this means either (a) "
                  f"bench.eval's scoring code genuinely changed (expected, not a bug), "
                  f"or (b) nondeterministic execution (e.g. a flaky timeout under CPU "
                  f"load) -- this function can't tell which:")
            for m in mismatches:
                print(f"  {m}")

    fresh["_eval_cache_fingerprint"] = current_fp
    tmp_fd, tmp_name = tempfile.mkstemp(dir=metrics_path.parent, suffix=".tmp")
    try:
        with open(tmp_fd, "w") as f:
            json.dump(fresh, f, indent=2)
        os.replace(tmp_name, metrics_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return fresh


def _think_end_marker_for_model(model: str) -> str:
    model_lower = model.lower()
    for substr, marker in THINK_END_MARKER_BY_MODEL_SUBSTR:
        if substr in model_lower:
            return marker
    raise ValueError(
        f"No think-end-marker mapping for model {model!r} — add it to "
        f"THINK_END_MARKER_BY_MODEL_SUBSTR."
    )


def _termination_and_tokens(jsonl_path: Path, tokenizer, think_end_marker: str) -> dict[str, dict]:
    """Per-task_id: {closed_frac, mean_tokens, n_samples}."""
    out = {}
    if not jsonl_path.exists():
        return out
    for line in jsonl_path.open():
        r = json.loads(line)
        swt = r.get("samples_with_thinking", r.get("samples", []))
        if not swt:
            continue
        closed = sum(think_end_marker in s for s in swt)
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
    ap.add_argument("--force-reeval", action="store_true",
                     help="Bypass the eval cache and re-run bench.eval even for "
                          "configs whose results file hasn't changed. Any deviation "
                          "from the cached metrics is printed as a warning (either an "
                          "intentional scoring-code change not yet reflected in the "
                          "cache, or genuine nondeterminism on identical input).")
    args = ap.parse_args()
    if args.configs is None:
        args.configs = _default_configs_for_model(args.model)
    think_end_marker = _think_end_marker_for_model(args.model)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.model} (CPU-only, no GPU needed)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_slug = args.model.replace("/", "--")
    out_root = REPO / OUT_ROOT_TEMPLATE.format(model_slug=model_slug)

    from bench.eval.headline_table import TERMINAL_COLUMNS, headline_table
    from bench.eval.metrics import compute_cover_at_t, compute_pass_at_k

    K_VALUES = [1, 3, 5, 8, 10]
    T_VALUES = [0.3, 0.5, 0.7]

    all_rows = {}
    failed_configs: list[str] = []
    pending_configs: list[str] = []
    complete_table_rows: list[tuple[str, str]] = []  # (tmp_json_path, config) -- combined-platform metrics
    tmpdir = tempfile.TemporaryDirectory()
    try:
        for config in args.configs:
            task_meta: dict[str, dict] = {}
            combined_pass_results: list[dict] = []
            timeout_counts: list[int] = []
            sample_counts: list[int] = []
            num_samples_per_task = 10
            try:
                for platform in args.platforms:
                    # temperature varies by config (qwen_recipe=0.6, rest=1.0) — glob for it.
                    candidates = list(out_root.glob(f"{config}_{platform}_think_t*.jsonl"))
                    if not candidates:
                        continue
                    jsonl_path = candidates[0]
                    m = _run_bench_eval(jsonl_path, force_reeval=args.force_reeval)
                    tm = _termination_and_tokens(jsonl_path, tokenizer, think_end_marker)
                    task_meta.update(tm)
                    if m:
                        for t in m.get("per_task", []):
                            combined_pass_results.append(t)
                        num_samples_per_task = m.get("num_samples_per_task", num_samples_per_task)
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
                pending_configs.append(config)
                all_rows[config] = None
                continue

            pak = compute_pass_at_k(
                [{"task_id": t["task_id"], "num_correct": t.get("num_correct", 0),
                  "pass_results": t.get("pass_results", [None] * t.get("num_samples", 10))}
                 for t in combined_pass_results],
                K_VALUES,
            )
            cov, cov_distinct = compute_cover_at_t(
                combined_pass_results, T_VALUES, num_samples_per_task
            )
            solved = sum(1 for t in combined_pass_results if t.get("num_correct", 0) > 0)
            n_total = len(combined_pass_results)
            term_fracs = [v["closed_frac"] for v in task_meta.values()]
            tok_means = [v["mean_tokens"] for v in task_meta.values()]
            term_pct = 100 * sum(term_fracs) / max(len(term_fracs), 1)
            mean_tok = sum(tok_means) / max(len(tok_means), 1)
            timeout_pct = 100 * sum(timeout_counts) / max(sum(sample_counts), 1) if sample_counts else 0.0

            # Combined-platform metrics in the same shape build_metrics_output()
            # writes, so headline_table() renders this identically to any other
            # metrics JSON -- one canonical format, not a hand-rolled subset.
            synthetic_metrics = {
                "num_tasks": n_total,
                "pass_at_k": pak,
                "cover_at_t": cov,
                "cover_at_t_distinct": cov_distinct,
                "termination_pct": term_pct,
                "mean_token_size": mean_tok,
                "timeout_pct": timeout_pct,
                "structural_diversity": None,
                "codebleu_diversity": None,
            }
            tmp_path = Path(tmpdir.name) / f"{config}.json"
            tmp_path.write_text(json.dumps(synthetic_metrics))
            complete_table_rows.append((tmp_path, config))

            all_rows[config] = {
                "pass_at_k": pak, "solved": solved, "n_total": n_total,
                "cover_at_t": cov, "cover_at_t_distinct": cov_distinct,
                "termination_pct": term_pct, "mean_tokens": mean_tok,
                "timeout_pct": timeout_pct, "task_meta": task_meta,
            }

        print(f"\n{args.model} -- LCB v6 stratified-60 (canonical columns; N=60 means both "
              f"platforms fully done)")
        print(headline_table(complete_table_rows, columns=TERMINAL_COLUMNS))
        if pending_configs:
            print(f"\nPending (no data yet, omitted above): {', '.join(pending_configs)}")
        if failed_configs:
            print(f"Eval failed (omitted above): {', '.join(failed_configs)}")
    finally:
        tmpdir.cleanup()
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
