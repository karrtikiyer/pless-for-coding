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
from datetime import datetime, timezone
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
    "pless_default", "pless_effk", "pless_effk_nofloor", "pless_min2",
    "pless_topmass", "pless_topmass_nofloor", "pless_gini", "pless_gini_nofloor", "pless_top1half",
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
    """Per-task_id: {closed_frac, mean_tokens, n_samples}.

    ``closed_frac`` is None (not a deterministic 0.0) when the record's own
    ``enable_thinking`` field is explicitly False -- same fix as
    ``bench.eval.metrics.compute_termination_and_tokens`` (independent
    review, 2026-10-08): a disabled think phase can structurally never
    contain the closing marker, so counting it as "closed_frac=0" would
    silently misrepresent "no think phase by design" as "always rambles."
    """
    out = {}
    if not jsonl_path.exists():
        return out
    for line in jsonl_path.open():
        r = json.loads(line)
        swt = r.get("samples_with_thinking", r.get("samples", []))
        if not swt:
            continue
        token_counts = [len(tokenizer.encode(s)) for s in swt]
        think_phase_expected = bool(r.get("enable_thinking", True))
        closed_frac = (sum(think_end_marker in s for s in swt) / len(swt)) if think_phase_expected else None
        out[str(r["task_id"])] = {
            "closed_frac": closed_frac,
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
    ap.add_argument("--no-thinking", action="store_true",
                     help="Look for enable_thinking=False ('*_nothink_t*.jsonl') results "
                          "instead of the default reasoning-mode ('*_think_t*.jsonl') "
                          "files -- matches run_pless_variants_qwen3_8b_lcb_cuda.py's "
                          "--no-thinking flag. Independent review (2026-10-08) found the "
                          "discovery glob was 'think'-only, so nothink data would silently "
                          "show as pending/no-data rather than actually being reported.")
    args = ap.parse_args()
    if args.configs is None:
        args.configs = _default_configs_for_model(args.model)
    think_end_marker = _think_end_marker_for_model(args.model)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.model} (CPU-only, no GPU needed)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_slug = args.model.replace("/", "--")
    out_root = REPO / OUT_ROOT_TEMPLATE.format(model_slug=model_slug)

    from bench.eval.headline_table import COLUMN_DESCRIPTIONS, TERMINAL_COLUMNS, headline_table
    from bench.eval.metrics import compute_cover_at_t, compute_pass_at_k

    K_VALUES = [1, 3, 5, 8, 10]
    T_VALUES = [0.3, 0.5, 0.7]

    all_rows = {}
    failed_configs: list[str] = []
    pending_configs: list[str] = []
    complete_table_rows: list[tuple[str, str, float]] = []  # (tmp_json_path, config, pass@1) -- combined-platform metrics
    combined_results_by_config: dict[str, list[dict]] = {}
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
                    # Suffix is explicit ("think" vs "nothink"), not a wildcard: once
                    # both flavors exist for the same config, a wildcard would match
                    # both and candidates[0] would arbitrarily pick one with no warning.
                    think_suffix = "nothink" if args.no_thinking else "think"
                    candidates = list(out_root.glob(f"{config}_{platform}_{think_suffix}_t*.jsonl"))
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
            # closed_frac is None for enable_thinking=False records (no
            # think phase by design) -- exclude from the average rather
            # than crashing or silently treating None as 0.
            term_fracs = [v["closed_frac"] for v in task_meta.values() if v["closed_frac"] is not None]
            tok_means = [v["mean_tokens"] for v in task_meta.values()]
            term_pct = 100 * sum(term_fracs) / len(term_fracs) if term_fracs else None
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
            complete_table_rows.append((tmp_path, config, pak.get("1", 0.0)))

            all_rows[config] = {
                "pass_at_k": pak, "solved": solved, "n_total": n_total,
                "cover_at_t": cov, "cover_at_t_distinct": cov_distinct,
                "termination_pct": term_pct, "mean_tokens": mean_tok,
                "timeout_pct": timeout_pct, "task_meta": task_meta,
            }
            # Kept out of all_rows (and thus out of the JSON dump below) --
            # full per-task pass_results/extracted_codes would bloat the
            # summary JSON and duplicate data already in metrics/*.json.
            # Only used in-process for the stratified markdown table.
            combined_results_by_config[config] = combined_pass_results

    # Table-building and the markdown write both live in `finally` below (not
    # here) so a partial/failed run still regenerates both output files
    # (review finding, 2026-10-08: they used to live after the whole
    # try/finally, so an exception anywhere in the per-config loop skipped
    # them entirely, leaving progress_report.md stale behind a successfully-
    # written JSON -- silently reintroducing the exact staleness bug (A65)
    # this feature exists to eliminate).
    finally:
        # Leaderboard table must be built BEFORE tmpdir.cleanup() -- it reads
        # the per-config temp JSON files headline_table() expects on disk.
        # Sort by pass@1 descending -- the leaderboard's whole point is
        # ranking configs, so the raw per-config-loop order (whatever order
        # --configs happened to list them in) isn't useful to leave as-is.
        sorted_table_rows = sorted(complete_table_rows, key=lambda row: -row[2])
        leaderboard_rows = [(path, label) for path, label, _ in sorted_table_rows]
        leaderboard_md = headline_table(leaderboard_rows, columns=TERMINAL_COLUMNS)
        tmpdir.cleanup()

        # Write whatever was collected even if a config eval above crashed
        # unexpectedly outside the per-config try/except (e.g. KeyboardInterrupt
        # is re-raised, but ordinary exceptions during the loop body should
        # already be caught above) — partial results beat no results when this
        # runs unattended.
        # "_nothink" output discriminator -- same collision class the input-side
        # glob fix addresses: without it, a --no-thinking run would silently
        # overwrite the normal reasoning-mode report (review finding, 2026-10-08).
        out_suffix = "_nothink" if args.no_thinking else ""
        out_json = out_root / "analysis" / f"lcb_v6_stratified60_report{out_suffix}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(all_rows, out_json.open("w"), indent=2, default=str)
        print(f"\nWrote {out_json}")
        if failed_configs:
            print(f"WARNING: {len(failed_configs)} config(s) failed eval and were skipped: {failed_configs}")

        print(f"\n{args.model} -- LCB v6 stratified-60 (canonical columns; N=60 means both "
              f"platforms fully done)")
        print(leaderboard_md)
        if pending_configs:
            print(f"\nPending (no data yet, omitted above): {', '.join(pending_configs)}")
        if failed_configs:
            print(f"Eval failed (omitted above): {', '.join(failed_configs)}")

        # Stratified pass@1 by (platform, difficulty) -- real per-cell pass@1,
        # not just task counts (the previous version of this script only
        # printed n=10 sanity counts per cell, never actual stratified
        # performance numbers). Every populated cell shows its own task count
        # (n=X) alongside the percentage -- a config with a partial/in-progress
        # cell (e.g. 3 of the intended 10 tasks) must never render in a way
        # that's visually indistinguishable from a fully-populated cell
        # (review finding, 2026-10-08: the first version only distinguished
        # "zero data -> —" from "any data -> {pct}%", so a 3-task partial cell
        # rendered as a plain, equally-confident-looking percentage).
        strat_header = ["Config"] + [
            f"{plat[:2].upper()}-{diff[:1].upper()}" for plat in args.platforms for diff in ("easy", "medium", "hard")
        ]
        strat_rows_md = ["| " + " | ".join(strat_header) + " |",
                         "|" + "|".join([":---"] + ["---:"] * (len(strat_header) - 1)) + "|"]
        for config in args.configs:
            row = all_rows.get(config)
            if row is None:
                continue
            results = combined_results_by_config.get(config, [])
            task_meta = row["task_meta"]
            cells_vals = []
            for plat in args.platforms:
                for diff in ("easy", "medium", "hard"):
                    cell_results = [
                        r for r in results
                        if task_meta.get(str(r["task_id"]), {}).get("platform") == plat
                        and task_meta.get(str(r["task_id"]), {}).get("difficulty") == diff
                    ]
                    if not cell_results:
                        cells_vals.append("—")
                        continue
                    cell_pak = compute_pass_at_k(
                        [{"task_id": r["task_id"], "num_correct": r.get("num_correct", 0),
                          "pass_results": r.get("pass_results", [None] * r.get("num_samples", 10))}
                         for r in cell_results],
                        [1],
                    )
                    # Only flag the count when it deviates from the stratified
                    # design's intended 10-tasks-per-(platform,difficulty)-cell
                    # -- showing "(n=10)" on every normal cell is redundant;
                    # showing it on a genuinely partial cell (n<10) is the
                    # signal that actually matters (ship-blocking review
                    # finding, 2026-10-08 -- see docs/theory/todos.md E12).
                    n_cell = len(cell_results)
                    suffix = "" if n_cell == 10 else f" (n={n_cell})"
                    cells_vals.append(f"{100*cell_pak.get('1', 0):.1f}%{suffix}")
            strat_rows_md.append("| " + " | ".join([config] + cells_vals) + " |")
        stratified_md = "\n".join(strat_rows_md)
        print(f"\n{'='*20} Stratified pass@1 (by platform x difficulty) {'='*20}")
        print(stratified_md)

        # Auto-generated markdown report -- single source of truth, regenerated
        # fresh every run (including partial/crashed runs, per the fix above),
        # impossible to go stale the way a hand-edited doc did (see
        # docs/theory/todos.md A65's staleness incident). Same format for
        # every model; only the title/model line differs.
        report_md = [
            f"# LCB v6 Stratified-60 — Progress Report ({args.model})",
            "",
            "**Auto-generated by `scripts/report_lcb_v6_stratified60.py` — do not hand-edit.** "
            "Regenerate with the same command shown below; this file and "
            "`analysis/lcb_v6_stratified60_report.json` are always in sync (same run, same data).",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Dataset: LiveCodeBench v6, 60 problems stratified 10/10/10 (easy/medium/hard) "
            "across AtCoder + LeetCode. n=10 samples/task, cap=25,000 tokens, seed=42 "
            "(vendor-recipe configs use their own temperature; all others T=1.0).",
            "",
            f"Regenerate: `uv run python scripts/report_lcb_v6_stratified60.py --model {args.model}"
            f"{' --no-thinking' if args.no_thinking else ''}`",
            "",
            "## Leaderboard (sorted by pass@1, descending)",
            "",
            leaderboard_md,
            "",
            "### Column definitions",
            "",
        ]
        report_md += [
            "| Column | Meaning |",
            "|:---|:---|",
        ] + [f"| `{col}` | {COLUMN_DESCRIPTIONS[col]} |" for col in TERMINAL_COLUMNS] + [""]
        if pending_configs:
            report_md += [f"**Pending** (no data yet): {', '.join(pending_configs)}", ""]
        if failed_configs:
            report_md += [f"**Eval failed**: {', '.join(failed_configs)}", ""]
        report_md += [
            "## Stratified pass@1 (by platform × difficulty)",
            "",
            "`AT`/`LE` = AtCoder/LeetCode; `E`/`M`/`H` = easy/medium/hard. Each cell is "
            "pass@1 within that 10-task stratum. A cell shows `(n=X)` only when X "
            "deviates from the intended 10 tasks — i.e. a partial/in-progress config, "
            "not a fully-reliable stratum estimate; no suffix means n=10 (complete).",
            "",
            stratified_md,
            "",
        ]
        out_md = out_root / "analysis" / f"progress_report{out_suffix}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(report_md))
        print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
