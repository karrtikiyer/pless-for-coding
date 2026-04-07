"""Backfill missing CodeBLEU subcomponent fields in existing metrics JSONs.

Adds per-task `self_ngram_match` and `self_weighted_ngram_match` fields,
plus top-level `ngram_match_diversity` and `weighted_ngram_match_diversity`.

Does NOT re-run code execution — only re-computes CodeBLEU pairwise scores
using correct samples identified by existing `pass_results`.

Usage:
    uv run python -m bench.eval.patch_codebleu_subcomponents [--dry-run] [--workers 4]
"""

import argparse
import ctypes
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean as _mean

from bench.eval.executor import strip_code_fences
from bench.eval.fingerprint import ast_fingerprint


RESULTS_ROOT = Path("results")

# Directories to skip entirely
_SKIP_PATTERNS = ("_backup", "metrics_before_fix", "consolidated_metrics")


def _patch_tree_sitter_capsule():
    """Patch tree-sitter 0.22 Language to accept PyCapsule from tree-sitter-python 0.23+."""
    from tree_sitter import Language

    if getattr(Language, "_capsule_patched", False):
        return
    _orig_init = Language.__init__

    def _patched_init(self, *args, **kwargs):
        if args and type(args[0]).__name__ == "PyCapsule":
            ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
            ptr_fn.restype = ctypes.c_void_p
            ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
            ptr = ptr_fn(args[0], b"tree_sitter.Language")
            return _orig_init(self, ptr, *args[1:], **kwargs)
        return _orig_init(self, *args, **kwargs)

    Language.__init__ = _patched_init
    Language._capsule_patched = True


def discover_metrics_files() -> list[Path]:
    """Find all canonical metrics JSONs under results/, skipping copies and backups."""
    all_files = []
    for p in RESULTS_ROOT.rglob("*_metrics.json"):
        path_str = str(p)
        if any(skip in path_str for skip in _SKIP_PATTERNS):
            continue
        # Only include files inside a `metrics/` directory (canonical location)
        if p.parent.name != "metrics":
            continue
        all_files.append(p)
    return sorted(all_files)


def derive_jsonl_path(metrics_path: Path) -> Path | None:
    """Map metrics/{stem}_metrics.json -> ../{stem}.jsonl

    Tries two locations:
    1. ../{stem}.jsonl (standard MBPP layout)
    2. ../humaneval/{stem}.jsonl (HumanEval temperature sweep layout)
    """
    stem = metrics_path.stem  # e.g. pless_t0.7_metrics
    if stem.endswith("_metrics"):
        stem = stem[: -len("_metrics")]
    model_dir = metrics_path.parent.parent
    # Standard layout
    jsonl_path = model_dir / f"{stem}.jsonl"
    if jsonl_path.exists():
        return jsonl_path
    # HumanEval layout: humaneval/ subdirectory
    jsonl_path = model_dir / "humaneval" / f"{stem}.jsonl"
    if jsonl_path.exists():
        return jsonl_path
    return None


def compute_codebleu_for_task(
    task_result: dict, samples: list[str]
) -> dict[str, float | None]:
    """Compute all 5 CodeBLEU diversity fields for a single task."""
    _patch_tree_sitter_capsule()
    from codebleu import calc_codebleu

    pass_results = task_result["pass_results"]
    num_correct = task_result.get("num_correct", sum(pass_results))

    null_result = {
        "self_codebleu": None,
        "self_syntax_match": None,
        "self_dataflow_match": None,
        "self_ngram_match": None,
        "self_weighted_ngram_match": None,
    }

    if num_correct < 2:
        return null_result

    correct_codes = [
        strip_code_fences(sample)
        for sample, passed in zip(samples, pass_results)
        if passed
    ]

    if len(correct_codes) < 2:
        return null_result

    # Deduplicate by AST fingerprint
    seen_fps: dict[str, str] = {}
    unique_codes = []
    for code in correct_codes:
        fp = ast_fingerprint(code)
        key = fp if fp is not None else code
        if key not in seen_fps:
            seen_fps[key] = code
            unique_codes.append(code)

    if len(unique_codes) < 2:
        return {
            "self_codebleu": 0.0,
            "self_syntax_match": 0.0,
            "self_dataflow_match": 0.0,
            "self_ngram_match": 0.0,
            "self_weighted_ngram_match": 0.0,
        }

    bleu_scores, syntax_scores, dataflow_scores = [], [], []
    ngram_scores, weighted_ngram_scores = [], []

    for i in range(len(unique_codes)):
        for j in range(i + 1, len(unique_codes)):
            try:
                res = calc_codebleu(
                    references=[unique_codes[i]],
                    predictions=[unique_codes[j]],
                    lang="python",
                )
                bleu_scores.append(res["codebleu"])
                syntax_scores.append(res["syntax_match_score"])
                dataflow_scores.append(res["dataflow_match_score"])
                ngram_scores.append(res["ngram_match_score"])
                weighted_ngram_scores.append(res["weighted_ngram_match_score"])
            except Exception:
                continue

    if not bleu_scores:
        return null_result

    return {
        "self_codebleu": round(1.0 - _mean(bleu_scores), 4),
        "self_syntax_match": round(1.0 - _mean(syntax_scores), 4),
        "self_dataflow_match": round(1.0 - _mean(dataflow_scores), 4),
        "self_ngram_match": round(1.0 - _mean(ngram_scores), 4),
        "self_weighted_ngram_match": round(1.0 - _mean(weighted_ngram_scores), 4),
    }


def compute_aggregate_diversity(per_task: list[dict]) -> dict[str, float]:
    """Compute top-level diversity aggregates from per-task results."""
    metrics = {}
    for key in (
        "self_codebleu", "self_syntax_match", "self_dataflow_match",
        "self_ngram_match", "self_weighted_ngram_match",
    ):
        values = [
            t[key] for t in per_task
            if t.get("num_correct", 0) >= 2 and t.get(key) is not None
        ]
        agg_key = key.replace("self_", "") + "_diversity"
        metrics[agg_key] = round(_mean(values), 4) if values else 0.0
    return metrics


def patch_single_file(metrics_path: Path, dry_run: bool = False) -> dict:
    """Patch a single metrics JSON with missing CodeBLEU subcomponent fields."""
    result = {"path": str(metrics_path), "status": "skipped", "tasks_patched": 0}

    # Check if already fully patched
    with open(metrics_path) as f:
        metrics = json.load(f)

    if "ngram_match_diversity" in metrics and "weighted_ngram_match_diversity" in metrics:
        result["status"] = "already_patched"
        return result

    # Find corresponding JSONL
    jsonl_path = derive_jsonl_path(metrics_path)
    if jsonl_path is None:
        result["status"] = "no_jsonl"
        return result

    # Load JSONL records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    record_by_id = {r["task_id"]: r for r in records}

    per_task = metrics.get("per_task", [])
    if not per_task:
        result["status"] = "no_per_task"
        return result

    tasks_patched = 0
    for task in per_task:
        task_id = task["task_id"]
        record = record_by_id.get(task_id)
        if record is None:
            continue

        samples = record.get("samples", [])
        codebleu_fields = compute_codebleu_for_task(task, samples)

        for key, val in codebleu_fields.items():
            task[key] = val

        tasks_patched += 1

    # Compute aggregates
    agg = compute_aggregate_diversity(per_task)
    metrics.update(agg)

    if not dry_run:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    result["status"] = "patched"
    result["tasks_patched"] = tasks_patched
    result["aggregates"] = agg
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be patched without writing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only patch files matching this substring")
    args = parser.parse_args()

    files = discover_metrics_files()
    if args.filter:
        files = [f for f in files if args.filter in str(f)]

    print(f"Discovered {len(files)} canonical metrics files")

    if args.workers > 1:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(patch_single_file, f, args.dry_run): f
                for f in files
            }
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                if r["status"] == "patched":
                    print(f"  Patched: {r['path']} ({r['tasks_patched']} tasks)")
                elif r["status"] == "no_jsonl":
                    print(f"  Skipped (no JSONL): {r['path']}")
    else:
        results = []
        for i, f in enumerate(files):
            r = patch_single_file(f, args.dry_run)
            results.append(r)
            if r["status"] == "patched":
                print(f"  [{i+1}/{len(files)}] Patched: {r['path']} ({r['tasks_patched']} tasks)")
            elif r["status"] == "no_jsonl":
                print(f"  [{i+1}/{len(files)}] Skipped (no JSONL): {r['path']}")
            elif r["status"] == "already_patched":
                print(f"  [{i+1}/{len(files)}] Already patched: {r['path']}")

    # Summary
    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    print(f"\nSummary: {by_status}")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
