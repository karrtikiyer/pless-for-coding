"""Build algosim-compatible parquet requests from APPS JSONLs.

Sibling of :mod:`bench.eval.algosim_export` (MBPP). Differences from the MBPP
exporter:

  * **Input shape.** APPS JSONLs live under
    ``results/pless_apps_results/<model>/<source>_<difficulty>/<method>.jsonl``
    rather than the MBPP flat directory.
  * **Per-task metadata.** Each JSONL record has ``source`` and
    ``difficulty`` fields (in addition to ``task_id``, ``samples``,
    ``prompt_text``, etc.).
  * **No correctness gating.** We are not executing APPS tests in v1, so
    every sample is fed to algosim — algorithmic-diversity clustering does
    not require functional correctness on the input. (The paper's protocol
    does cluster only Passed samples, which we'll be able to apply once
    APPS execution is wired in.)
  * **problem_id format.** ``"<SOURCE>_<config>_<difficulty>_<task_id>"`` —
    so algosim's hardcoded ATCODER / CODEFORCES prefix filter is happy and
    the response parquet can be bucketed back into (source, difficulty)
    cells for the report.

Usage::

    uv run python -m bench.eval.algosim_export_apps \\
        --results-dir results/pless_apps_results/Qwen--Qwen3-8B \\
        --source ATCODER --difficulty competition \\
        --configs H7P,H8P,H9P,T15P,T15N,P15 \\
        --output-dir algosim_data/apps/requests/atcoder_competition
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bench.eval.executor import extract_python_code  # noqa: F401  (legacy)
from bench.eval.apps_extractor import extract_python_code_apps

# Method-key fragment for each config so we can locate the JSONL file by
# scanning the bucket directory. These should match what
# ``run_apps_qwen3_top_configs.sh`` produces (see ``_method_key`` in
# bench/apps/runner.py).
CONFIG_FILE_PATTERNS = {
    "H7P":  "split_temp_pure_t1.5_pless_t1.0_think_t*.jsonl",
    "H8P":  "split_temp_pure_t1.5_pless_t1.5_think_t*.jsonl",
    "H9P":  "split_temp_pure_t1.5_pless_t2.0_think_t*.jsonl",
    "T15P": "split_temp_pure_t1.5_temp_pure_t1.5_think_t*.jsonl",
    # T15N / P15 are the non-split (thinking-on) configs. The APPS runner's
    # _method_key appends "_think_t{args.temperature}" when --enable-thinking
    # is set, and _output_path further appends "_t{temperature}.jsonl", which
    # produces filenames like "temp_think_t1.5_t1.5.jsonl" (the trailing
    # _t1.5 is redundant but preserved for filename-compatibility with the
    # MBPP runner). Use wildcards on the trailing suffix so we match either form.
    "T15N": "temp_think_t1.5_t*.jsonl",
    "P15":  "pless_think_t1.5_t*.jsonl",
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_jsonl(bucket_dir: Path, pattern: str) -> Path | None:
    matches = sorted(bucket_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [warn] {pattern} matched {len(matches)} files in {bucket_dir}; "
              f"picking the first: {matches[0].name}")
    return matches[0]


def _coerce_task_id(tid):
    """Match algosim_export.py: APPS task_ids are int but keep generic."""
    if isinstance(tid, int):
        return tid
    s = str(tid)
    try:
        return int(s)
    except ValueError:
        return s


def _load_metrics_pass_results(metrics_path: Path) -> dict[int | str, list[bool]] | None:
    """Read ``per_task[].pass_results`` from a metrics JSON.

    Returns dict keyed by task_id → list of bools (one per sample), or
    None if the metrics JSON doesn't exist (e.g. eval hasn't run yet).
    """
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
    except Exception:
        return None
    return {
        _coerce_task_id(t["task_id"]): list(t.get("pass_results", []))
        for t in data.get("per_task", [])
    }


def export_config(
    *,
    results_dir: Path,
    source: str,
    difficulty: str,
    config_key: str,
    output_dir: Path,
    correct_only: bool = True,
) -> dict | None:
    """Export one (source, difficulty, config) bucket to a parquet.

    Two ways to resolve the JSONL file:
      1. If ``config_key`` is a registered split-decoding key (H7P/T15P/etc),
         use its glob pattern in CONFIG_FILE_PATTERNS.
      2. Otherwise (e.g. ``"pless_alpha_a2.0_t1.0"``), treat the key as a
         raw filename basename — look up ``<bucket_dir>/<config_key>.jsonl``.

    If ``correct_only=True`` (default), reads the sibling metrics JSON
    (``<bucket_dir>/metrics/<config_key>_metrics.json``) and keeps only
    samples where ``pass_results[i] == True`` — matches the MBPP/HE
    NAUADC methodology. If no metrics JSON exists, falls back to keeping
    all samples (legacy behavior) and notes this in the return dict.
    """
    bucket_dir = results_dir / f"{source}_{difficulty}"
    if config_key in CONFIG_FILE_PATTERNS:
        jsonl_path = _resolve_jsonl(bucket_dir, CONFIG_FILE_PATTERNS[config_key])
    else:
        # Treat as raw filename basename (α-arm keys land here)
        candidate = bucket_dir / f"{config_key}.jsonl"
        jsonl_path = candidate if candidate.exists() else None

    if jsonl_path is None:
        print(f"  [skip] no JSONL for {config_key} in {bucket_dir}")
        return None

    # Optional correctness filter
    pass_results_by_id: dict | None = None
    if correct_only:
        metrics_path = bucket_dir / "metrics" / f"{config_key}_metrics.json"
        pass_results_by_id = _load_metrics_pass_results(metrics_path)
        if pass_results_by_id is None:
            print(f"  [warn] no metrics JSON for {config_key} at {metrics_path}; "
                  f"falling back to all samples (no correctness filter)")

    rows = []
    n_samples = 0
    n_correct = 0
    n_dropped_empty = 0
    n_dropped_wrong = 0
    for_keep_all = pass_results_by_id is None
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            task_id = _coerce_task_id(rec["task_id"])
            samples = rec["samples"]
            n_samples += len(samples)
            if for_keep_all:
                pass_mask = [True] * len(samples)
            else:
                pass_mask = pass_results_by_id.get(task_id, [False] * len(samples))
                if len(pass_mask) < len(samples):
                    pass_mask = pass_mask + [False] * (len(samples) - len(pass_mask))
            keep: list[str] = []
            for sample, passed in zip(samples, pass_mask):
                if not passed:
                    n_dropped_wrong += 1
                    continue
                n_correct += 1
                # Use the APPS-aware extractor (prefix/window-strip rescue
                # for un-fenced code) so the judge sees the same clean code
                # the executor ran. Switching here from extract_python_code
                # (MBPP-style, fence-or-die) closes the
                # ~33%-of-samples-dropped gap on Deepseek-Coder samples
                # (see bench/eval/apps_extractor.py docstring + Phase A audit).
                result = extract_python_code_apps(sample)
                if result.success and result.code.strip():
                    keep.append(result.code)
                else:
                    n_dropped_empty += 1
            if not keep:
                continue
            rows.append({
                "problem_id": f"{source}_{config_key}_{difficulty}_{task_id}",
                "question": rec["prompt_text"],
                "solutions": keep,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config_key}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return {
        "config": config_key,
        "source": source,
        "difficulty": difficulty,
        "correct_only": not for_keep_all,
        "n_correct_samples": n_correct,
        "n_dropped_wrong": n_dropped_wrong,
        "source_jsonl": str(jsonl_path),
        "source_jsonl_sha256": _file_sha256(jsonl_path),
        "parquet": str(out_path),
        "n_problems": len(rows),
        "n_samples_total": n_samples,
        "n_samples_kept": int(sum(len(r["solutions"]) for r in rows)),
        "n_dropped_empty": n_dropped_empty,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="e.g. results/pless_apps_results/Qwen--Qwen3-8B")
    p.add_argument("--source", required=True, choices=["ATCODER", "CODEFORCES"])
    p.add_argument("--difficulty", required=True,
                   choices=["introductory", "interview", "competition"])
    p.add_argument("--configs", type=str,
                   default="H7P,H8P,H9P,T15P,T15N,P15",
                   help="Comma-separated config keys. Either registered "
                        "split-decoding keys (H7P/H8P/...) or raw JSONL "
                        "basenames like 'pless_alpha_a2.0_t1.0'.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--correct-only", dest="correct_only",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Filter samples to passing-test only via the "
                        "sibling metrics JSON. Default ON to match "
                        "MBPP/HE NAUADC methodology. Use --no-correct-only "
                        "for the legacy 'all samples' behavior.")
    return p.parse_args()


def main():
    args = parse_args()
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for k in keys:
        print(f"[apps_export] exporting {k} ({args.source}/{args.difficulty}) "
              f"correct_only={args.correct_only} ...")
        entry = export_config(
            results_dir=args.results_dir,
            source=args.source,
            difficulty=args.difficulty,
            config_key=k,
            output_dir=args.output_dir,
            correct_only=args.correct_only,
        )
        if entry is None:
            continue
        print(
            f"  → {entry['parquet']} "
            f"({entry['n_problems']} problems, "
            f"{entry['n_samples_kept']}/{entry['n_samples_total']} samples; "
            f"{entry['n_dropped_empty']} dropped as empty after extraction)"
        )
        entries.append(entry)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(args.results_dir),
        "source": args.source,
        "difficulty": args.difficulty,
        "filter": "correct_samples_only" if args.correct_only else "all_samples_unfiltered",
        "configs": entries,
    }
    manifest_path = args.output_dir.parent / f"manifest_{args.source}_{args.difficulty}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[apps_export] wrote manifest → {manifest_path}")


if __name__ == "__main__":
    main()
