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

from bench.eval.executor import extract_python_code

# Method-key fragment for each config so we can locate the JSONL file by
# scanning the bucket directory. These should match what
# ``run_apps_qwen3_top_configs.sh`` produces (see ``_method_key`` in
# bench/apps/runner.py).
CONFIG_FILE_PATTERNS = {
    "H7P":  "split_temp_pure_t1.5_pless_t1.0_think_t*.jsonl",
    "H8P":  "split_temp_pure_t1.5_pless_t1.5_think_t*.jsonl",
    "H9P":  "split_temp_pure_t1.5_pless_t2.0_think_t*.jsonl",
    "T15P": "split_temp_pure_t1.5_temp_pure_t1.5_think_t*.jsonl",
    "T15N": "temp_think_t1.5.jsonl",
    "P15":  "pless_think_t1.5.jsonl",
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


def export_config(
    *,
    results_dir: Path,
    source: str,
    difficulty: str,
    config_key: str,
    output_dir: Path,
) -> dict | None:
    if config_key not in CONFIG_FILE_PATTERNS:
        raise SystemExit(
            f"Unknown config key {config_key!r}; known: {sorted(CONFIG_FILE_PATTERNS)}"
        )
    bucket_dir = results_dir / f"{source}_{difficulty}"
    jsonl_path = _resolve_jsonl(bucket_dir, CONFIG_FILE_PATTERNS[config_key])
    if jsonl_path is None:
        print(f"  [skip] no JSONL for {config_key} in {bucket_dir}")
        return None

    rows = []
    n_samples = 0
    n_dropped_empty = 0
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            task_id = rec["task_id"]
            samples = rec["samples"]
            n_samples += len(samples)
            keep: list[str] = []
            for sample in samples:
                code = extract_python_code(sample)
                if code.strip():
                    keep.append(code)
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
                   help="Comma-separated config keys to export.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for k in keys:
        print(f"[apps_export] exporting {k} ({args.source}/{args.difficulty}) ...")
        entry = export_config(
            results_dir=args.results_dir,
            source=args.source,
            difficulty=args.difficulty,
            config_key=k,
            output_dir=args.output_dir,
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
        "filter": "all_samples_unfiltered",
        "configs": entries,
    }
    manifest_path = args.output_dir.parent / f"manifest_{args.source}_{args.difficulty}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[apps_export] wrote manifest → {manifest_path}")


if __name__ == "__main__":
    main()
