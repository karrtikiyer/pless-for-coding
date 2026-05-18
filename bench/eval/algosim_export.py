"""Build algosim-compatible parquet requests from Qwen3-8B split-decoding JSONLs.

algosim (https://github.com/sh0416/algosim) expects a directory of parquet files,
each row containing:
  - problem_id (str, must start with "ATCODER" or "CODEFORCES" to pass the
    hardcoded filter in clustering_solutions.py / compute_metrics.py)
  - question  (str)
  - solutions (list[str])

We keep only the **functionally correct** samples (pass_results[i] is True),
run them through the same extract_python_code() as our pass@k pipeline, and
emit one parquet per config. Tasks with zero correct samples are dropped.

Usage:
    uv run python -m bench.eval.algosim_export \
        --configs A,C,T15,P15,H7P,H8P,H9P,H10,H11P,H12P \
        --output-dir algosim_data/requests
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bench.eval.executor import extract_python_code
from bench.eval.split_decoding_analysis import CONFIGS

# Default for the Qwen3-8B 14-config layout (backwards compatibility).
DEFAULT_RESULTS_ROOT = Path("results/pless_full_mbpp_results")
DEFAULT_MODEL_SLUG = "Qwen--Qwen3-8B"


def _load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["task_id"])] = rec
    return out


def _load_metrics(path: Path) -> dict[int, list[bool]]:
    data = json.loads(path.read_text())
    return {int(t["task_id"]): list(t["pass_results"]) for t in data["per_task"]}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_config_paths(
    config_key: str, model_dir: Path
) -> tuple[Path, Path, str]:
    """Resolve (jsonl_path, metrics_path, label) for a config.

    Two modes:
      1. ``config_key`` is a key in the legacy ``CONFIGS`` dict — use
         the registered filename + label (Qwen3-8B 14-config layout).
      2. Otherwise, treat ``config_key`` as the raw JSONL basename
         (without ``.jsonl``) and look it up directly in ``model_dir``.
         Used for the new α-sweep exports under
         ``results/pless_alpha_full/{model_slug}/``.
    """
    if config_key in CONFIGS:
        cfg = CONFIGS[config_key]
        jsonl_filename = cfg["file"]
        label = cfg["label"]
    else:
        jsonl_filename = f"{config_key}.jsonl"
        label = config_key
    jsonl_path = model_dir / jsonl_filename
    metrics_path = model_dir / "metrics" / jsonl_filename.replace(".jsonl", "_metrics.json")
    return jsonl_path, metrics_path, label


def export_config(
    config_key: str,
    output_dir: Path,
    model_dir: Path = None,
) -> dict:
    """Export one config to parquet. Returns a manifest entry."""
    if model_dir is None:
        model_dir = DEFAULT_RESULTS_ROOT / DEFAULT_MODEL_SLUG
    jsonl_path, metrics_path, label = _resolve_config_paths(config_key, model_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing JSONL: {jsonl_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    records = _load_jsonl(jsonl_path)
    pass_map = _load_metrics(metrics_path)

    rows = []
    total_samples = 0
    correct_samples = 0
    for task_id, rec in records.items():
        passes = pass_map.get(task_id)
        if passes is None:
            continue
        samples = rec["samples"]
        total_samples += len(samples)
        keep: list[str] = []
        for i, sample in enumerate(samples):
            if i >= len(passes) or not passes[i]:
                continue
            code = extract_python_code(sample)
            if code.strip():
                keep.append(code)
        correct_samples += len(keep)
        if not keep:
            continue
        rows.append({
            "problem_id": f"ATCODER_{config_key}_{task_id:04d}",
            "question": rec["prompt_text"],
            "solutions": keep,
        })

    df = pd.DataFrame(rows)
    out_path = output_dir / f"{config_key}.parquet"
    df.to_parquet(out_path, index=False)

    return {
        "config": config_key,
        "label": label,
        "source_jsonl": str(jsonl_path),
        "source_jsonl_sha256": _file_sha256(jsonl_path),
        "parquet": str(out_path),
        "n_tasks_with_correct": len(rows),
        "n_correct_samples": correct_samples,
        "n_total_samples": total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=str,
        default="A,B,C,D,E,T15N,P15,T15P,H7P,H8P,H9P,H10P,H11P,H12P",
        help="Comma-separated config keys to export. Either keys in the "
             "legacy CONFIGS dict (A/B/C/.../H12P for the Qwen3-8B layout) or "
             "raw JSONL basenames (without .jsonl) for arbitrary models — "
             "e.g. 'pless_alpha_a2.0_t1.0'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("algosim_data/requests"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Parent dir containing per-model subdirs (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--model-slug",
        type=str,
        default=DEFAULT_MODEL_SLUG,
        help=f"Per-model subdir under --results-dir; HF id with '/' → '--' "
             f"(default: {DEFAULT_MODEL_SLUG}).",
    )
    args = parser.parse_args()

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]

    # Resolve the per-model dir and validate that each config has a JSONL there.
    model_dir = args.results_dir / args.model_slug
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")
    missing = []
    for key in keys:
        jsonl_path, metrics_path, _ = _resolve_config_paths(key, model_dir)
        if not jsonl_path.exists() or not metrics_path.exists():
            missing.append((key, jsonl_path, metrics_path))
    if missing:
        msgs = "\n  ".join(f"{k}: jsonl={p}, metrics={m}" for k, p, m in missing)
        raise SystemExit(f"Missing files for configs:\n  {msgs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for key in keys:
        _, _, label = _resolve_config_paths(key, model_dir)
        print(f"[algosim_export] exporting {key} ({label}) ...")
        entry = export_config(key, args.output_dir, model_dir=model_dir)
        print(
            f"  → {entry['parquet']} "
            f"({entry['n_tasks_with_correct']} tasks, "
            f"{entry['n_correct_samples']}/{entry['n_total_samples']} correct samples)"
        )
        manifest_entries.append(entry)

    # Derive HF model id from slug ('--' → '/'). Falls back to slug if no '--'.
    model_id = args.model_slug.replace("--", "/")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "model_slug": args.model_slug,
        "results_dir": str(args.results_dir),
        "dataset": "mbpp",
        "filter": "correct_samples_only",
        "configs": manifest_entries,
    }
    # Write manifest inside the output dir so multi-model runs don't collide.
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[algosim_export] wrote manifest → {manifest_path}")


if __name__ == "__main__":
    main()
