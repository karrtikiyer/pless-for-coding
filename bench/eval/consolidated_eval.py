"""Consolidated metrics evaluation pipeline.

Discovers all result files across MBPP and HumanEval (3 formats), re-executes
every code sample from scratch, computes uniform metrics (including new
diversity metrics), and generates a consolidated report.

Usage:
    python -m bench.eval.consolidated_eval [--workers 4] [--timeout 5] [--force]
    python -m bench.eval.consolidated_eval --verify-only
    python -m bench.eval.consolidated_eval --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from human_eval.data import read_problems

from bench.eval.executor import evaluate_all
from bench.eval.loader import load_results
from bench.eval.metrics import build_metrics_output

RESULTS_ROOT = Path("results")
MBPP_ROOT = RESULTS_ROOT / "pless_mbpp_results"
MBPP_FULL_ROOT = RESULTS_ROOT / "pless_full_mbpp_results"
HE_FULL_ROOT = RESULTS_ROOT / "pless_human_eval_results" / "full_precision_results"
HE_TEMP_ROOT = RESULTS_ROOT / "pless_human_eval_results" / "temprature_results"

# Maps HF-style dir names → short names used in consolidated_metrics/ subdirs.
# Keeps naming consistent across old and new runs.
_MODEL_SHORT_NAMES: dict[str, str] = {
    "codellama--CodeLlama-7b-Instruct-hf": "CodeLlama-7B",
    "mistralai--Codestral-22B-v0.1":        "Codestral-22B",
    "Qwen--Qwen2.5-Coder-7B-Instruct":      "Qwen2.5-Coder-7B",
    "Qwen--Qwen3-Coder-30B-A3B-Instruct":   "Qwen3-Coder-30B",
    "Qwen--Qwen2.5-Coder-3B-Instruct":      "Qwen2.5-Coder-3B-Instruct",
}


def _model_short_name(dir_name: str) -> str:
    return _MODEL_SHORT_NAMES.get(dir_name, dir_name)


K_VALUES = [1, 3, 5, 10]
T_VALUES = [0.1, 0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvalUnit:
    """One configuration to evaluate (model + method + temperature + dataset)."""

    source_path: Path
    model: str
    method: str
    temperature: float
    dataset: str  # "mbpp" or "humaneval"
    format: str  # "mbpp_jsonl", "he_full_json", "he_temp_jsonl"
    output_dir: Path = field(default_factory=Path)

    @property
    def slug(self) -> str:
        return f"{self.method}_t{self.temperature}"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / f"{self.slug}_metrics.json"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_all() -> list[EvalUnit]:
    """Scan all data locations and return a list of EvalUnit instances."""
    units: list[EvalUnit] = []
    units.extend(_discover_mbpp())
    units.extend(_discover_mbpp_full())
    units.extend(_discover_he_full())
    units.extend(_discover_he_full_jsonl())
    units.extend(_discover_he_temp())
    return units


def _discover_mbpp() -> list[EvalUnit]:
    units = []
    if not MBPP_ROOT.exists():
        return units
    for model_dir in sorted(MBPP_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("analysis",):
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            # Parse method + temp from filename like pless_t1.0.jsonl
            stem = jsonl.stem  # e.g. pless_t1.0
            parts = stem.rsplit("_t", 1)
            method = parts[0] if len(parts) == 2 else stem
            try:
                temp = float(parts[1]) if len(parts) == 2 else 0.0
            except ValueError:
                temp = 0.0

            out_dir = (
                MBPP_ROOT / "analysis" / "consolidated_metrics" / model_dir.name
            )
            units.append(EvalUnit(
                source_path=jsonl,
                model=model_dir.name,
                method=method,
                temperature=temp,
                dataset="mbpp",
                format="mbpp_jsonl",
                output_dir=out_dir,
            ))
    return units


def _discover_mbpp_full() -> list[EvalUnit]:
    units = []
    if not MBPP_FULL_ROOT.exists():
        return units
    for model_dir in sorted(MBPP_FULL_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("analysis",):
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            stem = jsonl.stem
            parts = stem.rsplit("_t", 1)
            method = parts[0] if len(parts) == 2 else stem
            try:
                temp = float(parts[1]) if len(parts) == 2 else 0.0
            except ValueError:
                temp = 0.0

            out_dir = (
                MBPP_FULL_ROOT / "analysis" / "consolidated_metrics" / model_dir.name
            )
            units.append(EvalUnit(
                source_path=jsonl,
                model=model_dir.name,
                method=method,
                temperature=temp,
                dataset="mbpp",
                format="mbpp_jsonl",
                output_dir=out_dir,
            ))
    return units


def _discover_he_full() -> list[EvalUnit]:
    units = []
    if not HE_FULL_ROOT.exists():
        return units
    for model_dir in sorted(HE_FULL_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("analysis",):
            continue
        # Use only the most recent detailed JSON per model to avoid duplicates
        detail_jsons = sorted(model_dir.glob("*_detailed.json"))
        if not detail_jsons:
            continue
        for detail_json in detail_jsons[-1:]:
            with open(detail_json) as f:
                data = json.load(f)
            for method_name in data.keys():
                # Infer temperature from method name
                temp = _temp_from_method(method_name)
                out_dir = (
                    HE_FULL_ROOT
                    / "analysis"
                    / "consolidated_metrics"
                    / _model_short_name(model_dir.name)
                )
                units.append(EvalUnit(
                    source_path=detail_json,
                    model=model_dir.name,
                    method=method_name,
                    temperature=temp,
                    dataset="humaneval",
                    format="he_full_json",
                    output_dir=out_dir,
                ))
    return units


def _discover_he_full_jsonl() -> list[EvalUnit]:
    """Discover HumanEval full-precision JSONL files (new method additions not in detailed.json)."""
    units = []
    if not HE_FULL_ROOT.exists():
        return units
    for model_dir in sorted(HE_FULL_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("analysis",):
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            stem = jsonl.stem  # e.g. "pless_t0.6", "top_p0.9_t1.0"
            parts = stem.rsplit("_t", 1)
            method = parts[0] if len(parts) == 2 else stem
            try:
                temp = float(parts[1]) if len(parts) == 2 else 0.0
            except ValueError:
                temp = 0.0
            out_dir = (
                HE_FULL_ROOT / "analysis" / "consolidated_metrics"
                / _model_short_name(model_dir.name)
            )
            units.append(EvalUnit(
                source_path=jsonl,
                model=_model_short_name(model_dir.name),
                method=method,
                temperature=temp,
                dataset="humaneval",
                format="he_full_jsonl",
                output_dir=out_dir,
            ))
    return units


def _discover_he_temp() -> list[EvalUnit]:
    units = []
    if not HE_TEMP_ROOT.exists():
        return units
    for model_dir in sorted(HE_TEMP_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("analysis",):
            continue
        for jsonl in sorted((model_dir / "humaneval").glob("*.jsonl")):
            stem = jsonl.stem
            parts = stem.rsplit("_t", 1)
            method = parts[0] if len(parts) == 2 else stem
            try:
                temp = float(parts[1]) if len(parts) == 2 else 0.0
            except ValueError:
                temp = 0.0

            out_dir = (
                HE_TEMP_ROOT
                / "analysis"
                / "consolidated_metrics"
                / model_dir.name
            )
            units.append(EvalUnit(
                source_path=jsonl,
                model=model_dir.name,
                method=method,
                temperature=temp,
                dataset="humaneval",
                format="he_temp_jsonl",
                output_dir=out_dir,
            ))
    return units


def _load_humaneval_full_jsonl(unit: EvalUnit) -> list[dict]:
    """Load HumanEval full-precision JSONL (new methods). Same format as temp-sweep JSONL."""
    return load_results(unit.source_path)


def _temp_from_method(method: str) -> float:
    """Guess temperature from method name like 'temp_0.7', 'p_less'."""
    if method.startswith("temp_"):
        try:
            return float(method.split("_", 1)[1])
        except ValueError:
            pass
    if method == "greedy":
        return 0.0
    if method in ("top_p_0.95",):
        return 1.0
    if method in ("p_less", "p_less_norm"):
        return 1.0
    if method in ("pless", "pless_norm"):
        return 0.6
    if method in ("top_p",):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Loaders (one per format)
# ---------------------------------------------------------------------------

# Cache HumanEval problems (prompt, test, entry_point) — loaded once
_HE_PROBLEMS: dict | None = None


def _get_he_problems() -> dict:
    global _HE_PROBLEMS
    if _HE_PROBLEMS is None:
        _HE_PROBLEMS = read_problems()
    return _HE_PROBLEMS


def _load_mbpp_jsonl(unit: EvalUnit) -> list[dict]:
    """Load MBPP JSONL — records already have samples + test_list."""
    return load_results(unit.source_path)


def _load_humaneval_full_json(unit: EvalUnit) -> list[dict]:
    """Load HumanEval full-precision detailed JSON for one method.

    Reconstructs complete code (prompt + body) and fetches test/entry_point
    from human_eval.data so we can re-execute from scratch.
    """
    problems = _get_he_problems()

    with open(unit.source_path) as f:
        data = json.load(f)

    records = []
    for task in data[unit.method]:
        task_id = task["task_id"]
        problem = problems[task_id]

        # Code in the JSON is an already-indented function body; prepend
        # the prompt (which ends with the function signature + docstring)
        # to reconstruct a complete, executable function.  Do NOT dedent —
        # the body must stay indented inside the function.
        samples = [problem["prompt"] + s["code"] for s in task["samples"]]

        records.append({
            "task_id": task_id,
            "samples": samples,
            "test": problem["test"],
            "entry_point": problem["entry_point"],
            "model": unit.model,
            "method": unit.method,
            "temperature": unit.temperature,
        })
    return records


def _load_humaneval_temp_jsonl(unit: EvalUnit) -> list[dict]:
    """Load HumanEval temperature-sweep JSONL — already has samples + test + entry_point."""
    return load_results(unit.source_path)


def load_unit(unit: EvalUnit) -> list[dict]:
    """Dispatch to the correct loader based on format."""
    if unit.format == "mbpp_jsonl":
        return _load_mbpp_jsonl(unit)
    elif unit.format == "he_full_json":
        return _load_humaneval_full_json(unit)
    elif unit.format == "he_temp_jsonl":
        return _load_humaneval_temp_jsonl(unit)
    elif unit.format == "he_full_jsonl":
        return _load_humaneval_full_jsonl(unit)
    else:
        raise ValueError(f"Unknown format: {unit.format}")


# ---------------------------------------------------------------------------
# Parse verification
# ---------------------------------------------------------------------------

def verify_parsing(units: list[EvalUnit]) -> None:
    """Load every unit and report parse statistics."""
    print(f"\n{'='*70}")
    print(f"PARSE VERIFICATION: {len(units)} configurations")
    print(f"{'='*70}\n")

    errors = []
    for i, unit in enumerate(units, 1):
        try:
            records = load_unit(unit)
            n_tasks = len(records)
            n_samples = len(records[0]["samples"]) if records else 0
            total = n_tasks * n_samples
            print(
                f"  [{i:3d}/{len(units)}] OK  {unit.model:40s} "
                f"{unit.method:15s} t={unit.temperature:.1f}  "
                f"{n_tasks:3d} tasks x {n_samples:2d} samples = {total:5d}"
            )
        except Exception as e:
            errors.append((unit, str(e)))
            print(
                f"  [{i:3d}/{len(units)}] ERR {unit.model:40s} "
                f"{unit.method:15s} — {e}"
            )

    print(f"\n{'='*70}")
    print(f"Parse results: {len(units) - len(errors)} OK, {len(errors)} errors")
    if errors:
        print("\nErrors:")
        for unit, err in errors:
            print(f"  {unit.source_path}: {err}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def evaluate_unit(
    unit: EvalUnit, workers: int, timeout: float
) -> dict:
    """Execute + compute all metrics for one config."""
    records = load_unit(unit)

    # Re-execute all code samples from scratch
    task_results = evaluate_all(records, unit.dataset, timeout, workers)

    # Infer model name from records if available
    model_name = unit.model
    if records and "model" in records[0]:
        model_name = records[0]["model"]

    return build_metrics_output(
        task_results,
        records,
        model=model_name,
        method=unit.method,
        temperature=unit.temperature,
        dataset=unit.dataset,
        k_values=K_VALUES,
        t_values=T_VALUES,
    )


# ---------------------------------------------------------------------------
# Consolidated report
# ---------------------------------------------------------------------------

def generate_consolidated_report(units: list[EvalUnit]) -> None:
    """Load all per-config metrics JSONs and produce a CSV + markdown report."""
    rows: list[dict] = []

    for unit in units:
        mp = unit.metrics_path
        if not mp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)

        row = {
            "dataset": unit.dataset,
            "format": unit.format,
            "model": m.get("model", unit.model),
            "method": m.get("method", unit.method),
            "temperature": m.get("temperature", unit.temperature),
            "num_tasks": m.get("num_tasks", 0),
            "num_samples_per_task": m.get("num_samples_per_task", 0),
        }
        # pass@k
        for k, v in m.get("pass_at_k", {}).items():
            row[f"pass@{k}"] = round(v, 4)
        # cover@t
        for t, v in m.get("cover_at_t", {}).items():
            row[f"cover@{t}"] = round(v, 2)
        # cover@t distinct
        for t, v in m.get("cover_at_t_distinct", {}).items():
            row[f"cover@{t}_distinct"] = round(v, 2)
        # diversity metrics
        row["structural_diversity"] = m.get("structural_diversity", 0.0)

        rows.append(row)

    if not rows:
        print("No metrics files found — run evaluation first.")
        return

    # Write CSV
    report_dir = RESULTS_ROOT / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "consolidated_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV written to {csv_path} ({len(rows)} rows)")

    # Write markdown report
    md_path = report_dir / "consolidated_report.md"
    with open(md_path, "w") as f:
        _write_markdown_report(f, rows)
    print(f"Markdown report written to {md_path}")


def _write_markdown_report(f, rows: list[dict]) -> None:
    """Generate a structured markdown report from metrics rows."""
    f.write("# Consolidated Metrics Report\n\n")

    # Summary table
    f.write("## Summary\n\n")
    f.write(f"Total configurations evaluated: **{len(rows)}**\n\n")

    # Group by dataset
    for dataset in ("mbpp", "humaneval"):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue

        f.write(f"## {dataset.upper()}\n\n")

        # Core metrics table
        f.write("| Model | Method | Temp | pass@1 | pass@5 | pass@10 "
                "| Diversity | Distinct-3 | LOC σ | CodeBLEU |\n")
        f.write("|-------|--------|------|--------|--------|---------|"
                "-----------|------------|-------|----------|\n")

        for r in sorted(ds_rows, key=lambda x: (x["model"], x["method"], x["temperature"])):
            f.write(
                f"| {r['model']} | {r['method']} | {r['temperature']:.1f} "
                f"| {r.get('pass@1', 0):.4f} "
                f"| {r.get('pass@5', 0):.4f} "
                f"| {r.get('pass@10', 0):.4f} "
                f"| {r.get('structural_diversity', 0):.4f} "
                f"| {r.get('mean_distinct_3', 0):.4f} "
                f"| {r.get('mean_loc_stdev', 0):.2f} "
                f"| {r.get('mean_self_codebleu', 0):.4f} |\n"
            )
        f.write("\n")

    # Research questions
    f.write("## Key Research Questions\n\n")

    # Q1: p-less vs temperature sampling
    f.write("### Does p-less sampling produce more diverse correct solutions?\n\n")
    for dataset in ("mbpp", "humaneval"):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        pless = [r for r in ds_rows if r["method"] == "pless"]
        temp = [r for r in ds_rows if r["method"].startswith("temp")]
        if pless and temp:
            avg_pless_div = sum(r.get("structural_diversity", 0) for r in pless) / len(pless)
            avg_temp_div = sum(r.get("structural_diversity", 0) for r in temp) / len(temp)
            avg_pless_p1 = sum(r.get("pass@1", 0) for r in pless) / len(pless)
            avg_temp_p1 = sum(r.get("pass@1", 0) for r in temp) / len(temp)
            f.write(f"**{dataset.upper()}**: p-less avg diversity = {avg_pless_div:.4f} "
                    f"(pass@1={avg_pless_p1:.4f}) vs temp avg diversity = {avg_temp_div:.4f} "
                    f"(pass@1={avg_temp_p1:.4f})\n\n")

    # Q2: p-less norm effect
    f.write("### Does normalization improve p-less sampling?\n\n")
    for dataset in ("mbpp", "humaneval"):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        pless = [r for r in ds_rows if r["method"] == "pless"]
        pless_norm = [r for r in ds_rows if r["method"] == "pless_norm"]
        if pless and pless_norm:
            for metric in ("pass@1", "structural_diversity", "mean_distinct_3"):
                avg_p = sum(r.get(metric, 0) for r in pless) / len(pless)
                avg_pn = sum(r.get(metric, 0) for r in pless_norm) / len(pless_norm)
                f.write(f"**{dataset.upper()}** {metric}: pless={avg_p:.4f}, "
                        f"pless_norm={avg_pn:.4f}\n\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidated metrics evaluation pipeline"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for code execution (default: 4)",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Per-sample execution timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if metrics JSON already exists",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify parsing — do not execute code",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only generate consolidated report from existing metrics",
    )
    parser.add_argument(
        "--dataset", choices=["humaneval", "mbpp", "all"], default="all",
        help="Only evaluate configs for this dataset (default: all)",
    )
    parser.add_argument(
        "--format", nargs="+", default=None,
        help="Only evaluate configs matching these formats "
             "(e.g. he_full_json he_full_jsonl mbpp_jsonl he_temp_jsonl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Phase 1: Discover all configurations
    print("Phase 1: Discovering configurations...")
    units = discover_all()
    if args.dataset != "all":
        units = [u for u in units if u.dataset == args.dataset]
    if args.format:
        allowed_formats = set(args.format)
        units = [u for u in units if u.format in allowed_formats]
    print(f"  Found {len(units)} configurations")

    by_format = {}
    for u in units:
        by_format.setdefault(u.format, []).append(u)
    for fmt, us in by_format.items():
        print(f"    {fmt}: {len(us)}")

    # Phase 2: Report-only shortcut
    if args.report_only:
        print("\nPhase 6: Generating consolidated report...")
        generate_consolidated_report(units)
        return

    # Phase 3: Verify parsing
    print("\nPhase 2-3: Loading and verifying...")
    verify_parsing(units)

    if args.verify_only:
        return

    # Phase 4-5: Execute + compute metrics
    print("\nPhase 4-5: Executing and computing metrics...")
    done, skipped, errors = 0, 0, 0
    for i, unit in enumerate(units, 1):
        if unit.metrics_path.exists() and not args.force:
            skipped += 1
            continue

        label = f"[{i}/{len(units)}] {unit.model} / {unit.method} t={unit.temperature}"
        print(f"\n{label}")

        try:
            metrics = evaluate_unit(unit, args.workers, args.timeout)

            unit.output_dir.mkdir(parents=True, exist_ok=True)
            with open(unit.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            p1 = metrics["pass_at_k"].get("1", 0)
            sd = metrics.get("structural_diversity", 0)
            d3 = metrics.get("mean_distinct_3", 0)
            print(f"  pass@1={p1:.3f}  diversity={sd:.4f}  distinct_3={d3:.4f}")
            done += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\nPhase 4-5 complete: {done} evaluated, {skipped} skipped, {errors} errors")

    # Phase 6: Consolidated report
    print("\nPhase 6: Generating consolidated report...")
    generate_consolidated_report(units)

    print("\nDone.")


if __name__ == "__main__":
    import platform
    if platform.system() == "Darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork")
    main()
