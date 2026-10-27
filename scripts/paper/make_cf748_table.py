"""CF748 decoder table in the paper's APPS-252 Table-5A layout (PAPER_COLUMNS).

Read-only on the metrics: it never computes cb_div, it only renders numbers already
in the metrics JSONs via bench.eval.headline_table.headline_table. Fails loudly on a
missing file, a wrong task count, or a missing self-CodeBLEU cache. Does NOT touch
paper/ or the paper repo (unlike make_llm4code_tables.py main()).

Reproduction order (cb_div comes from per-task `self_codebleu` cached into each metrics
JSON; the main eval runs with --skip-diversity so that cache does not exist yet):
  1. uv run python -m bench.eval --results-file <jsonl> --dataset apps --workers 8 \\
         --skip-diversity --k 1,3,5,8,10 --t 0.1,0.3,0.5,0.7        (per arm)
  2. SET=qwen_cf PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py
         (SET=deepseek_cf for DeepSeek; writes the cb_div cache into the metrics JSONs)
  3. uv run python scripts/paper/make_cf748_table.py --model qwen3

Usage: uv run python scripts/paper/make_cf748_table.py --model {qwen3,deepseek} [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.eval.headline_table import headline_table  # noqa: E402
from make_llm4code_tables import PAPER_COLUMNS  # noqa: E402

EXPECTED_TASKS = 748
TOP_P = ("0.7", "0.75", "0.8", "0.85", "0.9")
G_K = ("0.05", "0.1", "0.2", "0.4", "0.8", "1.6")

MODELS = {
    "qwen3": {
        "title": "Qwen3-8B",
        "dir": "results/_cf_interview_748/Qwen--Qwen3-8B/CODEFORCES_interview",
        "rec": "temp_p0.95_k20_think_t0.6_t0.6",
    },
    "deepseek": {
        "title": "DeepSeek-R1-Distill-Llama-8B",
        "dir": "results/_cf_interview_748/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/CODEFORCES_interview",
        "rec": "temp_p0.95_think_t0.6_t0.6",
        # DeepSeek CF also ran the densification arm k=0.35 (in the 252 DeepSeek grid); the table
        # includes it, in ascending position. Qwen keeps the default 6-point grid (G_K).
        "g_k": ("0.05", "0.1", "0.2", "0.35", "0.4", "0.8", "1.6"),
    },
}


def build_rows(model: str) -> list[tuple[str, str]]:
    """(label, metrics-file stem) in the paper table's order. pless_norm is omitted, as in Table 5A."""
    cfg = MODELS[model]
    rows = [("recommended", cfg["rec"])]
    rows += [(f"top-p @{p}", f"temp_p{p}_think_t1.0_t1.0") for p in TOP_P]
    rows += [("p-less baseline (k=2)", "pless_think_t1.0_t1.0")]
    rows += [(f"G_k={k}", f"pless_renyi_think_t1.0_k{k}_t1.0") for k in cfg.get("g_k", G_K)]
    return rows


def _metrics_path(metrics_dir: Path, stem: str) -> Path:
    return Path(metrics_dir) / f"{stem}_metrics.json"


def missing_labels(model: str, metrics_dir: Path) -> list[str]:
    """Labels (table order) whose metrics file does not exist yet."""
    return [label for label, stem in build_rows(model) if not _metrics_path(metrics_dir, stem).exists()]


def _validate(path: Path, expected_tasks: int) -> None:
    if not path.exists():
        raise SystemExit(f"missing metrics file: {path}")
    d = json.loads(path.read_text())
    if d.get("num_tasks") != expected_tasks:
        raise SystemExit(f"{path.name}: num_tasks={d.get('num_tasks')}, expected {expected_tasks}")
    pt = d.get("per_task") or []
    if not pt or "self_codebleu" not in pt[0]:
        raise SystemExit(
            f"{path.name}: no per-task self_codebleu cache (cb_div would read '—'). Run "
            "SET=<qwen_cf|deepseek_cf> PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py first."
        )


def render_table(model: str, metrics_dir: Path, expected_tasks: int = EXPECTED_TASKS,
                 allow_partial: bool = False) -> str:
    """Strict by default (every row must exist). allow_partial=True skips rows whose metrics file does not
    exist yet, but every file that IS present is still fully validated (task count, cb cache)."""
    rows = build_rows(model)
    paths = [(_metrics_path(metrics_dir, stem), label) for label, stem in rows]
    if allow_partial:
        paths = [(p, label) for p, label in paths if p.exists()]
        if not paths:
            raise SystemExit(f"no metrics files found under {metrics_dir} for model {model!r}")
    for p, _ in paths:
        _validate(p, expected_tasks)
    return headline_table(paths, label_header="Decoder", columns=PAPER_COLUMNS)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="render only configs that already have a metrics file and list the rest as pending "
                         "(default: fail if any row is missing)")
    args = ap.parse_args(argv)
    cfg = MODELS[args.model]
    base = REPO_ROOT / cfg["dir"]
    table = render_table(args.model, base / "metrics", allow_partial=args.allow_partial)
    pending = missing_labels(args.model, base / "metrics") if args.allow_partial else []
    n_rows = len(build_rows(args.model))
    partial_note = (
        f"**Partial table: {n_rows - len(pending)} of {n_rows} configs.** Pending (no scored metrics yet): "
        f"{', '.join(pending)}.\n\n" if pending else ""
    )
    gk = ", ".join(cfg.get("g_k", G_K))
    out = args.out or base / "analysis" / f"apps_cf748_paper_table_{args.model}.md"
    header = (
        f"# APPS CODEFORCES-interview ({EXPECTED_TASKS}-task subset, n=10, thinking-ON) — {cfg['title']}\n\n"
        "Same columns as the APPS-252 paper table (Table 5A; pless_norm omitted as there). "
        f"G_k rows: k = {gk} (the CF grid, not the 252 grid). cb_div = mean per-task self-CodeBLEU over "
        "tasks with >=2 correct samples (cached per-task values), so rows average over different task subsets.\n\n"
        f"{partial_note}"
        f"Generated by `uv run python scripts/paper/make_cf748_table.py --model {args.model}"
        f"{' --allow-partial' if args.allow_partial else ''}` at git {_git_sha()}.\n\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + table + "\n")
    print(table)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
