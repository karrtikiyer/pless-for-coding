"""Pooled APPS table: ATCODER-interview (252) + CODEFORCES-interview (748 subset) = 1000 tasks.

Same columns as the paper's Table 5A (PAPER_COLUMNS), for every config present on BOTH sources.
Pooled pass@k / cov@t are task-weighted means over the union (NOT the average of the two tables);
pooled cb_div is the mean of per-task self_codebleu over the union (tasks with >=2 correct),
NOT the average of the two printed cb_div values. Per-source tables are emitted alongside.
Read-only on the metrics; no re-evaluation. Does not touch paper/ or the paper repo.

Reuse: row/path maps from make_llm4code_tables.APPS_RENYI (252) and make_cf748_table.build_rows (CF);
estimators from bench.eval.metrics.compute_pass_at_k / compute_cover_at_t; formatting from
bench.eval.headline_table.headline_table.

Prereqs: both sources' metrics JSONs carry the per-task self_codebleu cache (CF:
`SET=qwen_cf PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py`).
Usage: uv run python scripts/paper/make_pooled_table.py --model {qwen3,deepseek} [--sort-by pass@1] [--ref LABEL ...] [--boot N] [--seed S] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.eval.headline_table import headline_table  # noqa: E402
from bench.eval.metrics import compute_cover_at_t, compute_pass_at_k  # noqa: E402
import make_cf748_table as cf  # noqa: E402
from make_llm4code_tables import APPS_RENYI, PAPER_COLUMNS  # noqa: E402

EXPECTED = (252, 748)
N_SAMPLES = 10
K_VALUES = [1, 3, 5, 8, 10]
T_VALUES = [0.1, 0.3, 0.5, 0.7]
APPS_KEY = {"qwen3": "Qwen3-8B", "deepseek": "DeepSeek-R1-Distill-Llama-8B"}


def _cf_rows(model: str) -> list[tuple[str, str]]:
    return cf.build_rows(model)


def common_rows(model: str, cf_metrics_dir: Path) -> list[tuple[str, Path, Path]]:
    """(label, atcoder-252 metrics path, cf metrics path) for labels present on both sides, CF order."""
    a_by_label = {label: REPO_ROOT / p for label, p in APPS_RENYI[APPS_KEY[model]]}
    return [
        (label, a_by_label[label], Path(cf_metrics_dir) / f"{stem}_metrics.json")
        for label, stem in _cf_rows(model)
        if label in a_by_label
    ]


def _load(path: Path, expected_n: int) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing metrics file: {path}")
    pt = json.loads(path.read_text()).get("per_task") or []
    if len(pt) != expected_n:
        raise SystemExit(f"{path.name}: {len(pt)} per-task rows, expected {expected_n}")
    if len({t["task_id"] for t in pt}) != len(pt):
        raise SystemExit(f"{path.name}: duplicate task_ids")
    if any(len(t["pass_results"]) != N_SAMPLES for t in pt):
        raise SystemExit(f"{path.name}: expected exactly {N_SAMPLES} samples (n=10) per task")
    if "self_codebleu" not in pt[0]:
        raise SystemExit(
            f"{path.name}: no per-task self_codebleu cache; run build_decoder_comparison_table.py first"
        )
    return pt


def pool_metrics(a_path: Path, c_path: Path, expected: tuple[int, int] = EXPECTED) -> dict:
    a, c = _load(a_path, expected[0]), _load(c_path, expected[1])
    overlap = {t["task_id"] for t in a} & {t["task_id"] for t in c}
    if overlap:
        raise SystemExit(f"task_id overlap between sources ({len(overlap)}): {sorted(overlap)[:5]}")
    per_task = [dict(t, source="ATCODER") for t in a] + [dict(t, source="CODEFORCES") for t in c]
    cov, _ = compute_cover_at_t(per_task, T_VALUES, N_SAMPLES)
    return {
        "num_tasks": len(per_task),
        "pass_at_k": compute_pass_at_k(per_task, K_VALUES),
        "cover_at_t": cov,
        "codebleu_diversity": None,
        "per_task": per_task,
    }


def _stamp(path: Path) -> str:
    return str(json.loads(Path(path).read_text()).get("timestamp"))


def pass_matrix(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """((N, n) bool pass matrix, (N,) int source index 0=ATCODER/1=CODEFORCES), rows sorted by
    (source, task_id) so matrices from different configs are row-aligned."""
    tasks = sorted(m["per_task"], key=lambda t: (t["source"], t["task_id"]))
    x = np.array([t["pass_results"] for t in tasks], dtype=bool)
    s = np.array([0 if t["source"] == "ATCODER" else 1 for t in tasks], dtype=int)
    return x, s


def paired_bootstrap(a: np.ndarray, b: np.ndarray, strata: np.ndarray, boot: int = 5000,
                     seed: int = 0) -> dict:
    """Stratified problem-level paired bootstrap of delta = A - B for pass@1 and pass@10.

    Per-problem statistics are pass@1_i = c_i/n and pass@10_i = any(draws) (n=10 => pass@10 is a
    0/1 indicator). Problems are resampled WITHIN each stratum (source mix preserved), with the same
    indices for A and B (paired). Draws are NOT resampled again: the per-problem pass@1_i already
    carries the draw noise, so a second level double-counts it (CIs ~1.4x too wide) and, for
    any-of-n pass@10, biases the replicate distribution (resampled draws drop observed passes).
    Also returns the analytic stratified paired SE/z for pass@1 (used for the Holm column) and, when
    there are exactly two strata, the interaction delta(stratum0) - delta(stratum1) with its CI.
    """
    rng = np.random.default_rng(seed)
    N = a.shape[0]
    d1 = a.mean(1) - b.mean(1)
    d10 = a.any(1).astype(np.float64) - b.any(1)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    r1 = np.zeros(boot)
    r10 = np.zeros(boot)
    per1 = []
    for idx in groups:
        pick = rng.integers(0, len(idx), (boot, len(idx)))
        m1 = d1[idx][pick].mean(1)
        r1 += (len(idx) / N) * m1
        r10 += (len(idx) / N) * d10[idx][pick].mean(1)
        per1.append(m1)
    var = sum(len(idx) * d1[idx].var(ddof=1) for idx in groups if len(idx) > 1)
    se1 = float(np.sqrt(var) / N)
    out = {
        "dp1": float(d1.mean()),
        "dp1_lo": float(np.percentile(r1, 2.5)), "dp1_hi": float(np.percentile(r1, 97.5)),
        "dp10": float(d10.mean()),
        "dp10_lo": float(np.percentile(r10, 2.5)), "dp10_hi": float(np.percentile(r10, 97.5)),
        "se1": se1,
        "z1": float(d1.mean() / se1) if se1 > 0 else float("inf") if d1.mean() else 0.0,
    }
    out["p1"] = float(math.erfc(abs(out["z1"]) / math.sqrt(2))) if np.isfinite(out["z1"]) else 0.0
    if len(groups) == 2:
        diff = per1[0] - per1[1]
        out["int"] = float(d1[groups[0]].mean() - d1[groups[1]].mean())
        out["int_lo"], out["int_hi"] = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    return out


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _fmt(r: dict, k: str) -> str:
    lo, hi = r[f"{k}_lo"], r[f"{k}_hi"]
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{100 * r[k]:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{star}"


def _paired_section(ref: str, labels: list[str], mats: dict, boot: int, seed: int) -> str:
    xr, sr = mats[ref]
    cmp_labels = [lab for lab in labels if lab != ref]
    res = {lab: paired_bootstrap(mats[lab][0], xr, sr, boot, seed) for lab in cmp_labels}
    holm = dict(zip(cmp_labels, holm_adjust([res[lab]["p1"] for lab in cmp_labels])))
    out = [
        f"\n## Paired differences vs {ref} (config − ref, percentage points; stratified problem-level "
        f"bootstrap 95% CI, {boot} reps, seed {seed}; * = CI excludes 0, unadjusted)\n",
        "| Config | Δpass@1 pooled | Holm (pass@1) | Δpass@10 pooled | Δpass@1 ATCODER | Δpass@1 CODEFORCES | ATCODER−CODEFORCES |",
        "|:---|---:|:---:|---:|---:|---:|---:|",
    ]
    for lab in cmp_labels:
        x = mats[lab][0]
        r = res[lab]
        per = [paired_bootstrap(x[sr == g], xr[sr == g], np.zeros(int((sr == g).sum()), dtype=int), boot, seed)
               for g in (0, 1)]
        out.append(
            f"| {lab} | {_fmt(r, 'dp1')} | {'yes' if holm[lab] < 0.05 else 'no'} (p={holm[lab]:.3g}) | "
            f"{_fmt(r, 'dp10')} | {_fmt(per[0], 'dp1')} | {_fmt(per[1], 'dp1')} | {_fmt(r, 'int')} |"
        )
    out.append(
        "\nCIs: problems resampled within source (same indices for config and ref); draws are not "
        "resampled because per-problem pass@1 / any-of-10 already carry draw noise. `*` is unadjusted; "
        f"Holm = Holm-adjusted two-sided normal-approx p < 0.05 across the {len(cmp_labels)} comparisons "
        "for this ref (analytic stratified paired SE). CIs cover task sampling within the fixed 252 + 748 "
        "sets, not the choice of the CF subset or the 3:1 weighting. Δpass@10 is driven by discordant "
        "problems (0.1 pp grid on 1000 tasks). ATCODER−CODEFORCES is the difference of the per-source "
        "Δpass@1 (heterogeneity check)."
    )
    return "\n".join(out)


SORT_KEYS = {"pass@1": ("pass_at_k", "1")}


def render(rows: list[tuple[str, Path, Path]], expected: tuple[int, int] = EXPECTED,
           workdir: Path | None = None, sort_by: str | None = None,
           refs: list[str] | None = None, boot: int = 5000, seed: int = 0) -> str:
    """sort_by: None keeps the given row order; 'pass@1' orders rows by POOLED pass@1 descending,
    and that same order is used in the per-source sections so rows stay aligned across tables."""
    if sort_by is not None and sort_by not in SORT_KEYS:
        raise SystemExit(f"unknown sort key {sort_by!r}; choose from {sorted(SORT_KEYS)}")
    labels_all = [r[0] for r in rows]
    for ref in refs or []:
        if ref not in labels_all:
            raise SystemExit(f"unknown --ref {ref!r}; available: {labels_all}")
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="pooled_"))
    workdir.mkdir(parents=True, exist_ok=True)
    pooled_metrics = [pool_metrics(a, c, expected) for _, a, c in rows]
    if sort_by is not None:
        field, key = SORT_KEYS[sort_by]
        order = sorted(range(len(rows)), key=lambda i: pooled_metrics[i][field][key], reverse=True)
        rows = [rows[i] for i in order]
        pooled_metrics = [pooled_metrics[i] for i in order]
    pooled = []
    for i, ((label, _, _), m) in enumerate(zip(rows, pooled_metrics)):
        p = workdir / f"pooled_{i:02d}.json"
        p.write_text(json.dumps(m))
        pooled.append((p, label))
    kw = dict(label_header="Decoder", columns=PAPER_COLUMNS)
    n = sum(expected)
    out = [
        f"## Pooled (N={n}: {expected[0]} ATCODER + {expected[1]} CODEFORCES)\n",
        headline_table(pooled, **kw),
        f"\n## ATCODER-interview (N={expected[0]})\n",
        headline_table([(a, label) for label, a, _ in rows], **kw),
        f"\n## CODEFORCES-interview subset (N={expected[1]})\n",
        headline_table([(c, label) for label, _, c in rows], **kw),
        "\n## Inputs (path — stored timestamp)\n",
    ]
    for label, a, c in rows:
        out.append(f"- {label}: `{a}` — {_stamp(a)}; `{c}` — {_stamp(c)}")
    if refs and boot > 0:
        labels = [r[0] for r in rows]
        mats = {lab: pass_matrix(m) for lab, m in zip(labels, pooled_metrics)}
        for ref in refs:
            out.append(_paired_section(ref, labels, mats, boot, seed))
    return "\n".join(out) + "\n"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(APPS_KEY), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sort-by", choices=sorted(SORT_KEYS), default=None,
                    help="order rows by pooled metric, descending (default: paper row order)")
    ap.add_argument("--ref", action="append", default=None,
                    help="reference config label for paired CIs (repeatable); default: top-p @0.9 and p-less baseline (k=2)")
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap replicates (0 disables CIs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    refs = args.ref if args.ref is not None else ["top-p @0.9", "p-less baseline (k=2)"]
    cf_dir = REPO_ROOT / cf.MODELS[args.model]["dir"]
    rows = common_rows(args.model, cf_dir / "metrics")
    ids_file = REPO_ROOT / "data" / "cf_interview_748_ids.txt"
    want = {l.strip() for l in ids_file.read_text().split() if l.strip()}
    got = {str(t["task_id"]) for t in _load(rows[0][2], EXPECTED[1])}
    if got != want:
        raise SystemExit(f"CF task ids differ from {ids_file.name}: {len(got ^ want)} mismatched")
    body = render(rows, sort_by=args.sort_by, refs=refs, boot=args.boot, seed=args.seed)
    title = cf.MODELS[args.model]["title"]
    header = (
        f"# APPS interview, pooled ATCODER-252 + CODEFORCES-748 subset (n=10, thinking-ON) — {title}\n\n"
        f"Common configs only ({len(rows)}). Pooled = task-weighted over the {sum(EXPECTED)}-task union "
        "(a seeded random 748-of-2386 CF subset plus all 252 ATCODER tasks; CF is weighted ~3:1). "
        "Pooled cb_div = mean per-task self-CodeBLEU over the union's tasks with >=2 correct, "
        "so it is not the average of the per-source values.\n\n"
        f"Generated by `uv run python scripts/paper/make_pooled_table.py --model {args.model}{f" --sort-by {args.sort_by}" if args.sort_by else ""}{"".join(f" --ref '{r}'" for r in (args.ref or []))} --boot {args.boot} --seed {args.seed}` at git {_git_sha()}.\n\n"
    )
    out = args.out or REPO_ROOT / "results" / "_pooled_apps_1000" / cf_dir.parents[0].name / f"pooled_table_{args.model}{'_sorted_' + args.sort_by.replace('@', 'at') if args.sort_by else ''}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body)
    print(header + body)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
