"""Pooled AtCoder+LeetCode leaderboard for the Ministral-3-3B-Reasoning LCB v6 re-evaluation
(GPU2 batches 1+2).

One row per config, ranked by pooled pass@1 over all tasks of both platforms
(= mean over tasks of per-task c/n). Config name and temperature are taken from the
metrics FILENAME only -- the ``top_p``/``temperature`` fields inside the JSONs are not
reliable for native top-p configs.

Usage:
    uv run python scripts/ministral_lcb_leaderboard.py                       # real paths
    uv run python scripts/ministral_lcb_leaderboard.py --out-dir /tmp/lb_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.eval.headline_table import CANONICAL_COLUMNS, headline_table  # noqa: E402
from bench.eval.metrics import (  # noqa: E402
    compute_cover_at_t,
    compute_self_codebleu_diversity,
    estimate_pass_at_k,
)

R_DEFAULT = (REPO_ROOT / "results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512"
             / "LCB_v6_full1046")
MD_NAME = "ministral_lcb_batch12_leaderboard.md"
CSV_NAME = "ministral_lcb_batch12_leaderboard.csv"
SCRIPT_REL = "scripts/ministral_lcb_leaderboard.py"
RUNNER_REL = "scripts/run_pless_variants_ministral_3b_lcb_cuda.py"

PLATFORMS = ("atcoder", "leetcode")
K_VALUES = (1, 3, 5, 8, 10)
N_BOOT = 2000
SEED = 42
POOL_TOL = 1e-9
DAGGER_CONFIGS = frozenset({"top_p_0_7"})
EXPECTED_TASKS = {"atcoder": 242, "leetcode": 178}  # GPU2 batches 1+2
AUX_FIELDS = (("termination_pct", "term%"), ("mean_token_size", "mean_tok"),
              ("timeout_pct", "timeout%"))
COV_T = (0.3, 0.5, 0.7)
CANONICAL_SECTION_HEADING = "## Canonical tables (pooled, then per platform)"
POOLING_RULE_SENTENCE = (
    "Pooled table: pass@k, cov@t, cov@t distinct are task-weighted over all {N} tasks (each "
    "task judged against its own platform's n); cb_div is the mean per-task self-CodeBLEU "
    "diversity over the pooled tasks with >=2 correct samples (the same task set `bench.eval` "
    "averages over per platform), so it is not a task-count blend of the two platform values; "
    "term%, mean_tok, timeout% are sample-weighted; struct_div is not pooled (skipped).")

_FNAME_RE = re.compile(
    r"^(?P<config>.+)_(?P<platform>atcoder|leetcode)_think_t(?P<T>\d+(?:\.\d+)?)_metrics\.json$"
)

# Order matters: more specific prefixes first.
_FAMILIES = (
    ("ministral_recipe", "Ministral recipe (T=0.7, top-p 0.95)", True),
    ("top_nsigma_1_5", "top-nsigma", True),
    ("gk_", "G_k (Renyi)", False),
    ("pless_", "p-less variants", False),
    ("top_p_", "top-p", False),
)


# ---------------------------------------------------------------- parsing / labels
def parse_metrics_filename(name: str) -> tuple[str, str, float]:
    """``<config>_<platform>_think_t<T>_metrics.json`` -> (config, platform, T)."""
    m = _FNAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"unrecognised metrics filename: {name!r}")
    return m["config"], m["platform"], float(m["T"])


def family_for(config: str) -> str:
    for prefix, family, exact in _FAMILIES:
        if (config == prefix) if exact else config.startswith(prefix):
            return family
    raise ValueError(f"no family mapping for config {config!r}")


def config_label(config: str) -> str:
    return f"{config}†" if config in DAGGER_CONFIGS else config


def t_label(t: float) -> str:
    return f"{t:.1f}" if t == 1.0 else f"{t:.1f}‡"


# ---------------------------------------------------------------- loading / pooling
def discover(metrics_dir: Path) -> dict[str, dict[str, tuple[Path, float]]]:
    """config -> platform -> (path, T). Raises ValueError if a config lacks a platform."""
    found: dict[str, dict[str, tuple[Path, float]]] = {}
    for p in sorted(Path(metrics_dir).glob("*_metrics.json")):
        config, platform, t = parse_metrics_filename(p.name)
        if platform in found.setdefault(config, {}):
            raise ValueError(f"duplicate {platform} metrics file for config {config!r}")
        found[config][platform] = (p, t)
    if not found:
        raise ValueError(f"no *_metrics.json files in {metrics_dir}")
    for config, plats in found.items():
        missing = [pl for pl in PLATFORMS if pl not in plats]
        if missing:
            raise ValueError(f"config {config!r} is missing platform(s) {missing}: "
                             f"have only {sorted(plats)}")
        temps = {plats[pl][1] for pl in PLATFORMS}
        if len(temps) != 1:
            raise ValueError(f"config {config!r} has inconsistent temperatures {temps}")
    return found


def load_platform(path: Path) -> dict:
    """Load one metrics JSON and validate it; raises ValueError on any inconsistency."""
    path = Path(path)
    d = json.loads(path.read_text())
    per_task = d["per_task"]
    if len(per_task) != d["num_tasks"]:
        raise ValueError(f"{path.name}: per_task length {len(per_task)} != num_tasks "
                         f"{d['num_tasks']}")
    ids = [t["task_id"] for t in per_task]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ValueError(f"{path.name}: duplicate task_id(s) {dups[:5]}")
    n = np.array([len(t["pass_results"]) for t in per_task])
    if len(set(n.tolist())) != 1:
        raise ValueError(f"{path.name}: samples per task not uniform: {sorted(set(n.tolist()))}")
    bad = [t["task_id"] for t in per_task if t["num_correct"] != sum(t["pass_results"])]
    if bad:
        raise ValueError(f"{path.name}: num_correct != sum(pass_results) for {bad[:5]}")
    c = np.array([t["num_correct"] for t in per_task])
    for src, _ in AUX_FIELDS:
        v = d.get(src)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"{path.name}: {src} must be a finite number, got {v!r}")
    return {"d": d, "n": n, "c": c, "task_ids": ids}


def _cb_eligible(per_task: list[dict]) -> list[float]:
    """The per-task self_codebleu values ``compute_self_codebleu_diversity`` averages over."""
    return [t["self_codebleu"] for t in per_task
            if t.get("num_correct", 0) >= 2 and t.get("self_codebleu") is not None]


def _check(name: str, what: str, reported: float, recomputed: float) -> None:
    if not abs(reported - recomputed) <= POOL_TOL:
        raise ValueError(f"{name}: reported {what}={reported} != per_task recompute {recomputed}")


def pool_platform_metrics(loaded: dict[str, dict], name: str = "") -> dict:
    """Metrics dict (``headline_table`` input format) pooled over all tasks of PLATFORMS.

    Each platform aggregate is first re-derived from that platform's per_task data and must
    equal the reported value (ValueError otherwise). Pooling rules:
      pass@k             task-weighted: mean over all tasks of the per-task unbiased estimate;
      cov@t, cov@t dist. task-weighted: share of all tasks meeting count >= t*n (own n);
      cb_div             mean per-task self_codebleu over all tasks with >=2 correct samples
                         and a value (bench.eval's own task set; NOT a task-count blend);
      term%/mean_tok/timeout%  sample-weighted (every platform value is a per-sample mean);
      struct_div         not pooled (None -> "—").
    """
    tasks = {pl: len(loaded[pl]["n"]) for pl in PLATFORMS}
    n_tasks = sum(tasks.values())
    samples = {pl: int(loaded[pl]["n"].sum()) for pl in PLATFORMS}
    n_all = np.concatenate([loaded[pl]["n"] for pl in PLATFORMS])
    c_all = np.concatenate([loaded[pl]["c"] for pl in PLATFORMS])
    per_tasks = {pl: loaded[pl]["d"]["per_task"] for pl in PLATFORMS}

    cov_sum = {key: {str(t): 0.0 for t in COV_T} for key in ("cover_at_t", "cover_at_t_distinct")}
    for pl in PLATFORMS:
        d = loaded[pl]["d"]
        recomputed = dict(zip(("cover_at_t", "cover_at_t_distinct"),
                              compute_cover_at_t(per_tasks[pl], list(COV_T),
                                                 int(loaded[pl]["n"][0]))))
        for key, vals in recomputed.items():
            for t, v in vals.items():
                _check(f"{name}/{pl}", f"{key}[{t}]", d[key][t], v)
                cov_sum[key][t] += tasks[pl] * v

    cb_reported = {pl: loaded[pl]["d"].get("codebleu_diversity") for pl in PLATFORMS}
    cb_div = None
    if all(v is not None for v in cb_reported.values()):
        for pl in PLATFORMS:
            _check(f"{name}/{pl}", "codebleu_diversity", cb_reported[pl],
                   compute_self_codebleu_diversity(per_tasks[pl])["codebleu_diversity"])
        elig = [v for pl in PLATFORMS for v in _cb_eligible(per_tasks[pl])]
        cb_div = float(statistics.mean(elig)) if elig else 0.0
        _check(name, "pooled codebleu_diversity (rounded)",
               compute_self_codebleu_diversity(
                   [t for pl in PLATFORMS for t in per_tasks[pl]])["codebleu_diversity"],
               round(cb_div, 4))

    n_uniform = {int(x) for x in n_all}
    out = {
        "num_tasks": n_tasks,
        "num_samples_per_task": n_uniform.pop() if len(n_uniform) == 1 else None,
        "pass_at_k": {str(k): float(estimate_pass_at_k(n_all, c_all, k).mean())
                      for k in K_VALUES},
        **{key: {t: s / n_tasks for t, s in vals.items()} for key, vals in cov_sum.items()},
        "structural_diversity": None,
        "codebleu_diversity": cb_div,
    }
    for src, _ in AUX_FIELDS:
        out[src] = (sum(samples[pl] * loaded[pl]["d"][src] for pl in PLATFORMS)
                    / sum(samples.values()))
    return out


def build_rows(metrics_dir: Path, expected_tasks: dict[str, int] | None = None) -> list[dict]:
    """One dict per config with pooled + per-platform metrics (fractions, full precision).

    ``expected_tasks`` (platform -> task count), if given, is enforced exactly. Within a
    platform every config must have the identical task_id sequence (set AND order): the
    bootstrap resamples task POSITIONS shared across configs.
    """
    rows = []
    ref_ids: dict[str, tuple[str, list[str]]] = {}
    boot_idx = None
    for config, plats in sorted(discover(metrics_dir).items()):
        loaded = {pl: load_platform(plats[pl][0]) for pl in PLATFORMS}
        for pl in PLATFORMS:
            ids, fname = loaded[pl]["task_ids"], plats[pl][0].name
            if expected_tasks is not None and len(ids) != expected_tasks[pl]:
                raise ValueError(f"{fname}: {pl} has {len(ids)} tasks, expected "
                                 f"{expected_tasks[pl]}")
            if pl not in ref_ids:
                ref_ids[pl] = (fname, ids)
            elif ids != ref_ids[pl][1]:
                what = "task_id set" if set(ids) != set(ref_ids[pl][1]) else "task_id order"
                raise ValueError(f"{fname}: {what} differs from {ref_ids[pl][0]}")
        row: dict = {"config": config, "family": family_for(config),
                     "temperature": plats["atcoder"][1]}
        for pl in PLATFORMS:
            row[f"{pl}_file"] = plats[pl][0].name
        n_all = np.concatenate([loaded[pl]["n"] for pl in PLATFORMS])
        c_all = np.concatenate([loaded[pl]["c"] for pl in PLATFORMS])
        tasks = {pl: len(loaded[pl]["n"]) for pl in PLATFORMS}
        row["N"] = int(len(n_all))
        for pl in PLATFORMS:
            row[f"{pl}_N"] = tasks[pl]

        for k in K_VALUES:
            pooled = float(estimate_pass_at_k(n_all, c_all, k).mean())
            row[f"pooled_pass@{k}"] = pooled
            for pl in PLATFORMS:
                rep = loaded[pl]["d"]["pass_at_k"][str(k)]
                recomputed = float(estimate_pass_at_k(loaded[pl]["n"], loaded[pl]["c"], k).mean())
                if not abs(rep - recomputed) <= POOL_TOL:
                    raise ValueError(f"{config}/{pl}: reported pass@{k}={rep} != per_task "
                                     f"recompute {recomputed}")
                row[f"{pl}_pass@{k}"] = rep
            weighted = sum(tasks[pl] * row[f"{pl}_pass@{k}"] for pl in PLATFORMS) / row["N"]
            if not abs(pooled - weighted) <= POOL_TOL:
                raise ValueError(f"{config}: pooled pass@{k}={pooled} != task-weighted mean "
                                 f"{weighted}")

        per_task_p1 = c_all / n_all
        row["_per_task_pass1"] = per_task_p1
        if boot_idx is None:  # one matrix for ALL configs (same N guaranteed by the id check)
            boot_idx = bootstrap_indices(len(per_task_p1), n_boot=N_BOOT, seed=SEED)
        row["_boot_idx"] = boot_idx
        row["ci_lo"], row["ci_hi"] = bootstrap_ci(per_task_p1, idx=boot_idx)
        row["json_top_p"] = {pl: loaded[pl]["d"].get("top_p") for pl in PLATFORMS}

        row["samples"] = sum(int(loaded[pl]["n"].sum()) for pl in PLATFORMS)
        pooled = pool_platform_metrics(loaded, config)
        row["_pooled_metrics"] = pooled
        for src, key in AUX_FIELDS:
            for pl in PLATFORMS:
                row[f"{pl}_{key}"] = loaded[pl]["d"][src]
            row[f"pooled_{key}"] = pooled[src]  # sample-weighted
        rows.append(row)
    return rows


# ---------------------------------------------------------------- stats / ranking
def bootstrap_indices(n_tasks: int, *, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n_tasks) task-position resample matrix; row b = resample b."""
    return np.random.default_rng(seed).integers(0, n_tasks, size=(n_boot, n_tasks))


def bootstrap_ci(vec, *, n_boot: int = N_BOOT, seed: int = SEED,
                 level: float = 0.95, idx: np.ndarray | None = None) -> tuple[float, float]:
    """Percentile bootstrap of the mean, resampling elements (tasks) with replacement.

    Pass ``idx`` (from :func:`bootstrap_indices`) to share resamples across configs.
    """
    vec = np.asarray(vec, dtype=float)
    if idx is None:
        idx = bootstrap_indices(len(vec), n_boot=n_boot, seed=seed)
    if idx.shape[1] != len(vec):
        raise ValueError(f"resample matrix width {idx.shape[1]} != vector length {len(vec)}")
    means = vec[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


SORT_DECIMALS = 12  # absorbs float summation-order noise (~1e-16) so exact ties stay ties


def rank_rows(rows: list[dict]) -> list[dict]:
    ranked = sorted(rows, key=lambda r: (-round(r["pooled_pass@1"], SORT_DECIMALS),
                                         -round(r["pooled_pass@10"], SORT_DECIMALS),
                                         r["config"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def add_paired_deltas(ranked: list[dict]) -> list[dict]:
    """Rank-1 minus each other config: pooled pass@1 delta (pp) + paired 95% bootstrap CI.

    Uses the SAME resample index matrix as the per-config CIs (paired over tasks).
    """
    top = ranked[0]
    top["delta_vs_rank1_pp"] = top["delta_lo_pp"] = top["delta_hi_pp"] = None
    for r in ranked[1:]:
        if r["_boot_idx"] is not top["_boot_idx"] and \
                not np.array_equal(r["_boot_idx"], top["_boot_idx"]):
            raise ValueError(f"{r['config']}: resample matrix differs from rank-1 config")
        d = top["_per_task_pass1"] - r["_per_task_pass1"]
        lo, hi = bootstrap_ci(d, idx=top["_boot_idx"])
        r["delta_vs_rank1_pp"] = 100 * (top["pooled_pass@1"] - r["pooled_pass@1"])
        r["delta_lo_pp"], r["delta_hi_pp"] = 100 * lo, 100 * hi
    return ranked


def delta_excludes_zero(r: dict) -> bool:
    return r["delta_lo_pp"] > 0 or r["delta_hi_pp"] < 0


def render_paired_section(ranked: list[dict]) -> list[str]:
    others = ranked[1:]
    m = len(others)
    n_tasks = ranked[0]["N"]
    lines = [
        "## Paired difference vs rank 1", "",
        f"delta = pooled pass@1 of rank 1 ({ranked[0]['config']}) minus this config, in "
        f"percentage points; paired 95% percentile bootstrap over the {n_tasks} tasks "
        f"({N_BOOT} resamples, seed {SEED}, the same resample index matrix as the per-config "
        "CIs).", "",
        "| Rank | Config | delta pass@1 vs rank 1 (pp) | paired 95% CI (pp) | CI excludes 0? |",
        "|---:|:---|---:|:---:|:---:|",
    ]
    for r in others:
        lines.append(f"| {r['rank']} | {config_label(r['config'])} | "
                     f"{r['delta_vs_rank1_pp']:.2f} | "
                     f"{r['delta_lo_pp']:.2f}–{r['delta_hi_pp']:.2f} | "
                     f"{'yes' if delta_excludes_zero(r) else 'no'} |")
    indist = [f"{r['rank']} ({r['config']})" for r in others if not delta_excludes_zero(r)]
    lines += [
        "",
        f"{m} comparison{'s' if m != 1 else ''}, no multiplicity correction; borderline cases "
        "(a CI bound close to 0) are not solid.",
        "Ranks not distinguishable from rank 1 (paired 95% CI includes 0): "
        f"{', '.join(indist) if indist else 'none'}.",
    ]
    return lines


def _pct2(x: float) -> str:
    return f"{100 * x:.2f}"


def displayed_pass1_ties(ranked: list[dict]) -> list[list[str]]:
    """Groups (in ranked order) of >=2 configs whose displayed pooled pass@1 is identical."""
    groups: list[list[str]] = []
    prev = None
    for r in ranked:
        shown = _pct2(r["pooled_pass@1"])
        if shown == prev:
            groups[-1].append(r["config"])
        else:
            groups.append([r["config"]])
        prev = shown
    return [g for g in groups if len(g) > 1]


# ---------------------------------------------------------------- rendering
LEADERBOARD_HEADER = (
    "| Rank | Config | Family | T | pass@1 (pooled, N={N}) | 95% CI | AtCoder pass@1 | "
    "LeetCode pass@1 | pass@10 (pooled) | term% (pooled, sample-weighted) | "
    "mean_tok (pooled, sample-weighted) | timeout% (pooled, sample-weighted) |"
)


def render_leaderboard(ranked: list[dict]) -> str:
    n_vals = {r["N"] for r in ranked}
    n_label = str(n_vals.pop()) if len(n_vals) == 1 else "varies by config"
    lines = [LEADERBOARD_HEADER.format(N=n_label),
             "|---:|:---|:---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for r in ranked:
        lines.append("| " + " | ".join([
            str(r["rank"]), config_label(r["config"]), r["family"], t_label(r["temperature"]),
            _pct2(r["pooled_pass@1"]), f"{_pct2(r['ci_lo'])}–{_pct2(r['ci_hi'])}",
            _pct2(r["atcoder_pass@1"]), _pct2(r["leetcode_pass@1"]),
            _pct2(r["pooled_pass@10"]), f"{r['pooled_term%']:.1f}",
            f"{r['pooled_mean_tok']:.0f}", f"{r['pooled_timeout%']:.1f}",
        ]) + " |")
    return "\n".join(lines)


def render_pooled_table(ranked: list[dict]) -> str:
    """Canonical table of the pooled metrics, rendered by the repo's own ``headline_table``.

    The pooled dicts are written as JSON into a fresh temporary directory (never the metrics
    dir) and read back; raises ValueError if any rendered pass@k differs from the ranking's
    pooled value by more than POOL_TOL.
    """
    with tempfile.TemporaryDirectory(prefix="pooled_metrics_") as tmp:
        rows = []
        for i, r in enumerate(ranked):
            p = Path(tmp) / f"{i:03d}_{r['config']}_pooled_metrics.json"
            p.write_text(json.dumps(r["_pooled_metrics"]))
            written = json.loads(p.read_text())["pass_at_k"]
            for k in K_VALUES:
                if not abs(written[str(k)] - r[f"pooled_pass@{k}"]) <= POOL_TOL:
                    raise ValueError(f"{r['config']}: pooled-table pass@{k}={written[str(k)]} "
                                     f"!= ranking pooled pass@{k}={r[f'pooled_pass@{k}']}")
            rows.append((p, config_label(r["config"])))
        return headline_table(rows, columns=CANONICAL_COLUMNS)


def render_markdown(ranked: list[dict], metrics_dir: Path) -> str:
    def _uniq(key: str) -> str:
        vals = sorted({r[key] for r in ranked})
        return str(vals[0]) if len(vals) == 1 else "/".join(map(str, vals))

    ties = displayed_pass1_ties(ranked)
    out = [
        "# Ministral-3-3B-Reasoning — LiveCodeBench v6, GPU2 batches 1+2: pooled leaderboard",
        "",
        f"Ranked by pooled pass@1 over all {_uniq('N')} tasks (AtCoder {_uniq('atcoder_N')} + "
        f"LeetCode {_uniq('leetcode_N')}); pooled pass@1 = mean over tasks of per-task c/n. "
        "Ties broken by pooled pass@10 (desc), then config name (asc). All values in %.",
        "",
        render_leaderboard(ranked),
        "",
        "Caveat: single generation seed per task; the per-config 95% CIs describe each config "
        "alone; differences between configs must be judged with the PAIRED task-bootstrap "
        "below.",
        "",
        f"- 95% CI: percentile bootstrap over tasks ({N_BOOT} resamples with replacement, "
        f"numpy `default_rng(seed={SEED})`) of the pooled per-task pass@1 vector; the same seed "
        "is used for every config, so all configs share the same task resamples.",
        "- † top_p_0_7: AtCoder metrics file provenance is UNVERIFIED (could not confirm from "
        "files that it was generated with the post-fix nucleus sampler; user accepted this risk "
        "on 2026-11-08).",
        "- ‡ ministral_recipe uses T=0.7; all other configs use T=1.0.",
        "- Sampler parameters such as the top-p cutoff are NOT stored in the run records "
        f"(the records' `top_p` field reads {_json_top_p_summary(ranked)} for all top-p "
        "configs); the cutoff values in the config names come from the runner configuration "
        f"(`{RUNNER_REL}`); only T is recorded.",
    ]
    if ties:
        for g in ties:
            out.append(f"- Tie on displayed pooled pass@1: {', '.join(g)} (ordered by the "
                       "pass@10 / name tie-break, not by pass@1).")
    else:
        out.append("- No ties on displayed pooled pass@1.")

    if len(ranked) > 1:
        out += [""] + render_paired_section(ranked)

    out += ["", CANONICAL_SECTION_HEADING, "",
            "Generated with `bench.eval.headline_table.headline_table(..., "
            "columns=CANONICAL_COLUMNS)`; rows in the pooled-ranking order. struct_div shows — "
            "because it was intentionally skipped (`--skip-structural-diversity`). "
            + POOLING_RULE_SENTENCE.format(N=_uniq("N")), ""]
    out += [f"### Pooled (AtCoder + LeetCode, N={_uniq('N')})", "",
            render_pooled_table(ranked), ""]
    for pl, title in (("atcoder", "AtCoder"), ("leetcode", "LeetCode")):
        rows = [(metrics_dir / r[f"{pl}_file"],
                 config_label(r["config"]) if pl == "atcoder" else r["config"])
                for r in ranked]
        out += [f"### {title}", "", headline_table(rows, columns=CANONICAL_COLUMNS), ""]
    out.append("(† in the pooled and AtCoder tables marks top_p_0_7, whose AtCoder metrics file "
               "provenance is unverified as noted above; its LeetCode file is not affected and "
               "the LeetCode table carries no †.)")

    out += [
        "", "## Provenance", "",
        "1. Data = GPU2 batches 1+2 only, 242 AtCoder + 178 LeetCode tasks = 420/1046; "
        "batches 0/3/4 not included.",
        "2. 38 metrics files generated by `python -m bench.eval --dataset livecodebench "
        "--workers 4 --skip-structural-diversity` (6.0s per-test timeout), re-evaluated locally "
        "2026-11-07/08.",
        "3. results-validator verdict: PASS with one flag.",
        "4. FLAG: top_p_0_7 AtCoder file provenance is UNVERIFIED (could not confirm from files "
        "that it was generated with the post-fix nucleus sampler; user accepted this risk on "
        "2026-11-08); the top_p_0_7 row is marked †.",
        "5. Excluded: legacy partial top_nsigma (n=1.0) AtCoder file (111/242 tasks), no "
        "LeetCode counterpart.",
        "6. Not included in this table: Mac mlx gk_0_025 run and MSI batch-0 runs.",
        "7. ministral_recipe uses T=0.7, all others T=1.0.",
        "",
        f"Generated by `{SCRIPT_REL}` (script git status at generation: "
        f"{_git_tracked_status(SCRIPT_REL)}; repo HEAD at generation: {_git_commit()}) "
        f"from `{_rel(metrics_dir)}`.",
        "",
    ]
    return "\n".join(out)


CSV_FIELDS = (
    ["rank", "config", "family", "temperature", "N", "atcoder_N", "leetcode_N", "samples"]
    + [f"pooled_pass@{k}" for k in K_VALUES] + ["ci_lo", "ci_hi"]
    + ["pooled_term%", "pooled_mean_tok", "pooled_timeout%"]
    + [f"{pl}_{m}" for pl in PLATFORMS
       for m in [f"pass@{k}" for k in K_VALUES] + ["term%", "mean_tok", "timeout%"]]
    + ["provenance_flag", "atcoder_file", "leetcode_file"]
    + ["delta_vs_rank1_pp", "delta_lo_pp", "delta_hi_pp"]  # appended; earlier cols unchanged
)


def write_csv(ranked: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in ranked:
            rec = {k: r.get(k) for k in CSV_FIELDS}
            rec["provenance_flag"] = ("atcoder_file_provenance_unverified"
                                      if r["config"] in DAGGER_CONFIGS else "")
            for k, v in rec.items():
                if isinstance(v, float):
                    rec[k] = repr(v)
            w.writerow(rec)


def _json_top_p_summary(ranked: list[dict]) -> str:
    vals = sorted({v for r in ranked if r["family"] == "top-p"
                   for v in r.get("json_top_p", {}).values()}, key=str)
    return "/".join(map(str, vals)) if vals else "n/a (no top-p configs)"


def _git_tracked_status(rel_path: str) -> str:
    """Read-only: 'tracked', 'tracked, with uncommitted modifications', 'untracked'."""
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel_path], cwd=REPO_ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return "untracked"
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel_path],
                               cwd=REPO_ROOT, capture_output=True).returncode
        return "tracked, with uncommitted modifications" if dirty else "tracked"
    except Exception:
        return "unknown (git unavailable)"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics-dir", type=Path, default=R_DEFAULT / "metrics")
    ap.add_argument("--out-dir", type=Path, default=R_DEFAULT / "analysis")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing leaderboard .md/.csv")
    ap.add_argument("--expected-atcoder-tasks", type=int, default=EXPECTED_TASKS["atcoder"])
    ap.add_argument("--expected-leetcode-tasks", type=int, default=EXPECTED_TASKS["leetcode"])
    return ap.parse_args(argv)


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    md_path, csv_path = args.out_dir / MD_NAME, args.out_dir / CSV_NAME
    existing = [p for p in (md_path, csv_path) if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"refusing to overwrite existing {', '.join(map(str, existing))} "
                 "(pass --overwrite)")
    expected = {"atcoder": args.expected_atcoder_tasks,
                "leetcode": args.expected_leetcode_tasks}
    ranked = add_paired_deltas(rank_rows(build_rows(args.metrics_dir, expected)))
    md = render_markdown(ranked, args.metrics_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    write_csv(ranked, csv_path)
    print(f"wrote {md_path}\nwrote {csv_path}\nconfigs={len(ranked)} "
          f"pooled==weighted-mean checked for all configs x k={list(K_VALUES)} (tol {POOL_TOL})")
    return ranked


if __name__ == "__main__":
    main()
