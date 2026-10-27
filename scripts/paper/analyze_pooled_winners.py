"""Winners analysis over the pooled APPS table (ATCODER-252 + CODEFORCES-748 subset), common configs.

PRE-SPECIFIED (fixed before looking at any ranking; advisor-reviewed):
  * Primary metrics: pass@1, pass@10, cov@0.7  (mean accuracy / reach / reliability). Everything else
    (pass@3/5/7, cov@0.3/0.5, cov-curve, cb_div, cost) is exploratory/appendix. pass@k (n=10) and cov@t are
    all linear functionals of per-task num_correct, so "wins on many metrics" is NOT independent evidence.
  * Reference for fixed-reference deltas: 'top-p @0.9'. Holm within each primary across the other configs.
  * "Top group" = configs whose deficit to the best cannot be established by a joint (same resampled problems
    for all configs, stratified by source, no draw resampling) simultaneous one-sided 95% max-t interval for
    theta_j = mu_j - max_{i!=j} mu_i. A single config is called "the winner" only if the group has size 1.
  * P(rank 1) and rank intervals are DESCRIPTIVE only. cb_div is a side metric (correct samples only; the
    own-population mean mixes task subsets) and never enters a winner verdict.
Read-only; no re-evaluation. pass@k recomputed from num_correct (unbiased estimator, n=10).

Usage: uv run python scripts/paper/analyze_pooled_winners.py --model qwen3 [--boot 5000] [--seed 0] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.eval.metrics import estimate_pass_at_k  # noqa: E402
import make_pooled_table as mp  # noqa: E402

N_SAMPLES = 10
PRIMARIES = ["pass@1", "pass@10", "cov@0.7"]
METRICS = ["pass@1", "pass@3", "pass@5", "pass@7", "pass@10", "cov@0.3", "cov@0.5", "cov@0.7"]
REF = "top-p @0.9"
MARGIN_PP = 1.0
SRC_NAMES = ["ATCODER", "CODEFORCES"]


def lookup(metric: str) -> np.ndarray:
    """Per-task metric value as a function of c = num_correct (0..10)."""
    c = np.arange(N_SAMPLES + 1)
    kind, val = metric.split("@")
    if kind == "pass":
        return np.asarray(estimate_pass_at_k(np.full(N_SAMPLES + 1, N_SAMPLES), c, int(val)), dtype=float)
    return (c >= round(float(val) * N_SAMPLES)).astype(float)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank 1 = highest value; ties get the average rank."""
    return rankdata(-np.asarray(values), method="average")


def joint_bootstrap_means(V: np.ndarray, strata: np.ndarray, boot: int, seed: int, chunk: int = 250) -> np.ndarray:
    """(boot, J) replicate means of per-task values V (J, N). The SAME stratified problem indices are used
    for every config, so cross-config dependence (shared problems) is preserved. Draws are not resampled."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    out = []
    for start in range(0, boot, chunk):
        m = min(chunk, boot - start)
        pidx = np.concatenate([idx[rng.integers(0, len(idx), (m, len(idx)))] for idx in groups], axis=1)
        out.append(V[:, pidx].mean(2).T)
    return np.concatenate(out, axis=0)


def _max_other(X: np.ndarray) -> np.ndarray:
    srt = np.sort(X, axis=-1)
    top, second = srt[..., -1:], srt[..., -2:-1]
    return np.where(X == top, second, top)


def top_group(obs: np.ndarray, R: np.ndarray, alpha: float = 0.05) -> tuple[list[int], float]:
    """Simultaneous one-sided max-t 'not separable from best' set (Hsu-style, joint bootstrap).
    theta_j = mu_j - max_{i!=j} mu_i; j stays in the group if its upper bound >= 0."""
    theta = obs - _max_other(obs)
    theta_star = R - _max_other(R)
    Q = float(np.percentile((theta_star - theta).min(axis=1), 100 * alpha))
    upper = theta - Q
    return [int(j) for j in np.flatnonzero(upper >= 0)], Q


def _strat_mean_boot(d: np.ndarray, strata: np.ndarray, boot: int, rng) -> np.ndarray:
    N = len(d)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    r = np.zeros(boot)
    for idx in groups:
        r += (len(idx) / N) * d[idx][rng.integers(0, len(idx), (boot, len(idx)))].mean(1)
    return r


def delta_ci(v: np.ndarray, vref: np.ndarray, strata: np.ndarray, boot: int, seed: int) -> dict:
    """Paired stratified problem-level bootstrap CI + analytic stratified z/p for mean(v - vref)."""
    d = np.asarray(v, float) - np.asarray(vref, float)
    N = len(d)
    r = _strat_mean_boot(d, strata, boot, np.random.default_rng(seed))
    var = sum(len(idx) * d[idx].var(ddof=1) for idx in (np.flatnonzero(strata == g) for g in np.unique(strata))
              if len(idx) > 1)
    se = float(np.sqrt(var) / N)
    mean = float(d.mean())
    z = mean / se if se > 0 else (0.0 if mean == 0 else float("inf"))
    p = float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else 0.0
    return {"d": mean, "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5)),
            "se": se, "z": z, "p": p}


def cb_delta(cb_a: np.ndarray, cb_b: np.ndarray, strata: np.ndarray, boot: int, seed: int) -> dict:
    """Paired cb_div delta on the pairwise-intersection subset (tasks valid, i.e. >=2 correct, in both)."""
    mask = ~np.isnan(cb_a) & ~np.isnan(cb_b)
    r = delta_ci(cb_a[mask], cb_b[mask], strata[mask], boot, seed)
    r["n"] = int(mask.sum())
    return r


def informative_share(C: np.ndarray) -> float:
    """Fraction of tasks whose num_correct is not identical across all configs. C: (J, N)."""
    return float((C.max(0) != C.min(0)).mean())


def aligned_vectors(m: dict):
    """(num_correct, self_codebleu [nan if None], num_distinct_correct [nan if absent], source idx),
    rows sorted by (source, task_id) exactly like make_pooled_table.pass_matrix."""
    tasks = sorted(m["per_task"], key=lambda t: (t["source"], t["task_id"]))
    c = np.array([t["num_correct"] for t in tasks], dtype=int)
    cb = np.array([np.nan if t.get("self_codebleu") is None else t["self_codebleu"] for t in tasks], dtype=float)
    nd = np.array([np.nan if t.get("num_distinct_correct") is None else t["num_distinct_correct"] for t in tasks],
                  dtype=float)
    s = np.array([0 if t["source"] == "ATCODER" else 1 for t in tasks], dtype=int)
    return c, cb, nd, s


# ----------------------------------------------------------------------------- report


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|:---|" + "---:|" * (len(header) - 1)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _fmt_d(r: dict, scale: float = 100.0) -> str:
    star = "*" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return f"{scale * r['d']:+.2f} [{scale * r['lo']:+.2f}, {scale * r['hi']:+.2f}]{star}"


def analyze(rows, expected=mp.EXPECTED, boot: int = 5000, seed: int = 0, cf_metrics_paths: list | None = None) -> str:
    labels = [r[0] for r in rows]
    if REF not in labels:
        raise SystemExit(f"reference {REF!r} not among configs {labels}")
    pooled = [mp.pool_metrics(a, c, expected) for _, a, c in rows]
    vecs = [aligned_vectors(m) for m in pooled]
    keys0 = [(t["source"], t["task_id"]) for t in sorted(pooled[0]["per_task"], key=lambda t: (t["source"], t["task_id"]))]
    for lab, m in zip(labels, pooled):
        if [(t["source"], t["task_id"]) for t in sorted(m["per_task"], key=lambda t: (t["source"], t["task_id"]))] != keys0:
            raise SystemExit(f"{lab}: task set/alignment differs from {labels[0]}")
    C = np.vstack([v[0] for v in vecs])
    CB = np.vstack([v[1] for v in vecs])
    ND = np.vstack([v[2] for v in vecs])
    S = vecs[0][3]
    J, N = C.shape
    ref_i = labels.index(REF)
    V = {m: lookup(m)[C] for m in set(METRICS) | {f"cov@{t / 10:.1f}" for t in range(1, 11)}}
    masks = {"pooled": np.ones(N, bool), "ATCODER": S == 0, "CODEFORCES": S == 1}
    L: list[str] = []

    # 0. pre-specification + provenance
    L.append("## Pre-specification (locked in code; advisor-reviewed before running)\n")
    L.append(f"- Primary metrics: {', '.join(PRIMARIES)}. Everything else is exploratory. Reference: `{REF}`. "
             f"Practical-significance margin considered: {MARGIN_PP:g} pp. Bootstrap: {boot} reps, seed {seed}, "
             "stratified by source, problems resampled jointly across configs, draws not resampled.")
    L.append(f"- N={N} tasks ({int((S == 0).sum())} ATCODER + {int((S == 1).sum())} CODEFORCES), n={N_SAMPLES} samples/task, "
             f"{J} configs. Pooled = task-weighted (CF ~3:1).\n")

    # 1. metric tables
    order = np.argsort(-V["pass@1"].mean(1))
    for scope in ("pooled", "ATCODER", "CODEFORCES"):
        mk = masks[scope]
        L.append(f"\n## Metrics — {scope} (N={int(mk.sum())}; percent; rows sorted by pooled pass@1)\n")
        L.append(_table(["Config"] + METRICS, [[labels[j]] + [_pct(V[m][j][mk].mean()) for m in METRICS] for j in order]))
    L.append("\n## cov@t curve (pooled; t = c/10, c=1..10; cov@0.1 == pass@10 for n=10)\n")
    cov_keys = [f"cov@{t / 10:.1f}" for t in range(1, 11)]
    L.append(_table(["Config"] + cov_keys, [[labels[j]] + [_pct(V[k][j].mean()) for k in cov_keys] for j in order]))

    # 2. rank / top group per primary, per scope
    L.append("\n## Ranks and top group (primary metrics)\n")
    L.append("`top group` = configs not separable from the best by the simultaneous max-t set (✓). Rank interval = "
             "bootstrap 2.5–97.5 pct of the config's rank; P(rank 1) is descriptive, NOT the probability of being truly best.\n")
    tg: dict = {}
    seed_check: dict = {}
    for prim in PRIMARIES:
        for scope in ("pooled", "ATCODER", "CODEFORCES"):
            mk = masks[scope]
            Vp = V[prim][:, mk]
            R = joint_bootstrap_means(Vp, S[mk], boot, seed)
            obs = Vp.mean(1)
            grp, Q = top_group(obs, R)
            tg[(prim, scope)] = (set(grp), Q, obs, R)
            if scope == "pooled":
                R2 = joint_bootstrap_means(Vp, S[mk], boot, seed + 1)
                g2, Q2 = top_group(obs, R2)
                seed_check[prim] = (sorted(labels[j] for j in grp), Q, sorted(labels[j] for j in g2), Q2)
    for prim in PRIMARIES:
        grp, Q, obs, R = tg[(prim, "pooled")]
        ranks = rankdata(-R, axis=1, method="average")
        rk = average_ranks(obs)
        tab = []
        for j in np.argsort(-obs):
            lo, hi = np.percentile(ranks[:, j], [2.5, 97.5])
            marks = " ".join("✓" if j in tg[(prim, sc)][0] else "·" for sc in ("pooled", "ATCODER", "CODEFORCES"))
            tab.append([labels[j], _pct(obs[j]), f"{rk[j]:.1f}", f"[{lo:.1f}, {hi:.1f}]",
                        f"{float((ranks[:, j] == 1).mean()):.2f}", marks])
        L.append(f"\n### {prim}  (pooled critical deficit |Q| = {100 * abs(Q):.2f} pp)\n")
        L.append(_table(["Config", "pooled %", "avg rank", "rank interval", "P(rank 1)", "top group: pooled ATC CF"], tab))
        sizes = {sc: len(tg[(prim, sc)][0]) for sc in ("pooled", "ATCODER", "CODEFORCES")}
        L.append(f"\nTop-group size — pooled {sizes['pooled']}, ATCODER {sizes['ATCODER']}, CODEFORCES {sizes['CODEFORCES']} "
                 f"(of {J}); a single 'winner' requires size 1.")

    # 3. fixed-reference deltas
    L.append(f"\n## Paired differences vs {REF} (config − ref, percentage points; stratified problem-level bootstrap CI; "
             "* = CI excludes 0, unadjusted)\n")
    L.append("Holm = Holm-adjusted two-sided normal-approx p < 0.05 within the metric across the other configs. "
             "`sign-consistent` = ATCODER and CODEFORCES deltas have the same sign.\n")
    others = [j for j in range(J) if j != ref_i]
    for prim in PRIMARIES:
        res = {j: delta_ci(V[prim][j], V[prim][ref_i], S, boot, seed) for j in others}
        holm = dict(zip(others, mp.holm_adjust([res[j]["p"] for j in others])))
        tab = []
        for j in sorted(others, key=lambda j: -res[j]["d"]):
            per = [delta_ci(V[prim][j][S == g], V[prim][ref_i][S == g], np.zeros(int((S == g).sum()), int), boot, seed)
                   for g in (0, 1)]
            cons = "yes" if per[0]["d"] * per[1]["d"] > 0 else "no"
            tab.append([labels[j], _fmt_d(res[j]), f"{'yes' if holm[j] < 0.05 else 'no'} (p={holm[j]:.3g})",
                        _fmt_d(per[0]), _fmt_d(per[1]), cons])
        L.append(f"\n### {prim}\n")
        L.append(_table(["Config", "Δ pooled", "Holm", "Δ ATCODER", "Δ CODEFORCES", "sign-consistent"], tab))
    se1 = [delta_ci(V["pass@1"][j], V["pass@1"][ref_i], S, boot, seed)["se"] for j in others]
    L.append(f"\nResolution: pooled pass@1 delta SE ranges {100 * min(se1):.2f}–{100 * max(se1):.2f} pp, so differences "
             f"smaller than ~{2 * 100 * max(se1):.1f} pp cannot be resolved within the top group; non-significance is not equivalence.")

    # 4. cb_div (side metric)
    valid = ~np.isnan(CB)
    common = valid.all(0)
    L.append("\n## cb_div (side metric; correct samples only; NOT used for winner verdicts)\n")
    L.append(f"Own-population = mean over each config's tasks with >=2 correct. 13-way common subset = tasks valid for all "
             f"{J} configs (n={int(common.sum())}: ATCODER {int((common & (S == 0)).sum())}, CODEFORCES "
             f"{int((common & (S == 1)).sum())}; easier tasks only). cb=0 share = tasks whose correct samples are all "
             "AST-identical, among tasks valid for that config.\n")
    tab = []
    for j in np.argsort(-np.nanmean(CB, axis=1)):
        d = cb_delta(CB[j], CB[ref_i], S, boot, seed) if j != ref_i else None
        tab.append([labels[j], f"{np.nanmean(CB[j]):.4f}", f"{np.nanmean(CB[j][common]):.4f}",
                    f"{100 * float((CB[j][valid[j]] == 0).mean()):.1f}%", f"{np.nanmean(ND[j][valid[j]]):.2f}",
                    "—" if d is None else f"{d['d']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{'*' if (d['lo'] > 0 or d['hi'] < 0) else ''} (n={d['n']})"])
    L.append(_table(["Config", "own-pop", "common-subset", "cb=0 share", "mean #distinct correct",
                     f"paired Δ vs {REF} (pairwise-intersection)"], tab))

    # 5. informative tasks
    L.append("\n## Information content\n")
    allzero = float((C.max(0) == 0).mean())
    allten = float((C.min(0) == N_SAMPLES).mean())
    L.append(f"- Tasks informative for ranking (num_correct not identical across all configs): pooled "
             f"{100 * informative_share(C):.1f}%, ATCODER {100 * informative_share(C[:, S == 0]):.1f}%, "
             f"CODEFORCES {100 * informative_share(C[:, S == 1]):.1f}%.")
    L.append(f"- Unsolved by every config (c=0 everywhere): {100 * allzero:.1f}%; solved 10/10 by every config: {100 * allten:.1f}%.")

    # 6. CF-only cost
    if cf_metrics_paths:
        L.append("\n## Cost context — CODEFORCES only (ATCODER-252 metrics have no termination/token fields)\n")
        tab = []
        for lab, p in zip(labels, cf_metrics_paths):
            d = json.loads(Path(p).read_text())
            f = lambda k, fmt: "—" if d.get(k) is None else fmt.format(d[k])
            tab.append([lab, f("termination_pct", "{:.1f}"), f("mean_token_size", "{:.0f}"), f("timeout_pct", "{:.1f}")])
        L.append(_table(["Config", "term%", "mean_tok", "timeout%"], tab))

    # 7. seed stability
    L.append("\n## Seed stability (pooled top group, seed vs seed+1)\n")
    tab = []
    for prim in PRIMARIES:
        g1, q1, g2, q2 = seed_check[prim]
        tab.append([prim, ", ".join(g1), f"{100 * abs(q1):.2f}", "yes" if g1 == g2 else "NO", f"{100 * abs(q2):.2f}"])
    L.append(_table(["Metric", f"top group (seed {seed})", "|Q| pp", "same set at seed+1", "|Q| pp (seed+1)"], tab))

    L.append("\n## Caveats (do not over-read)\n")
    L.append("- Winner = a top group unless size 1; P(rank 1)/rank intervals are descriptive. Metrics are correlated "
             "functionals of num_correct; do not count 'metrics won'.\n"
             "- CIs are over tasks only (draws taken as given); they do not cover batch/infra differences between run "
             "directories (recommended/p-less baselines vs sweep dirs; recommended is T0.6/p0.95/k20, others T1.0), "
             "the choice of the CF subset, or the 3:1 weighting.\n"
             "- Qwen3-8B, thinking-ON, interview-level tasks only. No cost claim pooled (ATCODER lacks token fields).")
    return "\n".join(L) + "\n"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(mp.APPS_KEY), required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    cf_dir = REPO_ROOT / mp.cf.MODELS[args.model]["dir"]
    rows = mp.common_rows(args.model, cf_dir / "metrics")
    body = analyze(rows, boot=args.boot, seed=args.seed, cf_metrics_paths=[r[2] for r in rows])
    header = (f"# Winners analysis — pooled APPS ATCODER-252 + CODEFORCES-748 subset — {mp.cf.MODELS[args.model]['title']}\n\n"
              f"Generated by `uv run python scripts/paper/analyze_pooled_winners.py --model {args.model} --boot {args.boot} "
              f"--seed {args.seed}` at git {_git_sha()}.\n\n")
    out = args.out or REPO_ROOT / "results" / "_pooled_apps_1000" / cf_dir.parents[0].name / f"winners_analysis_{args.model}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body)
    print(header + body)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
