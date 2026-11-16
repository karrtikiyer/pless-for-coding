"""Pool gk_0_45 (+context configs) to 420 tasks and run paired task bootstrap.

Run from repo root:  uv run python /tmp/eval_gk045/pool_and_bootstrap.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/karrtikiyer/projects/airesearch/pless-for-coding")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from bench.eval.metrics import estimate_pass_at_k  # noqa: E402
import ministral_lcb_leaderboard as lb  # noqa: E402

M = REPO / "results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/metrics"
CONFIGS = [("gk_0_45", "1.0"), ("gk_0_4", "1.0"), ("top_p_0_9", "1.0"),
           ("top_p_0_95", "1.0"), ("ministral_recipe", "0.7")]

loaded, rows = {}, []
for cfg, T in CONFIGS:
    paths = {pl: M / f"{cfg}_{pl}_think_t{T}_metrics.json" for pl in lb.PLATFORMS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(f"SKIP {cfg}: missing {missing}")
        continue
    L = {pl: lb.load_platform(paths[pl]) for pl in lb.PLATFORMS}
    for pl in lb.PLATFORMS:
        assert len(L[pl]["task_ids"]) == lb.EXPECTED_TASKS[pl], (cfg, pl, len(L[pl]["task_ids"]))
        assert set(L[pl]["n"].tolist()) == {10}, (cfg, pl)
    pooled = lb.pool_platform_metrics(L, cfg)
    loaded[cfg] = L
    rows.append({"config": cfg + " pooled", "_pooled_metrics": pooled,
                 **{f"pooled_pass@{k}": pooled["pass_at_k"][str(k)] for k in lb.K_VALUES}})
    print(f"{cfg}: atcoder N={len(L['atcoder']['task_ids'])} leetcode N={len(L['leetcode']['task_ids'])} "
          f"pooled N={pooled['num_tasks']} n={pooled['num_samples_per_task']}")

print("\n### Pooled canonical table (N=420)\n")
print(lb.render_pooled_table(rows))
print("\n### gk_0_45 per platform\n")
print(lb.headline_table([(M / "gk_0_45_atcoder_think_t1.0_metrics.json", "gk_0_45 atcoder"),
                         (M / "gk_0_45_leetcode_think_t1.0_metrics.json", "gk_0_45 leetcode")],
                        columns=lb.CANONICAL_COLUMNS))


def per_task(cfg, k):
    L = loaded[cfg]
    ids = [f"{pl}::{t}" for pl in lb.PLATFORMS for t in L[pl]["task_ids"]]
    n = np.concatenate([L[pl]["n"] for pl in lb.PLATFORMS])
    c = np.concatenate([L[pl]["c"] for pl in lb.PLATFORMS])
    return dict(zip(ids, estimate_pass_at_k(n, c, k).astype(float)))


print("\n### Paired task bootstrap (tasks = resampling unit; seed 42; percentile 95% CI)\n")
print("| comparison | metric | resamples | observed diff (pp) | 95% CI (pp) | CI excludes 0 | share diff<=0 |")
print("|:--|:--|--:|--:|:--:|:--:|--:|")
for other in ("gk_0_4", "top_p_0_9"):
    if other not in loaded:
        print(f"SKIP gk_0_45 - {other}: not loaded")
        continue
    for k in (1, 10):
        a, b = per_task("gk_0_45", k), per_task(other, k)
        same = set(a) == set(b) and len(a) == len(b) == 420
        order_same = list(a) == list(b)
        if not same:
            raise SystemExit(f"task id sets differ gk_0_45 vs {other}")
        ids = list(a)  # platform-concatenated order, as in the leaderboard (paired by id lookup)
        d = np.array([a[i] - b[i] for i in ids])
        for nb in (2000, 5000):
            idx = lb.bootstrap_indices(len(d), n_boot=nb, seed=lb.SEED)
            lo, hi = lb.bootstrap_ci(d, idx=idx)
            means = d[idx].mean(axis=1)
            share = float((means <= 0).mean())
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            print(f"| gk_0_45 - {other} | pass@{k} | {nb} | {100*d.mean():.2f} | "
                  f"{100*lo:.2f} to {100*hi:.2f} | {excl} | {share:.4f} |")
        if k == 1:
            print(f"<!-- id check gk_0_45 vs {other}: identical 420-id set={same}, "
                  f"same concatenation order={order_same} -->")
