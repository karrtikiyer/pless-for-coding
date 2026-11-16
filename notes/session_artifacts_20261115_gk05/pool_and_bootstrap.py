"""Pool gk_0_5 (+context configs) to 420 tasks and run paired task bootstrap.

Adapted from notes/session_artifacts_20261114/eval_gk035/pool_and_bootstrap.py.
Run from repo root:  uv run python /tmp/eval_gk05/pool_and_bootstrap.py
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
CONFIGS = [("gk_0_5", "1.0"), ("gk_0_35", "1.0"), ("gk_0_4", "1.0"), ("gk_0_45", "1.0"), ("top_p_0_9", "1.0"),
           ("top_p_0_95", "1.0"), ("ministral_recipe", "0.7")]

loaded, rows, pooled_by = {}, [], {}
for cfg, T in CONFIGS:
    paths = {pl: M / f"{cfg}_{pl}_think_t{T}_metrics.json" for pl in lb.PLATFORMS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"MISSING {cfg}: {missing}")
    L = {pl: lb.load_platform(paths[pl]) for pl in lb.PLATFORMS}
    for pl in lb.PLATFORMS:
        assert len(L[pl]["task_ids"]) == lb.EXPECTED_TASKS[pl], (cfg, pl, len(L[pl]["task_ids"]))
        assert set(L[pl]["n"].tolist()) == {10}, (cfg, pl)
    pooled = lb.pool_platform_metrics(L, cfg)
    loaded[cfg] = L
    pooled_by[cfg] = pooled
    rows.append({"config": cfg + " pooled", "_pooled_metrics": pooled,
                 **{f"pooled_pass@{k}": pooled["pass_at_k"][str(k)] for k in lb.K_VALUES}})
    print(f"{cfg}: atcoder N={len(L['atcoder']['task_ids'])} leetcode N={len(L['leetcode']['task_ids'])} "
          f"pooled N={pooled['num_tasks']} n={pooled['num_samples_per_task']}")

print("\n### Pooled canonical table (N=420)\n")
print(lb.render_pooled_table(rows))
print("\n### gk_0_5 per platform\n")
print(lb.headline_table([(M / "gk_0_5_atcoder_think_t1.0_metrics.json", "gk_0_5 atcoder"),
                         (M / "gk_0_5_leetcode_think_t1.0_metrics.json", "gk_0_5 leetcode")],
                        columns=lb.CANONICAL_COLUMNS))


def per_task(cfg, k):
    L = loaded[cfg]
    ids = [f"{pl}::{t}" for pl in lb.PLATFORMS for t in L[pl]["task_ids"]]
    n = np.concatenate([L[pl]["n"] for pl in lb.PLATFORMS])
    c = np.concatenate([L[pl]["c"] for pl in lb.PLATFORMS])
    return dict(zip(ids, estimate_pass_at_k(n, c, k).astype(float)))


def paired(a_cfg, b_cfg, k):
    a, b = per_task(a_cfg, k), per_task(b_cfg, k)
    same = set(a) == set(b) and len(a) == len(b) == 420
    order_same = list(a) == list(b)
    if not same:
        raise SystemExit(f"task id sets differ {a_cfg} vs {b_cfg}")
    ids = list(a)
    return np.array([a[i] - b[i] for i in ids]), same, order_same


def boot_rows(pairs):
    print("| comparison | metric | resamples | observed diff (pp) | 95% CI (pp) | CI excludes 0 | share diff<=0 |")
    print("|:--|:--|--:|--:|:--:|:--:|--:|")
    for a_cfg, b_cfg in pairs:
        for k in (1, 10):
            d, same, order_same = paired(a_cfg, b_cfg, k)
            for nb in (2000, 5000):
                idx = lb.bootstrap_indices(len(d), n_boot=nb, seed=lb.SEED)
                lo, hi = lb.bootstrap_ci(d, idx=idx)
                share = float((d[idx].mean(axis=1) <= 0).mean())
                excl = "yes" if (lo > 0 or hi < 0) else "no"
                print(f"| {a_cfg} - {b_cfg} | pass@{k} | {nb} | {100*d.mean():.2f} | "
                      f"{100*lo:.2f} to {100*hi:.2f} | {excl} | {share:.4f} |")
            if k == 1:
                print(f"<!-- id check {a_cfg} vs {b_cfg}: identical 420-id set={same}, "
                      f"same concatenation order={order_same} -->")


print("\n### Paired task bootstrap (tasks = resampling unit; seed 42; percentile 95% CI)\n")
boot_rows([("gk_0_5", "gk_0_35"), ("gk_0_5", "gk_0_4"), ("gk_0_5", "gk_0_45"), ("gk_0_5", "top_p_0_9")])

print("\n### Descriptive: GPU2 k values (one run per config; pooled N=420)\n")
print("| k | config | pass@1 | pass@10 | term% | mean_tok |")
print("|--:|:--|--:|--:|--:|--:|")
for k_, cfg in (("0.35", "gk_0_35"), ("0.4", "gk_0_4"), ("0.45", "gk_0_45"), ("0.5", "gk_0_5")):
    p = pooled_by[cfg]
    print(f"| {k_} | {cfg} | {100*p['pass_at_k']['1']:.2f}% | {100*p['pass_at_k']['10']:.2f}% | "
          f"{p['termination_pct']:.1f}% | {p['mean_token_size']:.0f} |")
print("\nAdjacent-k paired differences (same bootstrap):\n")
boot_rows([("gk_0_4", "gk_0_35"), ("gk_0_45", "gk_0_4"), ("gk_0_5", "gk_0_45")])
