import json
import numpy as np
from math import comb
import sys
D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx_vs_vllm_60"  # dir with mlx/ and vllm/ subdirs
def load(side):
    out = {}
    for p in ("atcoder", "leetcode"):
        for t in json.load(open(f"{D}/{side}/pless_default_{p}_think_t1.0_metrics.json"))["per_task"]:
            out[t["task_id"]] = t["num_correct"]
    return out
m, v = load("mlx"), load("vllm")
ids = sorted(m)
def pk(c, k, n=10):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
def pooled(cs, k): return np.mean([pk(c, k) for c in cs])
obs = {k: pooled([m[t] for t in ids], k) - pooled([v[t] for t in ids], k) for k in (1, 5, 10)}
rng = np.random.default_rng(0)
tot = np.array([m[t] + v[t] for t in ids])
null = {k: [] for k in (1, 5, 10)}
for _ in range(20000):
    # under H0, mlx's 10 samples are a random 10 of the task's 20 pooled samples
    cm = rng.hypergeometric(tot, 20 - tot, 10)
    cv = tot - cm
    for k in (1, 5, 10):
        null[k].append(pooled(cm, k) - pooled(cv, k))
for k in (1, 5, 10):
    a = np.array(null[k])
    p = np.mean(np.abs(a) >= abs(obs[k]) - 1e-12)
    print(f"pass@{k}: observed Δ = {obs[k]*100:+.2f} pp, within-task permutation two-sided p = {p:.3f}")
once_m = sum(m[t] > 0 for t in ids); once_v = sum(v[t] > 0 for t in ids)
print(f"tasks solved at least once: mlx {once_m}/60, vllm {once_v}/60; differ only on",
      [t for t in ids if (m[t] > 0) != (v[t] > 0)], [(m[t], v[t]) for t in ids if (m[t] > 0) != (v[t] > 0)])
low = [t for t in ids if 0 < m[t] + v[t] and min(m[t], v[t]) < 5]
print("low-count tasks (either side <5 correct), (mlx,vllm):", [(t, m[t], v[t]) for t in low])
