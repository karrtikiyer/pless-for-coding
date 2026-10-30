import json
import numpy as np
from bench.eval.metrics import compute_pass_at_k

import sys
D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx_vs_vllm_60"  # dir with mlx/ and vllm/ subdirs
K = [1, 5, 10]


def load(side):
    per, meta = {}, {}
    for plat in ("atcoder", "leetcode"):
        m = json.load(open(f"{D}/{side}/pless_default_{plat}_think_t1.0_metrics.json"))
        for t in m["per_task"]:
            per[t["task_id"]] = t
        for line in open(f"{D}/{side}/pless_default_{plat}_think_t1.0.jsonl"):
            r = json.loads(line)
            meta[r["task_id"]] = (plat, r["difficulty"])
    return per, meta


mlx, meta = load("mlx")
vllm, _ = load("vllm")
ids = sorted(mlx, key=lambda t: ({"easy": 0, "medium": 1, "hard": 2}[meta[t][1]], meta[t][0], t))
assert set(ids) == set(vllm) and len(ids) == 60


def pk(per, subset):
    return compute_pass_at_k([per[t] for t in subset], K)


def fmt(x):
    return f"{x*100:5.1f}"


print("## Per task (num_correct out of 10, pass@k in %)\n")
print("| task | plat | diff | mlx c | vllm c | Δc | mlx p@1 | vllm p@1 | mlx p@5 | vllm p@5 | mlx p@10 | vllm p@10 |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for t in ids:
    a, b = pk(mlx, [t]), pk(vllm, [t])
    cm, cv = mlx[t]["num_correct"], vllm[t]["num_correct"]
    print(f"| {t} | {meta[t][0]} | {meta[t][1]} | {cm} | {cv} | {cm-cv:+d} | "
          + " | ".join(f"{fmt(a[str(k)])} | {fmt(b[str(k)])}" for k in K) + " |")

rng = np.random.default_rng(0)


def boot_ci(subset, k, n_boot=20000):
    mc = [mlx[t] for t in subset]
    vc = [vllm[t] for t in subset]
    per_m = np.array([compute_pass_at_k([x], [k])[str(k)] for x in mc])
    per_v = np.array([compute_pass_at_k([x], [k])[str(k)] for x in vc])
    diff = per_m - per_v
    idx = rng.integers(0, len(subset), (n_boot, len(subset)))
    bs = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("\n## Aggregate (pass@k in %, Δ = mlx − vllm, paired task-bootstrap 95% CI)\n")
print("| subset | N | metric | mlx | vllm | Δ (pp) | 95% CI (pp) |")
print("|---|---:|---|---:|---:|---:|---|")
groups = [("all", ids)]
groups += [(d, [t for t in ids if meta[t][1] == d]) for d in ("easy", "medium", "hard")]
groups += [(p, [t for t in ids if meta[t][0] == p]) for p in ("atcoder", "leetcode")]
for name, sub in groups:
    a, b = pk(mlx, sub), pk(vllm, sub)
    for k in K:
        d, lo, hi = boot_ci(sub, k)
        print(f"| {name} | {len(sub)} | pass@{k} | {fmt(a[str(k)])} | {fmt(b[str(k)])} | "
              f"{d*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] |")

deltas = [mlx[t]["num_correct"] - vllm[t]["num_correct"] for t in ids]
nz = [x for x in deltas if x]
print(f"\nTasks identical (same num_correct): {deltas.count(0)}/60; favour mlx: "
      f"{sum(x > 0 for x in nz)}, favour vllm: {sum(x < 0 for x in nz)}; net Δc = {sum(deltas):+d} of 600 samples")
from scipy import stats
pos, neg = sum(x > 0 for x in nz), sum(x < 0 for x in nz)
if pos + neg:
    print(f"Sign test p = {stats.binomtest(pos, pos+neg, 0.5).pvalue:.3f}")
sat = sum(1 for t in ids if vllm[t]["num_correct"] in (0, 10) and mlx[t]["num_correct"] in (0, 10))
print(f"Tasks saturated (0/10 or 10/10) on BOTH backends: {sat}/60")
for side, per in (("mlx", mlx), ("vllm", vllm)):
    st = {}
    for t in ids:
        for s in per[t]["statuses"]:
            st[s] = st.get(s, 0) + 1
    print(f"{side} status counts: {st}")
