"""Paired comparison of the BOS check: MBPP-full greedy, default (double BOS) vs --single-bos.

Reads, for one model, the two arms written by run_bos_check_mbpp_chat.sh and graded locally with bench.eval:
    <out-root>/{double_bos,single_bos}/<model dir>/greedy_t1.0.jsonl
    <out-root>/{double_bos,single_bos}/<model dir>/metrics/greedy_t1.0_metrics.json
and reports
  * the number of tasks whose greedy output text differs between the arms;
  * pass@1 of each arm and the paired difference single - double (percentage points), with a paired bootstrap 95%
    percentile CI over tasks (--n-boot resamples, default 5000; --seed, default 0);
  * an exact two-sided sign test on the discordant tasks (single-only vs double-only passes; ties dropped).

    uv run python scripts/compare_bos_check.py --out-root results/bos_check_2026_11 \
        --model codellama/CodeLlama-7b-Instruct-hf [--json-out cmp.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ("double_bos", "single_bos")
STEM = "greedy_t1.0"


def sign_test_p(b: int, c: int) -> float:
    """Exact two-sided sign test: P under Binomial(b + c, 1/2) of a split at least as extreme as (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_stats(double: dict, single: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    """double/single: {task_id: passed (bool)} over the same task set."""
    if set(double) != set(single):
        raise ValueError(f"task sets differ: {len(set(double) ^ set(single))} task ids are in only one arm")
    ids = sorted(double)
    d = np.array([bool(double[i]) for i in ids], dtype=float)
    s = np.array([bool(single[i]) for i in ids], dtype=float)
    delta = s - d
    n = len(ids)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    b = int(((s == 1) & (d == 0)).sum())
    c = int(((s == 0) & (d == 1)).sum())
    return {"n_tasks": n, "pass1_double": float(d.mean() * 100), "pass1_single": float(s.mean() * 100),
            "diff_pp": float(delta.mean() * 100), "ci95_pp": [float(lo), float(hi)],
            "single_only": b, "double_only": c, "sign_test_p": sign_test_p(b, c),
            "n_boot": n_boot, "bootstrap_seed": seed}


def _load_arm(root: Path, arm: str, model: str):
    d = root / arm / model.replace("/", "--")
    recs = [json.loads(line) for line in open(d / f"{STEM}.jsonl") if line.strip()]
    ids = [r["task_id"] for r in recs]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{d / STEM}.jsonl: duplicate task_ids")
    want = True if arm == "single_bos" else None
    bad = [r["task_id"] for r in recs if r.get("single_bos") is not want]
    if bad:
        raise SystemExit(f"{arm}: {len(bad)} records have single_bos={not want!s} (arms swapped or mixed?)")
    texts = {r["task_id"]: r["samples"][0] for r in recs}
    metrics = json.load(open(d / "metrics" / f"{STEM}_metrics.json"))
    passes = {t["task_id"]: bool(t["pass_results"][0]) for t in metrics["per_task"]}
    if set(passes) != set(texts):
        raise SystemExit(f"{arm}: metrics cover {len(passes)} tasks but the jsonl has {len(texts)}; re-grade it")
    return texts, passes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    (t_d, p_d), (t_s, p_s) = (_load_arm(args.out_root, a, args.model) for a in ARMS)
    if set(t_d) != set(t_s):
        raise SystemExit(f"arms cover different tasks ({len(t_d)} vs {len(t_s)})")
    differs = sorted(i for i in t_d if t_d[i] != t_s[i])
    res = {"model": args.model, "out_root": str(args.out_root), "n_text_differs": len(differs),
           "text_differs_task_ids": differs, **paired_stats(p_d, p_s, n_boot=args.n_boot, seed=args.seed)}
    lo, hi = res["ci95_pp"]
    print(f"{args.model}  ({res['n_tasks']} tasks, {args.out_root})")
    print(f"  greedy outputs differ: {len(differs)} / {res['n_tasks']}")
    print(f"  pass@1 double_bos {res['pass1_double']:.2f}   single_bos {res['pass1_single']:.2f}")
    print(f"  paired diff (single - double) {res['diff_pp']:+.2f} pp, bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}] "
          f"({args.n_boot} resamples, seed {args.seed})")
    print(f"  discordant: single-only {res['single_only']}, double-only {res['double_only']}; "
          f"exact sign test p = {res['sign_test_p']:.4g}")
    if args.json_out:
        args.json_out.write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
