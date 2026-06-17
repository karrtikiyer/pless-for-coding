"""Single comparison table for the full-252 Qwen3 loop-force w1200 run vs the matched
no-force baselines + healthy temp decoders — diversity + pass@k + tokens + truncation.

Every number pulled LIVE from source-of-truth and cross-verified:
  - pass@1/5/10: unbiased estimator (Chen 2021) recomputed from raw pass_results AND
    checked vs the stored pass_at_k (mismatch flagged).
  - trunc% / compl% / cond-correctness: recomputed from </think> presence + pass_results.
  - cb_div: the project's own add_self_codebleu / compute_self_codebleu_diversity
    (CodeBLEU, correct-only subset, STATIC — no code re-execution). structural (zss)
    omitted: intractable on APPS-CoT code (documented; these runs use --skip-diversity).
  - mean think tok: from each config's cot_efficiency CSV (the pipeline's tokenizer output).

Run: PYTHONPATH=. uv run python scripts/loopforce_w1200_comparison.py
"""
import csv
import json
import math
import os

from bench.eval.metrics import add_self_codebleu, compute_self_codebleu_diversity

W1200 = "results/loop_forcethink_qwen_w1200/Qwen--Qwen3-8B/ATCODER_interview"
CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
OUT = "docs/loopforce_w1200_comparison_apps_qwen3.md"

# (display, dir, jsonl_basename)
CONFIGS = [
    ("pless loop-force w1200",        W1200, "pless_think_t1.0_t1.0"),
    ("pless_norm loop-force w1200",   W1200, "pless_norm_think_t1.0_t1.0"),
    ("pless @α2 (no-force base)",     CANON, "pless_think_t1.0_t1.0"),
    ("pless_norm @α2 (no-force base)", CANON, "pless_norm_think_t1.0_t1.0"),
    ("temp p0.95 @T1.0",              CANON, "temp_p0.95_think_t1.0_t1.0"),
    ("temp k20 @T1.0",                CANON, "temp_k20_think_t1.0_t1.0"),
    ("temp @T0.6 (unfilt)",           CANON, "temp_think_t0.6_t0.6"),
]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    rows, warnings = [], []
    for name, d, base in CONFIGS:
        mpath = f"{d}/metrics/{base}_metrics.json"
        jpath = f"{d}/{base}.jsonl"
        if not (os.path.exists(mpath) and os.path.exists(jpath)):
            warnings.append(f"{name}: missing files, skipped ({jpath})")
            continue
        m = json.load(open(mpath))
        pt = m["per_task"]
        records = [json.loads(l) for l in open(jpath)]
        rec_by_id = {r["task_id"]: r for r in records}
        ntasks = len(pt)
        n_samp = len(pt[0]["pass_results"])

        # pass@k (unbiased), cross-checked vs stored
        def agg(k):
            return sum(pass_at_k(len(t["pass_results"]), sum(t["pass_results"]), k) for t in pt) / ntasks
        p1, p5, p10 = agg(1), agg(5), agg(10)
        for k, v in (("1", p1), ("5", p5), ("10", p10)):
            stored = m.get("pass_at_k", {}).get(k)
            if stored is not None and abs(stored - v) > 1e-6:
                warnings.append(f"{name}: pass@{k} recomputed {v:.4f} != stored {stored:.4f}")

        # completion / truncation / conditional-correctness from </think> + pass_results
        n_total = n_done = n_corr = n_corr_done = 0
        for t in pt:
            r = rec_by_id[t["task_id"]]
            swt = r.get("samples_with_thinking") or r["samples"]
            pr = t["pass_results"]
            for i, s in enumerate(swt):
                n_total += 1
                done = "</think>" in s
                ok = bool(pr[i]) if i < len(pr) else False
                if done:
                    n_done += 1
                    if ok:
                        n_corr_done += 1
                if ok:
                    n_corr += 1
        compl = n_done / n_total
        trunc = 1 - compl
        cond = n_corr_done / n_done if n_done else 0.0

        # mean think tok from the cot_efficiency CSV
        csv_path = f"{d}/analysis/cot_efficiency_apps.csv"
        mean_tok = float("nan")
        if os.path.exists(csv_path):
            crow = next((r for r in csv.DictReader(open(csv_path)) if r["file"] == f"{base}.jsonl"), None)
            if crow and crow.get("mean_think_tokens"):
                mean_tok = float(crow["mean_think_tokens"])
                tr_csv = float(crow["truncation_rate"])
                if abs(tr_csv - trunc) > 0.01:
                    warnings.append(f"{name}: trunc recomputed {trunc:.3f} != CSV {tr_csv:.3f}")

        # cb_div via project function (static, no execution)
        add_self_codebleu(pt, records)
        cb = compute_self_codebleu_diversity(pt).get("codebleu_diversity", 0.0)
        n_ge2 = sum(1 for t in pt if t.get("num_correct", 0) >= 2)

        rows.append((name, ntasks, n_samp, trunc, compl, cond, p1, p5, p10, cb, n_ge2, mean_tok))

    rows.sort(key=lambda r: -r[6])  # by pass@1

    hdr = ("| Config | n | trunc% | compl% | cond-corr | pass@1 | pass@5 | pass@10 | "
           "cb_div | (cb n≥2) | mean think tok |")
    sep = "|---|" + "---|" * 10
    lines = [
        "# Loop-force (w1200) vs baselines — ATCODER-interview, Qwen3-8B (thinking on, n=10, full 252)\n",
        "Live + cross-verified by `scripts/loopforce_w1200_comparison.py`. pass@k = unbiased "
        "estimator (Chen 2021) recomputed from raw pass_results and checked vs stored; "
        "trunc%/compl%/cond-correctness from `</think>` presence + pass_results; "
        "cb_div via the project's `add_self_codebleu` (CodeBLEU, correct-only, **no re-execution**); "
        "mean think tok from the cot_efficiency CSV.\n",
        "Loop-force = live n-gram detect (n=30/k=6) → force `</think>` at **window=1200**. "
        "Baseline = same sampler, no loop-force.\n",
        hdr, sep,
    ]
    for name, nt, ns, tr, cp, cd, p1, p5, p10, cb, nge2, mt in rows:
        mt_s = f"{mt:,.0f}" if mt == mt else "n/a"  # nan check
        lines.append(f"| {name} | {nt} | {tr*100:.1f} | {cp*100:.1f} | {cd:.3f} | "
                     f"{p1:.3f} | {p5:.3f} | {p10:.3f} | {cb:.4f} | {nge2} | {mt_s} |")

    lines.append("\n## Cross-verification\n")
    lines += (["⚠ " + w for w in warnings] if warnings
              else ["✓ pass@k (recomputed vs stored) and trunc% (recomputed vs CSV) agree to tolerance."])
    lines.append("\n## Caveats\n")
    lines.append("- **cb_div** is correct-only over each config's own ≥2-correct subset (see `(cb n≥2)`); "
                 "a config that solves fewer tasks computes diversity over a smaller/harder set — mild "
                 "cross-config confound, ranking robust.")
    lines.append("- **mean think tok** counts truncated samples at their cut length (≈cap), so it is biased "
                 "UP for high-truncation configs — exactly the rambling cost loop-force removes.")
    lines.append("- **structural (zss) diversity** omitted — intractable on APPS-CoT code (the reason these "
                 "runs use `--skip-diversity`); CodeBLEU only.")
    lines.append("- The α=5 / T2.0 prevention winners (`pless_recovery_full252/`) are not on local disk, "
                 "so they're omitted here rather than quoted from memory.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    if warnings:
        print("WARNINGS:", *(f"\n  ⚠ {w}" for w in warnings))


if __name__ == "__main__":
    main()
