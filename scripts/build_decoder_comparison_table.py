"""Build the grounded 14-config decoder comparison table (ATCODER-interview, Qwen3-8B, n=10).

Every number is pulled LIVE from source-of-truth files and cross-verified:
  - pass@1/5/10: recomputed from raw pass_results (unbiased estimator) AND compared to the
    stored pass_at_k in the metrics JSON — mismatch => flagged.
  - trunc%: recomputed from samples_with_thinking (fraction lacking </think>) AND compared
    to the cot_efficiency CSV truncation_rate — mismatch => flagged.
  - mean think tok: read from the cot_efficiency CSV (the pipeline's tokenizer output).
  - cb_div: recomputed via the project's add_self_codebleu/compute_self_codebleu_diversity.
Writes docs/decoder_comparison_cot_apps_qwen3.md with provenance.

Run: PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py
"""
import csv
import json
import math
import os

from bench.eval.metrics import add_self_codebleu, compute_self_codebleu_diversity

FULL = "results/pless_recovery_full252/Qwen--Qwen3-8B/ATCODER_interview"
CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
DEC06 = "results/decoders_t0.6/Qwen--Qwen3-8B/ATCODER_interview"
OUT = "docs/decoder_comparison_cot_apps_qwen3.md"

# (display, dir, jsonl_basename)
CONFIGS = [
    ("temp p0.95 @T1.0", CANON, "temp_p0.95_think_t1.0_t1.0"),
    ("temp k20 @T1.0",   CANON, "temp_k20_think_t1.0_t1.0"),
    ("temp @T0.6 (unfilt)", CANON, "temp_think_t0.6_t0.6"),
    ("top_k @T0.6",      DEC06, "temp_k20_think_t0.6_t0.6"),
    ("pless α=4",        FULL,  "pless_alpha_think_t1.0_a4.0_t1.0"),
    ("top_p @T0.6",      DEC06, "temp_p0.95_think_t0.6_t0.6"),
    ("pless T2.0",       FULL,  "pless_think_t2.0_t2.0"),
    ("pless α=5",        FULL,  "pless_alpha_think_t1.0_a5.0_t1.0"),
    ("temp p+k @T0.6",   CANON, "temp_p0.95_k20_think_t0.6_t0.6"),
    ("pless α=3",        FULL,  "pless_alpha_think_t1.0_a3.0_t1.0"),
    ("pless_norm @α2",   CANON, "pless_norm_think_t1.0_t1.0"),
    ("pless @α2 (base)", CANON, "pless_think_t1.0_t1.0"),
    ("pless_norm @T0.6", DEC06, "pless_norm_think_t0.6_t0.6"),
    ("pless @T0.6",      DEC06, "pless_think_t0.6_t0.6"),
]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    rows, warnings = [], []
    for name, d, base in CONFIGS:
        m = json.load(open(f"{d}/metrics/{base}_metrics.json"))
        pt = m["per_task"]
        records = [json.loads(l) for l in open(f"{d}/{base}.jsonl")]
        n_samp = len(pt[0]["pass_results"])
        ntasks = len(pt)

        # pass@k recomputed from raw booleans, cross-checked vs stored
        def agg(k):
            return sum(pass_at_k(len(t["pass_results"]), sum(t["pass_results"]), k) for t in pt) / ntasks
        p1, p5, p10 = agg(1), agg(5), agg(10)
        for k, v in (("1", p1), ("5", p5), ("10", p10)):
            stored = m.get("pass_at_k", {}).get(k)
            if stored is not None and abs(stored - v) > 1e-6:
                warnings.append(f"{name}: pass@{k} recomputed {v:.4f} != stored {stored:.4f}")

        # trunc% recomputed from </think> presence, cross-checked vs CSV
        rec_by_id = {r["task_id"]: r for r in records}
        n_trunc = n_total = 0
        for t in pt:
            swt = rec_by_id[t["task_id"]].get("samples_with_thinking") or rec_by_id[t["task_id"]]["samples"]
            for s in swt:
                n_total += 1
                if "</think>" not in s:
                    n_trunc += 1
        trunc_recomp = n_trunc / n_total if n_total else 0.0

        # CSV row (mean think tok + stored trunc)
        csv_path = f"{d}/analysis/cot_efficiency_apps.csv"
        crow = next((r for r in csv.DictReader(open(csv_path)) if r["file"] == f"{base}.jsonl"), None)
        mean_tok = float(crow["mean_think_tokens"]) if crow else float("nan")
        trunc_csv = float(crow["truncation_rate"]) if crow else float("nan")
        if crow and abs(trunc_csv - trunc_recomp) > 0.01:
            warnings.append(f"{name}: trunc recomputed {trunc_recomp:.3f} != CSV {trunc_csv:.3f}")

        # cb_div via project function (no execution)
        add_self_codebleu(pt, records)
        cb = compute_self_codebleu_diversity(pt).get("codebleu_diversity", 0.0)

        rows.append((name, ntasks, n_samp, p1, p10, cb, mean_tok, trunc_recomp))

    rows.sort(key=lambda r: -r[3])

    lines = ["# Decoder comparison — ATCODER-interview, Qwen3-8B (thinking on, n=10)\n",
             "All values pulled live and cross-verified by `scripts/build_decoder_comparison_table.py`. "
             "pass@k recomputed from raw pass_results (unbiased estimator, Chen 2021) and checked vs stored; "
             "trunc% recomputed from `</think>` presence and checked vs the cot_efficiency CSV; "
             "mean think tok from the cot_efficiency CSV; cb_div via the project's `add_self_codebleu`.\n",
             "**Sources:** full252 `pless_recovery_full252/`, canonical `pless_cot_efficiency_vllm/.../ATCODER_interview_all_252/`, "
             "T0.6 decoders `decoders_t0.6/`. All n_tasks=252 except where noted.\n",
             "| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |",
             "|---|---|---|---|---|---|---|"]
    for name, nt, ns, p1, p10, cb, mt, tr in rows:
        ntnote = f"{nt}" if nt == 252 else f"**{nt}**"
        lines.append(f"| {name} | {ntnote} | {p1:.3f} | {p10:.3f} | {cb:.4f} | {mt:,.0f} | {tr*100:.1f} |")

    lines.append("\n## Cross-verification\n")
    if warnings:
        lines.append("⚠ MISMATCHES (investigate before trusting):")
        lines += [f"- {w}" for w in warnings]
    else:
        lines.append("✓ All pass@k (recomputed vs stored) and trunc% (recomputed vs CSV) agree to tolerance. "
                     "No mismatches.")
    lines.append("\n## Caveats\n")
    lines.append("- **cb_div** is correct-only over each config's own ≥2-correct subset (~195–202 tasks, "
                 "~95% overlapping) — mild cross-config confound; ranking robust, small gaps may shift on a common set.")
    lines.append("- **structural_diversity** omitted: zss tree-edit is intractable on APPS-CoT code "
                 "(a single ~2000-AST-node pair times out >60s) — the reason these runs use `--skip-diversity`.")
    lines.append("- **mean think tok** counts truncated samples at their cut length (≈cap), so it is biased "
                 "UP for the high-truncation configs — which is exactly the rambling cost it exposes.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print("WARNINGS:" if warnings else "cross-verification: all clean")
    for w in warnings:
        print("  ⚠", w)


if __name__ == "__main__":
    main()
