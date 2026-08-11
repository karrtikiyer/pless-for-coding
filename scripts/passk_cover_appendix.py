#!/usr/bin/env python
"""Emit the Paper B appendix tables: full pass@k {1,3,5,8,10} and cover@t {0.3,0.5,0.7}
(non-distinct + distinct) for the Table-1 config set, both models.

All values recomputed LIVE from each config's metrics JSON per_task (pass_results,
num_correct, num_distinct_correct) — pass@k via the unbiased Chen-2021 estimator, cover@t as
the fraction of problems with >= t*n correct (num_distinct_correct for the distinct variant).
Cross-checks every recomputed value against the stored pass_at_k / cover_at_t (staleness guard)
and aborts on mismatch. No execution / no GPU. Prints LaTeX for both tables.
"""
from __future__ import annotations
import json, math, sys

_Q_CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
_DS_FIX = "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
_Q_RENYI = "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview"
_DS_RENYI = "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
_Q_TOPP = "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview"
_DS_TOPP = "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"


def _gk(d):
    return [(f"$G_k{{=}}{k}$", d, f"pless_renyi_think_t1.0_k{k}_t1.0")
            for k in ("1.6", "0.8", "0.4", "0.2", "0.1", "0.05")]


# Same rows, same order, as main.tex Table 1.
SETS = {
    "DeepSeek-R1-Distill-Llama-8B": (
        [("p-less $k{=}2$ (default)", _DS_FIX, "pless_think_t1.0_t1.0"),
         ("p-less-norm (default)", _DS_FIX, "pless_norm_think_t1.0_t1.0")]
        + _gk(_DS_RENYI)
        + [("temp $T{=}0.6,p0.95$ (rec)", _DS_FIX, "temp_p0.95_think_t0.6_t0.6"),
           ("temp $T{=}1.0$ (unfilt) $^\\dagger$", _DS_TOPP, "temp_think_t1.0_t1.0")]),
    "Qwen3-8B": (
        [("p-less $k{=}2$ (default)", _Q_CANON, "pless_think_t1.0_t1.0"),
         ("p-less-norm (default)", _Q_CANON, "pless_norm_think_t1.0_t1.0")]
        + _gk(_Q_RENYI)
        + [("temp $T{=}0.6,p0.95,k20$ (rec)", _Q_CANON, "temp_p0.95_k20_think_t0.6_t0.6"),
           ("temp $T{=}1.0$ (unfilt) $^\\dagger$", _Q_TOPP, "temp_think_t1.0_t1.0")]),
}

KS = [1, 3, 5, 8, 10]
TS = [0.3, 0.5, 0.7]


def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(dirp, base):
    m = json.load(open(f"{dirp}/metrics/{base}_metrics.json"))
    pt = m["per_task"]
    n = len(pt[0]["pass_results"])
    N = len(pt)
    passk = {k: sum(pak(len(t["pass_results"]), t["num_correct"], k) for t in pt) / N for k in KS}
    cov = {t: sum(1 for t_ in pt if t_["num_correct"] >= t * n) / N * 100 for t in TS}
    covd = {t: sum(1 for t_ in pt if t_.get("num_distinct_correct", 0) >= t * n) / N * 100 for t in TS}
    # staleness cross-check against stored
    warn = []
    sk = m.get("pass_at_k", {})
    for k in (1, 3, 5, 10):
        if str(k) in sk and abs(sk[str(k)] - passk[k]) > 1e-6:
            warn.append(f"pass@{k} stored {sk[str(k)]:.4f} != live {passk[k]:.4f}")
    sc, scd = m.get("cover_at_t", {}), m.get("cover_at_t_distinct", {})
    for t in TS:
        if str(t) in sc and abs(sc[str(t)] - cov[t]) > 1e-6:
            warn.append(f"cover@{t} stored {sc[str(t)]:.2f} != live {cov[t]:.2f}")
        if str(t) in scd and abs(scd[str(t)] - covd[t]) > 1e-6:
            warn.append(f"cover_d@{t} stored {scd[str(t)]:.2f} != live {covd[t]:.2f}")
    return passk, cov, covd, warn


def main():
    allwarn = []
    rows = {}
    for model, cfgs in SETS.items():
        rows[model] = []
        for label, d, base in cfgs:
            passk, cov, covd, warn = load(d, base)
            rows[model].append((label, passk, cov, covd))
            allwarn += [f"{model} {label}: {w}" for w in warn]
    if allwarn:
        print("STALENESS MISMATCHES (aborting):", file=sys.stderr)
        print("\n".join(allwarn), file=sys.stderr)
        sys.exit(1)

    # ---- persist a machine-readable markdown artifact (pass@8 is NOT stored in the metrics
    # JSONs, so this is the durable home for the full pass@k / cover@t values) ----
    md = ["# Full pass@k and cover@t — APPS-interview, n=10, thinking on (Paper B Table-1 config set)\n",
          "Recomputed live from each config's `per_task` (pass_results / num_correct / "
          "num_distinct_correct) by `scripts/passk_cover_appendix.py`; stored pass@{1,3,5,10} and "
          "cover@{.3,.5,.7} cross-verified (no staleness). **pass@8 is derived here — it is not stored "
          "in the metrics JSONs** (bench.eval default k=1,3,5,10). cover@t = %% problems solved on ≥t·n "
          "samples; `-d` = ≥t·n AST-distinct correct solutions.\n"]
    for model, rws in rows.items():
        md.append(f"\n## {model}\n")
        md.append("| Config | pass@1 | pass@3 | pass@5 | pass@8 | pass@10 | cov@.3 | cov@.5 | cov@.7 | cov@.3-d | cov@.5-d | cov@.7-d |")
        md.append("|" + "---|" * 12)
        for label, pk, cov, covd in rws:
            lab = label.replace("$", "").replace("{=}", "=").replace("^\\dagger", "†").replace("\\", "")
            cells = [f"{pk[k]:.3f}" for k in KS] + [f"{cov[t]:.1f}" for t in TS] + [f"{covd[t]:.1f}" for t in TS]
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
    import os
    os.makedirs("docs", exist_ok=True)
    open("docs/decoder_passk_cover_apps.md", "w").write("\n".join(md) + "\n")
    print("% wrote docs/decoder_passk_cover_apps.md")
    print("% cross-check: all recomputed pass@k / cover@t match stored values (no staleness).\n")

    # ---- pass@k table ----
    print(r"\begin{table}[ht]\centering\small")
    print(r"\caption{Full pass@$k$ (APPS-interview, $n{=}10$, thinking on), recomputed from the same "
          r"runs as Table~\ref{tab:main} (unbiased estimator; pass@1/@10 match Table~\ref{tab:main}). "
          r"$^\dagger$ = strongest swept temperature.}")
    print(r"\label{tab:passk}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Config & pass@1 & pass@3 & pass@5 & pass@8 & pass@10 \\")
    for model, rws in rows.items():
        print(r"\midrule")
        print(r"\multicolumn{6}{l}{\textit{" + model + r"}}\\")
        for label, passk, _, _ in rws:
            print(f"{label} & " + " & ".join(f"{passk[k]:.3f}" for k in KS) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ---- cover@t table ----
    print(r"\begin{table}[ht]\centering\small")
    print(r"\caption{cover@$t$ (\% of problems solved on $\ge t$ of the $n{=}10$ samples) and its "
          r"distinct variant cover@$t$-d (\% with $\ge tn$ AST-\emph{distinct} correct solutions). "
          r"Same runs / config set as Table~\ref{tab:main}; cover@$t$-d $\le$ cover@$t$ always. "
          r"cover@$0.1$ equals pass@10 and is omitted.}")
    print(r"\label{tab:cover}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Config & cov@.3 & cov@.5 & cov@.7 & cov@.3-d & cov@.5-d & cov@.7-d \\")
    for model, rws in rows.items():
        print(r"\midrule")
        print(r"\multicolumn{7}{l}{\textit{" + model + r"}}\\")
        for label, _, cov, covd in rws:
            cells = [f"{cov[t]:.1f}" for t in TS] + [f"{covd[t]:.1f}" for t in TS]
            print(f"{label} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
