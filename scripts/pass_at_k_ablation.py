#!/usr/bin/env python
"""Ablation: where does the Rényi G_k pass@k gain come from, vs the p-less (alpha=2) baseline?

Pairs every scored G_k arm against the alpha=2 baseline (=G_2=pless) at the PROBLEM level
(same task set; samples are independent draws across configs, so only per-problem pass RATES
are comparable — never individual-sample fates). For each k it reports:

  A. Winner/loser ledger + significance from two-level bootstrap 95% CIs on Δpass@1 and Δpass@10
     (resample problems AND the 10 draws), plus cov-McNemar on the coverage-status change.
  B. Difficulty-stratified deltas (bins by baseline pass@1) — Matthew vs loop-escape.
  C. Loop-escape attribution: correlate delta pass@1 with baseline truncation; split the pass@1
     gain by each improved problem's baseline failure mode (truncation- vs wrong-answer-dominated).
  D. pass@1 vs pass@10 divergence (reliability vs coverage).
  E. delta pass@1 decomposed into 0->positive (newly solvable) vs partial->more.

Auto-discovers G_k arms from the results folder, so new k's are picked up on re-run; arms without
a metrics JSON yet are listed as pending and skipped.

Usage:
  PYTHONPATH=. uv run python scripts/pass_at_k_ablation.py [--model qwen|deepseek] [--boot 5000]
Output: results/_renyi_sweep_full252/analysis/pass_at_k_ablation_<model>.md  (+ stdout summary)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
from scipy import stats as ss

# ---- per-model config: (alpha=2 baseline, G_k sweep dir) ------------------
MODELS = {
    "qwen": {
        "name": "Qwen3-8B",
        "base_dir": "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252",
        "base_file": "pless_think_t1.0_t1.0",
        "renyi_dir": "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview",
    },
    "deepseek": {
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "base_dir": "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview",
        "base_file": "pless_think_t1.0_t1.0",
        "renyi_dir": "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview",
    },
}


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_config(d: str, base: str):
    """Return {task_id: {'pass':[bool], 'nc':int, 'n':int, 'trunc':[bool], 'statuses':[str]}} or None."""
    mpath = f"{d}/metrics/{base}_metrics.json"
    jpath = f"{d}/{base}.jsonl"
    if not (os.path.exists(mpath) and os.path.exists(jpath)):
        return None
    m = json.load(open(mpath))
    rec_by_id = {r["task_id"]: r for r in (json.loads(l) for l in open(jpath))}
    out = {}
    for t in m["per_task"]:
        tid = t["task_id"]
        rec = rec_by_id.get(tid)
        # Truncation is read from samples_with_thinking; it MUST be present and index-aligned
        # to pass_results, else every sample would falsely read as truncated (no </think> in
        # extracted code) and inflate loop-escape attribution. Fail loud rather than fall back.
        sw = rec.get("samples_with_thinking") if rec else None
        if sw is None or len(sw) != len(t["pass_results"]):
            raise SystemExit(
                f"{base} task {tid}: samples_with_thinking missing or misaligned "
                f"({0 if sw is None else len(sw)} vs {len(t['pass_results'])} pass_results) "
                f"— truncation/loop-escape stats would be wrong.")
        trunc = ["</think>" not in s for s in sw]
        out[tid] = {
            "pass": t["pass_results"],
            "nc": t.get("num_correct", sum(t["pass_results"])),
            "n": len(t["pass_results"]),
            "trunc": trunc,
            "statuses": t.get("statuses") or [],
        }
    return out


def stratum(p1: float) -> str:
    if p1 == 0.0:
        return "dead (0)"
    if p1 <= 0.3:
        return "hard (0,0.3]"
    if p1 <= 0.7:
        return "mid (0.3,0.7]"
    return "easy (0.7,1]"


STRATA = ["dead (0)", "hard (0,0.3]", "mid (0.3,0.7]", "easy (0.7,1]"]


def analyse(base, arm, boot=5000, seed=0):
    ids = sorted(set(base) & set(arm))
    n = base[ids[0]]["n"]
    b_p1 = np.array([base[i]["nc"] / base[i]["n"] for i in ids])
    k_p1 = np.array([arm[i]["nc"] / arm[i]["n"] for i in ids])
    d_p1 = k_p1 - b_p1
    b_p10 = np.array([pass_at_k(base[i]["n"], base[i]["nc"], 10) for i in ids])
    k_p10 = np.array([pass_at_k(arm[i]["n"], arm[i]["nc"], 10) for i in ids])
    b_nt = np.array([np.mean(base[i]["trunc"]) if base[i]["trunc"] else 0.0 for i in ids])
    k_nt = np.array([np.mean(arm[i]["trunc"]) if arm[i]["trunc"] else 0.0 for i in ids])

    R = {"n_problems": len(ids), "n_samples": n}
    # A. ledger
    R["improved"] = int((d_p1 > 0).sum())
    R["deteriorated"] = int((d_p1 < 0).sum())
    R["unchanged"] = int((d_p1 == 0).sum())
    R["mean_dp1"] = float(d_p1.mean())
    R["base_p1"], R["arm_p1"] = float(b_p1.mean()), float(k_p1.mean())
    R["base_p10"], R["arm_p10"] = float(b_p10.mean()), float(k_p10.mean())
    R["mean_dp10"] = float((k_p10 - b_p10).mean())
    # McNemar on solved-at-least-once
    bs = b_p1 > 0
    ks = k_p1 > 0
    b_only = int((bs & ~ks).sum())   # regressions
    k_only = int((~bs & ks).sum())   # new solves
    R["solve_lost"], R["solve_gained"] = b_only, k_only
    R["mcnemar_p"] = float(ss.binomtest(min(b_only, k_only), b_only + k_only, 0.5).pvalue) if (b_only + k_only) else 1.0
    # Two-level bootstrap CIs for Δpass@1 AND Δpass@10: resample problems, and within each
    # resampled problem resample its n draws (base/arm independently, since samples are unpaired
    # across configs). This propagates within-problem sampling noise, unlike a problem-only bootstrap.
    rng = np.random.default_rng(seed)
    Bp = np.array([base[i]["pass"] for i in ids], dtype=bool)   # (N, n)
    Ap = np.array([arm[i]["pass"] for i in ids], dtype=bool)    # (N, n)
    N = len(ids)
    pidx = rng.integers(0, N, (boot, N), dtype=np.int32)
    dib = rng.integers(0, n, (boot, N, n), dtype=np.int16)
    dia = rng.integers(0, n, (boot, N, n), dtype=np.int16)
    Bres = np.take_along_axis(Bp[pidx], dib, axis=2)           # (boot, N, n)
    Ares = np.take_along_axis(Ap[pidx], dia, axis=2)
    dp1_boot = (Ares.mean(2) - Bres.mean(2)).mean(1)                          # Δpass@1 per replicate
    dp10_boot = (Ares.any(2).astype(np.float64) - Bres.any(2)).mean(1)        # Δpass@10 per replicate
    R["boot_lo"], R["boot_hi"] = float(np.percentile(dp1_boot, 2.5)), float(np.percentile(dp1_boot, 97.5))
    R["dp10_lo"], R["dp10_hi"] = float(np.percentile(dp10_boot, 2.5)), float(np.percentile(dp10_boot, 97.5))

    # B. strata (+ D per-stratum pass@10) + transition matrix
    d_p10 = k_p10 - b_p10
    b_str = np.array([stratum(x) for x in b_p1])
    k_str = np.array([stratum(x) for x in k_p1])
    R["strata"] = {}
    for s in STRATA:
        msk = b_str == s
        R["strata"][s] = {
            "n": int(msk.sum()),
            "mean_dp1": float(d_p1[msk].mean()) if msk.any() else float("nan"),
            "mean_dp10": float(d_p10[msk].mean()) if msk.any() else float("nan"),
            "share_of_gain": float(d_p1[msk].sum() / d_p1[d_p1 > 0].sum()) if (d_p1 > 0).any() and msk.any() else 0.0,
        }
    # migration: baseline stratum -> k stratum counts
    R["transition"] = {sb: {sk: int(((b_str == sb) & (k_str == sk)).sum()) for sk in STRATA} for sb in STRATA}
    # C x B: within each baseline stratum, baseline truncation + how much of the stratum's
    # gain comes from loop-escape (improved problems whose baseline failures were truncation-dominated).
    R["strata_attr"] = {s: {"base_trunc": float(b_nt[b_str == s].mean()) if (b_str == s).any() else float("nan"),
                            "k_trunc": float(k_nt[b_str == s].mean()) if (b_str == s).any() else float("nan"),
                            "esc": 0.0, "rea": 0.0} for s in STRATA}
    for j, i in enumerate(ids):
        if d_p1[j] <= 0:
            continue
        s = str(b_str[j])
        fails = sum(1 for p in base[i]["pass"] if not p)
        if fails == 0:
            R["strata_attr"][s]["rea"] += float(d_p1[j])
            continue
        trunc_fail = sum(1 for p, tr in zip(base[i]["pass"], base[i]["trunc"]) if (not p) and tr)
        key = "esc" if trunc_fail / fails >= 0.5 else "rea"
        R["strata_attr"][s][key] += float(d_p1[j])

    # C. loop-escape attribution
    R["spearman_dp1_vs_base_trunc"] = float(ss.spearmanr(b_nt, d_p1).statistic)
    loopy = b_nt >= 0.3
    R["loopy_n"] = int(loopy.sum())
    R["loopy_mean_dp1"] = float(d_p1[loopy].mean()) if loopy.any() else float("nan")
    R["clean_mean_dp1"] = float(d_p1[~loopy].mean()) if (~loopy).any() else float("nan")
    tot_gain = float(d_p1[d_p1 > 0].sum())
    R["loopy_share_of_gain"] = float(d_p1[loopy & (d_p1 > 0)].sum() / tot_gain) if tot_gain else 0.0
    # attribute improved problems by baseline failure mode (of FAILING baseline samples)
    esc_mass = rea_mass = 0.0
    for j, i in enumerate(ids):
        if d_p1[j] <= 0:
            continue
        fails = [not p for p in base[i]["pass"]]
        n_fail = sum(fails)
        if n_fail == 0:
            rea_mass += d_p1[j]           # was already all-correct, gain impossible-ish; bucket as non-loop
            continue
        trunc_fail = sum(1 for p, tr in zip(base[i]["pass"], base[i]["trunc"]) if (not p) and tr)
        if trunc_fail / n_fail >= 0.5:
            esc_mass += d_p1[j]
        else:
            rea_mass += d_p1[j]
    tg = esc_mass + rea_mass
    R["gain_from_loop_escape"] = float(esc_mass / tg) if tg else 0.0
    R["gain_from_reasoning"] = float(rea_mass / tg) if tg else 0.0

    # E. newly-solvable vs partial->more
    newly = d_p1[(b_p1 == 0) & (k_p1 > 0)].sum()
    partial = d_p1[(b_p1 > 0)].sum()
    R["gain_newly_solvable"] = float(newly / tot_gain) if tot_gain else 0.0
    R["gain_partial_more"] = float(partial / tot_gain) if tot_gain else 0.0

    # baseline overall failure-mode profile (context)
    all_fail = [(p, tr) for i in ids for p, tr in zip(base[i]["pass"], base[i]["trunc"])]
    nf = sum(1 for p, _ in all_fail if not p)
    R["base_fail_trunc_pct"] = float(100 * sum(1 for p, tr in all_fail if (not p) and tr) / nf) if nf else 0.0
    return R


def kval(fname: str) -> float:
    m = re.search(r"_k([0-9.]+)_", fname)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="qwen")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    cfg = MODELS[args.model]

    base = load_config(cfg["base_dir"], cfg["base_file"])
    if base is None:
        raise SystemExit(f"baseline missing: {cfg['base_dir']}/{cfg['base_file']}")
    _bf = [(p, tr) for i in base for p, tr in zip(base[i]["pass"], base[i]["trunc"])]
    _nf = sum(1 for p, _ in _bf if not p)
    base_fail_trunc_pct = 100 * sum(1 for p, tr in _bf if (not p) and tr) / _nf if _nf else 0.0
    base_p1 = float(np.mean([base[i]["nc"] / base[i]["n"] for i in base]))

    arms, pending = [], []
    for j in sorted(glob.glob(f"{cfg['renyi_dir']}/pless_renyi_think_t1.0_k*_t1.0.jsonl"),
                    key=lambda p: -kval(os.path.basename(p))):
        b = os.path.basename(j)[:-6]
        cfgd = load_config(cfg["renyi_dir"], b)
        (arms.append((kval(b), b, cfgd)) if cfgd else pending.append(f"k{kval(b)}"))

    L = [f"# pass@k ablation — G_k vs p-less (α=2) baseline, {cfg['name']} (ATCODER-interview, n=10)\n",
         f"Baseline α=2 (=G_2): `{cfg['base_dir']}/{cfg['base_file']}.jsonl` "
         f"(pass@1={base_p1:.3f}, "
         f"of failing samples {base_fail_trunc_pct:.1f}% are truncated loops).",
         "Problem-level paired design (same tasks; samples are independent draws across configs, so "
         "individual-sample fates are NOT tracked — only per-problem pass rates).\n"]
    if pending:
        L.append(f"**Pending arms (no metrics yet):** {', '.join(pending)}.\n")

    # summary table across k
    L += ["## Summary across k\n",
          "Significance is read from the **two-level bootstrap 95% CIs** (resample the 252 problems AND the "
          "10 within-problem draws, base/arm independently) — an interval excluding 0 is a significant shift, "
          "and the two-level resampling accounts for within-problem sampling noise. **cov-McNemar p** separately "
          "tests the *coverage-status* change (new-solve vs lost-solve counts), i.e. whether pass@10 membership "
          "shifts. **loop-escape (esc%)** is the coarse problem-level heuristic; the rigorous upper bound is "
          "Δtrunc in the C×B section. No multiple-comparisons correction across the 6 k arms; arms are unpaired, "
          "so differences *between* k arms are not significance-tested.\n",
          "| k | pass@1 (Δ) | pass@10 (Δ) | win / lose / net | new-solve / lost-solve | cov-McNemar p | Δpass@1 95% CI | Δpass@10 95% CI | loop-escape (esc%) |",
          "|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for k, b, arm in arms:
        R = analyse(base, arm, boot=args.boot)
        net = R["improved"] - R["deteriorated"]
        L.append(f"| {k} | {R['arm_p1']:.3f} ({R['mean_dp1']:+.3f}) | {R['arm_p10']:.3f} ({R['mean_dp10']:+.3f}) | "
                 f"{R['improved']}/{R['deteriorated']}/{net:+d} | {R['solve_gained']}/{R['solve_lost']} | "
                 f"{R['mcnemar_p']:.2g} | [{R['boot_lo']:+.3f},{R['boot_hi']:+.3f}] | [{R['dp10_lo']:+.3f},{R['dp10_hi']:+.3f}] | "
                 f"{R['gain_from_loop_escape']*100:.0f}% |")
        detail.append((k, R))

    # B + D: strata with per-stratum pass@1 AND pass@10
    n_by = {s: detail[0][1]["strata"][s]["n"] for s in STRATA} if detail else {}
    L.append("\n## B+D. Difficulty strata — Δpass@1 / Δpass@10 (net Δ contribution)\n")
    L.append("Buckets fixed by baseline pass@1; n constant across k. Cell = mean Δpass@1 / mean Δpass@10 (contrib%). "
             "**contrib%** = that stratum's *net* Δpass@1 as a fraction of the *gross winner* gain "
             "(Σ of positive per-problem Δ) — so a net-losing stratum shows a **negative** contrib%, and the "
             "columns sum to <100% by the total loss fraction (not an error). "
             "Δpass@1 ≫ Δpass@10 within a bucket ⇒ reliability (fewer auto-fails), not new coverage.")
    L.append("| k | " + " | ".join(f"{s} n={n_by.get(s, '?')}" for s in STRATA) + " |")
    L.append("|---|" + "|".join("---" for _ in STRATA) + "|")
    for k, R in detail:
        cells = []
        for s in STRATA:
            st = R["strata"][s]
            cells.append(f"{st['mean_dp1']:+.3f} / {st['mean_dp10']:+.3f} ({st['share_of_gain']*100:.0f}%)")
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    # B: migration matrix
    L.append("\n## B. Migration matrix (baseline stratum → k stratum, problem counts)\n")
    L.append("Rows = where a problem sat at α=2; columns = where it sits at that k. Mass above the diagonal "
             "(toward *easy*) is upward migration; below is regression. The **dead→(non-dead)** cells are exactly "
             "the *newly-solvable* problems of decomposition E; **(non-dead)→dead** are solves lost.")
    for k, R in detail:
        L.append(f"\n**k = {k}**\n")
        L.append("| baseline ↓ / k → | " + " | ".join(STRATA) + " |")
        L.append("|---|" + "|".join("---" for _ in STRATA) + "|")
        for sb in STRATA:
            L.append(f"| **{sb}** | " + " | ".join(str(R["transition"][sb][sk]) for sk in STRATA) + " |")

    L.append("\n## C+E. Attribution & decomposition\n")
    L.append("| k | Δpass@1 | loopy(n≥0.3 trunc) Δ | clean Δ | ρ(Δp1, base-trunc) | gain: loop-escape / reasoning | gain: newly-solvable / partial |")
    L.append("|---|---|---|---|---|---|---|")
    for k, R in detail:
        L.append(f"| {k} | {R['mean_dp1']:+.3f} | {R['loopy_mean_dp1']:+.3f} (n={R['loopy_n']}) | {R['clean_mean_dp1']:+.3f} | "
                 f"{R['spearman_dp1_vs_base_trunc']:+.2f} | {R['gain_from_loop_escape']*100:.0f}% / {R['gain_from_reasoning']*100:.0f}% | "
                 f"{R['gain_newly_solvable']*100:.0f}% / {R['gain_partial_more']*100:.0f}% |")

    # C x B: is each stratum's gain loop-driven?
    L.append("\n## C×B. Is each stratum's gain due to loops? (baseline truncation + loop-escape share of gain)\n")
    L.append("Per baseline stratum, cell = **Δtrunc** (α=2 trunc% → k trunc%; how much looping actually fell) · "
             "**esc%** (share of that stratum's positive Δpass@1 from truncation-dominated-failure problems). "
             "**Δtrunc bounds the loop contribution: loop-escape can lift pass@1 by at most |Δtrunc|** — under the "
             "premise that a truncated sample always fails (true: an unclosed thinking phase yields no gradable "
             "answer) and given truncation only *falls* with looser k here (Δtrunc ≤ 0 in every stratum, so the "
             "aggregate rate change equals the max rescuable mass). If truncation barely fell in a stratum, its gain "
             "is NOT loops. esc% is a coarser problem-level heuristic (rounds a whole problem's gain to loop-escape "
             "when ≥50% of its α=2 failures were truncations) and tends to *over*-credit loops.")
    L.append("| k | " + " | ".join(f"{s}" for s in STRATA) + " |")
    L.append("|---|" + "|".join("---" for _ in STRATA) + "|")
    for k, R in detail:
        cells = []
        for s in STRATA:
            a = R["strata_attr"][s]
            tot = a["esc"] + a["rea"]
            esc = f"{a['esc']/tot*100:.0f}%" if tot > 1e-9 else "—"
            cells.append(f"{a['base_trunc']*100:.0f}→{a['k_trunc']*100:.0f}% (Δ{-(a['base_trunc']-a['k_trunc'])*100:+.0f}) · esc {esc}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    L += ["\n## How to read (A–E)\n",
          "- **A (summary)**: win/lose/net shows how many problems improved vs regressed; significance of the "
          "pass@1 and pass@10 shifts is read from the **two-level bootstrap CIs** (excluding 0), which resample "
          "problems and the 10 draws; cov-McNemar separately tests the coverage-status (new vs lost solve) change.",
          "- **B (strata + migration)**: if gain concentrates in *dead/hard/mid* → loop-escape/coverage; in *easy* → "
          "Matthew (H4). The migration matrix shows which buckets move up (mid→easy) vs stay (dead→dead).",
          "- **C (loop-escape share / ρ(Δp1, base-trunc))**: if Δpass@1 tracks how much a problem truncated at α=2, "
          "and most gain is from truncation-dominated problems, the win is loops escaping the token cap, not new reasoning (H1).",
          "- **D (Δpass@1 ≫ Δpass@10, incl. per-stratum)**: reliability (fewer auto-fail draws), not coverage (new solutions) (H2).",
          "- **E (newly-solvable vs partial→more)**: newly-solvable ⇔ dead→non-dead migration; partial→more ⇔ within/among "
          "the non-dead buckets. Dominant partial→more with tiny newly-solvable ⇒ consolidating borderline problems, not unlocking new ones.",
          "\nGenerated by `scripts/pass_at_k_ablation.py`. Every number recomputed live from metrics JSON + jsonl."]

    outdir = f"{cfg['renyi_dir']}/analysis"
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/pass_at_k_ablation_{args.model}.md"
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:3]))
    print(f"\nScored arms: {[k for k, _, _ in arms]}   Pending: {pending or 'none'}")
    print(f"Report -> {outp}")


if __name__ == "__main__":
    main()
