"""Pilot 1 — Phase 3b-1: FAITHFUL probe+CUSUM detector (arXiv:2601.05693, Section 4).

Reimplements the paper's deployed early-loop detector and its exact metrics on our
naturalistic ATCODER loops. See docs/pilot1_circular_reasoning_replication.md for the
locked design + deviation decisions. Differences from the exploratory Phase 3
(pilot1_analyze.py):
  * faithful layer = LAST layer (36); 6/16/24 reported as OUR extension.
  * CUSUM grid-searched over alpha in [1,2], p in {3,4,5} (paper Appendix D.2),
    reported as a full grid (no single "best" cherry-pick -> no selection bias).
  * paper's exact metrics: EDR, FPR, ASE (sentence earliness), ATE (token earliness),
    per group (terminal / transient), Eqs 3-6.

Deviations (documented): onset = our validated n-gram (n=30/k=6/w=1200), NOT the
paper's 3-rep textual onset -> EDR/ASE/ATE are NOT numerically comparable to their
Table 3, only directionally. Probe = leave-trace-out StratifiedGroupKFold CV (rigor).

Usage:
  uv run python scripts/pilot1_replicate.py \
      --manifest results/pilot1_hidden/manifest.jsonl --out-dir results/pilot1_hidden \
      [--layers 36] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0]   # paper: alpha in [1.0, 2.0]
PS = [3, 4, 5]                          # paper: p in {3,4,5,...}
N_SPLITS = 5
LOOP_CLSES = ("looping_truncated", "looping_completed")


def trace_key(rec: dict) -> str:
    return f"{rec['cls']}__{rec['task_id']}__{rec['sample_idx']}"


def cusum_alert(scores: np.ndarray, r: float, h: float, p: int) -> int | None:
    """First sentence index where S_i=max(0,S_{i-1}+(x_i-r)) stays > h for p consecutive."""
    S = 0.0; run = 0
    for i, x in enumerate(scores):
        xi = r if np.isnan(x) else float(x)
        S = max(0.0, S + (xi - r))
        if S > h:
            run += 1
            if run >= p:
                return i - p + 1
        else:
            run = 0
    return None


def load_traces(manifest: Path, vec_dir: Path):
    recs = {trace_key(r): r for r in (json.loads(l) for l in open(manifest))}
    out = []
    for f in sorted(vec_dir.glob("*.npz")):
        rec = recs.get(f.stem)
        if rec is None:
            continue
        z = np.load(f)
        vecs = z["vecs"].astype(np.float32)
        layers = [int(x) for x in z["layers"]]
        assert vecs.shape[0] == len(rec["sentences"]), f"{f.stem}: n_sent mismatch"
        out.append({"rec": rec, "vecs": vecs, "layers": layers})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="default: faithful last layer first, then the rest as extension")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    traces = load_traces(args.manifest, args.out_dir / "vectors")
    if not traces:
        print("no vectors found"); return
    all_layers = traces[0]["layers"]
    faithful = max(all_layers)                       # last layer = paper's choice
    layers = args.layers or ([faithful] + [L for L in all_layers if L != faithful])
    from collections import Counter
    print(f"traces: {len(traces)}  by cls: {dict(Counter(t['rec']['cls'] for t in traces))}")
    print(f"faithful layer (paper) = {faithful}; reporting layers {layers}")

    # per-sentence table
    rows = []
    for ti, t in enumerate(traces):
        for s in t["rec"]["sentences"]:
            rows.append({"ti": ti, "y": 1 if s["label"] == "loop" else 0})
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["ti"] for r in rows])
    counts = [t["vecs"].shape[0] for t in traces]
    offs = np.cumsum([0] + counts)
    eff_splits = max(2, min(N_SPLITS, len(set(groups.tolist()))))

    # clean split for CUSUM calibration (r,h on calib; FPR on test)
    clean_tis = [ti for ti, t in enumerate(traces) if t["rec"]["cls"] == "clean"]
    rng.shuffle(clean_tis)
    half = len(clean_tis) // 2
    calib_clean, test_clean = set(clean_tis[:half]), set(clean_tis[half:])
    assert calib_clean and test_clean, "need >=2 clean traces to calibrate + test"

    results = {"faithful_layer": faithful, "by_layer": {}, "alphas": ALPHAS, "ps": PS,
               "n_calib_clean": len(calib_clean), "n_test_clean": len(test_clean)}

    for L in layers:
        li = all_layers.index(L)
        X = np.concatenate([t["vecs"][:, li, :] for t in traces], axis=0).astype(np.float32)
        # held-out probe score per sentence (leave-trace-out, per-layer std in-pipeline)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedGroupKFold(n_splits=eff_splits)
        proba = cross_val_predict(pipe, X, y, groups=groups, cv=cv,
                                  method="predict_proba")[:, 1]
        seq_by_trace = [proba[offs[ti]:offs[ti + 1]] for ti in range(len(traces))]

        # reference r = mean probe score over calib-clean sentences
        calib_rows = np.concatenate([seq_by_trace[ti] for ti in calib_clean])
        r = float(np.nanmean(calib_rows))

        grid = []
        for alpha in ALPHAS:
            # h depends on alpha and the calib-clean CUSUM max (independent of p)
            maxS = 0.0
            for ti in calib_clean:
                S = 0.0
                for x in seq_by_trace[ti]:
                    xi = r if np.isnan(x) else float(x)
                    S = max(0.0, S + (xi - r)); maxS = max(maxS, S)
            h = float("inf") if maxS <= 0 else alpha * maxS
            for p in PS:
                # FPR on held-out clean
                fp = sum(1 for ti in test_clean
                         if cusum_alert(seq_by_trace[ti], r, h, p) is not None)
                fpr = fp / len(test_clean)
                cell = {"alpha": alpha, "p": p, "fpr": fpr, "h": h, "r": r}
                # per loop group: EDR / ASE / ATE (Eqs 3-6)
                for cls in LOOP_CLSES:
                    tis = [ti for ti, t in enumerate(traces) if t["rec"]["cls"] == cls]
                    n_loop = len(tis)
                    n_early = 0; ase = []; ate = []
                    for ti in tis:
                        rec = traces[ti]["rec"]
                        a = cusum_alert(seq_by_trace[ti], r, h, p)
                        if a is None:
                            continue
                        t_alert = rec["sentences"][a]["tok_start"]
                        if t_alert < rec["onset_token"]:        # early = fired before onset
                            n_early += 1
                            ase.append(rec["onset_sentence"] - a)
                            ate.append(rec["onset_token"] - t_alert)
                    cell[cls] = {
                        "EDR": n_early / n_loop if n_loop else None,
                        "ASE": float(np.mean(ase)) if ase else None,
                        "ATE": float(np.mean(ate)) if ate else None,
                        "n_loop": n_loop, "n_early": n_early,
                    }
                grid.append(cell)
        results["by_layer"][L] = {"r": r, "grid": grid}

        # ---- print: operating points at FPR <= 0.35 (paper's range), best EDR ----
        print(f"\n=== layer {L}{'  (FAITHFUL last layer)' if L == faithful else '  (extension)'} ===")
        for cls in LOOP_CLSES:
            ok = [c for c in grid if c["fpr"] <= 0.35 and c[cls]["EDR"] is not None]
            best = max(ok, key=lambda c: c[cls]["EDR"], default=None)
            if best is None:
                print(f"  {cls:>18}: no config with FPR<=0.35"); continue
            b = best[cls]
            print(f"  {cls:>18}: best EDR@FPR<=0.35 = {b['EDR']:.2f} "
                  f"(alpha={best['alpha']}, p={best['p']}, FPR={best['fpr']:.2f})  "
                  f"ASE={b['ASE']} sent  ATE={b['ATE']} tok  "
                  f"[early {b['n_early']}/{b['n_loop']}]")

    out_path = args.out_dir / "pilot1_replicate_results.json"
    out_path.write_text(json.dumps(results, indent=2,
                                   default=lambda o: float(o) if isinstance(o, np.floating)
                                   else int(o) if isinstance(o, np.integer) else str(o)))
    print(f"\nfull grid -> {out_path}")
    print("NOTE: onset = n-gram (not the paper's 3-rep) -> EDR/ASE/ATE are directional, "
          "NOT numerically comparable to their Table 3 (see the design doc).")


if __name__ == "__main__":
    main()
