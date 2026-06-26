"""Pilot 1 — Phase 3b-2: cluster-trajectory PERIODICITY precursor (arXiv:2601.05693 Sec 3.1).

Probe-free test of the paper's mechanistic claim: "semantic circularity precedes explicit
textual repetition." If periodicity precedes the n-gram onset even where the linear probe
(3b-1) failed, the signal EXISTS in the geometry and the probe was a poor readout; if not,
it corroborates that the signal is genuinely weak on naturalistic loops.

Two operationalizations of periodicity (see docs/pilot1_circular_reasoning_replication.md
D-periodicity); the paper leaves the detection method unspecified, so both are OURS:

  Method 1 (faithful, keeps K-means K=200): label-MATCH autocorrelation.
      labels = KMeans(min(200,n)).fit_predict(layer-36 sentence vectors)
      tau(d) = mean_i [label_i == label_{i+d}];  strength = max_d (tau(d) - chance),
      chance = sum_k (n_k/N)^2  (prob two random positions share a label).
  Method 2 (K-means-free cross-check): mean-centred trajectory autocorrelation.
      c_i = vec_i - mean(vec);  rho(d) = mean_i<c_i,c_{i+d}> / mean_i<c_i,c_i>;
      strength = max_{d>=D_MIN} rho(d).

Reported per group (terminal/transient/clean):
  (i)  pre-onset strength vs clean  -> does the precursor exist BEFORE textual onset (AUC).
  (ii) semantic-onset lead vs n-gram onset (sliding-window periodicity, theta calibrated
       on held-out clean), per loop group + clean false-periodicity rate.

Usage:
  uv run python scripts/pilot1_periodicity.py \
      --manifest results/pilot1_hidden/manifest.jsonl --out-dir results/pilot1_hidden
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

K_CLUSTERS = 200
D_MIN = 3
WIN = 50          # sliding window (sentences) for onset detection
D_MAX_FULL = 200  # cap on lag for whole-region strength
P_CONSEC = 3
LOOP_CLSES = ("looping_truncated", "looping_completed")


def trace_key(rec: dict) -> str:
    return f"{rec['cls']}__{rec['task_id']}__{rec['sample_idx']}"


def label_match_strength(labels: np.ndarray, d_min=D_MIN, d_max=D_MAX_FULL) -> float:
    """max_d (tau(d) - chance), tau(d)=mean[label_i==label_{i+d}]."""
    n = len(labels)
    if n <= d_min:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    chance = float(((counts / n) ** 2).sum())
    best = 0.0
    for d in range(d_min, min(d_max, n - 1) + 1):
        tau = float(np.mean(labels[: n - d] == labels[d:]))
        best = max(best, tau - chance)
    return best


def centered_strength(vecs: np.ndarray, d_min=D_MIN, d_max=D_MAX_FULL) -> float:
    """max_{d} rho(d) on the mean-centred trajectory."""
    n = len(vecs)
    if n <= d_min:
        return 0.0
    c = vecs - vecs.mean(axis=0, keepdims=True)
    denom = float(np.mean(np.sum(c * c, axis=1)))
    if denom <= 0:
        return 0.0
    best = -1.0
    for d in range(d_min, min(d_max, n - 1) + 1):
        num = float(np.mean(np.sum(c[: n - d] * c[d:], axis=1)))
        best = max(best, num / denom)
    return best


def onset_via_sliding(metric_fn, seq, win=WIN, p=P_CONSEC, theta=None):
    """Return (metric_per_window_end, fire_sentence_index_or_None).
    metric_fn(window_slice)->float; seq is the per-sentence array (labels or vecs)."""
    n = len(seq)
    ends, mvals = [], []
    for e in range(win - 1, n):            # window covers [e-win+1 .. e]
        m = metric_fn(seq[e - win + 1: e + 1])
        ends.append(e); mvals.append(m)
    fire = None
    if theta is not None:
        run = 0
        for j, m in enumerate(mvals):
            if m > theta:
                run += 1
                if run >= p:
                    fire = ends[j - p + 1]; break
            else:
                run = 0
    return np.array(mvals), fire


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
    ap.add_argument("--layer", type=int, default=None, help="default: faithful last layer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    traces = load_traces(args.manifest, args.out_dir / "vectors")
    if not traces:
        print("no vectors found"); return
    all_layers = traces[0]["layers"]
    L = args.layer or max(all_layers)
    li = all_layers.index(L)
    from collections import Counter
    print(f"traces: {len(traces)}  by cls: {dict(Counter(t['rec']['cls'] for t in traces))}  layer={L}")

    # precompute per-trace: K-means labels, centred vecs, full-region + pre-onset strengths
    for t in traces:
        v = t["vecs"][:, li, :].astype(np.float32)
        n = v.shape[0]
        k = min(K_CLUSTERS, n)
        t["labels"] = KMeans(n_clusters=k, n_init=3, max_iter=100,
                             random_state=args.seed).fit_predict(v)
        t["vL"] = v
        t["m1_full"] = label_match_strength(t["labels"])
        t["m2_full"] = centered_strength(v)
        os = t["rec"]["onset_sentence"]
        if os is not None and os > D_MIN:
            t["m1_pre"] = label_match_strength(t["labels"][:os])
            t["m2_pre"] = centered_strength(v[:os])
        else:
            t["m1_pre"] = t["m2_pre"] = None

    cls_of = [t["rec"]["cls"] for t in traces]
    is_clean = np.array([c == "clean" for c in cls_of])

    # ---- (i) pre-onset strength vs clean: does the precursor exist BEFORE onset? ----
    def preonset_auc(key_pre, key_full):
        # positives = loop traces' PRE-onset strength; negatives = clean whole-trace strength
        pos = [t[key_pre] for t in traces if t["rec"]["cls"] in LOOP_CLSES and t[key_pre] is not None]
        neg = [t[key_full] for t in traces if t["rec"]["cls"] == "clean"]
        if not pos or not neg:
            return None
        yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        sv = np.r_[pos, neg]
        return float(roc_auc_score(yv, sv))

    auc1_pre = preonset_auc("m1_pre", "m1_full")
    auc2_pre = preonset_auc("m2_pre", "m2_full")

    # ---- (ii) sliding-window onset + lead, theta calibrated on held-out clean ----
    clean_tis = [ti for ti, t in enumerate(traces) if t["rec"]["cls"] == "clean"]
    rng.shuffle(clean_tis)
    half = len(clean_tis) // 2
    calib_clean, test_clean = set(clean_tis[:half]), set(clean_tis[half:])

    def win_metric_fn(method):
        if method == 1:
            return lambda w: label_match_strength(w, d_min=D_MIN, d_max=WIN // 2)
        return lambda w: centered_strength(w, d_min=D_MIN, d_max=WIN // 2)

    def run_method(method):
        seq_key = "labels" if method == 1 else "vL"
        mfn = win_metric_fn(method)
        # theta = max windowed metric over calib-clean traces (strict; clean rarely exceeds)
        theta = 0.0
        for ti in calib_clean:
            mvals, _ = onset_via_sliding(mfn, traces[ti][seq_key], theta=None)
            if len(mvals):
                theta = max(theta, float(mvals.max()))
        # FPR on held-out clean
        fp = 0
        for ti in test_clean:
            _, fire = onset_via_sliding(mfn, traces[ti][seq_key], theta=theta)
            fp += int(fire is not None)
        fpr = fp / len(test_clean) if test_clean else None
        out = {"theta": theta, "fpr_heldout_clean": fpr}
        for cls in LOOP_CLSES:
            tis = [ti for ti, t in enumerate(traces) if t["rec"]["cls"] == cls]
            leads, n_early = [], 0
            for ti in tis:
                rec = traces[ti]["rec"]
                _, fire = onset_via_sliding(mfn, traces[ti][seq_key], theta=theta)
                if fire is None:
                    continue
                t_alert = rec["sentences"][fire]["tok_start"]
                lead = rec["onset_token"] - t_alert
                leads.append(lead)
                if lead > 0:
                    n_early += 1
            out[cls] = {
                "n_loop": len(tis), "fired": len(leads), "fired_before_onset": n_early,
                "lead_median": float(np.median(leads)) if leads else None,
            }
        return out

    m1 = run_method(1)
    m2 = run_method(2)

    results = {"layer": L, "n_traces": len(traces),
               "preonset_auc": {"method1_labelmatch": auc1_pre, "method2_centered": auc2_pre},
               "onset_lead": {"method1_labelmatch": m1, "method2_centered": m2}}
    out_path = args.out_dir / "pilot1_periodicity_results.json"
    out_path.write_text(json.dumps(results, indent=2,
                        default=lambda o: float(o) if isinstance(o, np.floating)
                        else int(o) if isinstance(o, np.integer) else str(o)))

    print(f"\n(i) PRE-ONSET periodicity strength, loop-pre-onset vs clean (AUC>0.5 = precursor exists):")
    print(f"    method1 (label-match): {auc1_pre}")
    print(f"    method2 (centered):    {auc2_pre}")
    for name, m in (("method1 label-match", m1), ("method2 centered", m2)):
        print(f"\n(ii) {name}: semantic-onset lead vs n-gram onset  (theta on held-out clean, FPR={m['fpr_heldout_clean']})")
        for cls in LOOP_CLSES:
            c = m[cls]
            print(f"    {cls:>18}: fired {c['fired']}/{c['n_loop']}  "
                  f"before-onset {c['fired_before_onset']}  lead_median={c['lead_median']}")
    print(f"\nfull results -> {out_path}")
    print("NOTE: periodicity-detection method is OUR operationalization (paper underspecifies); "
          "onset=n-gram so leads are directional, not comparable to their ASE/ATE.")


if __name__ == "__main__":
    main()
