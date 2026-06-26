"""Pilot 1 — Phase 3: does the hidden-state signal separate / predict loops? (CPU)

Consumes the manifest (Phase 1) + per-trace sentence vectors (Phase 2). Answers,
PER LAYER and PER GROUP (terminal / transient / clean — never pooled):

  SEPARATION (Gate 1: does any signal exist?)
    - probe AUC: logistic regression on per-sentence hidden vectors, with PER-LAYER
      standardization fit inside a leave-trace-out (GroupKFold) CV — held-out P(loop)
      vs the normal/loop label. This is the supervised UPPER BOUND.
    - cosine AUC: unsupervised max-cosine-to-prior-sentence (the deployable signal).
      Reported all-loop-vs-normal, terminal-vs-clean, transient-vs-clean.

  EARLINESS (Gate 3: does it fire before the n-gram onset?)
    - CUSUM over each trace's per-sentence score; reference r and threshold h are
      calibrated on HELD-OUT clean traces (half), so FPR is measured on the other
      half (not circular). Lead-time = onset_token - fire_token, per loop group.

  SEMANTIC-vs-SURFACE (Gate 2): compare the early-control layer (6) to the middle
  layers — if early separates as well, the signal is likely surface repetition.

Decision gates printed at the end. NOTE the pre/post-norm asymmetry: layer 36 is
post-final-norm, 6/16/24 are pre-norm — the per-layer StandardScaler (probe) and
the scale-invariance of cosine both neutralize this WITHIN a layer; cross-layer we
only compare standardized AUCs, never raw vectors.

Usage:
  uv run python scripts/pilot1_analyze.py \
      --manifest results/pilot1_hidden/manifest.jsonl --out-dir results/pilot1_hidden
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GAP = 3            # cosine: compare to priors >= GAP positions back (skips self + GAP-1 nearest; avoids trivial adjacency)
CUSUM_ALPHA = 1.5  # h = alpha * max-CUSUM-on-calibration-clean
CUSUM_P = 3        # consecutive steps above h required to fire (paper uses p>=3)
N_SPLITS = 5       # grouped CV folds


def trace_key(rec: dict) -> str:
    return f"{rec['cls']}__{rec['task_id']}__{rec['sample_idx']}"


def max_cos_to_prior(vec_layer: np.ndarray, gap: int = GAP) -> np.ndarray:
    x = vec_layer.astype(np.float32)
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    xn = x / nrm
    out = np.full(len(xn), np.nan, dtype=np.float32)
    for i in range(gap, len(xn)):
        out[i] = float((xn[: i - gap + 1] @ xn[i]).max())
    return out


def cusum_fire(scores: np.ndarray, r: float, h: float, p: int = CUSUM_P) -> int | None:
    """First index where the CUSUM stat S_i=max(0,S_{i-1}+(x_i-r)) stays > h for p
    consecutive steps. NaNs (early cosine) treated as no-contribution (x_i=r)."""
    S = 0.0; run = 0
    for i, x in enumerate(scores):
        xi = r if np.isnan(x) else float(x)      # np.isnan handles np.float32 (isinstance(float) would not)
        S = max(0.0, S + (xi - r))
        if S > h:
            run += 1
            if run >= p:
                return i - p + 1
        else:
            run = 0
    return None


def load_traces(manifest: Path, vec_dir: Path, want_layers):
    recs = {trace_key(r): r for r in (json.loads(l) for l in open(manifest))}
    out = []
    for f in sorted(vec_dir.glob("*.npz")):
        rec = recs.get(f.stem)
        if rec is None:
            continue
        z = np.load(f)
        vecs = z["vecs"].astype(np.float32)          # (n_sent, n_layers, dim)
        layers = list(int(x) for x in z["layers"])
        assert vecs.shape[0] == len(rec["sentences"]), f"{f.stem}: n_sent mismatch"
        out.append({"rec": rec, "vecs": vecs, "layers": layers})
    return out, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    traces, _ = load_traces(args.manifest, args.out_dir / "vectors", None)
    if not traces:
        print("no vectors found — run Phase 2 first"); return
    layers = traces[0]["layers"]
    from collections import Counter
    print(f"traces with vectors: {len(traces)}  "
          f"by cls: {dict(Counter(t['rec']['cls'] for t in traces))}  layers={layers}")

    # ---- assemble per-sentence table (layer-agnostic columns) ----
    # row per sentence: trace_idx, cls, label(1=loop), is_clean, tok_start, and a
    # pointer to (trace, sent) so we can fetch per-layer vectors/cosine.
    rows = []
    for ti, t in enumerate(traces):
        rec = t["rec"]; cls = rec["cls"]
        for si, s in enumerate(rec["sentences"]):
            rows.append({"ti": ti, "si": si, "cls": cls,
                         "y": 1 if s["label"] == "loop" else 0})
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["ti"] for r in rows])
    cls_arr = np.array([r["cls"] for r in rows])
    print(f"sentences: {len(rows)}  loop={int(y.sum())}  normal={int((1-y).sum())}")

    # contiguous row offsets per trace (rows are built trace-by-trace, in order)
    counts = [t["vecs"].shape[0] for t in traces]
    offs = np.cumsum([0] + counts)        # len == n_traces+1
    n_groups = len(set(groups.tolist()))
    eff_splits = max(2, min(N_SPLITS, n_groups))

    # clean-trace split for CUSUM calibration (calib vs test), by TRACE
    clean_tis = [ti for ti, t in enumerate(traces) if t["rec"]["cls"] == "clean"]
    rng.shuffle(clean_tis)
    half = len(clean_tis) // 2
    calib_clean = set(clean_tis[:half]); test_clean = set(clean_tis[half:])

    def auc_framings(score):
        """score: per-row signal (len == len(rows)); returns AUC dicts. Drops NaN rows."""
        s = np.asarray(score, dtype=float)
        ok = ~np.isnan(s)
        def _auc(mask):
            m = mask & ok
            yy = y[m]
            if yy.size == 0 or yy.min() == yy.max():  # empty, or only one class present
                return None
            return float(roc_auc_score(yy, s[m]))
        loop = y == 1
        term = (cls_arr == "looping_truncated")
        trans = (cls_arr == "looping_completed")
        clean = (cls_arr == "clean")
        return {
            "all_loop_vs_normal": _auc(np.ones(len(s), bool)),
            "terminal_vs_clean": _auc((loop & term) | clean),
            "transient_vs_clean": _auc((loop & trans) | clean),
            # within-loop-trace contrast: post-onset vs pre-onset of the SAME traces.
            # Controls for the trace-identity confound that can inflate the
            # *_vs_clean framings (clean traces may differ from loop traces for
            # reasons unrelated to onset).
            "pre_vs_post_within_loop": _auc(term | trans),
        }

    results = {"layers": layers, "n_traces": len(traces), "by_layer": {}}

    for li, L in enumerate(layers):
        # feature matrix for this layer
        X = np.concatenate([t["vecs"][:, li, :] for t in traces], axis=0).astype(np.float32)
        # ---- (C) supervised probe: per-layer std fit inside grouped CV ----
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
        # StratifiedGroupKFold keeps each trace wholly in one fold (no leakage)
        # AND balances loop/normal across folds (avoids a single-class train fold,
        # which crashes LogisticRegression).
        cv = StratifiedGroupKFold(n_splits=eff_splits)
        proba = cross_val_predict(pipe, X, y, groups=groups, cv=cv,
                                  method="predict_proba")[:, 1]
        probe_auc = auc_framings(proba)

        # ---- (B) unsupervised cosine ----
        cos = np.full(len(rows), np.nan, dtype=float)
        off = 0
        cos_seq_by_trace = []
        for t in traces:
            n = t["vecs"].shape[0]
            c = max_cos_to_prior(t["vecs"][:, li, :])
            cos[off:off + n] = c
            cos_seq_by_trace.append(c)
            off += n
        cos_auc = auc_framings(cos)

        # ---- earliness via CUSUM, per signal ----
        def leadtime_and_fpr(score_per_row, seq_by_trace):
            # reference r = mean score over calibration-clean sentences
            calib_mask = np.array([rows[i]["ti"] in calib_clean for i in range(len(rows))])
            rvals = np.asarray(score_per_row)[calib_mask]
            rvals = rvals[~np.isnan(rvals)]
            r = float(rvals.mean()) if rvals.size else 0.0
            # h = alpha * max CUSUM stat reached on calibration-clean traces
            maxS = 0.0
            for ti in calib_clean:
                seq = seq_by_trace[ti]; S = 0.0
                for x in seq:
                    xi = r if np.isnan(x) else float(x)   # float() -> keep python float (json-safe, matches cusum_fire)
                    S = max(0.0, S + (xi - r)); maxS = max(maxS, S)
            h = CUSUM_ALPHA * maxS
            if maxS <= 0:           # cannot calibrate (no/degenerate clean signal)
                h = float("inf")    # -> never fire, rather than h=0 firing everything
            # FPR on held-out clean (None if no held-out clean to measure on)
            fp = sum(1 for ti in test_clean
                     if cusum_fire(seq_by_trace[ti], r, h) is not None)
            fpr = (fp / len(test_clean)) if test_clean else None
            # lead-time on loop traces, per group
            lead = {"looping_truncated": [], "looping_completed": []}
            fired = {"looping_truncated": 0, "looping_completed": 0}
            ntot = {"looping_truncated": 0, "looping_completed": 0}
            for ti, t in enumerate(traces):
                cls = t["rec"]["cls"]
                if cls not in lead:
                    continue
                ntot[cls] += 1
                fire = cusum_fire(seq_by_trace[ti], r, h)
                if fire is None:
                    continue
                fired[cls] += 1
                fire_tok = t["rec"]["sentences"][fire]["tok_start"]
                lead[cls].append(t["rec"]["onset_token"] - fire_tok)
            summ = {"r": r, "h": h, "fpr_heldout_clean": fpr}
            for cls in lead:
                lt = lead[cls]
                summ[cls] = {
                    "n": ntot[cls], "fired": fired[cls],
                    "lead_median": float(np.median(lt)) if lt else None,
                    "pct_fired_before_onset": (
                        float(100 * sum(1 for x in lt if x > 0) / len(lt)) if lt else None),
                }
            return summ

        probe_seq_by_trace = [proba[offs[ti]:offs[ti + 1]] for ti in range(len(traces))]
        probe_lead = leadtime_and_fpr(proba, probe_seq_by_trace)
        cos_lead = leadtime_and_fpr(cos, cos_seq_by_trace)

        results["by_layer"][L] = {
            "probe_auc": probe_auc, "cosine_auc": cos_auc,
            "probe_cusum": probe_lead, "cosine_cusum": cos_lead,
        }
        def _a(v):  # AUC formatter (None-safe)
            return f"{v:.3f}" if v is not None else "  -  "
        def _f(v):  # FPR formatter (None-safe)
            return f"{v:.2f}" if v is not None else "n/a"
        print(f"\n=== layer {L} ===")
        print(f"  probe  AUC  all={_a(probe_auc['all_loop_vs_normal'])}  "
              f"term_v_clean={_a(probe_auc['terminal_vs_clean'])}  "
              f"trans_v_clean={_a(probe_auc['transient_vs_clean'])}  "
              f"pre_v_post={_a(probe_auc['pre_vs_post_within_loop'])}")
        print(f"  cosine AUC  all={_a(cos_auc['all_loop_vs_normal'])}  "
              f"term_v_clean={_a(cos_auc['terminal_vs_clean'])}  "
              f"trans_v_clean={_a(cos_auc['transient_vs_clean'])}  "
              f"pre_v_post={_a(cos_auc['pre_vs_post_within_loop'])}")
        print(f"  probe  CUSUM  FPR={_f(probe_lead['fpr_heldout_clean'])}  "
              f"term lead={probe_lead['looping_truncated']['lead_median']} "
              f"(fired {probe_lead['looping_truncated']['fired']}/{probe_lead['looping_truncated']['n']})  "
              f"trans lead={probe_lead['looping_completed']['lead_median']} "
              f"(fired {probe_lead['looping_completed']['fired']}/{probe_lead['looping_completed']['n']})")
        print(f"  cosine CUSUM  FPR={_f(cos_lead['fpr_heldout_clean'])}  "
              f"term lead={cos_lead['looping_truncated']['lead_median']} "
              f"trans lead={cos_lead['looping_completed']['lead_median']}")

    def _json_default(o):  # coerce any stray numpy scalars
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(f"not serializable: {type(o)}")
    out_path = args.out_dir / "pilot1_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default))

    # ---- decision gates ----
    best = max(layers, key=lambda L: (results["by_layer"][L]["probe_auc"]["all_loop_vs_normal"] or 0))
    bestauc = results["by_layer"][best]["probe_auc"]["all_loop_vs_normal"] or 0
    early_auc = results["by_layer"][layers[0]]["probe_auc"]["all_loop_vs_normal"] or 0
    print("\n" + "=" * 64)
    print("DECISION GATES")
    print("=" * 64)
    print(f"  Gate1 (signal exists):   best probe AUC = {bestauc:.3f} @ layer {best}  "
          f"-> {'PASS' if bestauc > 0.7 else 'WEAK/STOP' if bestauc < 0.6 else 'MARGINAL'}")
    print(f"  Gate2 (semantic>surface): early(L{layers[0]}) AUC={early_auc:.3f} vs best={bestauc:.3f}  "
          f"-> {'semantic (mid wins)' if bestauc - early_auc > 0.03 else 'surface-like (early ties)'}")
    tl = results["by_layer"][best]["probe_cusum"]["looping_truncated"]["lead_median"]
    print(f"  Gate3 (early warning):   best-layer terminal lead-time median = {tl} tokens  "
          f"-> {'EARLY' if (tl or -1) > 0 else 'not early'}")
    print(f"\n  full results -> {out_path}")


if __name__ == "__main__":
    main()
