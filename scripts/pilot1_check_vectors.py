"""Pilot 1 — verify extracted hidden-state vectors (CPU; runs on whatever vectors exist).

Works on a partial run (e.g. the --limit 2 smoke) — only checks traces that have a
.npz in <out-dir>/vectors/. Verifies:
  1. shape + ALIGNMENT with the manifest (n_sent matches the manifest's sentence
     count for that trace; n_layers, hidden_dim as expected),
  2. finiteness,
  3. per-layer L2-norm scale (expect layer 36 = post-final-norm to differ from the
     pre-norm layers 6/16/24 — confirms layer indexing & motivates per-layer std),
  4. a cosine self-similarity PEEK: per layer, median max-cosine-to-prior-sentence
     in the PRE-onset region vs the LOOP region (loop traces) and overall (clean).
     A first read on whether the signal points the hypothesized way (loop > pre).

This is a sanity check, NOT the Phase-3 analysis (no probe, no CUSUM, no lead-time,
no per-layer standardization yet).

Usage (pod or local, wherever the vectors are):
  uv run python scripts/pilot1_check_vectors.py \
      --manifest results/pilot1_hidden/manifest.jsonl --out-dir results/pilot1_hidden
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GAP = 3   # skip the GAP most-recent sentences (avoid trivial adjacency) in the cosine peek


def trace_key(rec: dict) -> str:
    return f"{rec['cls']}__{rec['task_id']}__{rec['sample_idx']}"


def max_cos_to_prior(vec_layer: np.ndarray, gap: int = GAP) -> np.ndarray:
    """vec_layer: (n_sent, dim) float32. Returns per-sentence max cosine to any
    prior sentence at least `gap` back; NaN where no eligible prior exists."""
    x = vec_layer.astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    xn = x / norm
    n = len(xn)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(gap, n):
        sims = xn[: i - gap + 1] @ xn[i]      # cosine to priors [0 .. i-gap]
        if sims.size:
            out[i] = float(sims.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    recs = {trace_key(r): r for r in (json.loads(l) for l in open(args.manifest))}
    vec_dir = args.out_dir / "vectors"
    files = sorted(vec_dir.glob("*.npz"))
    print(f"found {len(files)} vector files; manifest has {len(recs)} traces")
    if not files:
        return

    shape_fail, align_fail, finite_fail = [], [], []
    layers = None
    per_layer_norm = {}      # layer-idx-position -> list of mean sentence norms
    cos_summary = []         # (cls, key, layer_pos, pre_median, loop_median, all_median)

    for f in files:
        key = f.stem
        rec = recs.get(key)
        z = np.load(f)
        vecs = z["vecs"].astype(np.float32)   # (n_sent, n_layers, dim)
        layers = list(z["layers"])
        if rec is None:
            align_fail.append((key, "no manifest match")); continue
        n_sent = len(rec["sentences"])
        if vecs.shape[0] != n_sent:
            align_fail.append((key, f"n_sent {vecs.shape[0]} != manifest {n_sent}"))
        if vecs.shape[1] != len(layers) or vecs.shape[2] != 4096:
            shape_fail.append((key, f"shape {vecs.shape}"))
        if not np.isfinite(vecs).all():
            finite_fail.append(key)

        for li in range(vecs.shape[1]):
            per_layer_norm.setdefault(layers[li], []).append(
                float(np.linalg.norm(vecs[:, li, :], axis=1).mean()))

        # cosine peek per layer
        onset_s = rec.get("onset_sentence")
        for li in range(vecs.shape[1]):
            mc = max_cos_to_prior(vecs[:, li, :])
            allmed = float(np.nanmedian(mc))
            if onset_s is not None:
                pre = mc[GAP:onset_s]
                loop = mc[onset_s:]
                premed = float(np.nanmedian(pre)) if pre.size else float("nan")
                loopmed = float(np.nanmedian(loop)) if loop.size else float("nan")
            else:
                premed = loopmed = float("nan")
            cos_summary.append((rec["cls"], key, layers[li], premed, loopmed, allmed))

    print()
    print("1-2. shape/alignment/finite failures:")
    print("   shape :", shape_fail or "none")
    print("   align :", align_fail or "none")
    print("   finite:", finite_fail or "none")

    print()
    print("3. per-layer mean sentence L2-norm (expect layer", layers[-1] if layers else "?",
          "= post-norm to differ):")
    for L in layers:
        vals = per_layer_norm[L]
        print(f"   layer {L:>2}: mean L2 = {np.mean(vals):8.2f}   (over {len(vals)} traces)")

    print()
    print("4. cosine self-similarity peek (max-cos-to-prior; loop>pre would support the signal):")
    print(f"   {'cls':>18} {'layer':>5} {'pre':>6} {'loop':>6} {'all':>6}")
    # aggregate by (cls, layer)
    import collections
    agg = collections.defaultdict(lambda: {"pre": [], "loop": [], "all": []})
    for cls, key, L, pre, loop, allm in cos_summary:
        if not np.isnan(pre): agg[(cls, L)]["pre"].append(pre)
        if not np.isnan(loop): agg[(cls, L)]["loop"].append(loop)
        if not np.isnan(allm): agg[(cls, L)]["all"].append(allm)
    for (cls, L) in sorted(agg):
        a = agg[(cls, L)]
        pm = f"{np.median(a['pre']):.3f}" if a["pre"] else "  -  "
        lm = f"{np.median(a['loop']):.3f}" if a["loop"] else "  -  "
        am = f"{np.median(a['all']):.3f}" if a["all"] else "  -  "
        print(f"   {cls:>18} {L:>5} {pm:>6} {lm:>6} {am:>6}")
    print()
    print("NOTE: this is a raw-cosine sanity peek (no per-layer standardization, no CUSUM,")
    print("      no lead-time). Real separation/earliness comes from Phase 3.")


if __name__ == "__main__":
    main()
