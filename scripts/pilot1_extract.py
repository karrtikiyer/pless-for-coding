"""Pilot 1 — Phase 2: hidden-state extraction via teacher forcing (GPU).

Reads the manifest from Phase 1 (scripts/pilot1_segment.py). For each trace:
  1. Build full_input = prompt_token_ids + think_token_ids (the faithful sequence).
  2. ONE forward pass with output_hidden_states=True (no generation, no sampling).
  3. For each target layer, mean-pool the per-token hidden states over each
     sentence's token span -> one vector per (sentence, layer).
  4. Save the small (n_sentences, n_layers, hidden_dim) array; discard raw states.

Index math (verified, no predictive shift — hidden state at position p IS the
representation of token p, unlike logits):
  * think token i (0-indexed in the think stream) sits at full-input position
    n_prompt + i.
  * sentence span [tok_start, tok_end) in think coords -> full-input positions
    [n_prompt + tok_start, n_prompt + tok_end).
  * hidden_states is a tuple of (num_layers+1) tensors; index 0 = embeddings,
    index L = output after L transformer layers. LAYERS are indices into this
    tuple (so 36 = last layer for a 36-layer model).

IMPORTANT — pre/post-norm asymmetry: the LAST tuple entry (index = num_layers) is
POST final-RMSNorm, whereas intermediate entries (6, 16, 24) are raw pre-norm
residual-stream outputs. So the layer-36 vectors live on a different scale than
the others. This is faithful to the paper's "last layer hidden state", but it
means Phase 3 MUST standardize per-layer (z-score per layer) before any
cross-layer comparison or the layer-36 result is not comparable to the rest.

Memory: output_hidden_states materializes ALL (num_layers+1) tensors on GPU at
once — for the manifest's 38000-token cap that's ~37 x 38000 x 4096 x 2B ≈ 12 GB
of hidden states on top of the ~16 GB model. Safe on an 80 GB card; would OOM a
40 GB card on the longest traces (switch to per-layer forward hooks if needed).

Output: one compressed .npz per trace under <out-dir>/vectors/, plus the run config.
Metadata (labels, spans, onset_sentence) stays in the manifest — Phase 3 joins by key.

Usage (on a CUDA pod):
  HF_HUB_OFFLINE=1 uv run python scripts/pilot1_extract.py \
      --manifest results/pilot1_hidden/manifest.jsonl \
      --model Qwen/Qwen3-8B --layers 6 16 24 36 \
      --out-dir results/pilot1_hidden [--limit 2]   # --limit for a GPU smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def trace_key(rec: dict) -> str:
    return f"{rec['cls']}__{rec['task_id']}__{rec['sample_idx']}"


def pool_sentences(hidden_by_layer, n_prompt: int, sentences: list[dict]):
    """hidden_by_layer: list over target layers of (seq_len, hidden_dim) float32
    tensors (already on CPU). Returns np.ndarray (n_sent, n_layers, hidden_dim)."""
    n_sent = len(sentences)
    n_layers = len(hidden_by_layer)
    hidden_dim = hidden_by_layer[0].shape[-1]
    out = np.empty((n_sent, n_layers, hidden_dim), dtype=np.float32)
    for si, s in enumerate(sentences):
        p0 = n_prompt + s["tok_start"]
        p1 = n_prompt + s["tok_end"]
        for li, hs in enumerate(hidden_by_layer):
            out[si, li] = hs[p0:p1].mean(axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 16, 24, 36])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--limit", type=int, default=None, help="process only first N traces (GPU smoke)")
    args = ap.parse_args()

    import torch
    from bench.generator import load_model_and_tokenizer

    recs = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        recs = recs[: args.limit]
    vec_dir = args.out_dir / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] model={args.model}  layers={args.layers}  traces={len(recs)}")
    model, _tok = load_model_and_tokenizer(args.model, dtype=args.dtype)
    model.eval()
    device = next(model.parameters()).device
    n_model_layers = model.config.num_hidden_layers
    assert max(args.layers) <= n_model_layers, \
        f"layer {max(args.layers)} > num_hidden_layers {n_model_layers}"
    print(f"[extract] model on {device}, {n_model_layers} layers "
          f"(hidden_states tuple length {n_model_layers + 1})")

    done = skipped = failed = 0
    for i, rec in enumerate(recs):
        key = trace_key(rec)
        out_path = vec_dir / f"{key}.npz"
        if out_path.exists():
            skipped += 1
            continue

        prompt_ids = rec["prompt_token_ids"]
        think_ids = rec["think_token_ids"]
        n_prompt = rec["n_prompt_tokens"]
        assert n_prompt == len(prompt_ids), f"{key}: n_prompt mismatch"
        assert rec["n_think_tokens"] == len(think_ids), f"{key}: n_think mismatch"
        # last sentence must end exactly at n_think (Phase-1 invariant I2) — guard here too
        assert rec["sentences"][-1]["tok_end"] == len(think_ids), f"{key}: span/think mismatch"

        full_ids = torch.tensor([prompt_ids + think_ids], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                out = model(full_ids, output_hidden_states=True, use_cache=False)
            hs_all = out.hidden_states  # tuple len = n_model_layers+1, each (1, seq, dim)
            # pull only target layers, to CPU float32, then free everything on GPU
            hidden_by_layer = [hs_all[L][0].float().cpu().numpy() for L in args.layers]
            del out, hs_all
        except Exception as e:
            print(f"  [warn] {key} failed: {e}", file=sys.stderr)
            failed += 1
            continue
        finally:
            # the finally always runs (incl. on the except's continue), so one
            # empty_cache() here covers both success and OOM paths.
            del full_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        vecs = pool_sentences(hidden_by_layer, n_prompt, rec["sentences"])
        # sanity (fp32): no NaN/inf (a degenerate empty span would produce NaN means)
        assert np.isfinite(vecs).all(), f"{key}: non-finite pooled vector (fp32)"
        vecs16 = vecs.astype(np.float16)
        # re-check after downcast: late-layer "massive activations" can exceed
        # fp16's 65504 max and silently become inf.
        assert np.isfinite(vecs16).all(), f"{key}: fp16 overflow on store (massive activations)"
        np.savez_compressed(
            out_path,
            vecs=vecs16,                           # half-precision storage; cast back in Phase 3
            layers=np.array(args.layers, dtype=np.int32),
        )
        done += 1
        print(f"  [{i+1}/{len(recs)}] {key}  vecs={vecs.shape}  "
              f"(seq={n_prompt + rec['n_think_tokens']})")

    # write run config for provenance
    (args.out_dir / "extract_config.json").write_text(json.dumps({
        "model": args.model, "layers": args.layers, "dtype": args.dtype,
        "n_done": done, "n_skipped": skipped, "n_failed": failed,
        "manifest": str(args.manifest),
    }, indent=2))
    print(f"[extract] done. extracted={done} skipped={skipped} failed={failed}  -> {vec_dir}")


if __name__ == "__main__":
    main()
