"""Phase 1 (GPU) — teacher-force the screened loop traces and extract the two
signals needed to replicate Circular-Reasoning Figs 3b & 4 (arXiv:2601.05693):

  Fig 3b  — per-token TOP-1 PROBABILITY (max softmax) and ENTROPY of the model's
            raw next-token distribution, across the whole think region.
  Fig 4   — ALL-LAYER hidden states at the anchor token's per-cycle positions
            (loop) and at a normal recurring token's positions (baseline).

Reads the manifest from scripts/loop_collapse_screen.py. ONE forward pass per
trace with output_hidden_states=True (no generation). Index math is identical to
pilot1_extract.py:
  * think token i sits at full-input position n_prompt + i.
  * hidden_states[L] (tuple, len num_layers+1; 0 = embeddings) at position p IS the
    representation of token p — NO predictive shift.
  * logits[t] predicts token t+1, so think token i's predictive logit is at
    logits[n_prompt-1+i] (used only for the per-token distribution stats).

Faithful-to-paper choices:
  * prob/entropy are computed on the RAW model softmax (the model's intrinsic
    determinism), not the p-less-filtered distribution actually sampled. This is
    what Fig 3 measures ("entropy collapse / probability surge").
  * Fig-4 cosine/L2 are RAW per layer (no per-layer standardization). Cosine is
    scale-invariant; the last tuple entry is post-final-RMSNorm so its L2 lives on
    a different scale — faithful to the paper's raw layer-wise plot.

Memory: output_hidden_states materializes all (num_layers+1) hidden tensors AND
the logits at once (~12 GB each at the 38k cap on top of the ~16 GB model). Run on
an ~80 GB card (same assumption as pilot1_extract.py). Only 3–5 traces/model.

Usage (CUDA pod):
  HF_HUB_OFFLINE=1 uv run python scripts/loop_collapse_extract.py \
      --manifest results/loop_collapse_replication/Qwen--Qwen3-8B/manifest.jsonl \
      --model Qwen/Qwen3-8B \
      --out-dir results/loop_collapse_replication/Qwen--Qwen3-8B [--limit 1]
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


def per_token_stats(logits, prompt_len: int, chunk_size: int = 512):
    """(seq,vocab) logits -> (max_prob[], entropy_nats[]) for think positions.

    think token i's predictive logit is logits[prompt_len-1+i]. Chunked softmax so
    peak extra memory is chunk_size × vocab × 4B (~0.3 GB), moved to CPU per chunk.
    """
    import torch
    n_think = logits.shape[0] - prompt_len
    if n_think <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    think_logits = logits[prompt_len - 1: prompt_len - 1 + n_think]
    maxp_parts, H_parts = [], []
    with torch.no_grad():
        for s in range(0, n_think, chunk_size):
            chunk = think_logits[s: s + chunk_size].float()
            log_p = torch.log_softmax(chunk, dim=-1)
            p = log_p.exp()
            maxp_parts.append(p.max(dim=-1).values.cpu())
            H_parts.append(-(p * log_p).sum(dim=-1).cpu())
            del chunk, log_p, p
    maxp = torch.cat(maxp_parts).numpy().astype(np.float32)
    H = torch.cat(H_parts).numpy().astype(np.float32)
    return maxp, H


def gather_anchor_acts(hs_all, n_prompt: int, positions: list[int]) -> np.ndarray:
    """hs_all: tuple of (1, seq, hidden). positions in think coords. Returns
    (n_pos, num_layers+1, hidden) float32 — every layer at each anchor position."""
    n_layers = len(hs_all)
    hidden = hs_all[0].shape[-1]
    out = np.empty((len(positions), n_layers, hidden), dtype=np.float32)
    for pi, p in enumerate(positions):
        fp = n_prompt + p
        for L in range(n_layers):
            out[pi, L] = hs_all[L][0, fp].float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-cycles", type=int, default=12,
                    help="forward only far enough to cover this many loop cycles "
                         "(attention is causal, so truncating the suffix is lossless "
                         "for the kept positions) — keeps MPS/40GB memory in check")
    ap.add_argument("--signal-window", type=int, default=800,
                    help="ensure the forward covers onset + this many tokens (Fig 3b)")
    args = ap.parse_args()

    import torch
    from bench.generator import load_model_and_tokenizer

    recs = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        recs = recs[: args.limit]
    vec_dir = args.out_dir / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] model={args.model}  traces={len(recs)}")
    model, _tok = load_model_and_tokenizer(args.model, dtype=args.dtype)
    model.eval()
    device = next(model.parameters()).device
    n_model_layers = model.config.num_hidden_layers
    print(f"[extract] on {device}, {n_model_layers} layers "
          f"(hidden_states tuple length {n_model_layers + 1})")

    done = skipped = failed = 0
    for i, rec in enumerate(recs):
        key = f"{rec['task_id']}__{rec['sample_idx']}"
        out_path = vec_dir / f"{key}.npz"
        if out_path.exists():
            skipped += 1
            continue

        prompt_ids = rec["prompt_token_ids"]
        think_ids = rec["think_token_ids"]
        n_prompt = rec["n_prompt_tokens"]
        assert n_prompt == len(prompt_ids), f"{key}: n_prompt mismatch"
        onset = int(rec["onset_token"])
        # keep only the first --keep-cycles loop cycles (enough for Repeat 1..k)
        loop_pos = rec["loop_anchor_positions"][: args.keep_cycles]
        normal_pos = rec["normal_anchor_positions"]
        # identical-token guarantee (the heart of the Fig-4 method)
        for p in loop_pos:
            assert think_ids[p] == rec["anchor_token_id"], f"{key}: loop anchor mismatch @ {p}"

        # Causal attention ⇒ truncating the think SUFFIX is lossless for kept positions.
        # Forward far enough to cover (a) the onset + signal window for Fig 3b and
        # (b) the last kept anchor (loop or normal) for Fig 4, + a small margin.
        need = max([onset + args.signal_window]
                   + [p for p in loop_pos] + [p for p in normal_pos]) + 8
        fwd_think = min(len(think_ids), need)
        think_fwd = think_ids[:fwd_think]
        full_ids = torch.tensor([prompt_ids + think_fwd], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                out = model(full_ids, output_hidden_states=True, use_cache=False)
            maxp, H = per_token_stats(out.logits[0], n_prompt)
            hs_all = out.hidden_states
            loop_acts = gather_anchor_acts(hs_all, n_prompt, loop_pos)
            normal_acts = (gather_anchor_acts(hs_all, n_prompt, normal_pos)
                           if normal_pos else np.zeros((0, n_model_layers + 1,
                                                         hs_all[0].shape[-1]), np.float32))
            del out, hs_all
        except Exception as e:
            print(f"  [warn] {key} failed: {e}", file=sys.stderr)
            failed += 1
            continue
        finally:
            del full_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # sanity
        assert maxp.shape[0] == fwd_think, f"{key}: signal len {maxp.shape[0]} != fwd {fwd_think}"
        assert ((maxp >= 0) & (maxp <= 1.0001)).all(), f"{key}: prob out of [0,1]"
        assert (H >= -1e-4).all() and np.isfinite(H).all(), f"{key}: bad entropy"
        assert np.isfinite(loop_acts).all(), f"{key}: non-finite loop acts"
        assert loop_acts.shape == (len(loop_pos), n_model_layers + 1, model.config.hidden_size), \
            f"{key}: loop_acts shape {loop_acts.shape}"

        np.savez_compressed(
            out_path,
            prob=maxp, entropy=H,
            onset_token=np.int64(rec["onset_token"]),
            n_prompt=np.int64(n_prompt),
            n_think=np.int64(fwd_think),                 # length of prob/entropy arrays (forwarded)
            n_think_full=np.int64(len(think_ids)),       # full think length (provenance)
            loop_acts=loop_acts, normal_acts=normal_acts,
            loop_positions=np.array(loop_pos, dtype=np.int64),
            normal_positions=np.array(normal_pos, dtype=np.int64),
            anchor_token_id=np.int64(rec["anchor_token_id"]),
            num_layers=np.int64(n_model_layers),
        )
        done += 1
        print(f"  [{i+1}/{len(recs)}] {key}  signals={maxp.shape[0]}  "
              f"loop_acts={loop_acts.shape}  normal_acts={normal_acts.shape}  "
              f"(seq={n_prompt + len(think_ids)})")

    (args.out_dir / "extract_config.json").write_text(json.dumps({
        "model": args.model, "dtype": args.dtype,
        "n_done": done, "n_skipped": skipped, "n_failed": failed,
        "manifest": str(args.manifest),
    }, indent=2))
    print(f"[extract] done. extracted={done} skipped={skipped} failed={failed}  -> {vec_dir}")


if __name__ == "__main__":
    main()
