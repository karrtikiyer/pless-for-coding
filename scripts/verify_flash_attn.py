"""Verify flash_attention_2 vs sdpa — RUN ON THE POD (needs CUDA + flash_attn).

Answers three questions with evidence, so the FA2 decision isn't a guess:
  1. AVAILABLE?  can we import flash_attn and load the model with attn_implementation=fa2.
  2. QUALITY?    are FA2's logits the same as sdpa's? (batch=1, no padding → clean compare;
                 argmax must agree — FP-level Δ is fine, an argmax flip / large Δ is not.)
  3. FASTER?     tokens/sec of a batched decode loop under each, same inputs.

Loads the model once per backend (freeing between) to keep memory modest.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=2 \
     uv run python scripts/verify_flash_attn.py
"""
import gc
import os
import time

import torch

from bench.generator import load_model_and_tokenizer


def run_backend(attn_impl, model_id, seqs, decode_steps, batch):
    """Return (last_logits_per_seq, tokens_per_sec) for the given attn implementation, or
    raise if it can't be loaded (→ not available)."""
    model, tok = load_model_and_tokenizer(model_id, dtype="bfloat16", attn_impl=attn_impl)
    dev = model.device
    # (2) quality: batch=1, no padding, last-position logits per sequence
    logits = []
    with torch.no_grad():
        for s in seqs:
            out = model(input_ids=torch.tensor([s], device=dev), use_cache=True, logits_to_keep=1)
            logits.append(out.logits[0, -1].float().cpu())
    # (3) speed: batched decode loop of `batch` copies of the longest seq, timed
    seq = max(seqs, key=len)
    ids = torch.tensor([seq] * batch, device=dev)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1).view(batch, 1)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(decode_steps):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1).view(batch, 1)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
    tps = batch * decode_steps / dt
    del model, past, out
    gc.collect(); torch.cuda.empty_cache()
    return logits, tps


def main():
    model_id = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    steps = int(os.environ.get("DECODE_STEPS", "300"))
    batch = int(os.environ.get("BATCH", "8"))
    tol = float(os.environ.get("TOL", "0.1"))

    import importlib.util
    have_fa2 = importlib.util.find_spec("flash_attn") is not None
    print(f"[1] flash_attn importable: {have_fa2}", flush=True)

    texts = ["def f():\n    return sum(range(10))\n",
             "Given an array, return the max subarray sum. " * 8,
             "import sys\n" + "x=1\n" * 60]
    tok = load_model_and_tokenizer(model_id, dtype="bfloat16")[1]
    seqs = [tok.encode(t) for t in texts]

    print(f"[3] timing: {batch}-wide decode x {steps} steps", flush=True)
    sdpa_logits, sdpa_tps = run_backend("sdpa", model_id, seqs, steps, batch)
    print(f"    sdpa: {sdpa_tps:.1f} tok/s", flush=True)

    if not have_fa2:
        print("[2/3] flash_attention_2 NOT available on this pod → sdpa stays. Done.", flush=True)
        raise SystemExit(0)
    try:
        fa2_logits, fa2_tps = run_backend("flash_attention_2", model_id, seqs, steps, batch)
    except Exception as e:
        print(f"[2/3] FA2 failed to load/run: {type(e).__name__}: {str(e)[:200]}", flush=True)
        raise SystemExit(0)
    print(f"    fa2 : {fa2_tps:.1f} tok/s  |  speedup vs sdpa: {fa2_tps / sdpa_tps:.2f}x", flush=True)

    print("[2] quality: FA2 vs sdpa last-logits (argmax must agree; Δ small = FP only)", flush=True)
    ok = True
    for i in range(len(seqs)):
        d = (sdpa_logits[i] - fa2_logits[i]).abs().max().item()
        am = int(sdpa_logits[i].argmax()) == int(fa2_logits[i].argmax())
        ok = ok and am
        print(f"    seq {i} (len {len(seqs[i])}): max|Δlogit|={d:.4f} argmax_match={am} "
              f"{'OK' if am and d <= tol else 'CHECK'}", flush=True)
    print(f"\nVERDICT: FA2 {'faster' if fa2_tps > sdpa_tps else 'NOT faster'} "
          f"({fa2_tps / sdpa_tps:.2f}x), outputs {'argmax-equivalent' if ok else 'DIVERGENT'} "
          f"(FP-level Δ; not bit-identical → future runs only).", flush=True)


if __name__ == "__main__":
    main()
