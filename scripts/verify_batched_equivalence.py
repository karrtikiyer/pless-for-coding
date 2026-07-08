"""Logit-equivalence harness for the batched decode path — RUN ON THE SPARE GPU.

Proves the batched, left-padded forward computes the SAME thing per row as running that
row alone — independent of sampling/RNG (teacher-forced: we compare logits, not samples).
This is the rigorous "is it doing what it's supposed to" check; it would have caught the
left-pad position_ids class of bug before a full run.

Two tests, each per row, comparing batched-vs-solo:
  A. PREFILL   — last-position logits after the initial forward.
  B. DECODE    — one incremental step (append a token), exercising the cur_mask extension
                 + position_ids = real_len + step logic used inside batched_gen_round.

Pass criterion: argmax agrees for every row (robust to bf16), and max|Δlogit| is small
(bf16 reduction-order noise is ~<=0.1; a real bug shows argmax mismatch and huge Δ).

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=2 \
     uv run python scripts/verify_batched_equivalence.py
"""
import os

import torch

from bench.generator import load_model_and_tokenizer
from scripts.batched_gen import left_pad_batch


def last_logits_solo(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    return out.logits[0, -1].float().cpu(), out.past_key_values


def main():
    model_id = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    dtype = os.environ.get("DTYPE", "bfloat16")          # set float32 for the airtight control
    tol = float(os.environ.get("TOL", "0.1"))
    model, tok = load_model_and_tokenizer(model_id, dtype=dtype)
    if dtype == "float32":                              # loader falls back to bf16 otherwise
        model = model.float()
    dev = model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    # Varied-length real token sequences (exercise left padding).
    texts = [
        "def solve():\n    n = int(input())\n    print(n * 2)\n",
        "Given an array of integers, return the maximum subarray sum. " * 6,
        "The quick brown fox. " * 30,
        "import sys\n" + "x = 1\n" * 80,
    ]
    seqs = [tok.encode(t) for t in texts]
    print(f"model={model_id} dtype={dtype} | {len(seqs)} rows, "
          f"lengths={[len(s) for s in seqs]}", flush=True)

    # Noise floor: same solo forward twice → intrinsic CUDA/dtype run-to-run wobble. The
    # batched-vs-solo diff is a real divergence only if it clears this floor (and flips argmax).
    _a, _ = last_logits_solo(model, torch.tensor([seqs[0]], device=dev))
    _b, _ = last_logits_solo(model, torch.tensor([seqs[0]], device=dev))
    noise_floor = (_a - _b).abs().max().item()
    thresh = max(tol, 3 * noise_floor)
    print(f"solo-vs-solo noise floor={noise_floor:.4f} | Δ pass threshold={thresh:.4f} "
          f"(argmax agreement is the primary criterion)", flush=True)

    # --- solo: each row alone (batch=1, no padding) ---
    solo_prefill, solo_past, solo_next = [], [], []
    for s in seqs:
        lg, past = last_logits_solo(model, torch.tensor([s], device=dev))
        solo_prefill.append(lg)
        solo_past.append(past)
        solo_next.append(int(lg.argmax()))              # deterministic next token (greedy)
    # solo decode: feed each row its own argmax token
    solo_decode = []
    for i, s in enumerate(seqs):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[solo_next[i]]], device=dev),
                        past_key_values=solo_past[i], use_cache=True, logits_to_keep=1)
        solo_decode.append(out.logits[0, -1].float().cpu())

    # --- batched: left-padded, all rows together ---
    ids, mask, pos = left_pad_batch(seqs, pad_id)
    ids, mask, pos = ids.to(dev), mask.to(dev), pos.to(dev)
    real_len = mask.sum(-1)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    use_cache=True, logits_to_keep=1)
    batch_prefill = out.logits[:, -1].float().cpu()
    past = out.past_key_values
    # batched decode: feed the SAME token solo used, one step
    nxt = torch.tensor(solo_next, device=dev).view(-1, 1)
    cur_mask = torch.cat([mask, torch.ones((len(seqs), 1), dtype=mask.dtype, device=dev)], dim=1)
    step_pos = (real_len + 0).view(-1, 1)               # first decode step
    with torch.no_grad():
        out = model(input_ids=nxt, attention_mask=cur_mask, position_ids=step_pos,
                    past_key_values=past, use_cache=True, logits_to_keep=1)
    batch_decode = out.logits[:, -1].float().cpu()

    # --- compare ---
    ok = True
    for i in range(len(seqs)):
        for tag, solo, bat in [("prefill", solo_prefill[i], batch_prefill[i]),
                               ("decode", solo_decode[i], batch_decode[i])]:
            d = (solo - bat).abs().max().item()
            am = int(solo.argmax()) == int(bat.argmax())
            good = am and d <= thresh
            ok = ok and good
            print(f"  row {i} [{tag}] max|Δlogit|={d:.4f} argmax_match={am} "
                  f"{'OK' if good else 'FAIL'}", flush=True)
    print("\n" + ("PASS — batched forward is per-row equivalent to solo (argmax agrees; "
                  "Δ within noise)" if ok else
                  "FAIL — batched path diverges beyond noise / flips argmax (real bug)"),
          flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
