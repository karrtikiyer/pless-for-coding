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

from bench.generator import load_model_and_tokenizer, _expand_past_key_values
from scripts.batched_gen import left_pad_batch


def last_logits_solo(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    return out.logits[0, -1].float().cpu(), out.past_key_values


def teacher_decode_row0(model, prompt_ids, n, steps, dev, drop_step=None, keep=None):
    """Compaction primitive check. Prefill n copies of a shared prompt, then teacher-force a
    per-row-DISTINCT fixed token (row with original id o always gets token 1000+o) so rows
    diverge. Optionally at `drop_step` compact the batch to `keep` (via batch_select_indices).
    Return row-0's next-logits at each step. Row 0 is always kept and always fed 1000, so a
    correct compaction must leave row-0's logits unchanged whether or not other rows are
    dropped — any divergence means batch_select_indices corrupted/misaligned the cache.
    """
    with torch.no_grad():
        pf = model(input_ids=torch.tensor([prompt_ids], device=dev), use_cache=True,
                   logits_to_keep=1)
    past = _expand_past_key_values(pf.past_key_values, n)
    active = list(range(n))
    row0 = []
    for step in range(steps):
        toks = torch.tensor([[1000 + o] for o in active], device=dev)
        with torch.no_grad():
            out = model(input_ids=toks, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
        row0.append(logits[active.index(0)].cpu())
        if drop_step is not None and step == drop_step:
            kp = [p for p, o in enumerate(active) if o in keep]
            past.batch_select_indices(torch.tensor(kp, device=dev))
            active = [active[p] for p in kp]
    return row0


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
    # --- Test C: compaction primitive — dropping rows must not change a survivor ---
    print("\n[Test C] compaction: drop rows {1,3} at step 2, row-0 logits must be unchanged",
          flush=True)
    full = teacher_decode_row0(model, seqs[1], n=4, steps=6, dev=dev)
    drop = teacher_decode_row0(model, seqs[1], n=4, steps=6, dev=dev, drop_step=2, keep={0, 2})
    for t in range(len(full)):
        d = (full[t] - drop[t]).abs().max().item()
        am = int(full[t].argmax()) == int(drop[t].argmax())
        good = am and d <= thresh
        ok = ok and good
        tagpost = " (post-drop)" if t > 2 else ""
        print(f"  step {t}{tagpost} max|Δlogit|={d:.4f} argmax_match={am} "
              f"{'OK' if good else 'FAIL'}", flush=True)

    print("\n" + ("PASS — batched forward + compaction are per-row equivalent (argmax agrees; "
                  "Δ within noise)" if ok else
                  "FAIL — a path diverges beyond noise / flips argmax (real bug)"),
          flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
