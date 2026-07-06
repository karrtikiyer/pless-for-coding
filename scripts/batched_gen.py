"""Left-padded batched generation primitives for the live adaptive decoder.

The full-252 runs need to batch sequences with DIFFERENT-length prefixes:
  - Phase 2: the fired rows' chopped prefixes (prompt + trace[:onset]) differ per row.
  - Phase 1 (optional cross-task): different tasks' prompts differ per row.

HF decoder-only models require LEFT padding for batched generation (so every row's last
real token is at the same final position and the next-token logits line up at index -1).
The subtle, bug-prone part is position_ids: with left padding, positions must start at 0 at
each row's FIRST REAL token, i.e. position_ids = cumsum(attention_mask) - 1 (clamped >=0 on
pad). This module isolates that construction so it is unit-tested WITHOUT a model/GPU; the
model-bound decode loop (in live_adaptive_decode) consumes these tensors.
"""
import torch

from scripts.repeat_detector import RepeatDetector


def left_pad_batch(prefix_lists, pad_id):
    """Left-pad a list of ragged token-id lists into a batched tensor.

    Returns (input_ids, attention_mask, position_ids), each (B, max_len):
      input_ids     : pad_id in the left gap, real tokens right-aligned
      attention_mask: 0 over the pad gap, 1 over real tokens
      position_ids  : 0 at each row's first real token, increasing rightward; the pad
                      positions are set to 0 (masked out anyway) to stay in-range.
    """
    if not prefix_lists:
        raise ValueError("left_pad_batch: empty prefix_lists")
    B = len(prefix_lists)
    max_len = max(len(p) for p in prefix_lists)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, p in enumerate(prefix_lists):
        L = len(p)
        if L:
            input_ids[i, max_len - L:] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, max_len - L:] = 1
    # position_ids: cumsum of the mask minus 1 → first real token gets 0; clamp pad to 0.
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    return input_ids, attention_mask, position_ids


def batched_gen_round(model, prefixes, sampler, temp, max_new, eos_id, think_end_id,
                      n_g, k_g, w_g, pad_id, label="", log_every=512):
    """One batched decode round over RAGGED prefixes at a single sampler. Left-pads the
    prefixes, prefills once (batched), then decodes token-by-token; each row stops on
    eos / loop-detected(onset) / cap. Rows that emit </think> keep generating code
    (detection off). NO re-chop inside — a re-looped row just stops with reason='loop' and
    the orchestrator re-chops it in the next round (avoids mid-batch KV surgery).

    position_ids/attention_mask are threaded explicitly through every step (left padding
    makes HF's auto-computation wrong). Returns per row {gen, reason, onset, closed_think}.
    GPU-validated by the smoke, not unit tests (needs a model); the orchestration that
    calls it (batched_phase2) IS unit-tested with a mock round_fn.
    """
    dev = model.device
    ids, mask, pos = left_pad_batch(prefixes, pad_id)
    ids, mask, pos = ids.to(dev), mask.to(dev), pos.to(dev)
    B = ids.shape[0]
    real_len = mask.sum(-1)                          # (B,) per-row real token count
    with torch.no_grad():
        # logits_to_keep=1: only materialize the last-position logits. Without it the prefill
        # builds (B, seq_len, vocab) — 40+ GB for 10 long fired prefixes → OOM. Mirrors decode_round.
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                    logits_to_keep=1)
    past = out.past_key_values
    logits = out.logits[:, -1].float()               # (B,vocab); left-pad → col -1 aligns

    dets = [RepeatDetector(n=n_g, k=k_g, window=w_g) for _ in range(B)]
    in_code = [False] * B
    reason = [None] * B
    onset = [None] * B
    gens = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_mask = mask
    for step in range(max_new):
        lg = logits / temp if temp != 1.0 else logits
        probs = torch.softmax(lg, dim=-1)
        nxt = sampler(probs.clone()).view(B)
        nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
        for i in range(B):
            if bool(finished[i]):
                continue
            tid = int(nxt[i])
            gens[i].append(tid)
            if tid == eos_id:
                finished[i] = True; reason[i] = "eos"
            elif tid == think_end_id:
                in_code[i] = True
            elif not in_code[i] and dets[i].update(tid):
                finished[i] = True; reason[i] = "loop"; onset[i] = dets[i].onset
        if log_every and step % log_every == 0:
            print(f"    {label} step {step}/{max_new} finished={int(finished.sum())}/{B}",
                  flush=True)
        if bool(finished.all()) or step == max_new - 1:
            break
        cur_mask = torch.cat([cur_mask, torch.ones((B, 1), dtype=cur_mask.dtype, device=dev)], dim=1)
        step_pos = (real_len + step).view(B, 1)       # new token's position per row
        with torch.no_grad():
            out = model(input_ids=nxt.view(B, 1), attention_mask=cur_mask,
                        position_ids=step_pos, past_key_values=past, use_cache=True,
                        logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
    for i in range(B):
        if reason[i] is None:
            reason[i] = "cap"
    return [{"gen": gens[i], "reason": reason[i], "onset": onset[i],
             "closed_think": in_code[i]} for i in range(B)]


def batched_phase2(fired, round_fn, max_chops, round_cap):
    """Batched analog of adaptive_continue's re-chop loop, across MANY fired rows at once.

    fired: list of dicts, each with 'prefix' (list[int], = prompt+trace[:onset]) and
    'budget' (int, tokens allowed for the whole continuation). Mutates each with
    'cont' (accumulated continuation tokens), 'chops', 'reason', 'done'.

    round_fn(prefixes:list[list[int]], max_new:int) -> list[{gen, reason, onset}] runs one
    batched decode round (real impl wraps batched_gen_round). A 'loop' row (under max_chops)
    keeps gen[:onset] and re-enters next round; any other reason finalizes it. Re-chop = a
    new batched round, so the active set shrinks each round — no mid-batch surgery.
    """
    for f in fired:
        f["cont"] = []
        f["chops"] = 0
        f["reason"] = None
        f["done"] = False
    for rnd in range(max_chops + 1):
        batch = [f for f in fired if not f["done"]]
        if not batch:
            break
        prefixes = [f["prefix"] + f["cont"] for f in batch]
        rem = [max(0, f["budget"] - len(f["cont"])) for f in batch]
        max_new = max(1, min(max(rem), round_cap))
        res = round_fn(prefixes, max_new)
        for f, r in zip(batch, res):
            if r["reason"] == "loop" and rnd < max_chops:
                f["cont"] = f["cont"] + r["gen"][:r["onset"]]
                f["chops"] += 1
            else:
                f["cont"] = f["cont"] + r["gen"]
                f["done"] = True
                f["reason"] = r["reason"]
    return fired
