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
