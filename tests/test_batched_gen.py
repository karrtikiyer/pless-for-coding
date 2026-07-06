"""TDD for the left-padded batch construction (scripts/batched_gen.py) — the bug-prone
ragged-prefill piece. No model/GPU: pure tensor logic.
"""
import torch

from scripts.batched_gen import left_pad_batch, batched_phase2


def test_shapes_and_left_alignment():
    ids, mask, pos = left_pad_batch([[5, 6, 7], [9]], pad_id=0)
    assert ids.shape == (2, 3) == mask.shape == pos.shape
    # row 0 full (no pad); row 1 left-padded with two pads
    assert ids.tolist() == [[5, 6, 7], [0, 0, 9]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_position_ids_start_at_zero_on_first_real_token():
    _, _, pos = left_pad_batch([[5, 6, 7], [9]], pad_id=0)
    # row 0: positions 0,1,2 ; row 1: pads clamped to 0, real token at position 0
    assert pos.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_equal_length_no_padding():
    ids, mask, pos = left_pad_batch([[1, 2], [3, 4]], pad_id=-1)
    assert mask.tolist() == [[1, 1], [1, 1]]
    assert pos.tolist() == [[0, 1], [0, 1]]
    assert (ids == torch.tensor([[1, 2], [3, 4]])).all()


def test_three_rows_varied_lengths():
    ids, mask, pos = left_pad_batch([[1, 1, 1, 1], [2, 2], [3]], pad_id=0)
    assert ids.tolist() == [[1, 1, 1, 1], [0, 0, 2, 2], [0, 0, 0, 3]]
    assert mask.tolist() == [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
    # each row's last real token sits at the final column → logits[:, -1] lines up
    assert [m[-1] for m in mask.tolist()] == [1, 1, 1]
    # positions: real tokens count from 0 per row
    assert pos.tolist() == [[0, 1, 2, 3], [0, 0, 0, 1], [0, 0, 0, 0]]


def test_empty_raises():
    try:
        left_pad_batch([], pad_id=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- batched_phase2: re-chop orchestration across many rows, via a mock round_fn ------

def _scripted_round(scripts):
    """round_fn that pops the next scripted per-round result list, and records the
    prefixes/max_new it was called with."""
    calls = []
    it = iter(scripts)

    def round_fn(prefixes, max_new):
        calls.append({"prefixes": [list(p) for p in prefixes], "max_new": max_new})
        return next(it)

    round_fn.calls = calls
    return round_fn


def test_phase2_no_reloop_finalizes_first_round():
    fired = [{"prefix": [1, 2], "budget": 100}, {"prefix": [3], "budget": 100}]
    rf = _scripted_round([[{"gen": [7, 8], "reason": "eos", "onset": None},
                           {"gen": [9], "reason": "eos", "onset": None}]])
    batched_phase2(fired, rf, max_chops=3, round_cap=8192)
    assert fired[0]["cont"] == [7, 8] and fired[0]["chops"] == 0 and fired[0]["done"]
    assert fired[1]["cont"] == [9] and fired[1]["reason"] == "eos"
    assert len(rf.calls) == 1                       # one round, all finalized


def test_phase2_reloop_then_close_accumulates_and_rebatches():
    fired = [{"prefix": [1], "budget": 100}]
    rf = _scripted_round([
        [{"gen": [10, 11, 12], "reason": "loop", "onset": 2}],   # keep [10,11], re-chop
        [{"gen": [20, 21], "reason": "eos", "onset": None}],     # then close
    ])
    batched_phase2(fired, rf, max_chops=3, round_cap=8192)
    assert fired[0]["cont"] == [10, 11, 20, 21]     # chopped tail dropped, continuation appended
    assert fired[0]["chops"] == 1 and fired[0]["reason"] == "eos"
    # round 2's prefix must be prefix + accumulated cont so far
    assert rf.calls[1]["prefixes"] == [[1, 10, 11]]


def test_phase2_respects_max_chops_and_shrinks_batch():
    # two rows: A closes round 1; B loops forever -> stops re-chopping after max_chops
    fired = [{"prefix": [1], "budget": 100}, {"prefix": [2], "budget": 100}]
    rf = _scripted_round([
        [{"gen": [10], "reason": "eos", "onset": None},
         {"gen": [20, 21], "reason": "loop", "onset": 1}],       # A done, B re-chops
        [{"gen": [30, 31], "reason": "loop", "onset": 1}],       # only B in batch now
        [{"gen": [40, 41], "reason": "loop", "onset": 1}],       # B, chop 2
    ])
    batched_phase2(fired, rf, max_chops=2, round_cap=8192)
    assert fired[0]["done"] and fired[0]["cont"] == [10]
    assert fired[1]["chops"] == 2 and fired[1]["reason"] == "loop"
    # after A finalized, later rounds batch only B (shrinking active set)
    assert rf.calls[1]["prefixes"] == [[2, 20]]
    assert rf.calls[2]["prefixes"] == [[2, 20, 30]]


def test_phase2_round_cap_bounds_max_new():
    fired = [{"prefix": [1], "budget": 5}]
    rf = _scripted_round([[{"gen": [7], "reason": "eos", "onset": None}]])
    batched_phase2(fired, rf, max_chops=3, round_cap=8192)
    assert rf.calls[0]["max_new"] == 5             # min(remaining budget, round_cap)
