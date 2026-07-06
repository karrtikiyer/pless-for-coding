"""TDD for the left-padded batch construction (scripts/batched_gen.py) — the bug-prone
ragged-prefill piece. No model/GPU: pure tensor logic.
"""
import torch

from scripts.batched_gen import left_pad_batch


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
