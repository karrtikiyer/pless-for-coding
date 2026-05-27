"""Tests for the runner-level n_samples chunker used to bound HF-backend
KV memory on Phase A's N=100 cells (CUDA OOM on H100 80GB without chunking).

Helper under test: ``bench.apps.runner._chunk_sizes(n_samples, hf_batch_size)``
returns the list of per-call sample counts. The runner uses this to drive
multiple ``generate_samples`` / ``generate_samples_split`` calls when
hf_batch_size < n_samples, preserving sample independence.
"""
from __future__ import annotations

import pytest


def test_chunk_sizes_default_none_is_single_chunk():
    """hf_batch_size=None preserves current behavior — one call of n_samples."""
    from bench.apps.runner import _chunk_sizes
    assert _chunk_sizes(10, None) == [10]


def test_chunk_sizes_zero_or_negative_treated_as_no_chunking():
    """0 / negative chunk sizes are no-ops (single call)."""
    from bench.apps.runner import _chunk_sizes
    assert _chunk_sizes(10, 0) == [10]
    assert _chunk_sizes(10, -1) == [10]


def test_chunk_sizes_smaller_than_n_samples_splits():
    """n_samples=10, chunk=3 → [3, 3, 3, 1]."""
    from bench.apps.runner import _chunk_sizes
    assert _chunk_sizes(10, 3) == [3, 3, 3, 1]


def test_chunk_sizes_equal_to_n_samples_single_chunk():
    """n_samples=10, chunk=10 → single chunk (no overhead)."""
    from bench.apps.runner import _chunk_sizes
    assert _chunk_sizes(10, 10) == [10]


def test_chunk_sizes_larger_than_n_samples_clamps():
    """hf_batch_size > n_samples should clamp to n_samples (single chunk)."""
    from bench.apps.runner import _chunk_sizes
    assert _chunk_sizes(5, 100) == [5]


def test_chunk_sizes_phase_a_shape():
    """Phase A's actual shape: N=100, chunk=10 → ten chunks of 10."""
    from bench.apps.runner import _chunk_sizes
    assert _chunk_sizes(100, 10) == [10] * 10


def test_chunk_sizes_sum_equals_n_samples():
    """Across many (n, chunk) combos: sum of chunks must equal n_samples."""
    from bench.apps.runner import _chunk_sizes
    for n in [1, 5, 10, 17, 50, 100]:
        for c in [None, 0, 1, 3, 5, 10, 100]:
            chunks = _chunk_sizes(n, c)
            assert sum(chunks) == n, (
                f"n_samples={n}, chunk={c} → chunks={chunks} sums to "
                f"{sum(chunks)}, expected {n}"
            )
            assert all(b > 0 for b in chunks), (
                f"n_samples={n}, chunk={c} → empty chunk: {chunks}"
            )
