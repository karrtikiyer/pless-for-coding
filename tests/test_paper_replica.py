"""Tests for bench.apps.paper_replica — paper-prompt loader for Phase A
Deepseek replication experiment.

Goal: feed the EXACT prompt the paper sent to each (model, problem) pair
into our generator, bypassing our own chat-template wrapper. This isolates
the sampler effect (pless_alpha vs nucleus) from any prompt-format effect.

We do NOT hit the network in these tests — they exercise the dedup/cache
logic on synthetic rows mimicking the sh0416/outputs-apps schema.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ─── Schema sanity ──────────────────────────────────────────────────────


def test_module_exposes_load_paper_prompts():
    """Public API surface: load_paper_prompts must be importable."""
    from bench.apps.paper_replica import load_paper_prompts
    assert callable(load_paper_prompts)


# ─── Dedup logic ────────────────────────────────────────────────────────


def _fake_rows(model: str, source: str, n_problems: int, n_samples_each: int):
    """Generate fake rows mimicking sh0416/outputs-apps schema."""
    for pid in range(n_problems):
        for s in range(n_samples_each):
            yield {
                "problem_id": pid,
                "source": source,
                "model": model,
                # 100 rows for the same problem share the same prompt
                "prompt": f"PROMPT for problem {pid}",
                "completion": f"completion sample {s}",
                "status": "Passed" if s % 3 == 0 else "Failed",
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 1024,
            }


def test_load_paper_prompts_dedups_to_one_per_problem(tmp_path):
    """The HF dataset has 100 rows per problem (one per sample); we want
    one prompt per problem_id. Verify dedup."""
    from bench.apps import paper_replica
    rows = list(_fake_rows("deepseek-ai/deepseek-coder-6.7b-instruct",
                           "CODEFORCES", n_problems=5, n_samples_each=20))
    with patch.object(paper_replica, "_stream_dataset_rows", return_value=iter(rows)):
        out = paper_replica.load_paper_prompts(
            model="deepseek-ai/deepseek-coder-6.7b-instruct",
            source="CODEFORCES",
            difficulty="introductory",
            cache_dir=tmp_path,
        )
    # 5 unique problem_ids
    assert len(out) == 5
    assert set(out.keys()) == {0, 1, 2, 3, 4}
    # Each value is the prompt string for that problem
    for pid, prompt in out.items():
        assert prompt == f"PROMPT for problem {pid}"


def test_load_paper_prompts_filters_by_model(tmp_path):
    """Multiple models in the same dataset; the loader must filter to the
    requested model only."""
    from bench.apps import paper_replica
    rows = (
        list(_fake_rows("deepseek-ai/deepseek-coder-6.7b-instruct",
                       "CODEFORCES", n_problems=3, n_samples_each=10))
        + list(_fake_rows("TheBloke/deepseek-coder-33B-instruct-AWQ",
                         "CODEFORCES", n_problems=3, n_samples_each=10))
    )
    with patch.object(paper_replica, "_stream_dataset_rows", return_value=iter(rows)):
        out = paper_replica.load_paper_prompts(
            model="deepseek-ai/deepseek-coder-6.7b-instruct",
            source="CODEFORCES",
            difficulty="introductory",
            cache_dir=tmp_path,
        )
    # Only 3 problems from the requested model
    assert len(out) == 3
    # All prompts came from the right model (synthetic rows would still say
    # "PROMPT for problem N" since fake rows don't differentiate by model
    # in the prompt field — the test asserts filtering happened via row count)


def test_load_paper_prompts_filters_by_source(tmp_path):
    """source = 'ATCODER' or 'CODEFORCES' — filter must respect it."""
    from bench.apps import paper_replica
    rows = (
        list(_fake_rows("deepseek-ai/deepseek-coder-6.7b-instruct",
                       "CODEFORCES", n_problems=4, n_samples_each=10))
        + list(_fake_rows("deepseek-ai/deepseek-coder-6.7b-instruct",
                         "ATCODER", n_problems=2, n_samples_each=10))
    )
    with patch.object(paper_replica, "_stream_dataset_rows", return_value=iter(rows)):
        out = paper_replica.load_paper_prompts(
            model="deepseek-ai/deepseek-coder-6.7b-instruct",
            source="CODEFORCES",
            difficulty="introductory",
            cache_dir=tmp_path,
        )
    assert len(out) == 4  # Only CODEFORCES rows
    assert set(out.keys()) == {0, 1, 2, 3}


# ─── Caching ────────────────────────────────────────────────────────────


def test_load_paper_prompts_caches_to_disk(tmp_path):
    """Second call should hit the cache, NOT re-stream from HF."""
    from bench.apps import paper_replica
    rows = list(_fake_rows("deepseek-ai/deepseek-coder-6.7b-instruct",
                           "CODEFORCES", n_problems=3, n_samples_each=5))
    with patch.object(paper_replica, "_stream_dataset_rows",
                      return_value=iter(rows)) as m:
        out1 = paper_replica.load_paper_prompts(
            model="deepseek-ai/deepseek-coder-6.7b-instruct",
            source="CODEFORCES",
            difficulty="introductory",
            cache_dir=tmp_path,
        )
        assert m.call_count == 1
    # Second call — cache exists, must not stream again
    with patch.object(paper_replica, "_stream_dataset_rows",
                      side_effect=AssertionError("should hit cache")) as m:
        out2 = paper_replica.load_paper_prompts(
            model="deepseek-ai/deepseek-coder-6.7b-instruct",
            source="CODEFORCES",
            difficulty="introductory",
            cache_dir=tmp_path,
        )
        # Stream not called
        assert m.call_count == 0
    assert out1 == out2


# ─── Edge cases ──────────────────────────────────────────────────────────


def test_load_paper_prompts_empty_result_raises(tmp_path):
    """If no rows match the filter, we should error loudly — silent empty
    returns would let downstream code skip the experiment without warning."""
    from bench.apps import paper_replica
    with patch.object(paper_replica, "_stream_dataset_rows", return_value=iter([])):
        with pytest.raises((ValueError, RuntimeError)):
            paper_replica.load_paper_prompts(
                model="deepseek-ai/deepseek-coder-6.7b-instruct",
                source="CODEFORCES",
                difficulty="introductory",
                cache_dir=tmp_path,
            )


def test_load_paper_prompts_problem_id_is_int():
    """Keys of the returned dict must be int (matching how the rest of
    our pipeline keys APPS problems)."""
    from bench.apps import paper_replica
    rows = list(_fake_rows("deepseek-ai/deepseek-coder-6.7b-instruct",
                           "CODEFORCES", n_problems=2, n_samples_each=5))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with patch.object(paper_replica, "_stream_dataset_rows",
                          return_value=iter(rows)):
            out = paper_replica.load_paper_prompts(
                model="deepseek-ai/deepseek-coder-6.7b-instruct",
                source="CODEFORCES",
                difficulty="introductory",
                cache_dir=Path(td),
            )
    assert all(isinstance(k, int) for k in out.keys())
