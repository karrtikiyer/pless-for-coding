"""Tests for the self-CodeBLEU write-through cache (bench.eval.metrics.add_self_codebleu_cached).

The cache persists per-task self_codebleu into the metrics JSON so expensive all-pairs
CodeBLEU is computed once. Cache key = presence of the "self_codebleu" key on the first
per-task result. Cache hit ⇒ no recompute and records untouched.
"""
from __future__ import annotations

import json

import bench.eval.metrics as M
from bench.eval.metrics import add_self_codebleu_cached, compute_self_codebleu_diversity


def _write_metrics(tmp_path, per_task):
    m = {"dataset": "apps", "num_tasks": len(per_task), "per_task": per_task}
    p = tmp_path / "m_metrics.json"
    p.write_text(json.dumps(m))
    return str(p), m


def test_cache_miss_computes_and_persists(tmp_path, monkeypatch):
    """First call: computes (stubbed), returns True, and writes self_codebleu to the file."""
    calls = {"n": 0}

    def stub(task_results, records):
        calls["n"] += 1
        for t in task_results:
            t["self_codebleu"] = 0.42  # pretend-computed
    monkeypatch.setattr(M, "add_self_codebleu", stub)

    pt = [{"task_id": 1, "pass_results": [True, True], "num_correct": 2}]
    path, m = _write_metrics(tmp_path, pt)
    wrote = add_self_codebleu_cached(pt, records=[{"task_id": 1, "samples": ["a", "b"]}],
                                     metrics_path=path, full_metrics=m)
    assert wrote is True
    assert calls["n"] == 1
    assert pt[0]["self_codebleu"] == 0.42
    on_disk = json.loads(open(path).read())
    assert on_disk["per_task"][0]["self_codebleu"] == 0.42  # persisted


def test_cache_hit_is_noop_and_ignores_records(tmp_path, monkeypatch):
    """Second call (key already present): returns False, does NOT recompute, ignores records."""
    calls = {"n": 0}

    def stub(task_results, records):
        calls["n"] += 1
    monkeypatch.setattr(M, "add_self_codebleu", stub)

    pt = [{"task_id": 1, "pass_results": [True, True], "num_correct": 2, "self_codebleu": 0.7}]
    path, m = _write_metrics(tmp_path, pt)
    # records deliberately empty/garbage — a cache hit must not touch them
    wrote = add_self_codebleu_cached(pt, records=[], metrics_path=path, full_metrics=m)
    assert wrote is False
    assert calls["n"] == 0
    assert pt[0]["self_codebleu"] == 0.7


def test_real_codebleu_roundtrip_and_second_call_cached(tmp_path):
    """End-to-end with the real CodeBLEU path: first call persists a numeric cb_div;
    second call is a cache hit that reproduces the aggregate without recompute."""
    pt = [{"task_id": 1, "pass_results": [True, True], "num_correct": 2}]
    records = [{"task_id": 1, "samples": ["def f(x):\n    return x + 1\n",
                                          "def f(x):\n    y = x\n    return y + 2\n"]}]
    path, m = _write_metrics(tmp_path, pt)

    wrote1 = add_self_codebleu_cached(pt, records, path, full_metrics=m)
    assert wrote1 is True
    cb1 = compute_self_codebleu_diversity(pt)["codebleu_diversity"]
    assert isinstance(pt[0]["self_codebleu"], float)
    assert 0.0 <= cb1 <= 1.0

    # reload fresh from disk (simulating a later run) → cache hit, records not needed
    m2 = json.loads(open(path).read())
    pt2 = m2["per_task"]
    assert "self_codebleu" in pt2[0]  # persisted
    wrote2 = add_self_codebleu_cached(pt2, records=[], metrics_path=path, full_metrics=m2)
    assert wrote2 is False
    cb2 = compute_self_codebleu_diversity(pt2)["codebleu_diversity"]
    assert cb2 == cb1  # identical, from cache
