"""Tests for scripts/paper/analyze_pooled_winners.py (joint-bootstrap winners analysis)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "paper" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("analyze_pooled_winners")


def test_lookup_matches_unbiased_estimator_and_closed_form(mod):
    from bench.eval.metrics import estimate_pass_at_k
    for k in (1, 3, 5, 7, 10):
        lut = mod.lookup(f"pass@{k}")
        ref = estimate_pass_at_k(np.full(11, 10), np.arange(11), k)
        assert lut == pytest.approx(ref)
    assert mod.lookup("pass@7")[3] == pytest.approx(1 - 1 / math.comb(10, 7))
    assert list(mod.lookup("cov@0.7")) == [0.0] * 7 + [1.0] * 4          # c >= 7
    assert list(mod.lookup("cov@0.3")) == [0.0] * 3 + [1.0] * 8          # c >= 3


def test_pass_at_k_is_monotone_in_k(mod):
    for c in range(11):
        vals = [mod.lookup(f"pass@{k}")[c] for k in (1, 3, 5, 7, 10)]
        assert vals == sorted(vals)


def test_average_ranks_handles_ties(mod):
    assert list(mod.average_ranks(np.array([0.9, 0.5, 0.9, 0.1]))) == [1.5, 3.0, 1.5, 4.0]


def test_joint_bootstrap_uses_same_problems_for_all_configs(mod):
    # config 1 is exactly 1 - config 0 per task, so replicate means must sum to exactly 1
    rng = np.random.default_rng(0)
    v0 = rng.random(120)
    V = np.vstack([v0, 1 - v0])
    strata = np.array([0] * 30 + [1] * 90)
    R = mod.joint_bootstrap_means(V, strata, boot=200, seed=0)
    assert R.shape == (200, 2)
    assert R.sum(1) == pytest.approx(np.ones(200))


def test_joint_bootstrap_deterministic(mod):
    V = np.random.default_rng(1).random((3, 60))
    s = np.zeros(60, dtype=int)
    assert (mod.joint_bootstrap_means(V, s, 100, 5) == mod.joint_bootstrap_means(V, s, 100, 5)).all()


def test_top_group_single_clear_winner(mod):
    rng = np.random.default_rng(2)
    V = np.vstack([rng.random(400) < 0.8, rng.random(400) < 0.5, rng.random(400) < 0.2]).astype(float)
    obs = V.mean(1)
    R = mod.joint_bootstrap_means(V, np.zeros(400, dtype=int), boot=2000, seed=0)
    group, q = mod.top_group(obs, R)
    assert group == [0]


def test_top_group_includes_statistical_ties(mod):
    rng = np.random.default_rng(3)
    V = np.vstack([rng.random(400) < 0.70, rng.random(400) < 0.70, rng.random(400) < 0.30]).astype(float)
    obs = V.mean(1)
    R = mod.joint_bootstrap_means(V, np.zeros(400, dtype=int), boot=2000, seed=0)
    group, _ = mod.top_group(obs, R)
    assert sorted(group) == [0, 1]


def test_delta_ci_constant_difference(mod):
    v = np.full(50, 0.6)
    r = mod.delta_ci(v, np.full(50, 0.5), np.zeros(50, dtype=int), boot=100, seed=0)
    assert r["d"] == pytest.approx(0.1) and r["lo"] == pytest.approx(0.1) and r["hi"] == pytest.approx(0.1)


def test_delta_ci_zero_difference_has_p_one(mod):
    v = np.random.default_rng(4).random(80)
    r = mod.delta_ci(v, v.copy(), np.zeros(80, dtype=int), boot=100, seed=0)
    assert r["p"] == pytest.approx(1.0)


def test_aligned_vectors_sort_by_source_and_task_id(mod):
    tasks = [
        {"task_id": 5, "source": "CODEFORCES", "num_correct": 4, "self_codebleu": 0.3, "num_distinct_correct": 2},
        {"task_id": 2, "source": "ATCODER", "num_correct": 1, "self_codebleu": None, "num_distinct_correct": 1},
        {"task_id": 1, "source": "ATCODER", "num_correct": 9, "self_codebleu": 0.5, "num_distinct_correct": 3},
    ]
    m1 = {"per_task": tasks}
    m2 = {"per_task": list(reversed(tasks))}
    c1, cb1, nd1, s1 = mod.aligned_vectors(m1)
    c2, cb2, nd2, s2 = mod.aligned_vectors(m2)
    assert list(c1) == [9, 1, 4] and list(s1) == [0, 0, 1]
    assert (c1 == c2).all() and (s1 == s2).all() and (nd1 == nd2).all()
    assert np.isnan(cb1[1]) and cb1[0] == 0.5 and cb1[2] == 0.3
    assert np.array_equal(np.isnan(cb1), np.isnan(cb2))


def test_pairwise_cb_delta_uses_only_tasks_valid_in_both(mod):
    cbA = np.array([0.5, np.nan, 0.4, 0.6])
    cbB = np.array([0.3, 0.2, np.nan, 0.4])
    r = mod.cb_delta(cbA, cbB, np.array([0, 0, 1, 1]), boot=100, seed=0)
    assert r["n"] == 2                                   # tasks 0 and 3
    assert r["d"] == pytest.approx(((0.5 - 0.3) + (0.6 - 0.4)) / 2)


def test_informative_share(mod):
    C = np.array([[0, 5, 10, 3], [0, 6, 10, 3], [0, 5, 10, 4]])   # configs x tasks
    assert mod.informative_share(C) == pytest.approx(0.5)         # tasks 1 and 3 vary


# ---------------------------------------------------------------- model-aware caveats + --allow-partial wiring


def _write_cfg(tmp_path, tag, seed, n_a, n_c):
    import json
    rng = np.random.default_rng(seed)
    def mk(path, ids):
        per = []
        for tid in ids:
            nc = int(rng.integers(0, 11))
            per.append({"task_id": tid, "num_correct": nc, "pass_results": [True] * nc + [False] * (10 - nc),
                        "self_codebleu": 0.4 if nc >= 2 else None, "num_distinct_correct": min(nc, 3)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"num_tasks": len(per), "timestamp": "2026-01-01T00:00:00+00:00", "per_task": per}))
        return path
    return mk(tmp_path / "a" / f"{tag}.json", range(1, n_a + 1)), mk(tmp_path / "c" / f"{tag}.json", range(1001, 1001 + n_c))


def _rows(tmp_path, labels):
    return [(lab, *_write_cfg(tmp_path, f"x{i}", i, 6, 10)) for i, lab in enumerate(labels)]


def test_caveats_use_the_model_title_not_qwen(mod):
    import tempfile
    rows = _rows(Path(tempfile.mkdtemp()), ["top-p @0.9", "G_k=0.35", "p-less baseline (k=2)"])
    out = mod.analyze(rows, expected=(6, 10), boot=50, seed=0, model_title="DeepSeek-R1-Distill-Llama-8B")
    assert "DeepSeek-R1-Distill-Llama-8B, thinking-ON" in out
    assert "Qwen3-8B" not in out
    assert "recommended is T0.6/p0.95/k20" not in out      # no `recommended` row => no such sentence


def test_recommended_clause_only_when_a_recommended_row_exists(mod):
    import tempfile
    rows = _rows(Path(tempfile.mkdtemp()), ["recommended", "top-p @0.9", "p-less baseline (k=2)"])
    out = mod.analyze(rows, expected=(6, 10), boot=50, seed=0, model_title="Qwen3-8B")
    assert "recommended" in out.split("## Caveats")[1]
    assert "Qwen3-8B, thinking-ON" in out


def test_main_allow_partial_end_to_end(mod, tmp_path, monkeypatch):
    mp = mod.mp
    stems = dict(mp.cf.build_rows("deepseek"))
    labels = ["top-p @0.9", "p-less baseline (k=2)", "G_k=0.35", "G_k=1.6"]
    apps = []
    for i, lab in enumerate(labels):
        a, c = _write_cfg(tmp_path, f"cfg{i}", i, 252, 748)
        apps.append((lab, str(a)))
        if lab != "G_k=1.6":   # no CF file for the 4th config
            dest = tmp_path / mp.cf.MODELS["deepseek"]["dir"] / "metrics" / f"{stems[lab]}_metrics.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(c.read_text())
    monkeypatch.setattr(mp, "APPS_RENYI", {"DeepSeek-R1-Distill-Llama-8B": apps})
    monkeypatch.setattr(mp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    out = tmp_path / "w.md"
    mod.main(["--model", "deepseek", "--allow-partial", "--boot", "50", "--out", str(out)])
    text = out.read_text()
    assert "Partial: 3 of 4" in text
    assert "G_k=1.6" not in text.split("## Metrics")[1].split("## Ranks")[0]
    with pytest.raises(SystemExit):
        mod.main(["--model", "deepseek", "--boot", "50", "--out", str(tmp_path / "strict.md")])


def test_common_subset_label_uses_the_actual_config_count(mod):
    import tempfile
    rows = _rows(Path(tempfile.mkdtemp()), ["top-p @0.9", "G_k=0.35", "p-less baseline (k=2)"])
    out = mod.analyze(rows, expected=(6, 10), boot=50, seed=0, model_title="X")
    cb = out.split("## cb_div")[1].split("## Information")[0]
    assert "3-way common subset" in cb and "13-way" not in cb
