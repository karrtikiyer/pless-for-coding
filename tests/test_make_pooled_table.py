"""Tests for scripts/paper/make_pooled_table.py (APPS-252 + CF-748 pooled decoder table)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "paper" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("make_pooled_table")


def _metrics(path: Path, tasks: list[tuple[int, int, float | None]], n: int = 10):
    """tasks = [(task_id, num_correct, self_codebleu)]; writes a minimal metrics JSON."""
    per_task = []
    for tid, nc, cb in tasks:
        per_task.append({
            "task_id": tid, "num_correct": nc,
            "pass_results": [True] * nc + [False] * (n - nc),
            "self_codebleu": cb,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "num_tasks": len(tasks), "timestamp": "2026-01-01T00:00:00+00:00",
        "per_task": per_task,
    }))
    return path


def test_pooled_is_task_weighted_not_average_of_sources(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 10, 0.5)])                       # 1 task, all correct
    c = _metrics(tmp_path / "c.json", [(2, 0, None), (3, 0, None), (4, 0, None)])
    m = mod.pool_metrics(a, c, expected=(1, 3))
    assert m["num_tasks"] == 4
    assert m["pass_at_k"]["1"] == pytest.approx(0.25)          # simple average of sources would be 0.5
    assert m["cover_at_t"]["0.7"] == pytest.approx(25.0)


def test_pooled_cb_div_is_union_mean_not_mean_of_means(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)])
    c = _metrics(tmp_path / "c.json", [(2, 5, 0.3), (3, 5, 0.3)])
    m = mod.pool_metrics(a, c, expected=(1, 2))
    vals = [t["self_codebleu"] for t in m["per_task"] if t["self_codebleu"] is not None]
    assert sum(vals) / len(vals) == pytest.approx((0.5 + 0.3 + 0.3) / 3)   # not (0.5+0.3)/2


def test_overlapping_task_ids_fail(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)])
    c = _metrics(tmp_path / "c.json", [(1, 5, 0.3)])
    with pytest.raises(SystemExit) as e:
        mod.pool_metrics(a, c, expected=(1, 1))
    assert "overlap" in str(e.value)


def test_wrong_task_count_fails(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)])
    c = _metrics(tmp_path / "c.json", [(2, 5, 0.3)])
    with pytest.raises(SystemExit) as e:
        mod.pool_metrics(a, c, expected=(252, 1))
    assert "252" in str(e.value)


def test_wrong_samples_per_task_fails(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)], n=8)
    c = _metrics(tmp_path / "c.json", [(2, 5, 0.3)])
    with pytest.raises(SystemExit) as e:
        mod.pool_metrics(a, c, expected=(1, 1))
    assert "n=10" in str(e.value) or "10 samples" in str(e.value)


def test_missing_codebleu_cache_fails(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)])
    p = tmp_path / "c.json"
    _metrics(p, [(2, 5, 0.3)])
    d = json.loads(p.read_text())
    for t in d["per_task"]:
        t.pop("self_codebleu")
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        mod.pool_metrics(a, p, expected=(1, 1))
    assert "self_codebleu" in str(e.value)


def test_common_rows_qwen3(mod):
    rows = mod.common_rows("qwen3", REPO_ROOT / "results" / "cf_dir_placeholder")
    labels = [r[0] for r in rows]
    assert labels == [
        "recommended",
        "top-p @0.7", "top-p @0.75", "top-p @0.8", "top-p @0.85", "top-p @0.9",
        "p-less baseline (k=2)",
        "G_k=0.05", "G_k=0.1", "G_k=0.2", "G_k=0.4", "G_k=0.8", "G_k=1.6",
    ]
    assert all("ATCODER_interview" in str(r[1]) for r in rows)
    assert all(str(r[2]).endswith("_metrics.json") for r in rows)


def test_common_rows_drops_labels_missing_on_one_side(mod, monkeypatch):
    cf = _load("make_cf748_table")
    monkeypatch.setattr(mod, "_cf_rows", lambda model: [r for r in cf.build_rows(model) if r[0] != "G_k=0.2"])
    labels = [r[0] for r in mod.common_rows("qwen3", Path("x"))]
    assert "G_k=0.2" not in labels and "G_k=0.1" in labels


def test_render_has_three_sections_and_paper_columns(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 10, 0.5), (2, 4, 0.4)])
    c = _metrics(tmp_path / "c.json", [(3, 0, None), (4, 6, 0.3)])
    out = mod.render([("cfgX", a, c)], expected=(2, 2), workdir=tmp_path / "w")
    assert out.count("| Decoder | pass@1 | pass@3 | pass@5 | pass@10 | cov@0.3 | cov@0.7 | cb_div |") == 3
    assert "Pooled" in out and "ATCODER" in out and "CODEFORCES" in out
    assert "| cfgX |" in out


def test_sort_by_pass_at_1_orders_rows_descending_in_all_sections(mod, tmp_path):
    lo_a = _metrics(tmp_path / "lo_a.json", [(1, 2, 0.5), (2, 2, 0.4)])
    lo_c = _metrics(tmp_path / "lo_c.json", [(3, 2, 0.3), (4, 2, 0.3)])
    hi_a = _metrics(tmp_path / "hi_a.json", [(5, 9, 0.5), (6, 9, 0.4)])
    hi_c = _metrics(tmp_path / "hi_c.json", [(7, 9, 0.3), (8, 9, 0.3)])
    rows = [("low", lo_a, lo_c), ("high", hi_a, hi_c)]
    unsorted = mod.render(rows, expected=(2, 2), workdir=tmp_path / "w1")
    srt = mod.render(rows, expected=(2, 2), workdir=tmp_path / "w2", sort_by="pass@1")
    def order(text):
        return [l.split("|")[1].strip() for l in text.splitlines() if l.startswith("| low") or l.startswith("| high")]
    assert order(unsorted) == ["low", "high"] * 3
    assert order(srt) == ["high", "low"] * 3


def test_sort_by_rejects_unknown_key(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)])
    c = _metrics(tmp_path / "c.json", [(2, 5, 0.3)])
    with pytest.raises(SystemExit):
        mod.render([("x", a, c)], expected=(1, 1), workdir=tmp_path / "w", sort_by="nonsense")


import numpy as np


def test_bootstrap_all_pass_vs_all_fail(mod):
    a = np.ones((4, 10), dtype=bool)
    b = np.zeros((4, 10), dtype=bool)
    r = mod.paired_bootstrap(a, b, np.zeros(4, dtype=int), boot=200, seed=0)
    assert r["dp1"] == pytest.approx(1.0) and r["dp1_lo"] == pytest.approx(1.0) and r["dp1_hi"] == pytest.approx(1.0)
    assert r["dp10"] == pytest.approx(1.0)


def test_bootstrap_identical_configs_ci_contains_zero(mod):
    rng = np.random.default_rng(1)
    x = rng.random((60, 10)) < 0.5
    r = mod.paired_bootstrap(x, x.copy(), np.zeros(60, dtype=int), boot=400, seed=0)
    assert r["dp1"] == pytest.approx(0.0)
    assert r["dp1_lo"] <= 0.0 <= r["dp1_hi"]


def test_bootstrap_is_deterministic_for_a_seed(mod):
    rng = np.random.default_rng(2)
    a, b = rng.random((50, 10)) < 0.6, rng.random((50, 10)) < 0.5
    s = np.zeros(50, dtype=int)
    assert mod.paired_bootstrap(a, b, s, boot=300, seed=7) == mod.paired_bootstrap(a, b, s, boot=300, seed=7)


def test_bootstrap_is_stratified_by_source(mod):
    # stratum 0: 1 task, A passes all / B fails all  -> delta 1.0
    # stratum 1: 3 tasks identical                    -> delta 0.0
    a = np.array([[1] * 10, [1] * 10, [0] * 10, [1] * 10], dtype=bool)
    b = np.array([[0] * 10, [1] * 10, [0] * 10, [1] * 10], dtype=bool)
    strata = np.array([0, 1, 1, 1])
    r = mod.paired_bootstrap(a, b, strata, boot=300, seed=0)
    # stratified resampling always draws exactly 1 task from stratum 0 and 3 from stratum 1
    assert r["dp1_lo"] == pytest.approx(0.25) and r["dp1_hi"] == pytest.approx(0.25)


def test_pass_matrix_aligns_by_source_and_task_id(mod, tmp_path):
    a1 = _metrics(tmp_path / "a1.json", [(1, 10, 0.5), (2, 0, None)])
    c1 = _metrics(tmp_path / "c1.json", [(3, 5, 0.3), (4, 5, 0.3)])
    m1 = mod.pool_metrics(a1, c1, expected=(2, 2))
    m2 = dict(m1, per_task=list(reversed(m1["per_task"])))
    x1, s1 = mod.pass_matrix(m1)
    x2, s2 = mod.pass_matrix(m2)
    assert (x1 == x2).all() and (s1 == s2).all()


def test_render_adds_paired_difference_sections(mod, tmp_path):
    a1 = _metrics(tmp_path / "a1.json", [(1, 10, 0.5), (2, 9, 0.4)])
    c1 = _metrics(tmp_path / "c1.json", [(3, 8, 0.3), (4, 8, 0.3)])
    a2 = _metrics(tmp_path / "a2.json", [(1, 2, 0.5), (2, 1, 0.4)])
    c2 = _metrics(tmp_path / "c2.json", [(3, 2, 0.3), (4, 1, 0.3)])
    rows = [("good", a1, c1), ("bad", a2, c2)]
    out = mod.render(rows, expected=(2, 2), workdir=tmp_path / "w", refs=["bad"], boot=100, seed=0)
    assert "Paired differences vs bad" in out
    sec = out.split("Paired differences vs bad")[1]
    assert "| good |" in sec and "| bad |" not in sec.split("\n## ")[0]


def test_render_unknown_ref_fails(mod, tmp_path):
    a = _metrics(tmp_path / "a.json", [(1, 5, 0.5)])
    c = _metrics(tmp_path / "c.json", [(2, 5, 0.3)])
    with pytest.raises(SystemExit):
        mod.render([("x", a, c)], expected=(1, 1), workdir=tmp_path / "w", refs=["nope"], boot=10)


def test_bootstrap_halfwidth_matches_analytic_se(mod):
    rng = np.random.default_rng(3)
    # two strata, problems with heterogeneous pass rates, n=10 draws
    pa = rng.beta(2, 2, 400)
    pb = np.clip(pa - 0.03 + rng.normal(0, 0.1, 400), 0, 1)
    a = rng.random((400, 10)) < pa[:, None]
    b = rng.random((400, 10)) < pb[:, None]
    strata = np.array([0] * 100 + [1] * 300)
    r = mod.paired_bootstrap(a, b, strata, boot=4000, seed=0)
    half = (r["dp1_hi"] - r["dp1_lo"]) / 2
    assert half == pytest.approx(1.96 * r["se1"], rel=0.06)
    assert r["dp1"] / r["se1"] == pytest.approx(r["z1"])


def test_bootstrap_does_not_resample_draws_so_pass10_is_unbiased(mod):
    # every problem solved by exactly 1 of 10 draws in A, 0 in B -> delta pass@10 is exactly 1.0
    a = np.zeros((50, 10), dtype=bool)
    a[:, 0] = True
    b = np.zeros((50, 10), dtype=bool)
    r = mod.paired_bootstrap(a, b, np.zeros(50, dtype=int), boot=300, seed=0)
    assert r["dp10"] == pytest.approx(1.0)
    assert r["dp10_lo"] == pytest.approx(1.0) and r["dp10_hi"] == pytest.approx(1.0)


def test_interaction_is_difference_of_source_deltas(mod):
    a = np.array([[1] * 10, [1] * 10, [0] * 10, [0] * 10], dtype=bool)
    b = np.array([[0] * 10, [0] * 10, [0] * 10, [0] * 10], dtype=bool)
    r = mod.paired_bootstrap(a, b, np.array([0, 0, 1, 1]), boot=100, seed=0)
    assert r["int"] == pytest.approx(1.0 - 0.0)


def test_holm_adjust(mod):
    assert mod.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mod.holm_adjust([0.5]) == pytest.approx([0.5])


def test_render_paired_section_has_holm_interaction_and_caveats(mod, tmp_path):
    rows = []
    for i, (p_a, p_c) in enumerate([(9, 8), (2, 2), (5, 5)]):
        a = _metrics(tmp_path / f"a{i}.json", [(1, p_a, 0.5), (2, p_a, 0.4), (3, p_a, 0.4)])
        c = _metrics(tmp_path / f"c{i}.json", [(11, p_c, 0.3), (12, p_c, 0.3), (13, p_c, 0.3)])
        rows.append((f"cfg{i}", a, c))
    out = mod.render(rows, expected=(3, 3), workdir=tmp_path / "w", refs=["cfg2"], boot=200, seed=0)
    sec = out.split("Paired differences vs cfg2")[1]
    assert "Holm" in sec and "ATCODER−CODEFORCES" in sec
    assert "unadjusted" in sec and "problem-level" in sec
    assert "two-level" not in sec
