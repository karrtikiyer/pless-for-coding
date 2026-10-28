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


# ---------------------------------------------------------------- --allow-partial (rows present on BOTH sides only)


def _synthetic_pair(tmp_path, tag, seed):
    """252 ATCODER-side + 748 CF-side synthetic metrics for one config; returns (a_path, c_path)."""
    rng = np.random.default_rng(seed)
    def tasks(ids):
        out = []
        for tid in ids:
            nc = int(rng.integers(0, 11))
            out.append((tid, nc, 0.4 if nc >= 2 else None))
        return out
    a = _metrics(tmp_path / "a" / f"{tag}.json", tasks(range(1, 253)))
    c = _metrics(tmp_path / "c" / f"{tag}.json", tasks(range(1001, 1749)))
    return a, c


def test_common_rows_only_existing_keeps_configs_with_both_files(mod, tmp_path, monkeypatch):
    a_ok = _metrics(tmp_path / "a_ok.json", [(1, 5, 0.4)])
    a_only = _metrics(tmp_path / "a_only.json", [(1, 5, 0.4)])
    c_ok = _metrics(tmp_path / "cf" / "stemA_metrics.json", [(2, 5, 0.4)])
    monkeypatch.setattr(mod, "APPS_RENYI", {"Qwen3-8B": [("A", str(a_ok)), ("B", str(a_only))]})
    monkeypatch.setattr(mod, "_cf_rows", lambda model: [("A", "stemA"), ("B", "stemB"), ("C", "stemC")])
    cf_dir = tmp_path / "cf"
    # default: every label in both config maps (A, B), regardless of file existence
    assert [r[0] for r in mod.common_rows("qwen3", cf_dir)] == ["A", "B"]
    # only_existing: also requires BOTH metrics files on disk (B has no CF file)
    assert [r[0] for r in mod.common_rows("qwen3", cf_dir, only_existing=True)] == ["A"]


def _patch_deepseek_env(mod, tmp_path, monkeypatch, present_cf):
    """Patch the module to a tmp tree: 4 candidate configs, 252-side files for all, CF files for `present_cf`."""
    labels = ["top-p @0.9", "p-less baseline (k=2)", "G_k=0.35", "G_k=1.6"]
    stems = dict(mod.cf.build_rows("deepseek"))
    apps = []
    for i, lab in enumerate(labels):
        a, c = _synthetic_pair(tmp_path, f"cfg{i}", seed=i)
        apps.append((lab, str(a)))
        if lab in present_cf:
            dest = tmp_path / mod.cf.MODELS["deepseek"]["dir"] / "metrics" / f"{stems[lab]}_metrics.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(c.read_text())
    monkeypatch.setattr(mod, "APPS_RENYI", {"DeepSeek-R1-Distill-Llama-8B": apps})
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    ids = tmp_path / "data" / "cf_interview_748_ids.txt"
    ids.parent.mkdir(parents=True, exist_ok=True)
    ids.write_text(" ".join(str(i) for i in range(1001, 1749)))
    return labels


def test_main_allow_partial_end_to_end_lists_pending(mod, tmp_path, monkeypatch):
    _patch_deepseek_env(mod, tmp_path, monkeypatch, present_cf={"top-p @0.9", "p-less baseline (k=2)", "G_k=0.35"})
    out = tmp_path / "t.md"
    mod.main(["--model", "deepseek", "--allow-partial", "--boot", "50", "--out", str(out)])
    text = out.read_text()
    assert "Partial: 3 of 4" in text and "Pending" in text and "G_k=1.6" in text.split("Pending")[1].split("\n")[0]
    pooled = text.split("## Pooled")[1].split("## ATCODER")[0]
    assert "| G_k=0.35 |" in pooled and "| top-p @0.9 |" in pooled and "| G_k=1.6 |" not in pooled
    assert "--allow-partial" in text


def test_main_without_allow_partial_still_fails_on_missing_cf_file(mod, tmp_path, monkeypatch):
    _patch_deepseek_env(mod, tmp_path, monkeypatch, present_cf={"top-p @0.9"})
    with pytest.raises(SystemExit):
        mod.main(["--model", "deepseek", "--boot", "50", "--out", str(tmp_path / "t.md")])


def test_main_allow_partial_with_no_common_configs_fails(mod, tmp_path, monkeypatch):
    _patch_deepseek_env(mod, tmp_path, monkeypatch, present_cf=set())
    with pytest.raises(SystemExit) as e:
        mod.main(["--model", "deepseek", "--allow-partial", "--boot", "50", "--out", str(tmp_path / "t.md")])
    assert "no configs have metrics on both sides" in str(e.value)
