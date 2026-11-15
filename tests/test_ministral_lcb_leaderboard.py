"""Tests for scripts/ministral_lcb_leaderboard.py (pooled AtCoder+LeetCode leaderboard).

All tests use small SYNTHETIC metrics JSONs written into tmp_path.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "ministral_lcb_leaderboard"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lb = _load()


# ---------------------------------------------------------------- helpers
def _write_metrics(path: Path, correct: list[int], n: int = 10, *,
                   term: float = 80.0, mean_tok: float = 1000.0, timeout: float = 2.0,
                   json_top_p: float = 1.0, json_temp: float = 1.0,
                   task_ids: list[str] | None = None,
                   distinct: list[int] | None = None,
                   cb: list[float | None] | None = None,
                   overrides: dict | None = None) -> Path:
    """Write a minimal metrics JSON whose reported aggregates are consistent with per_task.

    ``distinct`` (default = ``correct``) -> per-task num_distinct_correct; ``cb`` (default 0.5
    for tasks with >=2 correct, else None) -> per-task self_codebleu. Reported cover_at_t /
    cover_at_t_distinct / codebleu_diversity are hand-derived here the way bench.eval defines
    them (fraction of tasks with count >= t*n, in %; round(mean self_codebleu over tasks with
    >=2 correct and a non-null value, 4)). ``overrides`` replaces top-level keys afterwards.
    """
    if task_ids is None:
        task_ids = [f"t{i}" for i in range(len(correct))]
    if distinct is None:
        distinct = list(correct)
    if cb is None:
        cb = [0.5 if c >= 2 else None for c in correct]
    per_task = []
    for tid, c, dc, s in zip(task_ids, correct, distinct, cb, strict=True):
        pr = [True] * c + [False] * (n - c)
        per_task.append({"task_id": tid, "num_correct": c, "pass_results": pr,
                         "num_distinct_correct": dc, "self_codebleu": s})
    nums = np.array([n] * len(correct))
    cs = np.array(correct)
    pk = {str(k): float(lb.estimate_pass_at_k(nums, cs, k).mean()) for k in (1, 3, 5, 8, 10)}
    cov = {str(t): 100 * sum(c >= t * n for c in correct) / len(correct) for t in (0.3, 0.5, 0.7)}
    covd = {str(t): 100 * sum(x >= t * n for x in distinct) / len(correct)
            for t in (0.3, 0.5, 0.7)}
    elig = [s for c, s in zip(correct, cb) if c >= 2 and s is not None]
    cb_div = round(sum(elig) / len(elig), 4) if elig else 0.0
    total = n * len(correct)
    n_timeout = round(timeout / 100 * total)
    d = {
        "model": "synthetic",
        "method": "whatever",
        "temperature": json_temp,  # deliberately unreliable -- must not be used
        "top_p": json_top_p,       # deliberately unreliable -- must not be used
        "dataset": "livecodebench",
        "num_tasks": len(correct),
        "num_samples_per_task": n,
        "pass_at_k": pk,
        "cover_at_t": cov,
        "cover_at_t_distinct": covd,
        "structural_diversity": None,
        "codebleu_diversity": cb_div,
        "termination_pct": term,
        "mean_token_size": mean_tok,
        "timeout_pct": 100 * n_timeout / total,
        "execution_diagnostics": {"by_status": {"Timeout": n_timeout, "Passed": total - n_timeout}},
        "per_task": per_task,
    }
    d.update(overrides or {})
    path.write_text(json.dumps(d))
    return path


def _write_pair(mdir: Path, config: str, atc: list[int], lc: list[int], t: str = "1.0", **kw):
    _write_metrics(mdir / f"{config}_atcoder_think_t{t}_metrics.json", atc, **kw)
    _write_metrics(mdir / f"{config}_leetcode_think_t{t}_metrics.json", lc, **kw)


def _run(mdir: Path, out: Path, n_atc: int, n_lc: int, *extra: str):
    """Call main() with the expected per-platform task counts of the synthetic data."""
    return lb.main(["--metrics-dir", str(mdir), "--out-dir", str(out),
                    "--expected-atcoder-tasks", str(n_atc),
                    "--expected-leetcode-tasks", str(n_lc), *extra])


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    "fname, expected",
    [
        ("gk_0_05_atcoder_think_t1.0_metrics.json", ("gk_0_05", "atcoder", 1.0)),
        ("top_p_0_85_leetcode_think_t1.0_metrics.json", ("top_p_0_85", "leetcode", 1.0)),
        ("ministral_recipe_atcoder_think_t0.7_metrics.json", ("ministral_recipe", "atcoder", 0.7)),
        ("pless_effk_nofloor_leetcode_think_t1.0_metrics.json",
         ("pless_effk_nofloor", "leetcode", 1.0)),
        ("top_nsigma_1_5_atcoder_think_t1.0_metrics.json", ("top_nsigma_1_5", "atcoder", 1.0)),
    ],
)
def test_parse_filename(fname, expected):
    assert lb.parse_metrics_filename(fname) == expected


def test_parse_filename_rejects_garbage():
    with pytest.raises(ValueError):
        lb.parse_metrics_filename("gk_0_05_codeforces_think_t1.0_metrics.json")
    with pytest.raises(ValueError):
        lb.parse_metrics_filename("notes.json")


def test_temperature_comes_from_filename_not_json(tmp_path):
    _write_pair(tmp_path, "ministral_recipe", [5, 5], [5], t="0.7", json_temp=1.0)
    _write_pair(tmp_path, "top_p_0_95", [5, 5], [5], json_top_p=1.0, json_temp=0.3)
    rows = {r["config"]: r for r in lb.build_rows(tmp_path)}
    assert rows["ministral_recipe"]["temperature"] == 0.7
    assert rows["top_p_0_95"]["temperature"] == 1.0


# ---------------------------------------------------------------- family
@pytest.mark.parametrize(
    "config, family",
    [
        ("gk_0_05", "G_k (Renyi)"),
        ("gk_1_6", "G_k (Renyi)"),
        ("pless_default", "p-less variants"),
        ("pless_effk_nofloor", "p-less variants"),
        ("top_p_0_7", "top-p"),
        ("top_p_0_95", "top-p"),
        ("top_nsigma_1_5", "top-nsigma"),
        ("ministral_recipe", "Ministral recipe (T=0.7, top-p 0.95)"),
    ],
)
def test_family(config, family):
    assert lb.family_for(config) == family


def test_family_unknown_raises():
    with pytest.raises(ValueError):
        lb.family_for("beam_search")


# ---------------------------------------------------------------- pooling
def test_pooled_pass1_is_task_weighted_mean(tmp_path):
    atc = [10, 0, 5]          # a1 = 0.5   (3 tasks)
    lc = [2, 4]               # l1 = 0.3   (2 tasks)
    _write_pair(tmp_path, "gk_0_1", atc, lc)
    (row,) = lb.build_rows(tmp_path)
    a1, l1 = 0.5, 0.3
    assert row["atcoder_pass@1"] == pytest.approx(a1, abs=1e-12)
    assert row["leetcode_pass@1"] == pytest.approx(l1, abs=1e-12)
    assert row["pooled_pass@1"] == pytest.approx((3 * a1 + 2 * l1) / 5, abs=1e-12)
    assert row["N"] == 5
    for k in (3, 5, 8, 10):
        assert f"pooled_pass@{k}" in row


def test_pooled_sample_weighted_aux_metrics(tmp_path):
    _write_metrics(tmp_path / "gk_0_1_atcoder_think_t1.0_metrics.json", [1, 1, 1],
                   term=90.0, mean_tok=3000.0, timeout=10.0)
    _write_metrics(tmp_path / "gk_0_1_leetcode_think_t1.0_metrics.json", [1],
                   term=50.0, mean_tok=1000.0, timeout=0.0)
    (row,) = lb.build_rows(tmp_path)
    # 30 vs 10 samples
    assert row["pooled_term%"] == pytest.approx((30 * 90 + 10 * 50) / 40)
    assert row["pooled_mean_tok"] == pytest.approx((30 * 3000 + 10 * 1000) / 40)
    assert row["pooled_timeout%"] == pytest.approx((30 * 10 + 10 * 0) / 40)


def test_pooling_assertion_fires_on_inconsistent_file(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    p = tmp_path / "gk_0_1_atcoder_think_t1.0_metrics.json"
    d = json.loads(p.read_text())
    d["pass_at_k"]["1"] = 0.9  # reported value disagrees with per_task
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="reported pass@1"):
        lb.build_rows(tmp_path)


def test_missing_platform_is_clear_error(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5], [5])
    _write_metrics(tmp_path / "top_nsigma_atcoder_think_t1.0_metrics.json", [3])
    with pytest.raises(ValueError, match="top_nsigma.*leetcode"):
        lb.build_rows(tmp_path)


# ---------------------------------------------------------------- ranking
def test_ranking_with_tie_and_tiebreak(tmp_path):
    # pooled pass@1: A=0.5, B=0.5 (tie), C=0.7, D=0.1
    # B has higher pass@10 than A -> B before A.
    _write_pair(tmp_path, "gk_0_2", [5, 5], [5])          # pass@10 = 1.0, pass@1 = .5
    _write_pair(tmp_path, "gk_0_1", [10, 0], [5])         # pass@10 = 2/3, pass@1 = .5
    _write_pair(tmp_path, "top_p_0_9", [7, 7], [7])
    _write_pair(tmp_path, "pless_gini", [1, 1], [1])
    ranked = lb.rank_rows(lb.build_rows(tmp_path))
    assert [r["config"] for r in ranked] == ["top_p_0_9", "gk_0_2", "gk_0_1", "pless_gini"]
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]
    ties = lb.displayed_pass1_ties(ranked)
    assert ties == [["gk_0_2", "gk_0_1"]]


def test_float_noise_tie_uses_tiebreak_not_ulp():
    # Exact ties computed via different summation orders can differ by 1 ulp; the ulp must
    # not decide the order -- pass@10 must.
    a = {"config": "a_cfg", "pooled_pass@1": 0.6757142857142857, "pooled_pass@10": 0.80}
    b = {"config": "b_cfg", "pooled_pass@1": 0.6757142857142856, "pooled_pass@10": 0.90}
    assert [r["config"] for r in lb.rank_rows([a, b])] == ["b_cfg", "a_cfg"]


def test_full_tie_falls_back_to_name(tmp_path):
    _write_pair(tmp_path, "gk_0_4", [5, 5], [5])
    _write_pair(tmp_path, "gk_0_2", [5, 5], [5])
    ranked = lb.rank_rows(lb.build_rows(tmp_path))
    assert [r["config"] for r in ranked] == ["gk_0_2", "gk_0_4"]


def test_tie_footnote_in_markdown(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "gk_0_2", [5, 5], [5])
    _write_pair(mdir, "gk_0_1", [10, 0], [5])
    _run(mdir, out, 2, 1)
    md = (out / lb.MD_NAME).read_text()
    assert "Tie on displayed pooled pass@1: gk_0_2, gk_0_1" in md
    assert "pass@1 (pooled, N=3)" in md
    assert "over all 3 tasks (AtCoder 2 + LeetCode 1)" in md


def test_no_tie_footnote_when_no_ties(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "gk_0_2", [5, 5], [5])
    _write_pair(mdir, "gk_0_1", [9, 9], [9])
    _run(mdir, out, 2, 1)
    md = (out / lb.MD_NAME).read_text()
    assert "Tie on displayed" not in md
    assert "No ties on displayed pooled pass@1." in md


# ---------------------------------------------------------------- dagger / labels
def test_dagger_on_top_p_0_7(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "top_p_0_7", [5, 5], [5])
    _write_pair(mdir, "top_p_0_75", [6, 6], [6])
    _write_pair(mdir, "ministral_recipe", [4, 4], [4], t="0.7")
    _run(mdir, out, 2, 1)
    md = (out / lb.MD_NAME).read_text()
    assert "top_p_0_7†" in md
    assert "top_p_0_75†" not in md
    assert lb.config_label("top_p_0_7") == "top_p_0_7†"
    assert lb.config_label("top_p_0_75") == "top_p_0_75"
    # T column flags ministral_recipe's T=0.7
    rec_line = next(line for line in md.splitlines()
                    if line.startswith("|") and "| ministral_recipe |" in line)
    assert "0.7" in rec_line


DAGGER_TABLE_NOTE = (
    "(† in the pooled and AtCoder tables marks top_p_0_7, whose AtCoder metrics file "
    "provenance is unverified as noted above; its LeetCode file is not affected and the "
    "LeetCode table carries no †.)")


def test_dagger_placement_and_table_note(tmp_path):
    # top_p_0_7 is NOT rank 1, so it also appears in the paired-difference section.
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "gk_0_1", [9, 9], [9])
    _write_pair(mdir, "top_p_0_7", [5, 5], [5])
    _run(mdir, out, 2, 1)
    lines = (out / lb.MD_NAME).read_text().splitlines()
    md = "\n".join(lines)
    assert "in the AtCoder table marks" not in md
    assert DAGGER_TABLE_NOTE in lines
    assert lines.index(DAGGER_TABLE_NOTE) > lines.index("### LeetCode")

    def table_rows(start: str, stop) -> list[str]:
        i = lines.index(start) if start in lines else \
            next(j for j, x in enumerate(lines) if x.startswith(start))
        j = stop(i)
        return [x for x in lines[i:j] if x.startswith("| ") and "top_p_0_7" in x]

    def next_heading(i):
        return next(j for j in range(i + 1, len(lines)) if lines[j].startswith("#"))

    main_rows = [x for x in lines[:lines.index("## Paired difference vs rank 1")]
                 if x.startswith("| ") and "top_p_0_7" in x]
    sections = {
        "main": main_rows,
        "paired": table_rows("## Paired difference vs rank 1", next_heading),
        "pooled": table_rows("### Pooled (AtCoder + LeetCode", next_heading),
        "atcoder": table_rows("### AtCoder", next_heading),
        "leetcode": table_rows("### LeetCode", lambda i: lines.index(DAGGER_TABLE_NOTE)),
    }
    for name in ("main", "paired", "pooled", "atcoder"):
        assert len(sections[name]) == 1, name
        assert "| top_p_0_7† |" in sections[name][0], name
    assert len(sections["leetcode"]) == 1
    assert "| top_p_0_7 |" in sections["leetcode"][0] and "†" not in sections["leetcode"][0]
    # no † on any LeetCode-table line at all, and no other config gets a †
    lc_i = lines.index("### LeetCode")
    assert not any("†" in x for x in lines[lc_i:lines.index(DAGGER_TABLE_NOTE)])
    assert not any("gk_0_1†" in x for x in lines)


# ---------------------------------------------------------------- outputs
def test_writes_md_and_csv_and_refuses_overwrite(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "gk_0_1", [5, 5], [5])
    _write_pair(mdir, "top_p_0_9", [7, 7], [7])
    _run(mdir, out, 2, 1)
    md_path, csv_path = out / lb.MD_NAME, out / lb.CSV_NAME
    assert md_path.exists() and csv_path.exists()
    first = md_path.read_text()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["config"] for r in rows] == ["top_p_0_9", "gk_0_1"]
    for col in ("pooled_pass@1", "pooled_pass@3", "pooled_pass@5", "pooled_pass@8",
                "pooled_pass@10", "atcoder_pass@1", "leetcode_pass@1", "ci_lo", "ci_hi"):
        assert col in rows[0]
    # canonical per-platform tables present
    assert "| N | pass@1 | pass@3 |" in first
    assert "struct_div" in first

    md_path.write_text("SENTINEL")
    with pytest.raises(SystemExit):
        _run(mdir, out, 2, 1)
    assert md_path.read_text() == "SENTINEL"
    _run(mdir, out, 2, 1, "--overwrite")
    assert md_path.read_text() == first


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_ci_deterministic():
    vec = np.random.default_rng(0).random(50)
    a = lb.bootstrap_ci(vec, n_boot=2000, seed=42)
    b = lb.bootstrap_ci(vec, n_boot=2000, seed=42)
    c = lb.bootstrap_ci(vec, n_boot=2000, seed=7)
    assert a == b
    assert a != c
    lo, hi = a
    assert lo <= vec.mean() <= hi


def test_bootstrap_ci_degenerate():
    lo, hi = lb.bootstrap_ci(np.full(20, 0.3), n_boot=200, seed=42)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


# ================================================================ review follow-ups (2026-11-08)
# ---------------------------------------------------------------- item 1: sample weighting
def test_pooled_aux_metrics_sample_weighted_with_unequal_n(tmp_path):
    # AtCoder: 3 tasks x n=10 = 30 samples; LeetCode: 2 tasks x n=4 = 8 samples.
    # Task weighting (3:2) and sample weighting (30:8) differ here.
    _write_metrics(tmp_path / "gk_0_1_atcoder_think_t1.0_metrics.json", [1, 5, 9], n=10,
                   term=90.0, mean_tok=3000.0, timeout=10.0)   # 3/30 timeouts -> 10.0
    _write_metrics(tmp_path / "gk_0_1_leetcode_think_t1.0_metrics.json", [1, 3], n=4,
                   term=50.0, mean_tok=1000.0, timeout=25.0)   # 2/8 timeouts -> 25.0
    (row,) = lb.build_rows(tmp_path)
    assert row["samples"] == 38
    assert row["N"] == 5
    assert row["pooled_term%"] == pytest.approx(3100 / 38, abs=1e-12)       # != 74.0
    assert row["pooled_mean_tok"] == pytest.approx(98000 / 38, abs=1e-9)    # != 2200.0
    assert row["pooled_timeout%"] == pytest.approx(500 / 38, abs=1e-12)     # != 16.0


# ---------------------------------------------------------------- item 3: pooled pass@k values
def test_pooled_pass_at_k_hand_computed(tmp_path):
    # n=10 everywhere. Per-task unbiased pass@k = 1 - C(n-c, k) / C(n, k):
    #   c=1: p@1=.1  p@3=1-84/120   p@5=1-126/252  p@8=1-9/45   p@10=1
    #   c=2: p@1=.2  p@3=1-56/120   p@5=1-56/252   p@8=1-1/45   p@10=1
    #   c=0: all 0;  c=10: all 1
    _write_pair(tmp_path, "gk_0_1", [1, 2, 0], [10])
    (row,) = lb.build_rows(tmp_path)
    exp = {
        1: (0.1 + 0.2 + 0 + 1) / 4,
        3: ((1 - 84 / 120) + (1 - 56 / 120) + 0 + 1) / 4,
        5: ((1 - 126 / 252) + (1 - 56 / 252) + 0 + 1) / 4,
        8: ((1 - 9 / 45) + (1 - 1 / 45) + 0 + 1) / 4,
        10: (1 + 1 + 0 + 1) / 4,
    }
    for k, v in exp.items():
        assert row[f"pooled_pass@{k}"] == pytest.approx(v, abs=1e-12), k
    # pass@10 (the tie-break key) must not be confused with pass@8
    assert row["pooled_pass@10"] == 0.75
    assert row["pooled_pass@8"] != pytest.approx(row["pooled_pass@10"])


# ---------------------------------------------------------------- item 2: bootstrap
_ATC = [10, 0, 5, 3, 7, 9, 1]
_LC = [2, 8, 6, 4, 10]


def _indep_idx(n_tasks: int) -> np.ndarray:
    return np.random.default_rng(42).integers(0, n_tasks, size=(2000, n_tasks))


def _indep_ci(vec: np.ndarray, idx: np.ndarray, q=(2.5, 97.5)) -> tuple[float, float]:
    means = np.array([vec[row].sum() / len(vec) for row in idx])  # one resample per idx ROW
    lo, hi = np.percentile(means, list(q))
    return float(lo), float(hi)


def test_bootstrap_constants():
    assert lb.N_BOOT == 2000
    assert lb.SEED == 42


def test_bootstrap_ci_matches_independent_per_row_recompute(tmp_path):
    _write_pair(tmp_path, "gk_0_1", _ATC, _LC)
    (row,) = lb.build_rows(tmp_path)
    vec = np.array(_ATC + _LC) / 10
    lo, hi = _indep_ci(vec, _indep_idx(len(vec)))
    assert row["ci_lo"] == pytest.approx(lo, abs=1e-12)
    assert row["ci_hi"] == pytest.approx(hi, abs=1e-12)


def test_bootstrap_ci_rejects_swapped_axis_and_other_level(tmp_path):
    _write_pair(tmp_path, "gk_0_1", _ATC, _LC)
    (row,) = lb.build_rows(tmp_path)
    vec = np.array(_ATC + _LC) / 10
    idx = _indep_idx(len(vec))
    got = (row["ci_lo"], row["ci_hi"])
    swapped = tuple(np.percentile(vec[idx].mean(axis=0), [2.5, 97.5]))
    level80 = _indep_ci(vec, idx, q=(10, 90))
    # the synthetic data must discriminate between the alternatives ...
    assert swapped != pytest.approx(got, abs=1e-6)
    assert level80 != pytest.approx(got, abs=1e-6)
    # ... and the function must produce the 95%, axis=1 (per-resample) answer
    assert got == pytest.approx(_indep_ci(vec, idx), abs=1e-12)


def test_all_configs_share_identical_resample_index_matrix(tmp_path):
    _write_pair(tmp_path, "gk_0_1", _ATC, _LC)
    _write_pair(tmp_path, "top_p_0_9", _LC + _ATC[:2], _ATC[2:])
    rows = lb.build_rows(tmp_path)
    assert len(rows) == 2
    a, b = (r["_boot_idx"] for r in rows)
    assert a.shape == (2000, 12)
    assert np.array_equal(a, b)
    assert np.array_equal(a, _indep_idx(12))


def test_identical_vectors_get_identical_ci_across_configs(tmp_path):
    _write_pair(tmp_path, "gk_0_1", _ATC, _LC)
    _write_pair(tmp_path, "gk_0_2", _ATC, _LC)
    r1, r2 = lb.build_rows(tmp_path)
    assert (r1["ci_lo"], r1["ci_hi"]) == (r2["ci_lo"], r2["ci_hi"])


# ---------------------------------------------------------------- item 4: input checks
def test_script_has_no_bare_asserts():
    import ast
    tree = ast.parse((REPO_ROOT / "scripts" / "ministral_lcb_leaderboard.py").read_text())
    assert not [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]


def test_default_expected_task_counts_are_242_and_178():
    args = lb.parse_args([])
    assert args.expected_atcoder_tasks == 242
    assert args.expected_leetcode_tasks == 178


def test_main_rejects_wrong_task_count_by_default(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "gk_0_1", [5, 5], [5])
    with pytest.raises(ValueError, match="atcoder.*242"):
        lb.main(["--metrics-dir", str(mdir), "--out-dir", str(out)])
    assert not out.exists()


def test_build_rows_rejects_wrong_task_count(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    with pytest.raises(ValueError, match="leetcode.*expected 2"):
        lb.build_rows(tmp_path, expected_tasks={"atcoder": 2, "leetcode": 2})
    assert len(lb.build_rows(tmp_path, expected_tasks={"atcoder": 2, "leetcode": 1})) == 1


def test_rejects_different_task_id_set_across_configs(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    _write_metrics(tmp_path / "gk_0_2_atcoder_think_t1.0_metrics.json", [5, 5],
                   task_ids=["t0", "t9"])
    _write_metrics(tmp_path / "gk_0_2_leetcode_think_t1.0_metrics.json", [5])
    with pytest.raises(ValueError, match="task_id"):
        lb.build_rows(tmp_path)


def test_rejects_different_task_id_order_across_configs(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    _write_metrics(tmp_path / "gk_0_2_atcoder_think_t1.0_metrics.json", [5, 5],
                   task_ids=["t1", "t0"])
    _write_metrics(tmp_path / "gk_0_2_leetcode_think_t1.0_metrics.json", [5])
    with pytest.raises(ValueError, match="task_id order"):
        lb.build_rows(tmp_path)


def test_rejects_non_uniform_n_within_file(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    p = tmp_path / "gk_0_1_atcoder_think_t1.0_metrics.json"
    d = json.loads(p.read_text())
    d["per_task"][1]["pass_results"] = [True, True, False, False]
    d["per_task"][1]["num_correct"] = 2
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="uniform"):
        lb.build_rows(tmp_path)


def test_rejects_duplicate_task_ids(tmp_path):
    _write_metrics(tmp_path / "gk_0_1_atcoder_think_t1.0_metrics.json", [5, 5],
                   task_ids=["t0", "t0"])
    _write_metrics(tmp_path / "gk_0_1_leetcode_think_t1.0_metrics.json", [5])
    with pytest.raises(ValueError, match="duplicate task_id"):
        lb.build_rows(tmp_path)


def test_rejects_num_correct_mismatch(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    p = tmp_path / "gk_0_1_leetcode_think_t1.0_metrics.json"
    d = json.loads(p.read_text())
    d["per_task"][0]["num_correct"] = 6
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="num_correct"):
        lb.build_rows(tmp_path)


def test_rejects_num_tasks_mismatch(tmp_path):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    p = tmp_path / "gk_0_1_leetcode_think_t1.0_metrics.json"
    d = json.loads(p.read_text())
    d["num_tasks"] = 3
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="num_tasks"):
        lb.build_rows(tmp_path)


@pytest.mark.parametrize("field", ["termination_pct", "mean_token_size", "timeout_pct"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_rejects_non_finite_aux_metrics(tmp_path, field, bad):
    _write_pair(tmp_path, "gk_0_1", [5, 5], [5])
    p = tmp_path / "gk_0_1_atcoder_think_t1.0_metrics.json"
    d = json.loads(p.read_text())
    d[field] = bad
    p.write_text(json.dumps(d))  # json writes NaN / Infinity, json.loads reads them back
    with pytest.raises(ValueError, match=field):
        lb.build_rows(tmp_path)


# ---------------------------------------------------------------- item 5: overwrite guard
def test_refuses_when_only_csv_exists(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    out.mkdir()
    _write_pair(mdir, "gk_0_1", [5, 5], [5])
    (out / lb.CSV_NAME).write_text("CSV_SENTINEL")
    with pytest.raises(SystemExit):
        _run(mdir, out, 2, 1)
    assert (out / lb.CSV_NAME).read_text() == "CSV_SENTINEL"
    assert not (out / lb.MD_NAME).exists()
    _run(mdir, out, 2, 1, "--overwrite")
    assert (out / lb.CSV_NAME).read_text() != "CSV_SENTINEL"


# ---------------------------------------------------------------- items 6/8/9: md wording
def _md_for(tmp_path, pairs, n_atc, n_lc):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    for cfg, atc, lc in pairs:
        _write_pair(mdir, cfg, atc, lc)
    ranked = _run(mdir, out, n_atc, n_lc)
    return (out / lb.MD_NAME).read_text(), out, ranked


def test_caveat_points_to_paired_bootstrap(tmp_path):
    md, _, _ = _md_for(tmp_path, [("gk_0_1", [5, 5], [5]), ("gk_0_2", [9, 9], [9])], 2, 1)
    assert "differences smaller than the CI widths" not in md
    assert ("per-config 95% CIs describe each config alone; differences between configs "
            "must be judged with the PAIRED task-bootstrap below") in md


def test_provenance_line_names_script_tracking_and_head(tmp_path):
    md, _, _ = _md_for(tmp_path, [("gk_0_1", [5, 5], [5])], 2, 1)
    assert "at git commit" not in md
    status = lb._git_tracked_status(lb.SCRIPT_REL)
    assert lb.SCRIPT_REL == "scripts/ministral_lcb_leaderboard.py"
    line = next(x for x in md.splitlines() if x.startswith("Generated by"))
    assert f"`{lb.SCRIPT_REL}`" in line
    assert f"script git status at generation: {status}" in line
    assert f"repo HEAD at generation: {lb._git_commit()}" in line


def test_git_tracked_status():
    assert lb._git_tracked_status("pyproject.toml").startswith("tracked")
    assert lb._git_tracked_status("scripts/__no_such_file_for_test__.py") == "untracked"


def test_sampler_params_footnote_after_t_footnote(tmp_path):
    md, _, _ = _md_for(tmp_path, [("gk_0_1", [5, 5], [5]), ("top_p_0_9", [9, 9], [9])], 2, 1)
    lines = md.splitlines()
    i = next(j for j, x in enumerate(lines) if x.startswith("- ‡ "))
    assert lines[i + 1] == (
        "- Sampler parameters such as the top-p cutoff are NOT stored in the run records "
        "(the records' `top_p` field reads 1.0 for all top-p configs); the cutoff values in the "
        "config names come from the runner configuration "
        "(`scripts/run_pless_variants_ministral_3b_lcb_cuda.py`); only T is recorded.")


def test_sampler_footnote_reports_observed_json_top_p(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _write_pair(mdir, "top_p_0_9", [9, 9], [9], json_top_p=0.9)
    _run(mdir, out, 2, 1)
    md = (out / lb.MD_NAME).read_text()
    assert "field reads 0.9 for all top-p configs" in md


# ---------------------------------------------------------------- item 7: paired deltas
# 20 AtCoder + 10 LeetCode tasks, n=10.
#  top_p_0_9 : c=8 everywhere                          -> rank 1, pass@1 0.8
#  gk_0_1    : same but one AtCoder task c=7           -> delta 0.1/30, CI includes 0
#  gk_0_2    : c=2 everywhere                          -> delta 0.6 exactly, CI excludes 0
_A1, _L1 = [8] * 20, [8] * 10
_A2, _L2 = [7] + [8] * 19, [8] * 10
_A3, _L3 = [2] * 20, [2] * 10


def _paired_setup(tmp_path):
    return _md_for(tmp_path, [("top_p_0_9", _A1, _L1), ("gk_0_1", _A2, _L2),
                              ("gk_0_2", _A3, _L3)], 20, 10)


def test_paired_deltas_values_and_ci(tmp_path):
    _, _, ranked = _paired_setup(tmp_path)
    assert [r["config"] for r in ranked] == ["top_p_0_9", "gk_0_1", "gk_0_2"]
    top = np.array(_A1 + _L1) / 10
    idx = _indep_idx(30)
    assert ranked[0]["delta_vs_rank1_pp"] is None
    for r, (a, l) in zip(ranked[1:], [(_A2, _L2), (_A3, _L3)]):
        d = top - np.array(a + l) / 10
        lo, hi = _indep_ci(d, idx)
        assert r["delta_vs_rank1_pp"] == pytest.approx(
            100 * (ranked[0]["pooled_pass@1"] - r["pooled_pass@1"]), abs=1e-12)
        assert r["delta_vs_rank1_pp"] == pytest.approx(100 * d.mean(), abs=1e-9)
        assert r["delta_lo_pp"] == pytest.approx(100 * lo, abs=1e-9)
        assert r["delta_hi_pp"] == pytest.approx(100 * hi, abs=1e-9)
    assert ranked[1]["delta_lo_pp"] == 0.0          # includes 0
    assert ranked[2]["delta_lo_pp"] == pytest.approx(60.0, abs=1e-9)
    assert ranked[2]["delta_hi_pp"] == pytest.approx(60.0, abs=1e-9)


def test_paired_section_in_markdown(tmp_path):
    md, _, _ = _paired_setup(tmp_path)
    lines = md.splitlines()
    i = lines.index("## Paired difference vs rank 1")
    # placed after the leaderboard notes, before the per-platform tables
    assert i < lines.index(lb.CANONICAL_SECTION_HEADING)
    assert i > next(j for j, x in enumerate(lines) if x.startswith("- ‡ "))
    body = "\n".join(lines[i:lines.index(lb.CANONICAL_SECTION_HEADING)])
    d = (np.array(_A1 + _L1) - np.array(_A2 + _L2)) / 10
    lo, hi = _indep_ci(d, _indep_idx(30))
    assert lo == 0.0
    assert f"| 2 | gk_0_1 | 0.33 | 0.00–{100 * hi:.2f} | no |" in body
    assert "| 3 | gk_0_2 | 60.00 | 60.00–60.00 | yes |" in body
    assert ("2 comparisons, no multiplicity correction; borderline cases (a CI bound close "
            "to 0) are not solid.") in body
    assert ("Ranks not distinguishable from rank 1 (paired 95% CI includes 0): "
            "2 (gk_0_1).") in body


def test_paired_sentence_when_all_distinguishable(tmp_path):
    md, _, _ = _md_for(tmp_path, [("top_p_0_9", _A1, _L1), ("gk_0_2", _A3, _L3)], 20, 10)
    assert ("Ranks not distinguishable from rank 1 (paired 95% CI includes 0): none.") in md
    assert "1 comparisons" not in md and "1 comparison, no multiplicity correction" in md


def test_csv_existing_columns_unchanged_and_three_trailing(tmp_path):
    old = (["rank", "config", "family", "temperature", "N", "atcoder_N", "leetcode_N",
            "samples"]
           + [f"pooled_pass@{k}" for k in (1, 3, 5, 8, 10)] + ["ci_lo", "ci_hi"]
           + ["pooled_term%", "pooled_mean_tok", "pooled_timeout%"]
           + [f"{pl}_{m}" for pl in ("atcoder", "leetcode")
              for m in [f"pass@{k}" for k in (1, 3, 5, 8, 10)] + ["term%", "mean_tok",
                                                                     "timeout%"]]
           + ["provenance_flag", "atcoder_file", "leetcode_file"])
    assert list(lb.CSV_FIELDS) == old + ["delta_vs_rank1_pp", "delta_lo_pp", "delta_hi_pp"]
    _, out, ranked = _paired_setup(tmp_path)
    with (out / lb.CSV_NAME).open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["delta_vs_rank1_pp"] == rows[0]["delta_lo_pp"] == ""
    assert float(rows[2]["delta_vs_rank1_pp"]) == ranked[2]["delta_vs_rank1_pp"]


def test_tiebreak_uses_pass10_not_pass8(tmp_path):
    # n=20, equal pass@1 (8/60). gk_0_2 tasks (1,1,6): pass@10 .6649 > .6378 but
    # pass@8 .5921 < .5986 vs gk_0_1 tasks (0,4,4). Name order alone would put gk_0_1 first.
    _write_pair(tmp_path, "gk_0_2", [1, 1], [6], n=20)
    _write_pair(tmp_path, "gk_0_1", [0, 4], [4], n=20)
    rows = {r["config"]: r for r in lb.build_rows(tmp_path)}
    a, b = rows["gk_0_2"], rows["gk_0_1"]
    assert a["pooled_pass@1"] == pytest.approx(b["pooled_pass@1"], abs=1e-15)
    assert a["pooled_pass@10"] > b["pooled_pass@10"] and a["pooled_pass@8"] < b["pooled_pass@8"]
    assert [r["config"] for r in lb.rank_rows(list(rows.values()))] == ["gk_0_2", "gk_0_1"]


# ================================================================ pooled canonical table (2026-11-08)
# Two platforms with DIFFERENT task counts and DIFFERENT n, so task weighting (5:3),
# sample weighting (50:12) and cb-eligible-task weighting (4:2) all give different numbers.
#   AtCoder : n=10, c=[10,7,5,3,0], distinct=[1,7,2,3,0], self_cb=[0.0,0.6,0.5,0.3,0.95]
#             term 90, mean_tok 3000, timeout 10% (5/50 samples)
#   LeetCode: n=4,  c=[4,2,1],      distinct=[4,1,1],     self_cb=[0.8,0.0,0.9]
#             term 50, mean_tok 1000, timeout 25% (3/12 samples)
# The c=0 AtCoder task (self_cb 0.95) and the c=1 LeetCode task (self_cb 0.9) carry NON-null
# self_codebleu on purpose: they are INELIGIBLE for cb_div (bench.eval requires >=2 correct),
# so dropping the `num_correct >= 2` filter in the pooling must change the result.
_PA = dict(correct=[10, 7, 5, 3, 0], n=10, distinct=[1, 7, 2, 3, 0],
           cb=[0.0, 0.6, 0.5, 0.3, 0.95], term=90.0, mean_tok=3000.0, timeout=10.0)
_PL = dict(correct=[4, 2, 1], n=4, distinct=[4, 1, 1], cb=[0.8, 0.0, 0.9],
           term=50.0, mean_tok=1000.0, timeout=25.0)

# Hand-computed pooled values (8 tasks, 62 samples).
_EXP_PASS1 = (1 + .7 + .5 + .3 + 0 + 1 + .5 + .25) / 8                        # 0.53125
# pass@3: AtCoder 1, 1-1/120, 1-10/120, 1-35/120, 0; LeetCode 1, 1 (n-c<k), 1-1/4
_EXP_PASS3 = (1 + 119 / 120 + 110 / 120 + 85 / 120 + 0 + 1 + 1 + 0.75) / 8
# cov@t: task meets t if num_correct >= t*n (its own n). AtCoder thresholds 3/5/7, LeetCode
# 1.2/2/2.8 -> AtCoder 4,3,2 of 5 tasks; LeetCode 2,2,1 of 3 tasks.
_EXP_COV = {"0.3": 100 * 6 / 8, "0.5": 100 * 5 / 8, "0.7": 100 * 3 / 8}
# distinct: AtCoder 2,1,1 of 5; LeetCode 1,1,1 of 3.
_EXP_COVD = {"0.3": 100 * 3 / 8, "0.5": 100 * 2 / 8, "0.7": 100 * 2 / 8}
# cb_div: mean self_cb over pooled tasks with >=2 correct and non-null value (4 + 2 tasks);
# the c=0 (0.95) and c=1 (0.9) tasks are excluded: 2.2 / 6 = 0.36667.
_EXP_CB = (0.0 + 0.6 + 0.5 + 0.3 + 0.8 + 0.0) / 6                              # 0.36667
# Wrong rule (filter dropped, every non-null value): (2.2 + 0.95 + 0.9) / 8 = 4.05 / 8.
_WRONG_CB_NO_FILTER = (0.0 + 0.6 + 0.5 + 0.3 + 0.95 + 0.8 + 0.0 + 0.9) / 8      # 0.50625
# Per-platform reported cb_div (eligible tasks only): AtCoder 1.4/4 = 0.35, LeetCode
# 0.8/2 = 0.4 -- unchanged by the ineligible non-null values.
_EXP_TERM = (50 * 90 + 12 * 50) / 62                                           # 82.258
_EXP_TOK = (50 * 3000 + 12 * 1000) / 62                                        # 2612.9
_EXP_TIMEOUT = 100 * (5 + 3) / 62                                              # 12.903


def _pooled_fixture(mdir: Path, config: str = "gk_0_1", t: str = "1.0", **over) -> None:
    a, l_ = dict(_PA), dict(_PL)
    a.update(over.get("atc", {}))
    l_.update(over.get("lc", {}))
    _write_metrics(mdir / f"{config}_atcoder_think_t{t}_metrics.json", a.pop("correct"), **a)
    _write_metrics(mdir / f"{config}_leetcode_think_t{t}_metrics.json", l_.pop("correct"), **l_)


def test_pooled_metrics_hand_computed(tmp_path):
    _pooled_fixture(tmp_path)
    (row,) = lb.build_rows(tmp_path)
    pm = row["_pooled_metrics"]
    assert pm["num_tasks"] == 8
    assert pm["pass_at_k"]["1"] == pytest.approx(_EXP_PASS1, abs=1e-12)
    assert pm["pass_at_k"]["3"] == pytest.approx(_EXP_PASS3, abs=1e-12)
    for k in (1, 3, 5, 8, 10):
        assert pm["pass_at_k"][str(k)] == pytest.approx(row[f"pooled_pass@{k}"], abs=1e-12)
    for t in ("0.3", "0.5", "0.7"):
        assert pm["cover_at_t"][t] == pytest.approx(_EXP_COV[t], abs=1e-9)
        assert pm["cover_at_t_distinct"][t] == pytest.approx(_EXP_COVD[t], abs=1e-9)
    assert pm["codebleu_diversity"] == pytest.approx(_EXP_CB, abs=1e-12)
    assert pm["termination_pct"] == pytest.approx(_EXP_TERM, abs=1e-9)
    assert pm["mean_token_size"] == pytest.approx(_EXP_TOK, abs=1e-9)
    assert pm["timeout_pct"] == pytest.approx(_EXP_TIMEOUT, abs=1e-9)
    assert pm["structural_diversity"] is None


def test_pooled_cov_is_not_sample_weighted_and_cb_is_not_task_or_sample_weighted(tmp_path):
    """Guards against the plausible wrong pooling rules (they all differ on this fixture)."""
    _pooled_fixture(tmp_path)
    (row,) = lb.build_rows(tmp_path)
    pm = row["_pooled_metrics"]
    cov_a, cov_l = 100 * 4 / 5, 100 * 2 / 3
    assert pm["cover_at_t"]["0.3"] != pytest.approx((50 * cov_a + 12 * cov_l) / 62, abs=1e-3)
    assert pm["codebleu_diversity"] != pytest.approx((5 * 0.35 + 3 * 0.4) / 8, abs=1e-4)
    assert pm["codebleu_diversity"] != pytest.approx((50 * 0.35 + 12 * 0.4) / 62, abs=1e-4)
    assert pm["codebleu_diversity"] != pytest.approx(_WRONG_CB_NO_FILTER, abs=1e-4)


def test_pooled_fixture_has_ineligible_non_null_self_codebleu():
    """The fixture must exercise the `num_correct >= 2` cb_div eligibility filter."""
    for fx in (_PA, _PL):
        assert any(c < 2 and s is not None for c, s in zip(fx["correct"], fx["cb"]))
    assert any(c == 0 and s is not None for c, s in zip(_PA["correct"], _PA["cb"]))
    assert any(c == 1 and s == 0.9 for c, s in zip(_PL["correct"], _PL["cb"]))


def test_pooled_cb_div_excludes_tasks_with_fewer_than_two_correct(tmp_path):
    _pooled_fixture(tmp_path)
    (row,) = lb.build_rows(tmp_path)
    assert row["_pooled_metrics"]["codebleu_diversity"] == pytest.approx(_EXP_CB, abs=1e-12)
    assert _EXP_CB == pytest.approx(2.2 / 6, abs=1e-12)
    assert _WRONG_CB_NO_FILTER == pytest.approx(4.05 / 8, abs=1e-12)


def _pooled_section(md: str) -> list[str]:
    lines = md.splitlines()
    i = next(j for j, x in enumerate(lines) if x.startswith("### Pooled (AtCoder + LeetCode"))
    j = lines.index("### AtCoder")
    return lines[i:j]


def test_pooled_table_row_strings(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _pooled_fixture(mdir)
    _run(mdir, out, 5, 3)
    sec = _pooled_section((out / lb.MD_NAME).read_text())
    assert sec[0] == "### Pooled (AtCoder + LeetCode, N=8)"
    assert sec[2] == "| Config | " + " | ".join(lb.CANONICAL_COLUMNS) + " |"
    row = next(x for x in sec if x.startswith("| gk_0_1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    got = dict(zip(["label"] + lb.CANONICAL_COLUMNS, cells, strict=True))
    assert got == {
        "label": "gk_0_1", "N": "8", "pass@1": f"{100 * _EXP_PASS1:.2f}%",
        "pass@3": "79.58%",
        "pass@5": got["pass@5"], "pass@8": got["pass@8"], "pass@10": got["pass@10"],
        "cov@0.3": "75.0%", "cov@0.5": "62.5%", "cov@0.7": "37.5%",
        "cov@0.3d": "37.5%", "cov@0.5d": "25.0%", "cov@0.7d": "25.0%",
        "term%": "82.3%", "mean_tok": "2613", "timeout%": "12.9%",
        "struct_div": "—", "cb_div": "0.3667",
    }


def test_pooled_table_row_order_and_dagger(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _pooled_fixture(mdir, "gk_0_1")
    _pooled_fixture(mdir, "top_p_0_7", atc={"correct": [10, 10, 10, 10, 10],
                                            "distinct": [1] * 5, "cb": [0.0] * 5})
    _pooled_fixture(mdir, "ministral_recipe", t="0.7",
                    atc={"correct": [0] * 5, "distinct": [0] * 5, "cb": [None] * 5})
    ranked = _run(mdir, out, 5, 3)
    assert [r["config"] for r in ranked] == ["top_p_0_7", "gk_0_1", "ministral_recipe"]
    sec = _pooled_section((out / lb.MD_NAME).read_text())
    labels = [x.split("|")[1].strip() for x in sec[4:] if x.startswith("| ")]
    assert labels == ["top_p_0_7†", "gk_0_1", "ministral_recipe"]


def test_canonical_section_heading_and_pooled_subsection_first(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _pooled_fixture(mdir)
    _run(mdir, out, 5, 3)
    lines = (out / lb.MD_NAME).read_text().splitlines()
    assert "## Per-platform canonical tables" not in lines
    h = lines.index("## Canonical tables (pooled, then per platform)")
    assert lb.CANONICAL_SECTION_HEADING == lines[h]
    subs = [x for x in lines[h:] if x.startswith("### ")]
    assert subs == ["### Pooled (AtCoder + LeetCode, N=8)", "### AtCoder", "### LeetCode"]
    intro = lines[h + 2]
    assert intro.startswith("Generated with `bench.eval.headline_table.headline_table(..., "
                            "columns=CANONICAL_COLUMNS)`; rows in the pooled-ranking order. "
                            "struct_div shows — because it was intentionally skipped "
                            "(`--skip-structural-diversity`).")
    for frag in ("pass@k, cov@t, cov@t distinct", "task-weighted", "cb_div",
                 "term%, mean_tok, timeout%", "sample-weighted", "struct_div"):
        assert frag in intro.split("skipped (`--skip-structural-diversity`).", 1)[1]
    assert lines.index("## Provenance") > lines.index("### LeetCode")


def test_pooled_crosscheck_raises_on_pass_at_k_mismatch(tmp_path):
    _pooled_fixture(tmp_path)
    ranked = lb.add_paired_deltas(lb.rank_rows(lb.build_rows(tmp_path)))
    lb.render_pooled_table(ranked)  # consistent -> no error
    ranked[0]["_pooled_metrics"]["pass_at_k"]["8"] += 1e-6
    with pytest.raises(ValueError, match="pass@8"):
        lb.render_pooled_table(ranked)


@pytest.mark.parametrize("override", [
    {"cover_at_t": {"0.3": 81.0, "0.5": 60.0, "0.7": 40.0}},
    {"cover_at_t_distinct": {"0.3": 40.0, "0.5": 20.0, "0.7": 21.0}},
    {"codebleu_diversity": 0.36},
])
def test_rejects_platform_aggregate_not_reproducible_from_per_task(tmp_path, override):
    _pooled_fixture(tmp_path, atc={"overrides": override})
    with pytest.raises(ValueError, match="per_task"):
        lb.build_rows(tmp_path)


def test_pooled_tmp_jsons_not_written_into_metrics_dir(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _pooled_fixture(mdir)
    before = sorted(p.name for p in mdir.iterdir())
    _run(mdir, out, 5, 3)
    assert sorted(p.name for p in mdir.iterdir()) == before
    assert sorted(p.name for p in out.iterdir()) == sorted([lb.MD_NAME, lb.CSV_NAME])


def test_csv_unaffected_by_pooled_table(tmp_path):
    mdir, out = tmp_path / "m", tmp_path / "out"
    mdir.mkdir()
    _pooled_fixture(mdir)
    _run(mdir, out, 5, 3)
    with (out / lb.CSV_NAME).open() as f:
        header = next(csv.reader(f))
    assert header == list(lb.CSV_FIELDS)
    assert not any(h.startswith(("cov", "cb_", "struct")) or "pooled_cov" in h for h in header)
