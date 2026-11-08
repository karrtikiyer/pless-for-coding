"""scripts/compare_bos_check.py: paired double-BOS vs single-BOS MBPP greedy comparison (synthetic inputs)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_bos_check", REPO / "scripts" / "compare_bos_check.py")
cbc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cbc)

MODEL = "codellama/CodeLlama-7b-Instruct-hf"


def test_sign_test_known_values():
    assert cbc.sign_test_p(0, 0) == 1.0
    assert cbc.sign_test_p(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert cbc.sign_test_p(2, 8) == cbc.sign_test_p(8, 2)
    assert cbc.sign_test_p(5, 5) == 1.0
    assert cbc.sign_test_p(10, 0) == pytest.approx(2 / 1024)


def test_sign_test_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for b, c in [(3, 9), (0, 4), (17, 11), (1, 1), (40, 25)]:
        assert cbc.sign_test_p(b, c) == pytest.approx(scipy_stats.binomtest(b, b + c, 0.5).pvalue)


def test_paired_stats():
    ids = list(range(100))
    double = {i: i < 50 for i in ids}                  # 50 %
    single = {i: (i < 50 and i != 0) or i in (60, 61, 62) for i in ids}   # lose 1, gain 3 -> 52 %
    s = cbc.paired_stats(double, single, n_boot=5000, seed=0)
    assert s["n_tasks"] == 100
    assert s["pass1_double"] == pytest.approx(50.0) and s["pass1_single"] == pytest.approx(52.0)
    assert s["diff_pp"] == pytest.approx(2.0)
    assert (s["single_only"], s["double_only"]) == (3, 1)
    assert s["sign_test_p"] == pytest.approx(cbc.sign_test_p(3, 1))
    lo, hi = s["ci95_pp"]
    assert lo <= 2.0 <= hi and lo < hi
    assert cbc.paired_stats(double, single, n_boot=5000, seed=0)["ci95_pp"] == [lo, hi]   # fixed seed


def test_identical_arms_give_zero_interval():
    d = {i: i % 3 == 0 for i in range(30)}
    s = cbc.paired_stats(d, dict(d), n_boot=5000, seed=0)
    assert s["diff_pp"] == 0 and s["ci95_pp"] == [0.0, 0.0] and s["sign_test_p"] == 1.0


def test_mismatched_task_sets_raise():
    with pytest.raises(ValueError, match="task"):
        cbc.paired_stats({1: True, 2: False}, {1: True, 3: False})


def _arm(root, arm, texts, passes, single_bos):
    d = root / arm / MODEL.replace("/", "--")
    (d / "metrics").mkdir(parents=True)
    with open(d / "greedy_t1.0.jsonl", "w") as f:
        for tid, t in texts.items():
            r = {"model": MODEL, "method": "greedy", "task_id": tid, "samples": [t]}
            if single_bos:
                r["single_bos"] = True
            f.write(json.dumps(r) + "\n")
    per_task = [{"task_id": tid, "num_correct": int(p), "pass_results": [p]} for tid, p in passes.items()]
    (d / "metrics" / "greedy_t1.0_metrics.json").write_text(json.dumps({"per_task": per_task}))


def test_cli_end_to_end(tmp_path, capsys):
    ids = [11, 12, 13, 14]
    _arm(tmp_path, "double_bos", {i: f"x{i}" for i in ids}, {11: True, 12: False, 13: True, 14: False}, False)
    _arm(tmp_path, "single_bos", {11: "x11", 12: "CHANGED", 13: "x13", 14: "CHANGED"},
         {11: True, 12: True, 13: True, 14: False}, True)
    out_json = tmp_path / "cmp.json"
    cbc.main(["--out-root", str(tmp_path), "--model", MODEL, "--json-out", str(out_json)])
    res = json.loads(out_json.read_text())
    assert res["n_text_differs"] == 2 and res["text_differs_task_ids"] == [12, 14]
    assert res["single_only"] == 1 and res["double_only"] == 0
    assert res["diff_pp"] == pytest.approx(25.0)
    assert res["n_boot"] == 5000 and res["bootstrap_seed"] == 0
    assert "greedy outputs differ: 2 / 4" in capsys.readouterr().out


def test_cli_refuses_swapped_arms(tmp_path):
    _arm(tmp_path, "double_bos", {11: "a"}, {11: True}, True)     # the double arm must NOT carry single_bos
    _arm(tmp_path, "single_bos", {11: "a"}, {11: True}, True)
    with pytest.raises(SystemExit, match="single_bos"):
        cbc.main(["--out-root", str(tmp_path), "--model", MODEL])


def test_cli_refuses_ungraded_mismatch(tmp_path):
    _arm(tmp_path, "double_bos", {11: "a", 12: "b"}, {11: True}, False)   # metrics cover fewer tasks than the jsonl
    _arm(tmp_path, "single_bos", {11: "a", 12: "b"}, {11: True, 12: True}, True)
    with pytest.raises(SystemExit, match="metrics"):
        cbc.main(["--out-root", str(tmp_path), "--model", MODEL])
