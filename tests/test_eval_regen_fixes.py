"""Regression tests for three eval-code defects found while validating the 2026-11 HumanEval regeneration.

1. cov@td: regen base samples are function bodies (no ``def``), so fingerprinting the raw sample gave
   ``ast_fingerprint(...) is None`` and num_distinct_correct == 0 for every task. Under
   ``--humaneval-grader official`` the fingerprint now covers the program the official grader executes,
   minus the test tail. The legacy/MBPP path must be unchanged.
2. cb_div: CodeBLEU's dataflow-match depends on PYTHONHASHSEED. self-CodeBLEU is now computed under a
   fixed hash seed (recorded in the metrics JSON as ``self_codebleu_hashseed``).
3. n=1 configs (greedy, beam): ``headline_table(..., na_for_single_sample=True)`` renders the
   non-applicable cells as "—"; the default output is unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.eval.executor import strip_code_fences
from bench.eval.fingerprint import ast_fingerprint
from bench.eval.headline_table import CANONICAL_COLUMNS, headline_row, headline_table
from bench.eval.metrics import (
    SELF_CODEBLEU_HASHSEED,
    add_distinct_counts,
    add_self_codebleu,
    build_metrics_output,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HE_PROMPT = 'def is_prime(n):\n    """Return true if a given number is prime, and false otherwise."""\n'
_HE_TEST = "def check(candidate):\n    assert candidate(2) is True\n    assert candidate(4) is False\n"

# Three distinct correct bodies of HumanEval/31 written by Qwen2.5-Coder-3B (pless_t0.7, regen run).
# On these, the pre-fix self-CodeBLEU changes with PYTHONHASHSEED.
_BODY_A = ('    "*** YOUR CODE HERE ***"\n    if n == 1:\n        return False\n    for i in range(2, n):\n'
           '        if n % i == 0:\n            return False\n    return True\n')
_BODY_B = ('    "*** YOUR CODE HERE ***"\n    if n == 1:\n        return False\n    if n == 2:\n        return True\n'
           '    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n')
_BODY_C = ('    "*** YOUR CODE HERE ***"\n    if n == 1:\n        return False\n    elif n == 2:\n'
           '        return True\n    else:\n        for i in range(2, n):\n            if n % i == 0:\n'
           '                return False\n        return True\n')
# Same structure as _BODY_A, different variable name -> same normalized fingerprint.
_BODY_A_RENAMED = _BODY_A.replace("for i in", "for j in").replace("n % i", "n % j")


def _he_record(samples, style="base", task_id="HumanEval/31"):
    return {"task_id": task_id, "samples": samples, "prompt_style": style, "task_prompt": _HE_PROMPT,
            "test": _HE_TEST, "entry_point": "is_prime"}


def _mbpp_records():
    """MBPP-style records: full functions, fenced replies, an unparseable sample, a repeat."""
    return [
        {"task_id": 1, "samples": [
            "def add(a, b):\n    return a + b\n",
            "def add(x, y):\n    return x + y\n",               # same fingerprint as sample 0
            "```python\ndef add(a, b):\n    s = a + b\n    return s\n```",
            "def add(a, b)\n    return a + b\n",                 # syntax error -> fingerprint None
        ]},
        {"task_id": 2, "samples": [
            "    return sorted(xs)\n",                           # body only -> fingerprint None
            "def srt(xs):\n    return sorted(xs)\n",
            "def srt(xs):\n    xs.sort()\n    return xs\n",
            "def srt(xs):\n    return list(sorted(xs))\n",
        ]},
    ]


def _mbpp_task_results():
    return [
        {"task_id": 1, "pass_results": [True, True, True, True], "num_correct": 4},
        {"task_id": 2, "pass_results": [True, True, False, True], "num_correct": 3},
    ]


# ---------------------------------------------------------------------------
# Defect 1: distinct-correct counts
# ---------------------------------------------------------------------------

def test_mbpp_distinct_counts_match_legacy_rule():
    """Default path is the legacy rule: fingerprint strip_code_fences(sample), skip None."""
    recs, tr = _mbpp_records(), _mbpp_task_results()
    add_distinct_counts(tr, recs)
    for t, r in zip(tr, recs):
        legacy = {ast_fingerprint(strip_code_fences(s)) for s, p in zip(r["samples"], t["pass_results"]) if p}
        legacy.discard(None)
        assert t["num_distinct_correct"] == len(legacy)
    # Pinned values (computed with the pre-fix code at HEAD eb58688).
    assert [t["num_distinct_correct"] for t in tr] == [2, 2]


def test_mbpp_build_metrics_output_unchanged_snapshot():
    """build_metrics_output on MBPP-style records, default arguments: per-task and aggregate values
    pinned to the pre-fix output (HEAD eb58688), excluding the timestamp and the new hash-seed key."""
    recs, tr = _mbpp_records(), _mbpp_task_results()
    out = build_metrics_output(tr, recs, model="m", method="temp", temperature=0.7, dataset="mbpp",
                               k_values=[1, 3], t_values=[0.3, 0.5, 0.7], skip_diversity=True)
    out.pop("timestamp")
    assert out == {
        "model": "m", "method": "temp", "temperature": 0.7, "dataset": "mbpp", "num_tasks": 2,
        "num_samples_per_task": 4, "pass_at_k": {"1": 0.875, "3": 1.0},
        "cover_at_t": {"0.3": 100.0, "0.5": 100.0, "0.7": 100.0},
        "cover_at_t_distinct": {"0.3": 100.0, "0.5": 100.0, "0.7": 0.0},
        "structural_diversity": None, "termination_pct": None, "mean_token_size": None,
        "timeout_pct": None,
        "per_task": [
            {"task_id": 1, "pass_results": [True, True, True, True], "num_correct": 4, "num_distinct_correct": 2},
            {"task_id": 2, "pass_results": [True, True, False, True], "num_correct": 3, "num_distinct_correct": 2},
        ],
    }


def test_official_fingerprint_source_is_program_without_test_tail():
    from bench.eval.humaneval_official import build_program, fingerprint_source

    for style, sample in [("base", _BODY_A),
                          ("chat", "```python\ndef is_prime(n):\n    return n > 1 and all(n % d for d in range(2, n))\n```"),
                          ("chat", _BODY_B)]:
        rec = _he_record([sample], style=style)
        program, _ = build_program(rec, sample)
        src = fingerprint_source(rec, sample)
        tail = "\n" + _HE_TEST + "\n" + "check(is_prime)"
        assert program == src + tail
        assert ast_fingerprint(src) is not None


def test_official_distinct_counts_nonzero_for_base_bodies():
    """10/10-correct base task (as HumanEval/0 of Qwen-3B pless_t0.7) must not report 0 distinct."""
    from bench.eval.humaneval_official import fingerprint_source

    samples = [_BODY_A, _BODY_A_RENAMED, _BODY_B, _BODY_C, _BODY_A]
    recs = [_he_record(samples)]
    tr = [{"task_id": "HumanEval/31", "pass_results": [True] * 5, "num_correct": 5}]
    add_distinct_counts(tr, recs)
    assert tr[0]["num_distinct_correct"] == 0          # the defect, still the legacy behaviour
    add_distinct_counts(tr, recs, code_for_sample=fingerprint_source)
    assert tr[0]["num_distinct_correct"] == 3          # A (== A renamed), B, C


def test_official_cli_fingerprints_regen_base_records(tmp_path):
    """End-to-end: `bench.eval --humaneval-grader official` on a regen-style file gives distinct > 0."""
    samples = [_BODY_A, _BODY_A_RENAMED, _BODY_B]
    rec = _he_record(samples)
    f = tmp_path / "pless_t0.7.jsonl"
    f.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "m.json"
    subprocess.run([sys.executable, "-m", "bench.eval", "--results-file", str(f), "--dataset", "humaneval",
                    "--humaneval-grader", "official", "--skip-diversity", "--k", "1", "--workers", "1",
                    "--output", str(out)], cwd=REPO_ROOT, check=True, capture_output=True)
    m = json.loads(out.read_text())
    assert m["per_task"][0]["num_correct"] == 3
    assert m["per_task"][0]["num_distinct_correct"] == 2


# ---------------------------------------------------------------------------
# Defect 2: deterministic self-CodeBLEU
# ---------------------------------------------------------------------------

_CB_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from bench.eval.metrics import add_self_codebleu
recs = json.loads(sys.argv[2])
tr = [{"task_id": r["task_id"], "pass_results": [True] * len(r["samples"]), "num_correct": len(r["samples"])}
      for r in recs]
add_self_codebleu(tr, recs)
print(json.dumps(tr))
"""


def _cb_under_hashseed(seed: str, recs) -> list[dict]:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run([sys.executable, "-c", _CB_CHILD, str(REPO_ROOT), json.dumps(recs)],
                          env=env, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_self_codebleu_independent_of_parent_hashseed():
    recs = [_he_record([_BODY_A, _BODY_B, _BODY_C]),
            {"task_id": 7, "samples": ["def f(x):\n    return x + 1\n", "def f(x):\n    y = x\n    return y + 2\n"]}]
    runs = [_cb_under_hashseed(s, recs) for s in ("0", "1", "2", "3", "4", "random")]
    for r in runs[1:]:
        assert r == runs[0]
    assert runs[0][0]["self_codebleu"] is not None


def test_self_codebleu_in_process_matches_seed0():
    """The in-process call (whatever this pytest process's hash seed) equals the PYTHONHASHSEED=0 child."""
    recs = [_he_record([_BODY_A, _BODY_B, _BODY_C])]
    tr = [{"task_id": "HumanEval/31", "pass_results": [True] * 3, "num_correct": 3}]
    add_self_codebleu(tr, recs)
    assert tr == _cb_under_hashseed("0", recs)


def test_self_codebleu_short_circuits_unchanged():
    """<2 correct -> None; all structurally identical -> 0.0 (no CodeBLEU call needed)."""
    recs = [{"task_id": 1, "samples": ["def f():\n    return 1\n", "x"]},
            {"task_id": 2, "samples": ["def f(a):\n    return a\n", "def g(b):\n    return b\n"]}]
    tr = [{"task_id": 1, "pass_results": [True, False], "num_correct": 1},
          {"task_id": 2, "pass_results": [True, True], "num_correct": 2}]
    add_self_codebleu(tr, recs)
    assert tr[0]["self_codebleu"] is None
    assert tr[1]["self_codebleu"] == 0.0 and tr[1]["self_dataflow_match"] == 0.0


def test_metrics_output_records_hashseed():
    recs, tr = _mbpp_records(), _mbpp_task_results()
    out = build_metrics_output(tr, recs, model="m", method="temp", temperature=0.7, dataset="mbpp",
                               k_values=[1], t_values=[0.5], skip_structural_diversity=True)
    assert SELF_CODEBLEU_HASHSEED == 0
    assert out["self_codebleu_hashseed"] == 0
    # Not computed -> not recorded.
    tr2 = _mbpp_task_results()
    out2 = build_metrics_output(tr2, _mbpp_records(), model="m", method="temp", temperature=0.7,
                                dataset="mbpp", k_values=[1], t_values=[0.5], skip_diversity=True)
    assert "self_codebleu_hashseed" not in out2


# ---------------------------------------------------------------------------
# Defect 3: n=1 cells as "—"
# ---------------------------------------------------------------------------

_NA_COLS = ["pass@3", "pass@5", "pass@8", "pass@10", "cov@0.3", "cov@0.5", "cov@0.7",
            "cov@0.3d", "cov@0.5d", "cov@0.7d", "struct_div", "cb_div"]


def _write_metrics(tmp_path, name, n):
    m = {"num_tasks": 164, "num_samples_per_task": n,
         "pass_at_k": {"1": 0.5, "3": 0.5, "5": 0.5, "8": 0.5, "10": 0.5},
         "cover_at_t": {"0.3": 50.0, "0.5": 50.0, "0.7": 50.0},
         "cover_at_t_distinct": {"0.3": 0.0, "0.5": 0.0, "0.7": 0.0},
         "termination_pct": None, "mean_token_size": 120.0, "timeout_pct": 1.0,
         "structural_diversity": 0.0, "codebleu_diversity": 0.0}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(m))
    return p


def test_headline_default_unchanged_for_n1(tmp_path):
    p = _write_metrics(tmp_path, "greedy", 1)
    row = headline_row(p, "greedy")
    assert row["pass@10"] == "50.00%" and row["cov@0.5d"] == "0.0%" and row["cb_div"] == "0.0000"
    assert headline_table([(p, "greedy")]) == headline_table([(p, "greedy")], na_for_single_sample=False)
    expected_body = ("| greedy | 164 | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.0% | 50.0% | 50.0% | "
                     "0.0% | 0.0% | 0.0% | — | 120 | 1.0% | 0.0000 | 0.0000 |")
    assert headline_table([(p, "greedy")]).splitlines()[2] == expected_body


def test_headline_na_for_single_sample(tmp_path):
    p1 = _write_metrics(tmp_path, "greedy", 1)
    p10 = _write_metrics(tmp_path, "temp", 10)
    row1 = headline_row(p1, "greedy", na_for_single_sample=True)
    for c in _NA_COLS:
        assert row1[c] == "—", c
    assert row1["pass@1"] == "50.00%" and row1["N"] == "164"
    assert row1["mean_tok"] == "120" and row1["timeout%"] == "1.0%"
    # n=10 rows are unaffected by the flag.
    assert headline_row(p10, "temp", na_for_single_sample=True) == headline_row(p10, "temp")
    table = headline_table([(p1, "greedy"), (p10, "temp")], na_for_single_sample=True)
    lines = table.splitlines()
    assert lines[2].count("—") == len(_NA_COLS) + 1   # + term% (None in both rows)
    assert lines[3] == headline_table([(p10, "temp")]).splitlines()[2]
    assert len(lines[2].split("|")) == len(CANONICAL_COLUMNS) + 3


def test_headline_na_uses_per_task_when_n_missing(tmp_path):
    m = json.loads(_write_metrics(tmp_path, "x", 1).read_text())
    del m["num_samples_per_task"]
    m["per_task"] = [{"pass_results": [True]}, {"pass_results": [False]}]
    p = tmp_path / "beam.json"
    p.write_text(json.dumps(m))
    assert headline_row(p, "beam", na_for_single_sample=True)["pass@5"] == "—"
