"""scripts/cf748_split_ids.py: deterministic even/odd (by sorted position) split of the CF-748 id list."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cf748_split_ids.py"
REAL_IDS = REPO_ROOT / "data" / "cf_interview_748_ids.txt"


def _mod():
    spec = importlib.util.spec_from_file_location("cf748_split_ids", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cli(*args, ok=True):
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


# ---------------------------------------------------------------- units
def test_split_even_odd_by_sorted_position():
    m = _mod()
    even, odd = m.split_even_odd([30, 10, 20, 50, 40])
    assert even == [10, 30, 50] and odd == [20, 40]


def test_read_ids_rejects_duplicates_and_non_ints(tmp_path):
    m = _mod()
    (tmp_path / "d.txt").write_text("1\n2\n2\n")
    with pytest.raises(ValueError, match="duplicate"):
        m.read_ids(tmp_path / "d.txt")
    (tmp_path / "x.txt").write_text("1\nabc\n")
    with pytest.raises(ValueError):
        m.read_ids(tmp_path / "x.txt")


def test_verify_split_catches_each_violation():
    m = _mod()
    ids = [1, 2, 3, 4]
    m.verify_split(ids, [1, 3], [2, 4], expected_count=4)
    with pytest.raises(ValueError, match="overlap"):
        m.verify_split(ids, [1, 3, 2], [2, 4], expected_count=4)
    with pytest.raises(ValueError, match="union"):
        m.verify_split(ids, [1], [2, 4], expected_count=4)
    with pytest.raises(ValueError, match="union"):
        m.verify_split(ids, [1, 3, 9], [2, 4], expected_count=4)
    with pytest.raises(ValueError, match="count"):
        m.verify_split(ids, [1, 3], [2, 4], expected_count=748)
    with pytest.raises(ValueError, match="duplicate"):
        m.verify_split(ids, [1, 3, 3], [2, 4], expected_count=4)


# ---------------------------------------------------------------- CLI on the real committed list
def test_cli_split_real_748(tmp_path):
    out = tmp_path / "split"
    r = _cli("split", "--input", str(REAL_IDS), "--out-dir", str(out))
    even, odd = out / "cf748_even_ids.txt", out / "cf748_odd_ids.txt"
    ev = [int(x) for x in even.read_text().split()]
    od = [int(x) for x in odd.read_text().split()]
    src = sorted(int(x) for x in REAL_IDS.read_text().split())
    assert len(ev) == 374 and len(od) == 374
    assert ev == src[0::2] and od == src[1::2]
    assert not set(ev) & set(od) and sorted(ev + od) == src
    man = json.loads((out / "split_manifest.json").read_text())
    assert man["input"]["sha256"] == _sha(REAL_IDS) and man["input"]["count"] == 748
    assert man["even"]["sha256"] == _sha(even) and man["even"]["count"] == 374
    assert man["odd"]["sha256"] == _sha(odd) and man["odd"]["count"] == 374
    assert "even" in man["rule"] and "sorted" in man["rule"]
    assert "374" in r.stdout
    # deterministic: a second split elsewhere is byte-identical
    out2 = tmp_path / "split2"
    _cli("split", "--input", str(REAL_IDS), "--out-dir", str(out2))
    assert _sha(out2 / "cf748_even_ids.txt") == _sha(even) and _sha(out2 / "cf748_odd_ids.txt") == _sha(odd)


def test_cli_refuses_overwrite(tmp_path):
    out = tmp_path / "split"
    _cli("split", "--input", str(REAL_IDS), "--out-dir", str(out))
    before = _sha(out / "cf748_even_ids.txt")
    r = _cli("split", "--input", str(REAL_IDS), "--out-dir", str(out), ok=False)
    assert r.returncode != 0 and "exists" in r.stderr
    assert _sha(out / "cf748_even_ids.txt") == before


def test_cli_wrong_count_fails(tmp_path):
    small = tmp_path / "ids.txt"
    small.write_text("5\n3\n1\n")
    r = _cli("split", "--input", str(small), "--out-dir", str(tmp_path / "o"), ok=False)
    assert r.returncode != 0 and "count" in r.stderr
    assert not (tmp_path / "o" / "cf748_even_ids.txt").exists()
    _cli("split", "--input", str(small), "--out-dir", str(tmp_path / "o"), "--expected-count", "3")
    assert (tmp_path / "o" / "cf748_even_ids.txt").read_text() == "1\n5\n"


def test_cli_verify(tmp_path):
    out = tmp_path / "split"
    _cli("split", "--input", str(REAL_IDS), "--out-dir", str(out))
    a, b = out / "cf748_even_ids.txt", out / "cf748_odd_ids.txt"
    r = _cli("verify", "--input", str(REAL_IDS), "--a", str(a), "--b", str(b))
    assert "OK" in r.stdout and _sha(a) in r.stdout and _sha(b) in r.stdout
    # same file twice -> overlap
    r = _cli("verify", "--input", str(REAL_IDS), "--a", str(a), "--b", str(a), ok=False)
    assert r.returncode != 0 and "overlap" in r.stderr
    # drop one id from b -> union mismatch
    bad = tmp_path / "b_bad.txt"
    bad.write_text("\n".join(b.read_text().split()[:-1]) + "\n")
    r = _cli("verify", "--input", str(REAL_IDS), "--a", str(a), "--b", str(bad), ok=False)
    assert r.returncode != 0 and "union" in r.stderr
    # missing file
    r = _cli("verify", "--input", str(REAL_IDS), "--a", str(a), "--b", str(tmp_path / "nope"), ok=False)
    assert r.returncode != 0


def test_verify_even_odd_rule_unit():
    m = _mod()
    m.verify_even_odd_rule([4, 1, 3, 2], [3, 1], [2, 4])   # order within a file is irrelevant
    with pytest.raises(ValueError, match="rule"):
        m.verify_even_odd_rule([1, 2, 3, 4], [1, 4], [2, 3])  # disjoint + union OK, rule violated
    with pytest.raises(ValueError, match="rule"):
        m.verify_even_odd_rule([1, 2, 3, 4], [2, 4], [1, 3])  # halves swapped


def test_cli_verify_rejects_rule_violation_with_valid_partition(tmp_path):
    src = sorted(int(x) for x in REAL_IDS.read_text().split())
    ev, od = src[0::2], src[1::2]
    ev[0], od[0] = od[0], ev[0]                                # still disjoint, union == input
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("".join(f"{i}\n" for i in ev)); b.write_text("".join(f"{i}\n" for i in od))
    r = _cli("verify", "--input", str(REAL_IDS), "--a", str(a), "--b", str(b), ok=False)
    assert r.returncode != 0 and "rule" in r.stderr
