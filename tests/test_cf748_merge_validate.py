"""scripts/cf748_merge_validate.py: merge two per-GPU CF-748 result dirs for one arm (tiny fixtures)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cf748_merge_validate.py"
METHOD = "pless_effk_nofloor"
FNAME = f"{METHOD}_think_t1.0_t1.0.jsonl"
SUB = Path("Qwen--Qwen3-8B") / "CODEFORCES_interview"
IDS = [3, 5, 8, 13, 21, 34]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rec(tid, n=3, **over):
    r = {"model": "Qwen/Qwen3-8B", "backend": "vllm", "method": METHOD, "temperature": 1.0, "top_p": 1.0,
         "top_k": 0, "task_id": tid, "source": "CODEFORCES", "difficulty": "interview", "prompt_text": "q",
         "samples": [f"s{i}" for i in range(n)], "timestamp": "t",
         "samples_with_thinking": [f"w{i}" for i in range(n)]}
    r.update(over)
    return r


def _write(root: Path, recs, raw_tail: str = "") -> Path:
    d = root / SUB
    d.mkdir(parents=True, exist_ok=True)
    p = d / FNAME
    p.write_text("".join(json.dumps(r) + "\n" for r in recs) + raw_tail)
    return p


@pytest.fixture
def fx(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("".join(f"{i}\n" for i in IDS))
    sa, sb = tmp_path / "even.txt", tmp_path / "odd.txt"
    sa.write_text("".join(f"{i}\n" for i in IDS[0::2]))
    sb.write_text("".join(f"{i}\n" for i in IDS[1::2]))
    a, b = tmp_path / "gpu0", tmp_path / "gpu1"
    # GPU0 writes its ids out of order (resume / task order) -> merged file must be sorted.
    _write(a, [_rec(t) for t in reversed(IDS[0::2])])
    _write(b, [_rec(t) for t in IDS[1::2]])
    return {"tmp": tmp_path, "ids": ids, "sa": sa, "sb": sb, "a": a, "b": b, "out": tmp_path / "merged"}


def _cli(fx, *extra, ok=True, out=True):
    args = [sys.executable, "-I", str(SCRIPT), "--in-a", str(fx["a"]), "--in-b", str(fx["b"]),
            "--ids-file", str(fx["ids"]), "--expected-count", str(len(IDS)), "--n-samples", "3",
            "--method", METHOD, "--temperature", "1.0"]
    if out:
        args += ["--out-results-dir", str(fx["out"])]
    r = subprocess.run(args + list(extra), capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_happy_merge(fx):
    r = _cli(fx, "--split-a", str(fx["sa"]), "--split-b", str(fx["sb"]))
    outf = fx["out"] / SUB / FNAME
    lines = outf.read_text().splitlines()
    assert [json.loads(l)["task_id"] for l in lines] == IDS
    # lines are copied verbatim from the inputs
    src = set((fx["a"] / SUB / FNAME).read_text().splitlines()) | set((fx["b"] / SUB / FNAME).read_text().splitlines())
    assert set(lines) == src
    man = json.loads(Path(str(outf) + ".merge_manifest.json").read_text())
    assert man["inputs"]["a"]["sha256"] == _sha(fx["a"] / SUB / FNAME)
    assert man["inputs"]["b"]["sha256"] == _sha(fx["b"] / SUB / FNAME)
    assert man["inputs"]["a"]["count"] == 3 and man["inputs"]["b"]["count"] == 3
    assert man["ids_file"]["sha256"] == _sha(fx["ids"])
    assert man["output"]["sha256"] == _sha(outf) and man["output"]["count"] == len(IDS)
    assert man["split_files"]["a"]["sha256"] == _sha(fx["sa"])
    assert "MERGE OK" in r.stdout


def test_refuses_to_overwrite(fx):
    _cli(fx)
    outf = fx["out"] / SUB / FNAME
    before = _sha(outf)
    r = _cli(fx, ok=False)
    assert r.returncode != 0 and "exists" in r.stderr
    assert _sha(outf) == before


def test_refuses_if_only_manifest_exists(fx):
    d = fx["out"] / SUB
    d.mkdir(parents=True)
    (d / (FNAME + ".merge_manifest.json")).write_text("{}")
    r = _cli(fx, ok=False)
    assert r.returncode != 0 and "exists" in r.stderr
    assert not (d / FNAME).exists()


def _assert_fails(fx, needle, *extra):
    r = _cli(fx, *extra, ok=False)
    assert r.returncode != 0, r.stdout
    assert needle in r.stderr, r.stderr
    assert not (fx["out"] / SUB / FNAME).exists()


def test_unparseable_line(fx):
    _write(fx["b"], [_rec(t) for t in IDS[1::2]], raw_tail='{"task_id": 99, "trunc')
    _assert_fails(fx, "unparseable")


def test_blank_line(fx):
    _write(fx["b"], [_rec(t) for t in IDS[1::2]], raw_tail="\n")
    _assert_fails(fx, "blank")


def test_wrong_sample_count(fx):
    _write(fx["b"], [_rec(5), _rec(13, n=2), _rec(34)])
    _assert_fails(fx, "samples")


def test_wrong_thinking_sample_count(fx):
    _write(fx["b"], [_rec(5), _rec(13, samples_with_thinking=["x"]), _rec(34)])
    _assert_fails(fx, "samples_with_thinking")


def test_wrong_method(fx):
    _write(fx["b"], [_rec(5), _rec(13, method="pless"), _rec(34)])
    _assert_fails(fx, "method")


def test_wrong_temperature(fx):
    _write(fx["b"], [_rec(5), _rec(13, temperature=0.6), _rec(34)])
    _assert_fails(fx, "temperature")


def test_duplicate_within_file(fx):
    _write(fx["b"], [_rec(5), _rec(13), _rec(34), _rec(13)])
    _assert_fails(fx, "duplicate")


def test_overlap_between_inputs(fx):
    _write(fx["b"], [_rec(t) for t in IDS[1::2]] + [_rec(3)])
    _assert_fails(fx, "overlap")


def test_union_missing_id(fx):
    _write(fx["b"], [_rec(5), _rec(13)])
    _assert_fails(fx, "union")


def test_id_not_in_ids_file(fx):
    _write(fx["b"], [_rec(t) for t in IDS[1::2]] + [_rec(999)])
    _assert_fails(fx, "not in ids file")


def test_expected_count_enforced(fx):
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--in-a", str(fx["a"]), "--in-b", str(fx["b"]),
                        "--ids-file", str(fx["ids"]), "--n-samples", "3", "--method", METHOD,
                        "--temperature", "1.0", "--out-results-dir", str(fx["out"])],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "748" in r.stderr


def test_split_file_mismatch(fx):
    # GPU0 dir must hold exactly the even ids when --split-a is given
    _assert_fails(fx, "split", "--split-a", str(fx["sb"]), "--split-b", str(fx["sa"]))


def test_same_input_twice_rejected(fx):
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--in-a", str(fx["a"]), "--in-b", str(fx["a"]),
                        "--ids-file", str(fx["ids"]), "--expected-count", str(len(IDS)), "--n-samples", "3",
                        "--method", METHOD, "--temperature", "1.0", "--out-results-dir", str(fx["out"])],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "same" in r.stderr


def test_missing_input_file(fx):
    (fx["b"] / SUB / FNAME).unlink()
    _assert_fails(fx, "missing")


def test_check_only_writes_nothing(fx):
    r = _cli(fx, "--check-only", out=False)
    assert "CHECK OK" in r.stdout
    assert not fx["out"].exists()


def test_check_only_allow_incomplete_reports_progress(fx):
    _write(fx["b"], [_rec(5)])
    r = _cli(fx, "--check-only", "--allow-incomplete", out=False)
    assert "a=3" in r.stdout and "b=1" in r.stdout and "missing=2" in r.stdout
    # still strict on corruption
    _write(fx["b"], [_rec(5), _rec(3)])
    r = _cli(fx, "--check-only", "--allow-incomplete", out=False, ok=False)
    assert r.returncode != 0 and "overlap" in r.stderr


def test_allow_incomplete_never_writes(fx):
    r = _cli(fx, "--allow-incomplete", ok=False)
    assert r.returncode != 0 and "check-only" in r.stderr


def test_deepseek_model_dir_and_model_check(tmp_path):
    """--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B -> model dir deepseek-ai--DeepSeek-R1-Distill-Llama-8B
    for inputs and output, and the records' model field is checked against it."""
    model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    sub = Path("deepseek-ai--DeepSeek-R1-Distill-Llama-8B") / "CODEFORCES_interview"
    ids = tmp_path / "ids.txt"
    ids.write_text("".join(f"{i}\n" for i in IDS))
    a, b, out = tmp_path / "gpu0", tmp_path / "gpu1", tmp_path / "merged"
    for root, part in ((a, IDS[0::2]), (b, IDS[1::2])):
        (root / sub).mkdir(parents=True)
        (root / sub / FNAME).write_text("".join(json.dumps(_rec(t, model=model)) + "\n" for t in part))
    base = [sys.executable, "-I", str(SCRIPT), "--in-a", str(a), "--in-b", str(b), "--ids-file", str(ids),
            "--expected-count", str(len(IDS)), "--n-samples", "3", "--method", METHOD, "--temperature", "1.0"]
    r = subprocess.run(base + ["--check-only"], capture_output=True, text=True)   # default (Qwen) model dir
    assert r.returncode != 0 and "Qwen--Qwen3-8B" in r.stderr, r.stdout + r.stderr
    r = subprocess.run(base + ["--model", model, "--out-results-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [json.loads(l)["task_id"] for l in (out / sub / FNAME).read_text().splitlines()] == IDS
    # a record carrying the wrong model name is rejected
    (b / sub / FNAME).write_text("".join(json.dumps(_rec(t)) + "\n" for t in IDS[1::2]))
    r = subprocess.run(base + ["--model", model, "--check-only"], capture_output=True, text=True)
    assert r.returncode != 0 and "model" in r.stderr, r.stdout + r.stderr


# ---------------------------------------------------------------- built-in-method arms (launcher ARM=topp0.9 / renyi_k0.35)
# The runner's file name for these is NOT default_filename(method, T) (it encodes top_p / renyi_k), so the
# launcher passes --filename; the record checks stay method/temperature/model/source/difficulty/backend.
@pytest.mark.parametrize("method,fname,fields", [
    ("temp", "temp_p0.9_think_t1.0_t1.0.jsonl", {"top_p": 0.9, "top_k": 0}),
    ("pless_renyi", "pless_renyi_think_t1.0_k0.35_t1.0.jsonl", {"renyi_k": 0.35}),
])
def test_builtin_method_arm_with_filename(tmp_path, method, fname, fields):
    ids = tmp_path / "ids.txt"
    ids.write_text("".join(f"{i}\n" for i in IDS))
    sa, sb = tmp_path / "even.txt", tmp_path / "odd.txt"
    sa.write_text("".join(f"{i}\n" for i in IDS[0::2]))
    sb.write_text("".join(f"{i}\n" for i in IDS[1::2]))
    a, b, out = tmp_path / "gpu0", tmp_path / "gpu1", tmp_path / "merged"
    for root, part in ((a, IDS[0::2]), (b, IDS[1::2])):
        (root / SUB).mkdir(parents=True)
        (root / SUB / fname).write_text("".join(json.dumps(_rec(t, method=method, **fields)) + "\n" for t in part))
    base = [sys.executable, "-I", str(SCRIPT), "--in-a", str(a), "--in-b", str(b), "--ids-file", str(ids),
            "--expected-count", str(len(IDS)), "--n-samples", "3", "--method", method, "--temperature", "1.0",
            "--split-a", str(sa), "--split-b", str(sb)]
    # without --filename the default name (<method>_think_t1.0_t1.0.jsonl) does not exist -> refused
    r = subprocess.run(base + ["--check-only"], capture_output=True, text=True)
    assert r.returncode != 0 and f"{method}_think_t1.0_t1.0.jsonl" in r.stderr, r.stdout + r.stderr
    r = subprocess.run(base + ["--filename", fname, "--check-only"], capture_output=True, text=True)
    assert r.returncode == 0 and "CHECK OK (complete)" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(base + ["--filename", fname, "--out-results-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "MERGE OK" in r.stdout, r.stdout + r.stderr
    recs = [json.loads(l) for l in (out / SUB / fname).read_text().splitlines()]
    assert [x["task_id"] for x in recs] == IDS and all(x["method"] == method for x in recs)
    # a record of another method in the same file is still rejected (method check unchanged)
    (b / SUB / fname).write_text("".join(json.dumps(_rec(t, method="pless_effk_nofloor")) + "\n" for t in IDS[1::2]))
    r = subprocess.run(base + ["--filename", fname, "--check-only"], capture_output=True, text=True)
    assert r.returncode != 0 and "method" in r.stderr, r.stdout + r.stderr
