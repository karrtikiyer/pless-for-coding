import json
import tempfile
from pathlib import Path

from bench.eval.loader import load_results


def test_load_results_basic():
    records = [
        {"task_id": 1, "samples": ["code1"]},
        {"task_id": 2, "samples": ["code2"]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name

    loaded = load_results(path)
    assert len(loaded) == 2
    assert loaded[0]["task_id"] == 1
    assert loaded[1]["task_id"] == 2


def test_load_results_skips_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"task_id": 1}) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps({"task_id": 2}) + "\n")
        path = f.name

    loaded = load_results(path)
    assert len(loaded) == 2


def test_load_results_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    loaded = load_results(path)
    assert loaded == []


def test_load_results_preserves_all_fields():
    record = {"task_id": 42, "model": "test", "samples": ["a", "b"], "extra": True}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(record) + "\n")
        path = f.name

    loaded = load_results(path)
    assert loaded[0] == record
