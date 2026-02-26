"""Integration test for the full eval pipeline via the CLI module."""

import json
import tempfile
from pathlib import Path

from bench.eval.__main__ import infer_output_path, main as cli_main


def test_infer_output_path():
    p = Path("results/Qwen--Qwen2.5-7B/pless_t1.0.jsonl")
    out = infer_output_path(p)
    assert out == Path("results/Qwen--Qwen2.5-7B/metrics/pless_t1.0_metrics.json")


def test_full_pipeline_mbpp(tmp_path, monkeypatch):
    """End-to-end test: create fake JSONL, run eval, check output."""
    # Create a fake results file with simple tasks
    results_file = tmp_path / "test_results.jsonl"
    records = [
        {
            "model": "test-model",
            "method": "pless",
            "temperature": 1.0,
            "task_id": 1,
            "samples": [
                "def add(a, b): return a + b",
                "def add(a, b): return a - b",  # wrong
                "def add(a, b): return a + b",  # duplicate correct
            ],
            "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        },
        {
            "model": "test-model",
            "method": "pless",
            "temperature": 1.0,
            "task_id": 2,
            "samples": [
                "def mul(a, b): return a * b",
                "def mul(a, b): return a * b",
                "def mul(a, b): return 0",  # wrong
            ],
            "test_list": ["assert mul(2, 3) == 6", "assert mul(0, 5) == 0"],
        },
    ]
    with open(results_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    output_file = tmp_path / "metrics.json"

    # Monkeypatch sys.argv for argparse
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench.eval",
            "--results-file", str(results_file),
            "--dataset", "mbpp",
            "--k", "1,3",
            "--t", "0.3,0.5,1.0",
            "--output", str(output_file),
            "--workers", "2",
        ],
    )

    cli_main()

    assert output_file.exists()
    with open(output_file) as f:
        output = json.load(f)

    assert output["model"] == "test-model"
    assert output["method"] == "pless"
    assert output["dataset"] == "mbpp"
    assert output["num_tasks"] == 2
    assert output["num_samples_per_task"] == 3

    # pass@1 should be between 0 and 1
    assert 0 < output["pass_at_k"]["1"] <= 1.0

    # cover@t values are percentages (0-100)
    assert 0 <= output["cover_at_t"]["0.3"] <= 100
    # cover@t should decrease as t increases
    assert output["cover_at_t"]["1.0"] <= output["cover_at_t"]["0.5"]

    # distinct <= non-distinct
    for t in ["0.3", "0.5", "1.0"]:
        assert output["cover_at_t_distinct"][t] <= output["cover_at_t"][t]

    # per_task should have 2 entries
    assert len(output["per_task"]) == 2

    # Each task result should have required fields
    for task_result in output["per_task"]:
        assert "task_id" in task_result
        assert "num_correct" in task_result
        assert "num_distinct_correct" in task_result
        assert "pass_results" in task_result
        assert len(task_result["pass_results"]) == 3
