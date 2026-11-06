"""Tests for the smoke-test checker (scripts/humaneval_regen_smoke_checks.py). Plan §8 step 3."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("smoke", REPO / "scripts" / "humaneval_regen_smoke_checks.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def test_first_body_indent():
    assert smoke.first_body_indent("    return 1\n") == 4
    assert smoke.first_body_indent("\n\n   x = 1\n    return x\n") == 3
    assert smoke.first_body_indent("") is None


def test_stop_hits_counts_raw_texts_containing_a_stop_sequence():
    assert smoke.stop_hits(["    return 1\n\ndef g():", "    return 2\n"], ["\ndef", "\nif"]) == 1


def test_identical_share():
    a = {"t1": ["x", "y"], "t2": ["z"]}
    b = {"t1": ["x", "y"], "t2": ["w"], "t3": ["q"]}
    assert smoke.identical_share(a, b) == (1, 2)   # only shared task ids are compared


def test_seconds_per_task_from_timestamps():
    ts = ["2026-11-06T10:00:00+00:00", "2026-11-06T10:00:04+00:00", "2026-11-06T10:00:10+00:00"]
    assert smoke.seconds_per_task(ts) == 5.0   # median of [4, 6]
    assert smoke.seconds_per_task(ts[:1]) is None


def test_main_end_to_end_on_synthetic_outputs(tmp_path, capsys):
    import json
    import pytest
    from bench.humaneval import regen
    try:
        tok = regen.load_tokenizer("Qwen/Qwen2.5-Coder-1.5B")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    task = {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "T", "entry_point": "f"}
    ids = regen.build_prompt_ids(tok, task["prompt"], "base")
    for root in ("a", "b"):
        for name, raw in (("greedy", ["    return 1\n"]), ("temp_t0.7", [f"    return {i}\n" for i in range(10)])):
            meta = [{"finish_reason": "stop", "stop_reason": "\ndef", "num_output_tokens": 5}] * len(raw)
            rec = regen.make_record(task, model_id="Qwen/Qwen2.5-Coder-1.5B", style="base",
                                    config=regen.CONFIGS[name], config_name=name, prompt_ids=ids, raw=raw,
                                    seed=42, max_new_tokens=512, versions={}, metas=meta, backend_stops=True)
            p = regen.output_path(tmp_path / root, "Qwen/Qwen2.5-Coder-1.5B", name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rec) + "\n")
    assert smoke.main(["--smoke-root", str(tmp_path / "a"), "--rerun-root", str(tmp_path / "b")]) == 0
    assert "HARD CHECKS: PASS" in capsys.readouterr().out


def _write_run(root, name, raws, metas, backend_stops=True, model="Qwen/Qwen2.5-Coder-1.5B"):
    import json
    from bench.humaneval import regen
    tok = regen.load_tokenizer(model)
    task = {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "T", "entry_point": "f"}
    rec = regen.make_record(task, model_id=model, style="base", config=regen.CONFIGS[name], config_name=name,
                            prompt_ids=regen.build_prompt_ids(tok, task["prompt"], "base"), raw=raws, seed=42,
                            max_new_tokens=512, versions={}, metas=metas, backend_stops=backend_stops)
    p = regen.output_path(root, model, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec) + "\n")


META_STOP = {"finish_reason": "stop", "stop_reason": "\ndef", "num_output_tokens": 5}


def test_backend_that_did_not_stop_is_a_hard_failure(tmp_path, capsys):
    import pytest
    try:
        _write_run(tmp_path / "a", "greedy", ["    return 1\n\ndef g():\n"], [META_STOP])   # stop text present in raw
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    assert smoke.main(["--smoke-root", str(tmp_path / "a")]) == 1
    assert "did not stop" in capsys.readouterr().out


def test_seeded_rerun_difference_is_a_hard_failure(tmp_path, capsys):
    import pytest
    try:
        _write_run(tmp_path / "a", "temp_t0.7", [f"    return {i}\n" for i in range(10)], [META_STOP] * 10)
        _write_run(tmp_path / "b", "temp_t0.7", [f"    return {i + 1}\n" for i in range(10)], [META_STOP] * 10)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    assert smoke.main(["--smoke-root", str(tmp_path / "a"), "--rerun-root", str(tmp_path / "b")]) == 1
    assert "rerun differs" in capsys.readouterr().out


def test_identical_samples_within_every_task_are_a_hard_failure(tmp_path, capsys):
    import pytest
    try:
        _write_run(tmp_path / "a", "temp_t0.7", ["    return 1\n"] * 10, [META_STOP] * 10)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    assert smoke.main(["--smoke-root", str(tmp_path / "a")]) == 1
    assert "identical within every task" in capsys.readouterr().out


def test_at_cap_counts_come_from_generation_metadata():
    metas = [{"finish_reason": "length", "num_output_tokens": 512}, {"finish_reason": "stop", "num_output_tokens": 9}]
    assert smoke.at_cap(metas) == 1
