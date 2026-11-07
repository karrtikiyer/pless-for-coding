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


def test_first_divergence_position():
    assert smoke.first_divergence(["abcd", "xy"], ["abXd", "xy"]) == (0, 2)
    assert smoke.first_divergence(["a"], ["a"]) is None


def test_smoke_configs_cover_every_vllm_sampler():
    import re
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / "run_regen_humaneval_8models.sh").read_text()
    smoke_line = next(ln for ln in script.splitlines() if "--max-problems 5" in ln and "beam4" not in ln and "greedy" in ln)
    for c in ("greedy", "temp_t0.7", "top_p0.9_t1.0", "pless_t0.7", "pless_norm_t0.7"):
        assert c in smoke_line, c


def test_first_body_indent_skips_the_leading_newline_of_a_stripped_prompt_completion():
    # --prompt-variant stripped: the prompt ends with '"""', so completions start with "\n    ".
    assert smoke.first_body_indent("\n    return 1\n") == 4
    assert smoke.first_body_indent("    return 1\n") == 4   # existing (unstripped) data unchanged


# ---------- --prompt-variant stripped: completion starts ----------

def test_classify_start():
    assert smoke.classify_start("\n    return 1\n") == "new_line"
    assert smoke.classify_start("  \n\n    return 1\n") == "new_line"   # trailing blanks on the docstring line
    assert smoke.classify_start(' x = 1\n    return x\n') == "same_line"   # continues on the '"""' line
    assert smoke.classify_start("") == "empty"
    assert smoke.classify_start("\n  \n") == "empty"


def test_stripped_start_counts():
    comps = ["\n    return 1\n", "\n   return 2\n", " x\n    return x\n", ""]
    assert smoke.stripped_start_counts(comps) == {"new_line_4sp": 1, "new_line_other": 1, "same_line": 1, "empty": 1}


def _write_variant_run(root, name, raws, variant):
    import json
    from bench.humaneval import regen
    model = "Qwen/Qwen2.5-Coder-1.5B"
    tok = regen.load_tokenizer(model)
    task = {"task_id": "HumanEval/0", "prompt": 'def f():\n    """ Doc. """\n', "test": "T", "entry_point": "f"}
    meta = [META_STOP] * len(raws)
    rec = regen.make_record(task, model_id=model, style="base", config=regen.CONFIGS[name], config_name=name,
                            prompt_ids=regen.build_prompt_ids(tok, regen.apply_prompt_variant(task["prompt"], "base", variant),
                                                              "base"),
                            raw=raws, seed=42, max_new_tokens=512, versions={}, metas=meta, backend_stops=True,
                            prompt_variant=variant)
    p = regen.output_path(root, model, name, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec) + "\n")


# 10 samples: 7 new-line + 4-space bodies, 2 continuations on the '"""' line, 1 empty (as seen in real stripped data)
MIXED = [f"\n    return {i}\n" for i in range(7)] + [" x = 1\n    return x\n", " y = 2\n    return y\n", ""]


def _main_out(root, capsys):
    rc = smoke.main(["--smoke-root", str(root)])
    return rc, capsys.readouterr().out


def test_stripped_same_line_and_empty_samples_do_not_fail_and_are_reported(tmp_path, capsys):
    import pytest
    try:
        _write_variant_run(tmp_path / "a", "temp_t0.7", MIXED, "stripped")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    rc, out = _main_out(tmp_path / "a", capsys)
    assert rc == 0 and "HARD CHECKS: PASS" in out, out
    table, starts = out.split("STRIPPED-PROMPT COMPLETION STARTS", 1)
    row = next(ln for ln in table.splitlines() if "temp_t0.7__prompt_stripped" in ln)   # file stem shown as config
    assert row.split()[5] == "100.0%"    # 4sp column: share of the 7 new-line starts
    nums = next(ln for ln in starts.splitlines() if "temp_t0.7__prompt_stripped" in ln).split()[-4:]
    assert nums == ["7", "0", "2", "1"]   # new line + 4sp, new line other indent, '"""' line, empty


def test_unstripped_behaviour_unchanged_for_the_same_samples(tmp_path, capsys):
    import pytest
    try:
        _write_variant_run(tmp_path / "a", "temp_t0.7", MIXED, "unstripped")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    rc, out = _main_out(tmp_path / "a", capsys)
    assert rc == 1 and "only 70.0% of completions start with 4 spaces" in out
    assert "STRIPPED-PROMPT COMPLETION STARTS" not in out


def test_stripped_three_space_indent_bug_is_a_hard_failure(tmp_path, capsys):
    import pytest
    try:
        _write_variant_run(tmp_path / "a", "temp_t0.7", [f"\n   return {i}\n" for i in range(10)], "stripped")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    rc, out = _main_out(tmp_path / "a", capsys)
    assert rc == 1 and "new-line-starting completions have a 4-space first body line" in out


def test_stripped_indent_threshold_is_90_percent_of_new_line_starts(tmp_path, capsys):
    import pytest
    ok = [f"\n    return {i}\n" for i in range(9)] + ["\n   return 9\n"]          # 9/10 = 90%: pass
    bad = [f"\n    return {i}\n" for i in range(8)] + ["\n   return 8\n", "\n   return 9\n"]   # 80%: fail
    try:
        _write_variant_run(tmp_path / "ok", "temp_t0.7", ok, "stripped")
        _write_variant_run(tmp_path / "bad", "temp_t0.7", bad, "stripped")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    assert _main_out(tmp_path / "ok", capsys)[0] == 0
    assert _main_out(tmp_path / "bad", capsys)[0] == 1


def test_stripped_lost_newlines_are_a_hard_failure(tmp_path, capsys):
    # A whitespace-losing decode ("\n    x" -> "x") would make every completion look like a '"""'-line
    # continuation and slip past the 4-space check; a majority of same-line starts is flagged.
    import pytest
    try:
        _write_variant_run(tmp_path / "a", "temp_t0.7", [f"return {i}\n" for i in range(10)], "stripped")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    rc, out = _main_out(tmp_path / "a", capsys)
    assert rc == 1 and "start on a new line" in out


def test_stripped_all_empty_is_a_hard_failure(tmp_path, capsys):
    import pytest
    try:
        _write_variant_run(tmp_path / "a", "greedy", [""], "stripped")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    rc, out = _main_out(tmp_path / "a", capsys)
    assert rc == 1 and "all completions are empty" in out
