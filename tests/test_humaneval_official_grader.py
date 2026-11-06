"""Tests for the official-composition HumanEval grader (bench/eval/humaneval_official.py).

Spec: docs/experiments/humaneval_regen_2026_11_plan.md §7.
  program = task_prompt + completion + "\\n" + test + "\\n" + check(entry_point)   (openai/human-eval execution.py)
  * base records: the stored sample is the completion only (cut at the BigCode stop list) -> prompt + sample
  * chat records: a reply that defines `def <entry_point>` at column 0 -> prompt + "\\n" + extracted code;
    otherwise (a body fragment) -> prompt + reply exactly as written
  * exit / quit / SystemExit / os._exit count as failures; timeout 3.0 s; venv interpreter
"""
import pytest

from bench.eval import humaneval_official as he

PROMPT = (
    "from typing import List\n\n\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    '    """ Check if any two numbers are closer than threshold. """\n'
)
TEST = (
    "\n\ndef check(candidate):\n"
    "    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n"
    "    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
)
BODY = (
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)
WHOLE_FN_NO_IMPORT = (
    "```python\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n" + BODY + "```\n"
)


def _rec(style, samples):
    return {
        "task_id": "HumanEval/0",
        "prompt_style": style,
        "task_prompt": PROMPT,
        "test": TEST,
        "entry_point": "has_close_elements",
        "samples": samples,
    }


# ---------- program composition ----------

def test_base_program_is_prompt_plus_completion():
    prog, mode = he.build_program(_rec("base", [BODY]), BODY)
    assert mode == "base"
    assert prog == PROMPT + BODY + "\n" + TEST + "\n" + "check(has_close_elements)"


def test_base_completion_is_recut_at_stop_sequences():
    sample = BODY + "\n\ndef check(candidate):\n    assert False\n"
    prog, _ = he.build_program(_rec("base", [sample]), sample)
    assert "assert False" not in prog


def test_chat_whole_function_gets_prompt_prepended():
    prog, mode = he.build_program(_rec("chat", [WHOLE_FN_NO_IMPORT]), WHOLE_FN_NO_IMPORT)
    assert mode == "chat_function"
    assert prog.startswith(PROMPT + "\n")
    assert "```" not in prog


def test_chat_body_fragment_is_appended_as_written():
    frag = "  return abs(number) % 1"
    prog, mode = he.build_program(_rec("chat", [frag]), frag)
    assert mode == "chat_body"
    assert prog.startswith(PROMPT + frag)


def test_unknown_prompt_style_raises():
    with pytest.raises(ValueError):
        he.build_program(_rec("instruct", ["x"]), "x")


# ---------- execution semantics ----------

def test_correct_base_completion_passes():
    ok, reason = he.run_program(he.build_program(_rec("base", [BODY]), BODY)[0], timeout=3.0)
    assert ok and reason == "passed"


def test_missing_import_passes_once_prompt_is_prepended():
    # The old no-prompt grader failed this with NameError: name 'List' is not defined.
    prog, _ = he.build_program(_rec("chat", [WHOLE_FN_NO_IMPORT]), WHOLE_FN_NO_IMPORT)
    ok, reason = he.run_program(prog, timeout=3.0)
    assert ok, reason


def test_reply_with_its_own_imports_passes():
    reply = "from typing import List\nimport math\n\ndef has_close_elements(numbers, threshold):\n" + BODY
    prog, _ = he.build_program(_rec("chat", [reply]), reply)
    assert he.run_program(prog, timeout=3.0)[0]


def test_empty_body_fails():
    prog, _ = he.build_program(_rec("base", [""]), "")
    assert not he.run_program(prog, timeout=3.0)[0]


def test_wrong_answer_fails_with_assertion():
    wrong = "    return True\n"
    ok, reason = he.run_program(he.build_program(_rec("base", [wrong]), wrong)[0], timeout=3.0)
    assert not ok and "AssertionError" in reason


@pytest.mark.parametrize("escape", [
    "    import sys; sys.exit(0)\n",
    "    exit(0)\n",
    "    quit()\n",
    "    raise SystemExit(0)\n",
    "    import os; os._exit(0)\n",
])
def test_exit_tricks_fail(escape):
    # A sample that exits cleanly before the tests run must not count as a pass.
    sample = escape + BODY
    ok, _ = he.run_program(he.build_program(_rec("base", [sample]), sample)[0], timeout=3.0)
    assert not ok


def test_dunder_main_block_in_reply_does_not_run():
    # The official harness execs with fresh globals, so `if __name__ == "__main__":` blocks are skipped.
    reply = "def has_close_elements(numbers, threshold):\n" + BODY + "\nif __name__ == '__main__':\n    assert 1 == 2\n"
    prog, _ = he.build_program(_rec("chat", [reply]), reply)
    assert he.run_program(prog, timeout=3.0)[0]


def test_timeout_fails():
    loop = "    while True:\n        pass\n"
    ok, reason = he.run_program(he.build_program(_rec("base", [loop]), loop)[0], timeout=1.0)
    assert not ok and reason == "timed out"


def test_interpreter_is_the_running_python():
    import sys
    assert he.PYTHON == sys.executable


def test_default_timeout_is_official():
    assert he.DEFAULT_TIMEOUT == 3.0


# ---------- task-level evaluation ----------

def test_evaluate_task_shapes_and_statuses():
    rec = _rec("base", [BODY, "    return True\n", "    while True:\n        pass\n"])
    out = he.evaluate_task(rec, timeout=1.0)
    assert out["task_id"] == "HumanEval/0"
    assert out["pass_results"] == [True, False, False]
    assert out["num_correct"] == 1
    assert out["statuses"] == ["Passed", "Failed", "Timeout"]
    assert out["grader_version"] == he.GRADER_VERSION


def test_evaluate_all_diagnostics_count_statuses():
    recs = [_rec("base", [BODY, "    return True\n"])]
    results, diag = he.evaluate_all(recs, timeout=3.0, workers=1)
    assert diag["by_status"] == {"Passed": 1, "Failed": 1}
    assert diag["grader_version"] == he.GRADER_VERSION
    assert results[0]["num_correct"] == 1


# ---------- known-answer test on the real dataset ----------

def test_canonical_solutions_pass_164_of_164():
    datasets = pytest.importorskip("datasets")
    try:
        ds = datasets.load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:  # offline
        pytest.skip(f"dataset unavailable: {e}")
    recs = [
        {"task_id": t["task_id"], "prompt_style": "base", "task_prompt": t["prompt"], "test": t["test"],
         "entry_point": t["entry_point"], "samples": [t["canonical_solution"]]}
        for t in ds
    ]
    results, diag = he.evaluate_all(recs, timeout=3.0, workers=8)
    failed = [r["task_id"] for r in results if r["num_correct"] != 1]
    assert len(recs) == 164
    assert failed == [], failed


# ---------- bench.eval CLI integration ----------

def test_cli_official_grader_end_to_end(tmp_path, monkeypatch):
    import json
    from bench.eval import __main__ as cli

    rec = _rec("base", [BODY, "    return True\n"])
    rec.update(model="m/x", method="temp", temperature=0.7)
    results = tmp_path / "temp_t0.7.jsonl"
    results.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "metrics.json"
    monkeypatch.setattr("sys.argv", ["bench.eval", "--results-file", str(results), "--dataset", "humaneval",
                                     "--humaneval-grader", "official", "--skip-diversity", "--workers", "1",
                                     "--k", "1", "--output", str(out)])
    cli.main()
    m = json.loads(out.read_text())
    assert m["pass_at_k"]["1"] == 0.5
    assert m["execution_diagnostics"]["grader_version"] == he.GRADER_VERSION
    assert m["execution_diagnostics"]["timeout_s"] == 3.0          # official default when --timeout is omitted
    assert m["timeout_pct"] == 0.0
    assert m["per_task"][0]["statuses"] == ["Passed", "Failed"]


def test_cli_legacy_humaneval_default_unchanged(tmp_path, monkeypatch):
    # Without the flag, --dataset humaneval keeps the old builder and the 5.0 s default.
    from bench.eval import __main__ as cli
    monkeypatch.setattr("sys.argv", ["bench.eval", "--results-file", "x.jsonl", "--dataset", "humaneval"])
    args = cli.parse_args()
    assert args.humaneval_grader == "legacy" and args.timeout == 5.0


def test_cli_official_grader_rejects_other_datasets(monkeypatch):
    from bench.eval import __main__ as cli
    monkeypatch.setattr("sys.argv", ["bench.eval", "--results-file", "x.jsonl", "--dataset", "mbpp",
                                     "--humaneval-grader", "official"])
    with pytest.raises(SystemExit):
        cli.parse_args()


# ---------- chat reply normalization (code review 2026-11-06, CRITICAL #1) ----------

FN = "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n" + BODY


@pytest.mark.parametrize("reply", [
    "  \n[PYTHON]\n" + FN + "[/PYTHON]\n",                                     # CodeLlama's own tags
    "Here is the code:\n[PYTHON]\n" + FN + "[/PYTHON]\nIt loops over pairs.",
    "  import math\n\n" + FN,                                                  # indented first line, column-0 def
    "```python\nprint(has_close_elements([1.0], 0.1))\n```\nand\n```python\n" + FN + "```\n",  # entry point in 2nd fence
    "Sure!\n```python\n" + FN + "```\nThis works.",
])
def test_correct_chat_replies_in_common_formats_pass(reply):
    prog, mode = he.build_program(_rec("chat", [reply]), reply)
    assert mode == "chat_function", mode
    ok, reason = he.run_program(prog, timeout=3.0)
    assert ok, reason


def test_fenced_body_fragment_keeps_its_indentation():
    reply = "```python\n" + BODY + "```"
    prog, mode = he.build_program(_rec("chat", [reply]), reply)
    assert mode == "chat_body"
    assert he.run_program(prog, timeout=3.0)[0]


def test_unfenced_body_fragment_still_appended_as_written():
    frag = "  return abs(number) % 1"
    prog, mode = he.build_program(_rec("chat", [frag]), frag)
    assert mode == "chat_body" and prog.startswith(PROMPT + frag)


def test_wrong_chat_reply_still_fails():
    reply = "[PYTHON]\ndef has_close_elements(numbers, threshold):\n    return True\n[/PYTHON]"
    assert not he.run_program(he.build_program(_rec("chat", [reply]), reply)[0], timeout=3.0)[0]


# ---------- sandbox (code review MEDIUM #8) ----------

def test_program_runs_outside_the_repo(tmp_path):
    import os
    here = os.getcwd()
    sample = f"    import os\n    assert os.getcwd() != {here!r}\n" + BODY
    assert he.run_program(he.build_program(_rec("base", [sample]), sample)[0], timeout=3.0)[0]


# ---------- legacy-grader guard (code review HIGH #2) ----------

def test_cli_refuses_legacy_grader_on_regen_records(tmp_path, monkeypatch):
    import json
    from bench.eval import __main__ as cli
    rec = _rec("base", [BODY])
    rec.update(model="m/x", method="temp", temperature=0.7)
    results = tmp_path / "temp_t0.7.jsonl"
    results.write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr("sys.argv", ["bench.eval", "--results-file", str(results), "--dataset", "humaneval",
                                     "--skip-diversity", "--workers", "1", "--output", str(tmp_path / "m.json")])
    with pytest.raises(SystemExit):
        cli.main()


# ---------- indented first def + column-0 helper (code re-review 2026-11-06, blocker) ----------

def test_indented_entry_def_with_column0_helper_passes():
    reply = ("  def has_close_elements(numbers, threshold):\n" + BODY.replace("return True", "return _hit()")
             + "\n\ndef _hit():\n    return True\n")
    prog, mode = he.build_program(_rec("chat", [reply]), reply)
    assert mode == "chat_function", mode
    assert he.run_program(prog, timeout=3.0)[0]


def test_uniformly_indented_reply_still_dedents():
    # Must not be broken by the first-line fix: the whole block is indented by 2.
    reply = "  from typing import List\n  def has_close_elements(numbers, threshold):\n" + "\n".join(
        "  " + ln for ln in BODY.splitlines()) + "\n"
    prog, mode = he.build_program(_rec("chat", [reply]), reply)
    assert mode == "chat_function", mode
    assert he.run_program(prog, timeout=3.0)[0]


def test_real_codellama_reply_humaneval_107():
    datasets = pytest.importorskip("datasets")
    try:
        task = next(t for t in datasets.load_dataset("openai/openai_humaneval", split="test") if t["task_id"] == "HumanEval/107")
    except Exception as e:
        pytest.skip(f"dataset unavailable: {e}")
    reply = ('  def even_odd_palindrome(n):\n    even_count = 0\n    odd_count = 0\n    for i in range(1, n+1):\n'
             '        if is_palindrome(i) and i % 2 == 0:\n            even_count += 1\n'
             '        elif is_palindrome(i) and i % 2 == 1:\n            odd_count += 1\n'
             '    return (even_count, odd_count)\n\ndef is_palindrome(n):\n    return str(n) == str(n)[::-1]')
    rec = {"task_id": task["task_id"], "prompt_style": "chat", "task_prompt": task["prompt"], "test": task["test"],
           "entry_point": task["entry_point"], "samples": [reply]}
    prog, mode = he.build_program(rec, reply)
    assert mode == "chat_function"
    assert he.run_program(prog, timeout=3.0)[0]
