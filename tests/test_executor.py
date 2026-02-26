import pytest

from bench.eval.executor import (
    _build_program_humaneval,
    _build_program_mbpp,
    check_sample,
    evaluate_task,
    evaluate_all,
    strip_code_fences,
)


# --- check_sample ---

def test_check_sample_passing():
    code = "def add(a, b): return a + b\nassert add(1, 2) == 3"
    assert check_sample("", code) is True


def test_check_sample_failing():
    code = "def add(a, b): return a - b\nassert add(1, 2) == 3"
    assert check_sample("", code) is False


def test_check_sample_syntax_error():
    code = "def add(a b): return a + b"
    assert check_sample("", code) is False


# --- strip_code_fences ---

def test_strip_code_fences_python():
    code = "```python\ndef f(x): return x + 1\n```"
    assert strip_code_fences(code) == "def f(x): return x + 1\n"


def test_strip_code_fences_no_language():
    code = "```\ndef f(x): return x + 1\n```"
    assert strip_code_fences(code) == "def f(x): return x + 1\n"


def test_strip_code_fences_no_fences():
    code = "def f(x): return x + 1"
    assert strip_code_fences(code) == "def f(x): return x + 1"


def test_strip_code_fences_with_whitespace():
    code = "  ```python\ndef f(x): return x + 1\n```  "
    assert strip_code_fences(code) == "def f(x): return x + 1\n"


def test_strip_code_fences_multiline():
    code = "```python\ndef f(x):\n    if x > 0:\n        return x\n    return -x\n```"
    expected = "def f(x):\n    if x > 0:\n        return x\n    return -x\n"
    assert strip_code_fences(code) == expected


def test_evaluate_task_with_code_fences():
    """Samples wrapped in code fences should still be evaluated correctly."""
    record = {
        "task_id": 1,
        "samples": [
            "```python\ndef add(a, b): return a + b\n```",
            "```python\ndef add(a, b): return a - b\n```",
        ],
        "test_list": ["assert add(1, 2) == 3"],
    }
    result = evaluate_task(record, "mbpp")
    assert result["num_correct"] == 1
    assert result["pass_results"] == [True, False]


def test_check_sample_timeout():
    code = "import time; time.sleep(100)"
    assert check_sample("", code, timeout=0.5) is False


def test_check_sample_runtime_error():
    code = "1 / 0"
    assert check_sample("", code) is False


# --- _build_program_mbpp ---

def test_build_program_mbpp():
    sample = "def foo(x): return x + 1"
    tests = ["assert foo(1) == 2", "assert foo(0) == 1"]
    program = _build_program_mbpp(sample, tests)
    assert "def foo(x): return x + 1" in program
    assert "assert foo(1) == 2" in program
    assert "assert foo(0) == 1" in program


# --- _build_program_humaneval ---

def test_build_program_humaneval():
    sample = "def has_close_elements(numbers, threshold):\n    return False"
    test = "def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) == False"
    program = _build_program_humaneval(sample, test, "has_close_elements")
    assert "def has_close_elements" in program
    assert "def check(candidate)" in program
    assert "check(has_close_elements)" in program


# --- evaluate_task ---

def test_evaluate_task_mbpp():
    record = {
        "task_id": 1,
        "samples": [
            "def add(a, b): return a + b",
            "def add(a, b): return a - b",
            "def add(a, b): return a + b",
        ],
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }
    result = evaluate_task(record, "mbpp")
    assert result["task_id"] == 1
    assert result["num_correct"] == 2
    assert result["pass_results"] == [True, False, True]


def test_evaluate_task_humaneval():
    record = {
        "task_id": "HumanEval/0",
        "samples": [
            "def double(x):\n    return x * 2",
            "def double(x):\n    return x + x",
            "def double(x):\n    return x",
        ],
        "test": "def check(candidate):\n    assert candidate(3) == 6\n    assert candidate(0) == 0",
        "entry_point": "double",
    }
    result = evaluate_task(record, "humaneval")
    assert result["task_id"] == "HumanEval/0"
    assert result["num_correct"] == 2
    assert result["pass_results"] == [True, True, False]


def test_evaluate_task_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        evaluate_task({"task_id": 1, "samples": ["pass"]}, "unknown")


# --- evaluate_all ---

def test_evaluate_all():
    records = [
        {
            "task_id": 1,
            "samples": ["def f(): return 1", "def f(): return 2"],
            "test_list": ["assert f() == 1"],
        },
        {
            "task_id": 2,
            "samples": ["def g(): return 'a'", "def g(): return 'b'"],
            "test_list": ["assert g() == 'a'"],
        },
    ]
    results = evaluate_all(records, "mbpp", timeout=5.0, workers=2)
    assert len(results) == 2
    # Should be sorted by task_id
    assert results[0]["task_id"] == 1
    assert results[1]["task_id"] == 2
    assert results[0]["num_correct"] == 1
    assert results[1]["num_correct"] == 1
