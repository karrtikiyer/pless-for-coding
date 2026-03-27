from __future__ import annotations

import re
import subprocess
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed

_CODE_FENCE_RE = re.compile(r"```\w*\n(.*?)```", re.DOTALL)
_CHECK_OR_MAIN_RE = re.compile(
    r"^(def check\(|if __name__\s*==)", re.MULTILINE,
)


def strip_code_fences(code: str) -> str:
    """Extract code from the first markdown code fence, if present."""
    m = _CODE_FENCE_RE.search(code.strip())
    return m.group(1) if m else code


def _strip_check_and_main(code: str) -> str:
    """Remove model-generated ``def check(...)`` and ``if __name__`` blocks.

    Models (especially Qwen) sometimes generate their own test harness after
    the solution.  When the evaluation pipeline appends the canonical HumanEval
    test, the model's ``if __name__ == '__main__': check(...)`` block runs
    first (because ``python -c`` sets ``__name__`` to ``'__main__'``), and any
    incorrect assertion in the model's check causes a false negative.

    This function removes such blocks by finding the first top-level
    ``def check(`` or ``if __name__`` line and truncating the code there.
    """
    m = _CHECK_OR_MAIN_RE.search(code)
    if m is None:
        return code
    # Truncate at the start of the match, strip trailing whitespace
    return code[: m.start()].rstrip()


def _strip_after_function(code: str) -> str:
    """Strip 'next problem' continuations that follow the target function.

    Base models often generate: def func(): ... \\n\\n\"\"\"Next problem\"\"\"\\nassert ...
    This is valid Python, so _trim_to_compilable accepts it, but the stray
    assert calls crash with NameError before real tests run.
    """
    lines = code.split('\n')
    # Find first top-level def
    def_idx = None
    for i, line in enumerate(lines):
        if line.startswith('def '):
            def_idx = i
            break
    if def_idx is None:
        return code

    # Walk past the function body
    found_body = False
    for i in range(def_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue  # skip blank lines
        if lines[i][0] in (' ', '\t'):
            found_body = True
        elif found_body:
            # Another top-level def is likely a helper function — keep it
            # and continue scanning past its body too.
            if lines[i].startswith('def '):
                found_body = False
                continue
            # First non-def, non-indented, non-blank line after body → truncate
            return '\n'.join(lines[:i]).rstrip()

    return code


def _trim_to_compilable(code: str) -> str | None:
    """Return the longest line-prefix of *code* that compiles, or None."""
    try:
        compile(code, "<sample>", "exec")
        return code
    except SyntaxError:
        pass
    except (MemoryError, RecursionError):
        # Pathologically long/deeply nested code (e.g. [BEGIN]\n[END]\n... cycling)
        # overflows the CPython parser stack — treat as uncompilable.
        return None

    lines = code.split("\n")
    for end in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            compile(candidate, "<sample>", "exec")
            return candidate
        except SyntaxError:
            continue
        except (MemoryError, RecursionError):
            continue
    return None


def extract_python_code(code: str) -> str:
    """Extract Python code, removing trailing non-Python garbage.

    Base models (e.g. Qwen-7B) often continue generating past the Python
    solution, appending Java, JavaScript, HTML, markdown, or repeated
    patterns.

    Strategy:
      1. Try trimming the raw code first (handles base-model output where the
         function is plain text followed by garbage).
      2. Fall back to stripping code fences and trimming (handles
         instruct-model output wrapped in ```python ... ```).
      3. Return the original if nothing compiles.

    As a final step, strip any model-generated ``def check()`` or
    ``if __name__`` blocks that would interfere with the canonical test
    harness appended by the evaluation pipeline.
    """
    # Strategy 1: trim raw code directly
    result = _trim_to_compilable(code)
    if result is not None:
        return _strip_check_and_main(_strip_after_function(result))

    # Strategy 2: strip code fences first, then trim
    stripped = strip_code_fences(code)
    if stripped != code:
        result = _trim_to_compilable(stripped)
        if result is not None:
            return _strip_check_and_main(_strip_after_function(result))
        # Also try dedenting the stripped code
        dedented = textwrap.dedent(stripped)
        if dedented != stripped:
            result = _trim_to_compilable(dedented)
            if result is not None:
                return _strip_check_and_main(_strip_after_function(result))

    # Strategy 3: dedent and retry (some models, e.g. CodeLlama-Instruct,
    # generate top-level code with leading indentation)
    dedented = textwrap.dedent(code)
    if dedented != code:
        result = _trim_to_compilable(dedented)
        if result is not None:
            return _strip_check_and_main(_strip_after_function(result))

    # Nothing compiled — return original and let execution handle it
    return code


def _build_program_mbpp(sample_code: str, test_list: list[str], test_setup_code: str = "") -> str:
    parts = []
    if test_setup_code:
        parts.append(test_setup_code)
    parts.append(sample_code)
    parts.append("\n".join(test_list))
    return "\n".join(parts)


def _build_program_humaneval(sample_code: str, test: str, entry_point: str) -> str:
    return sample_code + "\n" + test + f"\ncheck({entry_point})\n"


def check_sample(program: str, timeout: float = 5.0) -> bool:
    """Run program in a subprocess, return True if it exits cleanly."""
    try:
        result = subprocess.run(
            ["python3", "-c", program],
            timeout=timeout,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def evaluate_task(record: dict, dataset: str, timeout: float = 5.0) -> dict:
    """Run all samples for one task, return results dict."""
    samples = record["samples"]
    pass_results = []

    for sample in samples:
        sample = extract_python_code(sample)
        if dataset == "mbpp":
            program = _build_program_mbpp(
                sample, record["test_list"], record.get("test_setup_code", "")
            )
        elif dataset == "humaneval":
            program = _build_program_humaneval(
                sample, record["test"], record["entry_point"]
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        passed = check_sample(program, timeout=timeout)
        pass_results.append(passed)

    return {
        "task_id": record["task_id"],
        "num_correct": sum(pass_results),
        "pass_results": pass_results,
    }


def _evaluate_task_wrapper(args):
    """Wrapper for ProcessPoolExecutor (needs top-level picklable callable)."""
    record, dataset, timeout = args
    return evaluate_task(record, dataset, timeout)


def evaluate_all(
    records: list[dict],
    dataset: str,
    timeout: float = 5.0,
    workers: int = 4,
) -> list[dict]:
    """Evaluate all tasks in parallel."""
    args_list = [(r, dataset, timeout) for r in records]
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_evaluate_task_wrapper, args): args[0]["task_id"]
            for args in args_list
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by task_id for deterministic output
    results.sort(key=lambda r: r["task_id"])
    return results
