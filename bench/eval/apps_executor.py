"""APPS-aware test execution.

The MBPP/HumanEval executors run the model's code with ``stdin=DEVNULL`` and
call the function from a test harness appended inside the same ``python3 -c``
process. APPS doesn't work that way — every ATCODER/CODEFORCES problem in
the canonical dataset reads from stdin and writes to stdout, so we must:

  * Pipe each test case's input into the subprocess's stdin.
  * Capture stdout, compare to the expected output (with whitespace
    normalisation matching the algosim paper's protocol).
  * Halt on first mismatch and record which test case failed (so the
    diagnostics block can tell "wrong on test 0" apart from "wrong on
    test 47 of 50").

We also support the ``fn_name`` interface for forward-compatibility, even
though every ATCODER/CODEFORCES problem in scope is stdin/stdout — see
``bench/apps/dataset.py`` for the empirical confirmation.

Status taxonomy mirrors Lee et al. 2025 Table 7 to make our numbers
comparable with theirs:

  * ``Passed``       — all test cases matched
  * ``Failed``       — at least one test case had wrong stdout
  * ``RuntimeError`` — subprocess exited non-zero with a traceback
  * ``Timeout``      — at least one test case exceeded the per-case budget
  * ``SyntaxError``  — compile() raised before we ran any test
  * ``ParsingError`` — extraction failed (no compilable code) → ``pass_results=False``

``pass_results[i]`` (the boolean array fed to ``compute_pass_at_k``) is
True only for ``Passed``; the richer status is preserved in the
diagnostics block.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

from bench.apps.dataset import AppsProblem
from bench.eval.apps_extractor import (
    STRATEGY_NONE,
    ExtractionResult,
    extract_python_code_apps,
)

StatusT = Literal[
    "Passed", "Failed", "RuntimeError", "Timeout", "SyntaxError", "ParsingError"
]


@dataclass
class AppsSampleResult:
    status: StatusT
    n_tests_total: int = 0
    n_tests_passed: int = 0
    first_failing_idx: int | None = None
    stderr_excerpt: str | None = None


# ── Output normalisation ─────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _normalise_output(text: str) -> str:
    """Algosim-paper-style normalisation: strip per-line trailing whitespace,
    drop fully-blank trailing lines, collapse all-whitespace lines to ''."""
    if text is None:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _outputs_equal(actual: str, expected: str) -> bool:
    """Compare normalised stdout. Pure-numeric outputs get a tiny float tolerance."""
    a = _normalise_output(actual)
    e = _normalise_output(expected)
    if a == e:
        return True
    # Numeric tolerance — only when *both* sides are a single number
    if _NUMERIC_RE.match(a) and _NUMERIC_RE.match(e):
        try:
            return abs(float(a) - float(e)) < 1e-6
        except ValueError:
            return False
    return False


# ── Execution paths ──────────────────────────────────────────────────────────


def _run_stdin_test(
    code: str,
    stdin_str: str,
    timeout: float,
) -> tuple[int, str, str, bool]:
    """Run the program with ``stdin_str`` piped in.

    The program is written to a temp ``.py`` file and run as ``python3 <file>``
    rather than passed via ``python3 -c <code>``: Linux caps a single argv
    string at MAX_ARG_STRLEN (128 KiB), so long generated solutions passed via
    ``-c`` raise ``OSError: [Errno 7] Argument list too long``. Running a file
    is behaviour-equivalent (``__name__ == "__main__"`` still holds, stdin still
    piped) and has no length limit.

    Returns ``(returncode, stdout, stderr, timed_out)``.
    """
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        proc = subprocess.run(
            ["python3", path],
            input=stdin_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired:
        return -1, "", "", True
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _make_fn_call_harness(code: str, fn_name: str) -> str:
    """Wrap user code with a harness that consumes (inputs, outputs) JSON from argv."""
    # The harness reads two argv strings (inputs json, outputs json), iterates,
    # and exits with code 0 only if every assertion passes. Failures exit 1.
    return (
        code
        + "\n\n# ── harness injected by bench.eval.apps_executor ──\n"
        + "import json as _json\n"
        + "import sys as _sys\n"
        + "_inputs = _json.loads(_sys.argv[1])\n"
        + "_outputs = _json.loads(_sys.argv[2])\n"
        + "for _i, (_inp, _want) in enumerate(zip(_inputs, _outputs)):\n"
        + "    try:\n"
        + f"        _got = {fn_name}(*_inp) if isinstance(_inp, list) else {fn_name}(_inp)\n"
        + "    except Exception as _e:\n"
        + "        print(f'_FAIL_AT_{_i}_RUNTIME', file=_sys.stderr); _sys.exit(2)\n"
        + "    if _got != _want:\n"
        + "        print(f'_FAIL_AT_{_i}_MISMATCH', file=_sys.stderr); _sys.exit(1)\n"
    )


def _run_fn_call_tests(
    code: str,
    fn_name: str,
    inputs: list,
    outputs: list,
    timeout: float,
) -> AppsSampleResult:
    program = _make_fn_call_harness(code, fn_name)
    try:
        proc = subprocess.run(
            ["python3", "-c", program, json.dumps(inputs), json.dumps(outputs)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return AppsSampleResult(status="Timeout", n_tests_total=len(inputs))
    if proc.returncode == 0:
        return AppsSampleResult(
            status="Passed", n_tests_total=len(inputs), n_tests_passed=len(inputs)
        )
    err = proc.stderr or ""
    m = re.search(r"_FAIL_AT_(\d+)_(\w+)", err)
    first_idx = int(m.group(1)) if m else None
    fail_type = m.group(2) if m else "UNKNOWN"
    return AppsSampleResult(
        status="RuntimeError" if fail_type == "RUNTIME" else "Failed",
        n_tests_total=len(inputs),
        n_tests_passed=first_idx if first_idx is not None else 0,
        first_failing_idx=first_idx,
        stderr_excerpt=err[:200] if err else None,
    )


def _run_stdin_tests(
    code: str,
    inputs: list,
    outputs: list,
    per_test_timeout: float,
) -> AppsSampleResult:
    """Run a stdin/stdout program against each (input, output) pair, halting
    at the first failure. Paper protocol."""
    n_total = len(inputs)
    for i, (inp, expected) in enumerate(zip(inputs, outputs)):
        # APPS sometimes encodes inputs/outputs as lists of strings, sometimes
        # as a single string; coerce to a single string here for stdin piping.
        inp_str = inp if isinstance(inp, str) else "\n".join(map(str, inp)) + "\n"
        expected_str = expected if isinstance(expected, str) else str(expected)
        rc, stdout, stderr, timed_out = _run_stdin_test(code, inp_str, per_test_timeout)
        if timed_out:
            return AppsSampleResult(
                status="Timeout",
                n_tests_total=n_total,
                n_tests_passed=i,
                first_failing_idx=i,
                stderr_excerpt="(timeout)",
            )
        if rc != 0:
            return AppsSampleResult(
                status="RuntimeError",
                n_tests_total=n_total,
                n_tests_passed=i,
                first_failing_idx=i,
                stderr_excerpt=(stderr or "")[:200] or None,
            )
        if not _outputs_equal(stdout, expected_str):
            return AppsSampleResult(
                status="Failed",
                n_tests_total=n_total,
                n_tests_passed=i,
                first_failing_idx=i,
                stderr_excerpt=None,
            )
    return AppsSampleResult(status="Passed", n_tests_total=n_total, n_tests_passed=n_total)


def evaluate_apps_sample(
    raw_sample: str,
    problem: AppsProblem,
    per_test_timeout: float = 10.0,
) -> tuple[AppsSampleResult, ExtractionResult]:
    """End-to-end: extract → compile-check → run tests. Returns (status, extraction)."""
    ext = extract_python_code_apps(raw_sample, fn_name=problem.fn_name)
    if not ext.success or not ext.code:
        return AppsSampleResult(status="ParsingError",
                                n_tests_total=len(problem.inputs)), ext
    # Defensive: re-check the chosen candidate compiles (the extractor already
    # promised this, but verify so a bug there doesn't show up as RuntimeError).
    try:
        compile(ext.code, "<sample>", "exec")
    except SyntaxError as e:
        return AppsSampleResult(
            status="SyntaxError",
            n_tests_total=len(problem.inputs),
            stderr_excerpt=str(e)[:200],
        ), ext
    if not problem.has_test_data:
        # No tests means we can't evaluate — treat as a kind of ParsingError
        # for stats purposes (consistent with "we couldn't determine pass/fail").
        return AppsSampleResult(status="ParsingError",
                                n_tests_total=0,
                                stderr_excerpt="problem has no test data"), ext
    if problem.fn_name:
        res = _run_fn_call_tests(ext.code, problem.fn_name, problem.inputs,
                                 problem.outputs, per_test_timeout)
    else:
        res = _run_stdin_tests(ext.code, problem.inputs, problem.outputs,
                               per_test_timeout)
    return res, ext


# ── Parallel orchestration ───────────────────────────────────────────────────


@dataclass
class TaskResult:
    """One row in the per-task results list consumed by metrics.compute_*."""
    task_id: int
    num_correct: int
    pass_results: list[bool]
    statuses: list[str] = field(default_factory=list)
    n_tests_total: list[int] = field(default_factory=list)
    n_tests_passed: list[int] = field(default_factory=list)
    first_failing_idx: list[int | None] = field(default_factory=list)
    extracted_codes: list[str] = field(default_factory=list)   # for diversity metrics
    extraction_strategies: list[str] = field(default_factory=list)
    extraction_success: list[bool] = field(default_factory=list)


def _evaluate_one_sample(args):
    """Evaluate ONE sample of one task — the parallelism unit.

    Per-sample (not per-task) so the pool scales with the total number of
    samples, and a single slow problem can't serialize its samples behind one
    worker. ``problem_pickle`` is a plain dict because AppsProblem doesn't
    pickle cleanly through every executor backend; rehydrate it here.
    """
    tid, idx, sample, problem_pickle, per_test_timeout = args
    problem = AppsProblem(**problem_pickle)
    r, e = evaluate_apps_sample(sample, problem, per_test_timeout=per_test_timeout)
    return tid, idx, r, e


def _regroup_results(completed, n_samples_by_task):
    """Place per-sample ``(tid, idx, result, extraction)`` tuples — which arrive
    in ARBITRARY completion order from the pool — back into per-task lists
    ordered by sample index, so ``pass_results[i]`` stays aligned with
    ``samples[i]``. This index-alignment is the correctness invariant of the
    per-sample parallelization.
    """
    results_by_task = {tid: [None] * n for tid, n in n_samples_by_task.items()}
    extractions_by_task = {tid: [None] * n for tid, n in n_samples_by_task.items()}
    for tid, idx, r, e in completed:
        results_by_task[tid][idx] = r
        extractions_by_task[tid][idx] = e
    return results_by_task, extractions_by_task


def evaluate_all_apps(
    records: list[dict],
    problems_by_id: dict[int, AppsProblem],
    *,
    per_test_timeout: float = 10.0,
    workers: int = 4,
) -> tuple[list[TaskResult], dict, dict]:
    """Run the APPS evaluation across every (task, sample) in ``records``.

    Parallelizes over **(task, sample)** pairs, so throughput scales with the
    total number of samples (not the number of problems) and one slow problem
    no longer serializes its samples behind a single worker. Returns
    (per_task_results, extraction_diagnostics, execution_diagnostics);
    per-task results are shaped to plug into bench.eval.metrics' aggregators.
    """
    # Per-(task, sample) work items. (problem_pickle is duplicated across a
    # task's samples in the queue — small payloads, negligible vs the win.)
    work = []
    skipped_no_problem: list[int] = []
    n_samples_by_task: dict[int, int] = {}
    for rec in records:
        tid = int(rec["task_id"])
        problem = problems_by_id.get(tid)
        if problem is None:
            skipped_no_problem.append(tid)
            continue
        problem_pickle = dict(
            problem_id=problem.problem_id,
            source=problem.source,
            difficulty=problem.difficulty,
            question=problem.question,
            starter_code=problem.starter_code,
            fn_name=problem.fn_name,
            inputs=problem.inputs,
            outputs=problem.outputs,
        )
        samples = rec["samples"]
        n_samples_by_task[tid] = len(samples)
        for idx, s in enumerate(samples):
            work.append((tid, idx, s, problem_pickle, per_test_timeout))

    # Evaluate (parallel over samples; sequential fallback for workers <= 1).
    completed = []
    if workers <= 1:
        for w in work:
            completed.append(_evaluate_one_sample(w))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_evaluate_one_sample, w) for w in work]
            for fut in as_completed(futures):
                completed.append(fut.result())

    results_by_task, extractions_by_task = _regroup_results(completed, n_samples_by_task)

    # Per-task aggregation + diagnostics, in deterministic task order.
    task_results: list[TaskResult] = []
    n_samples_total = 0
    strat_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {s: 0 for s in
        ("Passed", "Failed", "RuntimeError", "Timeout", "SyntaxError", "ParsingError")}
    first_fail_idxs: list[int] = []
    for tid in sorted(n_samples_by_task):
        results = results_by_task[tid]
        extractions = extractions_by_task[tid]
        task_results.append(_to_task_result(tid, results, extractions))
        n_samples_total += len(results)
        for ext in extractions:
            strat_counts[ext.strategy] = strat_counts.get(ext.strategy, 0) + 1
        for r in results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
            if r.first_failing_idx is not None:
                first_fail_idxs.append(r.first_failing_idx)
    extraction_diag = {
        "n_samples_total": n_samples_total,
        "n_records_skipped_no_problem": len(skipped_no_problem),
        "by_strategy": dict(sorted(strat_counts.items(), key=lambda kv: -kv[1])),
        "n_extraction_success": n_samples_total - strat_counts.get(STRATEGY_NONE, 0),
        "n_extraction_failed":  strat_counts.get(STRATEGY_NONE, 0),
    }
    execution_diag = {
        "by_status": status_counts,
        "pass_rate": status_counts.get("Passed", 0) / n_samples_total if n_samples_total else 0.0,
        "first_failing_test_idx": _percentile_dict(first_fail_idxs),
    }
    return task_results, extraction_diag, execution_diag


def _to_task_result(
    task_id: int,
    results: list[AppsSampleResult],
    extractions: list[ExtractionResult],
) -> TaskResult:
    pass_results = [r.status == "Passed" for r in results]
    return TaskResult(
        task_id=task_id,
        num_correct=sum(pass_results),
        pass_results=pass_results,
        statuses=[r.status for r in results],
        n_tests_total=[r.n_tests_total for r in results],
        n_tests_passed=[r.n_tests_passed for r in results],
        first_failing_idx=[r.first_failing_idx for r in results],
        extracted_codes=[e.code for e in extractions],
        extraction_strategies=[e.strategy for e in extractions],
        extraction_success=[e.success for e in extractions],
    )


def _percentile_dict(xs: list[int]) -> dict[str, int | None]:
    if not xs:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    s = sorted(xs)
    n = len(s)
    return {
        "p50": s[int(0.5 * (n - 1))],
        "p90": s[int(0.9 * (n - 1))],
        "p99": s[int(0.99 * (n - 1))],
        "max": s[-1],
    }
