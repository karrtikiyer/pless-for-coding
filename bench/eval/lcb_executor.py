"""LiveCodeBench executor — mirrors bench.eval.apps_executor.

Reuses the APPS stdin runner, code extractor, per-task aggregation, and diagnostics
verbatim; the only LCB-specific piece is the FUNCTIONAL (LeetCode) harness, which
calls ``Solution().<fn_name>(*args)`` (a method on class Solution) with
``from typing import *`` injected — unlike APPS's bare-function harness. stdin
(AtCoder/CodeForces) problems go straight through the reused APPS stdin path.

Test-map problems come from bench.livecodebench.dataset.load_lcb_test_map, keyed by
task_id (== question_id, a string). fn_name set => functional; None => stdin.
"""
from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from bench.eval.apps_executor import (
    AppsSampleResult,
    _percentile_dict,
    _regroup_results,
    _run_stdin_tests,
    _to_task_result,
)
from bench.eval.apps_extractor import (
    STRATEGY_NONE,
    ExtractionResult,
    extract_python_code_apps,
)
from bench.livecodebench.dataset import LcbProblem


def _make_lcb_fn_harness(code: str, fn_name: str) -> str:
    """Wrap LeetCode-style code (defines class Solution) with a call harness.

    Reads (inputs, outputs) JSON from argv: inputs[i] is the arg-list for one call,
    outputs[i] the expected return. Exits 0 iff every call matches (float-tolerant
    for numeric returns); 1 on mismatch, 2 on runtime error — encoding the failing
    index in stderr for diagnostics (same protocol as apps_executor's harness)."""
    return (
        "from typing import *\n"
        "import json as _json\n"
        "import sys as _sys\n"
        + code
        + "\n\n# ── harness injected by bench.eval.lcb_executor ──\n"
        + "def _cmp(_g, _w):\n"
        + "    if _g == _w: return True\n"
        + "    try:\n"
        + "        if isinstance(_g,(int,float)) and isinstance(_w,(int,float)):\n"
        + "            return abs(float(_g)-float(_w)) < 1e-6\n"
        + "    except Exception: pass\n"
        + "    return False\n"
        + "_inputs = _json.loads(_sys.argv[1])\n"
        + "_outputs = _json.loads(_sys.argv[2])\n"
        + "_sol = Solution()\n"
        + f"_fn = getattr(_sol, {fn_name!r})\n"
        + "for _i, (_args, _want) in enumerate(zip(_inputs, _outputs)):\n"
        + "    try:\n"
        + "        _got = _fn(*_args)\n"
        + "    except Exception as _e:\n"
        + "        print(f'_FAIL_AT_{_i}_RUNTIME', file=_sys.stderr); _sys.exit(2)\n"
        + "    if not _cmp(_got, _want):\n"
        + "        print(f'_FAIL_AT_{_i}_MISMATCH', file=_sys.stderr); _sys.exit(1)\n"
    )


def _run_lcb_fn_tests(code: str, fn_name: str, inputs: list, outputs: list,
                      timeout: float) -> AppsSampleResult:
    """Functional (LeetCode) execution: Solution().fn(*args) vs expected return."""
    program = _make_lcb_fn_harness(code, fn_name)
    try:
        proc = subprocess.run(
            ["python3", "-c", program, json.dumps(inputs), json.dumps(outputs)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return AppsSampleResult(status="Timeout", n_tests_total=len(inputs))
    if proc.returncode == 0:
        return AppsSampleResult(status="Passed", n_tests_total=len(inputs),
                                n_tests_passed=len(inputs))
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


def evaluate_lcb_sample(raw_sample: str, problem: LcbProblem,
                        per_test_timeout: float = 10.0
                        ) -> tuple[AppsSampleResult, ExtractionResult]:
    """Extract → compile-check → run tests (stdin or functional)."""
    ext = extract_python_code_apps(raw_sample, fn_name=problem.fn_name)
    if not ext.success or not ext.code:
        return AppsSampleResult(status="ParsingError",
                                n_tests_total=len(problem.inputs)), ext
    try:
        compile(ext.code, "<sample>", "exec")
    except SyntaxError as e:
        return AppsSampleResult(status="SyntaxError", n_tests_total=len(problem.inputs),
                                stderr_excerpt=str(e)[:200]), ext
    if not problem.has_test_data:
        return AppsSampleResult(status="ParsingError", n_tests_total=0,
                                stderr_excerpt="problem has no test data"), ext
    if problem.fn_name:
        res = _run_lcb_fn_tests(ext.code, problem.fn_name, problem.inputs,
                                problem.outputs, per_test_timeout)
    else:
        res = _run_stdin_tests(ext.code, problem.inputs, problem.outputs,
                               per_test_timeout)
    return res, ext


def _evaluate_one_lcb_sample(args):
    tid, idx, sample, problem_pickle, per_test_timeout = args
    problem = LcbProblem(**problem_pickle)
    r, e = evaluate_lcb_sample(sample, problem, per_test_timeout=per_test_timeout)
    return tid, idx, r, e


def evaluate_all_lcb(
    records: list[dict],
    problems_by_id: dict[str, LcbProblem],
    *,
    per_test_timeout: float = 10.0,
    workers: int = 4,
) -> tuple[list, dict, dict]:
    """Run LCB evaluation across every (task, sample). Mirrors evaluate_all_apps;
    join key is the string task_id (== question_id)."""
    work = []
    skipped_no_problem: list[str] = []
    n_samples_by_task: dict[str, int] = {}
    for rec in records:
        tid = str(rec["task_id"])
        problem = problems_by_id.get(tid)
        if problem is None:
            skipped_no_problem.append(tid)
            continue
        problem_pickle = dict(
            task_id=problem.task_id, platform=problem.platform,
            difficulty=problem.difficulty, contest_date=problem.contest_date,
            question=problem.question, starter_code=problem.starter_code,
            fn_name=problem.fn_name, inputs=problem.inputs, outputs=problem.outputs,
        )
        samples = rec["samples"]
        n_samples_by_task[tid] = len(samples)
        for idx, s in enumerate(samples):
            work.append((tid, idx, s, problem_pickle, per_test_timeout))

    completed = []
    if workers <= 1:
        for w in work:
            completed.append(_evaluate_one_lcb_sample(w))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_evaluate_one_lcb_sample, w) for w in work]
            for fut in as_completed(futures):
                completed.append(fut.result())

    results_by_task, extractions_by_task = _regroup_results(completed, n_samples_by_task)

    task_results = []
    n_samples_total = 0
    strat_counts: dict[str, int] = {}
    status_counts = {s: 0 for s in
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
        "n_extraction_failed": strat_counts.get(STRATEGY_NONE, 0),
    }
    execution_diag = {
        "by_status": status_counts,
        "pass_rate": status_counts.get("Passed", 0) / n_samples_total if n_samples_total else 0.0,
        "first_failing_test_idx": _percentile_dict(first_fail_idxs),
    }
    return task_results, extraction_diag, execution_diag
