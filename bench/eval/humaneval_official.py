"""Official-composition HumanEval grader (2026-11 regeneration).

Spec: docs/experiments/humaneval_regen_2026_11_plan.md §7. Mirrors openai/human-eval
(human_eval/execution.py): the program is

    task_prompt + completion + "\\n" + test + "\\n" + f"check({entry_point})"

executed in fresh globals, with ``exit``/``quit`` disabled and any exception (including
``SystemExit``) counted as a failure; timeout 3.0 s (human_eval/evaluation.py default).

Records come from bench/humaneval/regen.py and carry ``task_prompt`` (the raw HumanEval prompt)
and ``prompt_style`` ("base" | "chat"):

* base: the stored sample is the completion only; it is re-cut at the BigCode stop list
  (idempotent when the generator already cut it) and appended to the prompt.
* chat: a reply whose extracted code defines ``def <entry_point>`` at column 0 is appended to the
  prompt after a newline, so the reply's function replaces the stub while the prompt's imports stay;
  any other reply (a body fragment) is appended exactly as written, as the official harness does with
  any completion.

The older no-prompt builder in bench/eval/executor.py is left unchanged (MBPP and historical files).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from bench.eval.executor import extract_python_code
from bench.humaneval.prompts import BIGCODE_HUMANEVAL_STOP_SEQUENCES, truncate_at_stop  # noqa: F401 (re-exported)

GRADER_VERSION = "humaneval-official-v1"
DEFAULT_TIMEOUT = 3.0
PYTHON = sys.executable  # the running (venv) interpreter, never `python3` from PATH

# Runs inside the child interpreter. The program arrives on stdin; success is signalled by a
# per-call random token written only after exec() returns, so a program that exits early through
# any route (sys.exit, os._exit, exit/quit) can never be counted as a pass.
_HARNESS = r"""
import builtins, os, sys
_token = sys.argv[1]
_src = sys.stdin.read()
_out, _exit = sys.__stdout__, os._exit
builtins.exit = None
builtins.quit = None
os._exit = None
try:
    exec(compile(_src, "<humaneval>", "exec"), {})
except BaseException as _e:
    sys.__stderr__.write("failed: %s: %s" % (type(_e).__name__, str(_e)[:300]))
    sys.__stderr__.flush()
    _exit(1)
_out.write(_token)
_out.flush()
_exit(0)
"""


MEMORY_LIMIT_BYTES = 4 * 1024**3   # per program, Linux only (macOS does not enforce RLIMIT_AS)


def _limit_memory() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def _defines_entry_point(code: str, entry_point: str) -> bool:
    return re.search(rf"^def\s+{re.escape(entry_point)}\s*\(", code, re.M) is not None


_FENCE_BLOCK_RE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.S)


def normalize_chat_reply(reply: str, entry_point: str) -> tuple[str, bool]:
    """Code text of a chat reply and whether it came from a fenced block.

    * CodeLlama-Instruct's own ``[PYTHON] ... [/PYTHON]`` tags are treated as a fence.
    * With several fenced blocks, the first one that defines ``def <entry_point>`` is used
      (else the first block).
    * The same rule is applied to every reply of every config, so it cannot favour a decoder.
    """
    text = reply.replace("[PYTHON]", "```python").replace("[/PYTHON]", "```")
    blocks = _FENCE_BLOCK_RE.findall(text)
    if blocks:
        chosen = next((b for b in blocks if _defines_entry_point(b, entry_point)), blocks[0])
        return chosen, True
    return text, False


def _strip_leading_indent(code: str, entry_point: str) -> str:
    """Repair mis-indented top-level lines at the start of a reply. Same rule for every config.

    1. A first line indented like ``  def <entry>(`` / ``  import`` / ``  from``, when the rest of the reply
       is NOT uniformly indented (it has a column-0 line, e.g. a helper ``def``), is un-indented
       (CodeLlama-Instruct writes ``  def f(n):\\n    body\\n\\ndef helper(...)``). A uniformly indented
       block is left for ``textwrap.dedent`` in ``extract_python_code``.
    2. When the entry point is then defined at column 0, indented top-level lines before the first column-0
       line (e.g. ``  import hashlib`` before ``def f``) are un-indented. Helper bodies are untouched.
    """
    lines = code.split("\n")
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is not None and lines[first][:1].isspace() and re.match(
            rf"\s+(def\s+{re.escape(entry_point)}\s*\(|import\s|from\s)", lines[first]):
        rest_has_col0 = any(ln.strip() and not ln[0].isspace() for ln in lines[first + 1:])
        if rest_has_col0:
            lines[first] = lines[first].lstrip()
    if not any(re.match(rf"def\s+{re.escape(entry_point)}\s*\(", ln) for ln in lines):
        return "\n".join(lines)
    first_col0 = next((i for i, ln in enumerate(lines) if ln.strip() and not ln[0].isspace()), len(lines))
    return "\n".join(ln.lstrip() if i < first_col0 else ln for i, ln in enumerate(lines))


def _build_solution(record: dict, sample: str) -> tuple[str, str]:
    """Return (prompt + completion, mode): the graded program without the test tail."""
    prompt, entry_point = record["task_prompt"], record["entry_point"]
    style = record.get("prompt_style")
    if style == "base":
        return prompt + truncate_at_stop(sample), "base"
    if style == "chat":
        text, fenced = normalize_chat_reply(sample, entry_point)
        text = _strip_leading_indent(text, entry_point)
        code = extract_python_code(text)
        if _defines_entry_point(code, entry_point):
            return prompt + "\n" + code, "chat_function"
        # A body fragment: appended as written (fence/tag markup removed, indentation kept).
        return prompt + (text if fenced else sample), "chat_body"
    raise ValueError(f"unknown prompt_style {style!r} for {record.get('task_id')}")


def build_program(record: dict, sample: str) -> tuple[str, str]:
    """Return (program, mode) for one sample; mode is base | chat_function | chat_body."""
    solution, mode = _build_solution(record, sample)
    return solution + "\n" + record["test"] + "\n" + f"check({record['entry_point']})", mode


def fingerprint_source(record: dict, sample: str) -> str:
    """Code used for AST fingerprints (cov@td): the program the grader executes minus the test tail.

    Base samples are function bodies (no ``def``) and do not parse alone; prompt + completion does.
    The prompt part is the same for every sample of a task, so it does not change which samples
    are distinct.
    """
    return _build_solution(record, sample)[0]


def run_program(program: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    """Execute ``program`` in a fresh interpreter; return (passed, reason)."""
    token = f"__HE_OK_{uuid.uuid4().hex}__"
    workdir = tempfile.mkdtemp(prefix="he_official_")   # never the repo directory
    try:
        proc = subprocess.run(
            [PYTHON, "-I", "-c", _HARNESS, token],
            input=program.encode("utf-8", "replace"),
            capture_output=True,
            timeout=timeout,
            cwd=workdir,
            preexec_fn=_limit_memory if sys.platform.startswith("linux") else None,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode == 0 and proc.stdout.endswith(token.encode()):
        return True, "passed"
    err = proc.stderr.decode("utf-8", "replace").strip()
    tagged = [ln for ln in err.splitlines() if ln.startswith("failed:")]
    if tagged:
        return False, tagged[-1][:300]
    last = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
    return False, f"failed: {last[:300]}"


def _status(passed: bool, reason: str) -> str:
    if passed:
        return "Passed"
    return "Timeout" if reason == "timed out" else "Failed"


def evaluate_task(record: dict, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Grade every sample of one task. Output keys extend bench.eval.executor.evaluate_task's."""
    pass_results, statuses, reasons, modes = [], [], [], []
    for sample in record["samples"]:
        program, mode = build_program(record, sample)
        passed, reason = run_program(program, timeout)
        pass_results.append(passed)
        statuses.append(_status(passed, reason))
        reasons.append(reason)
        modes.append(mode)
    return {
        "task_id": record["task_id"],
        "num_correct": sum(pass_results),
        "pass_results": pass_results,
        "statuses": statuses,
        "failure_reasons": reasons,
        "program_modes": modes,
        "grader_version": GRADER_VERSION,
    }


def _evaluate_task_wrapper(args):
    record, timeout = args
    return evaluate_task(record, timeout)


def evaluate_all(records: list[dict], timeout: float = DEFAULT_TIMEOUT, workers: int = 4) -> tuple[list[dict], dict]:
    """Grade all records (order preserved). Returns (task_results, execution_diagnostics)."""
    jobs = [(r, timeout) for r in records]
    if workers <= 1:
        results = [_evaluate_task_wrapper(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_evaluate_task_wrapper, jobs))
    by_status = Counter(s for r in results for s in r["statuses"])
    by_mode = Counter(m for r in results for m in r["program_modes"])
    diag = {
        "grader_version": GRADER_VERSION,
        "timeout_s": timeout,
        "by_status": dict(by_status),
        "by_program_mode": dict(by_mode),
    }
    return results, diag
