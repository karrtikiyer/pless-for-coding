"""APPS-aware code extraction.

Sibling of ``bench.eval.executor.extract_python_code``. The MBPP/HumanEval
extractor is tuned for function bodies and applies two transformations
that are **actively destructive** for APPS samples (full programs that
read from stdin and write to stdout):

  * ``_strip_check_and_main`` truncates at the first ``if __name__`` line,
    which is exactly the entry point an APPS program needs to drive its
    stdin/stdout I/O.
  * ``_strip_after_function`` truncates the first non-indented, non-``def``
    line after a function body — which would chop the ``if __name__`` block
    even if the previous step missed it, and chops any module-level
    statements (imports written after a helper def, top-level reads, etc.).

So we ship a separate extractor for APPS that *preserves* whole-program
structure. The strategy stack:

  1. Enumerate every candidate code block we can see:
     ``` ```python\\n...``` ``, ``` ```\\n...``` ``, and (as a last resort)
     the raw sample dedented.
  2. For each candidate, run an increasingly aggressive compile-rescue:
     compile as-is → dedent → ``_trim_to_compilable`` (longest
     compilable prefix). Each rescue step records which one succeeded
     so the diagnostics can attribute the win.
  3. Score the compilable candidates and pick the highest-scoring one.
     Ties broken by appearance order (we favour the *last* occurrence on
     a tie, since models often emit a sketch first and a final answer
     last).

Output is a structured ``ExtractionResult`` rather than a bare string so
the diagnostics block in the metrics JSON can attribute drops accurately:
how many samples got compiled-as-is vs dedent-rescued vs trim-rescued vs
gave up.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

_PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
# Plain triple-backtick fence (no python tag). We accept these but
# *score them lower* than ```python``` fences since they sometimes hold
# pseudocode / plain text.
_PLAIN_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


@dataclass
class ExtractionResult:
    code: str                   # the chosen program (may be "")
    success: bool               # whether ``code`` compiles
    strategy: str               # which path picked the winner (see below)
    n_candidates_seen: int      # how many code blocks were detected
    reason_if_failed: str | None = None  # populated when success is False


# Strategy strings — kept short and stable so they're useful as JSON keys
# in the diagnostics aggregator.
STRATEGY_PYTHON_FENCE_ASIS  = "python_fence_asis"
STRATEGY_PYTHON_FENCE_DEDENT = "python_fence_dedent"
STRATEGY_PYTHON_FENCE_TRIM  = "python_fence_trim"
STRATEGY_PLAIN_FENCE_ASIS   = "plain_fence_asis"
STRATEGY_PLAIN_FENCE_DEDENT = "plain_fence_dedent"
STRATEGY_PLAIN_FENCE_TRIM   = "plain_fence_trim"
STRATEGY_RAW_DEDENT         = "raw_dedent"
STRATEGY_RAW_TRIM           = "raw_trim"
STRATEGY_RAW_PREFIX_STRIP   = "raw_prefix_strip"
STRATEGY_RAW_WINDOW         = "raw_window"
STRATEGY_NONE               = "none"


def _try_compile(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        compile(code, "<sample>", "exec")
        return True
    except SyntaxError:
        return False
    except (MemoryError, RecursionError):
        return False


def _trim_to_compilable(code: str) -> str | None:
    """Return the longest line-prefix of ``code`` that compiles, or None."""
    if _try_compile(code):
        return code
    lines = code.split("\n")
    for end in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end])
        if _try_compile(candidate):
            return candidate
    return None


def _compile_with_optional_dedent(code: str) -> str | None:
    """Return ``code`` if it compiles, or its dedented form if dedenting
    yields something compilable, else None.

    Handles the common "all lines indented at consistent N" pattern
    that arises when a window-strip or prefix-strip yields a code block
    that was originally inside a (hallucinated) function body. The
    block-as-is is an IndentationError at module scope; ``textwrap.dedent``
    on it removes the common N-space prefix and the result compiles.
    """
    if _try_compile(code):
        return code
    dedented = textwrap.dedent(code)
    if dedented != code and _try_compile(dedented):
        return dedented
    return None


def _strip_leading_to_compilable(code: str) -> str | None:
    """Drop leading lines progressively until what remains compiles.

    Sibling of :func:`_trim_to_compilable` for the OTHER end of the
    sample. Useful when a model emits prose / hallucinated tokens
    BEFORE the actual code (a frequent Deepseek-Coder-Instruct
    failure mode on APPS — see ``tests/test_apps_extractor.py``).

    Each candidate suffix is checked first as-is, then with
    ``textwrap.dedent`` applied (to recover indented-code blocks that
    were inside a hallucinated function body).

    Returns the longest compilable suffix, or None.
    """
    if _try_compile(code):
        return code
    lines = code.split("\n")
    for start in range(1, len(lines)):
        candidate = "\n".join(lines[start:])
        recovered = _compile_with_optional_dedent(candidate)
        if recovered is not None:
            return recovered
    return None


def _window_to_compilable(code: str, max_iter: int = 100) -> str | None:
    """Drop BOTH leading AND trailing lines to find a compilable window.

    Only attempted when neither :func:`_trim_to_compilable` (suffix-only)
    nor :func:`_strip_leading_to_compilable` (prefix-only) succeeds.
    For each candidate start (capped at ``max_iter`` lines), search for
    the largest end such that ``lines[start:end]`` compiles (after an
    optional ``textwrap.dedent`` pass — necessary for the
    indented-code-buried-in-prose pattern), and return the first such
    window found. ``max_iter`` bounds the worst-case to
    O(max_iter × n) compile attempts, keeping cost predictable on
    pathologically long samples.
    """
    if _try_compile(code):
        return code
    lines = code.split("\n")
    n = len(lines)
    for start in range(1, min(n, max_iter)):
        # range(n, start, -1) gives [n, n-1, ..., start+1] so the smallest
        # window considered for this start is the single line lines[start:start+1].
        for end in range(n, start, -1):
            candidate = "\n".join(lines[start:end])
            recovered = _compile_with_optional_dedent(candidate)
            if recovered is not None:
                return recovered
    return None


def _rescue(code: str, fence_strategies: tuple[str, str, str]) -> tuple[str | None, str | None]:
    """Try compile-as-is → dedent → trim-to-compilable on a candidate body.

    ``fence_strategies`` is a triple of strategy strings for the three rescue
    levels (asis, dedent, trim) so the caller can label the winner.
    Returns (winning_code, winning_strategy) or (None, None).
    """
    strat_asis, strat_dedent, strat_trim = fence_strategies
    if _try_compile(code):
        return code, strat_asis
    dedented = textwrap.dedent(code)
    if dedented != code and _try_compile(dedented):
        return dedented, strat_dedent
    trimmed = _trim_to_compilable(code)
    if trimmed is not None:
        return trimmed, strat_trim
    # Last-ditch: trim the dedented version (handles cases where dedent
    # alone doesn't compile but a prefix of it does).
    if dedented != code:
        trimmed_dedented = _trim_to_compilable(dedented)
        if trimmed_dedented is not None:
            return trimmed_dedented, strat_trim
    return None, None


def _score(
    code: str,
    *,
    is_python_fence: bool,
    fn_name: str | None,
) -> int:
    """Score a compilable candidate. Higher = better.

    Heuristics, in order of decisiveness:
      + 100 if explicit ```python``` fence (vs plain ``` `` or raw)
      + 50  if contains ``input(`` or ``sys.stdin`` (it's *probably* a
              stdin/stdout program, which is what every ATCODER/CODEFORCES
              problem expects)
      + 50  if ``fn_name`` was supplied and the code defines that function
      + 1 per 100 chars (longer = more developed solution, but small weight)
    """
    score = 0
    if is_python_fence:
        score += 100
    if "input(" in code or "sys.stdin" in code:
        score += 50
    if fn_name and re.search(rf"\bdef\s+{re.escape(fn_name)}\b", code):
        score += 50
    score += len(code) // 100
    return score


def extract_python_code_apps(
    text: str,
    *,
    fn_name: str | None = None,
) -> ExtractionResult:
    """Extract a compilable Python program from a generated sample.

    Designed for the APPS protocol where the model emits a *full program*
    (reads stdin, writes stdout), not a function body. Preserves
    ``if __name__ == "__main__":`` blocks, multiple top-level definitions,
    and module-level statements.

    ``fn_name`` is forwarded from the APPS problem's ``input_output``
    field when present (None for the ATCODER/CODEFORCES competition
    buckets in scope). When supplied it gets a scoring boost for
    candidates that define a matching function.
    """
    if not isinstance(text, str) or not text.strip():
        return ExtractionResult(code="", success=False, strategy=STRATEGY_NONE,
                                n_candidates_seen=0,
                                reason_if_failed="empty_input")

    n_candidates_seen = 0
    winners: list[tuple[int, str, str]] = []  # (score, code, strategy) — last-wins on ties

    # 1. ```python``` fences (preferred)
    py_matches = list(_PYTHON_FENCE_RE.finditer(text))
    n_candidates_seen += len(py_matches)
    for m in py_matches:
        body = m.group(1)
        win, strat = _rescue(body, (
            STRATEGY_PYTHON_FENCE_ASIS,
            STRATEGY_PYTHON_FENCE_DEDENT,
            STRATEGY_PYTHON_FENCE_TRIM,
        ))
        if win is not None:
            winners.append((_score(win, is_python_fence=True, fn_name=fn_name), win, strat))

    # 2. Plain ``` fences (only if no ```python``` produced a winner — these
    #    are noisier and we'd rather not chase them unless necessary)
    if not winners:
        plain_matches = []
        for m in _PLAIN_FENCE_RE.finditer(text):
            # Skip overlapping ```python``` matches — they'd be double-counted
            body = m.group(1)
            if body.startswith("python\n"):
                continue
            plain_matches.append(m)
        n_candidates_seen += len(plain_matches)
        for m in plain_matches:
            body = m.group(1)
            win, strat = _rescue(body, (
                STRATEGY_PLAIN_FENCE_ASIS,
                STRATEGY_PLAIN_FENCE_DEDENT,
                STRATEGY_PLAIN_FENCE_TRIM,
            ))
            if win is not None:
                winners.append((_score(win, is_python_fence=False, fn_name=fn_name), win, strat))

    # 3. Raw text (no fence found, or no fence-based rescue succeeded)
    if not winners:
        n_candidates_seen += 1
        win, strat = _rescue(text, (STRATEGY_RAW_DEDENT, STRATEGY_RAW_DEDENT,
                                    STRATEGY_RAW_TRIM))
        if win is not None:
            winners.append((_score(win, is_python_fence=False, fn_name=fn_name), win, strat))

    # 4. Raw text with leading-line stripping (prefix-strip). When the
    #    model emits prose / hallucinated tokens BEFORE the code (e.g.
    #    '\\n\\nceed:\\ndef dfs(...)...'), neither dedent nor suffix-trim
    #    can recover it. ~67% of our Phase A Deepseek ParsingErrors fell
    #    into this pattern (cell1 corpus audit 2026-05-26).
    if not winners:
        recovered = _strip_leading_to_compilable(text)
        if recovered is not None:
            winners.append((
                _score(recovered, is_python_fence=False, fn_name=fn_name),
                recovered, STRATEGY_RAW_PREFIX_STRIP,
            ))

    # 5. Raw text with both leading AND trailing stripping. Catches
    #    samples with prose preamble AND trailing garbage around a valid
    #    code middle. Recovers another ~27% of our Phase A Deepseek PEs
    #    on top of prefix-strip alone. Bounded by max_iter=100 to keep
    #    cost predictable on huge samples.
    if not winners:
        recovered = _window_to_compilable(text, max_iter=100)
        if recovered is not None:
            winners.append((
                _score(recovered, is_python_fence=False, fn_name=fn_name),
                recovered, STRATEGY_RAW_WINDOW,
            ))

    if not winners:
        return ExtractionResult(code="", success=False, strategy=STRATEGY_NONE,
                                n_candidates_seen=n_candidates_seen,
                                reason_if_failed="no_compilable_candidate")

    # Highest score wins; ties go to the *last* (later block in the sample —
    # models often refine and emit a final answer at the end)
    best = max(range(len(winners)), key=lambda i: (winners[i][0], i))
    score, code, strategy = winners[best]
    return ExtractionResult(
        code=code,
        success=True,
        strategy=strategy,
        n_candidates_seen=n_candidates_seen,
    )
