"""LiveCodeBench (code_generation_lite) dataset loader.

Wraps the HF dataset ``livecodebench/code_generation_lite`` (script-based; needs
``trust_remote_code=True`` and ``datasets<4``). Mirrors the shape of
``bench/apps/dataset.py`` so the eval executor and generation runner can treat LCB
like APPS: problems carry ``fn_name`` (None => stdin/stdout; set => functional /
LeetCode-style) plus parallel ``inputs``/``outputs`` test lists.

Schema (all string columns): question_title, question_content, platform,
question_id, contest_id, contest_date, starter_code, difficulty,
public_test_cases, private_test_cases, metadata.

Test decoding (verified): public_test_cases is plain JSON; private_test_cases is
base64 -> zlib -> pickle -> json.loads(str) (pickle wraps a JSON *string*), with a
plain-JSON fallback. Each test is {input, output, testtype} with testtype in
{"stdin", "functional"}.
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from dataclasses import dataclass, field
from typing import Iterator, Literal

LCB_VERSION = "release_v6"
PLATFORMS = ("atcoder", "leetcode", "codeforces")
PlatformT = Literal["atcoder", "leetcode", "codeforces"]


@dataclass
class LcbProblem:
    """Test container mirroring AppsProblem's fields consumed by the executor."""
    task_id: str                       # == question_id (stable, unique join key)
    platform: str                      # "atcoder" | "leetcode" | "codeforces"
    difficulty: str                    # "easy" | "medium" | "hard"
    contest_date: str                  # ISO date (used for contamination windowing)
    question: str                      # == question_content (the problem statement)
    starter_code: str = ""             # non-empty for functional (LeetCode)
    fn_name: str | None = None         # set => functional; None => stdin/stdout
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)

    @property
    def has_test_data(self) -> bool:
        return bool(self.inputs) and bool(self.outputs)


def decode_tests(row: dict) -> tuple[list, list]:
    """Return (public_tests, private_tests), each a list of {input, output, testtype}."""
    public = json.loads(row["public_test_cases"]) if row.get("public_test_cases") else []
    priv_raw = row.get("private_test_cases") or ""
    if not priv_raw:
        private = []
    else:
        try:
            private = json.loads(priv_raw)            # some rows may be plain JSON
        except (json.JSONDecodeError, TypeError):
            private = json.loads(pickle.loads(zlib.decompress(
                base64.b64decode(priv_raw.encode("utf-8")))))
    return public, private


def _func_name(row: dict) -> str | None:
    try:
        md = json.loads(row.get("metadata") or "{}")
    except (json.JSONDecodeError, TypeError):
        md = {}
    fn = md.get("func_name")
    return fn if isinstance(fn, str) and fn.strip() else None


def _in_window(date: str, window: str | None) -> bool:
    """window: None (all) | 'YYYY-MM..YYYY-MM' inclusive-by-month | 'YYYY-MM+' (>=)."""
    if not window:
        return True
    ym = date[:7]
    if window.endswith("+"):
        return ym >= window[:-1]
    lo, hi = window.split("..")
    return lo <= ym <= hi


def _parse_tests(tests: list, fn_name: str | None,
                 max_bytes: int | None) -> tuple[list, list]:
    """Turn decoded {input,output,testtype} tests into parallel (inputs, outputs).

    stdin  -> inputs[i] = stdin string,           outputs[i] = expected stdout string.
    functional -> inputs[i] = [json-parsed args], outputs[i] = json-parsed expected return.
      (LCB encodes multi-arg functional inputs as one JSON value per line.)
    Tests whose input+output exceeds ``max_bytes`` are skipped (guards the ~189MB tail).
    """
    ins, outs = [], []
    for t in tests:
        ti, to = t.get("input", ""), t.get("output", "")
        if max_bytes is not None and (len(ti) + len(to)) > max_bytes:
            continue
        if fn_name:  # functional
            args = [json.loads(ln) for ln in ti.splitlines() if ln.strip() != ""]
            try:
                want = json.loads(to)
            except (json.JSONDecodeError, TypeError):
                want = to
            ins.append(args)
            outs.append(want)
        else:        # stdin
            ins.append(ti)
            outs.append(to)
    return ins, outs


def load_lcb(
    *,
    version: str = LCB_VERSION,
    platforms: tuple[str, ...] | None = None,
    window: str | None = None,
    task_ids: set[str] | None = None,
    with_tests: bool = True,
    max_private: int | None = None,
    max_test_bytes: int | None = None,
) -> Iterator[LcbProblem]:
    """Iterate LCB problems, filtered to ``platforms``, date ``window``, and/or an
    explicit ``task_ids`` set (question_ids).

    ``with_tests=False`` skips (expensive) test decoding — for the generation path,
    which only needs the statement + starter_code. ``max_private`` caps private tests
    per problem (public are always kept); ``max_test_bytes`` skips oversized tests.
    ``task_ids`` restricts decoding to just those problems (used by eval to avoid
    decoding the whole platform, incl. the ~189MB test tail).
    """
    from datasets import load_dataset

    ds = load_dataset("livecodebench/code_generation_lite",
                      version_tag=version, split="test", trust_remote_code=True)
    want_platforms = set(platforms) if platforms else None
    for row in ds:
        plat = row["platform"]
        if want_platforms is not None and plat not in want_platforms:
            continue
        if task_ids is not None and str(row["question_id"]) not in task_ids:
            continue
        if not _in_window(row["contest_date"][:10], window):
            continue
        fn = _func_name(row)
        prob = LcbProblem(
            task_id=str(row["question_id"]),
            platform=plat,
            difficulty=row["difficulty"],
            contest_date=row["contest_date"][:10],
            question=row["question_content"],
            starter_code=row.get("starter_code") or "",
            fn_name=fn,
        )
        if with_tests:
            public, private = decode_tests(row)
            if max_private is not None:
                private = private[:max_private]
            prob.inputs, prob.outputs = _parse_tests(public + private, fn, max_test_bytes)
        yield prob


def load_lcb_test_map(
    *,
    version: str = LCB_VERSION,
    platforms: tuple[str, ...] | None = None,
    window: str | None = None,
    task_ids: set[str] | None = None,
    max_private: int | None = None,
    max_test_bytes: int | None = None,
) -> dict[str, LcbProblem]:
    """{task_id -> LcbProblem with decoded tests}, for the eval executor."""
    return {
        p.task_id: p
        for p in load_lcb(version=version, platforms=platforms, window=window,
                          task_ids=task_ids, with_tests=True, max_private=max_private,
                          max_test_bytes=max_test_bytes)
    }


def load_lcb_date_map(*, version: str = LCB_VERSION) -> dict[str, str]:
    """{task_id -> contest_date} for post-hoc contamination-window slicing in analysis."""
    return {
        p.task_id: p.contest_date
        for p in load_lcb(version=version, with_tests=False)
    }
