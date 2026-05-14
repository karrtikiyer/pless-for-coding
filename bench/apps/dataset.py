"""APPS dataset loader.

Wraps the canonical ``codeparrot/apps`` dataset on HuggingFace. The schema:

  * ``problem_id`` (int)
  * ``question`` (str)
  * ``solutions`` (str, JSON-encoded list of human reference solutions)
  * ``input_output`` (str, JSON-encoded test I/O)
  * ``difficulty`` ("introductory" | "interview" | "competition")
  * ``url`` (str) — used to derive the *source* ("ATCODER" / "CODEFORCES" / "OTHER")
  * ``starter_code`` (str)

Source extraction matches the algosim paper's convention (Lee et al. 2025
Table 1): AtCoder problems live at ``atcoder.jp``, CodeForces at
``codeforces.com``. Other sources exist (Codewars etc.) but the paper restricts
to those two; we follow.

Bucket counts from our local cache (test split):

  | source     | introductory | interview | competition |
  |------------|-------------:|----------:|------------:|
  | ATCODER    | 403          | 252       | 41          |
  | CODEFORCES | 299          | 2386      | 268         |

These match the paper's Table 1 numbers to within ~1 % for CodeForces (the
paper reports 294 / 2376 / 264 — minor discrepancy likely from problems they
later excluded; we use the canonical dataset as-is).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator, Literal

DIFFICULTIES = ("introductory", "interview", "competition")
SOURCES = ("ATCODER", "CODEFORCES")

DifficultyT = Literal["introductory", "interview", "competition"]
SourceT = Literal["ATCODER", "CODEFORCES"]


def url_to_source(url: str) -> str:
    """Map a problem URL to its source string, matching the paper's convention."""
    if "atcoder.jp" in url:
        return "ATCODER"
    if "codeforces.com" in url:
        return "CODEFORCES"
    return "OTHER"


@dataclass
class AppsProblem:
    """One APPS problem with the test data needed to evaluate generations.

    All ATCODER and CODEFORCES problems in the canonical dataset are
    stdin/stdout (verified empirically: 3,649 / 3,649 have no ``fn_name``).
    ``fn_name`` is preserved in the dataclass for forward-compatibility if
    we ever extend scope to the OTHER sources (Codewars / Kattis), which
    do contain function-call problems.
    """
    problem_id: int
    source: str                       # "ATCODER" | "CODEFORCES"
    difficulty: str                   # "introductory" | "interview" | "competition"
    question: str
    starter_code: str = ""            # may be empty
    fn_name: str | None = None        # function-call interface (None for stdin/stdout)
    inputs: list = field(default_factory=list)   # parsed from input_output["inputs"]
    outputs: list = field(default_factory=list)  # parsed from input_output["outputs"]

    @property
    def has_test_data(self) -> bool:
        return bool(self.inputs) and bool(self.outputs)


def _parse_input_output(io_str: str) -> tuple[str | None, list, list]:
    """Parse the JSON-encoded input_output field. Returns (fn_name, inputs, outputs)."""
    if not io_str:
        return None, [], []
    try:
        obj = json.loads(io_str)
    except (json.JSONDecodeError, TypeError):
        return None, [], []
    if not isinstance(obj, dict):
        return None, [], []
    fn_name = obj.get("fn_name")
    if isinstance(fn_name, str) and not fn_name.strip():
        fn_name = None
    inputs = obj.get("inputs") or []
    outputs = obj.get("outputs") or []
    return fn_name, list(inputs), list(outputs)


def load_apps(
    *,
    source: SourceT | None = None,
    difficulty: DifficultyT | None = None,
    max_problems: int | None = None,
    with_tests: bool = True,
) -> Iterator[AppsProblem]:
    """Iterate APPS problems filtered to a (source, difficulty) bucket.

    Both ``source`` and ``difficulty`` default to all-of-that-axis when None.
    "OTHER" source problems (~1.3 K, mostly Codewars / Kattis) are always
    excluded to match the paper's scope.

    ``with_tests`` controls whether ``input_output`` is parsed and attached.
    The generation pipeline doesn't need it (saves a few seconds on startup);
    the eval pipeline does.
    """
    from datasets import load_dataset

    # codeparrot/apps ships as a script-based dataset (loading_script: apps.py).
    # `datasets` 3.x requires explicit `trust_remote_code=True` to execute it;
    # `datasets` 4.x removed script support entirely (hence our datasets<4 pin
    # in pyproject.toml). The script just unpacks the parquet shards and
    # exposes them — we've read it and it's safe.
    ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    yielded = 0
    for row in ds:
        src = url_to_source(row["url"])
        if src == "OTHER":
            continue
        if source is not None and src != source:
            continue
        if difficulty is not None and row["difficulty"] != difficulty:
            continue
        problem = AppsProblem(
            problem_id=int(row["problem_id"]),
            source=src,
            difficulty=row["difficulty"],
            question=row["question"],
            starter_code=row.get("starter_code", "") or "",
        )
        if with_tests:
            problem.fn_name, problem.inputs, problem.outputs = _parse_input_output(
                row.get("input_output", "")
            )
        yield problem
        yielded += 1
        if max_problems is not None and yielded >= max_problems:
            break


def load_apps_test_map(
    *,
    source: SourceT | None = None,
    difficulty: DifficultyT | None = None,
) -> dict[int, AppsProblem]:
    """Build a problem_id -> AppsProblem map for the eval pipeline.

    Test data (``inputs``, ``outputs``, ``fn_name``) is parsed and attached.
    Cheap one-time scan of the relevant subset of the HF dataset.
    """
    return {
        p.problem_id: p
        for p in load_apps(source=source, difficulty=difficulty, with_tests=True)
    }
