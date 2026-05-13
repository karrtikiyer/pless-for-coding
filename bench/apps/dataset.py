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

from dataclasses import dataclass
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
    problem_id: int
    source: str           # "ATCODER" | "CODEFORCES"
    difficulty: str       # "introductory" | "interview" | "competition"
    question: str
    starter_code: str     # may be empty


def load_apps(
    *,
    source: SourceT | None = None,
    difficulty: DifficultyT | None = None,
    max_problems: int | None = None,
) -> Iterator[AppsProblem]:
    """Iterate APPS problems filtered to a (source, difficulty) bucket.

    Both ``source`` and ``difficulty`` default to all-of-that-axis when None.
    "OTHER" source problems (~1.3 K, mostly Codewars / Kattis) are always
    excluded to match the paper's scope.
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
        yield AppsProblem(
            problem_id=int(row["problem_id"]),
            source=src,
            difficulty=row["difficulty"],
            question=row["question"],
            starter_code=row.get("starter_code", "") or "",
        )
        yielded += 1
        if max_problems is not None and yielded >= max_problems:
            break
