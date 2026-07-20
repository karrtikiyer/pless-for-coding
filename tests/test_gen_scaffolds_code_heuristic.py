"""The scaffold code-leak heuristic must catch real Python but not prose.

Regression test: earlier clauses (`\\bwhile\\b.*:$`, bare `^from `) fired on
ordinary algorithm prose ("Repeat while the queue is not empty:", "From the
leftmost element, ..."), causing false-positive re-requests and a spurious
validation-gate failure on an otherwise-fine scaffold.
"""
from __future__ import annotations

import pytest

from bench.apps.gen_scaffolds import _looks_like_code

PROSE_NOT_CODE = [
    "4. Repeat while the priority queue is not empty:\n   a. Pop the entry.",
    "From the leftmost element, scan rightward and accumulate the running max.",
    "Maintain a distance array `dist` where `dist[u]` is the layer of vertex u.",
    "For each candidate start vertex from 1 to N, do a breadth-first search:",
    "Set dp[i][j] to the best value achievable using the first i items.",
    "Print the smallest feasible time with 10 digits after the decimal point.",
    "Use a while loop conceptually: keep merging until only one interval remains.",
]

REAL_CODE = [
    "def solve(n):\n    return n + 1",
    "```python\nprint(x)\n```",
    "import sys\ndata = sys.stdin.read()",
    "for i in range(n):\n    total += a[i]",
    "n = input()",
    "from collections import deque",
]


@pytest.mark.parametrize("text", PROSE_NOT_CODE)
def test_prose_is_not_flagged_as_code(text):
    assert not _looks_like_code(text), f"false positive on prose: {text!r}"


@pytest.mark.parametrize("text", REAL_CODE)
def test_real_code_is_flagged(text):
    assert _looks_like_code(text), f"missed real code: {text!r}"
