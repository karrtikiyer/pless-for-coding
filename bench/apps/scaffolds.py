"""Load/save helpers for external algorithm scaffolds.

A scaffold file is a JSONL where each row is ``{"task_id": int, "scaffold":
str, ...}`` (extra keys such as ``model`` / ``timestamp`` are ignored on load).
Produced by :mod:`bench.apps.gen_scaffolds`, consumed by the APPS runner's
``--scaffold-file`` flag via :func:`load_scaffolds`.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_scaffolds(path: str | Path) -> dict[int, str]:
    """Read a scaffolds JSONL into a ``{task_id: scaffold}`` map.

    Blank lines are skipped. If a ``task_id`` appears more than once (e.g. a
    re-request appended during a resume), the last row wins. A missing file is
    a loud error — a wrong ``--scaffold-file`` path should not silently degrade
    the treatment run into the no-scaffold control.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"scaffold file not found: {path}")
    mapping: dict[int, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[int(row["task_id"])] = row["scaffold"]
    return mapping
