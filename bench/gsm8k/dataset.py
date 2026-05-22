"""GSM8K test split loader with reproducible random subsetting.

Dataset = ``openai/gsm8k`` "main" config, test split (1,319 problems).
Verified canonical source 2026-05-22 via HF Hub: maintained by OpenAI,
fields are exactly ``question`` and ``answer`` (no difficulty / topic
metadata to stratify by). Citation: Cobbe et al. 2021, arXiv:2110.14168.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Gsm8kProblem:
    task_id: str
    question: str
    gold_answer: str            # The numeric gold answer (string of digits)
    gold_solution: str          # Full ground-truth solution text
    raw_index: int              # 0-based index into the test split (for reproducibility)


def _gold_from_answer_field(answer: str) -> str:
    """GSM8K answers end with '#### N'. Extract the N and normalize."""
    m = re.search(r"####\s*([+-]?\d+(?:[.,]\d+)?)", answer)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def load_gsm8k_subset(
    n_problems: int | None = None,
    seed: int = 0,
) -> list[Gsm8kProblem]:
    """Load GSM8K test split, optionally subsetting to ``n_problems`` random items.

    With ``seed=0`` the subset is deterministic — same indices across runs.
    If ``n_problems`` is None or >= 1319, returns the full test split.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n_total = len(ds)
    if n_problems is None or n_problems >= n_total:
        indices = list(range(n_total))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(n_total, size=n_problems, replace=False).tolist())

    out: list[Gsm8kProblem] = []
    for i in indices:
        row = ds[int(i)]
        out.append(Gsm8kProblem(
            task_id=f"gsm8k_{i:04d}",
            question=row["question"],
            gold_answer=_gold_from_answer_field(row["answer"]),
            gold_solution=row["answer"],
            raw_index=int(i),
        ))
    return out
