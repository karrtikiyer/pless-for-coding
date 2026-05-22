"""Dataset loaders for the cross-domain entropy probe.

We unify GSM8K, MATH, and MBPP into a common shape (task_id, problem,
reference) so the probe runner is dataset-agnostic. Reference solutions
are not used by the probe itself — they're stored for later inspection
("was the model's own completion close to the reference?").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntropyProbeProblem:
    task_id: str
    problem: str          # the natural-language question / prompt body
    reference: str | None  # ground-truth completion, if available


def load_gsm8k(max_problems: int | None = None) -> list[EntropyProbeProblem]:
    """Load GSM8K test split via HuggingFace ``openai/gsm8k``."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out: list[EntropyProbeProblem] = []
    for i, row in enumerate(ds):
        out.append(EntropyProbeProblem(
            task_id=f"gsm8k_{i:04d}",
            problem=row["question"],
            reference=row.get("answer"),
        ))
        if max_problems is not None and len(out) >= max_problems:
            break
    return out


def load_math(max_problems: int | None = None) -> list[EntropyProbeProblem]:
    """Load competition MATH via HuggingFace ``lighteval/MATH``.

    The original ``hendrycks/competition_math`` dataset was removed from
    the Hub; ``lighteval/MATH`` is the canonical surviving mirror used
    by lighteval and many recent papers.
    """
    from datasets import load_dataset
    ds = load_dataset("lighteval/MATH", "all", split="test")
    out: list[EntropyProbeProblem] = []
    for i, row in enumerate(ds):
        out.append(EntropyProbeProblem(
            task_id=f"math_{i:04d}",
            problem=row["problem"],
            reference=row.get("solution"),
        ))
        if max_problems is not None and len(out) >= max_problems:
            break
    return out


def load_mbpp(max_problems: int | None = None) -> list[EntropyProbeProblem]:
    """Load MBPP test split — the code-domain control."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    out: list[EntropyProbeProblem] = []
    for i, row in enumerate(ds):
        out.append(EntropyProbeProblem(
            task_id=f"mbpp_{row['task_id']}",
            problem=row["text"],
            reference=row.get("code"),
        ))
        if max_problems is not None and len(out) >= max_problems:
            break
    return out


DATASETS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "mbpp": load_mbpp,
}
