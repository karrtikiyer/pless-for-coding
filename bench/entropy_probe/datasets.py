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


_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def load_math(max_problems: int | None = None) -> list[EntropyProbeProblem]:
    """Load competition MATH via HuggingFace ``EleutherAI/hendrycks_math``.

    Dataset choice (verified 2026-05-22 via HF Hub search):
      * The bare repo ``lighteval/MATH`` does not exist on the Hub.
        ``lighteval`` publishes ``lighteval/MATH-Hard`` (a hard-subset
        only, ~7.26k rows) and other variants, but no full-MATH repo.
      * ``EleutherAI/hendrycks_math`` is publicly accessible (no
        gating, no auth required) and contains the full Hendrycks 2021
        MATH source, organised into 7 subject subsets.

    We load all 7 subsets and round-robin across subjects so a
    ``max_problems`` cap gives a roughly balanced sample of problem
    types rather than 200 problems all from algebra. ``split="test"``
    yields ~5,000 problems total across the 7 subjects (the
    canonical MATH test split from Hendrycks 2021).
    """
    from datasets import load_dataset
    per_subject: list[list[dict]] = []
    for subject in _MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        per_subject.append([dict(row, _subject=subject) for row in ds])
    # Round-robin interleave so each subject contributes roughly equally
    # to any prefix of the merged list (matters when max_problems caps).
    out: list[EntropyProbeProblem] = []
    idx = 0
    counter = 0
    while True:
        any_added = False
        for subject_rows in per_subject:
            if idx < len(subject_rows):
                row = subject_rows[idx]
                out.append(EntropyProbeProblem(
                    task_id=f"math_{row['_subject']}_{idx:04d}",
                    problem=row["problem"],
                    reference=row.get("solution"),
                ))
                counter += 1
                any_added = True
                if max_problems is not None and counter >= max_problems:
                    return out
        if not any_added:
            break
        idx += 1
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
