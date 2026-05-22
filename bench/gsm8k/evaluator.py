"""GSM8K evaluation: extract numeric answers + compute pass@k.

The Wei 2022 8-shot CoT convention ends each answer with
"The answer is N." (or "The answer is $N." for currency problems).
We extract the last numeric match against that pattern.

For pass@k, we reuse ``bench.eval.metrics.compute_pass_at_k`` (Chen
et al. 2021 unbiased estimator) — same as MBPP/HumanEval, so the
numbers are computed identically across our code and math experiments.
"""
from __future__ import annotations

import re


_ANSWER_RE = re.compile(
    r"answer is\s*\$?\s*([+-]?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)


def extract_predicted_answer(completion: str) -> str | None:
    """Return the LAST 'answer is N' numeric extraction, or None.

    If the model emitted multiple "The answer is X" phrases (rare but
    possible), the last one is the final answer. Commas are stripped to
    normalize "1,000" → "1000".
    """
    matches = _ANSWER_RE.findall(completion)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def numeric_equals(predicted: str | None, gold: str) -> bool:
    """Compare predicted vs gold as floating-point numbers, tolerant to
    integer/float formatting (so '8' == '8.0', '0.5' == '.5')."""
    if predicted is None or not gold:
        return False
    try:
        return float(predicted) == float(gold)
    except (ValueError, TypeError):
        return predicted == gold
