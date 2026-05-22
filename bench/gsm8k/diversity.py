"""Pairwise BLEU-4 diversity on GSM8K reasoning chains.

Designed to be apples-to-apples with the code-side
``bench/eval/metrics.py:add_self_codebleu`` convention:

  * **Pairwise** (not leave-one-out — distinct from Texygen Self-BLEU).
    For all i < j, compute BLEU-4 between samples i and j, then average.
  * **Conditional on correctness**: only correct samples (verified by
    ``bench.gsm8k.evaluator.numeric_equals``) are used.
  * **Deduplicated**: identical normalized-reasoning strings are
    collapsed before the pairwise computation, so we don't reward the
    sampler for emitting verbatim copies.
  * **Diversity = 1 − mean(BLEU-4)** so higher = more diverse, matching
    the existing ``self_codebleu`` convention.

We compute BLEU on the **reasoning portion** of each completion — the
text before "The answer is N." — so the trailing answer phrase (which
is identical across correct samples, just "The answer is X") doesn't
inflate similarity. This matches the design rationale we discussed
2026-05-22: in math, diversity lives in the reasoning path, not the
answer.

BLEU-4 via NLTK with method-1 smoothing (avoids zero scores when
4-gram overlap happens to be empty in short reasoning).
"""
from __future__ import annotations

import re
from typing import Iterable


_ANSWER_SPLIT_RE = re.compile(r"\bthe\s+answer\s+is\b", re.IGNORECASE)


def extract_reasoning(completion: str) -> str:
    """Return the text BEFORE the 'The answer is' phrase.

    If the phrase isn't present, return the whole completion (the
    model emitted reasoning but never produced the answer line — this
    correctness path will be filtered out by the conditional-on-correct
    filter anyway).
    """
    m = _ANSWER_SPLIT_RE.search(completion)
    return completion[: m.start()].strip() if m else completion.strip()


def _tokenize(text: str) -> list[str]:
    """Whitespace + basic punctuation tokenization for BLEU.

    NLTK's BLEU is sensitive to tokenization; we lowercase and split on
    non-word boundaries so punctuation doesn't break n-gram matches
    artificially.
    """
    # Simple regex tokenizer: keep alphanumeric / digit groups
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def pairwise_bleu4_diversity(samples: Iterable[str]) -> float | None:
    """All-pairs BLEU-4 → 1 − mean. Higher = more diverse.

    Returns None if fewer than 2 samples are present (diversity undefined).
    Samples are deduplicated by normalized reasoning text first.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    reasonings = [extract_reasoning(s) for s in samples]
    # Dedup: collapse identical normalized strings
    seen: dict[str, str] = {}
    for r in reasonings:
        key = " ".join(_tokenize(r))
        if key and key not in seen:
            seen[key] = r
    unique = list(seen.values())
    if len(unique) < 2:
        # All correct samples produced identical reasoning (or only one)
        return 0.0 if len(unique) == 1 and len(reasonings) >= 2 else None

    tokenized = [_tokenize(r) for r in unique]
    smoother = SmoothingFunction().method1
    scores: list[float] = []
    weights = (0.25, 0.25, 0.25, 0.25)  # BLEU-4 uniform weights
    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            if not tokenized[i] or not tokenized[j]:
                continue
            try:
                s = sentence_bleu(
                    [tokenized[i]], tokenized[j],
                    weights=weights, smoothing_function=smoother,
                )
                scores.append(float(s))
            except Exception:
                continue
    if not scores:
        return None
    return round(1.0 - sum(scores) / len(scores), 4)


def compute_aggregate_diversity(
    per_task_records: list[dict],
) -> dict[str, float | int]:
    """Aggregate diversity across all tasks.

    Each per-task record must have a key ``self_bleu_diversity`` (the
    per-task value, computed earlier via ``pairwise_bleu4_diversity`` on
    that task's correct samples). Returns a dict with the cross-task
    mean and the count of tasks with at least 2 unique correct samples.
    """
    valid = [r["self_bleu_diversity"] for r in per_task_records
             if r.get("self_bleu_diversity") is not None]
    n = len(valid)
    if n == 0:
        return {"self_bleu_diversity": None, "n_tasks_with_diversity": 0}
    return {
        "self_bleu_diversity": round(sum(valid) / n, 4),
        "n_tasks_with_diversity": n,
    }
