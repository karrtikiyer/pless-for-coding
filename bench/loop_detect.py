"""Streaming n-gram loop detection for live think-phase termination.

Single canonical implementation used by the vLLM logits processor (live force-</think>)
and reusable by offline scripts. Content-agnostic: a loop is declared when any length-n
token n-gram recurs >= k times within the last `window` tokens.

Mirrors the validated detector in scripts/repeat_detector.py (same n-gram-count logic);
this module is the import target for bench/ code (scripts/ must not be imported by bench/).
"""
from collections import Counter


def ngram_loop_fired(token_ids, n: int = 8, k: int = 4, window: int = 400) -> bool:
    """True iff some length-n token sequence recurs >= k times in the last `window` tokens."""
    if len(token_ids) < n * k:
        return False
    t = token_ids[-window:]
    counts = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    # counts is non-empty because len(t) >= n*k >= n
    return max(counts.values()) >= k
