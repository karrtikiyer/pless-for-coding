"""Shared think-end-marker resolution, by model-name substring.

Centralizes the Qwen3/DeepSeek-R1 (``</think>``) vs Ministral (``[/THINK]``)
convention so eval/report code never hardcodes one model family's marker.
Previously duplicated ad hoc in ``scripts/report_lcb_v6_stratified60.py``
(found 2026-10-05: its termination% computation hardcoded ``"</think>"``,
silently reading ~0% for every Ministral config).
"""
from __future__ import annotations

THINK_END_MARKER_BY_MODEL_SUBSTR = [
    ("qwen", "</think>"),
    ("deepseek", "</think>"),
    ("ministral", "[/THINK]"),
]


def resolve_think_end_marker(model: str) -> str | None:
    """Return the model's think-end marker string, or None if unmapped.

    Returns ``None`` (rather than raising) for models with no reasoning
    phase at all (CodeLlama, Codestral, ...) — callers that find a
    ``samples_with_thinking`` field but get ``None`` here should raise
    loudly themselves, since that means a new reasoning model needs wiring.
    """
    model_lower = model.lower()
    for substr, marker in THINK_END_MARKER_BY_MODEL_SUBSTR:
        if substr in model_lower:
            return marker
    return None
