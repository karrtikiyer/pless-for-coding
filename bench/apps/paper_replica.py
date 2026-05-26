"""Load the EXACT prompts the paper sent to each (model, problem) pair.

For Phase A of the Deepseek-6.7B-Instruct comparison experiment, we need
to isolate the sampler effect (our pless_alpha vs paper's nucleus) from
any prompt-format effect. The cleanest way is to use the paper's own
prompts — published in the dataset
`sh0416/outputs-apps` (alongside the 100-sample-per-problem completions
behind their Table 2).

Each (model, problem_id) appears in ~100 rows (one per sample) with the
same `prompt` field; we deduplicate to one prompt per problem_id.

Cached to disk on first load so repeat calls don't re-stream the full
1.67 GB dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


_DATASET_NAME = "sh0416/outputs-apps"
_VALID_DIFFICULTIES = {"introductory", "interview", "competition"}


def _stream_dataset_rows(difficulty: str) -> Iterator[dict]:
    """Stream rows of the sh0416/outputs-apps test split.

    Isolated as a function so tests can monkeypatch it with a synthetic
    row generator (no network required for unit tests).
    """
    from datasets import load_dataset
    ds = load_dataset(_DATASET_NAME, difficulty, split="test", streaming=True)
    for row in ds:
        yield row


def _cache_path(cache_dir: Path, model: str, source: str,
                difficulty: str) -> Path:
    """Per-(model, source, difficulty) cached parquet."""
    model_slug = model.replace("/", "--")
    return cache_dir / f"{model_slug}__{source}__{difficulty}.parquet"


def load_paper_prompts(
    model: str,
    source: str,
    difficulty: str,
    *,
    cache_dir: Path | None = None,
) -> dict[int, str]:
    """Return {problem_id: prompt_str} for the paper's run of one cell.

    Args:
        model: HF model id, e.g. ``"deepseek-ai/deepseek-coder-6.7b-instruct"``.
        source: ``"ATCODER"`` or ``"CODEFORCES"``.
        difficulty: ``"introductory"``, ``"interview"``, or ``"competition"``.
        cache_dir: optional dir for the dedup'd parquet cache. If the cache
            file exists, skips the stream entirely.

    Raises:
        ValueError: if no rows match (avoids silent skipping of the experiment).
    """
    if difficulty not in _VALID_DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of {_VALID_DIFFICULTIES}, got {difficulty!r}"
        )

    # Cache hit short-circuit
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cf = _cache_path(cache_dir, model, source, difficulty)
        if cf.exists():
            df = pd.read_parquet(cf)
            return {int(r["problem_id"]): str(r["prompt"])
                    for _, r in df.iterrows()}

    # Stream + dedup by problem_id, filtering to (model, source).
    # First-seen prompt wins (all 100 rows share the same prompt for a
    # given (model, problem_id), so order doesn't matter).
    prompts: dict[int, str] = {}
    for row in _stream_dataset_rows(difficulty):
        if row.get("model") != model:
            continue
        if row.get("source") != source:
            continue
        pid = int(row["problem_id"])
        if pid in prompts:
            continue
        prompts[pid] = str(row["prompt"])

    if not prompts:
        raise ValueError(
            f"No paper-replica prompts found for "
            f"(model={model!r}, source={source!r}, difficulty={difficulty!r}). "
            f"Check that the model exists in {_DATASET_NAME} for this bucket."
        )

    # Persist cache
    if cache_dir is not None:
        cf = _cache_path(cache_dir, model, source, difficulty)
        pd.DataFrame(
            [{"problem_id": pid, "prompt": p} for pid, p in prompts.items()]
        ).to_parquet(cf, index=False)

    return prompts
