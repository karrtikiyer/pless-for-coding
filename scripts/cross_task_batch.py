"""Cross-task batching scheduler bookkeeping (pure, model-free) — the #1 lever for the
~25% GPU-util bottleneck: pool (task, sample) sequences across MANY tasks into wider
batches so the card isn't starved by per-task batches of only n=10.

This module is ONLY the scheduling/reassembly logic — the part where bugs mean *result
misattribution* (a sample's output landing under the wrong task/sample). The numerical
correctness of the ragged left-padded batched decode it feeds is already proven by
scripts/verify_batched_equivalence.py (Test A/B: batched row == solo; Test C: compaction).
The GPU decode loop over pooled ragged prompts (generalizing batched_phase1) + the K
memory cap are the remaining, GPU-validated pieces; this scaffolding is opt-in and does
not touch the live pipeline.
"""


def flatten_workitems(task_ids, n):
    """Every (task_id, sample_idx) to generate, in a stable order."""
    return [(t, s) for t in task_ids for s in range(n)]


def chunk_items(items, max_seqs):
    """Split work-items into batches of at most `max_seqs` concurrent sequences (K).
    K is the OOM knob: total live tokens ~ K x context, so cap K below the memory ceiling."""
    if max_seqs < 1:
        raise ValueError("max_seqs must be >= 1")
    return [items[i:i + max_seqs] for i in range(0, len(items), max_seqs)]


def regroup_by_task(flat_results, n):
    """Reassemble a flat list of per-sequence results back into per-task, sample-ordered
    lists. Each result must carry its own (task_id, sample) — the anti-misattribution
    contract. Raises if a task is missing samples or has dupes, so a scheduling bug can't
    silently mis-file a result.

    flat_results: list of dicts, each with 'task_id' and 'sample'.
    Returns: {task_id: [result_for_sample_0, ..., result_for_sample_{n-1}]}.
    """
    by_task = {}
    for r in flat_results:
        t, s = r["task_id"], r["sample"]
        slot = by_task.setdefault(t, [None] * n)
        if not (0 <= s < n):
            raise ValueError(f"sample {s} out of range for n={n} (task {t})")
        if slot[s] is not None:
            raise ValueError(f"duplicate result for task {t} sample {s}")
        slot[s] = r
    for t, slot in by_task.items():
        missing = [i for i, v in enumerate(slot) if v is None]
        if missing:
            raise ValueError(f"task {t} missing samples {missing}")
    return by_task
