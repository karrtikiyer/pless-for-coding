"""Paired HF-vs-vLLM backend-delta comparison logic (pure; no GPU, no model).

The question this answers: at a *matched* config (same model, sampler, budget, prompt,
task_ids), do the HF and vLLM backends produce the same pless behaviour? For Qwen3-8B
they did (truncation 13.0% vs 12.3%, pass@1 0.631 vs 0.625 on the shared 100). For
DeepSeek-R1-Distill the reconstructed numbers diverge (truncation ~40% vs 64.9%, pass@1
0.386 vs 0.174) — this module scores a controlled run to confirm/deny that on identical
task_ids.

Scoring is NOT reimplemented: pass@k routes through the canonical
``bench.eval.metrics.compute_pass_at_k`` (the same unbiased estimator behind the α=2/α=5
columns), so the comparison is apple-to-apple with the standard eval pipeline. This
module only pairs tasks, bootstraps the paired gap, and counts truncation.
"""
import random

from bench.eval.metrics import compute_pass_at_k


def index_by_task(per_task):
    """{task_id -> per_task dict}; raises on duplicate task_id."""
    out = {}
    for r in per_task:
        t = r["task_id"]
        if t in out:
            raise ValueError(f"duplicate task_id {t}")
        out[t] = r
    return out


def paired_task_results(hf_per_task, vllm_per_task, subset=None):
    """Align two backends' per_task lists on their shared task_ids.

    Returns ``(task_ids, hf_results, vllm_results)`` where the two result lists are the
    per_task dicts ordered by task_id (so index i is the same task in both). If ``subset``
    is given, restrict to it and raise if any requested id is absent from either backend.
    """
    hf = index_by_task(hf_per_task)
    v = index_by_task(vllm_per_task)
    shared = set(hf) & set(v)
    if subset is not None:
        want = set(subset)
        missing = want - shared
        if missing:
            raise ValueError(
                f"subset task_ids missing from one/both backends: {sorted(missing)}")
        shared &= want
    ids = sorted(shared)
    if not ids:
        raise ValueError("no shared task_ids to compare")
    return ids, [hf[t] for t in ids], [v[t] for t in ids]


def pass_at_k(task_results, k_values=(1,)):
    """Canonical pass@k over a set of per_task dicts (reuses bench.eval.metrics)."""
    return compute_pass_at_k(list(task_results), list(k_values))


def bootstrap_gap(hf_results, vllm_results, k=1, iters=2000, seed=0):
    """Paired task-level bootstrap of the (HF - vLLM) pass@k gap.

    Resamples task *indices* with replacement (same indices applied to both backends →
    keeps the pairing), recomputes pass@k on each resample via the canonical estimator,
    and returns ``(point, lo, hi)`` at the 95% percentile interval. Deterministic for a
    given ``seed``.
    """
    if len(hf_results) != len(vllm_results):
        raise ValueError("paired inputs must have equal length")
    n = len(hf_results)
    ks = str(k)
    point = pass_at_k(hf_results, [k])[ks] - pass_at_k(vllm_results, [k])[ks]
    rng = random.Random(seed)
    gaps = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        h = [hf_results[i] for i in idx]
        v = [vllm_results[i] for i in idx]
        gaps.append(pass_at_k(h, [k])[ks] - pass_at_k(v, [k])[ks])
    gaps.sort()
    lo = gaps[int(0.025 * iters)]
    hi = gaps[min(int(0.975 * iters), iters - 1)]
    return point, lo, hi


def truncation_rate(records, subset=None):
    """Fraction of samples whose raw generation never closes ``</think>`` (the loop-
    truncation proxy). ``records`` are run-JSONL dicts with ``samples_with_thinking``.

    Returns ``(n_truncated, n_samples, rate)``. Uses an exact substring test — no
    tokenizer, so it is immune to the decode→re-encode token-count drift.
    """
    if subset is not None:
        want = set(subset)
        records = [r for r in records if r["task_id"] in want]
    n = 0
    trunc = 0
    for r in records:
        for s in r.get("samples_with_thinking", []):
            n += 1
            if "</think>" not in s:
                trunc += 1
    if n == 0:
        raise ValueError("no samples")
    return trunc, n, trunc / n
