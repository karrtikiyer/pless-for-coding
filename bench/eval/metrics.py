import numpy as np
from human_eval.evaluation import estimate_pass_at_k

from bench.eval.executor import strip_code_fences
from bench.eval.fingerprint import ast_fingerprint


def compute_pass_at_k(
    task_results: list[dict],
    k_values: list[int],
) -> dict[str, float]:
    """Compute pass@k averaged across all tasks.

    Uses the unbiased estimator from the human-eval paper.
    """
    num_samples = np.array([len(r["pass_results"]) for r in task_results])
    num_correct = np.array([r["num_correct"] for r in task_results])

    pass_at_k = {}
    for k in k_values:
        if k > num_samples.min():
            continue
        estimates = estimate_pass_at_k(num_samples, num_correct, k)
        pass_at_k[str(k)] = float(estimates.mean())

    return pass_at_k


def compute_cover_at_t(
    task_results: list[dict],
    t_values: list[float],
    num_samples_per_task: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute cover@t (non-distinct and distinct).

    t is a fractional threshold (e.g. 0.5 means "at least 50% of samples correct").
    Returns the percentage of tasks meeting the threshold.
    """
    num_tasks = len(task_results)
    cover_at_t = {}
    cover_at_t_distinct = {}

    for t in t_values:
        min_correct = t * num_samples_per_task
        cover_at_t[str(t)] = sum(
            1 for r in task_results if r["num_correct"] >= min_correct
        ) / num_tasks * 100

        cover_at_t_distinct[str(t)] = sum(
            1 for r in task_results
            if r.get("num_distinct_correct", 0) >= min_correct
        ) / num_tasks * 100

    return cover_at_t, cover_at_t_distinct


def add_distinct_counts(task_results: list[dict], records: list[dict]) -> None:
    """Add num_distinct_correct to each task result by fingerprinting correct samples."""
    record_by_id = {r["task_id"]: r for r in records}

    for result in task_results:
        record = record_by_id[result["task_id"]]
        samples = record["samples"]
        pass_results = result["pass_results"]

        fingerprints = set()
        for sample, passed in zip(samples, pass_results):
            if passed:
                fp = ast_fingerprint(strip_code_fences(sample))
                if fp is not None:
                    fingerprints.add(fp)

        result["num_distinct_correct"] = len(fingerprints)
