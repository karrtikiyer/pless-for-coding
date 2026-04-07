import ctypes
from datetime import datetime, timezone
from statistics import mean as _mean

import numpy as np
from human_eval.evaluation import estimate_pass_at_k

from bench.eval.executor import strip_code_fences
from bench.eval.fingerprint import ast_fingerprint, pairwise_diversity


def _patch_tree_sitter_capsule():
    """Patch tree-sitter 0.22 Language to accept PyCapsule from tree-sitter-python 0.23+."""
    from tree_sitter import Language

    if getattr(Language, "_capsule_patched", False):
        return
    _orig_init = Language.__init__

    def _patched_init(self, *args, **kwargs):
        if args and type(args[0]).__name__ == "PyCapsule":
            ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
            ptr_fn.restype = ctypes.c_void_p
            ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
            ptr = ptr_fn(args[0], b"tree_sitter.Language")
            return _orig_init(self, ptr, *args[1:], **kwargs)
        return _orig_init(self, *args, **kwargs)

    Language.__init__ = _patched_init
    Language._capsule_patched = True


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
    pass_at_1 = None
    for k in k_values:
        if k > num_samples.min():
            # For single-sample methods (greedy, beam): copy pass@1
            if pass_at_1 is not None:
                pass_at_k[str(k)] = pass_at_1
            continue
        estimates = estimate_pass_at_k(num_samples, num_correct, k)
        val = float(estimates.mean())
        pass_at_k[str(k)] = val
        if k == 1:
            pass_at_1 = val

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


def add_structural_diversity(
    task_results: list[dict],
    records: list[dict],
    cluster_threshold: float = 0.8,
) -> None:
    """Add pairwise AST edit distance metrics to each task result.

    Adds `mean_pairwise_distance` and `num_ast_clusters` to each task result dict.
    """
    record_by_id = {r["task_id"]: r for r in records}

    for result in task_results:
        record = record_by_id[result["task_id"]]
        samples = record["samples"]
        pass_results = result["pass_results"]

        correct_codes = [
            strip_code_fences(sample)
            for sample, passed in zip(samples, pass_results)
            if passed
        ]

        if len(correct_codes) < 2:
            result["mean_pairwise_distance"] = 0.0
            result["num_ast_clusters"] = len(correct_codes)
            continue

        diversity = pairwise_diversity(correct_codes, cluster_threshold)
        result["mean_pairwise_distance"] = diversity["mean_distance"]
        result["num_ast_clusters"] = diversity["num_clusters"]


def add_self_codebleu(
    task_results: list[dict],
    records: list[dict],
) -> None:
    """Add pairwise CodeBLEU diversity metrics to each task result.

    Computes all-pairs CodeBLEU among correct samples, storing diversity
    (1 - similarity) for the composite score and sub-components.
    """
    _patch_tree_sitter_capsule()
    from codebleu import calc_codebleu

    record_by_id = {r["task_id"]: r for r in records}

    for result in task_results:
        record = record_by_id[result["task_id"]]
        samples = record["samples"]
        pass_results = result["pass_results"]

        correct_codes = [
            strip_code_fences(sample)
            for sample, passed in zip(samples, pass_results)
            if passed
        ]

        if len(correct_codes) < 2:
            result["self_codebleu"] = None
            result["self_syntax_match"] = None
            result["self_dataflow_match"] = None
            continue

        # Deduplicate by AST fingerprint to avoid redundant expensive comparisons
        seen_fps: dict[str, str] = {}  # fingerprint -> code
        unique_codes = []
        for code in correct_codes:
            fp = ast_fingerprint(code)
            key = fp if fp is not None else code
            if key not in seen_fps:
                seen_fps[key] = code
                unique_codes.append(code)

        if len(unique_codes) < 2:
            # All correct samples are structurally identical
            result["self_codebleu"] = 0.0
            result["self_syntax_match"] = 0.0
            result["self_dataflow_match"] = 0.0
            continue

        bleu_scores, syntax_scores, dataflow_scores = [], [], []
        for i in range(len(unique_codes)):
            for j in range(i + 1, len(unique_codes)):
                try:
                    res = calc_codebleu(
                        references=[unique_codes[i]],
                        predictions=[unique_codes[j]],
                        lang="python",
                    )
                    bleu_scores.append(res["codebleu"])
                    syntax_scores.append(res["syntax_match_score"])
                    dataflow_scores.append(res["dataflow_match_score"])
                except Exception:
                    continue

        if not bleu_scores:
            result["self_codebleu"] = None
            result["self_syntax_match"] = None
            result["self_dataflow_match"] = None
            continue

        result["self_codebleu"] = round(1.0 - _mean(bleu_scores), 4)
        result["self_syntax_match"] = round(1.0 - _mean(syntax_scores), 4)
        result["self_dataflow_match"] = round(1.0 - _mean(dataflow_scores), 4)


def compute_self_codebleu_diversity(task_results: list[dict]) -> dict[str, float]:
    """Compute aggregate CodeBLEU diversity metrics across tasks with >=2 correct solutions."""
    metrics = {}
    for key in ("self_codebleu", "self_syntax_match", "self_dataflow_match"):
        values = [
            r[key] for r in task_results
            if r.get("num_correct", 0) >= 2 and r.get(key) is not None
        ]
        agg_key = key.replace("self_", "") + "_diversity"
        metrics[agg_key] = round(_mean(values), 4) if values else 0.0
    return metrics


def build_metrics_output(
    task_results: list[dict],
    records: list[dict],
    *,
    model: str,
    method: str,
    temperature: float,
    top_p: float | None = None,
    dataset: str,
    k_values: list[int],
    t_values: list[float],
) -> dict:
    """Aggregate all metrics into a single output dict.

    Expects task_results from evaluate_all() and the original records.
    Mutates task_results in-place (adds fingerprint/diversity fields).
    """
    add_distinct_counts(task_results, records)
    add_structural_diversity(task_results, records)
    add_self_codebleu(task_results, records)

    num_samples_per_task = len(records[0]["samples"]) if records else 0

    pass_at_k = compute_pass_at_k(task_results, k_values)
    cover_at_t, cover_at_t_distinct = compute_cover_at_t(
        task_results, t_values, num_samples_per_task
    )
    structural_diversity = compute_structural_diversity(task_results)
    codebleu_diversity = compute_self_codebleu_diversity(task_results)

    return {
        "model": model,
        "method": method,
        "temperature": temperature,
        **({"top_p": top_p} if top_p is not None else {}),
        "dataset": dataset,
        "num_tasks": len(records),
        "num_samples_per_task": num_samples_per_task,
        "pass_at_k": pass_at_k,
        "cover_at_t": cover_at_t,
        "cover_at_t_distinct": cover_at_t_distinct,
        "structural_diversity": structural_diversity,
        **codebleu_diversity,
        "per_task": task_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def compute_structural_diversity(task_results: list[dict]) -> float:
    """Compute aggregate structural diversity: mean of per-task mean_pairwise_distance.

    Only considers tasks with >=2 correct solutions.
    """
    distances = [
        r["mean_pairwise_distance"]
        for r in task_results
        if r.get("num_correct", 0) >= 2 and "mean_pairwise_distance" in r
    ]
    if not distances:
        return 0.0
    return round(sum(distances) / len(distances), 4)
