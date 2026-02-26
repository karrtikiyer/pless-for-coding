import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def _build_program_mbpp(sample_code: str, test_list: list[str]) -> str:
    return sample_code + "\n" + "\n".join(test_list)


def _build_program_humaneval(sample_code: str, test: str, entry_point: str) -> str:
    return sample_code + "\n" + test + f"\ncheck({entry_point})\n"


def check_sample(code: str, tests: str, timeout: float = 5.0) -> bool:
    """Run code+tests in a subprocess, return True if it exits cleanly."""
    try:
        result = subprocess.run(
            ["python3", "-c", tests],
            timeout=timeout,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def evaluate_task(record: dict, dataset: str, timeout: float = 5.0) -> dict:
    """Run all samples for one task, return results dict."""
    samples = record["samples"]
    pass_results = []

    for sample in samples:
        if dataset == "mbpp":
            program = _build_program_mbpp(sample, record["test_list"])
        elif dataset == "humaneval":
            program = _build_program_humaneval(
                sample, record["test"], record["entry_point"]
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        passed = check_sample(sample, program, timeout=timeout)
        pass_results.append(passed)

    return {
        "task_id": record["task_id"],
        "num_correct": sum(pass_results),
        "pass_results": pass_results,
    }


def _evaluate_task_wrapper(args):
    """Wrapper for ProcessPoolExecutor (needs top-level picklable callable)."""
    record, dataset, timeout = args
    return evaluate_task(record, dataset, timeout)


def evaluate_all(
    records: list[dict],
    dataset: str,
    timeout: float = 5.0,
    workers: int = 4,
) -> list[dict]:
    """Evaluate all tasks in parallel."""
    args_list = [(r, dataset, timeout) for r in records]
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_evaluate_task_wrapper, args): args[0]["task_id"]
            for args in args_list
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by task_id for deterministic output
    results.sort(key=lambda r: r["task_id"])
    return results
