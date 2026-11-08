import fcntl
import json
from pathlib import Path


def get_output_path(results_dir: str, model_id: str, method: str, temperature: float, benchmark: str = None) -> Path:
    """Build the output JSONL path: results/<model-name>/[benchmark/]<method>_t<temp>.jsonl"""
    model_name = model_id.replace("/", "--")
    out_dir = Path(results_dir) / model_name
    if benchmark:
        out_dir = out_dir / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{method}_t{temperature}.jsonl"


def load_completed_ids(path: Path) -> set:
    """Read existing JSONL and return the set of completed task_ids."""
    completed = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def check_resume_compatible(path: Path, requested: dict) -> None:
    """Refuse to resume into a file whose records were generated under different opt-in settings.

    ``requested`` maps record keys to the values this run will write; ``None`` means "key absent" (the default runs,
    i.e. every historical file). E.g. ``{"seed": None, "single_bos": None}`` for a default run. Raises SystemExit
    naming the conflicting values and the fix (``--no-resume`` or another ``--results-dir``). Missing file: OK.
    """
    if not path.exists():
        return
    found = {k: set() for k in requested}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in requested:
                found[k].add(json.dumps(record.get(k)))
    bad = {k: sorted(v) for k, v in found.items() if v and v != {json.dumps(requested[k])}}
    if bad:
        details = "; ".join(f"{k}: file has {', '.join(v)}, this run would write {json.dumps(requested[k])}"
                            for k, v in bad.items())
        raise SystemExit(
            f"Refusing to resume {path}: its records were generated with different settings ({details}; "
            f"null = key absent). Use --no-resume to overwrite it, or a different --results-dir.")


def append_result(path: Path, record: dict) -> None:
    """Append a single JSON record to the JSONL file and flush.

    Holds an exclusive flock for the duration of the write. Records can be
    multi-MB (many generated samples per task), so a single os.write() is not
    guaranteed to be one atomic syscall -- without this lock, two processes
    appending to the same shared file (e.g. two full-dataset batches writing
    disjoint task_ids into one output file) could interleave partial writes
    and corrupt a line. No-op cost for the single-writer case (every other
    caller of this function).
    """
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
