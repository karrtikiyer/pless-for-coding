import json
from pathlib import Path


def load_results(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file, return list of dicts."""
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
