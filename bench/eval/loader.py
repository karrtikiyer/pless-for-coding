import gzip
import json
import lzma
from pathlib import Path


def load_results(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file (plain, gzip, or xz/lzma)."""
    path = Path(path)
    if path.suffix == ".gz":
        opener = gzip.open
    elif path.suffix == ".xz":
        opener = lzma.open
    else:
        opener = open
    records = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
