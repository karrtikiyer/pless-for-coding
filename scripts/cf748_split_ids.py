#!/usr/bin/env python
"""Split the CF-interview 748-id list into two disjoint halves for a two-GPU run.

Rule (k0.25 precedent): sort the ids ascending; ids at even sorted positions (0, 2, 4, ...) go to
the EVEN file (GPU0), ids at odd positions (1, 3, 5, ...) to the ODD file (GPU1). One id per line,
ascending, trailing newline (same format as data/cf_interview_748_ids.txt). Stdlib only, so it can
run with any python (``python -I``) on the GPU box.

Checks (any failure -> non-zero exit, nothing written): input has no duplicates / non-integers;
input count == --expected-count (748); halves disjoint, no duplicates, union == input; and (also in
`verify`) the halves follow the even/odd-by-sorted-position rule.
Writes <out-dir>/{cf748_even_ids.txt, cf748_odd_ids.txt, split_manifest.json}; refuses to
overwrite any of them.

Usage:
  uv run python scripts/cf748_split_ids.py split --input data/cf_interview_748_ids.txt --out-dir data/cf748_split
  python -I scripts/cf748_split_ids.py verify --input data/cf_interview_748_ids.txt \
      --a data/cf748_split/cf748_even_ids.txt --b data/cf748_split/cf748_odd_ids.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EVEN_NAME = "cf748_even_ids.txt"
ODD_NAME = "cf748_odd_ids.txt"
MANIFEST_NAME = "split_manifest.json"
RULE = ("ids sorted ascending; even = sorted positions 0,2,4,... (GPU0); "
        "odd = sorted positions 1,3,5,... (GPU1)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> list[int]:
    """Whitespace-separated integer ids (same parsing as the runner's --task-ids-file)."""
    toks = Path(path).read_text().split()
    try:
        ids = [int(t) for t in toks]
    except ValueError as exc:
        raise ValueError(f"{path}: non-integer id ({exc})") from None
    if len(set(ids)) != len(ids):
        seen, dups = set(), set()
        for i in ids:
            (dups if i in seen else seen).add(i)
        raise ValueError(f"{path}: duplicate ids {sorted(dups)[:10]}")
    return ids


def split_even_odd(ids: list[int]) -> tuple[list[int], list[int]]:
    s = sorted(ids)
    return s[0::2], s[1::2]


def verify_split(input_ids: list[int], a: list[int], b: list[int], expected_count: int) -> None:
    """Raise ValueError unless a/b are duplicate-free, disjoint, and union to input (of expected size)."""
    for name, xs in (("a", a), ("b", b), ("input", input_ids)):
        if len(set(xs)) != len(xs):
            raise ValueError(f"{name}: duplicate ids")
    if len(input_ids) != expected_count:
        raise ValueError(f"input count {len(input_ids)} != expected count {expected_count}")
    overlap = set(a) & set(b)
    if overlap:
        raise ValueError(f"a/b overlap on {len(overlap)} ids, e.g. {sorted(overlap)[:10]}")
    union = set(a) | set(b)
    if union != set(input_ids):
        missing = sorted(set(input_ids) - union)
        extra = sorted(union - set(input_ids))
        raise ValueError(f"union of a/b != input: missing={missing[:10]} ({len(missing)}) "
                         f"extra={extra[:10]} ({len(extra)})")
    if len(a) + len(b) != expected_count:
        raise ValueError(f"count a+b={len(a) + len(b)} != expected count {expected_count}")


def verify_even_odd_rule(input_ids: list[int], a: list[int], b: list[int]) -> None:
    """Raise ValueError unless a == even sorted positions and b == odd sorted positions of input."""
    even, odd = split_even_odd(input_ids)
    if sorted(a) != even or sorted(b) != odd:
        raise ValueError("split rule violated: a must be the ids at even sorted positions and b the ids "
                         "at odd sorted positions of the input (" + RULE + ")")


def _write_ids(path: Path, ids: list[int]) -> None:
    path.write_text("".join(f"{i}\n" for i in ids))


def cmd_split(args) -> int:
    src = Path(args.input)
    ids = read_ids(src)
    even, odd = split_even_odd(ids)
    verify_split(ids, even, odd, args.expected_count)
    out = Path(args.out_dir)
    targets = [out / EVEN_NAME, out / ODD_NAME, out / MANIFEST_NAME]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing} exists")
    out.mkdir(parents=True, exist_ok=True)
    _write_ids(targets[0], even)
    _write_ids(targets[1], odd)
    # Re-read what was written and re-verify (guards against any write/parse asymmetry).
    verify_split(ids, read_ids(targets[0]), read_ids(targets[1]), args.expected_count)
    verify_even_odd_rule(ids, read_ids(targets[0]), read_ids(targets[1]))
    manifest = {
        "rule": RULE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/cf748_split_ids.py",
        "script_sha256": sha256_file(Path(__file__)),
        "input": {"path": str(src), "sha256": sha256_file(src), "count": len(ids)},
        "even": {"path": str(targets[0]), "sha256": sha256_file(targets[0]), "count": len(even),
                 "first": even[:3], "last": even[-3:]},
        "odd": {"path": str(targets[1]), "sha256": sha256_file(targets[1]), "count": len(odd),
                "first": odd[:3], "last": odd[-3:]},
        "checks": ["no duplicates", "disjoint", "union == input", f"count == {args.expected_count}"],
    }
    targets[2].write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"split OK: input={len(ids)} even={len(even)} odd={len(odd)}")
    for k in ("input", "even", "odd"):
        print(f"  {k}: {manifest[k]['path']} sha256={manifest[k]['sha256']} count={manifest[k]['count']}")
    print(f"  manifest: {targets[2]}")
    return 0


def cmd_verify(args) -> int:
    ids = read_ids(Path(args.input))
    a, b = read_ids(Path(args.a)), read_ids(Path(args.b))
    verify_split(ids, a, b, args.expected_count)
    verify_even_odd_rule(ids, a, b)
    print(f"verify OK: input={len(ids)} a={len(a)} b={len(b)} disjoint, union == input, even/odd rule holds")
    for name, p in (("input", args.input), ("a", args.a), ("b", args.b)):
        print(f"  {name}: {p} sha256={sha256_file(Path(p))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="write the even/odd id files + manifest")
    s.add_argument("--input", default="data/cf_interview_748_ids.txt")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--expected-count", type=int, default=748)
    s.set_defaults(fn=cmd_split)
    v = sub.add_parser("verify", help="re-check two id files against the input (read-only)")
    v.add_argument("--input", default="data/cf_interview_748_ids.txt")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)
    v.add_argument("--expected-count", type=int, default=748)
    v.set_defaults(fn=cmd_verify)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
