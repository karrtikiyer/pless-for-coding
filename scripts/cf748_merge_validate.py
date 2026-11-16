#!/usr/bin/env python
"""Merge + validate the two per-GPU result dirs of ONE CF-748 arm into the standard layout.

Each GPU's run (separate RESULTS_DIR, disjoint task-id halves from scripts/cf748_split_ids.py)
writes <RESULTS_DIR>/<model_dir>/<SOURCE>_<difficulty>/<method_key>_t<T>.jsonl, the same layout as
results/_cf_interview_748/Qwen--Qwen3-8B/CODEFORCES_interview/. This script writes the union into
<out-results-dir>/<model_dir>/<SOURCE>_<difficulty>/<same filename>, lines copied VERBATIM (bytes
unchanged, no re-serialisation) and sorted by task_id, plus <file>.merge_manifest.json.

Checks (any failure -> non-zero exit, nothing written):
  every line parses as a JSON object with an int task_id (blank lines are errors);
  per record: len(samples) == --n-samples (and samples_with_thinking, if present);
  method / temperature / model / source / difficulty / backend match the expected values;
  no duplicate task_id within a file; every id is in --ids-file; the two inputs are disjoint;
  union == --ids-file ids; --ids-file has --expected-count (748) ids;
  optional --split-a/--split-b: input a's ids == split-a's ids (b likewise);
  refuses to overwrite an existing output file or manifest.
Stdlib only (runs with ``python -I`` anywhere).

  --check-only            validate, write nothing (exit 0 only if mergeable)
  --check-only --allow-incomplete
                          progress/integrity check of a running job: corruption checks stay
                          strict, missing ids are only reported. Never writes.

Usage:
  python -I scripts/cf748_merge_validate.py \
      --in-a results/_cf748_effk_nofloor_split/gpu0 --in-b results/_cf748_effk_nofloor_split/gpu1 \
      --split-a data/cf748_split/cf748_even_ids.txt --split-b data/cf748_split/cf748_odd_ids.txt \
      --method pless_effk_nofloor --temperature 1.0 --out-results-dir results/_cf748_effk_nofloor_merged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class MergeError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_id_file(path: Path) -> list[int]:
    toks = Path(path).read_text().split()
    try:
        ids = [int(t) for t in toks]
    except ValueError as exc:
        raise MergeError(f"{path}: non-integer id ({exc})") from None
    if len(set(ids)) != len(ids):
        raise MergeError(f"{path}: duplicate ids in id file")
    return ids


def default_filename(method: str, temperature: float) -> str:
    # bench.apps.runner._method_key for a plain (non-temp, non-split, non-alpha/renyi) method with
    # --enable-thinking, then _output_path's f"{key}_t{T}.jsonl".
    return f"{method}_think_t{temperature}_t{temperature}.jsonl"


def resolve_input(p: str, model_dir: str, bucket: str, filename: str) -> Path:
    path = Path(p)
    if path.is_file():
        return path
    f = path / model_dir / bucket / filename
    if not f.is_file():
        raise MergeError(f"input missing: {f}")
    return f


def load_records(path: Path, exp: dict, n_samples: int, allowed_ids: set[int]) -> dict[int, bytes]:
    """Return {task_id: raw line bytes (with trailing newline)}; raise MergeError on any violation."""
    out: dict[int, bytes] = {}
    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]  # trailing newline after the last record
    for ln, raw in enumerate(lines, 1):
        where = f"{path}:{ln}"
        if not raw.strip():
            raise MergeError(f"{where}: blank line")
        try:
            rec = json.loads(raw)
        except Exception as exc:
            raise MergeError(f"{where}: unparseable JSON ({type(exc).__name__})") from None
        if not isinstance(rec, dict):
            raise MergeError(f"{where}: unparseable record (not a JSON object)")
        tid = rec.get("task_id")
        if not isinstance(tid, int) or isinstance(tid, bool):
            raise MergeError(f"{where}: task_id missing or not an int ({tid!r})")
        for key, want in exp.items():
            got = rec.get(key)
            if got != want:
                raise MergeError(f"{where}: task_id={tid} {key}={got!r} != expected {want!r}")
        s = rec.get("samples")
        if not isinstance(s, list) or len(s) != n_samples:
            raise MergeError(f"{where}: task_id={tid} samples has "
                             f"{len(s) if isinstance(s, list) else type(s).__name__} != {n_samples}")
        if "samples_with_thinking" in rec:
            sw = rec["samples_with_thinking"]
            if not isinstance(sw, list) or len(sw) != n_samples:
                raise MergeError(f"{where}: task_id={tid} samples_with_thinking has "
                                 f"{len(sw) if isinstance(sw, list) else type(sw).__name__} != {n_samples}")
        if tid in out:
            raise MergeError(f"{where}: duplicate task_id {tid} within {path}")
        if tid not in allowed_ids:
            raise MergeError(f"{where}: task_id {tid} not in ids file")
        out[tid] = raw + b"\n"
    return out


def run(args) -> int:
    if args.allow_incomplete and not args.check_only:
        raise MergeError("--allow-incomplete is only valid with --check-only (an incomplete merge is never written)")
    if not args.check_only and not args.out_results_dir:
        raise MergeError("--out-results-dir is required unless --check-only")
    bucket = f"{args.source}_{args.difficulty}"
    filename = args.filename or default_filename(args.method, args.temperature)
    ids = read_id_file(Path(args.ids_file))
    if len(ids) != args.expected_count:
        raise MergeError(f"ids file {args.ids_file} has {len(ids)} ids, expected count {args.expected_count}")
    allowed = set(ids)
    pa = resolve_input(args.in_a, args.model_dir, bucket, filename)
    pb = resolve_input(args.in_b, args.model_dir, bucket, filename)
    if pa.resolve() == pb.resolve():
        raise MergeError(f"--in-a and --in-b resolve to the same file: {pa}")
    exp = {"method": args.method, "temperature": args.temperature, "model": args.model,
           "source": args.source, "difficulty": args.difficulty, "backend": args.backend}
    sha_a, sha_b = sha256_file(pa), sha256_file(pb)
    ra = load_records(pa, exp, args.n_samples, allowed)
    rb = load_records(pb, exp, args.n_samples, allowed)
    overlap = set(ra) & set(rb)
    if overlap:
        raise MergeError(f"inputs overlap on {len(overlap)} task_ids, e.g. {sorted(overlap)[:10]}")
    split_info = {}
    for side, recs, sp in (("a", ra, args.split_a), ("b", rb, args.split_b)):
        if sp is None:
            continue
        want = set(read_id_file(Path(sp)))
        got = set(recs)
        bad = (got != want) if not args.allow_incomplete else not got <= want
        if bad:
            raise MergeError(f"input {side} ids do not match split file {sp}: "
                             f"missing={sorted(want - got)[:10]} ({len(want - got)}) "
                             f"extra={sorted(got - want)[:10]} ({len(got - want)})")
        split_info[side] = {"path": str(sp), "sha256": sha256_file(Path(sp)), "count": len(want)}
    union = set(ra) | set(rb)
    missing = sorted(allowed - union)
    summary = f"a={len(ra)} b={len(rb)} union={len(union)} expected={len(ids)} missing={len(missing)}"
    if missing and not args.allow_incomplete:
        raise MergeError(f"union of inputs != ids file: {summary}; missing e.g. {missing[:10]}")
    if args.check_only:
        print(f"CHECK OK ({'incomplete allowed' if args.allow_incomplete else 'complete'}): {summary}")
        print(f"  a: {pa} sha256={sha_a}\n  b: {pb} sha256={sha_b}")
        return 0

    out_dir = Path(args.out_results_dir) / args.model_dir / bucket
    out_file = out_dir / filename
    man_file = Path(str(out_file) + ".merge_manifest.json")
    existing = [str(p) for p in (out_file, man_file) if p.exists()]
    if existing:
        raise MergeError(f"refusing to overwrite: {existing} exists")
    # Inputs must not have changed while we validated them.
    if sha256_file(pa) != sha_a or sha256_file(pb) != sha_b:
        raise MergeError("an input file changed during validation (still being written?)")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = {**ra, **rb}
    tmp = out_dir / f".{filename}.tmp.{os.getpid()}"
    with open(tmp, "xb") as fh:
        for tid in sorted(merged):
            fh.write(merged[tid])
    # Re-validate what was written before publishing it.
    back = load_records(tmp, exp, args.n_samples, allowed)
    if sorted(back) != sorted(ids) or any(back[t] != merged[t] for t in back):
        tmp.unlink()
        raise MergeError("post-write re-validation failed")
    if out_file.exists():
        tmp.unlink()
        raise MergeError(f"refusing to overwrite: {out_file} exists")
    os.replace(tmp, out_file)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/cf748_merge_validate.py",
        "script_sha256": sha256_file(Path(__file__)),
        "expected": {**exp, "n_samples": args.n_samples, "expected_count": args.expected_count},
        "ids_file": {"path": str(args.ids_file), "sha256": sha256_file(Path(args.ids_file)), "count": len(ids)},
        "inputs": {"a": {"path": str(pa), "sha256": sha_a, "count": len(ra)},
                   "b": {"path": str(pb), "sha256": sha_b, "count": len(rb)}},
        "split_files": split_info,
        "output": {"path": str(out_file), "sha256": sha256_file(out_file), "count": len(merged),
                   "order": "sorted by task_id; lines copied verbatim from inputs"},
        "checks": ["every line parses", f"{args.n_samples} samples per record", "method/temperature/model/"
                   "source/difficulty/backend match", "no duplicate task_ids", "inputs disjoint",
                   "union == ids file", "ids file count == expected", "post-write re-validation"],
    }
    with open(man_file, "x") as fh:
        fh.write(json.dumps(manifest, indent=2) + "\n")
    print(f"MERGE OK: {summary}")
    print(f"  out: {out_file} sha256={manifest['output']['sha256']}")
    print(f"  manifest: {man_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-a", required=True, help="GPU0 RESULTS_DIR (or the jsonl itself)")
    ap.add_argument("--in-b", required=True, help="GPU1 RESULTS_DIR (or the jsonl itself)")
    ap.add_argument("--out-results-dir", default=None)
    ap.add_argument("--ids-file", default="data/cf_interview_748_ids.txt")
    ap.add_argument("--expected-count", type=int, default=748)
    ap.add_argument("--split-a", default=None, help="id file GPU0 was given (exact match required)")
    ap.add_argument("--split-b", default=None, help="id file GPU1 was given (exact match required)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-dir", default=None, help="default: --model with '/' -> '--'")
    ap.add_argument("--source", default="CODEFORCES")
    ap.add_argument("--difficulty", default="interview")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--filename", default=None, help="default: <method>_think_t<T>_t<T>.jsonl")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.model_dir is None:
        args.model_dir = args.model.replace("/", "--")
    try:
        return run(args)
    except (MergeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
