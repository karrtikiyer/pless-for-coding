"""Consolidate the Nov-2026 FULL regeneration results into ``results/regen_2026_11/``.

Dry run by default: builds the full old -> new file mapping, checks it, prints a per-target summary and
writes nothing (``--plan-dir DIR`` writes a preview of ``path_mapping.tsv`` / ``README.md`` into DIR,
which must lie outside ``<repo>/results/``).

``--execute``: refuses on ANY planning error (missing source root, filename collision between merged
sources, unmapped top-level entry, lock file, symlink, destination already present, mapping TSV already
present) or if a source and its destination are on different filesystems. Otherwise it writes
``results/regen_2026_11/path_mapping.tsv`` (old_path, new_path, sha256, bytes), moves every file with
``os.rename`` (never copy + delete), re-hashes every moved file against the sha256 recorded before the
move, writes ``results/regen_2026_11/README.md``, then removes the source directories that are empty
(``os.rmdir`` only; a directory still holding a file is left in place and reported).

Smoke / diag / fit / BOS-check roots are not sources and are never touched.

Usage (repo root):
    uv run python scripts/consolidate_regen_2026_11.py                       # dry run
    uv run python scripts/consolidate_regen_2026_11.py --plan-dir /tmp/regen_plan
    uv run python scripts/consolidate_regen_2026_11.py --execute
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CODE_REPO = Path(__file__).resolve().parents[1]
TARGET_ROOT = "results/regen_2026_11"
MAPPING_TSV = f"{TARGET_ROOT}/path_mapping.tsv"
README = f"{TARGET_ROOT}/README.md"
TSV_HEADER = "old_path\tnew_path\tsha256\tbytes"

_META_RE = re.compile(r"^(MANIFEST|README)((?:\.[^.]+)*)\.(json|md)$")   # MANIFEST[.host].json, README[.host].md


class ConsolidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    """One old root. ``mode``:
    * ``as_is``: every file keeps its path relative to the root.
    * ``merge``: top-level model dirs (``org--name``) keep their path; top-level dirs in ``dir_map`` are
      re-rooted; top-level files in ``file_map`` keep/rename; MANIFEST*/README* files get ``.<tag>``
      inserted after the stem when ``tag`` is set. Anything else is an "unmapped" error."""
    old_root: str
    new_root: str
    label: str
    benchmark: str
    mode: str = "merge"
    tag: str | None = None
    dir_map: dict = field(default_factory=dict)
    file_map: dict = field(default_factory=dict)


SOURCES: list[Source] = [
    Source("results/humaneval_regen_2026_11_stripped", f"{TARGET_ROOT}/humaneval/main",
           "HumanEval main (stripped protocol, 8 paper models)", "humaneval",
           tag="stripped", dir_map={"analysis": "analysis/main"}),
    Source("results/humaneval_regen_2026_11_stripped_extra", f"{TARGET_ROOT}/humaneval/main",
           "HumanEval extra temperatures (p-less / p-less-norm T=1.0, T=0.6; analysis only)", "humaneval",
           tag="stripped_extra", dir_map={"analysis": "analysis/extra_temps"}),
    Source("results/humaneval_regen_2026_11_instruct7b", f"{TARGET_ROOT}/humaneval/main",
           "HumanEval Qwen2.5-Coder-7B-Instruct (chat protocol; analysis only)", "humaneval",
           tag="instruct7b", dir_map={"analysis": "analysis/main"}),
    Source("results/humaneval_regen_2026_11_instruct7b_extra", f"{TARGET_ROOT}/humaneval/main",
           "HumanEval Qwen2.5-Coder-7B-Instruct extra temperatures (analysis only)", "humaneval",
           tag="instruct7b_extra", dir_map={"analysis": "analysis/extra_temps"}),
    Source("results/humaneval_regen_2026_11", f"{TARGET_ROOT}/humaneval/superseded_unstripped",
           "HumanEval unstripped protocol (SUPERSEDED by the stripped protocol)", "humaneval", mode="as_is"),
    Source("results/mbpp_seeded_2026_11", f"{TARGET_ROOT}/mbpp/main",
           "MBPP-full seeded (9 models)", "mbpp",
           dir_map={"analysis": "analysis", "gpu2_logs": "logs"},
           file_map={"grading_log_2026_11.tsv": "grading_log_2026_11.tsv"}),
    Source("results/mbpp_qwen25coder7b_2026_11", f"{TARGET_ROOT}/mbpp/unseeded_qwen25coder7b",
           "MBPP-full unseeded Qwen2.5-Coder-7B (base, BigCode prompt)", "mbpp", mode="as_is"),
    Source("results/mbpp_qwen25coder7binstruct_2026_11", f"{TARGET_ROOT}/mbpp/unseeded_qwen25coder7b",
           "MBPP-full unseeded Qwen2.5-Coder-7B-Instruct (chat template)", "mbpp", mode="as_is"),
]


@dataclass(frozen=True)
class Entry:
    old: str          # repo-relative POSIX path
    new: str
    sha256: str
    size: int
    source: str       # Source.old_root


@dataclass
class Plan:
    entries: list[Entry]
    errors: list[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tagged_meta(name: str, tag: str) -> str | None:
    m = _META_RE.match(name)
    if not m:
        return None
    stem, middle, ext = m.groups()
    return f"{stem}.{tag}{middle}.{ext}"


def _map_rel(src: Source, rel: Path) -> str | None:
    """New path (relative to src.new_root) for ``rel`` (relative to src.old_root); None if unmapped."""
    if src.mode == "as_is":
        return rel.as_posix()
    top, rest = rel.parts[0], rel.parts[1:]
    if not rest:   # a top-level file
        if top in src.file_map:
            return src.file_map[top]
        if src.tag:
            return _tagged_meta(top, src.tag)
        return None
    if top in src.dir_map:
        return Path(src.dir_map[top], *rest).as_posix()
    if "--" in top:   # model dir, e.g. Qwen--Qwen2.5-Coder-3B
        return rel.as_posix()
    return None


def build_plan(repo_root: Path, sources: list[Source] = SOURCES) -> Plan:
    repo_root = Path(repo_root)
    entries: list[Entry] = []
    errors: list[str] = []
    unmapped_tops: set[str] = set()
    for src in sources:
        root = repo_root / src.old_root
        if not root.is_dir():
            errors.append(f"missing source root: {src.old_root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for d in dirnames:
                if (Path(dirpath) / d).is_symlink():
                    errors.append(f"symlink (directory) in source: {(Path(dirpath) / d).relative_to(repo_root)}")
            for f in sorted(filenames):
                p = Path(dirpath) / f
                rel = p.relative_to(root)
                old = p.relative_to(repo_root).as_posix()
                if p.is_symlink():
                    errors.append(f"symlink in source: {old}")
                    continue
                if f.endswith(".lock"):
                    errors.append(f"lock file in source (a writer may be active; remove it by hand first): {old}")
                    continue
                new_rel = _map_rel(src, rel)
                if new_rel is None:
                    top = f"{src.old_root}/{rel.parts[0]}"
                    if top not in unmapped_tops:
                        unmapped_tops.add(top)
                        errors.append(f"unmapped top-level entry: {top}")
                    continue
                entries.append(Entry(old, f"{src.new_root}/{new_rel}", sha256_file(p), p.stat().st_size,
                                     src.old_root))
    by_new: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_new[e.new].append(e.old)
    for new, olds in sorted(by_new.items()):
        if len(olds) > 1:
            errors.append(f"collision: {new} <- {', '.join(olds)}")
    reserved = {MAPPING_TSV, README}
    for e in entries:
        if e.new in reserved:
            errors.append(f"collision: {e.new} is reserved for the consolidation index <- {e.old}")
        if (repo_root / e.new).exists() or (repo_root / e.new).is_symlink():
            errors.append(f"destination exists: {e.new}")
    for r in sorted(reserved):
        if (repo_root / r).exists():
            errors.append(f"destination exists: {r} (already consolidated?)")
    return Plan(entries, errors)


# ---------------------------------------------------------------- index files

def render_mapping_tsv(plan: Plan) -> str:
    return "\n".join([TSV_HEADER] + [f"{e.old}\t{e.new}\t{e.sha256}\t{e.size}" for e in plan.entries]) + "\n"


def _first_record(repo_root: Path, rel: str) -> dict:
    try:
        with open(repo_root / rel) as f:
            line = f.readline()
        return json.loads(line) if line.strip() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _protocol_rows(repo_root: Path, plan: Plan, new_root: str, use_new: bool, mbpp: bool = False) -> list[str]:
    """Per (model, protocol fields) the config stems, read from the first record of each JSONL. MBPP tables add
    ``single_bos`` (the record's value; ``-`` = field absent, i.e. the runner default: no --single-bos)."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for e in plan.entries:
        if not e.new.startswith(new_root + "/") or not e.new.endswith(".jsonl"):
            continue
        parts = Path(e.new).relative_to(new_root).parts
        if len(parts) != 2 or "--" not in parts[0]:
            continue
        rec = _first_record(repo_root, e.new if use_new else e.old)
        key = (parts[0], Path(e.old).parts[1], rec.get("backend", "?"), rec.get("prompt_style", "-"),
               rec.get("prompt_variant", "-"), rec.get("seed", "unset"))
        if mbpp:
            key += (rec.get("single_bos", "-"),)
        groups[key].append(Path(parts[1]).stem)
    cols = ["model dir", "from root", "backend", "prompt_style", "prompt_variant", "seed"] + (["single_bos"] if mbpp else [])
    rows = ["| " + " | ".join(cols + ["config files"]) + " |", "|" + "---|" * (len(cols) + 1)]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows.append("| " + " | ".join(str(x) for x in key) + " | " + ", ".join(sorted(groups[key])) + " |")
    return rows


def _old_root_re() -> re.Pattern:
    """Any moved source root's folder name as a whole word (``humaneval_regen_2026_11`` does not match ``..._smoke``)."""
    names = sorted({Path(s.old_root).name for s in SOURCES}, key=len, reverse=True)
    return re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, names)) + r")(?![\w])")


def _stale_analysis_files(repo_root: Path, plan: Plan, use_new: bool) -> dict[str, list[str]]:
    """Per moved ``analysis/`` dir: the archived non-.md files whose text names a pre-move source root."""
    pat = _old_root_re()
    out: dict[str, list[str]] = defaultdict(list)
    for e in plan.entries:
        if "/analysis/" not in e.new or e.new.endswith(".md"):
            continue
        try:
            text = (repo_root / (e.new if use_new else e.old)).read_text(errors="ignore")
        except OSError:
            continue
        if pat.search(text):
            adir, rel = e.new.split("/analysis/", 1)
            out[f"{adir}/analysis"].append(rel)
    return {k: sorted(v) for k, v in sorted(out.items())}


def render_readme(plan: Plan, provenance: str, repo_root: Path | None = None, use_new: bool = False) -> str:
    repo_root = Path(repo_root) if repo_root else CODE_REPO
    counts = Counter(e.source for e in plan.entries)
    sizes = Counter()
    for e in plan.entries:
        sizes[e.source] += e.size
    lines = [
        "# Nov-2026 regeneration results (consolidated)",
        "",
        "Written by `scripts/consolidate_regen_2026_11.py`. Every moved file is listed in `path_mapping.tsv`",
        "(old_path, new_path, sha256, bytes; sha256 recorded before the move and re-verified after it).",
        "",
        f"Provenance: {provenance}",
        "",
        "## Old root -> new location",
        "",
        "| old root | new location | content | files | bytes |",
        "|---|---|---|---:|---:|",
    ]
    for s in SOURCES:
        lines.append(f"| `{s.old_root}/` | `{s.new_root}/` | {s.label} | {counts[s.old_root]} | {sizes[s.old_root]} |")
    lines += [
        "",
        "Within `humaneval/main/`: model files of the four merged roots share `<model>/` (file names never",
        "collide: the consolidation refuses on any collision); analysis files of the stripped / instruct7b roots",
        "are under `analysis/main/`, those of the `*_extra` roots under `analysis/extra_temps/`. Each root's own",
        "MANIFEST / README is kept with the source tag inserted (e.g. `MANIFEST.gpu2.json` of",
        "`humaneval_regen_2026_11_stripped` -> `MANIFEST.stripped.gpu2.json`). `mbpp/main/logs/` is the old",
        "`gpu2_logs/` of `mbpp_seeded_2026_11`.",
        "",
        "Logs folder names differ on purpose and are not renamed: `mbpp/main/logs/` (was `mbpp_seeded_2026_11/gpu2_logs/`)",
        "vs `mbpp/unseeded_qwen25coder7b/gpu2_logs/` (moved as-is from `mbpp_qwen25coder7b_2026_11/gpu2_logs/`).",
        "Paths in `path_mapping.tsv` and the archived scripts depend on these names.",
        "",
        "## Index files kept from the source roots",
        "",
    ]
    for e in plan.entries:
        name = Path(e.new).name
        if _META_RE.match(name) and Path(e.new).parent.as_posix() in {s.new_root for s in SOURCES}:
            lines.append(f"- `{e.new}` (was `{e.old}`)")
    stale = _stale_analysis_files(repo_root, plan, use_new)
    lines += [
        "",
        "## Archived analysis scripts hardcode the pre-move roots",
        "",
        "The scripts / JSON / logs inside the moved `analysis/` dirs were written before the move and name the old",
        "roots (e.g. `results/mbpp_seeded_2026_11/...`). Do not rerun them as-is (they would read missing paths or,",
        "if an old root is re-created, the wrong files); translate paths with `path_mapping.tsv` (old_path ->",
        "new_path) first. Files (non-.md) whose text names a moved source root, per analysis dir:",
        "",
    ]
    for adir, rels in stale.items():
        lines.append(f"- `{adir}/`: {len(rels)} files (" + ", ".join(f"`{r}`" for r in rels) + ")")
    if not stale:
        lines.append("- none found")
    lines += ["", "## Protocol per file (from the first record of each JSONL)", "",
              "Each row is read from the FIRST record of each JSONL only (the fields are per run, not per task).",
              "MBPP records store the raw MBPP problem text (`prompt_text`) only, not the formatted prompt and no",
              "prompt-style field: MBPP prompt formats are inferred from the file stems + runner code",
              "(`bench/runner.py`, `bench/prompts.py`, `scripts/mbpp_staging_common.sh`): `_bigcode` stems = BigCode",
              "zero-shot (`--prompt-style bigcode`, base models); other stems = the runner default, i.e. the 3-shot",
              "prompt for base models (CodeLlama-7b-hf, Llama-2-7b-hf) and the chat template for chat/instruct models",
              "(`bench/prompts.py:is_instruct_model`). MBPP `single_bos`: the record's value (`True` = `--single-bos`,",
              "one BOS); `-` = field absent = runner default (no `--single-bos`).",
              ""]
    for new_root in dict.fromkeys(s.new_root for s in SOURCES):
        mbpp = any(s.benchmark == "mbpp" for s in SOURCES if s.new_root == new_root)
        lines += [f"### `{new_root}/`", ""] + _protocol_rows(repo_root, plan, new_root, use_new, mbpp) + [""]
    lines += [
        "File-name conventions: HumanEval `<config>__prompt_stripped.jsonl` = `prompt.strip()` (base models),",
        "`<config>.jsonl` = chat template or unstripped prompt (each record's `prompt_variant` is authoritative;",
        "`bench/humaneval/regen.py`). MBPP `_bigcode` stems = `--prompt-style bigcode` (base models), other stems =",
        "the runner's default prompt (`scripts/mbpp_staging_common.sh`). Plan and run log:",
        "`docs/experiments/humaneval_regen_2026_11_plan.md`, `docs/experiments/humaneval_regen_2026_11_runlog.md`.",
        "",
    ]
    return "\n".join(lines)


def _provenance(argv: list[str]) -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=CODE_REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE_REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        commit += " (working tree dirty)" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    ist = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S IST")
    return f"{ist}; git {commit}; command: `uv run python {' '.join(shlex.quote(a) for a in argv)}`"


# ---------------------------------------------------------------- execution

def _nearest_existing(p: Path) -> Path:
    while not p.exists():
        p = p.parent
    return p


def _same_device(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(_nearest_existing(b)).st_dev


def verify_moved(repo_root: Path, entries: list[Entry]) -> None:
    bad = []
    for e in entries:
        dest = repo_root / e.new
        if not dest.is_file():
            bad.append(f"missing after move: {e.new}")
        elif sha256_file(dest) != e.sha256:
            bad.append(f"sha256 mismatch after move: {e.new}")
        if (repo_root / e.old).exists():
            bad.append(f"source still present after move: {e.old}")
    if bad:
        raise ConsolidationError("; ".join(bad[:20]) + (f" (+{len(bad) - 20} more)" if len(bad) > 20 else ""))


def _remove_empty_dirs(root: Path) -> list[str]:
    """os.rmdir every empty directory under (and including) root, bottom-up; return the ones left."""
    left = []
    if not root.is_dir():
        return left
    for dirpath, _, _ in os.walk(root, topdown=False):
        try:
            os.rmdir(dirpath)
        except OSError:
            left.append(dirpath)
    return left


def execute_plan(repo_root: Path, plan: Plan, provenance: str) -> list[str]:
    repo_root = Path(repo_root)
    if plan.errors:
        raise ConsolidationError(f"{len(plan.errors)} planning error(s); first: {plan.errors[0]}")
    for e in plan.entries:   # re-check content and destination right before touching anything
        if sha256_file(repo_root / e.old) != e.sha256:
            raise ConsolidationError(f"source changed since planning: {e.old}")
    cross = sorted({e.source for e in plan.entries if not _same_device(repo_root / e.old, repo_root / e.new)})
    if cross:
        raise ConsolidationError(f"source and destination on different filesystems: {cross}")
    (repo_root / TARGET_ROOT).mkdir(parents=True, exist_ok=True)
    tsv = repo_root / MAPPING_TSV
    if tsv.exists():
        raise ConsolidationError(f"{MAPPING_TSV} exists")
    tsv.write_text(render_mapping_tsv(plan))   # written first: the record needed to recover a partial move
    for e in plan.entries:
        src, dest = repo_root / e.old, repo_root / e.new
        if dest.exists() or dest.is_symlink():
            raise ConsolidationError(f"destination appeared during the move: {e.new}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dest)
    verify_moved(repo_root, plan.entries)
    (repo_root / README).write_text(render_readme(plan, provenance, repo_root, use_new=True))
    left = []
    for root in dict.fromkeys(s.old_root for s in SOURCES):
        left += _remove_empty_dirs(repo_root / root)
    return left


def summarize(plan: Plan) -> list[str]:
    per_target = Counter()
    for e in plan.entries:
        parts = Path(e.new).relative_to(TARGET_ROOT).parts
        sub = parts[2] if len(parts) > 3 and parts[2] == "analysis" else ("<model dirs>" if "--" in parts[2]
                                                                           else parts[2])
        per_target[f"{parts[0]}/{parts[1]}  [{sub}]"] += 1
    lines = [f"{n:5d}  {TARGET_ROOT}/{k}" for k, n in sorted(per_target.items())]
    per_src = Counter(e.source for e in plan.entries)
    lines += [f"{per_src[s.old_root]:5d}  from {s.old_root} -> {s.new_root}" for s in SOURCES]
    lines.append(f"{len(plan.entries):5d}  files total, {sum(e.size for e in plan.entries)} bytes")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=CODE_REPO)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True, help="(default) plan and check only")
    g.add_argument("--execute", action="store_true", help="move the files (os.rename) and write the index")
    ap.add_argument("--plan-dir", type=Path, default=None,
                    help="dry run: write a preview path_mapping.tsv / README.md here (must be outside <repo>/results)")
    args = ap.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.plan_dir is not None:
        pd = args.plan_dir.resolve()
        if pd == repo_root / "results" or (repo_root / "results") in pd.parents:
            ap.error(f"--plan-dir must be outside {repo_root / 'results'}; got {pd}")
    plan = build_plan(repo_root)
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"[{mode}] repo {repo_root}")
    for line in summarize(plan):
        print(line)
    if plan.errors:
        print(f"REFUSING: {len(plan.errors)} error(s):")
        for err in plan.errors:
            print(f"  - {err}")
        return 2
    prov = _provenance(sys.argv if argv is None else ["scripts/consolidate_regen_2026_11.py", *argv])
    if not args.execute:
        if args.plan_dir is not None:
            args.plan_dir.mkdir(parents=True, exist_ok=True)
            (args.plan_dir / "path_mapping.tsv").write_text(render_mapping_tsv(plan))
            (args.plan_dir / "README.md").write_text(render_readme(plan, prov + " (DRY-RUN PREVIEW)", repo_root))
            print(f"preview written to {args.plan_dir}")
        print("DRY RUN: nothing moved. Re-run with --execute to move.")
        return 0
    try:
        left = execute_plan(repo_root, plan, prov)
    except ConsolidationError as exc:
        print(f"REFUSING / FAILED: {exc}")
        return 2
    print(f"moved and verified {len(plan.entries)} files; wrote {MAPPING_TSV} and {README}")
    if left:
        print("non-empty source directories left in place:")
        for d in left:
            print(f"  - {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
