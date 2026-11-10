"""Tests for scripts/consolidate_regen_2026_11.py on synthetic tmp trees (never the real results/)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "consolidate_regen_2026_11.py"


def _load():
    name = "consolidate_regen_2026_11"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load()

M = "Qwen--Qwen2.5-Coder-3B"
I7 = "Qwen--Qwen2.5-Coder-7B-Instruct"

# old relative path -> expected new relative path (one file per mapping rule)
EXPECTED = {
    # HumanEval stripped (main)
    f"results/humaneval_regen_2026_11_stripped/{M}/greedy__prompt_stripped.jsonl":
        f"results/regen_2026_11/humaneval/main/{M}/greedy__prompt_stripped.jsonl",
    f"results/humaneval_regen_2026_11_stripped/{M}/metrics/greedy__prompt_stripped_metrics.json":
        f"results/regen_2026_11/humaneval/main/{M}/metrics/greedy__prompt_stripped_metrics.json",
    "results/humaneval_regen_2026_11_stripped/analysis/final_pass_tables.md":
        "results/regen_2026_11/humaneval/main/analysis/main/final_pass_tables.md",
    "results/humaneval_regen_2026_11_stripped/analysis/grading_scripts/check.py":
        "results/regen_2026_11/humaneval/main/analysis/main/grading_scripts/check.py",
    "results/humaneval_regen_2026_11_stripped/MANIFEST.gpu2.json":
        "results/regen_2026_11/humaneval/main/MANIFEST.stripped.gpu2.json",
    "results/humaneval_regen_2026_11_stripped/MANIFEST.msi.json":
        "results/regen_2026_11/humaneval/main/MANIFEST.stripped.msi.json",
    "results/humaneval_regen_2026_11_stripped/README.gpu2.md":
        "results/regen_2026_11/humaneval/main/README.stripped.gpu2.md",
    # HumanEval stripped extra
    f"results/humaneval_regen_2026_11_stripped_extra/{M}/pless_t1.0__prompt_stripped.jsonl":
        f"results/regen_2026_11/humaneval/main/{M}/pless_t1.0__prompt_stripped.jsonl",
    "results/humaneval_regen_2026_11_stripped_extra/analysis/grading_scripts/check.py":
        "results/regen_2026_11/humaneval/main/analysis/extra_temps/grading_scripts/check.py",
    "results/humaneval_regen_2026_11_stripped_extra/MANIFEST.gpu2.json":
        "results/regen_2026_11/humaneval/main/MANIFEST.stripped_extra.gpu2.json",
    "results/humaneval_regen_2026_11_stripped_extra/README.msi.md":
        "results/regen_2026_11/humaneval/main/README.stripped_extra.msi.md",
    # instruct7b (+ extra)
    f"results/humaneval_regen_2026_11_instruct7b/{I7}/greedy.jsonl":
        f"results/regen_2026_11/humaneval/main/{I7}/greedy.jsonl",
    "results/humaneval_regen_2026_11_instruct7b/MANIFEST.gpu2.json":
        "results/regen_2026_11/humaneval/main/MANIFEST.instruct7b.gpu2.json",
    f"results/humaneval_regen_2026_11_instruct7b_extra/{I7}/metrics/pless_t0.6_metrics.json":
        f"results/regen_2026_11/humaneval/main/{I7}/metrics/pless_t0.6_metrics.json",
    "results/humaneval_regen_2026_11_instruct7b_extra/README.gpu2.md":
        "results/regen_2026_11/humaneval/main/README.instruct7b_extra.gpu2.md",
    # superseded unstripped, as-is (incl. its own MANIFEST.json / README.md)
    "results/humaneval_regen_2026_11/MANIFEST.json":
        "results/regen_2026_11/humaneval/superseded_unstripped/MANIFEST.json",
    "results/humaneval_regen_2026_11/README.md":
        "results/regen_2026_11/humaneval/superseded_unstripped/README.md",
    f"results/humaneval_regen_2026_11/{M}/greedy.jsonl":
        f"results/regen_2026_11/humaneval/superseded_unstripped/{M}/greedy.jsonl",
    # MBPP seeded
    f"results/mbpp_seeded_2026_11/{M}/greedy_bigcode_t1.0.jsonl":
        f"results/regen_2026_11/mbpp/main/{M}/greedy_bigcode_t1.0.jsonl",
    f"results/mbpp_seeded_2026_11/{M}/metrics/greedy_bigcode_t1.0_metrics.json":
        f"results/regen_2026_11/mbpp/main/{M}/metrics/greedy_bigcode_t1.0_metrics.json",
    "results/mbpp_seeded_2026_11/analysis/stats.json": "results/regen_2026_11/mbpp/main/analysis/stats.json",
    "results/mbpp_seeded_2026_11/gpu2_logs/q3b_seeded_driver.log":
        "results/regen_2026_11/mbpp/main/logs/q3b_seeded_driver.log",
    "results/mbpp_seeded_2026_11/grading_log_2026_11.tsv": "results/regen_2026_11/mbpp/main/grading_log_2026_11.tsv",
    # MBPP unseeded Qwen2.5-Coder-7B base + instruct, merged as-is
    "results/mbpp_qwen25coder7b_2026_11/Qwen--Qwen2.5-Coder-7B/greedy_bigcode_t1.0.jsonl":
        "results/regen_2026_11/mbpp/unseeded_qwen25coder7b/Qwen--Qwen2.5-Coder-7B/greedy_bigcode_t1.0.jsonl",
    "results/mbpp_qwen25coder7b_2026_11/gpu2_logs/q1_driver.log":
        "results/regen_2026_11/mbpp/unseeded_qwen25coder7b/gpu2_logs/q1_driver.log",
    "results/mbpp_qwen25coder7b_2026_11/analysis/qwen25coder7b_tables.md":
        "results/regen_2026_11/mbpp/unseeded_qwen25coder7b/analysis/qwen25coder7b_tables.md",
    f"results/mbpp_qwen25coder7binstruct_2026_11/{I7}/greedy_t1.0.jsonl":
        f"results/regen_2026_11/mbpp/unseeded_qwen25coder7b/{I7}/greedy_t1.0.jsonl",
}

# Roots that must be left alone (smoke / diag / fit / BOS-check)
UNTOUCHED = [
    f"results/humaneval_regen_2026_11_smoke/{M}/greedy.jsonl",
    f"results/humaneval_regen_2026_11_diag/{M}/greedy_hf.jsonl",
    f"results/humaneval_regen_2026_11_stripped_fit/{M}/greedy__prompt_stripped.jsonl",
    f"results/humaneval_regen_2026_11_instruct7b_smoke/{I7}/greedy.jsonl",
    "results/bos_check_2026_11/x.jsonl",
]


def _jsonl(path: Path) -> str:
    return '{"task_id": "T/0", "prompt_variant": "stripped", "prompt_style": "base", "seed": 42, "backend": "vllm"}\n'


def make_tree(root: Path, files=None) -> Path:
    for rel in list(EXPECTED if files is None else files) + UNTOUCHED:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_jsonl(p) if p.suffix == ".jsonl" else f"content of {rel}\n")
    return root


def snapshot(root: Path) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            out[str(Path(dirpath, d).relative_to(root)) + "/"] = None
        for f in filenames:
            p = Path(dirpath, f)
            st = p.stat()
            out[str(p.relative_to(root))] = (hashlib.sha256(p.read_bytes()).hexdigest(), st.st_mtime_ns, st.st_ino)
    return out


# ---------------------------------------------------------------- planning

def test_plan_maps_every_source_file_to_expected_target(tmp_path):
    make_tree(tmp_path)
    plan = cr.build_plan(tmp_path)
    assert plan.errors == []
    got = {e.old: e.new for e in plan.entries}
    assert got == EXPECTED


def test_plan_is_complete_every_file_under_sources_mapped_once(tmp_path):
    make_tree(tmp_path)
    plan = cr.build_plan(tmp_path)
    on_disk = set()
    for src in cr.SOURCES:
        for dirpath, _, filenames in os.walk(tmp_path / src.old_root):
            on_disk |= {str(Path(dirpath, f).relative_to(tmp_path)) for f in filenames}
    olds = [e.old for e in plan.entries]
    assert sorted(olds) == sorted(on_disk) and len(olds) == len(set(olds))
    assert not any(u in olds for u in UNTOUCHED)


def test_plan_records_sha256_and_size(tmp_path):
    make_tree(tmp_path)
    plan = cr.build_plan(tmp_path)
    for e in plan.entries:
        data = (tmp_path / e.old).read_bytes()
        assert e.sha256 == hashlib.sha256(data).hexdigest() and e.size == len(data)


def test_manifest_and_readme_get_source_tag(tmp_path):
    files = dict(EXPECTED)
    files["results/humaneval_regen_2026_11_stripped/MANIFEST.json"] = None
    files["results/humaneval_regen_2026_11_stripped/README.md"] = None
    make_tree(tmp_path, files)
    got = {e.old: e.new for e in cr.build_plan(tmp_path).entries}
    assert got["results/humaneval_regen_2026_11_stripped/MANIFEST.json"] == \
        "results/regen_2026_11/humaneval/main/MANIFEST.stripped.json"
    assert got["results/humaneval_regen_2026_11_stripped/README.md"] == \
        "results/regen_2026_11/humaneval/main/README.stripped.md"


def test_collision_between_merged_sources_is_refused(tmp_path):
    files = dict(EXPECTED)
    # same model file in the main and the extra root -> both map to humaneval/main/<model>/greedy__prompt_stripped.jsonl
    files[f"results/humaneval_regen_2026_11_stripped_extra/{M}/greedy__prompt_stripped.jsonl"] = None
    make_tree(tmp_path, files)
    plan = cr.build_plan(tmp_path)
    coll = [e for e in plan.errors if e.startswith("collision")]
    assert len(coll) == 1 and f"{M}/greedy__prompt_stripped.jsonl" in coll[0]
    assert "humaneval_regen_2026_11_stripped/" in coll[0] and "humaneval_regen_2026_11_stripped_extra/" in coll[0]


def test_collision_unseeded_merge_is_refused(tmp_path):
    files = dict(EXPECTED)
    files["results/mbpp_qwen25coder7binstruct_2026_11/gpu2_logs/q1_driver.log"] = None
    make_tree(tmp_path, files)
    assert any(e.startswith("collision") and "q1_driver.log" in e for e in cr.build_plan(tmp_path).errors)


def test_missing_source_root_is_refused(tmp_path):
    files = {k: v for k, v in EXPECTED.items() if not k.startswith("results/humaneval_regen_2026_11_instruct7b_extra/")}
    make_tree(tmp_path, files)
    errs = cr.build_plan(tmp_path).errors
    assert any(e.startswith("missing source") and "humaneval_regen_2026_11_instruct7b_extra" in e for e in errs)


def test_unmapped_top_level_entry_is_refused(tmp_path):
    files = dict(EXPECTED)
    files["results/humaneval_regen_2026_11_stripped/notes.txt"] = None
    files["results/mbpp_seeded_2026_11/stray_dir/x.log"] = None
    make_tree(tmp_path, files)
    errs = cr.build_plan(tmp_path).errors
    assert any(e.startswith("unmapped") and "humaneval_regen_2026_11_stripped/notes.txt" in e for e in errs)
    assert any(e.startswith("unmapped") and "mbpp_seeded_2026_11/stray_dir" in e for e in errs)


def test_existing_destination_is_refused(tmp_path):
    make_tree(tmp_path)
    dest = tmp_path / EXPECTED["results/mbpp_seeded_2026_11/analysis/stats.json"]
    dest.parent.mkdir(parents=True)
    dest.write_text("already here\n")
    errs = cr.build_plan(tmp_path).errors
    assert any(e.startswith("destination exists") and "analysis/stats.json" in e for e in errs)


def test_existing_mapping_tsv_is_refused(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "results/regen_2026_11").mkdir(parents=True)
    (tmp_path / cr.MAPPING_TSV).write_text("old_path\tnew_path\n")
    assert any(e.startswith("destination exists") and "path_mapping.tsv" in e for e in cr.build_plan(tmp_path).errors)


def test_symlink_in_source_is_refused(tmp_path):
    make_tree(tmp_path)
    (tmp_path / f"results/mbpp_seeded_2026_11/{M}/link.jsonl").symlink_to(
        tmp_path / f"results/mbpp_seeded_2026_11/{M}/greedy_bigcode_t1.0.jsonl")
    assert any(e.startswith("symlink") for e in cr.build_plan(tmp_path).errors)


def test_lock_files_are_refused_as_unmapped_state(tmp_path):
    files = dict(EXPECTED)
    files["results/humaneval_regen_2026_11_stripped/MANIFEST.json.lock"] = None
    make_tree(tmp_path, files)
    assert any("MANIFEST.json.lock" in e for e in cr.build_plan(tmp_path).errors)


def test_mapping_tsv_has_header_and_one_row_per_file(tmp_path):
    make_tree(tmp_path)
    plan = cr.build_plan(tmp_path)
    text = cr.render_mapping_tsv(plan)
    lines = text.splitlines()
    assert lines[0] == "old_path\tnew_path\tsha256\tbytes"
    rows = [ln.split("\t") for ln in lines[1:]]
    assert {(r[0], r[1]) for r in rows} == set(EXPECTED.items()) and len(rows) == len(EXPECTED)


def test_readme_lists_root_mapping_and_models(tmp_path):
    make_tree(tmp_path)
    text = cr.render_readme(cr.build_plan(tmp_path), provenance="PROV")
    for src in cr.SOURCES:
        assert src.old_root in text and src.new_root in text
    assert M in text and I7 in text and "PROV" in text and "path_mapping.tsv" in text
    assert "MANIFEST.stripped.gpu2.json" in text   # manifests are indexed under their new names


# ---------------------------------------------------------------- CLI: dry run

def test_dry_run_is_default_and_touches_nothing(tmp_path, capsys):
    make_tree(tmp_path)
    before = snapshot(tmp_path)
    rc = cr.main(["--repo-root", str(tmp_path)])
    assert rc == 0
    assert snapshot(tmp_path) == before
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "results/regen_2026_11/humaneval/main" in out


def test_dry_run_plan_dir_writes_preview_only_there(tmp_path):
    repo = make_tree(tmp_path / "repo")
    before = snapshot(repo)
    plan_dir = tmp_path / "preview"
    assert cr.main(["--repo-root", str(repo), "--plan-dir", str(plan_dir)]) == 0
    assert snapshot(repo) == before
    assert (plan_dir / "path_mapping.tsv").is_file() and (plan_dir / "README.md").is_file()


def test_plan_dir_inside_repo_results_is_refused(tmp_path):
    repo = make_tree(tmp_path / "repo")
    with pytest.raises(SystemExit):
        cr.main(["--repo-root", str(repo), "--plan-dir", str(repo / "results" / "regen_2026_11")])
    assert not (repo / "results" / "regen_2026_11").exists()


def test_dry_run_with_errors_returns_nonzero_and_touches_nothing(tmp_path):
    files = dict(EXPECTED)
    files[f"results/humaneval_regen_2026_11_stripped_extra/{M}/greedy__prompt_stripped.jsonl"] = None
    make_tree(tmp_path, files)
    before = snapshot(tmp_path)
    assert cr.main(["--repo-root", str(tmp_path)]) == 2
    assert snapshot(tmp_path) == before


# ---------------------------------------------------------------- CLI: execute

def test_execute_refuses_on_collision_and_moves_nothing(tmp_path):
    files = dict(EXPECTED)
    files[f"results/humaneval_regen_2026_11_stripped_extra/{M}/greedy__prompt_stripped.jsonl"] = None
    make_tree(tmp_path, files)
    before = snapshot(tmp_path)
    assert cr.main(["--repo-root", str(tmp_path), "--execute"]) == 2
    assert snapshot(tmp_path) == before


def test_execute_moves_by_rename_and_verifies(tmp_path):
    make_tree(tmp_path)
    before = snapshot(tmp_path)
    assert cr.main(["--repo-root", str(tmp_path), "--execute"]) == 0
    after = snapshot(tmp_path)
    for old, new in EXPECTED.items():
        assert old not in after
        sha, _, ino = before[old]
        assert after[new][0] == sha          # same content
        assert after[new][2] == ino          # same inode: renamed, not copied
    for u in UNTOUCHED:
        assert after[u] == before[u]
    for src in cr.SOURCES:                    # old roots removed (were empty after the move)
        assert not (tmp_path / src.old_root).exists()
    tsv = (tmp_path / cr.MAPPING_TSV).read_text().splitlines()
    assert len(tsv) == 1 + len(EXPECTED)
    assert (tmp_path / cr.README).is_file()


def test_execute_never_copies(tmp_path, monkeypatch):
    import shutil
    make_tree(tmp_path)
    for name in ("copy", "copy2", "copyfile", "copytree", "move"):
        monkeypatch.setattr(shutil, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("copy used")))
    assert cr.main(["--repo-root", str(tmp_path), "--execute"]) == 0


def test_execute_refuses_cross_device(tmp_path, monkeypatch):
    make_tree(tmp_path)
    before = snapshot(tmp_path)
    monkeypatch.setattr(cr, "_same_device", lambda a, b: False)
    assert cr.main(["--repo-root", str(tmp_path), "--execute"]) == 2
    assert snapshot(tmp_path) == before


def test_verify_detects_changed_content(tmp_path):
    make_tree(tmp_path)
    plan = cr.build_plan(tmp_path)
    e = plan.entries[0]
    dest = tmp_path / e.new
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.rename(tmp_path / e.old, dest)
    dest.write_text("tampered\n")
    with pytest.raises(cr.ConsolidationError, match="sha256"):
        cr.verify_moved(tmp_path, [e])


def test_leftover_files_keep_old_root(tmp_path):
    """A file created in a source root after planning is not moved, and its directory is not removed."""
    make_tree(tmp_path)
    plan = cr.build_plan(tmp_path)
    late = tmp_path / "results/mbpp_seeded_2026_11/analysis/late.json"
    late.write_text("{}\n")
    cr.execute_plan(tmp_path, plan, provenance="PROV")
    assert late.is_file()
    assert not (tmp_path / "results/humaneval_regen_2026_11_stripped").exists()


def test_real_sources_configured_match_agreed_layout():
    roots = {s.old_root: s.new_root for s in cr.SOURCES}
    assert roots == {
        "results/humaneval_regen_2026_11_stripped": "results/regen_2026_11/humaneval/main",
        "results/humaneval_regen_2026_11_stripped_extra": "results/regen_2026_11/humaneval/main",
        "results/humaneval_regen_2026_11_instruct7b": "results/regen_2026_11/humaneval/main",
        "results/humaneval_regen_2026_11_instruct7b_extra": "results/regen_2026_11/humaneval/main",
        "results/humaneval_regen_2026_11": "results/regen_2026_11/humaneval/superseded_unstripped",
        "results/mbpp_seeded_2026_11": "results/regen_2026_11/mbpp/main",
        "results/mbpp_qwen25coder7b_2026_11": "results/regen_2026_11/mbpp/unseeded_qwen25coder7b",
        "results/mbpp_qwen25coder7binstruct_2026_11": "results/regen_2026_11/mbpp/unseeded_qwen25coder7b",
    }


# ---------------------------------------------------------------- README notes (2026-11-10 follow-up)

def _readme_with(tmp_path, extra_files: dict):
    """make_tree + overwrite/add files with given content; README rendered from the OLD paths (pre-move)."""
    make_tree(tmp_path)
    for rel, content in extra_files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return cr.render_readme(cr.build_plan(tmp_path), provenance="PROV", repo_root=tmp_path)


def _table_rows(text, heading):
    block = text.split(f"### `{heading}/`", 1)[1].split("\n\n", 2)[1]
    return [ln for ln in block.splitlines() if ln.startswith("| ")]


def test_mbpp_protocol_table_has_single_bos_from_records(tmp_path):
    chat = f"results/mbpp_seeded_2026_11/{I7}/greedy_t1.0.jsonl"
    text = _readme_with(tmp_path, {
        chat: '{"task_id": 1, "backend": "hf", "seed": 42, "single_bos": true}\n',
        f"results/mbpp_seeded_2026_11/{M}/greedy_bigcode_t1.0.jsonl": '{"task_id": 1, "backend": "hf", "seed": 42}\n'})
    rows = _table_rows(text, "results/regen_2026_11/mbpp/main")
    assert rows[0].split(" | ")[-2] == "single_bos"
    by_model = {r.split(" | ")[0].lstrip("| "): r.split(" | ") for r in rows[1:]}   # rows[0] = header
    assert by_model[I7][-2] == "True" and by_model[M][-2] == "-"
    he = _table_rows(text, "results/regen_2026_11/humaneval/main")
    assert "single_bos" not in he[0]   # HumanEval tables unchanged


def test_readme_notes_prompt_inference_first_record_and_logs(tmp_path):
    text = _readme_with(tmp_path, {})
    assert "inferred from the file stems" in text and "prompt_text" in text
    assert "first record" in text
    assert "`mbpp/main/logs/`" in text and "`mbpp/unseeded_qwen25coder7b/gpu2_logs/`" in text
    assert "not renamed" in text


def test_readme_lists_archived_analysis_files_hardcoding_old_roots(tmp_path):
    text = _readme_with(tmp_path, {
        "results/mbpp_seeded_2026_11/analysis/validator_scripts/stats.py": 'ROOT = "results/mbpp_seeded_2026_11"\n',
        "results/mbpp_seeded_2026_11/analysis/integrity.json": '{"root": "mbpp_seeded_2026_11/x"}\n',
        "results/mbpp_seeded_2026_11/analysis/clean.py": 'ROOT = "results/regen_2026_11/mbpp/main"\n',
        "results/mbpp_seeded_2026_11/analysis/notes.md": "results/mbpp_seeded_2026_11\n",          # .md: not a script
        "results/humaneval_regen_2026_11_stripped/analysis/grading_scripts/check.py":
            'R = "results/humaneval_regen_2026_11_smoke"\n'})   # a non-moved root: not counted
    assert "path_mapping.tsv" in text and "do not rerun" in text.lower()
    # 3 = the two above + the fixture's analysis/stats.json, whose default content ("content of <old path>") names it
    assert "`results/regen_2026_11/mbpp/main/analysis/`: 3 files (`integrity.json`, `stats.json`, " \
           "`validator_scripts/stats.py`)" in text
    assert "clean.py" not in text and "notes.md" not in text
    assert "humaneval/main/analysis/main/grading_scripts/check.py" not in text
