"""Tests for ``--humaneval-table {regen,legacy}`` and the output-dir overrides of
scripts/paper/make_llm4code_tables.py.

Every build writes to pytest temp dirs via ``--out-dir`` / ``--paper-out-dir``. The real
output dirs (both repos' tables/ and figures/) are only READ; a module-scoped autouse guard
asserts their checksums are unchanged after the tests (and, when present, that they still
match the protection snapshot /tmp/paperdirs_snapshot.sha).

The "legacy" golden is produced at test time by the frozen PRE-CHANGE script
(tests/fixtures/make_llm4code_tables_prechange.py, sha256 pinned below), loaded from a temp
copy and pointed at temp dirs. That pins the legacy logic without freezing the data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "paper" / "make_llm4code_tables.py"
PRECHANGE = REPO_ROOT / "tests" / "fixtures" / "make_llm4code_tables_prechange.py"
PRECHANGE_SHA256 = "50ca4514059cd500ade1fa79bb8b51e00878518bfa6e4168b1fafc008e8db8c9"
PAPER_REPO = REPO_ROOT.parent / "pless-for-coding-paper"
REAL_TABLES_DIR = REPO_ROOT / "paper" / "llm4code" / "tables"
REAL_PAPER_DIR = PAPER_REPO / "tables"
PROTECTED_DIRS = (REAL_TABLES_DIR, REPO_ROOT / "paper" / "llm4code" / "figures", REAL_PAPER_DIR, PAPER_REPO / "figures")
SNAPSHOT_FILE = Path("/tmp/paperdirs_snapshot.sha")  # paths relative to REPO_ROOT.parent

SLUG_3B = "table_3b_humaneval"
HE_FILES = (f"{SLUG_3B}.md", f"{SLUG_3B}.tex")
# Tables whose regen output must equal the paper repo's current file byte for byte.
PAPER_IDENTICAL_TEX = ("table_3a_mbpp.tex", "table_3b_humaneval.tex", "table_a1_mbpp_appendix.tex",
                       "table_a2_humaneval_appendix.tex")
CAPTION_CHANGED_TEX = ("table_3a_mbpp.tex", "table_a1_mbpp_appendix.tex", "table_a2_humaneval_appendix.tex")
OTHER_SLUGS = ("table_3c_small_models_2507", "table_5a_apps_renyi", "table_a3_cf748_qwen")


def _load(name: str, path: Path):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


mlt = _load("make_llm4code_tables_under_test", SCRIPT)


def _sha_tree(d: Path) -> dict[str, str]:
    if not d.is_dir():
        return {}
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def _check_snapshot_file() -> None:
    if not SNAPSHOT_FILE.is_file():
        return
    for line in SNAPSHOT_FILE.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            p = REPO_ROOT.parent / rel.strip()
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f"protected file changed: {p}"


@pytest.fixture(scope="module", autouse=True)
def real_dirs_untouched():
    """Module-level guard: the real output dirs must be byte-for-byte unchanged by these tests."""
    _check_snapshot_file()
    before = [_sha_tree(d) for d in PROTECTED_DIRS]
    yield
    after = [_sha_tree(d) for d in PROTECTED_DIRS]
    assert before == after, "tests modified a real paper tables/figures dir"
    _check_snapshot_file()


def _files(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()} if d.is_dir() else {}


def _paper_file(name: str) -> bytes:
    p = REAL_PAPER_DIR / name
    if not p.is_file():
        pytest.skip(f"sibling paper repo file absent: {p}")
    return p.read_bytes()


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """(tables-dir files, paper-dir files) written by the frozen pre-change script into temp dirs."""
    assert hashlib.sha256(PRECHANGE.read_bytes()).hexdigest() == PRECHANGE_SHA256, "pre-change fixture was edited"
    root = tmp_path_factory.mktemp("golden")
    src = root / "make_llm4code_tables_prechange.py"
    shutil.copy(PRECHANGE, src)
    tdir, pdir = root / "tables", root / "paper"
    pdir.mkdir()
    pre = _load("make_llm4code_tables_prechange_golden", src)
    pre.REPO_ROOT, pre.TABLES_DIR, pre.TABLES_DIR_PAPER = REPO_ROOT, tdir, pdir
    assert pre.TABLES_DIR == tdir and pre.TABLES_DIR_PAPER == pdir
    try:
        pre.main()
    except ValueError as e:  # its final stderr summary calls TABLES_DIR.relative_to(REPO_ROOT)
        assert "subpath" in str(e) or "relative" in str(e), e
    g_t, g_p = _files(tdir), _files(pdir)
    assert len(g_t) == 14 and g_p == g_t, sorted(g_t)
    return g_t, g_p


def _build(tmp_path: Path, *args: str, seed_paper: dict[str, bytes] | None = None,
           seed_tables: dict[str, bytes] | None = None) -> tuple[Path, Path]:
    tdir, pdir = tmp_path / "tables", tmp_path / "paper"
    tdir.mkdir()
    pdir.mkdir()
    for d, seed in ((pdir, seed_paper), (tdir, seed_tables)):
        for name, data in (seed or {}).items():
            (d / name).write_bytes(data)
    mlt.main(["--out-dir", str(tdir), "--paper-out-dir", str(pdir), *args])
    return tdir, pdir


def _slug_files(files: dict[str, bytes], slugs) -> dict[str, bytes]:
    return {k: v for k, v in files.items() if k.rsplit(".", 1)[0] in slugs}


SENTINEL = {f"{SLUG_3B}.md": b"SENTINEL md\n", f"{SLUG_3B}.tex": b"SENTINEL tex\n"}


# ---------------------------------------------------------------- CLI surface

def test_default_mode_is_regen_and_default_dirs_are_the_real_ones():
    args = mlt.parse_args([])
    assert args.humaneval_table == "regen"
    assert args.allow_legacy_paper_overwrite is False
    assert args.out_dir == mlt.TABLES_DIR == REAL_TABLES_DIR
    assert args.paper_out_dir == mlt.TABLES_DIR_PAPER == REAL_PAPER_DIR


def test_overwrite_flag_rejected_outside_legacy_mode():
    with pytest.raises(SystemExit):
        mlt.parse_args(["--allow-legacy-paper-overwrite"])


def test_main_writes_nothing_outside_the_given_dirs(tmp_path):
    tdir, pdir = _build(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["paper", "tables"]
    assert len(_files(tdir)) == 14 and not (tdir / "legacy").exists()


# ---------------------------------------------------------------- regen mode (default)

def test_regen_tex_byte_identical_to_paper_repo(tmp_path):
    tdir, pdir = _build(tmp_path)
    for name in PAPER_IDENTICAL_TEX:
        want = _paper_file(name)
        assert (tdir / name).read_bytes() == want, f"out-dir {name} differs from the paper repo"
        assert (pdir / name).read_bytes() == want, f"paper-out {name} differs from the paper repo"
    assert (tdir / f"{SLUG_3B}.md").read_bytes() == _paper_file(f"{SLUG_3B}.md")


def test_regen_caption_tables_change_only_caption_and_hspace_lines_vs_prechange(tmp_path, golden):
    """3a / a1 / a2: data rows untouched; only caption (+ commented old caption) and group-header lines differ."""
    tdir, _ = _build(tmp_path)
    for name in CAPTION_CHANGED_TEX:
        new, old = (tdir / name).read_text().splitlines(), golden[0][name].decode().splitlines()
        rows = lambda ls: [l for l in ls if "&" in l]
        assert rows(new) == rows(old), name
        changed = {l for l in set(new) ^ set(old)}
        assert all(("caption" in l) or (r"\textit{" in l) for l in changed), (name, changed)


def test_regen_md_files_other_than_3b_equal_prechange(tmp_path, golden):
    tdir, _ = _build(tmp_path)
    md = {k: v for k, v in _files(tdir).items() if k.endswith(".md") and not k.startswith(SLUG_3B)}
    assert md == {k: v for k, v in golden[0].items() if k.endswith(".md") and not k.startswith(SLUG_3B)}


def test_regen_other_tables_identical_to_prechange(tmp_path, golden):
    tdir, _ = _build(tmp_path)
    assert _slug_files(_files(tdir), OTHER_SLUGS) == _slug_files(golden[0], OTHER_SLUGS)


def test_regen_3b_uses_staging_builder(tmp_path):
    st = mlt._load_stage_module()
    tdir, _ = _build(tmp_path)
    assert (tdir / f"{SLUG_3B}.tex").read_text() == st.render_paper_tex(st.build_groups(st.RESULTS_ROOT))


# ---------------------------------------------------------------- mirroring into the paper dir

def test_mirror_skips_overleaf_edited_tables_by_default(tmp_path, capsys):
    seed = {f"{s}.{e}": f"OVERLEAF {s}.{e}\n".encode() for s in mlt.OVERLEAF_EDITED for e in ("tex", "md")}
    _, pdir = _build(tmp_path, seed_paper=seed)
    assert _slug_files(_files(pdir), mlt.OVERLEAF_EDITED) == seed
    assert "table_3c_small_models_2507.tex" in _files(pdir)
    err = capsys.readouterr().err
    assert "WARNING" in err and all(s in err for s in mlt.OVERLEAF_EDITED)


def test_mirror_only_can_still_overwrite_an_overleaf_edited_table(tmp_path, golden, capsys):
    _, pdir = _build(tmp_path, "--mirror-only", "table_5a_apps_renyi", seed_paper={"table_5a_apps_renyi.tex": b"X\n"})
    assert _files(pdir) == _slug_files(golden[1], ("table_5a_apps_renyi",))
    # A deliberate overwrite of an Overleaf-edited table must be loud (review 2026-11-08).
    err = capsys.readouterr().err
    assert "WARNING: overwriting Overleaf-edited" in err and "table_5a_apps_renyi" in err


def test_mirror_never_copies_files_this_run_did_not_write(tmp_path):
    _, pdir = _build(tmp_path, seed_tables={"table_4_truncation.tex": b"from another script\n"})
    assert "table_4_truncation.tex" not in _files(pdir)


# ---------------------------------------------------------------- legacy mode

def test_legacy_with_overwrite_3b_byte_identical_to_prechange(tmp_path, golden):
    g_t, g_p = golden
    tdir, pdir = _build(tmp_path, "--humaneval-table", "legacy", "--allow-legacy-paper-overwrite",
                        seed_paper=SENTINEL, seed_tables=SENTINEL)
    assert {k: _files(tdir)[k] for k in HE_FILES} == {k: g_t[k] for k in HE_FILES}
    assert {k: _files(pdir)[k] for k in HE_FILES} == {k: g_p[k] for k in HE_FILES}
    assert _slug_files(_files(tdir), OTHER_SLUGS) == _slug_files(g_t, OTHER_SLUGS)
    assert not (tdir / "legacy").exists()


def test_legacy_without_overwrite_writes_3b_to_legacy_subdir_only(tmp_path, golden, capsys):
    g_t, _ = golden
    tdir, pdir = _build(tmp_path, "--humaneval-table", "legacy", seed_paper=SENTINEL, seed_tables=SENTINEL)
    assert {k: _files(pdir)[k] for k in HE_FILES} == SENTINEL  # paper repo 3b untouched
    assert {k: _files(tdir)[k] for k in HE_FILES} == SENTINEL  # code-repo root 3b untouched
    assert _files(tdir / "legacy") == {k: g_t[k] for k in HE_FILES}
    assert not (pdir / "legacy").exists()
    err = capsys.readouterr().err
    assert "WARNING" in err and "legacy" in err and "--allow-legacy-paper-overwrite" in err


def test_legacy_without_overwrite_ignores_mirror_only_3b(tmp_path, golden):
    _, pdir = _build(tmp_path, "--humaneval-table", "legacy", "--mirror-only", SLUG_3B, "table_3c_small_models_2507",
                     seed_paper=SENTINEL)
    assert {k: _files(pdir)[k] for k in HE_FILES} == SENTINEL
    assert _files(pdir).keys() - set(HE_FILES) == {"table_3c_small_models_2507.md", "table_3c_small_models_2507.tex"}
    assert _files(pdir)["table_3c_small_models_2507.tex"] == golden[1]["table_3c_small_models_2507.tex"]
