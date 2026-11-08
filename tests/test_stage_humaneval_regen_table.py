"""Tests for scripts/paper/stage_humaneval_regen_table.py (HumanEval Table II, STAGING only)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "paper" / "stage_humaneval_regen_table.py"
_spec = importlib.util.spec_from_file_location("stage_humaneval_regen_table", _SCRIPT)
st = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = st  # dataclasses need the module registered
_spec.loader.exec_module(st)

REAL_ROOT = REPO_ROOT / "results" / "humaneval_regen_2026_11_stripped"
MBPP_TABLE_I = REPO_ROOT.parent / "pless-for-coding-paper" / "tables" / "table_3a_mbpp.tex"  # read only

EXPECTED_DISPLAY = [
    "CodeLlama-7b-hf", "CodeLlama-7b-Instruct-hf", "OpenCodeInterpreter-DS-1.3B", "Llama-2-7b-hf",
    "Llama-2-7b-chat-hf", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-3B", "Qwen2.5-Coder-7B",
]
EXPECTED_LABELS = [
    "greedy", "beam@4", "T@0.7", "top-p@0.9 with T=1.0", "p-less with T=0.7", "p-less-norm with T=0.7",
]


def _mbpp_table_i() -> tuple[list[str], dict[str, list[str]]]:
    """(model group names in order, model -> decoder labels) parsed from MBPP Table I."""
    import re
    if not MBPP_TABLE_I.is_file():
        pytest.skip(f"MBPP Table I not available: {MBPP_TABLE_I}")
    models: list[str] = []
    labels: dict[str, list[str]] = {}
    for line in MBPP_TABLE_I.read_text().splitlines():
        m = re.search(r"\\textit\{(.+?)\}\}", line)
        if m:
            models.append(m.group(1))
            labels[m.group(1)] = []
        elif models and "&" in line and not line.startswith("Decoder"):
            labels[models[-1]].append(line.strip().split(" & ")[0])
    return models, labels

# config key -> (method, temperature, top_p, n)
_META = {
    "greedy": ("greedy", 0.0, 1.0, 1),
    "beam4": ("beam4", 1.0, 1.0, 1),
    "temp_t0.7": ("temp", 0.7, 1.0, 10),
    "top_p0.9_t1.0": ("top_p", 1.0, 0.9, 10),
    "top_p0.95_t0.2": ("top_p", 0.2, 0.95, 10),
    "pless_t0.7": ("pless", 0.7, 1.0, 10),
    "pless_norm_t0.7": ("pless_norm", 0.7, 1.0, 10),
}


def _metrics(config: str, p1: float, num_tasks: int = 164) -> dict:
    method, t, top_p, n = _META[config]
    if n == 1:
        pk = {str(k): p1 for k in (1, 3, 5, 8, 10)}
        cov = {"0.3": 100 * p1, "0.5": 100 * p1, "0.7": 100 * p1}
        cb = 0.0
    else:
        pk = {"1": p1, "3": p1 + 0.05, "5": p1 + 0.07, "8": p1 + 0.08, "10": p1 + 0.09}
        cov = {"0.3": 100 * p1 + 3, "0.5": 100 * p1, "0.7": 100 * p1 - 3}
        cb = 0.1234
    return {
        "method": method, "temperature": t, "top_p": top_p, "dataset": "humaneval",
        "num_tasks": num_tasks, "num_samples_per_task": n, "pass_at_k": pk,
        "cover_at_t": cov, "cover_at_t_distinct": cov, "codebleu_diversity": cb,
        "structural_diversity": None,
    }


def _make_fake_root(tmp_path: Path, p_less_equal_norm: bool = True) -> Path:
    root = tmp_path / "fake_root"
    for i, m in enumerate(st.MODELS):
        mdir = root / m.model_dir / "metrics"
        mdir.mkdir(parents=True)
        for c in st.configs_for(m):
            p1 = 0.30 + 0.01 * i
            if c.key == "pless_norm_t0.7" and not p_less_equal_norm:
                p1 += 0.001
            (mdir / st.metrics_filename(m, c.key)).write_text(json.dumps(_metrics(c.key, p1)))
    return root


# ---------------------------------------------------------------- file resolution

def test_model_order_and_display_names():
    assert [m.display for m in st.MODELS] == EXPECTED_DISPLAY


def test_display_names_and_order_equal_mbpp_table_i():
    mbpp_models, _ = _mbpp_table_i()
    assert len(mbpp_models) == 7
    assert [m.display for m in st.MODELS][:7] == mbpp_models
    assert st.MODELS[-1].display == "Qwen2.5-Coder-7B"


def test_config_order_and_labels_equal_mbpp_table_i():
    _, mbpp_labels = _mbpp_table_i()
    for model, ls in mbpp_labels.items():
        shared = [l for l in ls if "(Hasan)" not in l]
        assert shared == EXPECTED_LABELS, model
    for m in st.MODELS:
        assert [c.label for c in st.configs_for(m)] == EXPECTED_LABELS


def test_hasan_excluded_by_default_and_reenableable(monkeypatch):
    assert st.INCLUDE_HASAN is False
    assert all("0.95" not in c.key for m in st.MODELS for c in st.configs_for(m))
    monkeypatch.setattr(st, "INCLUDE_HASAN", True)
    have = {m.display for m in st.MODELS if any(c.hasan_only for c in st.configs_for(m))}
    assert have == {"OpenCodeInterpreter-DS-1.3B", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-3B", "Qwen2.5-Coder-7B"}


def test_real_table_has_48_rows_and_no_top_p095():
    groups = st.build_groups(REAL_ROOT)
    assert [g.model.display for g in groups] == EXPECTED_DISPLAY
    assert sum(len(g.rows) for g in groups) == 48
    for g in groups:
        assert [c.label for c, _, _ in g.rows] == EXPECTED_LABELS
    for na in (True, False):
        md = st.render_md(groups, na_single_sample=na)
        tex = st.render_tex(groups, na_single_sample=na)
        assert "0.95" not in md and "0.95" not in tex
        rows = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| Decoder")
                and not l.startswith("| *")]
        assert len(rows) == 48
        assert sum(1 for l in tex.splitlines() if "&" in l and not l.startswith("Decoder")) == 48


def test_draft_caption_has_no_hasan_and_new_prompting_sentence(tmp_path):
    assert "Hasan" not in st.DRAFT_CAPTION and "0.95" not in st.DRAFT_CAPTION and "cite" not in st.DRAFT_CAPTION
    assert ("CodeLlama-7b-Instruct-hf and Llama-2-7b-chat-hf use their chat template; all other models "
            "receive the raw (stripped) HumanEval prompt") in st.DRAFT_CAPTION
    assert st.DRAFT_CAPTION.startswith(r"HumanEval (164 problems). Sampling decoders draw $n{=}10$ samples "
                                       r"per problem; greedy and beam@4 draw one.")
    tex = st.render_tex(st.build_groups(_make_fake_root(tmp_path)), na_single_sample=True)
    assert "Hasan" not in tex


def test_filename_pattern_stripped_vs_chat():
    by = {m.display: m for m in st.MODELS}
    assert st.metrics_filename(by["Qwen2.5-Coder-7B"], "greedy") == "greedy__prompt_stripped_metrics.json"
    assert st.metrics_filename(by["Llama-2-7b-hf"], "pless_t0.7") == "pless_t0.7__prompt_stripped_metrics.json"
    assert st.metrics_filename(by["CodeLlama-7b-Instruct-hf"], "greedy") == "greedy_metrics.json"
    assert st.metrics_filename(by["Llama-2-7b-chat-hf"], "beam4") == "beam4_metrics.json"
    for name in ("OpenCodeInterpreter-DS-1.3B", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-3B", "Qwen2.5-Coder-7B",
                 "CodeLlama-7b-hf", "Llama-2-7b-hf"):
        assert by[name].prompt == "stripped"
    for name in ("CodeLlama-7b-Instruct-hf", "Llama-2-7b-chat-hf"):
        assert by[name].prompt == "chat"


def test_real_paths_resolve_to_expected_files():
    paths = {(g.model.display, c.key): p for g in st.build_groups(REAL_ROOT) for c, p, _ in g.rows}
    assert paths[("Qwen2.5-Coder-1.5B", "temp_t0.7")] == \
        REAL_ROOT / "Qwen--Qwen2.5-Coder-1.5B/metrics/temp_t0.7__prompt_stripped_metrics.json"
    assert paths[("Llama-2-7b-chat-hf", "pless_norm_t0.7")] == \
        REAL_ROOT / "meta-llama--Llama-2-7b-chat-hf/metrics/pless_norm_t0.7_metrics.json"
    assert paths[("CodeLlama-7b-hf", "beam4")] == \
        REAL_ROOT / "codellama--CodeLlama-7b-hf/metrics/beam4__prompt_stripped_metrics.json"


# ---------------------------------------------------------------- fail loudly

def test_missing_file_raises(tmp_path):
    root = _make_fake_root(tmp_path)
    m = next(m for m in st.MODELS if m.display == "CodeLlama-7b-hf")
    (root / m.model_dir / "metrics" / st.metrics_filename(m, "temp_t0.7")).unlink()
    with pytest.raises(FileNotFoundError, match="temp_t0.7"):
        st.build_groups(root)


def test_wrong_num_tasks_raises(tmp_path):
    root = _make_fake_root(tmp_path)
    m = next(m for m in st.MODELS if m.display == "CodeLlama-7b-Instruct-hf")
    f = root / m.model_dir / "metrics" / st.metrics_filename(m, "greedy")
    f.write_text(json.dumps(_metrics("greedy", 0.4, num_tasks=163)))
    with pytest.raises(ValueError, match="num_tasks"):
        st.build_groups(root)


def test_config_metadata_mismatch_raises(tmp_path):
    root = _make_fake_root(tmp_path)
    m = st.MODELS[0]
    f = root / m.model_dir / "metrics" / st.metrics_filename(m, "pless_norm_t0.7")
    f.write_text(json.dumps(_metrics("pless_t0.7", 0.4)))  # a p-less file under the p-less-norm name
    with pytest.raises(ValueError, match="method"):
        st.build_groups(root)


# ---------------------------------------------------------------- rows

def _labels_by_model(md: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in md.splitlines():
        if not line.startswith("| ") or line.startswith("| Decoder"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0].startswith("*") and cells[0].endswith("*"):
            cur = cells[0].strip("*")
            out[cur] = []
        elif cur is not None:
            out[cur].append(cells[0])
    return out


def test_pless_norm_has_own_row_even_when_equal(tmp_path):
    root = _make_fake_root(tmp_path, p_less_equal_norm=True)
    md = st.render_md(st.build_groups(root), na_single_sample=True)
    labels = _labels_by_model(md)
    for m in st.MODELS:
        assert labels[m.display].count("p-less with T=0.7") == 1
        assert labels[m.display].count("p-less-norm with T=0.7") == 1


def test_no_hasan_rows_by_default(tmp_path):
    root = _make_fake_root(tmp_path)
    md = st.render_md(st.build_groups(root), na_single_sample=True)
    labels = _labels_by_model(md)
    assert list(labels) == EXPECTED_DISPLAY
    for m in EXPECTED_DISPLAY:
        assert labels[m] == EXPECTED_LABELS
    assert "top-p@0.95" not in md


def test_hasan_rows_only_for_four_models_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "INCLUDE_HASAN", True)
    root = _make_fake_root(tmp_path)
    md = st.render_md(st.build_groups(root), na_single_sample=True)
    labels = _labels_by_model(md)
    have = {m for m, ls in labels.items() if "top-p@0.95 with T=0.2" in ls}
    assert have == {"OpenCodeInterpreter-DS-1.3B", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-3B", "Qwen2.5-Coder-7B"}
    assert labels["Llama-2-7b-chat-hf"] == EXPECTED_LABELS


def test_hasan_config_missing_is_not_required(tmp_path):
    # with Hasan excluded no model looks for the top_p0.95 file
    root = _make_fake_root(tmp_path)
    for m in st.MODELS:
        assert not (root / m.model_dir / "metrics" / st.metrics_filename(m, "top_p0.95_t0.2")).exists()
    st.build_groups(root)  # no raise


# ---------------------------------------------------------------- n=1 rendering

def _row(md: str, model: str, label: str) -> list[str]:
    cur = None
    for line in md.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == f"*{model}*":
            cur = model
        elif cur == model and cells[0] == label:
            return cells[1:]
    raise AssertionError(f"row {model}/{label} not found")


def test_n1_dash_rendering_md_and_tex(tmp_path):
    groups = st.build_groups(_make_fake_root(tmp_path))
    md = st.render_md(groups, na_single_sample=True)
    for label in ("greedy", "beam@4"):
        cells = _row(md, "Qwen2.5-Coder-3B", label)
        assert cells[0].endswith("%") and cells[0] != "—"
        assert cells[1:] == ["—"] * 6
    assert "single sample per problem; not applicable" in md
    # sampled rows are untouched
    assert "—" not in _row(md, "Qwen2.5-Coder-3B", "T@0.7")
    tex = st.render_tex(groups, na_single_sample=True)
    greedy_line = next(l for l in tex.splitlines() if l.strip().startswith("greedy &"))
    assert greedy_line.count("---") == 6
    assert "single sample per problem; not applicable" in tex


def test_n1_nodash_rendering_matches_headline_table(tmp_path):
    groups = st.build_groups(_make_fake_root(tmp_path))
    md = st.render_md(groups, na_single_sample=False)
    cells = _row(md, "Qwen2.5-Coder-3B", "greedy")
    assert "—" not in cells
    assert cells[0] == cells[1] == cells[2] == cells[3]  # pass@k copies pass@1
    assert cells[6] == "0.0000"
    assert "not applicable" not in md
    tex = st.render_tex(groups, na_single_sample=False)
    assert "---" not in "\n".join(l for l in tex.splitlines() if "&" in l)


# ---------------------------------------------------------------- tex style

def test_tex_style(tmp_path):
    tex = st.render_tex(st.build_groups(_make_fake_root(tmp_path)), na_single_sample=True)
    assert r"\label{tab:3b_humaneval}" in tex
    # n=10 applies to sampling decoders only; greedy and beam produce one sample per problem (review 2026-11-08).
    assert (r"\caption{Performance of eight code models on HumanEval (164 problems; $n{=}10$ samples for "
            r"sampling decoders, one for greedy and beam)}") in tex
    assert "n{=}10$)}" not in tex
    assert "% DRAFT CAPTION — user to review" in tex
    assert r"\resizebox{\columnwidth}{!}{%" in tex
    assert r"\begin{tabular}{lrrrrrrr}" in tex
    assert r"Decoder & pass@1 & pass@3 & pass@5 & pass@10 & cov@0.3 & cov@0.7 & cb\_div \\" in tex
    assert r"\multicolumn{8}{l}{\hspace{-1.5mm}\textit{Llama-2-7b-chat-hf}} \\" in tex
    assert r"\multicolumn{8}{l}{\hspace{-1.5mm}\textit{CodeLlama-7b-hf}} \\" in tex
    assert r"\textbf" not in tex and "\\cellcolor" not in tex
    assert tex.count(r"\midrule") == 1 + len(st.MODELS) - 1


# ---------------------------------------------------------------- output guard

def test_output_guard_refuses_outside_staging(tmp_path):
    with pytest.raises(ValueError, match="staging"):
        st.guard_output_path(REPO_ROOT / "paper" / "llm4code" / "tables" / "table_ii_staging.md")
    with pytest.raises(ValueError, match="staging"):
        st.guard_output_path(st.STAGING_DIR / ".." / "table_ii_staging.md")
    with pytest.raises(ValueError, match="staging"):
        st.guard_output_path(REPO_ROOT.parent / "pless-for-coding-paper" / "tables" / "x.tex")
    with pytest.raises(ValueError):
        st.guard_output_path(st.STAGING_DIR / "some_other_name.md")  # only the 4 allowed names
    assert st.guard_output_path(st.STAGING_DIR / "table_ii_staging.tex") == \
        (st.STAGING_DIR / "table_ii_staging.tex").resolve()


def test_write_outputs_refuses_bad_out_dir(tmp_path):
    groups = st.build_groups(_make_fake_root(tmp_path))
    with pytest.raises(ValueError, match="staging"):
        st.write_outputs(groups, out_dir=tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()


def test_write_outputs_into_allowed_dir(tmp_path):
    groups = st.build_groups(_make_fake_root(tmp_path))
    out = tmp_path / "staging"
    out.mkdir()
    written = st.write_outputs(groups, out_dir=out, staging_dir=out, provenance="test")
    assert sorted(p.name for p in written) == sorted(st.OUTPUT_NAMES)


# ---------------------------------------------------------------- real-data spot-check

def test_md_numbers_equal_metrics_json():
    md = st.render_md(st.build_groups(REAL_ROOT), na_single_sample=True)
    cols = st.PAPER_COLUMNS

    def j(model_dir, fname):
        return json.loads((REAL_ROOT / model_dir / "metrics" / fname).read_text())

    d = j("Qwen--Qwen2.5-Coder-7B", "pless_t0.7__prompt_stripped_metrics.json")
    assert _row(md, "Qwen2.5-Coder-7B", "p-less with T=0.7")[cols.index("pass@1")] == \
        f"{100 * d['pass_at_k']['1']:.2f}%"
    d = j("meta-llama--Llama-2-7b-chat-hf", "pless_norm_t0.7_metrics.json")
    assert _row(md, "Llama-2-7b-chat-hf", "p-less-norm with T=0.7")[cols.index("cov@0.7")] == \
        f"{d['cover_at_t']['0.7']:.1f}%"
    d = j("codellama--CodeLlama-7b-hf", "temp_t0.7__prompt_stripped_metrics.json")
    assert _row(md, "CodeLlama-7b-hf", "T@0.7")[cols.index("cb_div")] == f"{d['codebleu_diversity']:.4f}"
    d = j("m-a-p--OpenCodeInterpreter-DS-1.3B", "beam4__prompt_stripped_metrics.json")
    assert _row(md, "OpenCodeInterpreter-DS-1.3B", "beam@4")[cols.index("pass@1")] == \
        f"{100 * d['pass_at_k']['1']:.2f}%"
