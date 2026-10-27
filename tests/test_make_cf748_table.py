"""Tests for scripts/paper/make_cf748_table.py (CF748 paper-style decoder table)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "make_cf748_table", REPO_ROOT / "scripts" / "paper" / "make_cf748_table.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _write_metrics(path: Path, *, num_tasks=748, cache=True, p1=0.5):
    per_task = []
    for i in range(num_tasks):
        t = {"task_id": i, "num_correct": 3}
        if cache:
            t["self_codebleu"] = 0.4
        per_task.append(t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "num_tasks": num_tasks,
        "pass_at_k": {"1": p1, "3": 0.6, "5": 0.7, "10": 0.8},
        "cover_at_t": {"0.3": 70.0, "0.7": 60.0},
        "codebleu_diversity": None,
        "per_task": per_task,
    }))


def _populate(mod, root: Path, model: str, **kw):
    for _label, stem in mod.build_rows(model):
        _write_metrics(root / f"{stem}_metrics.json", **kw)


def test_columns_are_the_paper_columns(mod):
    spec = importlib.util.spec_from_file_location(
        "make_llm4code_tables", REPO_ROOT / "scripts" / "paper" / "make_llm4code_tables.py"
    )
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    assert mod.PAPER_COLUMNS == paper.PAPER_COLUMNS


def test_rows_qwen3_order_and_membership(mod):
    rows = mod.build_rows("qwen3")
    labels = [l for l, _ in rows]
    assert labels == [
        "recommended",
        "top-p @0.7", "top-p @0.75", "top-p @0.8", "top-p @0.85", "top-p @0.9",
        "p-less baseline (k=2)",
        "G_k=0.05", "G_k=0.1", "G_k=0.2", "G_k=0.4", "G_k=0.8", "G_k=1.6",
    ]
    assert "pless_norm" not in " ".join(s for _, s in rows)
    assert dict(rows)["recommended"] == "temp_p0.95_k20_think_t0.6_t0.6"


def test_rows_deepseek_recommended_has_no_top_k(mod):
    assert dict(mod.build_rows("deepseek"))["recommended"] == "temp_p0.95_think_t0.6_t0.6"


def test_unknown_model_raises(mod):
    with pytest.raises(KeyError):
        mod.build_rows("nope")


def test_render_ok(mod, tmp_path):
    _populate(mod, tmp_path, "qwen3")
    out = mod.render_table("qwen3", tmp_path, expected_tasks=748)
    assert out.splitlines()[0].startswith("| Decoder | pass@1 | pass@3 | pass@5 | pass@10 | cov@0.3 | cov@0.7 | cb_div |")
    assert "| G_k=1.6 |" in out
    assert "50.00%" in out
    assert "0.4000" in out


def test_missing_metrics_file_fails_loudly(mod, tmp_path):
    _populate(mod, tmp_path, "qwen3")
    (tmp_path / "pless_think_t1.0_t1.0_metrics.json").unlink()
    with pytest.raises(SystemExit) as e:
        mod.render_table("qwen3", tmp_path, expected_tasks=748)
    assert "pless_think_t1.0_t1.0_metrics.json" in str(e.value)


def test_wrong_task_count_fails_loudly(mod, tmp_path):
    _populate(mod, tmp_path, "qwen3")
    _write_metrics(tmp_path / "temp_p0.9_think_t1.0_t1.0_metrics.json", num_tasks=708)
    with pytest.raises(SystemExit) as e:
        mod.render_table("qwen3", tmp_path, expected_tasks=748)
    assert "708" in str(e.value) and "748" in str(e.value)


def test_missing_codebleu_cache_fails_loudly(mod, tmp_path):
    _populate(mod, tmp_path, "qwen3")
    _write_metrics(tmp_path / "temp_p0.9_think_t1.0_t1.0_metrics.json", cache=False)
    with pytest.raises(SystemExit) as e:
        mod.render_table("qwen3", tmp_path, expected_tasks=748)
    assert "build_decoder_comparison_table.py" in str(e.value)


# ---------------------------------------------------------------- partial tables / per-model G_k grid


def test_deepseek_grid_includes_the_035_arm_in_ascending_position(mod):
    labels = [l for l, _ in mod.build_rows("deepseek")]
    gk = [l for l in labels if l.startswith("G_k=")]
    assert gk == ["G_k=0.05", "G_k=0.1", "G_k=0.2", "G_k=0.35", "G_k=0.4", "G_k=0.8", "G_k=1.6"]
    assert dict(mod.build_rows("deepseek"))["G_k=0.35"] == "pless_renyi_think_t1.0_k0.35_t1.0"


def test_qwen_grid_is_unchanged_by_the_deepseek_grid(mod):
    gk = [l for l, _ in mod.build_rows("qwen3") if l.startswith("G_k=")]
    assert gk == ["G_k=0.05", "G_k=0.1", "G_k=0.2", "G_k=0.4", "G_k=0.8", "G_k=1.6"]


def test_strict_mode_is_still_the_default(mod, tmp_path):
    _populate(mod, tmp_path, "deepseek")
    (tmp_path / "pless_think_t1.0_t1.0_metrics.json").unlink()
    with pytest.raises(SystemExit):
        mod.render_table("deepseek", tmp_path, expected_tasks=748)


def test_allow_partial_renders_only_present_rows(mod, tmp_path):
    _populate(mod, tmp_path, "deepseek")
    for stem in ("temp_p0.7_think_t1.0_t1.0", "pless_renyi_think_t1.0_k0.05_t1.0"):
        (tmp_path / f"{stem}_metrics.json").unlink()
    out = mod.render_table("deepseek", tmp_path, expected_tasks=748, allow_partial=True)
    assert "| G_k=0.35 |" in out and "| p-less baseline (k=2) |" in out
    assert "| top-p @0.7 |" not in out and "| G_k=0.05 |" not in out


def test_missing_rows_lists_pending_labels_in_table_order(mod, tmp_path):
    _populate(mod, tmp_path, "deepseek")
    for stem in ("temp_p0.7_think_t1.0_t1.0", "pless_renyi_think_t1.0_k0.05_t1.0"):
        (tmp_path / f"{stem}_metrics.json").unlink()
    assert mod.missing_labels("deepseek", tmp_path) == ["top-p @0.7", "G_k=0.05"]


def test_allow_partial_still_validates_present_files(mod, tmp_path):
    _populate(mod, tmp_path, "deepseek")
    (tmp_path / "temp_p0.7_think_t1.0_t1.0_metrics.json").unlink()
    _write_metrics(tmp_path / "temp_p0.9_think_t1.0_t1.0_metrics.json", num_tasks=708)
    with pytest.raises(SystemExit) as e:
        mod.render_table("deepseek", tmp_path, expected_tasks=748, allow_partial=True)
    assert "708" in str(e.value)
    _write_metrics(tmp_path / "temp_p0.9_think_t1.0_t1.0_metrics.json", cache=False)
    with pytest.raises(SystemExit) as e:
        mod.render_table("deepseek", tmp_path, expected_tasks=748, allow_partial=True)
    assert "build_decoder_comparison_table.py" in str(e.value)


def test_allow_partial_with_no_files_fails(mod, tmp_path):
    with pytest.raises(SystemExit):
        mod.render_table("deepseek", tmp_path, expected_tasks=748, allow_partial=True)


def test_main_allow_partial_header_lists_pending_and_row_count(mod, tmp_path, monkeypatch):
    base = tmp_path / mod.MODELS["deepseek"]["dir"]
    for label, stem in mod.build_rows("deepseek"):
        if label in ("p-less baseline (k=2)", "G_k=0.35", "top-p @0.9"):
            _write_metrics(base / "metrics" / f"{stem}_metrics.json")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    out = tmp_path / "t.md"
    mod.main(["--model", "deepseek", "--allow-partial", "--out", str(out)])
    text = out.read_text()
    assert "3 of " in text and "Pending" in text
    assert "top-p @0.7" in text.split("Pending")[1] and "recommended" in text.split("Pending")[1]
    assert "| G_k=0.35 |" in text


def test_main_without_allow_partial_fails_on_incomplete_set(mod, tmp_path, monkeypatch):
    base = tmp_path / mod.MODELS["deepseek"]["dir"]
    _write_metrics(base / "metrics" / "pless_think_t1.0_t1.0_metrics.json")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        mod.main(["--model", "deepseek", "--out", str(tmp_path / "t.md")])
