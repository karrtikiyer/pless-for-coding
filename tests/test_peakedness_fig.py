"""Tests for the per-class L[P] histogram stats behind the peakedness figure.

Rules under test (see scripts/paper/make_llm4code_fig_peaked.py):
  - clean traces: every thinking token counts
  - loop classes (looping_completed / looping_truncated): only tokens BEFORE loop onset count
    (truncated traces were clipped at onset+500 before scoring, so the tail is not comparable)
  - a loop-class trace with onset None keeps all its tokens
  - "greedy share" = fraction of counted tokens with L[P] > 1/2 (strict; p-less keeps only the top token there)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_llm4code_fig_peaked", REPO_ROOT / "scripts" / "paper" / "make_llm4code_fig_peaked.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _write(tmp_path, rows):
    p = tmp_path / "per_sample.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_clean_counts_all_tokens(tmp_path):
    p = _write(tmp_path, [{"cls": "clean", "onset_token": None, "coll": [0.9, 0.2, 0.6]}])
    s = mod.class_stats(p, bins=10)
    assert s["clean"]["n"] == 3
    assert s["clean"]["n_gt_half"] == 2


def test_loop_class_uses_only_pre_onset_tokens(tmp_path):
    rows = [{"cls": "looping_truncated", "onset_token": 2, "coll": [0.9, 0.9, 0.1, 0.1]}]
    s = mod.class_stats(_write(tmp_path, rows), bins=10)
    assert s["looping_truncated"]["n"] == 2
    assert s["looping_truncated"]["n_gt_half"] == 2


def test_loop_class_without_onset_keeps_all_tokens(tmp_path):
    rows = [{"cls": "looping_completed", "onset_token": None, "coll": [0.9, 0.1]}]
    s = mod.class_stats(_write(tmp_path, rows), bins=10)
    assert s["looping_completed"]["n"] == 2


def test_exactly_half_is_not_greedy(tmp_path):
    p = _write(tmp_path, [{"cls": "clean", "onset_token": None, "coll": [0.5, 0.5000001]}])
    assert mod.class_stats(p, bins=10)["clean"]["n_gt_half"] == 1


def test_histogram_counts_sum_to_n_and_include_one(tmp_path):
    p = _write(tmp_path, [{"cls": "clean", "onset_token": None, "coll": [0.0, 0.25, 1.0, 1.0]}])
    s = mod.class_stats(p, bins=10)["clean"]
    assert int(np.sum(s["counts"])) == s["n"] == 4
    assert s["counts"][-1] == 2  # L[P] == 1.0 lands in the last bin


def test_pooled_sums_classes(tmp_path):
    rows = [
        {"cls": "clean", "onset_token": None, "coll": [0.9, 0.1]},
        {"cls": "looping_truncated", "onset_token": 3, "coll": [0.8, 0.7, 0.6, 0.1, 0.1]},
    ]
    pooled = mod.pooled_stats(mod.class_stats(_write(tmp_path, rows), bins=10))
    assert pooled["n"] == 2 + 3  # clean: all tokens; truncated: pre-onset only
    assert pooled["n_gt_half"] == 1 + 3
    assert int(np.sum(pooled["counts"])) == pooled["n"]


def _fig(tmp_path):
    rows = [{"cls": "clean", "onset_token": None, "coll": [0.1, 0.3, 0.6, 0.9, 0.99, 1.0]}]
    return mod.make_figure(mod.pooled_stats(mod.class_stats(_write(tmp_path, rows), bins=40)), bins=40)


def test_all_bars_one_colour_and_sum_to_100(tmp_path):
    """One fill colour for every bar (user, 2026-11-06); the bars still cover all tokens."""
    bars = _fig(tmp_path).axes[0].patches
    assert len(bars) == 40
    assert len({p.get_facecolor() for p in bars}) == 1
    assert abs(sum(p.get_height() for p in bars) - 100.0) < 1e-9


def test_no_threshold_line_and_no_in_chart_text(tmp_path):
    """No dashed L[P] = 1/2 line and no '82% ...' annotation (user, 2026-11-06); the share goes in the text."""
    ax = _fig(tmp_path).axes[0]
    assert len(ax.get_lines()) == 0
    assert len(ax.texts) == 0


def test_axis_labels(tmp_path):
    """Axis names chosen by the user (2026-11-06); y keeps the (%) unit like the sibling figures' 'pass@k (%)'."""
    ax = _fig(tmp_path).axes[0]
    assert ax.get_ylabel().replace("\n", " ") == "Frequency of Token Distributions (%)"
    assert ax.get_xlabel() == "$p$-less Truncation Threshold, $L[P]$"


def test_canvas_matches_sibling_figures(tmp_path):
    """Drawn at the same native width as Fig 1 (fig3a, 4.5 in) so that, scaled to the column, its default-size
    text prints at the same size as the other figures instead of oversized (a 3.5 in canvas prints at ~10 pt)."""
    rows = [{"cls": "clean", "onset_token": None, "coll": [0.2, 0.9]}]
    fig = mod.make_figure(mod.pooled_stats(mod.class_stats(_write(tmp_path, rows), bins=40)), bins=40)
    assert fig.get_size_inches()[0] == 4.5
    assert mod.SAVE_DPI == 180
