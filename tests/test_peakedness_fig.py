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


def test_blue_bars_equal_share_above_half(tmp_path):
    """Bars right of 1/2 must carry the headline share (counts in bins >= 0.5 vs n_gt_half)."""
    rows = [{"cls": "clean", "onset_token": None, "coll": [0.1, 0.3, 0.6, 0.9, 0.99, 1.0]}]
    pooled = mod.pooled_stats(mod.class_stats(_write(tmp_path, rows), bins=40))
    fig = mod.make_figure(pooled, bins=40)
    blue = [p for p in fig.axes[0].patches if p.get_facecolor()[:3] == (0x2a / 255, 0x78 / 255, 0xd6 / 255)]
    assert sum(p.get_height() for p in blue) == 100.0 * pooled["n_gt_half"] / pooled["n"]


def test_canvas_matches_sibling_figures(tmp_path):
    """Drawn at the same native width as Fig 1 (fig3a, 4.5 in) so that, scaled to the column, its default-size
    text prints at the same size as the other figures instead of oversized (a 3.5 in canvas prints at ~10 pt)."""
    rows = [{"cls": "clean", "onset_token": None, "coll": [0.2, 0.9]}]
    fig = mod.make_figure(mod.pooled_stats(mod.class_stats(_write(tmp_path, rows), bins=40)), bins=40)
    assert fig.get_size_inches()[0] == 4.5
    assert mod.SAVE_DPI == 180
