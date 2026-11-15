"""set_print_fonts: enlarge axis names / tick labels / legend text (one factor per kind of text,
capped at the printed targets) only as far as the layout allows: no text clipped by the figure
edge, no new overlaps between text/legend/axes, saved image not wider. Titles untouched."""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "print_fonts", Path(__file__).resolve().parents[1] / "scripts" / "paper" / "print_fonts.py")
pf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pf)


def _printed(fig, t):
    return t.get_fontsize() * pf.COLUMN_IN / pf.saved_width_in(fig)


def _roomy_fig():
    """Small text in a roomy single-axes figure: every target is reachable."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot([1, 2, 3], [1, 4, 9], label="a")
    ax.plot([1, 2, 3], [2, 3, 5], label="b")
    ax.set_xlabel("x", fontsize=5)
    ax.set_ylabel("y", fontsize=5)
    ax.tick_params(labelsize=5)
    ax.set_xlim(0, 4)
    ax.set_xticks([1, 2, 3])                      # no tick label at the plot corner
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 5, 8])
    ax.legend(fontsize=4, loc="upper left")
    return fig, ax


def _crowded_fig():
    """Rotated long tick labels above a below-axes legend: text cannot reach 7pt."""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    names = [f"Model-name-{i}-long" for i in range(7)]
    for j in range(4):
        ax.bar([i + 0.2 * j for i in range(7)], [5 + i for i in range(7)], 0.2, label=f"decoder {j} with T=0.7")
    ax.set_xticks(range(7))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("lift (pp)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=4)
    return fig, ax


def _as_saved_originally(make):
    """The figure as the original scripts saved it: one tight_layout pass."""
    fig, ax = make()
    fig.tight_layout()
    return fig, ax


def test_targets_reached_when_there_is_room():
    fig, ax = _roomy_fig()
    pf.set_print_fonts(fig, relayout=fig.tight_layout)
    assert _printed(fig, ax.xaxis.label) == pytest.approx(8, abs=0.15)
    assert _printed(fig, ax.get_yticklabels()[1]) == pytest.approx(7, abs=0.15)
    assert _printed(fig, ax.get_legend().get_texts()[0]) == pytest.approx(7, abs=0.15)
    plt.close(fig)


def test_crowded_figure_enlarged_without_clipping_overlap_or_widening():
    ref, rax = _as_saved_originally(_crowded_fig)
    w0 = pf.saved_width_in(ref, 180)
    before = _printed(ref, rax.get_xticklabels()[0])
    r = ref.canvas.get_renderer()
    ax_w0, ax_h0 = rax.get_window_extent(r).width, rax.get_window_extent(r).height
    fig, ax = _crowded_fig()
    fs = pf.set_print_fonts(fig, relayout=fig.tight_layout)
    assert max(fs.values()) > 1.0
    assert pf.layout_problems(fig, baseline=set()) == set()
    w1 = pf.saved_width_in(fig, 180)
    assert w1 - w0 <= pf.WIDTH_GROW_TOL_IN + 1e-6 and w0 - w1 <= pf.WIDTH_TOL_IN
    assert _printed(fig, ax.get_xticklabels()[0]) > before
    r = fig.canvas.get_renderer()
    assert ax.get_window_extent(r).width >= pf.MIN_AXES_FRAC * ax_w0
    assert ax.get_window_extent(r).height >= pf.MIN_AXES_FRAC * ax_h0
    plt.close(fig)
    plt.close(ref)


def test_never_shrinks_and_titles_untouched():
    def make():
        fig, ax = _roomy_fig()
        ax.set_title("panel title")
        ax.xaxis.label.set_fontsize(30)           # already above target: must keep its size
        return fig, ax
    ref, rax = _as_saved_originally(make)
    big = _printed(ref, rax.xaxis.label)
    fig, ax = make()
    title = ax.title.get_fontsize()
    pf.set_print_fonts(fig, relayout=fig.tight_layout)
    assert ax.title.get_fontsize() == title
    assert _printed(fig, ax.xaxis.label) >= big - 0.1
    plt.close(fig)


def test_factor_one_reproduces_the_original_figure():
    """Text that is already at/above every target: nothing changes, pixel for pixel."""
    import io
    import matplotlib.image as mi

    def make():
        fig, ax = _roomy_fig()
        for t in [ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts():
            t.set_fontsize(40)
        ax.tick_params(labelsize=40)
        return fig, ax
    imgs = []
    for use_helper in (False, True):
        fig, _ = make()
        if use_helper:
            assert set(pf.set_print_fonts(fig, relayout=fig.tight_layout).values()) == {1.0}
        else:
            fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
        buf.seek(0)
        imgs.append(mi.imread(buf))
        plt.close(fig)
    assert imgs[0].shape == imgs[1].shape and (imgs[0] == imgs[1]).all()


def test_layout_problems_detects_clipping_and_overlap():
    fig, ax = _as_saved_originally(_crowded_fig)
    assert pf.layout_problems(fig, baseline=set()) == set()
    ax.set_ylabel("y " * 80, fontsize=20)          # far taller than the figure: clipped
    for t in ax.get_xticklabels():
        t.set_fontsize(40)                         # tick labels run into the legend
    probs = pf.layout_problems(fig, baseline=set())
    assert any(p[0] == "clipped" for p in probs)
    assert any(p[0] == "overlap" for p in probs)
    plt.close(fig)
