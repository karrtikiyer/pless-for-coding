"""Enlarge axis names, tick labels and legend text of a paper figure as far as its layout allows.

The LLM4Code paper includes every figure at \\columnwidth (IEEEtran conference, 3.5 in), so a
figure saved W inches wide is scaled by 3.5 / W in print: s pt in the figure prints at
s * 3.5 / W pt.

``set_print_fonts`` multiplies the sizes of the axis names (x/y labels), tick labels and legend
entries of a figure by one factor per kind of text. Each element is capped at its printed target (axis names 8 pt,
ticks and legend 7 pt); an element already printing above its target keeps its size. The
factor is the largest one for which, with the figure's own layout routine re-run,
  * no text is clipped by the figure edge,
  * no new overlap appears between text, legends and axes (overlaps already present in the
    original figure are ignored), and
  * the saved image is at most WIDTH_GROW_TOL_IN wider and WIDTH_TOL_IN narrower than the
    original (print scale essentially unchanged; nothing prints smaller than before),
  * every plot area keeps at least MIN_AXES_FRAC of its original width and height.
Each candidate layout is ONE pass of the figure's own layout routine from the original subplot
positions, as in the original script, so factor 1.0 reproduces the original figure.
Titles, annotations, data, colours and legend placement are not touched.
"""
from __future__ import annotations

import io
import struct
from itertools import combinations
from typing import Callable

import matplotlib as mpl

COLUMN_IN = 3.5          # IEEEtran conference \columnwidth = 252 pt = 3.5 in
AXIS_NAME_PT = 8.0       # printed target for x/y axis names
TICK_PT = 7.0            # printed target for tick labels
LEGEND_PT = 7.0          # printed target for legend entries
WIDTH_TOL_IN = 0.05      # saved image width may move by at most this much
WIDTH_GROW_TOL_IN = 0.02  # ... and may grow by at most this much
MIN_AXES_FRAC = 0.85     # each plot area keeps at least this share of its width and height
_SEARCH_ITERS = 14
_PX_TOL = 1.0            # pixels of slack for clipping / overlap tests


def saved_width_in(fig, dpi: float | None = None) -> float:
    """Width in inches of ``fig.savefig(..., bbox_inches="tight")``.

    With ``dpi``: exact -- the figure is saved to memory at that dpi and the PNG width read back
    (text metrics differ slightly between dpis, so this matches the file the script writes).
    Without: tight bbox at the figure's own dpi plus both pads (fast estimate)."""
    if dpi is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return struct.unpack(">I", buf.getvalue()[16:20])[0] / dpi
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return bbox.width + 2 * mpl.rcParams["savefig.pad_inches"]


def _boxes(fig):
    """name -> display-space bbox of every text / legend / axes frame that must not collide."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = {}

    def add(name, artist):
        if hasattr(artist, "get_text") and not artist.get_text().strip():
            return
        if hasattr(artist, "get_visible") and not artist.get_visible():
            return
        out[name] = artist.get_window_extent(r)

    for i, ax in enumerate(fig.axes):
        out[("axes", i)] = ax.get_window_extent(r)
        add(("xlabel", i), ax.xaxis.label)
        add(("ylabel", i), ax.yaxis.label)
        add(("title", i), ax.title)
        for kind, axis in (("xtick", ax.xaxis), ("ytick", ax.yaxis)):
            lo, hi = sorted(axis.get_view_interval())
            for j, tick in enumerate(axis.get_major_ticks()):   # only ticks actually drawn
                if lo <= tick.get_loc() <= hi:
                    add((kind, i, j), tick.label1)
        if ax.get_legend() is not None:
            out[("legend", i)] = ax.get_legend().get_window_extent(r)
    for k, lg in enumerate(fig.legends):
        out[("figlegend", k)] = lg.get_window_extent(r)
    return out


def _hit(a, b) -> bool:
    return (a.x0 < b.x1 - _PX_TOL and b.x0 < a.x1 - _PX_TOL
            and a.y0 < b.y1 - _PX_TOL and b.y0 < a.y1 - _PX_TOL)


def layout_problems(fig, baseline: set) -> set:
    """Clipped text and colliding elements, minus those already present in ``baseline``."""
    boxes = _boxes(fig)
    # area kept by savefig(bbox_inches="tight"), in display pixels
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    dpi = fig.dpi
    x0, y0, x1, y1 = tb.x0 * dpi, tb.y0 * dpi, tb.x1 * dpi, tb.y1 * dpi
    probs = set()
    for name, b in boxes.items():
        if name[0] == "axes":
            continue
        if b.x0 < x0 - _PX_TOL or b.y0 < y0 - _PX_TOL or b.x1 > x1 + _PX_TOL or b.y1 > y1 + _PX_TOL:
            probs.add(("clipped", name[0]))
    for (n1, b1), (n2, b2) in combinations(boxes.items(), 2):
        if n1[0] == "axes" and n2[0] == "axes":
            continue
        # neighbouring (possibly rotated) tick labels of one axis: their axis-aligned boxes
        # overlap even when the glyphs do not, so this pair is not tested
        if n1[0] == n2[0] and n1[0] in ("xtick", "ytick") and n1[1] == n2[1]:
            continue
        # an axes' own labels / ticks / title sit against its frame by design
        if "axes" in (n1[0], n2[0]):
            other = n2 if n1[0] == "axes" else n1
            ax_i = (n1 if n1[0] == "axes" else n2)[1]
            if other[0] in ("xlabel", "ylabel", "title", "xtick", "ytick") and other[1] == ax_i:
                continue
        if _hit(b1, b2):
            probs.add(("overlap",) + tuple(sorted((n1[0], n2[0]))))
    return probs - baseline


def set_print_fonts(fig, relayout: Callable[[], None] | None = None, *,
                    axis_name_pt: float = AXIS_NAME_PT, tick_pt: float = TICK_PT,
                    legend_pt: float = LEGEND_PT, column_in: float = COLUMN_IN,
                    save_dpi: float = 180) -> dict:
    """Enlarge the text as described in the module docstring; returns {kind: factor}."""
    # Every layout is computed from the subplot positions the figure has NOW (before its
    # layout routine ever ran) with exactly ONE pass of that routine -- as the original script
    # did. Repeated tight_layout passes do not settle for wide below-axes legends: they keep
    # shrinking the axes.
    start = {k: getattr(fig.subplotpars, k) for k in ("left", "right", "bottom", "top", "wspace", "hspace")}

    def lay():
        fig.subplots_adjust(**start)
        if relayout is not None:
            relayout()

    lay()
    w0 = saved_width_in(fig, save_dpi)
    to_fig = w0 / column_in                               # printed pt -> figure pt
    baseline = layout_problems(fig, set())
    r = fig.canvas.get_renderer()
    axes0 = [(ax.get_window_extent(r).width, ax.get_window_extent(r).height) for ax in fig.axes]

    items = []                                            # (kind, setter, original size, cap), figure pt
    for ax in fig.axes:
        for lab in (ax.xaxis.label, ax.yaxis.label):
            items.append(("axis", lab.set_fontsize, lab.get_fontsize(), axis_name_pt * to_fig))
        for which, labels in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            if not labels:
                continue
            size0 = max(t.get_fontsize() for t in labels)

            def set_ticks(s, ax=ax, which=which):
                ax.tick_params(axis=which, which="major", labelsize=s)
                for t in (ax.get_xticklabels() if which == "x" else ax.get_yticklabels()):
                    t.set_fontsize(s)
            items.append(("tick", set_ticks, size0, tick_pt * to_fig))
        if ax.get_legend() is not None:
            items += [("legend", t.set_fontsize, t.get_fontsize(), legend_pt * to_fig)
                      for t in ax.get_legend().get_texts()]
    for lg in fig.legends:
        items += [("legend", t.set_fontsize, t.get_fontsize(), legend_pt * to_fig) for t in lg.get_texts()]

    def apply(fs):
        for kind, setter, size0, cap in items:
            setter(max(size0, min(size0 * fs[kind], cap)))
        lay()

    def ok(fs):
        apply(fs)
        w = saved_width_in(fig, save_dpi)
        # wider image -> everything prints smaller (existing text must not shrink): ~no growth;
        # slightly narrower is allowed
        if layout_problems(fig, baseline) or w - w0 > WIDTH_GROW_TOL_IN or w0 - w > WIDTH_TOL_IN:
            return False
        rr = fig.canvas.get_renderer()
        return all(ax.get_window_extent(rr).width >= MIN_AXES_FRAC * w
                   and ax.get_window_extent(rr).height >= MIN_AXES_FRAC * h
                   for ax, (w, h) in zip(fig.axes, axes0))

    # One factor per kind of text, maximised in turn: a long axis name that is near its limit
    # must not hold back tick labels or legend entries that still have room.
    fs = {"tick": 1.0, "legend": 1.0, "axis": 1.0}
    for kind in ("tick", "legend", "axis"):
        f_full = max([cap / size0 for k, _, size0, cap in items if k == kind and size0 > 0] + [1.0])
        if ok({**fs, kind: f_full}):
            fs[kind] = f_full
            continue
        lo, hi = 1.0, f_full
        for _ in range(_SEARCH_ITERS):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ok({**fs, kind: mid}) else (lo, mid)
        fs[kind] = lo
    while not ok(fs):                                     # leave the figure in a checked state
        if all(v == 1.0 for v in fs.values()):
            apply(fs)
            break
        fs = {k: max(1.0, v - 0.01) for k, v in fs.items()}
    return fs
