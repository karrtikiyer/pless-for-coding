"""Generate visualizations from metrics JSON files."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Qualitative model colours (up to 8 models)
# ---------------------------------------------------------------------------
_MODEL_COLORS = [
    "#2B6CB0",  # blue
    "#C05621",  # orange
    "#2F855A",  # green
    "#9B2C2C",  # red
    "#6B46C1",  # purple
    "#B7791F",  # gold
    "#2C7A7B",  # teal
    "#702459",  # pink
]

# ---------------------------------------------------------------------------
# Per-method linestyle + marker
# ---------------------------------------------------------------------------
_METHOD_STYLES: dict[str, dict] = {
    "greedy":      dict(linestyle="-",   marker="o"),
    "temp_0.2":    dict(linestyle="-",   marker="^"),
    "temp_0.7":    dict(linestyle="-",   marker="s"),
    "top_p_0.95":  dict(linestyle="-",   marker="D"),
    "temp":         dict(linestyle="-",   marker="s"),
    "top_p":       dict(linestyle="--",  marker="v"),   # Fix A: top_p (p=0.9, t=1.0)
    "p_less":      dict(linestyle="-",   marker="o"),   # canonical default
    "p_less_norm": dict(linestyle="-.",  marker="X"),   # canonical default norm
    # t=0.6 variants — dashed + distinct markers (Fix B)
    "pless":       dict(linestyle="--",  marker="D"),
    "pless_norm":  dict(linestyle="--",  marker="P"),
    "top_p0.9":    dict(linestyle="--",  marker="v"),   # top-p from JSONL discovery
    # BigCode variants (same shapes as non-bigcode equivalents)
    "pless_bigcode":       dict(linestyle="--",  marker="D"),
    "pless_norm_bigcode":  dict(linestyle="--",  marker="P"),
    "temp_bigcode":        dict(linestyle="-",   marker="s"),
    "top_p0.9_bigcode":    dict(linestyle="--",  marker="v"),
    "top_p0.95_bigcode":   dict(linestyle="-",   marker="D"),
}

# ---------------------------------------------------------------------------
# Human-readable display names for legend labels (Fix C)
# ---------------------------------------------------------------------------
_METHOD_DISPLAY_NAMES: dict[str, str] = {
    "greedy":      "greedy",
    "temp_0.2":    "temp (t=0.2)",
    "temp_0.7":    "temp (t=0.7)",
    "top_p_0.95":  "top-p (p=0.95)",
    "temp":         "temp (t=0.7)",
    "top_p":       "top-p (p=0.9, t=1.0)",
    "p_less":      "p-less (default)",
    "p_less_norm": "p-less-norm (default)",
    "pless":       "p-less (t=0.6)",
    "pless_norm":  "p-less-norm (t=0.6)",
    "top_p0.9":    "top-p (p=0.9, t=1.0)",
    # BigCode variants — strip suffix for cleaner legends
    "pless_bigcode":       "p-less (t=0.6)",
    "pless_norm_bigcode":  "p-less-norm (t=0.6)",
    "temp_bigcode":        "temp (t=0.7)",
    "top_p0.9_bigcode":    "top-p (p=0.9, t=1.0)",
    "top_p0.95_bigcode":   "top-p (p=0.95, t=0.2)",
}


def _config_key(m: dict) -> str:
    """Composite identifier for a sampler configuration.

    Two metrics dicts that share this key represent the same (method,
    temperature, top_p, top_k) tuple. Used to give each paper-baseline
    variant a distinct color and legend entry, instead of collapsing
    e.g. ``temp@0.3`` and ``temp@0.7`` under a single ``temp`` entry.
    """
    parts = [m["method"]]
    p = m.get("top_p")
    k = m.get("top_k")
    t = m.get("temperature")
    if p is not None:
        parts.append(f"p{p}")
    if k is not None:
        parts.append(f"k{k}")
    if t is not None:
        parts.append(f"t{t}")
    return "_".join(parts)


def _config_display(m: dict) -> str:
    """Human-readable legend label for a sampler configuration."""
    method = m["method"]
    t = m.get("temperature")
    p = m.get("top_p")
    k = m.get("top_k")
    if method == "greedy":
        return "greedy"
    if method.startswith("beam"):
        return method
    if method == "top_p":
        return f"top-p (p={p}, t={t})"
    if method == "top_k":
        return f"top-k (k={k}, t={t})"
    if method == "pless":
        return f"p-less (t={t})"
    if method == "pless_norm":
        return f"p-less-norm (t={t})"
    if method == "temp":
        return f"temp (t={t})"
    return method


def _build_config_palette(metrics_list: list[dict]) -> dict[str, str]:
    """Assign a distinct color to each unique sampler configuration.

    Configs are ordered by first appearance in ``metrics_list`` so the
    same input always produces the same color assignment.
    """
    import matplotlib.colors as mcolors

    keys = list(dict.fromkeys(_config_key(m) for m in metrics_list))
    cmap = plt.get_cmap("tab20")
    return {k: mcolors.to_hex(cmap(i % 20)) for i, k in enumerate(keys)}


def _family_key(m: dict) -> str:
    """Coarser palette key than ``_config_key``: collapses paper-baseline
    families (temp, top_p, top_k, beam) and the p-less / p-less-norm
    families to one color each. Temperature is *not* encoded in the color.

    Used for headline figures where the story is "p-less vs the paper's
    per-model recommended config" — within-family T-trajectory is read off
    the two same-color points per model.
    """
    method = m["method"]
    if method in ("pless", "p_less"):
        return "pless"
    if method in ("pless_norm", "p_less_norm"):
        return "pless_norm"
    if method.startswith("beam"):
        return "beam"
    return method


def _family_display(key: str) -> str:
    """Human-readable legend label for ``_family_key`` outputs."""
    if key == "beam":
        return "beam (4, 8)"
    if key == "top_p":
        return "top-p"
    if key == "top_k":
        return "top-k"
    if key == "temp":
        return "temp"
    if key == "greedy":
        return "greedy"
    if key == "pless":
        return "p-less (t=0.6, 1.0)"
    if key == "pless_norm":
        return "p-less-norm (t=0.6, 1.0)"
    return key


def _build_family_palette(metrics_list: list[dict]) -> dict[str, str]:
    """Assign a distinct color per family key (see ``_family_key``)."""
    import matplotlib.colors as mcolors

    keys = list(dict.fromkeys(_family_key(m) for m in metrics_list))
    cmap = plt.get_cmap("tab10")
    return {k: mcolors.to_hex(cmap(i % 10)) for i, k in enumerate(keys)}


_TOP_P_FILENAME_RE = re.compile(r"top_p(\d+(?:\.\d+)?)")
_TOP_K_FILENAME_RE = re.compile(r"top_k(\d+)")


def _backfill_truncation_params(m: dict, source_path: Path) -> None:
    """Recover ``top_p``/``top_k`` from the source filename when missing in JSON.

    Older runs wrote metrics JSONs with these fields nulled even when the
    filename clearly encodes the value (e.g. ``top_p0.9_t1.0_metrics.json``).
    Backfilling here keeps ``_config_key`` and the legend correct without
    needing to regenerate the JSONs.
    """
    stem = source_path.stem
    if m["method"] == "top_p" and m.get("top_p") is None:
        match = _TOP_P_FILENAME_RE.search(stem)
        if match:
            m["top_p"] = float(match.group(1))
    if m["method"] == "top_k" and m.get("top_k") is None:
        match = _TOP_K_FILENAME_RE.search(stem)
        if match:
            m["top_k"] = int(match.group(1))


def load_metrics(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            m = json.load(f)
        # Normalize model names: directory-style "org--model" → HF-style "org/model"
        if "model" in m:
            m["model"] = m["model"].replace("--", "/", 1)
        _backfill_truncation_params(m, Path(p))
        results.append(m)
    return results


def _label_for(m: dict) -> str:
    model = m["model"].split("/")[-1] if "/" in m["model"] else m["model"]
    method_display = _METHOD_DISPLAY_NAMES.get(m["method"], m["method"])
    return f"{model} ({method_display})"


def _build_style_map(metrics_list: list[dict]) -> dict[tuple[str, str], dict]:
    """Pre-compute a colour/style dict keyed by (model, method)."""
    models = list(dict.fromkeys(m["model"] for m in metrics_list))
    model_color = {name: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, name in enumerate(models)}

    style_map: dict[tuple[str, str], dict] = {}
    for m in metrics_list:
        key = (m["model"], m["method"])
        if key in style_map:
            continue
        method_style = _METHOD_STYLES.get(m["method"], dict(linestyle="-", marker="x"))
        style_map[key] = dict(
            color=model_color[m["model"]],
            linestyle=method_style["linestyle"],
            marker=method_style["marker"],
            linewidth=2,
            markersize=6,
        )
    return style_map


def _style_for(m: dict, style_map: dict[tuple[str, str], dict] | None = None) -> dict:
    """Return color, linestyle, and marker for a config.

    If *style_map* is provided (from ``_build_style_map``), look up the entry.
    Otherwise fall back to the legacy 2-model heuristic for backward compat.
    """
    if style_map is not None:
        key = (m["model"], m["method"])
        if key in style_map:
            return style_map[key]

    # Legacy fallback (2-model MBPP case)
    is_instruct = "instruct" in m["model"].lower() or "coder" in m["model"].lower()
    is_norm = m["method"] in ("pless_norm", "p_less_norm")
    color = "#C05621" if is_instruct else "#2B6CB0"
    linestyle = "--" if is_norm else "-"
    marker = "s" if is_norm else "o"
    if is_instruct and is_norm:
        color = "#ED8936"
    elif not is_instruct and is_norm:
        color = "#63B3ED"
    return dict(color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6)


def plot_aggregate_lines(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Line plot with 3 subplots: pass@k, cover@t, cover@t (distinct).

    Each subplot has one line per model+method configuration.
    """
    style_map = _build_style_map(metrics_list)
    n_lines = len(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for m in metrics_list:
        label = _label_for(m)
        style = _style_for(m, style_map)

        # pass@k
        ks = sorted(m["pass_at_k"], key=lambda x: int(x))
        axes[0].plot(
            [int(k) for k in ks],
            [m["pass_at_k"][k] * 100 for k in ks],
            label=label, **style,
        )

        # cover@t
        ts = sorted(m["cover_at_t"], key=lambda x: float(x))
        axes[1].plot(
            [float(t) for t in ts],
            [m["cover_at_t"][t] for t in ts],
            label=label, **style,
        )

        # cover@t (distinct)
        ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
        axes[2].plot(
            [float(t) for t in ts_d],
            [m["cover_at_t_distinct"][t] for t in ts_d],
            label=label, **style,
        )

    axes[0].set_xlabel("k")
    axes[0].set_title("pass@k")
    axes[0].set_xticks([int(k) for k in ks])

    axes[1].set_xlabel("t")
    axes[1].set_title("cover@t")
    axes[1].set_xticks([float(t) for t in ts])

    axes[2].set_xlabel("t")
    axes[2].set_title("cover@t (distinct)")
    axes[2].set_xticks([float(t) for t in ts_d])

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.grid(alpha=0.3)

    # Adjust legend font size based on number of lines
    legend_fontsize = 5 if n_lines > 10 else 7
    axes[2].legend(fontsize=legend_fontsize, loc="upper right")
    fig.suptitle(f"{dataset_name}: Metrics Overview", fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-method colours for faceted plots (model is implicit from the row)
# ---------------------------------------------------------------------------
_METHOD_COLORS: dict[str, str] = {
    "greedy":      "#2B6CB0",  # blue
    "temp_0.2":    "#C05621",  # orange
    "temp_0.7":    "#2F855A",  # green
    "top_p_0.95":  "#9B2C2C",  # red
    "temp":         "#2F855A",  # green (same family as temp_0.7)
    "top_p":       "#2C7A7B",  # teal (Fix A)
    "p_less":      "#6B46C1",  # dark purple (canonical default)
    "p_less_norm": "#B7791F",  # dark gold   (canonical default norm)
    # t=0.6 variants — lighter shades to indicate same algorithm (Fix B)
    "pless":       "#9F7AEA",  # light purple
    "pless_norm":  "#ECC94B",  # yellow-gold
    "top_p0.9":    "#2C7A7B",  # teal (same family as top_p)
    # BigCode variants — same colors as non-bigcode equivalents
    "pless_bigcode":       "#9F7AEA",
    "pless_norm_bigcode":  "#ECC94B",
    "temp_bigcode":        "#2F855A",
    "top_p0.9_bigcode":    "#2C7A7B",
    "top_p0.95_bigcode":   "#9B2C2C",
}


def plot_aggregate_lines_faceted(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Faceted line plot: one row per model, columns for pass@k / cover@t / cover@t distinct.

    Designed for many configs (e.g. 4 models × 6 methods = 24) where a single
    row of 3 subplots would be unreadable.
    """
    # Group by model, preserving insertion order
    models_order: list[str] = list(dict.fromkeys(m["model"] for m in metrics_list))
    by_model: dict[str, list[dict]] = {model: [] for model in models_order}
    for m in metrics_list:
        by_model[m["model"]].append(m)

    n_models = len(models_order)
    col_titles = ["pass@k", "cover@t", "cover@t (distinct)"]

    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(15, 4 * n_models),
        sharey=True,
        squeeze=False,
    )

    # Track which methods we've added to the legend
    legend_handles: dict[str, matplotlib.lines.Line2D] = {}

    for row, model in enumerate(models_order):
        for m in by_model[model]:
            method = m["method"]
            method_style = _METHOD_STYLES.get(method, dict(linestyle="-", marker="x"))
            color = _METHOD_COLORS.get(method, "#333333")
            style = dict(
                color=color,
                linestyle=method_style["linestyle"],
                marker=method_style["marker"],
                linewidth=2,
                markersize=6,
            )

            # pass@k
            ks = sorted(m["pass_at_k"], key=lambda x: int(x))
            line, = axes[row, 0].plot(
                [int(k) for k in ks],
                [m["pass_at_k"][k] * 100 for k in ks],
                **style,
            )
            if method not in legend_handles:
                legend_handles[method] = line

            # cover@t
            ts = sorted(m["cover_at_t"], key=lambda x: float(x))
            axes[row, 1].plot(
                [float(t) for t in ts],
                [m["cover_at_t"][t] for t in ts],
                **style,
            )

            # cover@t (distinct)
            ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
            axes[row, 2].plot(
                [float(t) for t in ts_d],
                [m["cover_at_t_distinct"][t] for t in ts_d],
                **style,
            )

        # Row label (model name) on the left
        short_model = model.split("/")[-1] if "/" in model else model
        axes[row, 0].set_ylabel(f"{short_model}\n\nPercentage", fontsize=9)

        for col in range(3):
            axes[row, col].set_ylim(0, 100)
            axes[row, col].grid(alpha=0.3)

        # x-tick labels only on the bottom row
        if row < n_models - 1:
            for col in range(3):
                axes[row, col].tick_params(labelbottom=False)

    # Column headers and x-labels on the bottom row
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)
    axes[-1, 0].set_xlabel("k")
    axes[-1, 1].set_xlabel("t")
    axes[-1, 2].set_xlabel("t")

    # Set x-ticks from the last plotted data
    if metrics_list:
        last = metrics_list[-1]
        ks = sorted(last["pass_at_k"], key=lambda x: int(x))
        ts = sorted(last["cover_at_t"], key=lambda x: float(x))
        ts_d = sorted(last["cover_at_t_distinct"], key=lambda x: float(x))
        for row in range(n_models):
            axes[row, 0].set_xticks([int(k) for k in ks])
            axes[row, 1].set_xticks([float(t) for t in ts])
            axes[row, 2].set_xticks([float(t) for t in ts_d])

    # Shared legend at the bottom — use display names (Fix C)
    legend_labels = [_METHOD_DISPLAY_NAMES.get(k, k) for k in legend_handles.keys()]
    fig.legend(
        legend_handles.values(),
        legend_labels,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(f"{dataset_name}: Metrics Overview (by model)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correctness_vs_diversity(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Bubble chart of num_correct vs num_distinct_correct per task.

    Uses only pless method configs (not pless_norm) to reduce clutter.
    """
    style_map = _build_style_map(metrics_list)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Fix E: deduplicate — prefer pless (t=0.6) over p_less (default) per model
    seen_models: set[str] = set()
    pless_configs = []
    for m in metrics_list:
        if m["method"] == "pless":
            seen_models.add(m["model"])
            pless_configs.append(m)
    for m in metrics_list:
        if m["method"] == "p_less" and m["model"] not in seen_models:
            pless_configs.append(m)
    if not pless_configs:
        pless_configs = metrics_list  # fallback

    for m in pless_configs:
        color = _style_for(m, style_map)["color"]
        label = _label_for(m)

        counts: Counter[tuple[int, int]] = Counter()
        for task in m["per_task"]:
            nc = task["num_correct"]
            ndc = task["num_distinct_correct"]
            counts[(nc, ndc)] += 1

        xs, ys, sizes = [], [], []
        for (nc, ndc), count in counts.items():
            xs.append(nc)
            ys.append(ndc)
            sizes.append(count)

        # Scale bubble sizes for visibility
        sizes_arr = np.array(sizes, dtype=float)
        ax.scatter(
            xs, ys,
            s=sizes_arr * 15,
            alpha=0.55,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            label=label,
        )

    # y=x reference line
    ax.plot([0, 10], [0, 10], "k--", alpha=0.3, label="y = x (max diversity)")

    # Fix E: bubble size legend (reference entries)
    for n_tasks in (1, 5, 10, 20):
        ax.scatter([], [], s=n_tasks * 15, c="gray", alpha=0.5,
                   label=f"{n_tasks} task{'s' if n_tasks > 1 else ''}")

    ax.set_xlabel("num_correct (out of 10)")
    ax.set_ylabel("num_distinct_correct (out of 10)")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_aspect("equal")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"{dataset_name}: Correctness vs Diversity per Task (p-less family)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_structural_diversity_bars(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Grouped bar chart of structural diversity per model × sampler config.

    One bar per unique (method, temperature, top_p, top_k) tuple — paper-
    baseline variants like ``temp@0.3`` and ``temp@0.7`` get separate
    bars instead of being silently collapsed to whichever appears first.
    """
    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    config_order = list(dict.fromkeys(_config_key(m) for m in metrics_list))
    config_palette = _build_config_palette(metrics_list)
    config_display = {_config_key(m): _config_display(m) for m in metrics_list}

    fig, ax = plt.subplots(figsize=(max(8, len(models_order) * 2.5), 5))

    bar_width = 0.8 / max(len(config_order), 1)
    x = np.arange(len(models_order))

    for i, ck in enumerate(config_order):
        values = []
        for model in models_order:
            match = [
                m for m in metrics_list
                if m["model"] == model and _config_key(m) == ck
                and "structural_diversity" in m
            ]
            values.append(match[0]["structural_diversity"] if match else 0.0)

        offset = (i - len(config_order) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, values, bar_width * 0.9,
            label=config_display[ck], color=config_palette[ck], alpha=0.85,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Structural Diversity\n(mean pairwise AST edit distance)")
    ax.set_title(f"{dataset_name}: Structural Diversity by Model & Sampler")
    ax.set_xticks(x)
    short_models = [m.split("/")[-1] if "/" in m else m for m in models_order]
    ax.set_xticklabels(short_models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, ncol=2 if len(config_order) > 8 else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pairwise_distance_distributions(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Box plot of per-task mean_pairwise_distance distributions.

    Fix D: 2×2 faceted layout (one panel per model) instead of a single
    ultra-wide figure.  Each panel shows one box per method.
    """
    import matplotlib.patches as mpatches

    # Group by model
    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    n_models = len(models_order)
    if n_models == 0:
        return

    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

    # Track which methods appeared (for shared legend)
    seen_methods: dict[str, str] = {}  # method -> color

    for idx, model in enumerate(models_order):
        ax = axes[idx // ncols][idx % ncols]
        model_metrics = [m for m in metrics_list if m["model"] == model]

        box_data = []
        box_labels = []
        box_colors = []

        for m in model_metrics:
            distances = [
                t["mean_pairwise_distance"]
                for t in m.get("per_task", [])
                if t.get("num_correct", 0) >= 2 and "mean_pairwise_distance" in t
            ]
            if not distances:
                continue
            method = m["method"]
            color = _METHOD_COLORS.get(method, "#333333")
            box_data.append(distances)
            box_labels.append(_METHOD_DISPLAY_NAMES.get(method, method))
            box_colors.append(color)
            seen_methods.setdefault(method, color)

        if not box_data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
        short_model = model.split("/")[-1] if "/" in model else model
        ax.set_title(short_model, fontsize=10)
        ax.set_ylabel("Mean Pairwise AST Edit Distance")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty panels if n_models is odd
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend using Patch handles
    legend_patches = [
        mpatches.Patch(facecolor=color, alpha=0.6,
                       label=_METHOD_DISPLAY_NAMES.get(method, method))
        for method, color in seen_methods.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(legend_patches), 5),
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(f"{dataset_name}: Distribution of Structural Diversity Across Tasks", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correctness_vs_diversity_multimethod(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "HUMANEVAL",
    methods: list[str] | None = None,
) -> None:
    """2×2 faceted bubble chart: correctness vs distinct-correct, one panel per model.

    Shows multiple methods per panel (color = method) to compare sampling strategies.
    """
    if methods is None:
        methods = list(dict.fromkeys(m["method"] for m in metrics_list))

    # Filter to requested methods
    filtered = [m for m in metrics_list if m["method"] in methods]
    if not filtered:
        return

    models_order = list(dict.fromkeys(m["model"] for m in filtered))
    n_models = len(models_order)
    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)

    for idx, model in enumerate(models_order):
        ax = axes[idx // ncols][idx % ncols]
        model_metrics = [m for m in filtered if m["model"] == model]

        for m in model_metrics:
            method = m["method"]
            color = _METHOD_COLORS.get(method, "#333333")
            label = _METHOD_DISPLAY_NAMES.get(method, method)

            counts: Counter[tuple[int, int]] = Counter()
            for task in m.get("per_task", []):
                nc = task["num_correct"]
                ndc = task.get("num_distinct_correct", 0)
                counts[(nc, ndc)] += 1

            xs, ys, sizes = [], [], []
            for (nc, ndc), count in counts.items():
                xs.append(nc)
                ys.append(ndc)
                sizes.append(count)

            ax.scatter(
                xs, ys,
                s=np.array(sizes, dtype=float) * 15,
                alpha=0.55,
                color=color,
                edgecolors="white",
                linewidth=0.5,
                label=label,
            )

        ax.plot([0, 10], [0, 10], "k--", alpha=0.3)
        short_model = model.split("/")[-1] if "/" in model else model
        ax.set_title(short_model, fontsize=10)
        ax.set_xlabel("num_correct (out of 10)")
        ax.set_ylabel("num_distinct_correct (out of 10)")
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticks(range(11))
        ax.set_yticks(range(11))
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper left")

    # Hide empty panels
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{dataset_name}: Correctness vs Diversity — Multi-Method Comparison",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_diversity_metrics_bars(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "HUMANEVAL",
    methods: list[str] | None = None,
) -> None:
    """Grouped bar chart comparing multiple diversity metrics across models and methods.

    Shows AST structural diversity, CodeBLEU diversity, and dataflow diversity side by side.
    """
    if methods is None:
        methods = list(dict.fromkeys(m["method"] for m in metrics_list))

    filtered = [m for m in metrics_list if m["method"] in methods]
    if not filtered:
        return

    models_order = list(dict.fromkeys(m["model"] for m in filtered))
    config_order = list(dict.fromkeys(_config_key(m) for m in filtered))
    config_palette = _build_config_palette(filtered)
    config_display = {_config_key(m): _config_display(m) for m in filtered}

    diversity_keys = [
        ("structural_diversity", "AST Edit Distance"),
        ("codebleu_diversity", "1 - CodeBLEU"),
        ("dataflow_match_diversity", "1 - Dataflow Match"),
    ]

    n_metrics = len(diversity_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)

    for col, (key, title) in enumerate(diversity_keys):
        ax = axes[0, col]
        bar_width = 0.8 / max(len(config_order), 1)
        x = np.arange(len(models_order))

        for i, ck in enumerate(config_order):
            values = []
            for model in models_order:
                match = [
                    m for m in filtered
                    if m["model"] == model and _config_key(m) == ck and key in m
                ]
                values.append(match[0][key] if match else 0.0)

            offset = (i - len(config_order) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, values, bar_width * 0.9,
                label=config_display[ck], color=config_palette[ck], alpha=0.85,
            )

        ax.set_title(title, fontsize=10)
        short_models = [m.split("/")[-1] if "/" in m else m for m in models_order]
        ax.set_xticks(x)
        ax.set_xticklabels(short_models, rotation=25, ha="right", fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("Diversity Score")
            ax.legend(fontsize=7, ncol=2 if len(config_order) > 8 else 1)

    fig.suptitle(
        f"{dataset_name}: Diversity Metrics by Model & Sampler",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Marker shapes for per-model encoding in scatter plots
# ---------------------------------------------------------------------------
_MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_pareto_scatter(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "HUMANEVAL",
    methods: list[str] | None = None,
    diversity_key: str = "codebleu_diversity",
    diversity_fallback: str = "structural_diversity",
    config_keys: list[str] | None = None,
    show_trajectories: bool = True,
    family_palette: bool = False,
) -> None:
    """Aggregate Pareto scatter: pass@1 vs mean diversity, one point per (model, method).

    Color encodes method, marker shape encodes model.  A dashed Pareto frontier
    connects non-dominated points (higher pass@1 AND higher diversity is better).
    """
    if methods is None:
        methods = list(dict.fromkeys(m["method"] for m in metrics_list))

    filtered = [m for m in metrics_list if m["method"] in methods]
    if config_keys is not None:
        wanted = set(config_keys)
        filtered = [m for m in filtered if _config_key(m) in wanted]
    if not filtered:
        return

    # Detect which diversity key is actually available
    has_primary = any(diversity_key in m for m in filtered)
    active_key = diversity_key if has_primary else diversity_fallback

    models_order = list(dict.fromkeys(m["model"] for m in filtered))
    model_marker = {name: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, name in enumerate(models_order)}

    if family_palette:
        palette = _build_family_palette(filtered)
        key_fn = _family_key
        display_map = {_family_key(m): _family_display(_family_key(m)) for m in filtered}
    else:
        palette = _build_config_palette(filtered)
        key_fn = _config_key
        display_map = {_config_key(m): _config_display(m) for m in filtered}
    key_order = list(dict.fromkeys(key_fn(m) for m in filtered))

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 7))

    # Collect (model, pass1, div) triples for per-model lines
    all_points: list[tuple[str, float, float]] = []

    for m in filtered:
        model = m["model"]
        pass1 = m.get("pass_at_k", {}).get("1", 0.0)
        div = m.get(active_key, 0.0)

        color = palette[key_fn(m)]
        marker = model_marker[model]

        ax.scatter(
            pass1, div,
            s=160,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
        all_points.append((model, pass1, div))

    # Per-model connecting lines: show within-model trade-off trajectory
    if show_trajectories:
        for model in models_order:
            model_pts = [(p1, d) for (m_model, p1, d) in all_points if m_model == model]
            if len(model_pts) < 2:
                continue
            model_pts.sort(key=lambda p: p[0])
            mx, my = zip(*model_pts)
            ax.plot(mx, my, "-", color="gray", alpha=0.3, linewidth=1.0, zorder=1)

    # --- Two separate legends with section titles ---
    # Method legend (color) — one entry per (method, temperature, top_p, top_k)
    method_handles: list[mpatches.Patch | Line2D] = [
        Line2D([0], [0], color="none", label="$\\bf{Sampler}$"),
    ]
    for ck in key_order:
        method_handles.append(mpatches.Patch(color=palette[ck], label=display_map[ck]))

    # Model legend (marker shape)
    model_handles: list[Line2D] = [
        Line2D([0], [0], color="none", label="$\\bf{Model}$"),  # section title
    ]
    for model in models_order:
        short = model.split("/")[-1] if "/" in model else model
        marker = model_marker[model]
        model_handles.append(
            Line2D([0], [0], marker=marker, color="gray", linestyle="None",
                   markersize=8, markeredgecolor="black", markeredgewidth=0.8,
                   label=short)
        )

    # Place both legends outside the plot area (right side)
    leg_method = ax.legend(
        handles=method_handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.9,
        borderaxespad=0,
    )
    ax.add_artist(leg_method)  # keep first legend when adding second
    ax.legend(
        handles=model_handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.42),
        framealpha=0.9,
        borderaxespad=0,
    )

    _DIVERSITY_LABELS = {
        "codebleu_diversity": "CodeBLEU Diversity",
        "structural_diversity": "Structural Diversity",
        "ngram_match_diversity": "N-gram Diversity",
        "weighted_ngram_match_diversity": "Weighted N-gram Diversity",
        "syntax_match_diversity": "Syntax Match Diversity",
        "dataflow_match_diversity": "Dataflow Diversity",
    }
    div_display = _DIVERSITY_LABELS.get(active_key, active_key)
    ax.set_xlabel("pass@1", fontsize=11)
    ax.set_ylabel(f"Mean {div_display}", fontsize=11)
    ax.set_title(f"{dataset_name}: Correctness vs Diversity Trade-off", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_method_heatmaps(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "HUMANEVAL",
    methods: list[str] | None = None,
    diversity_key: str = "codebleu_diversity",
    diversity_fallback: str = "structural_diversity",
) -> None:
    """Side-by-side heatmaps: model × method grids for pass@1 and mean diversity."""
    if methods is None:
        methods = list(dict.fromkeys(m["method"] for m in metrics_list))

    filtered = [m for m in metrics_list if m["method"] in methods]
    if not filtered:
        return

    # Detect which diversity key is actually available
    has_primary = any(diversity_key in m for m in filtered)
    active_key = diversity_key if has_primary else diversity_fallback

    models_order = list(dict.fromkeys(m["model"] for m in filtered))
    methods_order = [mt for mt in methods if any(f["method"] == mt for f in filtered)]

    n_models = len(models_order)
    n_methods = len(methods_order)

    # Build 2D arrays
    pass1_grid = np.full((n_models, n_methods), np.nan)
    div_grid = np.full((n_models, n_methods), np.nan)

    for m in filtered:
        row = models_order.index(m["model"])
        if m["method"] not in methods_order:
            continue
        col = methods_order.index(m["method"])
        pass1_grid[row, col] = m.get("pass_at_k", {}).get("1", np.nan)
        div_grid[row, col] = m.get(active_key, np.nan)

    short_models = [m.split("/")[-1] if "/" in m else m for m in models_order]
    method_labels = [_METHOD_DISPLAY_NAMES.get(mt, mt) for mt in methods_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 + n_methods * 1.2, 1.5 + n_models * 0.9))

    _DIVERSITY_LABELS = {
        "codebleu_diversity": "CodeBLEU Diversity",
        "structural_diversity": "Structural Diversity",
        "ngram_match_diversity": "N-gram Diversity",
        "weighted_ngram_match_diversity": "Weighted N-gram Diversity",
        "syntax_match_diversity": "Syntax Match Diversity",
        "dataflow_match_diversity": "Dataflow Diversity",
    }
    div_title = _DIVERSITY_LABELS.get(active_key, "Diversity")

    for ax, grid, cmap, title in [
        (ax1, pass1_grid, "Blues", "pass@1"),
        (ax2, div_grid, "Greens", div_title),
    ]:
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(n_models):
            for j in range(n_methods):
                val = grid[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
                else:
                    # Choose text color for readability
                    text_color = "white" if val > 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=text_color)

        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(method_labels, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(short_models, fontsize=8)
        ax.set_title(f"{dataset_name}: {title}", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _model_family_rank(model: str) -> tuple[int, int, str]:
    """Sort key that groups rows by model family for the heatmap.

    Lower tuples sort first. The family priority surfaces base→chat
    pairings (Llama-2 → CodeLlama → Qwen-7B → Qwen2.5-Coder →
    OpenCodeInterpreter), with sub-rank distinguishing base from
    chat/instruct variants.
    """
    short = model.split("/")[-1] if "/" in model else model
    s = short.lower()

    if "llama-2" in s:
        return (0, 1 if ("chat" in s or "instruct" in s) else 0, short)
    if "codellama" in s:
        return (1, 1 if ("instruct" in s or "chat" in s) else 0, short)
    if "qwen-7b" in s or s.startswith("qwen-7b"):
        return (2, 1 if ("chat" in s or "instruct" in s) else 0, short)
    if "qwen2.5-coder" in s:
        if "1.5b" in s:
            return (3, 0, short)
        if "3b" in s:
            return (3, 1, short)
        if "7b" in s:
            return (3, 2 if ("instruct" in s or "chat" in s) else 1, short)
        return (3, 9, short)
    if "opencodeinterpreter" in s:
        return (4, 1 if ("instruct" in s or "chat" in s) else 0, short)
    return (5, 0, short)


def plot_pass_at_1_heatmap(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
    sort_columns_by: str = "mean_rank",
    group_rows_by_family: bool = True,
) -> None:
    """Single-panel heatmap of pass@1 across (model, sampler).

    Rows are models, columns are sampler configurations. Cells are
    annotated with the pass@1 percentage; missing combinations render
    "—". The per-row best sampler gets a bold annotation and a black
    rectangular border. Designed to replace a multi-panel bar grid that
    fails to scale to paper-page width.
    """
    from matplotlib.patches import Rectangle

    if not metrics_list:
        return

    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    if group_rows_by_family:
        models_order.sort(key=_model_family_rank)
    else:
        models_order.sort(key=lambda m: (m.split("/")[-1] if "/" in m else m).lower())

    config_keys_seen: list[str] = []
    config_display: dict[str, str] = {}
    for m in metrics_list:
        k = _config_key(m)
        if k not in config_display:
            config_keys_seen.append(k)
            config_display[k] = _config_display(m)

    matrix: dict[tuple[str, str], float] = {}
    for m in metrics_list:
        p1 = m.get("pass_at_k", {}).get("1")
        if p1 is None:
            continue
        matrix[(m["model"], _config_key(m))] = p1 * 100.0

    if sort_columns_by == "mean_rank":
        means = {}
        for ck in config_keys_seen:
            vals = [matrix[(model, ck)] for model in models_order if (model, ck) in matrix]
            means[ck] = float(np.mean(vals)) if vals else float("-inf")
        config_keys_order = sorted(config_keys_seen, key=lambda k: means[k], reverse=True)
    elif sort_columns_by == "alpha":
        config_keys_order = sorted(config_keys_seen, key=lambda k: config_display[k].lower())
    else:
        config_keys_order = list(config_keys_seen)

    n_rows = len(models_order)
    n_cols = len(config_keys_order)

    data = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, model in enumerate(models_order):
        for j, ck in enumerate(config_keys_order):
            v = matrix.get((model, ck))
            if v is not None:
                data[i, j] = v

    fig_w = max(10.0, 0.55 * n_cols + 4)
    fig_h = max(4.0, 0.6 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("viridis")
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")
    cmap.set_bad(color="#E2E8F0")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("pass@1 (%)", fontsize=9)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [config_display[k] for k in config_keys_order],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [(m.split("/")[-1] if "/" in m else m) for m in models_order],
        fontsize=9,
    )

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    row_winners: dict[int, int] = {}
    for i in range(n_rows):
        row = data[i]
        if np.all(np.isnan(row)):
            continue
        row_winners[i] = int(np.nanargmax(row))

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#4A5568", fontsize=8)
                continue
            text_color = "white" if v < 50.0 else "black"
            is_winner = row_winners.get(i) == j
            ax.text(
                j, i, f"{v:.1f}",
                ha="center", va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold" if is_winner else "normal",
            )
            if is_winner:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=1.8,
                ))

    ax.set_title(
        f"{dataset_name}: pass@1 (%) — model × sampler",
        fontsize=11,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_overview(
    metrics_list: list[dict],
    output_path: Path,
    dataset_name: str = "MBPP",
) -> None:
    """Faceted line plot with tight y-limits: one row per model, 3 columns.

    Similar to plot_aggregate_lines_faceted but with zoomed y-axes to
    highlight differences between methods (matching the MBPP per-family style).
    """
    import math as _math

    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    by_model: dict[str, list[dict]] = {model: [] for model in models_order}
    for m in metrics_list:
        by_model[m["model"]].append(m)

    n_models = len(models_order)
    if n_models == 0:
        return

    col_titles = ["pass@k", "cover@t", "cover@t (distinct)"]
    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4 * n_models), sharey=False, squeeze=False,
    )

    legend_handles: dict[str, object] = {}
    col_vals: list[list[list[float]]] = [[[] for _ in range(3)] for _ in range(n_models)]

    for row, model in enumerate(models_order):
        for m in by_model[model]:
            method = m["method"]
            method_style = _METHOD_STYLES.get(method, dict(linestyle="-", marker="x"))
            color = _METHOD_COLORS.get(method, "#333333")
            label = _METHOD_DISPLAY_NAMES.get(method, method)
            style = dict(
                color=color,
                linestyle=method_style["linestyle"],
                marker=method_style["marker"],
                linewidth=2.5,
                markersize=8,
            )

            # pass@k
            ks = sorted(m["pass_at_k"], key=lambda x: int(x))
            vals0 = [m["pass_at_k"][k] * 100 for k in ks]
            col_vals[row][0].extend(vals0)
            line, = axes[row, 0].plot([int(k) for k in ks], vals0, **style)
            if label not in legend_handles:
                legend_handles[label] = line

            # cover@t
            ts = sorted(m["cover_at_t"], key=lambda x: float(x))
            vals1 = [m["cover_at_t"][t] for t in ts]
            col_vals[row][1].extend(vals1)
            axes[row, 1].plot([float(t) for t in ts], vals1, **style)

            # cover@t (distinct)
            ts_d = sorted(m["cover_at_t_distinct"], key=lambda x: float(x))
            vals2 = [m["cover_at_t_distinct"][t] for t in ts_d]
            col_vals[row][2].extend(vals2)
            axes[row, 2].plot([float(t) for t in ts_d], vals2, **style)

        short_model = model.split("/")[-1] if "/" in model else model
        axes[row, 0].set_ylabel(f"{short_model}\n\nPercentage", fontsize=9)
        for c in range(3):
            axes[row, c].grid(alpha=0.3)
        if row < n_models - 1:
            for c in range(3):
                axes[row, c].tick_params(labelbottom=False)

    # Tight per-model per-column y-limits
    for row in range(n_models):
        for c in range(3):
            vals = col_vals[row][c]
            if not vals:
                continue
            lo = max(0.0, _math.floor(min(vals)) - 3)
            hi = min(100.0, _math.ceil(max(vals)) + 3)
            if hi - lo < 15:
                mid = (lo + hi) / 2
                lo, hi = max(0.0, mid - 8), min(100.0, mid + 8)
            axes[row, c].set_ylim(lo, hi)

    # Column titles and x-labels
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11)
    axes[-1, 0].set_xlabel("k")
    axes[-1, 1].set_xlabel("t")
    axes[-1, 2].set_xlabel("t")

    # Set x-ticks from last plotted data
    if metrics_list:
        last = metrics_list[-1]
        ks = sorted(last["pass_at_k"], key=lambda x: int(x))
        ts = sorted(last["cover_at_t"], key=lambda x: float(x))
        ts_d = sorted(last["cover_at_t_distinct"], key=lambda x: float(x))
        for row in range(n_models):
            axes[row, 0].set_xticks([int(k) for k in ks])
            axes[row, 1].set_xticks([float(t) for t in ts])
            axes[row, 2].set_xticks([float(t) for t in ts_d])

    n_legend = len(legend_handles)
    fig.legend(
        legend_handles.values(), legend_handles.keys(),
        loc="lower center",
        ncol=min(n_legend, 5),
        fontsize=8.5,
        frameon=True,
    )
    fig.suptitle(f"{dataset_name}: Metrics Overview", fontsize=13)
    bottom_margin = 0.04 + 0.025 * max(0, (n_legend - 1) // 5)
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate metric visualizations from JSON files"
    )
    parser.add_argument(
        "--metrics", nargs="+", type=Path, required=True,
        help="Paths to metrics JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/pless_mbpp_results/analysis/figures"),
        help="Output directory for figures (default: results/pless_mbpp_results/analysis/figures/)",
    )
    parser.add_argument(
        "--dataset", type=str, default="MBPP",
        help="Dataset name used in plot titles (default: MBPP)",
    )
    parser.add_argument(
        "--methods", nargs="+",
        help="Filter to specific methods (e.g. pless pless_norm temp top_p0.9)",
    )
    parser.add_argument(
        "--config-keys", nargs="+",
        help="Filter the headline Pareto scatter to specific config keys "
             "(e.g. pless_t0.6 temp_t0.3 top_p_p0.8_t1.0). "
             "Format: method[_p<top_p>][_k<top_k>][_t<temperature>].",
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Filter to specific models by HF id (e.g. Qwen/Qwen-7B meta-llama/Llama-2-7b-hf).",
    )
    parser.add_argument(
        "--no-trajectories", action="store_true",
        help="Suppress per-model gray trajectory lines in the headline Pareto scatter.",
    )
    parser.add_argument(
        "--pareto-only", action="store_true",
        help="Only generate the headline Pareto scatter; skip aggregate lines, "
             "heatmaps, bars, and sub-component paretos.",
    )
    parser.add_argument(
        "--pareto-output-name", default="pareto_correctness_diversity.png",
        help="Filename for the headline Pareto PNG (default: pareto_correctness_diversity.png).",
    )
    parser.add_argument(
        "--family-palette", action="store_true",
        help="Collapse temp/top_p/top_k/beam variants to one color per family in the "
             "headline Pareto scatter; p-less variants stay distinct by temperature.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_list = load_metrics(args.metrics)

    if args.methods:
        allowed = set(args.methods)
        metrics_list = [m for m in metrics_list if m["method"] in allowed]

    if args.models:
        allowed_models = set(args.models)
        metrics_list = [m for m in metrics_list if m["model"] in allowed_models]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Determine how many distinct models are present
    n_models = len(dict.fromkeys(m["model"] for m in metrics_list))

    if not args.pareto_only:
        if n_models > 2:
            # Many configs — use the faceted (one-row-per-model) layout
            plot_aggregate_lines_faceted(
                metrics_list, out / "aggregate_lines_all.png", dataset_name=args.dataset,
            )
            print(f"Saved {out / 'aggregate_lines_all.png'}")

            # Also produce a pless-only faceted plot (Fix F: use faceted layout, not flat)
            pless_only = [
                m for m in metrics_list
                if m["method"] in ("pless", "pless_norm", "p_less", "p_less_norm")
            ]
            if pless_only:
                plot_aggregate_lines_faceted(
                    pless_only, out / "aggregate_lines_pless.png", dataset_name=args.dataset,
                )
                print(f"Saved {out / 'aggregate_lines_pless.png'}")
        else:
            plot_aggregate_lines(
                metrics_list, out / "aggregate_lines.png", dataset_name=args.dataset,
            )
            print(f"Saved {out / 'aggregate_lines.png'}")

        plot_correctness_vs_diversity(
            metrics_list, out / "correctness_vs_diversity.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'correctness_vs_diversity.png'}")

    # Primary: Pareto scatter — aggregate correctness vs diversity trade-off.
    # Headline figure honors --config-keys and --no-trajectories.
    plot_pareto_scatter(
        metrics_list, out / args.pareto_output_name,
        dataset_name=args.dataset,
        config_keys=args.config_keys,
        show_trajectories=not args.no_trajectories,
        family_palette=args.family_palette,
    )
    print(f"Saved {out / args.pareto_output_name}")

    if args.pareto_only:
        return

    # Per-subcomponent CodeBLEU pareto plots
    _SUBCOMPONENT_PARETOS = [
        ("ngram_match_diversity",          "pareto_ngram_diversity.png"),
        ("weighted_ngram_match_diversity", "pareto_weighted_ngram_diversity.png"),
        ("syntax_match_diversity",         "pareto_syntax_diversity.png"),
        ("dataflow_match_diversity",       "pareto_dataflow_diversity.png"),
    ]
    has_codebleu = any("codebleu_diversity" in m for m in metrics_list)
    if has_codebleu:
        for div_key, filename in _SUBCOMPONENT_PARETOS:
            plot_pareto_scatter(
                metrics_list, out / filename,
                dataset_name=args.dataset,
                diversity_key=div_key,
            )
            print(f"Saved {out / filename}")

    # Secondary: Heatmaps — model × method grid for pass@1 and diversity
    plot_method_heatmaps(
        metrics_list, out / "method_heatmaps.png",
        dataset_name=args.dataset,
    )
    print(f"Saved {out / 'method_heatmaps.png'}")

    # Structural diversity plots (only if data includes the new metrics)
    has_diversity = any("structural_diversity" in m for m in metrics_list)
    if has_diversity:
        plot_structural_diversity_bars(
            metrics_list, out / "structural_diversity_bars.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'structural_diversity_bars.png'}")

        plot_pairwise_distance_distributions(
            metrics_list, out / "pairwise_distance_distributions.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'pairwise_distance_distributions.png'}")

    # CodeBLEU diversity bar chart (only if data includes the new metrics)
    has_codebleu = any("codebleu_diversity" in m for m in metrics_list)
    if has_codebleu:
        plot_diversity_metrics_bars(
            metrics_list, out / "diversity_metrics_comparison.png",
            dataset_name=args.dataset,
        )
        print(f"Saved {out / 'diversity_metrics_comparison.png'}")

    # pass@1 comparison heatmap (model × sampler, single panel)
    plot_pass_at_1_heatmap(
        metrics_list, out / "pass_at_1_comparison.png", dataset_name=args.dataset,
    )
    print(f"Saved {out / 'pass_at_1_comparison.png'}")

    # Metrics overview with tight y-limits (zoomed faceted line plots)
    if n_models >= 1:
        plot_metrics_overview(
            metrics_list, out / "metrics_overview.png", dataset_name=args.dataset,
        )
        print(f"Saved {out / 'metrics_overview.png'}")


if __name__ == "__main__":
    main()
