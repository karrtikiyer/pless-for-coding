"""Generate CSV, markdown report, and visualizations for the temperature sweep.

Loads all metrics JSONs from temprature_results/*/metrics/ and produces:
  - temperature_sweep_summary.csv
  - temperature_sweep_report.md
  - figures/ directory with plots

Usage:
    python -m bench.eval.report_temperature_sweep
"""

import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path("results/pless_human_eval_results/temprature_results")
FIGURES_DIR = RESULTS_ROOT / "analysis" / "figures"

# ---------------------------------------------------------------------------
# Method colours and styles for temperature sweep plots
# ---------------------------------------------------------------------------
_METHOD_COLORS = {
    "temp": "#2F855A",       # green — standard temperature
    "pless": "#6B46C1",      # purple — p-less
    "pless_norm": "#B7791F",  # gold — p-less normalized
}

_METHOD_MARKERS = {
    "temp": "s",
    "pless": "o",
    "pless_norm": "X",
}

_METHOD_LINESTYLES = {
    "temp": "-",
    "pless": "-",
    "pless_norm": "-.",
}

_MODEL_SHORT_NAMES = {
    "Qwen/Qwen2.5-Coder-7B": "Qwen2.5-Coder-7B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder-30B",
    "codellama/CodeLlama-7b-hf": "CodeLlama-7b",
    "mistralai/Codestral-22B-v0.1": "Codestral-22B",
}


def _short_model(model: str) -> str:
    return _MODEL_SHORT_NAMES.get(model, model.split("/")[-1])


def _method_base(method: str) -> str:
    """Normalize method name: 'pless' / 'pless_norm' / 'temp'."""
    if method == "pless_norm":
        return "pless_norm"
    if method == "pless":
        return "pless"
    if method.startswith("temp"):
        return "temp"
    return method


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics(root: Path) -> list[dict]:
    """Load all metrics JSON files from root/*/metrics/."""
    metrics = []
    for p in sorted(root.glob("*/metrics/*_metrics.json")):
        with open(p) as f:
            metrics.append(json.load(f))
    return metrics


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(metrics_list: list[dict], output_path: Path) -> None:
    fieldnames = [
        "Model", "Method", "Temperature",
        "pass@1", "pass@3", "pass@5", "pass@10",
        "cover@0.5", "cover@0.7",
        "structural_diversity",
    ]
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": _short_model(m["model"]),
            "Method": m["method"],
            "Temperature": m["temperature"],
            "pass@1": f"{m['pass_at_k'].get('1', 0) * 100:.1f}",
            "pass@3": f"{m['pass_at_k'].get('3', 0) * 100:.1f}",
            "pass@5": f"{m['pass_at_k'].get('5', 0) * 100:.1f}",
            "pass@10": f"{m['pass_at_k'].get('10', 0) * 100:.1f}",
            "cover@0.5": f"{m['cover_at_t'].get('0.5', 0):.1f}",
            "cover@0.7": f"{m['cover_at_t'].get('0.7', 0):.1f}",
            "structural_diversity": f"{m.get('structural_diversity', 0):.4f}",
        })

    # Sort by model, method, temperature
    rows.sort(key=lambda r: (r["Model"], r["Method"], float(r["Temperature"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(metrics_list: list[dict], output_path: Path) -> None:
    models = sorted(set(_short_model(m["model"]) for m in metrics_list))
    methods = sorted(set(m["method"] for m in metrics_list))

    # Index for quick lookup
    idx = {}
    for m in metrics_list:
        key = (_short_model(m["model"]), m["method"], m["temperature"])
        idx[key] = m

    lines = []
    lines.append("# HumanEval Temperature Sweep Results\n")
    lines.append(f"**Models:** {', '.join(models)}  ")
    lines.append(f"**Methods:** {', '.join(methods)}  ")
    lines.append(f"**Temperatures:** 0.7, 1.0, 1.5, 2.0, 2.5, 3.0  ")
    lines.append(f"**Samples per task:** 10\n")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Model | Method | Temp | pass@1 | pass@5 | pass@10 | cover@0.5 | diversity |")
    lines.append("|-------|--------|------|--------|--------|---------|-----------|-----------|")

    sorted_metrics = sorted(
        metrics_list,
        key=lambda m: (_short_model(m["model"]), _method_base(m["method"]), m["temperature"]),
    )
    for m in sorted_metrics:
        p1 = m["pass_at_k"].get("1", 0) * 100
        p5 = m["pass_at_k"].get("5", 0) * 100
        p10 = m["pass_at_k"].get("10", 0) * 100
        c05 = m["cover_at_t"].get("0.5", 0)
        sd = m.get("structural_diversity", 0)
        lines.append(
            f"| {_short_model(m['model'])} | {m['method']} | {m['temperature']} "
            f"| {p1:.1f}% | {p5:.1f}% | {p10:.1f}% | {c05:.1f}% | {sd:.4f} |"
        )

    # Per-model analysis
    lines.append("\n## Per-Model Analysis\n")
    for model in models:
        lines.append(f"### {model}\n")
        lines.append("| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |")
        lines.append("|--------|-------|-------|-------|-------|-------|-------|")

        for method in methods:
            row = f"| {method}"
            for temp in [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
                key = (model, method, temp)
                if key in idx:
                    p1 = idx[key]["pass_at_k"].get("1", 0) * 100
                    row += f" | {p1:.1f}%"
                else:
                    row += " | —"
            row += " |"
            lines.append(row)
        lines.append("")

    # Key findings
    lines.append("## Key Findings\n")

    # Find best method per model at T=2.0 (highest temp with non-trivial results)
    lines.append("### Best Method at T=2.0\n")
    for model in models:
        best_method, best_p1 = None, -1
        for method in methods:
            key = (model, method, 2.0)
            if key in idx:
                p1 = idx[key]["pass_at_k"].get("1", 0) * 100
                if p1 > best_p1:
                    best_p1 = p1
                    best_method = method
        if best_method:
            lines.append(f"- **{model}**: **{best_method}** (pass@1={best_p1:.1f}%)")

    # Temperature robustness for pless methods: drop from T=0.7 to T=2.0
    lines.append("\n### Temperature Robustness (pass@1 drop from T=0.7 to T=2.0)\n")
    pless_methods = [m for m in methods if m in ("pless", "pless_norm")]
    for model in models:
        for method in pless_methods:
            key07 = (model, method, 0.7)
            key20 = (model, method, 2.0)
            if key07 in idx and key20 in idx:
                p1_07 = idx[key07]["pass_at_k"].get("1", 0) * 100
                p1_20 = idx[key20]["pass_at_k"].get("1", 0) * 100
                drop = p1_07 - p1_20
                lines.append(f"- {model} / {method}: {p1_07:.1f}% → {p1_20:.1f}% (Δ={drop:+.1f}pp)")

    # Temp baseline comparison at T=0.7 and T=1.0
    lines.append("\n### Standard Temperature Baselines (T=0.7 and T=1.0)\n")
    for model in models:
        for temp in [0.7, 1.0]:
            key_temp = (model, "temp", temp)
            if key_temp not in idx:
                continue
            p1_temp = idx[key_temp]["pass_at_k"].get("1", 0) * 100
            best_pless = None
            for method in pless_methods:
                key = (model, method, temp)
                if key in idx:
                    p1 = idx[key]["pass_at_k"].get("1", 0) * 100
                    if best_pless is None or p1 > best_pless[1]:
                        best_pless = (method, p1)
            if best_pless:
                diff = best_pless[1] - p1_temp
                lines.append(
                    f"- {model} T={temp}: temp={p1_temp:.1f}%, "
                    f"best pless ({best_pless[0]})={best_pless[1]:.1f}% (Δ={diff:+.1f}pp)"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _group_by_model(metrics_list: list[dict]) -> dict[str, list[dict]]:
    """Group metrics by model, preserving order."""
    models_order = list(dict.fromkeys(m["model"] for m in metrics_list))
    by_model = {model: [] for model in models_order}
    for m in metrics_list:
        by_model[m["model"]].append(m)
    return by_model


def plot_pass_at_k_vs_temperature(metrics_list: list[dict], output_dir: Path) -> None:
    """Line plot: pass@1 vs temperature, one subplot per model, one line per method."""
    by_model = _group_by_model(metrics_list)
    n_models = len(by_model)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for col, (model, model_metrics) in enumerate(by_model.items()):
        ax = axes[col]

        # Group by method base
        by_method: dict[str, list[dict]] = {}
        for m in model_metrics:
            mb = _method_base(m["method"])
            by_method.setdefault(mb, []).append(m)

        for method, mlist in sorted(by_method.items()):
            mlist.sort(key=lambda m: m["temperature"])
            temps = [m["temperature"] for m in mlist]
            p1s = [m["pass_at_k"].get("1", 0) * 100 for m in mlist]

            ax.plot(
                temps, p1s,
                label=method,
                color=_METHOD_COLORS.get(method, "#333"),
                marker=_METHOD_MARKERS.get(method, "x"),
                linestyle=_METHOD_LINESTYLES.get(method, "-"),
                linewidth=2, markersize=7,
            )

        ax.set_title(_short_model(model), fontsize=11)
        ax.set_xlabel("Temperature")
        ax.set_xticks([0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("pass@1 (%)")

    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle("HumanEval: pass@1 vs Temperature", fontsize=13)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pass_at_1_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_structural_diversity_vs_temperature(metrics_list: list[dict], output_dir: Path) -> None:
    """Line plot: structural diversity vs temperature, one subplot per model."""
    by_model = _group_by_model(metrics_list)
    n_models = len(by_model)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for col, (model, model_metrics) in enumerate(by_model.items()):
        ax = axes[col]

        by_method: dict[str, list[dict]] = {}
        for m in model_metrics:
            mb = _method_base(m["method"])
            by_method.setdefault(mb, []).append(m)

        for method, mlist in sorted(by_method.items()):
            mlist.sort(key=lambda m: m["temperature"])
            temps = [m["temperature"] for m in mlist]
            divs = [m.get("structural_diversity", 0) for m in mlist]

            ax.plot(
                temps, divs,
                label=method,
                color=_METHOD_COLORS.get(method, "#333"),
                marker=_METHOD_MARKERS.get(method, "x"),
                linestyle=_METHOD_LINESTYLES.get(method, "-"),
                linewidth=2, markersize=7,
            )

        ax.set_title(_short_model(model), fontsize=11)
        ax.set_xlabel("Temperature")
        ax.set_xticks([0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("Structural Diversity")

    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle("HumanEval: Structural Diversity vs Temperature", fontsize=13)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "structural_diversity_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_heatmaps(metrics_list: list[dict], output_dir: Path) -> None:
    """Heatmap: models × temperatures, one heatmap per method, cell color = pass@1."""
    methods = sorted(set(_method_base(m["method"]) for m in metrics_list))
    models = sorted(set(m["model"] for m in metrics_list))
    temps = sorted(set(m["temperature"] for m in metrics_list))

    # Index
    idx = {}
    for m in metrics_list:
        idx[(_method_base(m["method"]), m["model"], m["temperature"])] = m

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, max(3, len(models) * 0.8 + 1)),
                             squeeze=False)
    axes = axes[0]

    for col, method in enumerate(methods):
        ax = axes[col]
        data = np.zeros((len(models), len(temps)))

        for i, model in enumerate(models):
            for j, temp in enumerate(temps):
                key = (method, model, temp)
                if key in idx:
                    data[i, j] = idx[key]["pass_at_k"].get("1", 0) * 100

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=80)
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([str(t) for t in temps], fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_short_model(m) for m in models], fontsize=9)
        ax.set_xlabel("Temperature")
        ax.set_title(method, fontsize=11)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(temps)):
                val = data[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.suptitle("HumanEval: pass@1 (%) by Model × Temperature", fontsize=13)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="pass@1 (%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pass_at_1_heatmap.png", dpi=150)
    plt.close(fig)


def plot_pass_at_k_curves_by_temperature(metrics_list: list[dict], output_dir: Path) -> None:
    """Faceted pass@k curves: one row per model, one column per method.
    Lines colored by temperature."""
    by_model = _group_by_model(metrics_list)
    methods = sorted(set(_method_base(m["method"]) for m in metrics_list))
    n_models = len(by_model)
    n_methods = len(methods)

    # Temperature color map
    temps_all = sorted(set(m["temperature"] for m in metrics_list))
    cmap = plt.cm.coolwarm
    temp_colors = {t: cmap(i / max(len(temps_all) - 1, 1)) for i, t in enumerate(temps_all)}

    fig, axes = plt.subplots(n_models, n_methods,
                             figsize=(5 * n_methods, 4 * n_models),
                             sharey=True, squeeze=False)

    for row, (model, model_metrics) in enumerate(by_model.items()):
        by_method: dict[str, list[dict]] = {}
        for m in model_metrics:
            mb = _method_base(m["method"])
            by_method.setdefault(mb, []).append(m)

        for col, method in enumerate(methods):
            ax = axes[row, col]
            mlist = by_method.get(method, [])
            mlist.sort(key=lambda m: m["temperature"])

            for m in mlist:
                ks = sorted(m["pass_at_k"], key=lambda x: int(x))
                ax.plot(
                    [int(k) for k in ks],
                    [m["pass_at_k"][k] * 100 for k in ks],
                    label=f"T={m['temperature']}",
                    color=temp_colors[m["temperature"]],
                    marker="o", linewidth=1.5, markersize=5,
                )

            ax.set_xticks([1, 3, 5, 10])
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(method, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{_short_model(model)}\n\npass@k (%)", fontsize=9)
            if row == n_models - 1:
                ax.set_xlabel("k")

    # Legend from the last subplot
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(temps_all),
               fontsize=8, frameon=True)

    fig.suptitle("HumanEval: pass@k Curves by Temperature", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pass_at_k_by_temperature.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics_list = load_all_metrics(RESULTS_ROOT)
    if not metrics_list:
        print(f"No metrics JSONs found under {RESULTS_ROOT}/*/metrics/")
        print("Run `python -m bench.eval.eval_temperature_sweep` first.")
        return

    print(f"Loaded {len(metrics_list)} metrics files")

    # CSV
    csv_path = RESULTS_ROOT / "analysis" / "temperature_sweep_summary.csv"
    write_csv(metrics_list, csv_path)
    print(f"Wrote {csv_path}")

    # Markdown report
    md_path = RESULTS_ROOT / "analysis" / "temperature_sweep_report.md"
    write_markdown(metrics_list, md_path)
    print(f"Wrote {md_path}")

    # Visualizations
    print("Generating figures...")
    plot_pass_at_k_vs_temperature(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'pass_at_1_vs_temperature.png'}")

    plot_structural_diversity_vs_temperature(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'structural_diversity_vs_temperature.png'}")

    plot_heatmaps(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'pass_at_1_heatmap.png'}")

    plot_pass_at_k_curves_by_temperature(metrics_list, FIGURES_DIR)
    print(f"  → {FIGURES_DIR / 'pass_at_k_by_temperature.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
