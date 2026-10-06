"""Generate reasoning-model analogues of Fig 3a and Fig 3b for §5.

Fig 6a: pass@k curves per sampler on APPS-252 (2 panels: Qwen3-8B, DeepSeek).
Fig 6b: pass@10 - pass@1 lift bars per sampler per model.

Configs shown per model: p-less (k=2), temp, top-p, top-k, best-G_k.

Usage:
    uv run python scripts/paper/make_llm4code_fig6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR_CODE = REPO_ROOT / "paper" / "llm4code" / "figures"
FIG_DIR_PAPER = REPO_ROOT.parent / "pless-for-coding-paper" / "figures"

# (model, {config_label: metrics_json_path})
APPS_CELLS = [
    # Option C: 4 lines per panel — p-less, p-less-norm, model-recommended, best-k G_k
    # Qwen3-8B recommended (per model card, thinking mode): T=0.6, top_p=0.95, top_k=20
    # DeepSeek-R1-Distill recommended (per model card): T=0.6, top_p=0.95
    ("Qwen3-8B", {
        "p-less":            "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json",
        "p-less-norm":       "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_norm_think_t1.0_t1.0_metrics.json",
        "recommended":       "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/temp_p0.95_k20_think_t0.6_t0.6_metrics.json",
        "best top-p (swept)": "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/temp_p0.9_think_t1.0_t1.0_metrics.json",
        "G_k=0.1":           "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.1_t1.0_metrics.json",
    }),
    ("DeepSeek-R1-Distill-Llama-8B", {
        "p-less":            "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_think_t1.0_t1.0_metrics.json",
        "p-less-norm":       "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_norm_think_t1.0_t1.0_metrics.json",
        "recommended":       "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.95_think_t0.6_t0.6_metrics.json",
        "best top-p (swept)": "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.85_think_t1.0_t1.0_metrics.json",
        "G_k=0.35":          "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.35_t1.0_metrics.json",
    }),
]

CONFIG_COLORS = {
    "p-less": "#d62728",  # red
    "p-less-norm":  "#e377c2",  # pink
    "recommended":  "#ff7f0e",  # orange (model-recommended config)
    "G_k=0.1":      "#1f77b4",  # blue
    "G_k=0.35":     "#1f77b4",  # blue (same colour for G_k across models)
    "best top-p (swept)": "#2ca02c",  # green
}
CONFIG_ORDER = ["p-less", "p-less-norm", "top-p @0.95", "top-k @20"]  # G_k label varies per model
KS = [1, 3, 5, 10]


def _passk(path: Path, k: int) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("pass_at_k", {}).get(str(k))


def _make_fig6a() -> None:
    """pass@k curves per sampler on APPS-252, 2 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for i, (model, cells) in enumerate(APPS_CELLS):
        ax = axes[i]
        for label, rel in cells.items():
            path = REPO_ROOT / rel
            ys = [(_passk(path, k) or 0.0) * 100 for k in KS]
            colour = CONFIG_COLORS.get(label, "#666666")
            marker = "s" if label.startswith("G_k") else "o"
            ax.plot(KS, ys, marker=marker, label=label, color=colour, linewidth=1.5, markersize=5)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("$k$ in pass@$k$")
        if i == 0:
            ax.set_ylabel("pass@k (%)")
        short = model.replace("-R1-Distill-Llama-8B", "-R1-Distill")
        ax.set_title(short)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right", frameon=True)
    fig.tight_layout()
    for out in (FIG_DIR_CODE, FIG_DIR_PAPER):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig6a_reasoning_passk_curves.png", dpi=180, bbox_inches="tight")
    print("wrote fig6a_reasoning_passk_curves.png", file=sys.stderr)


def _make_fig6b() -> None:
    """Grouped bars: pass@10 - pass@1 lift per sampler per model."""
    n_models = len(APPS_CELLS)
    all_labels = list(APPS_CELLS[0][1].keys())  # 5 configs
    # For DeepSeek the G_k label differs; use position (last config) rather than name
    n_cfgs = len(all_labels)
    bar_width = 0.15
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    # Iterate by config position (aligns Qwen G_k=0.1 with DeepSeek G_k=0.35 in the last slot)
    for i in range(n_cfgs):
        lifts = []
        display_label = None
        for j, (_, cells) in enumerate(APPS_CELLS):
            lbl_j = list(cells.keys())[i]
            path = REPO_ROOT / cells[lbl_j]
            p1 = _passk(path, 1)
            p10 = _passk(path, 10)
            lifts.append((p10 - p1) * 100 if p1 is not None and p10 is not None else 0.0)
            if j == 0:
                display_label = lbl_j
        offset = (i - (n_cfgs - 1) / 2) * bar_width
        # For the last bar (G_k), use a combined label
        legend_label = display_label if not display_label.startswith("G_k") else "G_k (best k per model)"
        ax.bar(x + offset, lifts, bar_width, label=legend_label, color=CONFIG_COLORS.get(display_label, "#666666"))

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-R1-Distill-Llama-8B", "-R1-Distill") for m, _ in APPS_CELLS], fontsize=9)
    ax.set_ylabel("pass@10 $-$ pass@1 (pp)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="upper right", frameon=True, ncols=2)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    for out in (FIG_DIR_CODE, FIG_DIR_PAPER):
        fig.savefig(out / "fig6b_reasoning_lift_bars.png", dpi=180, bbox_inches="tight")
    print("wrote fig6b_reasoning_lift_bars.png", file=sys.stderr)


def main() -> None:
    _make_fig6a()
    _make_fig6b()


if __name__ == "__main__":
    main()
