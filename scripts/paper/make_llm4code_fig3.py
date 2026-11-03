"""Generate Fig 3a (pass@k curves per sampler) and Fig 3b (pass@10-pass@1 lift bars per model) for §3.

Reads the 7 MBPP models × 6 canonical configs used in Table 3A. Averages
pass@k across models for Fig 3a; per-model lift bars for Fig 3b.

Usage:
    uv run python scripts/paper/make_llm4code_fig3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR_CODE = REPO_ROOT / "paper" / "llm4code" / "figures"
FIG_DIR_PAPER = REPO_ROOT.parent / "pless-for-coding-paper" / "figures"

# 7 headline MBPP models × 6 canonical configs — mirrors extractor's MBPP dict.
# (model_short_label, {config_label: metrics_json_path})
MBPP_CELLS = [
    ("CodeLlama-7b", {
        "greedy":         "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/greedy_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/beam4_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/temp_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/top_p0.9_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/pless_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/pless_norm_t0.7_metrics.json",
    }),
    ("CodeLlama-Inst", {
        "greedy":         "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/greedy_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/beam4_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p0.9_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t0.7_metrics.json",
    }),
    ("OCI-1.3B", {
        "greedy":         "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/greedy_bigcode_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/beam4_bigcode_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/temp_bigcode_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/top_p0.9_bigcode_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/pless_bigcode_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/pless_norm_bigcode_t0.7_metrics.json",
    }),
    ("Llama-2-7b", {
        "greedy":         "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/greedy_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/beam4_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/temp_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/top_p0.9_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/pless_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/pless_norm_t0.7_metrics.json",
    }),
    ("Llama-2-7b-chat", {
        "greedy":         "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/greedy_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/beam4_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/temp_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/top_p0.9_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/pless_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/pless_norm_t0.7_metrics.json",
    }),
    ("Q2.5-Coder-1.5B", {
        "greedy":         "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/greedy_bigcode_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/beam4_bigcode_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/temp_bigcode_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/top_p0.9_bigcode_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_bigcode_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_norm_bigcode_t0.7_metrics.json",
    }),
    ("Q2.5-Coder-3B", {
        "greedy":         "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/greedy_bigcode_t1.0_metrics.json",
        "beam4":          "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/beam4_bigcode_t1.0_metrics.json",
        "temp@0.7":       "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/temp_bigcode_t0.7_metrics.json",
        "top-p@0.9":      "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/top_p0.9_bigcode_t1.0_metrics.json",
        "p-less":         "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_bigcode_t0.7_metrics.json",
        "p-less-norm":    "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_norm_bigcode_t0.7_metrics.json",
    }),
]

CONFIG_ORDER = ["greedy", "beam4", "temp@0.7", "top-p@0.9", "p-less", "p-less-norm"]
CONFIG_COLORS = {
    "greedy":      "#666666",  # gray
    "beam4":       "#888888",  # lighter gray
    "temp@0.7":    "#1f77b4",  # matplotlib blue
    "top-p@0.9":   "#ff7f0e",  # matplotlib orange
    "p-less":      "#d62728",  # matplotlib red
    "p-less-norm": "#e377c2",  # matplotlib pink
}
# Display text for figure legends only (CONFIG_ORDER / CONFIG_COLORS keys are
# unchanged -- still used for data lookup and color lookup throughout).
# top-p@0.9 is genuinely a T=1.0 config (see top_p0.9_bigcode_t1.0_metrics.json
# above); p-less/p-less-norm here are both T=0.7 (pless[_norm]_*t0.7*).
DISPLAY_LABEL = {
    "beam4":       "beam@4",
    "temp@0.7":    "T@0.7",
    "top-p@0.9":   "top-p@0.9 with T=1.0",
    "p-less":      "p-less with T=0.7",
    "p-less-norm": "p-less-norm with T=0.7",
}
KS = [1, 3, 5, 10]


def _passk(path: Path, k: int) -> float | None:
    if not path.exists(): return None
    d = json.loads(path.read_text())
    v = d.get("pass_at_k", {}).get(str(k))
    return v if v is not None else None


def _make_fig3a(log_scale: bool = False, out_name: str = "fig3a_passk_curves_mbpp.png") -> None:
    """Pass@k curves per sampler, averaged over 7 MBPP models.

    Linear x-axis (default) chosen over log after user review: with k in
    {1,3,5,10} (only a 10x range), log-scaling made the 1->3 gap look
    disproportionately large relative to 3->5 and 5->10 -- a real, verified
    visual artifact of log-scaling this irregular 4-point set, not a bug,
    but one the user preferred to avoid. Linear spacing reflects the actual
    numeric k-differences instead (5->10 is now the largest gap, matching
    the true magnitude difference of 5 vs. 2 for the other two gaps).

    Legend moved outside the axes (below the plot) rather than any in-plot
    corner. "Lower right" visually overlapped the greedy/beam4 flat lines
    and their k=10 markers; a follow-up "upper left" placement still came
    close to touching temp@0.7's steep rise at k=3 (y~47, near the legend's
    bottom-right corner). Placing the legend fully outside the data area
    is the only placement that's robust to curve shape by construction.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for cfg in CONFIG_ORDER:
        vals_per_k = []
        for k in KS:
            pks = [_passk(REPO_ROOT / cells[cfg], k) for _, cells in MBPP_CELLS if cfg in cells]
            pks = [v * 100 for v in pks if v is not None]
            vals_per_k.append(mean(pks) if pks else None)
        if all(v is None for v in vals_per_k): continue
        # p-less-norm overlaps p-less near-perfectly (within ~0.5 pp per Table 3A);
        # use a dashed stroke so both are visible without one hiding the other.
        linestyle = "--" if cfg == "p-less-norm" else "-"
        ax.plot(KS, vals_per_k, marker="o", label=DISPLAY_LABEL.get(cfg, cfg), color=CONFIG_COLORS[cfg], linewidth=1.5, markersize=5, linestyle=linestyle)
    if log_scale:
        ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("$k$ in pass@$k$")
    ax.set_ylabel("pass@k (%), mean over 7 MBPP models")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=True)
    fig.tight_layout()
    for out in (FIG_DIR_CODE, FIG_DIR_PAPER):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / out_name, dpi=180, bbox_inches="tight")
    print(f"wrote {out_name}", file=sys.stderr)


def _make_fig3b() -> None:
    """Grouped bars: pass@10 - pass@1 lift per (model, sampler)."""
    stochastic_cfgs = ["temp@0.7", "top-p@0.9", "p-less", "p-less-norm"]  # skip deterministic
    n_models = len(MBPP_CELLS)
    n_cfgs = len(stochastic_cfgs)
    bar_width = 0.18
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, cfg in enumerate(stochastic_cfgs):
        lifts = []
        for _, cells in MBPP_CELLS:
            p = REPO_ROOT / cells[cfg] if cfg in cells else None
            p1 = _passk(p, 1) if p else None
            p10 = _passk(p, 10) if p else None
            lifts.append((p10 - p1) * 100 if p1 is not None and p10 is not None else 0.0)
        offset = (i - (n_cfgs - 1) / 2) * bar_width
        ax.bar(x + offset, lifts, bar_width, label=DISPLAY_LABEL.get(cfg, cfg), color=CONFIG_COLORS[cfg])

    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in MBPP_CELLS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("pass@10 $-$ pass@1 (pp)")
    ax.grid(True, alpha=0.3, axis="y")
    # Legend moved outside (below) rather than "upper right" in-plot -- same
    # overlap-robustness rationale as fig3a: guaranteed clear of bars by
    # construction, regardless of bar heights per model.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncol=4, frameon=True)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    for out in (FIG_DIR_CODE, FIG_DIR_PAPER):
        fig.savefig(out / "fig3b_lift_bars_mbpp.png", dpi=180, bbox_inches="tight")
    print("wrote fig3b_lift_bars_mbpp.png", file=sys.stderr)


def main() -> None:
    _make_fig3a()
    _make_fig3b()


if __name__ == "__main__":
    main()
