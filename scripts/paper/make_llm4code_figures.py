"""Generate Fig 5A (pass@k vs k) and Fig 5B (truncation vs k) for §5.

Reads metrics JSONs from ``results/_renyi_sweep_full252/`` for Qwen3-8B and
DeepSeek-R1-Distill-Llama-8B, plus the p-less α=2 baselines. Produces
PNG figures under ``paper/llm4code/figures/`` and copies to the paper repo.

Usage:
    uv run python scripts/paper/make_llm4code_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR_CODE = REPO_ROOT / "paper" / "llm4code" / "figures"
FIG_DIR_PAPER = REPO_ROOT.parent / "pless-for-coding-paper" / "figures"


# k-sweeps per model. Baseline (α=2) plotted as a dashed horizontal line.
# Values must be sorted; matplotlib plots them in order.
QWEN3_KS = [0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.8, 1.6]
DEEPSEEK_KS = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.8, 1.6]

RENYI_DIR = {
    "Qwen3-8B": "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview",
    "DeepSeek-R1-Distill-Llama-8B": "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview",
}

BASELINE_PATH = {
    "Qwen3-8B": "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl",
    "DeepSeek-R1-Distill-Llama-8B": "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl",
}
BASELINE_METRICS = {
    "Qwen3-8B": "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json",
    "DeepSeek-R1-Distill-Llama-8B": "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_think_t1.0_t1.0_metrics.json",
}


def _renyi_metrics_path(model: str, k: float) -> Path:
    slug = "" if str(k).rstrip("0").rstrip(".") == "" else str(k)
    return REPO_ROOT / RENYI_DIR[model] / "metrics" / f"pless_renyi_think_t1.0_k{k}_t1.0_metrics.json"


def _passk_from_metrics(path: Path, key: str = "1") -> float | None:
    d = json.loads(path.read_text())
    v = d.get("pass_at_k", {}).get(str(key))
    return v if v is not None else None


def _truncation_from_jsonl(path: Path) -> float | None:
    truncated = 0
    total = 0
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            swt = d.get("samples_with_thinking") or []
            for s in swt:
                text = s if isinstance(s, str) else (s.get("text") if isinstance(s, dict) else "")
                total += 1
                if "</think>" not in (text or ""):
                    truncated += 1
    return 100.0 * truncated / total if total else None


def _renyi_jsonl_path(model: str, k: float) -> Path:
    return REPO_ROOT / RENYI_DIR[model] / f"pless_renyi_think_t1.0_k{k}_t1.0.jsonl"


def _collect(model: str, ks: list[float]) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (ks_present, pass1, pass10, truncation%) for k values present on disk."""
    ks_present = []
    p1 = []
    p10 = []
    tr = []
    for k in ks:
        mp = _renyi_metrics_path(model, k)
        jp = _renyi_jsonl_path(model, k)
        if not mp.exists() or not jp.exists():
            print(f"GAP: {model} k={k} — missing metrics or JSONL", file=sys.stderr)
            continue
        ks_present.append(k)
        p1.append(100.0 * (_passk_from_metrics(mp, "1") or 0.0))
        p10.append(100.0 * (_passk_from_metrics(mp, "10") or 0.0))
        tr.append(_truncation_from_jsonl(jp) or 0.0)
    return ks_present, p1, p10, tr


def _baseline(model: str) -> tuple[float, float, float]:
    mp = REPO_ROOT / BASELINE_METRICS[model]
    jp = REPO_ROOT / BASELINE_PATH[model]
    return (
        100.0 * (_passk_from_metrics(mp, "1") or 0.0),
        100.0 * (_passk_from_metrics(mp, "10") or 0.0),
        _truncation_from_jsonl(jp) or 0.0,
    )


def main() -> None:
    FIG_DIR_CODE.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PAPER.mkdir(parents=True, exist_ok=True)

    fig5a, ax5a = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    fig5b, ax5b = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)

    for i, (model, ks) in enumerate([("Qwen3-8B", QWEN3_KS), ("DeepSeek-R1-Distill-Llama-8B", DEEPSEEK_KS)]):
        ks_p, p1, p10, tr = _collect(model, ks)
        bp1, bp10, btr = _baseline(model)

        # Fig 5A: pass@1 and pass@10 vs k. Only first panel carries legend labels;
        # shared legend rendered below both panels via fig.legend after the loop.
        ax = ax5a[i]
        ax.plot(ks_p, p1, marker="o", color="C0", label=("pass@1" if i == 0 else None))
        ax.plot(ks_p, p10, marker="s", color="C1", label=("pass@10" if i == 0 else None))
        ax.axhline(bp1, linestyle="--", color="C0", alpha=0.5, linewidth=0.9,
                   label=("$k{=}2$ (p-less) pass@1" if i == 0 else None))
        ax.axhline(bp10, linestyle="--", color="C1", alpha=0.5, linewidth=0.9,
                   label=("$k{=}2$ (p-less) pass@10" if i == 0 else None))
        ax.set_xscale("log")
        # Sparse round-number ticks; data markers show all measured k values
        sparse_ticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        ax.set_xticks(sparse_ticks)
        ax.set_xticklabels([f"{k:g}" for k in sparse_ticks], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("k (Rényi order)")
        ax.set_title(model.replace("-R1-Distill-Llama-8B", "-R1-Distill"))
        if i == 0:
            ax.set_ylabel("pass@k (%)")
        ax.grid(True, alpha=0.3)

        # Fig 5B: truncation % vs k
        ax = ax5b[i]
        ax.plot(ks_p, tr, marker="o", color="C3",
                label=("truncation" if i == 0 else None))
        ax.axhline(btr, linestyle="--", color="C3", alpha=0.5, linewidth=0.9,
                   label=("$k{=}2$ (p-less) baseline" if i == 0 else None))
        ax.set_xscale("log")
        ax.set_xticks(sparse_ticks)
        ax.set_xticklabels([f"{k:g}" for k in sparse_ticks], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("k (Rényi order)")
        ax.set_title(model.replace("-R1-Distill-Llama-8B", "-R1-Distill"))
        if i == 0:
            ax.set_ylabel("truncation (%)")
        ax.grid(True, alpha=0.3)

    fig5a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4,
                 fontsize=8, frameon=True)
    fig5a.tight_layout(rect=[0, 0.08, 1, 1])
    fig5b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                 fontsize=8, frameon=True)
    fig5b.tight_layout(rect=[0, 0.08, 1, 1])

    for fig, name in [(fig5a, "fig5a_renyi_passk.png"), (fig5b, "fig5b_renyi_truncation.png")]:
        fig.savefig(FIG_DIR_CODE / name, dpi=180, bbox_inches="tight")
        fig.savefig(FIG_DIR_PAPER / name, dpi=180, bbox_inches="tight")
        print(f"Wrote {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
