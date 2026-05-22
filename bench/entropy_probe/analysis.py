"""Histogram, KDE, and Hartigan-Hartigan dip-test analysis."""
from __future__ import annotations

from pathlib import Path

import diptest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_dip_test(entropies: list[float]) -> dict:
    """Hartigan-Hartigan dip test for unimodality.

    Returns a summary dict with dip statistic, p-value, sample size,
    descriptive stats, and a categorical interpretation. Small p-value
    ⇒ reject unimodality (i.e., the distribution is multimodal).
    """
    arr = np.asarray(entropies, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        return {
            "error": "too_few_samples",
            "n_samples": int(len(arr)),
        }
    dip, pval = diptest.diptest(arr)
    if pval < 0.01:
        interp = "strongly_multimodal"
    elif pval < 0.05:
        interp = "multimodal"
    elif pval < 0.10:
        interp = "weakly_multimodal"
    else:
        interp = "consistent_with_unimodal"
    return {
        "dip_statistic": float(dip),
        "p_value": float(pval),
        "n_samples": int(len(arr)),
        "mean_entropy_nats": float(arr.mean()),
        "median_entropy_nats": float(np.median(arr)),
        "std_entropy_nats": float(arr.std()),
        "low_entropy_fraction": float((arr < 0.5).mean()),
        "high_entropy_fraction": float((arr > 2.0).mean()),
        "interpretation": interp,
    }


def plot_entropy_kde(
    entropies: list[float],
    out_path: Path,
    title: str,
) -> None:
    """Histogram + KDE overlay for a single (model, dataset) cell."""
    arr = np.asarray(entropies, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arr, bins=60, density=True, alpha=0.4, color="steelblue",
            edgecolor="black", label="histogram")
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr, bw_method=0.15)
        xs = np.linspace(max(0.0, arr.min() - 0.1), arr.max() + 0.1, 300)
        ax.plot(xs, kde(xs), color="darkred", linewidth=2, label="KDE")
    except Exception:
        pass  # KDE is optional decoration
    ax.set_xlabel("Per-token entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_overlay_kde(
    series: dict[str, list[float]],
    out_path: Path,
    title: str,
) -> None:
    """Overlay KDEs from multiple cells on one axis for visual comparison."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(series), 1)))
    try:
        from scipy.stats import gaussian_kde
        all_arrs = []
        for (label, vals), color in zip(series.items(), cmap):
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 2:
                continue
            kde = gaussian_kde(arr, bw_method=0.15)
            all_arrs.append(arr)
            xs = np.linspace(0.0, max(arr.max(), 5.0), 400)
            ax.plot(xs, kde(xs), linewidth=2, label=label, color=color)
    except Exception:
        pass
    ax.set_xlabel("Per-token entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
