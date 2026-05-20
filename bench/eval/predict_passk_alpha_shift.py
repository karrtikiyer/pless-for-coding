"""C7 v3 Step 5: predict per-task pass-rate shift from per-position entropy.

Hypothesis: per-task pass-rate at α > 2 can be predicted from
per-task pass-rate at α=2 + per-position entropy statistics at α=2.

For each model in {Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct} on MBPP:

  1. Stream the entropy sidecar (per-position records with sigma_p2,
     sigma_p3, sigma_p5, max_p over the full 500-task × 10-sample run).
  2. Aggregate to per-task entropy features (mean over positions & samples).
  3. Load per-task num_correct at α ∈ {2.0, 2.5, 3.0, 5.0} from the
     metrics JSONs.
  4. Fit two regressions for each target α ∈ {2.5, 3, 5}:
        (a) baseline: Δc_i  ~  intercept                       — null model
        (b) entropy: Δc_i   ~  intercept + entropy features    — full model
     Compare R² between models to test if entropy features add signal.
  5. Also fit a "kept-mass-ratio" feature derived from population-level
     mean log(σ_2 / σ_α) — closer to the v1 naive formula but applied
     at the per-task level.

Outputs:
  results/c7_validation/step5_entropy_prediction/per_task_features_{model}.parquet
  results/c7_validation/step5_entropy_prediction/regression_summary.json
  results/c7_validation/step5_entropy_prediction/regression_summary.md
  results/c7_validation/step5_entropy_prediction/scatter_{model}_alpha{α}.png

Status: smoke-test of feasibility. Cleanly-motivated features only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Per-position entropy → per-task aggregate features
# ---------------------------------------------------------------------------


@dataclass
class PerTaskAccum:
    """Streaming accumulator for one task across all samples & positions."""
    n_positions: int = 0
    sum_log_p2: float = 0.0  # = -Σ_pos log(σ_p2) = Σ_pos H_2 (collision entropy)
    sum_log_p3: float = 0.0
    sum_log_p5: float = 0.0
    sum_log_p2_sq: float = 0.0  # for variance
    sum_max_p: float = 0.0
    sum_frac_high_entropy: float = 0.0  # frac positions with σ_p2 < 0.5
    sum_log_ratio_23: float = 0.0  # Σ_pos log(σ_p2 / σ_p3)
    sum_log_ratio_25: float = 0.0  # Σ_pos log(σ_p2 / σ_p5)
    sum_log_ratio_25_sq: float = 0.0
    n_samples: int = 0
    sample_lengths: list[int] | None = None  # one entry per sample

    def __post_init__(self):
        if self.sample_lengths is None:
            self.sample_lengths = []


def update_from_record(accum: PerTaskAccum, rec: dict) -> None:
    sp2 = max(float(rec["sigma_p2"]), 1e-12)
    sp3 = max(float(rec["sigma_p3"]), 1e-12)
    sp5 = max(float(rec["sigma_p5"]), 1e-12)
    mp = float(rec["max_p"])
    log_p2 = math.log(sp2)
    log_p3 = math.log(sp3)
    log_p5 = math.log(sp5)
    accum.n_positions += 1
    accum.sum_log_p2 += -log_p2  # H_2
    accum.sum_log_p3 += -0.5 * log_p3  # H_3 = (1/(1-3)) log σ_3 = -0.5 log σ_3
    accum.sum_log_p5 += -0.25 * log_p5
    accum.sum_log_p2_sq += log_p2 * log_p2
    accum.sum_max_p += mp
    accum.sum_frac_high_entropy += 1.0 if sp2 < 0.5 else 0.0
    accum.sum_log_ratio_23 += log_p2 - log_p3  # log(σ_p2 / σ_p3) ≥ 0
    accum.sum_log_ratio_25 += log_p2 - log_p5
    accum.sum_log_ratio_25_sq += (log_p2 - log_p5) ** 2


def stream_entropy_to_per_task_features(entropy_path: Path) -> dict[int, dict[str, float]]:
    """Returns {task_id: {feature_name: value}}."""
    by_task: dict[int, PerTaskAccum] = defaultdict(PerTaskAccum)
    sample_pos_count: dict[tuple[int, int], int] = defaultdict(int)
    with entropy_path.open() as f:
        for line in f:
            rec = json.loads(line)
            t = int(rec["task_id"])
            s = int(rec["sample_id"])
            update_from_record(by_task[t], rec)
            sample_pos_count[(t, s)] += 1

    # Count samples and accumulate lengths per task
    for (t, _s), n in sample_pos_count.items():
        by_task[t].sample_lengths.append(n)

    features: dict[int, dict[str, float]] = {}
    for t, acc in by_task.items():
        n = max(acc.n_positions, 1)
        ns = len(acc.sample_lengths)
        mean_log_p2 = acc.sum_log_p2 / n  # mean H_2
        var_log_p2 = max(acc.sum_log_p2_sq / n - (acc.sum_log_p2 / n) ** 2, 0.0)
        # Note: sum_log_p2 = -Σ log σ_p2, so mean of log σ_p2 = -mean_log_p2
        # We computed sum_log_p2_sq = Σ (log σ_p2)^2 (not negated). So
        # var(log σ_p2) = E[(log σ_p2)^2] - (E[log σ_p2])^2.
        features[t] = {
            "n_samples": float(ns),
            "n_positions_total": float(n),
            "mean_positions_per_sample": float(np.mean(acc.sample_lengths) if ns else 0),
            "std_positions_per_sample": float(np.std(acc.sample_lengths) if ns > 1 else 0),
            # Mean per-position Rényi entropies (averaged across samples & positions)
            "mean_H2": mean_log_p2,
            "mean_H3": acc.sum_log_p3 / n,
            "mean_H5": acc.sum_log_p5 / n,
            "var_log_sigma_p2": var_log_p2,
            "mean_max_p": acc.sum_max_p / n,
            "frac_high_entropy_positions": acc.sum_frac_high_entropy / n,
            # Per-position threshold-ratio features (key for α-scaling)
            "mean_log_ratio_23": acc.sum_log_ratio_23 / n,
            "mean_log_ratio_25": acc.sum_log_ratio_25 / n,
            "var_log_ratio_25": max(
                acc.sum_log_ratio_25_sq / n - (acc.sum_log_ratio_25 / n) ** 2, 0.0
            ),
            # Cumulative log-ratio per sample (sum, not mean)
            "sum_log_ratio_23": acc.sum_log_ratio_23 / max(ns, 1),
            "sum_log_ratio_25": acc.sum_log_ratio_25 / max(ns, 1),
        }
    return features


# ---------------------------------------------------------------------------
# Per-task counts at each α
# ---------------------------------------------------------------------------


def load_per_task_counts(metrics_path: Path) -> dict[int, int]:
    with metrics_path.open() as f:
        d = json.load(f)
    return {int(t["task_id"]): int(t["num_correct"]) for t in d["per_task"]}


# ---------------------------------------------------------------------------
# Regression utilities (OLS via numpy, no sklearn)
# ---------------------------------------------------------------------------


@dataclass
class OLSFit:
    coefs: dict[str, float]
    r_squared: float
    n: int


def fit_ols(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> OLSFit:
    """OLS via lstsq. X already includes the intercept column if desired."""
    coef, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return OLSFit(
        coefs={feature_names[i]: float(coef[i]) for i in range(len(feature_names))},
        r_squared=r2,
        n=len(y),
    )


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


_MODELS_WITH_ENTROPY = {
    "Qwen2.5-Coder-7B-Instruct": "Qwen--Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct": "codellama--CodeLlama-7b-Instruct-hf",
}
_TARGET_ALPHAS = [2.5, 3.0, 5.0]
_FEATURES_FOR_REGRESSION = [
    "mean_H2",
    "mean_H3",
    "mean_H5",
    "var_log_sigma_p2",
    "mean_log_ratio_23",
    "mean_log_ratio_25",
    "var_log_ratio_25",
    "frac_high_entropy_positions",
    "mean_max_p",
    "mean_positions_per_sample",
]


@dataclass
class CellRegression:
    model: str
    target_alpha: float
    n_tasks: int
    # Δp_i = c_i^(α)/10 - c_i^(α=2)/10
    null_r2: float  # intercept only
    baseline_r2: float  # intercept + c_i^(α=2)
    full_r2: float  # intercept + c_i^(α=2) + entropy features
    delta_only_full_r2: float  # entropy features → Δp directly (no c_2 covariate)
    coefficients_full: dict[str, float]
    sum_log_ratio_only_r2: float  # entropy = mean_log_ratio_2α alone → Δp


def run_per_model(
    repo_root: Path,
    model_short: str,
    model_slug: str,
) -> tuple[list[CellRegression], dict[int, dict[str, float]], dict[float, dict[int, int]]]:
    entropy_path = (
        repo_root
        / "results"
        / "pless_alpha_entropy"
        / model_slug
        / "pless_t1.0.jsonl.entropy.jsonl"
    )
    print(f"[{model_short}] Streaming entropy from {entropy_path.relative_to(repo_root)}")
    features = stream_entropy_to_per_task_features(entropy_path)
    print(f"[{model_short}] Built features for {len(features)} tasks")

    counts_by_alpha: dict[float, dict[int, int]] = {}
    for alpha in [2.0] + _TARGET_ALPHAS:
        metrics_p = (
            repo_root
            / "results"
            / "pless_alpha_full"
            / model_slug
            / "metrics"
            / f"pless_alpha_a{alpha:.1f}_t1.0_metrics.json"
        )
        counts_by_alpha[alpha] = load_per_task_counts(metrics_p)

    results: list[CellRegression] = []
    task_ids = sorted(set(features.keys()) & set(counts_by_alpha[2.0].keys()))
    print(f"[{model_short}] {len(task_ids)} tasks shared between features and counts")

    n = 10  # samples per task

    for alpha in _TARGET_ALPHAS:
        c2 = np.array([counts_by_alpha[2.0][t] for t in task_ids], dtype=float)
        ca = np.array([counts_by_alpha[alpha][t] for t in task_ids], dtype=float)
        p2 = c2 / n
        pa = ca / n
        dp = pa - p2

        # null: intercept only
        X_null = np.ones((len(task_ids), 1))
        null = fit_ols(X_null, dp, ["intercept"])

        # baseline: intercept + p2 (does pass-rate at α=2 alone predict shift?)
        X_base = np.column_stack([np.ones(len(task_ids)), p2])
        base = fit_ols(X_base, dp, ["intercept", "p_alpha2"])

        # full: intercept + p2 + entropy features
        feat_cols = []
        for name in _FEATURES_FOR_REGRESSION:
            feat_cols.append(np.array([features[t][name] for t in task_ids], dtype=float))
        X_full = np.column_stack([np.ones(len(task_ids)), p2] + feat_cols)
        full = fit_ols(X_full, dp, ["intercept", "p_alpha2"] + _FEATURES_FOR_REGRESSION)

        # delta-only: just entropy features (no p2) → dp
        X_delta = np.column_stack([np.ones(len(task_ids))] + feat_cols)
        delta = fit_ols(X_delta, dp, ["intercept"] + _FEATURES_FOR_REGRESSION)

        # single-feature: just mean_log_ratio for this α
        single_feat = "mean_log_ratio_23" if alpha <= 3.0 else "mean_log_ratio_25"
        sf = np.array([features[t][single_feat] for t in task_ids], dtype=float)
        X_single = np.column_stack([np.ones(len(task_ids)), p2, sf])
        single = fit_ols(X_single, dp, ["intercept", "p_alpha2", single_feat])

        results.append(
            CellRegression(
                model=model_short,
                target_alpha=alpha,
                n_tasks=len(task_ids),
                null_r2=null.r_squared,
                baseline_r2=base.r_squared,
                full_r2=full.r_squared,
                delta_only_full_r2=delta.r_squared,
                coefficients_full={k: v for k, v in full.coefs.items()},
                sum_log_ratio_only_r2=single.r_squared,
            )
        )

    return results, features, counts_by_alpha


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def make_summary_md(all_results: list[CellRegression]) -> str:
    lines = []
    lines.append("# C7 v3 Step 5: predicting per-task pass-rate shift from α=2 entropy")
    lines.append("")
    lines.append(
        "Target: `Δp_i = c_i^(α)/10 − c_i^(α=2)/10` for each task `i`. "
        "Features extracted from per-position entropy log at α=2 only "
        "(streaming aggregate over all 500 tasks × 10 samples)."
    )
    lines.append("")
    lines.append("## R² comparison")
    lines.append("")
    lines.append(
        "- **null**: predict Δp = constant (just intercept) — sanity check, should be ~0\n"
        "- **baseline**: Δp = intercept + β·p_α2 — does pass-rate at α=2 alone predict shift?\n"
        "- **full**: Δp = intercept + β·p_α2 + Σ entropy features\n"
        "- **Δ-only**: Δp = intercept + Σ entropy features (no p_α2 covariate)\n"
        "- **single**: Δp = intercept + β·p_α2 + γ·mean_log_ratio (a single 'distance' feature)"
    )
    lines.append("")
    lines.append("| Model | α target | n | null R² | baseline R² | full R² | Δ-only R² | single-feat R² |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        lines.append(
            f"| {r.model} | {r.target_alpha:.1f} | {r.n_tasks} | "
            f"{r.null_r2:+.4f} | {r.baseline_r2:+.4f} | {r.full_r2:+.4f} | "
            f"{r.delta_only_full_r2:+.4f} | {r.sum_log_ratio_only_r2:+.4f} |"
        )
    lines.append("")
    lines.append("## Full-model coefficients (per cell)")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r.model} / α={r.target_alpha:.1f}")
        lines.append("")
        lines.append("| Feature | Coefficient |")
        lines.append("|---|---:|")
        for k, v in r.coefficients_full.items():
            lines.append(f"| `{k}` | {v:+.6f} |")
        lines.append("")
    lines.append("## Reading the result")
    lines.append("")
    lines.append(
        "If **full R² » baseline R²** and **Δ-only R² > 0**, entropy features add real "
        "predictive signal beyond knowing `pass-rate at α=2`. That would justify Step 5b "
        "(building a closed-form prediction of `(a_α, b_α)` from entropy)."
    )
    lines.append("")
    lines.append(
        "If **full R² ≈ baseline R²** (entropy adds nothing) **AND** baseline itself is low "
        "(<0.2), the per-task pass-rate shift is mostly noise at our n=10 sample size, and "
        "Step 5 should pivot to population-level prediction or accept that the empirical "
        "ν(α) regularity is the deliverable."
    )
    return "\n".join(lines) + "\n"


def plot_dp_vs_feature(
    model_short: str,
    features: dict[int, dict[str, float]],
    counts_by_alpha: dict[float, dict[int, int]],
    out_dir: Path,
    alphas: list[float] = (2.5, 3.0, 5.0),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 10
    task_ids = sorted(set(features.keys()) & set(counts_by_alpha[2.0].keys()))
    c2 = np.array([counts_by_alpha[2.0][t] for t in task_ids], dtype=float)

    for alpha in alphas:
        ca = np.array([counts_by_alpha[alpha][t] for t in task_ids], dtype=float)
        dp = (ca - c2) / n
        feat_name = "mean_log_ratio_23" if alpha <= 3.0 else "mean_log_ratio_25"
        x = np.array([features[t][feat_name] for t in task_ids], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, dp, alpha=0.4, s=14)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, dp, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, "r-", alpha=0.6,
                    label=f"linear fit: slope={slope:.3f}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"per-task {feat_name} (at α=2)")
        ax.set_ylabel(f"Δp = p_α={alpha} − p_α=2  (per task)")
        ax.set_title(f"{model_short}: per-task pass-rate shift vs {feat_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        slug = model_short.replace("/", "-").replace(" ", "_")
        fig.savefig(out_dir / f"scatter_{slug}_alpha{alpha:.1f}.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/c7_validation/step5_entropy_prediction"),
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir if args.out_dir.is_absolute() else (repo_root / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[CellRegression] = []
    per_task_features_by_model: dict[str, dict] = {}
    for short, slug in _MODELS_WITH_ENTROPY.items():
        cell_results, features, counts_by_alpha = run_per_model(repo_root, short, slug)
        all_results.extend(cell_results)
        per_task_features_by_model[short] = features
        # Per-task features dump (JSON; small enough — 500 tasks × ~15 feats)
        (out_dir / f"per_task_features_{short.replace('/', '_')}.json").write_text(
            json.dumps({str(k): v for k, v in features.items()}, indent=2)
        )
        # Scatters
        plot_dp_vs_feature(short, features, counts_by_alpha, out_dir)

    md = make_summary_md(all_results)
    (out_dir / "regression_summary.md").write_text(md)
    (out_dir / "regression_summary.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "model": r.model,
                        "target_alpha": r.target_alpha,
                        "n_tasks": r.n_tasks,
                        "null_r2": r.null_r2,
                        "baseline_r2": r.baseline_r2,
                        "full_r2": r.full_r2,
                        "delta_only_r2": r.delta_only_full_r2,
                        "single_feat_r2": r.sum_log_ratio_only_r2,
                        "coefficients_full": r.coefficients_full,
                    }
                    for r in all_results
                ]
            },
            indent=2,
        )
    )
    print(md)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
