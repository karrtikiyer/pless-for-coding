"""C7 v2: beta-binomial framework for pass@k(α).

For each (model, dataset, α), fit Beta(a, b) to the per-task pass-rate
via method of moments and reproduce dataset-level pass@k in closed form:

    pass@k = 1 - B(a, b+k)/B(a, b) = 1 - prod_{i=0..k-1} (b+i)/(a+b+i)

If Beta(a, b) fits the observed per-task distribution, this should
match the measured (Chen et al. unbiased) pass@k within sampling
noise. We also extract (a_α, b_α) trajectories across α to expose
how per-task heterogeneity shifts.

Inputs:
  results/pless_alpha_full/{model}/metrics/pless_alpha_a{α}_t1.0_metrics.json
  results/pless_alpha_full_humaneval/{model}/humaneval/metrics/pless_alpha_a{α}_t1.0_metrics.json

Outputs:
  results/c7_validation/beta_binomial/fit_summary.json
  results/c7_validation/beta_binomial/fit_summary.md
  results/c7_validation/beta_binomial/predicted_vs_measured_{cell}.png
  results/c7_validation/beta_binomial/alpha_trajectory_{model}.png

References:
  - arXiv:2510.05197 — Efficient Prediction of Pass@k Scaling (beta-binomial)
  - arXiv:2107.03374 — pass@k unbiased estimator (Chen et al. 2021)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Matplotlib import is deferred so import for tests stays cheap.


# ---------------------------------------------------------------------------
# Beta-binomial fit + pass@k closed form
# ---------------------------------------------------------------------------


@dataclass
class BetaFit:
    a: float
    b: float
    mean: float
    nu: float  # concentration = a + b
    method: str  # "mom" or "degenerate-all0" or "degenerate-all1"


def fit_beta_binomial_mom(c: np.ndarray, n: int) -> BetaFit:
    """Method-of-moments fit of Beta(a, b) to count data c (size M)."""
    c = np.asarray(c, dtype=float)
    m1 = c.mean() / n  # mean per-task pass rate

    if m1 <= 0.0:
        return BetaFit(a=0.0, b=float("inf"), mean=0.0, nu=float("inf"), method="degenerate-all0")
    if m1 >= 1.0:
        return BetaFit(a=float("inf"), b=0.0, mean=1.0, nu=float("inf"), method="degenerate-all1")

    # Second factorial moment estimator for E[p^2] under Beta-Binom:
    # E[c(c-1)] = n(n-1) E[p^2], so E[p^2] = mean(c*(c-1)) / (n(n-1))
    m2 = float(np.mean(c * (c - 1)) / (n * (n - 1)))
    var_p = m2 - m1 * m1

    # If E[p^2] - E[p]^2 <= 0, the per-task pass-rate has zero or
    # sub-Binomial dispersion — collapse to a point mass at m1
    # (recovers the iid-Binomial closed form 1 - (1-m1)^k).
    if var_p <= 0:
        return BetaFit(
            a=float("inf"),
            b=float("inf"),
            mean=m1,
            nu=float("inf"),
            method="degenerate-point-mass",
        )

    nu = m1 * (1 - m1) / var_p - 1.0
    if nu <= 0:
        return BetaFit(
            a=float("inf"),
            b=float("inf"),
            mean=m1,
            nu=float("inf"),
            method="degenerate-point-mass",
        )

    a = m1 * nu
    b = (1 - m1) * nu
    return BetaFit(a=a, b=b, mean=m1, nu=nu, method="mom")


def pass_at_k_beta(fit: BetaFit, k: int) -> float:
    """E_{p~Beta(a,b)}[1 - (1-p)^k] in closed form."""
    if fit.method == "degenerate-all0":
        return 0.0
    if fit.method == "degenerate-all1":
        return 1.0
    if fit.method == "degenerate-point-mass":
        return 1.0 - (1.0 - fit.mean) ** k
    # E[(1-p)^k] = prod_{i=0}^{k-1} (b+i) / (a+b+i)
    log_e = 0.0
    for i in range(k):
        log_e += math.log(fit.b + i) - math.log(fit.a + fit.b + i)
    return 1.0 - math.exp(log_e)


def measured_pass_at_k_chen(c: np.ndarray, n: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k = mean_task[1 - C(n-c, k) / C(n, k)]."""
    c = np.asarray(c, dtype=int)
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    # For each task: 1 - C(n-c, k) / C(n, k). If (n-c) < k, C(n-c, k) = 0
    # so pass@k = 1 (every k-subset includes a correct sample).
    denom = math.comb(n, k)
    per_task = np.array(
        [1.0 - (math.comb(n - ci, k) / denom if (n - ci) >= k else 0.0) for ci in c],
        dtype=float,
    )
    return float(per_task.mean())


# ---------------------------------------------------------------------------
# Cell loading
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    model: str  # short label
    dataset: str  # "MBPP" | "HumanEval"
    alpha: float
    metrics_path: Path


CELLS_TO_LOAD: list[Cell] = []

_MODELS = {
    "Qwen2.5-Coder-7B-Instruct": "Qwen--Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct": "codellama--CodeLlama-7b-Instruct-hf",
    "m-a-p-OCI-DS-1.3B": "m-a-p--OpenCodeInterpreter-DS-1.3B",
}
_ALPHAS = [2.0, 2.5, 3.0, 5.0]


def build_cells(repo_root: Path) -> list[Cell]:
    out: list[Cell] = []
    for short, slug in _MODELS.items():
        for alpha in _ALPHAS:
            # MBPP
            mbpp_p = (
                repo_root
                / "results"
                / "pless_alpha_full"
                / slug
                / "metrics"
                / f"pless_alpha_a{alpha:.1f}_t1.0_metrics.json"
            )
            if mbpp_p.exists():
                out.append(Cell(model=short, dataset="MBPP", alpha=alpha, metrics_path=mbpp_p))
            # HumanEval
            he_p = (
                repo_root
                / "results"
                / "pless_alpha_full_humaneval"
                / slug
                / "humaneval"
                / "metrics"
                / f"pless_alpha_a{alpha:.1f}_t1.0_metrics.json"
            )
            if he_p.exists():
                out.append(Cell(model=short, dataset="HumanEval", alpha=alpha, metrics_path=he_p))
    return out


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    model: str
    dataset: str
    alpha: float
    n_tasks: int
    n_samples_per_task: int
    a: float
    b: float
    mean: float
    nu: float
    fit_method: str
    predicted: dict[int, float]
    measured: dict[int, float]
    abs_err: dict[int, float]
    counts: list[int]


def evaluate_cell(cell: Cell, ks: list[int]) -> CellResult:
    with cell.metrics_path.open() as f:
        d = json.load(f)
    n = int(d["num_samples_per_task"])
    per_task = d["per_task"]
    counts = np.array([t["num_correct"] for t in per_task], dtype=int)

    fit = fit_beta_binomial_mom(counts, n)
    predicted = {k: pass_at_k_beta(fit, k) for k in ks}
    measured = {k: measured_pass_at_k_chen(counts, n, k) for k in ks}
    abs_err = {k: predicted[k] - measured[k] for k in ks}

    return CellResult(
        model=cell.model,
        dataset=cell.dataset,
        alpha=cell.alpha,
        n_tasks=len(counts),
        n_samples_per_task=n,
        a=float(fit.a) if math.isfinite(fit.a) else float("inf"),
        b=float(fit.b) if math.isfinite(fit.b) else float("inf"),
        mean=float(fit.mean),
        nu=float(fit.nu) if math.isfinite(fit.nu) else float("inf"),
        fit_method=fit.method,
        predicted=predicted,
        measured=measured,
        abs_err=abs_err,
        counts=counts.tolist(),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_summary_md(results: list[CellResult], ks: list[int]) -> str:
    lines: list[str] = []
    lines.append("# C7 v2: beta-binomial pass@k validation")
    lines.append("")
    lines.append(
        "Per-cell fit of `Beta(a, b)` to per-task pass-rate via method of moments, "
        "then closed-form pass@k. Compares against Chen et al. unbiased measured pass@k."
    )
    lines.append("")
    lines.append("## Per-cell fit quality")
    lines.append("")
    header = (
        "| Model | Dataset | α | n_tasks | a | b | mean | ν=a+b | "
        + " | ".join(f"pass@{k} pred / meas / err(pp)" for k in ks)
        + " |"
    )
    sep = "|" + "---|" * (8 + len(ks))
    lines.append(header)
    lines.append(sep)
    for r in results:
        row = [
            r.model,
            r.dataset,
            f"{r.alpha:.1f}",
            str(r.n_tasks),
            _fmt_finite(r.a),
            _fmt_finite(r.b),
            f"{r.mean:.4f}",
            _fmt_finite(r.nu),
        ]
        for k in ks:
            err_pp = 100.0 * r.abs_err[k]
            row.append(
                f"{100*r.predicted[k]:.2f} / {100*r.measured[k]:.2f} / {err_pp:+.2f}"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## α-trajectory of fitted Beta")
    lines.append("")
    lines.append("Shows how per-task heterogeneity (mean and concentration ν) shifts with α.")
    lines.append("Per-model, per-dataset.")
    lines.append("")
    by_model_ds: dict[tuple[str, str], list[CellResult]] = {}
    for r in results:
        by_model_ds.setdefault((r.model, r.dataset), []).append(r)
    for (model, dataset), cells in sorted(by_model_ds.items()):
        cells = sorted(cells, key=lambda r: r.alpha)
        lines.append(f"### {model} / {dataset}")
        lines.append("")
        lines.append("| α | mean p | ν = a+b | a | b |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in cells:
            lines.append(
                f"| {r.alpha:.1f} | {r.mean:.4f} | {_fmt_finite(r.nu)} | {_fmt_finite(r.a)} | {_fmt_finite(r.b)} |"
            )
        lines.append("")
    lines.append("## Verdict criteria")
    lines.append("")
    lines.append(
        "**Step 3 (sanity check): does fitted Beta reproduce measured pass@k?** "
        "Mean absolute error across all k in {1, 3, 5, 10} per cell:"
    )
    lines.append("")
    lines.append("| Model | Dataset | α | MAE (pp) | max-err (pp) |")
    lines.append("|---|---|---:|---:|---:|")
    for r in sorted(results, key=lambda r: (r.model, r.dataset, r.alpha)):
        errs = [abs(r.abs_err[k]) * 100.0 for k in ks]
        mae = sum(errs) / len(errs)
        maxerr = max(errs)
        lines.append(
            f"| {r.model} | {r.dataset} | {r.alpha:.1f} | {mae:.2f} | {maxerr:.2f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt_finite(x: float) -> str:
    if not math.isfinite(x):
        return "∞"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.3f}"


def plot_predicted_vs_measured(results: list[CellResult], out_dir: Path, ks: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cell: dict[tuple[str, str], list[CellResult]] = {}
    for r in results:
        by_cell.setdefault((r.model, r.dataset), []).append(r)

    for (model, dataset), cells in by_cell.items():
        cells = sorted(cells, key=lambda r: r.alpha)
        fig, ax = plt.subplots(figsize=(7, 5))
        alphas = [r.alpha for r in cells]
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(ks)))
        for i, k in enumerate(ks):
            pred = [r.predicted[k] for r in cells]
            meas = [r.measured[k] for r in cells]
            ax.plot(alphas, [100 * m for m in meas], "-o", color=colors[i], label=f"meas pass@{k}")
            ax.plot(
                alphas,
                [100 * p for p in pred],
                "--x",
                color=colors[i],
                label=f"pred pass@{k}",
                alpha=0.8,
            )
        ax.set_xlabel("α")
        ax.set_ylabel("pass@k (%)")
        ax.set_title(f"{model} / {dataset}: predicted (BetaBinom) vs measured")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        slug = f"{model}_{dataset}".replace("/", "-").replace(" ", "_")
        fig.savefig(out_dir / f"predicted_vs_measured_{slug}.png", dpi=140)
        plt.close(fig)


def plot_alpha_trajectory(results: list[CellResult], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[CellResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    for model, cells in by_model.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        by_ds: dict[str, list[CellResult]] = {}
        for r in cells:
            by_ds.setdefault(r.dataset, []).append(r)
        for ds, ds_cells in by_ds.items():
            ds_cells = sorted(ds_cells, key=lambda r: r.alpha)
            xs = [r.alpha for r in ds_cells]
            means = [r.mean for r in ds_cells]
            nus = [r.nu if math.isfinite(r.nu) else float("nan") for r in ds_cells]
            axes[0].plot(xs, means, "-o", label=ds)
            axes[1].plot(xs, nus, "-o", label=ds)
        axes[0].set_xlabel("α")
        axes[0].set_ylabel("mean p (a/(a+b))")
        axes[0].set_title(f"{model}: Beta mean across α")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)
        axes[1].set_xlabel("α")
        axes[1].set_ylabel("ν = a + b (concentration)")
        axes[1].set_title(f"{model}: Beta concentration across α")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        slug = model.replace("/", "-").replace(" ", "_")
        fig.savefig(out_dir / f"alpha_trajectory_{slug}.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/c7_validation/beta_binomial"),
        help="Output directory for fit_summary.{json,md} and plots.",
    )
    ap.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated k values to evaluate.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir if args.out_dir.is_absolute() else (repo_root / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cells = build_cells(repo_root)
    print(f"Loaded {len(cells)} cells:")
    for c in cells:
        print(f"  {c.model} / {c.dataset} / α={c.alpha}: {c.metrics_path.relative_to(repo_root)}")

    results = [evaluate_cell(c, ks) for c in cells]

    # JSON dump (drop large counts list out of summary for compactness; keep separately)
    summary_json = {
        "ks": ks,
        "results": [
            {
                k: v
                for k, v in r.__dict__.items()
                if k != "counts"
            }
            | {"counts_len": len(r.counts), "counts_sum": int(sum(r.counts))}
            for r in results
        ],
    }
    (out_dir / "fit_summary.json").write_text(json.dumps(summary_json, indent=2))

    md = format_summary_md(results, ks)
    (out_dir / "fit_summary.md").write_text(md)
    print(md)

    plot_predicted_vs_measured(results, out_dir, ks)
    plot_alpha_trajectory(results, out_dir)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
