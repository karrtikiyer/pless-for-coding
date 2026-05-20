"""Validate the C7 closed-form pass@k(α) prediction against empirical data.

Theory (see docs/theory/c7_pass_at_k_alpha_derivation.md):

    pass@1_task(α)  =  pass@1_task(α=2) · ∏_t (f_{t,α=2} / f_{t,α})

where f_{t,α} is the kept-probability mass at position t under α-truncation,
and the product is over positions in the samples generated for that task.

We approximate f_{t,α} from the top-32 probabilities logged in the entropy
sidecar — the tail beyond top-32 contributes <0.01% of mass at any α ≥ 2,
so the approximation is below sampling noise.

For pass@k aggregation, we use the iid binomial under predicted pass@1
per task.

Usage:

    uv run python -m bench.eval.validate_pass_at_k_prediction \\
        --entropy-dir results/pless_alpha_entropy \\
        --metrics-root results/pless_alpha_full \\
        --output-dir results/c7_validation
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALPHAS = [2.0, 2.5, 3.0, 5.0]
KS = [1, 3, 5, 10]


def kept_mass_at_alpha(top_probs: list[float], alpha: float) -> float:
    """Compute f_{t,α} = Σ_{i: p_i ≥ T_α} p_i from top-32 probabilities.

    The threshold T_α = Σ p_i^α is approximated by summing over top-32
    (the tail beyond top-32 contributes <0.01% to Σ p_i^α for α ≥ 2 on
    typical LM distributions). We compute the threshold and the kept
    mass jointly from the same top-32 vector.
    """
    p = np.asarray(top_probs, dtype=np.float64)
    threshold = float(np.sum(p ** alpha))
    return float(np.sum(p[p >= threshold]))


def load_entropy_sidecar(path: Path) -> dict[tuple[int, int], list[dict]]:
    """Load entropy JSONL grouped by (task_id, sample_id).

    Each value is a list of position records sorted by position index.
    """
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            grouped[(r["task_id"], r["sample_id"])].append(r)
    # Sort each list by position
    for k in grouped:
        grouped[k].sort(key=lambda r: r["position"])
    return dict(grouped)


def per_task_log_ratio(records: dict[tuple[int, int], list[dict]],
                       alpha_target: float, alpha_baseline: float = 2.0
                       ) -> dict[int, float]:
    """Compute mean over samples of Σ_t log(f_{t,α₀} / f_{t,α}) per task.

    Returns: {task_id -> mean log-ratio across samples for that task}.

    A larger log-ratio (more positive) → larger predicted pass@1 drop
    from α=2 baseline to α_target.
    """
    per_task_sample_ratios: dict[int, list[float]] = defaultdict(list)
    for (task_id, sample_id), positions in records.items():
        sample_log_ratio = 0.0
        for r in positions:
            top = r["top32_probs"]
            f_base = kept_mass_at_alpha(top, alpha_baseline)
            f_alpha = kept_mass_at_alpha(top, alpha_target)
            # log(f_base / f_alpha) — positive when α > base
            if f_base > 0 and f_alpha > 0:
                sample_log_ratio += np.log(f_base / f_alpha)
            else:
                # numerical guard — shouldn't trigger if top-32 ok
                pass
        per_task_sample_ratios[task_id].append(sample_log_ratio)
    return {task_id: float(np.mean(rs))
            for task_id, rs in per_task_sample_ratios.items()}


def load_per_task_pass(metrics_path: Path) -> dict[int, float]:
    """Load empirical per-task pass-rate (num_correct / num_samples) at α."""
    m = json.loads(metrics_path.read_text())
    n_samples = m["num_samples_per_task"]
    return {pt["task_id"]: pt["num_correct"] / n_samples for pt in m["per_task"]}


def pass_at_k_unbiased(c: int, n: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if c < 0 or k > n:
        return 0.0
    if n - c < k:
        return 1.0
    # Use logspace for numerical stability with large n
    log_c1 = sum(np.log(n - c - i) for i in range(k))
    log_c2 = sum(np.log(n - i) for i in range(k))
    return float(1.0 - np.exp(log_c1 - log_c2))


def predict_dataset_metrics(per_task_p: dict[int, float], n: int) -> dict[str, float]:
    """Given per-task predicted pass@1, compute pass@k by integrating over
    Binomial(n, p) per task and averaging across tasks.

    For each task with predicted p, E_c[pass@k(c, n)] is computed analytically
    via the binomial PMF.
    """
    out = {}
    for k in KS:
        # Average over tasks of the expected pass@k
        per_task_passk = []
        for task_id, p in per_task_p.items():
            # Compute E_c~Binom(n, p)[pass_at_k_unbiased(c, n, k)]
            # We sum over c=0..n
            ev = 0.0
            log_p = np.log(p) if p > 0 else -1e9
            log_1mp = np.log(1 - p) if p < 1 else -1e9
            for c in range(n + 1):
                # log Binom(n, c) p^c (1-p)^(n-c)
                log_binom = (
                    sum(np.log(n - i) for i in range(c))
                    - sum(np.log(i + 1) for i in range(c))
                )
                log_pmf = log_binom + c * log_p + (n - c) * log_1mp
                pmf = np.exp(log_pmf) if log_pmf > -700 else 0.0
                ev += pmf * pass_at_k_unbiased(c, n, k)
            per_task_passk.append(ev)
        out[f"pass@{k}"] = float(np.mean(per_task_passk))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy-dir", type=Path,
                        default=Path("results/pless_alpha_entropy"))
    parser.add_argument("--metrics-root", type=Path,
                        default=Path("results/pless_alpha_full"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/c7_validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_results = {}

    for model_dir in sorted(args.entropy_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "analysis":
            continue
        slug = model_dir.name
        sidecars = list(model_dir.glob("*.entropy.jsonl"))
        if not sidecars:
            print(f"[skip] {slug}: no entropy sidecar")
            continue

        # Pick the smaller file first as a sanity check
        sidecar = sidecars[0]
        print(f"[{slug}] loading {sidecar}")
        records = load_entropy_sidecar(sidecar)
        n_samples = max(s_id for _, s_id in records.keys()) + 1
        print(f"  {len(records)} (task, sample) groups; up to sample_id {n_samples - 1}")

        # Load empirical metrics per α
        metrics_dir = args.metrics_root / slug / "metrics"
        if not metrics_dir.is_dir():
            print(f"  [skip] no metrics dir at {metrics_dir}")
            continue

        # The baseline: per-task pass@1 at α=2.0
        baseline_metrics = metrics_dir / "pless_alpha_a2.0_t1.0_metrics.json"
        if not baseline_metrics.exists():
            print(f"  [skip] no α=2.0 baseline metrics")
            continue
        per_task_p_alpha2 = load_per_task_pass(baseline_metrics)
        baseline_n = json.loads(baseline_metrics.read_text())["num_samples_per_task"]
        print(f"  baseline α=2.0: {len(per_task_p_alpha2)} tasks, n_samples={baseline_n}")

        per_alpha_results = {}
        for alpha in ALPHAS:
            # Compute per-task log-ratio: Σ_t log(f_{α=2} / f_{α})
            log_ratio = per_task_log_ratio(records, alpha_target=alpha, alpha_baseline=2.0)

            # Predict per-task pass@1(α) = pass@1(α=2) * exp(-log_ratio)
            # (since the formula is pass@1(α) = pass@1(α=2) · ∏ (f_β/f_α);
            # and log_ratio = Σ log(f_β/f_α) so we use exp(log_ratio)
            # But wait: log_ratio = Σ log(f_β/f_α) is POSITIVE when α > β
            # because f_α >= f_β, so f_β/f_α <= 1, so log negative... let me re-check
            # f_{α=5} >= f_{α=2}. So f_{α=2}/f_{α=5} <= 1, log <= 0, log_ratio <= 0.
            # pass@1(α=5) = pass@1(α=2) * ∏ (f_{α=2}/f_{α=5}) <= pass@1(α=2). Good.
            # so predicted = baseline * exp(log_ratio) where log_ratio is NEGATIVE.
            predicted_per_task = {}
            for task_id, p2 in per_task_p_alpha2.items():
                if task_id not in log_ratio:
                    continue
                pred = p2 * np.exp(log_ratio[task_id])
                pred = float(np.clip(pred, 0.0, 1.0))
                predicted_per_task[task_id] = pred

            # Predicted pass@k aggregation via binomial under iid
            pred_dataset = predict_dataset_metrics(predicted_per_task, baseline_n)

            # Measured pass@k for comparison
            measured_metrics = metrics_dir / f"pless_alpha_a{alpha}_t1.0_metrics.json"
            if not measured_metrics.exists():
                print(f"  [skip α={alpha}] no measured metrics")
                continue
            measured = json.loads(measured_metrics.read_text())
            measured_passk = {f"pass@{k}": measured["pass_at_k"][str(k)] for k in KS}

            per_alpha_results[str(alpha)] = {
                "predicted": pred_dataset,
                "measured": measured_passk,
                "abs_err_pp": {f"pass@{k}":
                               abs(pred_dataset[f"pass@{k}"] - measured_passk[f"pass@{k}"]) * 100
                               for k in KS},
            }
            print(f"  α={alpha}: predicted pass@1={pred_dataset['pass@1']*100:.2f}% "
                  f"vs measured {measured_passk['pass@1']*100:.2f}% "
                  f"(err {per_alpha_results[str(alpha)]['abs_err_pp']['pass@1']:.2f} pp)")
            print(f"           predicted pass@10={pred_dataset['pass@10']*100:.2f}% "
                  f"vs measured {measured_passk['pass@10']*100:.2f}% "
                  f"(err {per_alpha_results[str(alpha)]['abs_err_pp']['pass@10']:.2f} pp)")

        overall_results[slug] = per_alpha_results

        # Plot: predicted vs measured pass@k curves per α
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, k in zip(axes, KS):
            xs = []
            ys_pred = []
            ys_meas = []
            for alpha in ALPHAS:
                if str(alpha) not in per_alpha_results:
                    continue
                xs.append(alpha)
                ys_pred.append(per_alpha_results[str(alpha)]["predicted"][f"pass@{k}"] * 100)
                ys_meas.append(per_alpha_results[str(alpha)]["measured"][f"pass@{k}"] * 100)
            ax.plot(xs, ys_pred, "o-", label="predicted (C7)", color="tab:blue")
            ax.plot(xs, ys_meas, "s--", label="measured", color="tab:orange")
            ax.set_xlabel("α")
            ax.set_ylabel(f"pass@{k} (%)")
            ax.set_title(f"pass@{k}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"C7 prediction vs measured — {slug}")
        fig.tight_layout()
        out_png = args.output_dir / f"predicted_vs_measured_{slug}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"  wrote {out_png}")

    # Summary JSON + markdown
    summary_path = args.output_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(overall_results, indent=2))
    print(f"\n[done] wrote {summary_path}")

    # Markdown summary
    md_lines = ["# C7 fit summary — predicted vs measured pass@k(α)",
                "",
                "Each cell shows |predicted − measured| in percentage points.",
                ""]
    for slug, per_alpha in overall_results.items():
        md_lines.append(f"## {slug}")
        md_lines.append("")
        md_lines.append("| α | pass@1 err | pass@5 err | pass@10 err |")
        md_lines.append("|---:|----------:|----------:|-----------:|")
        for alpha in ALPHAS:
            if str(alpha) not in per_alpha:
                continue
            errs = per_alpha[str(alpha)]["abs_err_pp"]
            md_lines.append(f"| {alpha} | {errs['pass@1']:.2f} pp "
                            f"| {errs['pass@5']:.2f} pp "
                            f"| {errs['pass@10']:.2f} pp |")
        md_lines.append("")
    md_path = args.output_dir / "fit_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
