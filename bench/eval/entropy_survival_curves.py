"""Central figure: survival mass vs entropy at α=2 vs α=5.

Bridges the bimodal-entropy observation and the α-knob mechanism claim.
For every recorded generation position with top-32 probability data,
compute:
  - H: entropy of the (renormalized) top-32 distribution (nats)
  - survived_α: total probability mass that survives the pless_alpha
    filter at α (using the stored σ_pᵅ threshold)
  - truncation_mass: 1 − sum(top32_probs), to quantify how much mass
    is past top-32

Bin positions by H and average per bin → two curves (α=2 and α=5)
per model that visualize *why* the α-knob produces diversity at the
secondary-mode region.

See ``docs/theory/central_figure_plan.md`` for the methodology and
the 4 standard-rigor validation checks.

CLI:
    uv run python -m bench.eval.entropy_survival_curves \\
        --models Qwen--Qwen2.5-Coder-7B-Instruct codellama--CodeLlama-7b-Instruct-hf \\
        --output-dir results/entropy_probe/_central_figure
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np


# ─── per-record math primitives ───────────────────────────────────────

_EPS = 1e-12


def compute_survival(p_vec, threshold: float) -> tuple[float, int]:
    """Survival mass under the pless_alpha filter.

    Matches the production sampler in ``bench/sampler_bridge.py``:
    - Tokens with p >= threshold survive (inclusive at threshold).
    - If no token meets the threshold, argmax falls back (one token).

    Args:
        p_vec: 1-D probability vector (need NOT be normalized; the
            survival mass is computed on the given values directly).
        threshold: the σ_pᵅ value for the chosen α.

    Returns:
        (survived_mass, n_surviving_tokens)
    """
    p = np.asarray(p_vec, dtype=float)
    mask = p >= threshold
    if not mask.any():
        # Argmax fallback (matches production make_pless_alpha_sampler)
        amax = float(p.max())
        return amax, 1
    return float(p[mask].sum()), int(mask.sum())


def compute_entropy(p_vec) -> float:
    """Shannon entropy in nats, accepts un-normalized input.

    Renormalizes internally so callers can pass top-K probability
    slices without first normalizing. Uses ε-smoothing for log(0).
    """
    p = np.asarray(p_vec, dtype=float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p_norm = p / total
    return float(-(p_norm * np.log(p_norm + _EPS)).sum())


def process_record(rec: dict) -> tuple[float, float, float, float]:
    """One per-token JSONL record → (H, surv_α2, surv_α5, truncation_mass).

    Survival is computed on the unnormalized top32_probs (matches
    what the stored σ_p² / σ_p⁵ were computed against — see
    ``bench/generator.py:_log_entropy_batch``: σ values come from the
    *full* softmax, but top-32 captures essentially all relevant mass
    for the α-knob threshold at our model scale; truncation_mass
    quantifies any leakage).
    """
    top32 = rec["top32_probs"]
    sigma_p2 = float(rec["sigma_p2"])
    sigma_p5 = float(rec["sigma_p5"])
    p = np.asarray(top32, dtype=float)
    truncation_mass = max(0.0, 1.0 - float(p.sum()))
    H = compute_entropy(p)
    surv_a2, _ = compute_survival(p, sigma_p2)
    surv_a5, _ = compute_survival(p, sigma_p5)
    return H, surv_a2, surv_a5, truncation_mass


# ─── streaming JSONL loader ───────────────────────────────────────────

def iter_records(jsonl_path: Path) -> Iterator[dict]:
    """Stream records from a .entropy.jsonl file. One JSON object per line."""
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ─── binning ──────────────────────────────────────────────────────────

def bin_records(
    records_4tuples,
    bin_width: float = 0.05,
    h_max: float = 4.0,
) -> list[dict]:
    """Bin records by entropy and aggregate within each bin.

    Args:
        records_4tuples: iterable of (H, surv_α2, surv_α5, truncation_mass).
        bin_width: bin width in nats. Default 0.05 gives ~80 bins for
            H in [0, 4] nats.
        h_max: upper edge of binning. Records above this are aggregated
            into the final bin.

    Returns:
        List of dicts, one per bin, each with:
          h_lo, h_hi (bin edges)
          n_positions
          mean_survival_alpha2, mean_survival_alpha5
          mean_truncation_mass
    """
    n_bins = int(math.ceil(h_max / bin_width))
    sums_a2 = np.zeros(n_bins, dtype=float)
    sums_a5 = np.zeros(n_bins, dtype=float)
    sums_trunc = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for H, sa2, sa5, trunc in records_4tuples:
        b = min(int(H / bin_width), n_bins - 1)
        if b < 0:
            b = 0
        sums_a2[b] += sa2
        sums_a5[b] += sa5
        sums_trunc[b] += trunc
        counts[b] += 1

    bins = []
    for b in range(n_bins):
        n = int(counts[b])
        if n > 0:
            ma2 = float(sums_a2[b] / n)
            ma5 = float(sums_a5[b] / n)
            mtr = float(sums_trunc[b] / n)
        else:
            ma2 = float("nan")
            ma5 = float("nan")
            mtr = float("nan")
        bins.append({
            "h_lo": round(b * bin_width, 6),
            "h_hi": round((b + 1) * bin_width, 6),
            "n_positions": n,
            "mean_survival_alpha2": ma2,
            "mean_survival_alpha5": ma5,
            "mean_truncation_mass": mtr,
        })
    return bins


# ─── validation checks ────────────────────────────────────────────────

def validate(jsonl_path: Path, n_sample: int = 500) -> dict:
    """Run the 4 standard-rigor validation checks on a random subsample.

    Checks per ``docs/theory/central_figure_plan.md``:
      1. Recomputed σ_p² matches stored sigma_p2 (within 1e-3, on
         top-32; small drift OK due to truncated tail).
      2. Top-32 truncation impact distribution.
      3. H recomputation: produces finite non-negative floats.
      4. (Per-bin sample size adequacy — done at the aggregation
         step, recorded in bins[].n_positions, not here.)

    Returns a dict with check name → result dict.
    """
    rng = np.random.default_rng(seed=0)
    # Reservoir sample n_sample records
    records: list[dict] = []
    for rec in iter_records(jsonl_path):
        if len(records) < n_sample:
            records.append(rec)
        else:
            i = int(rng.integers(0, len(records) + 1))
            if i < n_sample:
                records[i] = rec
    if not records:
        return {"error": "no records"}

    # Check 1: σ_p² recomputation
    deltas_sigma_p2 = []
    for r in records:
        p = np.asarray(r["top32_probs"], dtype=float)
        recomp = float((p ** 2).sum())
        stored = float(r["sigma_p2"])
        deltas_sigma_p2.append(abs(recomp - stored))
    deltas_sigma_p2_arr = np.array(deltas_sigma_p2)
    pct_within_1e_3 = float((deltas_sigma_p2_arr < 1e-3).mean())
    check1 = {
        "name": "sigma_p2_recomputation",
        "n_sampled": len(records),
        "max_delta": float(deltas_sigma_p2_arr.max()),
        "mean_delta": float(deltas_sigma_p2_arr.mean()),
        "fraction_within_1e-3": pct_within_1e_3,
        "acceptance_criterion": "fraction_within_1e-3 >= 0.99",
        "passed": pct_within_1e_3 >= 0.99,
    }

    # Check 2: truncation mass distribution
    trunc_masses = []
    for r in records:
        p = np.asarray(r["top32_probs"], dtype=float)
        trunc_masses.append(max(0.0, 1.0 - float(p.sum())))
    tm = np.array(trunc_masses)
    check2 = {
        "name": "top_32_truncation_mass",
        "n_sampled": len(records),
        "min": float(tm.min()),
        "median": float(np.median(tm)),
        "mean": float(tm.mean()),
        "p95": float(np.percentile(tm, 95)),
        "p99": float(np.percentile(tm, 99)),
        "max": float(tm.max()),
        "acceptance_criterion": "median <= 0.01 AND p99 <= 0.10 (most records have <1% tail leakage)",
        "passed": float(np.median(tm)) <= 0.01 and float(np.percentile(tm, 99)) <= 0.10,
    }

    # Check 3: H recomputation finite and non-negative
    Hs = []
    for r in records:
        H = compute_entropy(r["top32_probs"])
        Hs.append(H)
    H_arr = np.array(Hs)
    all_finite = bool(np.isfinite(H_arr).all())
    all_nonneg = bool((H_arr >= -1e-9).all())  # tiny float noise OK
    check3 = {
        "name": "H_recomputation_well_formed",
        "n_sampled": len(records),
        "min_H": float(H_arr.min()),
        "median_H": float(np.median(H_arr)),
        "max_H": float(H_arr.max()),
        "all_finite": all_finite,
        "all_nonneg": all_nonneg,
        "acceptance_criterion": "all H finite AND all H >= 0",
        "passed": all_finite and all_nonneg,
    }

    return {
        "validation_sample_size": len(records),
        "checks": [check1, check2, check3],
        "all_passed": all(c["passed"] for c in [check1, check2, check3]),
    }


# ─── orchestration ────────────────────────────────────────────────────

def process_model(
    jsonl_path: Path,
    bin_width: float = 0.05,
    h_max: float = 4.0,
) -> dict:
    """Stream a model's .entropy.jsonl, compute per-record + binned data."""
    records_processed = (process_record(r) for r in iter_records(jsonl_path))
    bins = bin_records(records_processed, bin_width=bin_width, h_max=h_max)
    return {
        "source_jsonl": str(jsonl_path),
        "bin_width_nats": bin_width,
        "h_max_nats": h_max,
        "bins": bins,
    }


def _density_fractions(bins: list[dict], h_low_max: float = 0.3,
                       h_high_min: float = 0.5) -> dict:
    """Per-model: fraction of positions in the low-entropy mode and the
    decision region. Used for the numeric annotation."""
    total = sum(b["n_positions"] for b in bins)
    if total == 0:
        return {"total": 0, "frac_below": 0.0, "frac_above": 0.0}
    low = sum(b["n_positions"] for b in bins if b["h_hi"] <= h_low_max)
    high = sum(b["n_positions"] for b in bins if b["h_lo"] >= h_high_min)
    return {
        "total": total,
        "frac_below": low / total,
        "frac_above": high / total,
        "h_low_max": h_low_max,
        "h_high_min": h_high_min,
    }


def make_plot(
    model_to_bins: dict[str, dict],
    output_path: Path,
) -> None:
    """Render the 2-subplot central figure: one subplot per model,
    two curves per subplot (α=2 and α=5), with:
      - dashed vertical reference line at H=0.5 nats (the divergence
        point matching the entropy distribution's secondary mode)
      - numeric position-density annotation in the lower-right corner
        (replaces the previous KDE shading, which was visually
        misleading on heavy-tailed entropy distributions — see
        ``docs/theory/central_figure_plan.md`` for the design rationale)
      - legend in lower-left corner (opposite side from annotation)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(model_to_bins), figsize=(7 * len(model_to_bins), 5),
        sharey=True,
    )
    if len(model_to_bins) == 1:
        axes = [axes]

    for ax, (model, data) in zip(axes, model_to_bins.items()):
        bins = data["bins"]
        h_centers = np.array([(b["h_lo"] + b["h_hi"]) / 2 for b in bins])
        n_pos = np.array([b["n_positions"] for b in bins])
        ma2 = np.array([b["mean_survival_alpha2"] for b in bins])
        ma5 = np.array([b["mean_survival_alpha5"] for b in bins])
        reliable = n_pos >= 50
        unreliable = (~reliable) & (n_pos > 0)

        # Solid curves on reliable bins
        ax.plot(h_centers[reliable], ma2[reliable], "-",
                color="tab:red", linewidth=2, label="α=2 (survived mass)")
        ax.plot(h_centers[reliable], ma5[reliable], "-",
                color="tab:blue", linewidth=2, label="α=5 (survived mass)")
        # Dotted continuation on low-count bins (signal noise)
        if unreliable.any():
            ax.plot(h_centers[unreliable], ma2[unreliable], ":",
                    color="tab:red", linewidth=1, alpha=0.5)
            ax.plot(h_centers[unreliable], ma5[unreliable], ":",
                    color="tab:blue", linewidth=1, alpha=0.5)

        # Vertical reference at H=0.5 (the divergence point)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.text(0.52, 0.97, "H=0.5 nats\n(divergence)",
                transform=ax.get_xaxis_transform(),
                fontsize=8, va="top", ha="left", alpha=0.7)

        # Numeric annotation — lower-right, computed from actual bin counts
        fr = _density_fractions(bins)
        ann = (
            f"Position density:\n"
            f"  H ≤ {fr['h_low_max']} nats: {fr['frac_below']*100:.1f}% (low-entropy mode,\n"
            f"                 α-knob has no effect)\n"
            f"  H ≥ {fr['h_high_min']} nats: {fr['frac_above']*100:.1f}% (decision region,\n"
            f"                 where α-knob diverges)\n"
            f"  N = {fr['total']:,} positions"
        )
        ax.text(0.98, 0.05, ann, transform=ax.transAxes,
                fontsize=8, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="gray", alpha=0.95))

        ax.set_xlim(0, h_centers[reliable].max() + 0.5 if reliable.any() else 4)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Per-token entropy H (nats)")
        ax.set_title(model.replace("--", "/"), fontsize=11)
        ax.grid(True, alpha=0.3)
        # Legend in lower-left (opposite corner from annotation)
        ax.legend(loc="lower left", fontsize=9)

    axes[0].set_ylabel("Mean surviving probability mass")
    fig.suptitle(
        "Survival mass vs entropy under p-less filter at α=2 and α=5\n"
        "(MBPP, top-32 truncation, recorded under α=2 sampling)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _model_jsonl_path(model_slug: str) -> Path:
    return Path(
        f"results/pless_alpha_entropy/{model_slug}/pless_t1.0.jsonl.entropy.jsonl"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Model slugs (e.g. Qwen--Qwen2.5-Coder-7B-Instruct).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bin-width", type=float, default=0.05,
                   help="Entropy bin width in nats.")
    p.add_argument("--h-max", type=float, default=4.0)
    p.add_argument("--validation-sample-size", type=int, default=500,
                   help="Random subsample for the 4 validation checks.")
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model processing
    model_to_bins: dict[str, dict] = {}
    validation_results: dict[str, dict] = {}
    for model in args.models:
        jsonl = _model_jsonl_path(model)
        if not jsonl.exists():
            print(f"[skip] no entropy jsonl for {model} at {jsonl}")
            continue
        print(f"[{model}] running validation on {args.validation_sample_size} records ...")
        validation_results[model] = validate(jsonl, n_sample=args.validation_sample_size)
        print(f"[{model}] processing full file: {jsonl}")
        data = process_model(jsonl, bin_width=args.bin_width, h_max=args.h_max)
        model_to_bins[model] = data
        n_total = sum(b["n_positions"] for b in data["bins"])
        print(f"[{model}] processed {n_total:,} positions into "
              f"{sum(1 for b in data['bins'] if b['n_positions'] > 0)} populated bins")

    # Persist data
    data_path = args.output_dir / "survival_vs_entropy_data.json"
    data_path.write_text(json.dumps({
        "bin_width_nats": args.bin_width,
        "h_max_nats": args.h_max,
        "per_model": model_to_bins,
    }, indent=2))
    print(f"\nWrote per-bin numerical data → {data_path}")

    # Validation report
    val_path = args.output_dir / "validation_report.md"
    lines = [
        "# Survival-curves validation report",
        "",
        "Standard rigor: 4 checks per `docs/theory/central_figure_plan.md`. Reported per model.",
        "",
    ]
    overall_pass = True
    for model, vr in validation_results.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"Random subsample size: **{vr['validation_sample_size']}**")
        lines.append("")
        for c in vr["checks"]:
            status = "✅ PASS" if c["passed"] else "❌ FAIL"
            lines.append(f"### {c['name']} — {status}")
            lines.append("")
            lines.append(f"- Acceptance: `{c['acceptance_criterion']}`")
            details = {k: v for k, v in c.items()
                       if k not in {"name", "passed", "acceptance_criterion"}}
            for k, v in details.items():
                lines.append(f"- {k}: `{v}`")
            lines.append("")
        # Check 4: per-bin sample size adequacy (computed from binned data)
        bins = model_to_bins.get(model, {}).get("bins", [])
        n_bins_total = len(bins)
        n_bins_populated = sum(1 for b in bins if b["n_positions"] > 0)
        n_bins_reliable = sum(1 for b in bins if b["n_positions"] >= 50)
        lines.append("### per_bin_sample_size_adequacy — informational")
        lines.append("")
        lines.append(f"- Total bins: {n_bins_total}")
        lines.append(f"- Populated bins (n_positions > 0): {n_bins_populated}")
        lines.append(f"- Reliable bins (n_positions ≥ 50): {n_bins_reliable}")
        lines.append("- The figure plots reliable bins as solid lines, "
                     "low-count bins as dotted/translucent so the reader "
                     "can see which range is statistically meaningful.")
        lines.append("")
        if not vr["all_passed"]:
            overall_pass = False
    lines.append("---")
    lines.append("")
    lines.append(f"**Overall**: {'✅ ALL CHECKS PASSED' if overall_pass else '❌ SOME CHECKS FAILED — DO NOT DECLARE FIGURE VALID'}")
    val_path.write_text("\n".join(lines))
    print(f"Wrote validation report → {val_path}")

    # Make plot
    if model_to_bins:
        plot_path = args.output_dir / "survival_vs_entropy.png"
        make_plot(model_to_bins, plot_path)
        print(f"Wrote figure → {plot_path}")


if __name__ == "__main__":
    main()
