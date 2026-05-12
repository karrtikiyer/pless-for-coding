"""Layer-Entropy Probe: per-layer next-token distribution analysis via teacher forcing.

Sibling of ``bench/eval/phase_entropy_probe.py``.  For each correct sample
the model produced on HumanEval, we run a single teacher-forced forward pass
**with hooks on every transformer block** to capture the residual stream after
each block.  Each residual is projected through ``model.model.norm`` followed
by ``model.lm_head`` (the raw logit lens — no learned adapter) to obtain a
per-layer next-token distribution at every code-token position.  Per
(layer, position) we record:

  * entropy of the projected distribution
  * KL divergence to the *final layer's* distribution
  * whether the layer's top-1 token matches the final's top-1
  * whether the layer's top-1 sits in the final's top-5

The intended use is to test the hypothesis that, on an RLHF-tuned code model,
the penultimate (or earlier) layer retains higher entropy on code tokens than
the final layer, and that the top-1 of that earlier layer still frequently
agrees with the final's top-1 — i.e., earlier layers preserve plausible
diversity without sacrificing correctness.  Compare an instruct model's curves
against its base model on the same prompts and samples to read off the
*RLHF-specific* layer signature.

This probe is read-only (teacher forcing, no generation).  No correctness
re-execution: we trust the existing per-task ``pass_results`` from the metrics
JSON if provided, else fall back to ``check_sample`` like phase_entropy_probe.

Usage::

    uv run python -m bench.eval.layer_entropy_probe \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --results-file results/pless_human_eval_results/temprature_results/\\
Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/temp_t0.7.jsonl \\
        --output-dir results/layer_entropy_probe/Qwen2.5-Coder-7B-Instruct \\
        --max-tasks 164 --samples-per-task 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from bench.eval.phase_entropy_probe import (
    PHASES,
    _PHASE_COLORS,
    align_tokens_to_phases,
    classify_bytes,
)


# ---------------------------------------------------------------------------
# Hook-based per-layer teacher-forced projection
# ---------------------------------------------------------------------------


def _find_decoder_layers(model) -> list:
    """Find the list of transformer blocks.

    Supports the common ``model.model.layers`` layout (Qwen2, Llama, Mistral)
    and falls back to a search over named modules.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Could not locate decoder layer list on model")


def _find_final_norm(model):
    """Locate the final pre-unembedding norm. Returns identity if absent."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return torch.nn.Identity()


def teacher_forced_per_layer(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> dict | None:
    """Single forward pass with hooks on every transformer block.

    Returns a dict of per-layer stats at the *code* positions only:

    .. code-block:: python

        {
            "n_code": int,
            "n_layers": int,
            "final_top1": list[int],          # length n_code
            "per_layer": [                    # length n_layers
                {
                    "entropy":        list[float],
                    "kl_to_final":    list[float],
                    "top1":           list[int],
                    "matches_final":  list[bool],
                    "in_top5_final":  list[bool],
                },
                ...
            ],
        }
    """
    decoder_layers = _find_decoder_layers(model)
    final_norm = _find_final_norm(model)
    lm_head = model.lm_head

    captured: list[torch.Tensor] = []

    def _make_hook(_idx):
        def _hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h.detach())
        return _hook

    handles = [
        block.register_forward_hook(_make_hook(i))
        for i, block in enumerate(decoder_layers)
    ]

    try:
        with torch.no_grad():
            out = model(input_ids)
        final_logits = out.logits[0]  # (seq, vocab)
    finally:
        for h in handles:
            h.remove()

    n_code = int(input_ids.shape[1] - prompt_len)
    if n_code <= 0 or not captured:
        return None

    # Logits at position t predict token at t+1, so code-prediction positions
    # are [prompt_len - 1, ..., prompt_len - 1 + n_code - 1].
    pos = torch.arange(prompt_len - 1, prompt_len - 1 + n_code,
                       device=final_logits.device)

    # Final-layer reference distribution at code positions.
    final_probs = torch.softmax(final_logits.index_select(0, pos).float(), dim=-1)
    log_final = torch.log(final_probs + 1e-12)
    final_top1 = final_probs.argmax(dim=-1)
    final_top5 = final_probs.topk(5, dim=-1).indices

    per_layer = []
    for resid in captured:
        # resid: (1, seq, hidden) — take only code positions then norm + unembed.
        h = resid[0].index_select(0, pos)
        h = final_norm(h)
        logits = lm_head(h)  # (n_code, vocab)
        probs = torch.softmax(logits.float(), dim=-1)
        log_p = torch.log(probs + 1e-12)
        ent = -(probs * log_p).sum(dim=-1)
        kl = (probs * (log_p - log_final)).sum(dim=-1)
        top1 = probs.argmax(dim=-1)
        matches_final = top1 == final_top1
        # Per-row "is top1 in final's top5?" — vectorised
        in_top5 = (final_top5 == top1.unsqueeze(-1)).any(dim=-1)
        per_layer.append({
            "entropy": ent.cpu().tolist(),
            "kl_to_final": kl.cpu().tolist(),
            "top1": top1.cpu().tolist(),
            "matches_final": matches_final.cpu().tolist(),
            "in_top5_final": in_top5.cpu().tolist(),
        })

    # Free GPU memory eagerly.
    del captured, final_probs, log_final, final_top5

    return {
        "n_code": n_code,
        "n_layers": len(per_layer),
        "final_top1": final_top1.cpu().tolist(),
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def collect_layer_entropies(
    model,
    tokenizer,
    results_file: Path,
    output_dir: Path,
    max_tasks: int,
    samples_per_task: int,
    device: str,
) -> tuple[list[dict], int]:
    """Run the probe over a HumanEval results JSONL.

    Returns (rows, n_layers).  Each row has the per-(task, sample, code-pos,
    layer) stats.  Rows are streamed to ``output_dir/per_token_data.csv`` as
    they are produced.
    """
    from human_eval.data import read_problems

    from bench.eval.executor import check_sample, extract_python_code
    from bench.eval.loader import load_results
    from bench.humaneval.prompts import format_prompt_instruct, is_instruct_model

    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(results_file)
    problems = read_problems()
    model_id = records[0]["model"] if records else "unknown"
    use_instruct = is_instruct_model(model_id)

    csv_path = output_dir / "per_token_data.csv"
    csv_fields = [
        "task_id", "sample_idx", "code_pos", "layer_idx", "phase", "token",
        "entropy", "kl_to_final", "top1_id", "matches_final", "in_top5_final",
    ]
    n_layers_seen = 0
    n_rows = 0
    n_passing = 0

    with csv_path.open("w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()

        for record in tqdm(records[:max_tasks], desc="Tasks"):
            task_id = record["task_id"]
            problem = problems.get(task_id)
            if problem is None:
                continue

            samples = record.get("samples", [])[:samples_per_task]
            test_code = record.get("test", problem.get("test", ""))
            entry_point = record.get("entry_point", problem.get("entry_point", ""))

            for s_idx, raw_code in enumerate(samples):
                code = extract_python_code(raw_code)
                if not code.strip():
                    continue

                full_program = f"{code}\n\n{test_code}\ncheck({entry_point})\n"
                if not check_sample(full_program, timeout=5.0):
                    continue
                n_passing += 1

                phase_bytes = classify_bytes(code)
                if not phase_bytes:
                    continue

                prompt = record.get("prompt_text", "")
                if not prompt:
                    if use_instruct:
                        prompt, _ = format_prompt_instruct(problem, tokenizer)
                    else:
                        prompt = problem.get("prompt", "")
                prompt_ids = (
                    prompt if isinstance(prompt, list)
                    else tokenizer.encode(prompt, add_special_tokens=False)
                )
                code_ids = tokenizer.encode(code, add_special_tokens=False)
                if not code_ids:
                    continue

                full_ids = torch.tensor([prompt_ids + code_ids], device=device)
                max_len = getattr(model.config, "max_position_embeddings", 8192)
                if full_ids.shape[1] > max_len:
                    continue

                try:
                    stats = teacher_forced_per_layer(model, full_ids, len(prompt_ids))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache() if device == "cuda" else None
                        print(f"  OOM on {task_id}[{s_idx}]: {e}")
                        continue
                    raise
                if stats is None:
                    continue
                n_layers_seen = stats["n_layers"]

                token_phases = align_tokens_to_phases(tokenizer, code, phase_bytes)
                if len(token_phases) != len(code_ids):
                    continue

                for layer_idx, layer_stats in enumerate(stats["per_layer"]):
                    for code_pos in range(len(code_ids)):
                        writer.writerow({
                            "task_id": task_id,
                            "sample_idx": s_idx,
                            "code_pos": code_pos,
                            "layer_idx": layer_idx,
                            "phase": token_phases[code_pos],
                            "token": tokenizer.decode([code_ids[code_pos]]),
                            "entropy": layer_stats["entropy"][code_pos],
                            "kl_to_final": layer_stats["kl_to_final"][code_pos],
                            "top1_id": layer_stats["top1"][code_pos],
                            "matches_final": int(layer_stats["matches_final"][code_pos]),
                            "in_top5_final": int(layer_stats["in_top5_final"][code_pos]),
                        })
                        n_rows += 1

    print(f"\nWrote {n_rows:,} rows from {n_passing} passing samples → {csv_path}")
    return csv_path, n_layers_seen


# ---------------------------------------------------------------------------
# Aggregation + stats
# ---------------------------------------------------------------------------


def _read_csv(csv_path: Path) -> dict:
    """Stream the CSV and accumulate per-(layer, phase) arrays."""
    data: dict[tuple[int, str], dict[str, list]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["layer_idx"]), row["phase"])
            d = data.setdefault(key, {
                "entropy": [], "kl_to_final": [],
                "matches_final": [], "in_top5_final": [],
            })
            d["entropy"].append(float(row["entropy"]))
            d["kl_to_final"].append(float(row["kl_to_final"]))
            d["matches_final"].append(int(row["matches_final"]))
            d["in_top5_final"].append(int(row["in_top5_final"]))
    return data


def compute_stats(csv_path: Path, n_layers: int) -> dict:
    """Aggregate per-layer (overall and per-phase) statistics from the CSV."""
    per = _read_csv(csv_path)

    # Helper: list -> summary
    def _summary(xs):
        a = np.asarray(xs, dtype=float)
        if a.size == 0:
            return {"count": 0}
        return {
            "count": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    # Per-layer aggregate (across all phases)
    per_layer_all = {}
    for layer_idx in range(n_layers):
        ent = []
        kl = []
        match = []
        top5 = []
        for phase in PHASES:
            d = per.get((layer_idx, phase))
            if not d:
                continue
            ent.extend(d["entropy"])
            kl.extend(d["kl_to_final"])
            match.extend(d["matches_final"])
            top5.extend(d["in_top5_final"])
        per_layer_all[layer_idx] = {
            "entropy": _summary(ent),
            "kl_to_final": _summary(kl),
            "top1_agreement": float(np.mean(match)) if match else 0.0,
            "top5_agreement": float(np.mean(top5)) if top5 else 0.0,
        }

    # Per-(layer, phase)
    per_layer_phase = {}
    for layer_idx in range(n_layers):
        per_layer_phase[layer_idx] = {}
        for phase in PHASES:
            d = per.get((layer_idx, phase))
            if not d:
                per_layer_phase[layer_idx][phase] = {"count": 0}
                continue
            per_layer_phase[layer_idx][phase] = {
                "entropy": _summary(d["entropy"]),
                "kl_to_final": _summary(d["kl_to_final"]),
                "top1_agreement": float(np.mean(d["matches_final"])),
                "top5_agreement": float(np.mean(d["in_top5_final"])),
            }

    # Headline: penultimate vs final entropy gap and top-1 agreement
    if n_layers >= 2:
        final_ent_mean = per_layer_all[n_layers - 1]["entropy"].get("mean", 0.0)
        penult_ent_mean = per_layer_all[n_layers - 2]["entropy"].get("mean", 0.0)
        penult_top1_agree = per_layer_all[n_layers - 2]["top1_agreement"]
    else:
        final_ent_mean = penult_ent_mean = penult_top1_agree = 0.0

    return {
        "n_layers": n_layers,
        "headline": {
            "final_layer_entropy_mean": final_ent_mean,
            "penultimate_layer_entropy_mean": penult_ent_mean,
            "penult_minus_final_entropy_gap": penult_ent_mean - final_ent_mean,
            "penultimate_top1_agreement_with_final": penult_top1_agree,
        },
        "per_layer_all": per_layer_all,
        "per_layer_phase": per_layer_phase,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _per_layer_means(stats: dict, key: str = "entropy") -> np.ndarray:
    n = stats["n_layers"]
    return np.array([
        stats["per_layer_all"][i][key].get("mean", np.nan)
        if isinstance(stats["per_layer_all"][i][key], dict)
        else stats["per_layer_all"][i][key]
        for i in range(n)
    ])


def plot_entropy_curve(stats: dict, output_path: Path):
    n = stats["n_layers"]
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall (across all phases) — bold
    ys = _per_layer_means(stats, "entropy")
    ax.plot(xs, ys, "-", color="black", linewidth=2.5, label="all phases",
            zorder=10)

    # Per-phase faceted curves
    for phase in PHASES:
        ys_p = np.array([
            stats["per_layer_phase"][i].get(phase, {}).get("entropy", {}).get(
                "mean", np.nan)
            for i in range(n)
        ])
        ax.plot(xs, ys_p, "--", color=_PHASE_COLORS[phase], alpha=0.85,
                linewidth=1.5, label=phase)

    ax.set_xlabel("Layer index (0 = first decoder block)")
    ax.set_ylabel("Mean entropy (nats)")
    ax.set_title("Per-Layer Next-Token Entropy at Code Positions")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


def plot_kl_curve(stats: dict, output_path: Path):
    n = stats["n_layers"]
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 6))

    ys = _per_layer_means(stats, "kl_to_final")
    ax.plot(xs, ys, "-", color="black", linewidth=2.5, label="all phases",
            zorder=10)
    for phase in PHASES:
        ys_p = np.array([
            stats["per_layer_phase"][i].get(phase, {}).get(
                "kl_to_final", {}).get("mean", np.nan)
            for i in range(n)
        ])
        ax.plot(xs, ys_p, "--", color=_PHASE_COLORS[phase], alpha=0.85,
                linewidth=1.5, label=phase)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("KL(layer || final), nats")
    ax.set_title("Divergence of Each Layer's Belief from the Final Layer")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


def plot_top1_agreement(stats: dict, output_path: Path):
    n = stats["n_layers"]
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 6))

    top1 = np.array([stats["per_layer_all"][i]["top1_agreement"]
                     for i in range(n)])
    top5 = np.array([stats["per_layer_all"][i]["top5_agreement"]
                     for i in range(n)])
    ax.plot(xs, top1, "-", color="#1565C0", linewidth=2.5,
            label="top-1 layer == top-1 final")
    ax.plot(xs, top5, "-", color="#E64A19", linewidth=2.0,
            label="top-1 layer ∈ top-5 final")
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Agreement fraction")
    ax.set_title("Per-Layer Top-1 Agreement with Final Layer (Code Positions)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--results-file", required=True, type=Path,
                        help="HumanEval JSONL results file (correct samples will be re-verified)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tasks", type=int, default=164,
                        help="Max tasks (default: full HumanEval = 164)")
    parser.add_argument("--samples-per-task", type=int, default=1,
                        help="Samples per task (default: 1; we care about distribution shape, not sample variance)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / mps / cpu (auto if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading model: {args.model}")
    from bench.generator import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    csv_path, n_layers = collect_layer_entropies(
        model, tokenizer, args.results_file, args.output_dir,
        max_tasks=args.max_tasks,
        samples_per_task=args.samples_per_task,
        device=device,
    )

    if n_layers == 0:
        print("No data collected — bailing.")
        return

    print("\nAggregating per-layer statistics...")
    stats = compute_stats(csv_path, n_layers)
    stats_path = args.output_dir / "layer_entropy_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved: {stats_path}")

    h = stats["headline"]
    print("\n" + "=" * 60)
    print("LAYER ENTROPY HEADLINE")
    print("=" * 60)
    print(f"  final-layer entropy (mean, code tokens):   {h['final_layer_entropy_mean']:.3f}")
    print(f"  penultimate-layer entropy:                 {h['penultimate_layer_entropy_mean']:.3f}")
    print(f"  gap (penult − final):                      {h['penult_minus_final_entropy_gap']:+.3f}")
    print(f"  penultimate top-1 agreement with final:    {h['penultimate_top1_agreement_with_final']:.3f}")

    print("\nGenerating plots...")
    plot_entropy_curve(stats, args.output_dir / "layer_entropy_curve.png")
    plot_kl_curve(stats, args.output_dir / "layer_kl_curve.png")
    plot_top1_agreement(stats, args.output_dir / "layer_top1_agreement.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
