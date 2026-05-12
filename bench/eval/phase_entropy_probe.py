"""D2 Phase-Oracle Probe: per-phase entropy analysis via teacher forcing.

Measures whether different AST phases (signature, body, docstring, operator)
exhibit meaningfully different entropy distributions in a code LLM.  Uses
teacher-forced forward passes over existing correct solutions — no generation.

See docs/experiments/d2_phase_entropy_probe.md for experiment design and
decision criteria.

Usage::

    uv run python -m bench.eval.phase_entropy_probe \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --results-file results/pless_human_eval_results/temprature_results/\
Qwen--Qwen2.5-Coder-7B-Instruct/humaneval/temp_t0.7.jsonl \
        --output-dir results/phase_entropy_probe/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tree-sitter setup
# ---------------------------------------------------------------------------

PHASES = ("signature", "body", "docstring", "operator")

_PHASE_COLORS = {
    "signature": "#2196F3",
    "body": "#4CAF50",
    "docstring": "#FF9800",
    "operator": "#9C27B0",
}


def _get_parser():
    """Create a tree-sitter Python parser (cached on first call)."""
    import tree_sitter_python as tsp
    from tree_sitter import Language, Parser

    from bench.eval.metrics import _patch_tree_sitter_capsule
    _patch_tree_sitter_capsule()

    py_lang = Language(tsp.language())
    parser = Parser()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser.set_language(py_lang)
    return parser


_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = _get_parser()
    return _PARSER


# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------


def classify_bytes(source: str) -> list[str]:
    """Classify every byte of Python source into an AST phase.

    Returns a list of length ``len(source.encode('utf-8'))`` where each entry
    is one of: "signature", "body", "docstring", "operator".

    Strategy: default every byte to "operator", then paint more specific phases
    on top as we walk the AST.
    """
    source_bytes = source.encode("utf-8")
    n = len(source_bytes)
    if n == 0:
        return []

    labels = ["operator"] * n
    tree = _parser().parse(source_bytes)

    def _paint(start: int, end: int, phase: str):
        for i in range(max(0, start), min(n, end)):
            labels[i] = phase

    def _walk_function(func_node):
        """Label children of a function_definition node."""
        block_node = None
        for child in func_node.children:
            if child.type == "block":
                block_node = child
                break

        if block_node is None:
            return

        # Paint everything from function start to block start as signature.
        # This catches def, name, params, ->, type, :, and inter-node whitespace.
        _paint(func_node.start_byte, block_node.start_byte, "signature")

        # Check for docstring: first child of block that is expression_statement
        # containing a string node
        first_stmt = block_node.children[0] if block_node.children else None
        docstring_end = -1
        if (first_stmt is not None
                and first_stmt.type == "expression_statement"
                and first_stmt.children
                and first_stmt.children[0].type == "string"):
            _paint(first_stmt.start_byte, first_stmt.end_byte, "docstring")
            docstring_end = first_stmt.end_byte

        # Everything else in the block is body (will be refined below)
        for child in block_node.children:
            if child.end_byte <= docstring_end:
                continue
            _paint(child.start_byte, child.end_byte, "body")

        # Recurse into nested functions
        _walk_block(block_node)

    def _walk_block(node):
        """Recursively find function_definitions inside a block/module."""
        for child in node.children:
            if child.type == "function_definition":
                _walk_function(child)
            elif child.type == "decorated_definition":
                # @decorator above a function/class — paint decorator as signature
                for cc in child.children:
                    if cc.type == "decorator":
                        _paint(cc.start_byte, cc.end_byte, "signature")
                    elif cc.type == "function_definition":
                        _walk_function(cc)
                    elif cc.type == "class_definition":
                        _walk_class(cc)
            elif child.type == "class_definition":
                _walk_class(child)
            elif child.type in ("if_statement", "for_statement",
                                "while_statement", "try_statement",
                                "with_statement", "elif_clause",
                                "else_clause", "except_clause",
                                "finally_clause"):
                for cc in child.children:
                    if cc.type == "block":
                        _walk_block(cc)

    def _walk_class(class_node):
        """Handle class_definition: recurse into its block."""
        class_block = None
        for cc in class_node.children:
            if cc.type == "block":
                class_block = cc
                break
        if class_block:
            _walk_block(class_block)

    # Also paint top-level comments as docstring
    def _paint_comments(node):
        if node.type == "comment":
            _paint(node.start_byte, node.end_byte, "docstring")
        for child in node.children:
            _paint_comments(child)

    # Walk module top-level
    root = tree.root_node
    for child in root.children:
        if child.type == "function_definition":
            _walk_function(child)
        elif child.type == "decorated_definition":
            for cc in child.children:
                if cc.type == "decorator":
                    _paint(cc.start_byte, cc.end_byte, "signature")
                elif cc.type == "function_definition":
                    _walk_function(cc)
                elif cc.type == "class_definition":
                    _walk_class(cc)
        elif child.type == "class_definition":
            _walk_class(child)
        elif child.type not in ("comment",):
            # Top-level non-function code (imports, assignments) → body
            _paint(child.start_byte, child.end_byte, "body")

    _paint_comments(root)

    return labels


# ---------------------------------------------------------------------------
# Teacher-forced entropy
# ---------------------------------------------------------------------------


def teacher_forced_entropy(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> list[float]:
    """Compute per-token entropy for code tokens via a single forward pass.

    Args:
        model: HuggingFace causal LM.
        input_ids: (1, seq_len) tensor of prompt + code token ids.
        prompt_len: Number of tokens in the prompt prefix.

    Returns:
        List of entropy values (nats), one per code token.  The i-th entry
        is the entropy of the model's predictive distribution at position
        ``prompt_len + i`` (i.e., predicting code token i given all preceding
        tokens).
    """
    device = input_ids.device
    with torch.no_grad():
        outputs = model(input_ids)
    # logits shape: (1, seq_len, vocab_size)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Code tokens start at position prompt_len in input_ids.
    # Logits at position t predict token at position t+1.
    # So logits[prompt_len - 1] predicts the first code token,
    # and logits[prompt_len - 1 + i] predicts code token i.
    n_code = input_ids.shape[1] - prompt_len
    if n_code <= 0:
        return []

    code_logits = logits[prompt_len - 1: prompt_len - 1 + n_code]  # (n_code, vocab)
    # Compute entropy in float32 for numerical stability
    probs = torch.softmax(code_logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropies = -(probs * log_probs).sum(dim=-1)  # (n_code,)
    return entropies.cpu().tolist()


# ---------------------------------------------------------------------------
# Token-to-phase alignment
# ---------------------------------------------------------------------------


def align_tokens_to_phases(
    tokenizer,
    code: str,
    phase_bytes: list[str],
) -> list[str]:
    """Map each token of ``code`` to its AST phase.

    Uses the tokenizer's offset mapping to find which byte(s) each token
    covers, then assigns the phase of the token's first byte.

    Returns a list of phase labels, one per token.
    """
    enc = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    token_phases = []
    for start, end in offsets:
        if start == end:
            # Zero-width token (BOS/EOS padding) — default to operator
            token_phases.append("operator")
        elif start < len(phase_bytes):
            token_phases.append(phase_bytes[start])
        else:
            token_phases.append("operator")
    return token_phases


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def collect_phase_entropies(
    model,
    tokenizer,
    results_file: Path,
    output_dir: Path,
    max_samples_per_task: int = 10,
    device: str = "cpu",
) -> list[dict]:
    """Run the full phase-entropy pipeline over HumanEval results.

    Returns list of per-token records: {token, phase, entropy, task_id, sample_idx}.
    Also saves per_token_data.csv to output_dir.
    """
    from human_eval.data import read_problems

    from bench.eval.executor import check_sample, extract_python_code
    from bench.eval.loader import load_results
    from bench.humaneval.prompts import format_prompt_instruct, is_instruct_model

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results and HumanEval problems
    records = load_results(results_file)
    problems = read_problems()
    model_id = records[0]["model"] if records else "unknown"
    use_instruct = is_instruct_model(model_id)

    all_rows = []
    n_passing = 0
    n_total = 0

    for record in tqdm(records, desc="Tasks"):
        task_id = record["task_id"]
        problem = problems.get(task_id)
        if problem is None:
            continue

        samples = record.get("samples", [])[:max_samples_per_task]
        test_code = record.get("test", problem.get("test", ""))
        entry_point = record.get("entry_point", problem.get("entry_point", ""))

        for s_idx, raw_code in enumerate(samples):
            n_total += 1
            code = extract_python_code(raw_code)
            if not code.strip():
                continue

            # Check correctness
            full_program = f"{code}\n\n{test_code}\ncheck({entry_point})\n"
            if not check_sample(full_program, timeout=5.0):
                continue
            n_passing += 1

            # Phase-classify the code bytes
            phase_bytes = classify_bytes(code)
            if not phase_bytes:
                continue

            # Use the stored prompt_text (already has chat template applied)
            prompt = record.get("prompt_text", "")
            if not prompt:
                # Fallback: reconstruct from HumanEval problem
                if use_instruct:
                    prompt, _ = format_prompt_instruct(problem, tokenizer)
                else:
                    prompt = problem.get("prompt", "")

            # Tokenize prompt and code separately
            if isinstance(prompt, list):
                # Already tokenized (some instruct formatters return token ids)
                prompt_ids = prompt
            else:
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            code_ids = tokenizer.encode(code, add_special_tokens=False)
            if not code_ids:
                continue

            # Build full input
            full_ids = torch.tensor([prompt_ids + code_ids], device=device)

            # Guard against exceeding model context
            max_len = getattr(model.config, "max_position_embeddings", 8192)
            if full_ids.shape[1] > max_len:
                continue

            # Teacher-forced entropy
            try:
                entropies = teacher_forced_entropy(model, full_ids, len(prompt_ids))
            except Exception as e:
                print(f"  Warning: forward pass failed for {task_id}[{s_idx}]: {e}")
                continue

            if len(entropies) != len(code_ids):
                # Length mismatch — skip (shouldn't happen but be safe)
                continue

            # Align tokens to phases
            token_phases = align_tokens_to_phases(tokenizer, code, phase_bytes)
            if len(token_phases) != len(code_ids):
                # Alignment mismatch
                continue

            # Collect rows
            for t_idx, (ent, phase) in enumerate(zip(entropies, token_phases)):
                token_text = tokenizer.decode([code_ids[t_idx]])
                all_rows.append({
                    "token": token_text,
                    "phase": phase,
                    "entropy": ent,
                    "task_id": task_id,
                    "sample_idx": s_idx,
                })

    print(f"\nCollected {len(all_rows)} token records from "
          f"{n_passing}/{n_total} passing samples")

    # Save CSV
    csv_path = output_dir / "per_token_data.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["token", "phase", "entropy",
                                                    "task_id", "sample_idx"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved: {csv_path}")

    return all_rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(rows: list[dict]) -> dict:
    """Compute per-phase summary statistics and pairwise effect sizes."""
    by_phase = {}
    for row in rows:
        by_phase.setdefault(row["phase"], []).append(row["entropy"])

    stats = {
        "total_tokens": len(rows),
        "per_phase": {},
        "pairwise_cohens_d": {},
        "pairwise_ks_pvalue": {},
    }

    for phase in PHASES:
        vals = np.array(by_phase.get(phase, []))
        if len(vals) == 0:
            stats["per_phase"][phase] = {"count": 0}
            continue
        stats["per_phase"][phase] = {
            "count": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "p5": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
        }

    # Pairwise Cohen's d and KS test
    from scipy.stats import ks_2samp

    for p1, p2 in combinations(PHASES, 2):
        v1 = np.array(by_phase.get(p1, []))
        v2 = np.array(by_phase.get(p2, []))
        key = f"{p1}_vs_{p2}"

        if len(v1) < 2 or len(v2) < 2:
            stats["pairwise_cohens_d"][key] = None
            stats["pairwise_ks_pvalue"][key] = None
            continue

        # Cohen's d (pooled)
        n1, n2 = len(v1), len(v2)
        s1, s2 = np.std(v1, ddof=1), np.std(v2, ddof=1)
        pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = abs(np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0.0
        stats["pairwise_cohens_d"][key] = round(d, 4)

        # KS test
        ks_stat, ks_p = ks_2samp(v1, v2)
        stats["pairwise_ks_pvalue"][key] = float(ks_p)

    # Conditional entropy bins (10 bins)
    all_ent = np.array([r["entropy"] for r in rows])
    bin_edges = np.linspace(all_ent.min(), all_ent.max() + 1e-9, 11)
    conditional_bins = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_label = f"{lo:.1f}-{hi:.1f}"
        bin_counts = {"bin": bin_label}
        for phase in PHASES:
            phase_ent = np.array(by_phase.get(phase, []))
            bin_counts[phase] = int(np.sum((phase_ent >= lo) & (phase_ent < hi)))
        conditional_bins.append(bin_counts)
    stats["conditional_entropy_bins"] = conditional_bins

    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_kde(rows: list[dict], output_path: Path):
    """Overlaid KDE curves per phase — the primary/decisive plot."""
    from scipy.stats import gaussian_kde

    by_phase = {}
    for row in rows:
        by_phase.setdefault(row["phase"], []).append(row["entropy"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for phase in PHASES:
        vals = np.array(by_phase.get(phase, []))
        if len(vals) < 10:
            continue
        kde = gaussian_kde(vals, bw_method=0.15)
        x = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 500)
        y = kde(x)
        color = _PHASE_COLORS[phase]
        ax.plot(x, y, color=color, linewidth=2,
                label=f"{phase} (n={len(vals):,}, med={np.median(vals):.2f})")
        ax.axvline(np.median(vals), color=color, linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Entropy (nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Per-Token Entropy by AST Phase", fontsize=14)
    ax.legend(fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


def plot_boxplot(rows: list[dict], output_path: Path):
    """Side-by-side box plots per phase."""
    by_phase = {}
    for row in rows:
        by_phase.setdefault(row["phase"], []).append(row["entropy"])

    data = [np.array(by_phase.get(p, [])) for p in PHASES]
    colors = [_PHASE_COLORS[p] for p in PHASES]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, labels=PHASES, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Entropy (nats)", fontsize=12)
    ax.set_title("Entropy Distribution by AST Phase", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


def plot_conditional(rows: list[dict], stats: dict, output_path: Path):
    """Conditional plot: within each entropy bin, show phase proportions.

    This is the diagnostic that separates 'phases differ in entropy' (already
    known) from 'phase adds info beyond entropy' (the actual question).
    """
    bins = stats["conditional_entropy_bins"]
    if not bins:
        return

    bin_labels = [b["bin"] for b in bins]
    phase_proportions = {p: [] for p in PHASES}

    for b in bins:
        total = sum(b.get(p, 0) for p in PHASES)
        for p in PHASES:
            phase_proportions[p].append(b.get(p, 0) / total if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bin_labels))
    bottom = np.zeros(len(bin_labels))

    for phase in PHASES:
        vals = np.array(phase_proportions[phase])
        ax.bar(x, vals, bottom=bottom, color=_PHASE_COLORS[phase],
               label=phase, alpha=0.8, width=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Entropy bin (nats)", fontsize=12)
    ax.set_ylabel("Phase proportion", fontsize=12)
    ax.set_title("Phase Distribution Within Entropy Bins\n"
                 "(Uniform = phase is redundant given entropy; "
                 "Non-uniform = phase adds info)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="D2 Phase-Oracle Probe: per-phase entropy analysis")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--results-file", required=True, type=Path,
                        help="Path to HumanEval JSONL results file")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for output plots, stats, and CSV")
    parser.add_argument("--max-samples-per-task", type=int, default=10,
                        help="Max samples to process per task (default: 10)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, or cpu (auto-detect if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    from bench.generator import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Run pipeline
    rows = collect_phase_entropies(
        model, tokenizer, args.results_file, args.output_dir,
        max_samples_per_task=args.max_samples_per_task,
        device=device,
    )

    if not rows:
        print("No data collected — check results file and model.")
        return

    # Statistics
    print("\nComputing statistics...")
    stats = compute_stats(rows)
    stats_path = args.output_dir / "phase_entropy_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE ENTROPY SUMMARY")
    print("=" * 60)
    for phase in PHASES:
        ps = stats["per_phase"].get(phase, {})
        if ps.get("count", 0) == 0:
            print(f"  {phase:12s}: no tokens")
            continue
        print(f"  {phase:12s}: n={ps['count']:>6,}  "
              f"mean={ps['mean']:.3f}  median={ps['median']:.3f}  "
              f"std={ps['std']:.3f}  [p5={ps['p5']:.2f}, p95={ps['p95']:.2f}]")

    print("\nPairwise Cohen's d:")
    max_d = 0.0
    for key, d in stats["pairwise_cohens_d"].items():
        if d is not None:
            marker = ""
            if d > 0.8:
                marker = " *** LARGE"
            elif d > 0.5:
                marker = " ** MEDIUM"
            elif d > 0.2:
                marker = " * SMALL"
            print(f"  {key:30s}: d={d:.4f}{marker}")
            max_d = max(max_d, d)

    print(f"\nMax Cohen's d: {max_d:.4f}")
    if max_d > 0.8:
        print("VERDICT: Strong phase separation — D2 premise holds")
    elif max_d > 0.5:
        print("VERDICT: Moderate phase separation — check conditional plot")
    elif max_d > 0.2:
        print("VERDICT: Weak phase separation — D2 unlikely to help much")
    else:
        print("VERDICT: No meaningful phase separation — D2 is dead")

    # Plots
    print("\nGenerating plots...")
    plot_kde(rows, args.output_dir / "phase_entropy_kde.png")
    plot_boxplot(rows, args.output_dir / "phase_entropy_boxplot.png")
    plot_conditional(rows, stats, args.output_dir / "phase_entropy_conditional.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
