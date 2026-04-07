"""Instrumented P-less generation for per-step token survivor analysis.

Loads a model, runs P-less generation on a stratified subset of MBPP tasks,
and captures per-step token survivor statistics. The instrumented sampler
wrapper intercepts probabilities BEFORE P-less modifies them in-place, records
statistics, then delegates to the real P-less sampler.

Usage::

    uv run python -m bench.eval.token_survivor_analysis \
        --model Qwen/Qwen2.5-Coder-3B-Instruct \
        --temperatures 0.6,1.0,1.5,2.0 \
        --n-samples 10 --n-tasks 30 --seed 42 \
        --output-dir results/token_survivor_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import p-less sampler via sys.path (same trick as bench/sampler_bridge.py)
# ---------------------------------------------------------------------------
_pless_dir = str(Path(__file__).resolve().parent.parent.parent / "p-less")
if _pless_dir not in sys.path:
    sys.path.insert(0, _pless_dir)

from p_less_samplers import p_less_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from bench.generator import generate_samples, load_model_and_tokenizer  # noqa: E402
from bench.prompts import (  # noqa: E402
    format_prompt_base_bigcode,
    format_prompt_instruct,
    is_instruct_model,
)

# Stop sequences for base models using bigcode prompt style.
# Duplicated from bench/runner.py:22 to avoid pulling in the full runner.
MBPP_BIGCODE_STOP_SEQUENCES = ["\nassert", "\nclass", "\nprint", '\n"""', "\nif __name__"]

# Smoothing constant — must match bench/generator.py:323
_PLESS_SMOOTH_ALPHA = 1e-3


# ── StepCollector ─────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """Statistics for a single generation step of a single sample."""

    step: int
    sample_idx: int
    broadcast: bool
    survivor_count: int
    threshold: float
    pre_entropy: float
    post_entropy: float
    max_survivor_prob: float
    survivor_prob_variance: float
    survivor_token_ids: list[int]
    survivor_probs: list[float]
    chosen_token_id: int
    mean_survivor_embedding_sim: float | None


@dataclass
class StepCollector:
    """Accumulates per-step generation records for survivor analysis.

    Parameters
    ----------
    eos_id : int
        End-of-sequence token id. Steps for already-finished samples are
        skipped.
    embedding_weight : torch.Tensor | None
        Model embedding matrix ``(vocab_size, hidden_dim)`` used for
        cosine similarity among survivors. ``None`` disables similarity.
    """

    eos_id: int
    embedding_weight: torch.Tensor | None = None
    records: list[StepRecord] = field(default_factory=list)
    _finished: set[int] = field(default_factory=set)
    _current_step: int = 0
    _is_broadcast: bool = False

    def set_step(self, step: int, *, broadcast: bool = False) -> None:
        self._current_step = step
        self._is_broadcast = broadcast

    def reset(self) -> None:
        """Reset state between tasks."""
        self.records.clear()
        self._finished.clear()
        self._current_step = 0
        self._is_broadcast = False

    def mark_finished(self, sample_idx: int) -> None:
        self._finished.add(sample_idx)

    def drain_records(self) -> list[dict]:
        """Return all records as plain dicts and clear the internal list."""
        out = []
        for r in self.records:
            d = {
                "step": r.step,
                "sample_idx": r.sample_idx,
                "broadcast": r.broadcast,
                "survivor_count": r.survivor_count,
                "threshold": r.threshold,
                "pre_entropy": r.pre_entropy,
                "post_entropy": r.post_entropy,
                "max_survivor_prob": r.max_survivor_prob,
                "survivor_prob_variance": r.survivor_prob_variance,
                "survivor_token_ids": r.survivor_token_ids,
                "survivor_probs": r.survivor_probs,
                "chosen_token_id": r.chosen_token_id,
                "mean_survivor_embedding_sim": r.mean_survivor_embedding_sim,
            }
            out.append(d)
        self.records.clear()
        return out

    # ── Statistics helpers ─────────────────────────────────────────────

    @staticmethod
    def _shannon_entropy(probs: torch.Tensor) -> float:
        """Shannon entropy in nats of a 1-D probability vector."""
        p = probs[probs > 0]
        return -(p * p.log()).sum().item()

    def _embedding_similarity(self, token_ids: torch.Tensor) -> float | None:
        """Mean pairwise cosine similarity among token embeddings.

        ``token_ids`` should contain at most 20 ids (caller caps).
        Returns ``None`` when there are fewer than 2 tokens or no
        embedding matrix is available.
        """
        if self.embedding_weight is None or len(token_ids) < 2:
            return None
        vecs = self.embedding_weight[token_ids]  # (k, hidden)
        vecs = torch.nn.functional.normalize(vecs, dim=-1)
        sim_matrix = vecs @ vecs.T  # (k, k)
        k = len(token_ids)
        # Mean of upper triangle (excluding diagonal)
        triu_sum = (sim_matrix.triu(diagonal=1)).sum().item()
        n_pairs = k * (k - 1) / 2
        return triu_sum / n_pairs if n_pairs > 0 else None

    # ── Main recording entry-point ────────────────────────────────────

    def record(
        self,
        sample_idx: int,
        pre_probs: torch.Tensor,
        threshold: torch.Tensor,
        survivor_mask: torch.Tensor,
        chosen_token_id: int,
    ) -> None:
        """Record statistics for one sample at the current step.

        All tensor arguments are expected to be 1-D (vocab_size,) and on the
        same device.
        """
        if sample_idx in self._finished:
            return

        survivor_probs_raw = pre_probs * survivor_mask.float()
        survivor_sum = survivor_probs_raw.sum()
        survivor_probs_renorm = survivor_probs_raw / survivor_sum if survivor_sum > 0 else survivor_probs_raw

        survivor_count = survivor_mask.sum().item()

        # Top-50 survivors by probability
        top_k = min(50, int(survivor_count))
        if top_k > 0:
            top_vals, top_ids = survivor_probs_renorm.topk(top_k)
            survivor_token_ids = top_ids.tolist()
            survivor_probs_list = [round(v, 6) for v in top_vals.tolist()]
        else:
            survivor_token_ids = []
            survivor_probs_list = []

        # Embedding similarity (cap at top 20)
        emb_ids = top_ids[:20] if top_k > 0 else torch.tensor([], dtype=torch.long)
        mean_emb_sim = self._embedding_similarity(emb_ids)

        # Entropy
        pre_entropy = self._shannon_entropy(pre_probs)
        post_entropy = self._shannon_entropy(survivor_probs_renorm)

        # Max and variance among renormalized survivors
        max_surv = survivor_probs_renorm.max().item() if survivor_count > 0 else 0.0
        surv_vals = survivor_probs_renorm[survivor_mask]
        variance = surv_vals.var().item() if len(surv_vals) > 1 else 0.0

        rec = StepRecord(
            step=self._current_step,
            sample_idx=sample_idx,
            broadcast=self._is_broadcast,
            survivor_count=int(survivor_count),
            threshold=round(threshold.item(), 8),
            pre_entropy=round(pre_entropy, 6),
            post_entropy=round(post_entropy, 6),
            max_survivor_prob=round(max_surv, 6),
            survivor_prob_variance=round(variance, 8),
            survivor_token_ids=survivor_token_ids,
            survivor_probs=survivor_probs_list,
            chosen_token_id=chosen_token_id,
            mean_survivor_embedding_sim=(
                round(mean_emb_sim, 6) if mean_emb_sim is not None else None
            ),
        )
        self.records.append(rec)

        # Track finished samples
        if chosen_token_id == self.eos_id:
            self._finished.add(sample_idx)


# ── Instrumented sampler factory ──────────────────────────────────────────


def make_instrumented_pless_sampler(collector: StepCollector):
    """Return a sampler_fn compatible with ``generate_samples``.

    The wrapper:
    1. Clones probs BEFORE ``p_less_decode`` mutates them in-place.
    2. Computes threshold and survivor mask on the clone.
    3. Delegates to the real ``p_less_decode(probs)`` for actual sampling.
    4. Records per-sample statistics in the collector.
    """

    def sampler(probs: torch.Tensor) -> torch.Tensor:
        # probs: (N, vocab) — may already have smoothing applied by generator
        N = probs.shape[0]

        # Clone BEFORE p_less_decode modifies in-place
        pre_probs = probs.clone()

        # Compute threshold and mask (mirrors p_less_samplers.py:25-26)
        threshold = pre_probs.square().sum(dim=-1, keepdim=True)  # (N, 1)
        survivor_mask = pre_probs >= threshold  # (N, vocab)

        # Delegate to real sampler
        next_tokens = p_less_decode(probs)  # (N, 1) — probs mutated in-place

        # Record per-sample statistics
        chosen_flat = next_tokens.view(N)
        for i in range(N):
            collector.record(
                sample_idx=i,
                pre_probs=pre_probs[i],
                threshold=threshold[i, 0],
                survivor_mask=survivor_mask[i],
                chosen_token_id=chosen_flat[i].item(),
            )

        return next_tokens

    return sampler


# ── Task selection ────────────────────────────────────────────────────────


def _load_task_tiers(
    metrics_path: Path,
    dataset,
) -> dict[int, str]:
    """Classify tasks into easy / medium / hard from a metrics JSON file.

    Returns a dict mapping task_id -> tier string.
    """
    with open(metrics_path) as f:
        metrics = json.load(f)

    per_task_raw = metrics.get("per_task", [])

    # per_task can be a list of dicts with "task_id" and "num_correct" keys
    # or a dict keyed by task_id string with "pass_rate" values.
    if isinstance(per_task_raw, list):
        per_task_map = {entry["task_id"]: entry for entry in per_task_raw}
    else:
        per_task_map = {int(k): v for k, v in per_task_raw.items()}

    tiers: dict[int, str] = {}
    dataset_ids = {task["task_id"] for task in dataset}

    for tid, entry in per_task_map.items():
        if tid not in dataset_ids:
            continue
        # Support both "num_correct" (direct count) and "pass_rate" (fraction)
        if "num_correct" in entry:
            n_correct = entry["num_correct"]
        else:
            n_correct = round(entry.get("pass_rate", 0.0) * 10)

        if n_correct == 10:
            tiers[tid] = "easy"
        elif 3 <= n_correct <= 7:
            tiers[tid] = "medium"
        elif n_correct <= 2:
            tiers[tid] = "hard"
        # Tasks with 8-9 correct are not cleanly tiered; skip them
    return tiers


def select_tasks(
    dataset,
    n_tasks: int,
    seed: int,
    metrics_path: Path | None,
    task_ids_override: list[int] | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Select a stratified subset of MBPP tasks.

    Returns ``(task_ids, task_tiers)`` where ``task_tiers`` maps each
    selected task_id to its difficulty tier.
    """
    if task_ids_override:
        tiers = {}
        if metrics_path and metrics_path.exists():
            tiers = _load_task_tiers(metrics_path, dataset)
        # Fill in "unknown" for tasks not in the metrics
        task_tiers = {tid: tiers.get(tid, "unknown") for tid in task_ids_override}
        return task_ids_override, task_tiers

    if metrics_path is None or not metrics_path.exists():
        warnings.warn(
            f"Metrics file not found ({metrics_path}). "
            "Falling back to random task selection without stratification.",
            stacklevel=2,
        )
        all_ids = [t["task_id"] for t in dataset]
        rng = random.Random(seed)
        selected = rng.sample(all_ids, min(n_tasks, len(all_ids)))
        return selected, {tid: "unknown" for tid in selected}

    tiers = _load_task_tiers(metrics_path, dataset)

    easy = [tid for tid, t in tiers.items() if t == "easy"]
    medium = [tid for tid, t in tiers.items() if t == "medium"]
    hard = [tid for tid, t in tiers.items() if t == "hard"]

    rng = random.Random(seed)
    per_tier = n_tasks // 3
    remainder = n_tasks - 3 * per_tier

    def _sample(pool: list[int], k: int) -> list[int]:
        return rng.sample(pool, min(k, len(pool)))

    selected_easy = _sample(easy, per_tier + (1 if remainder > 0 else 0))
    selected_medium = _sample(medium, per_tier + (1 if remainder > 1 else 0))
    selected_hard = _sample(hard, per_tier)

    selected = selected_easy + selected_medium + selected_hard
    task_tiers = {tid: tiers[tid] for tid in selected}

    print(f"Task selection: {len(selected_easy)} easy, "
          f"{len(selected_medium)} medium, {len(selected_hard)} hard "
          f"(total {len(selected)})")
    return selected, task_tiers


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instrumented P-less generation for token survivor analysis",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--temperatures", default="0.6,1.0,1.5,2.0",
        help="Comma-separated list of temperatures to sweep",
    )
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of samples per task")
    parser.add_argument("--n-tasks", type=int, default=30,
                        help="Number of MBPP tasks to run")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", default="results/token_survivor_analysis",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--metrics-file", default=None,
        help="Path to per-task metrics JSON for stratification. "
             "Defaults to results/full_mbpp_pre_post_temp_pless/"
             "<model>/metrics/pless_t1.5_metrics.json",
    )
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=None,
        help="Override task selection with specific task IDs",
    )
    return parser.parse_args()


# ── Summary printing ──────────────────────────────────────────────────────


def _print_summary(records: list[dict], temperature: float) -> None:
    """Print a quick summary of collected step data for one temperature."""
    n = len(records)
    if n == 0:
        print(f"  Temperature {temperature}: no records collected")
        return
    survivors = [r["survivor_count"] for r in records]
    mean_surv = sum(survivors) / n
    deterministic = sum(1 for s in survivors if s == 1)
    branching = sum(1 for s in survivors if s > 1)
    est_size_kb = len(json.dumps(records)) / 1024

    print(f"  Temperature {temperature}:")
    print(f"    Total steps recorded : {n}")
    print(f"    Mean survivor count  : {mean_surv:.2f}")
    print(f"    Deterministic (1 surv): {deterministic} ({100*deterministic/n:.1f}%)")
    print(f"    Branching (>1 surv)  : {branching} ({100*branching/n:.1f}%)")
    print(f"    Estimated JSON size  : {est_size_kb:.0f} KB")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]
    model_key = args.model.replace("/", "--")

    # Determine metrics file path
    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
    else:
        metrics_path = (
            Path("results/full_mbpp_pre_post_temp_pless")
            / model_key
            / "metrics"
            / "pless_t1.5_metrics.json"
        )

    # Load MBPP full test set
    print("Loading MBPP dataset...")
    dataset = load_dataset(
        "google-research-datasets/mbpp", "full", split="test",
    )
    dataset = dataset.map(lambda task: {"prompt": task["text"]})

    # Select tasks
    selected_ids, task_tiers = select_tasks(
        dataset=dataset,
        n_tasks=args.n_tasks,
        seed=args.seed,
        metrics_path=metrics_path,
        task_ids_override=args.task_ids,
    )
    task_id_set = set(selected_ids)
    tasks = [t for t in dataset if t["task_id"] in task_id_set]
    # Preserve selection order
    id_order = {tid: i for i, tid in enumerate(selected_ids)}
    tasks.sort(key=lambda t: id_order[t["task_id"]])
    print(f"Selected {len(tasks)} tasks")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    instruct = is_instruct_model(args.model)

    # Resolve EOS token id (same logic as bench/generator.py)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        for candidate in ("<|endoftext|>", "<|im_end|>"):
            cid = tokenizer.convert_tokens_to_ids(candidate)
            if cid is not None and cid != tokenizer.unk_token_id:
                eos_id = cid
                break
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Cannot determine eos_token_id for this tokenizer")

    # Get embedding matrix for similarity computation
    emb_weight = model.get_input_embeddings().weight.detach()

    # Stop sequences
    stop_strings: list[str] | None = None
    if not instruct:
        stop_strings = MBPP_BIGCODE_STOP_SEQUENCES

    # Output directory
    out_dir = Path(args.output_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run for each temperature
    for temperature in temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature: {temperature}")
        print(f"{'='*60}")

        collector = StepCollector(eos_id=eos_id, embedding_weight=emb_weight)
        instrumented_sampler = make_instrumented_pless_sampler(collector)

        all_task_steps: dict[str, list[dict]] = {}

        for task in tqdm(tasks, desc=f"pless @ T={temperature}"):
            task_id = task["task_id"]
            collector.reset()

            try:
                if instruct:
                    prompt_text, code_prefix = format_prompt_instruct(task, tokenizer)
                else:
                    prompt_text, code_prefix = format_prompt_base_bigcode(task)

                # Step 0 is a broadcast step (identical probs for all N samples)
                collector.set_step(0, broadcast=True)

                # Monkey-patch the collector's step counter to track steps
                # inside generate_samples. We wrap the sampler to update
                # the step counter automatically.
                _call_count = [0]  # mutable counter for closure
                _n_samples = args.n_samples

                def _tracking_sampler(probs: torch.Tensor) -> torch.Tensor:
                    step = _call_count[0]
                    collector.set_step(step, broadcast=(step == 0))
                    result = instrumented_sampler(probs)
                    _call_count[0] += 1
                    return result

                with torch.no_grad():
                    generate_samples(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        sampler_fn=_tracking_sampler,
                        n_samples=args.n_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temperature,
                        stop_strings=stop_strings,
                    )

                step_records = collector.drain_records()
                all_task_steps[str(task_id)] = step_records

                tqdm.write(
                    f"  task_id={task_id} ({task_tiers.get(task_id, 'unknown')}): "
                    f"{len(step_records)} step records"
                )

            except Exception as e:
                tqdm.write(f"  Error on task_id={task_id}: {e}")
                collector.reset()
                continue

        # Save results for this temperature
        output = {
            "model": args.model,
            "temperature": temperature,
            "n_samples": args.n_samples,
            "task_ids": selected_ids,
            "task_tiers": {str(tid): tier for tid, tier in task_tiers.items()},
            "tasks": all_task_steps,
        }

        out_path = out_dir / f"step_data_t{temperature}.json"
        with open(out_path, "w") as f:
            json.dump(output, f)
        print(f"Saved: {out_path}")

        # Print summary
        all_records = []
        for step_list in all_task_steps.values():
            all_records.extend(step_list)
        _print_summary(all_records, temperature)

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
