"""Claude-Sonnet judge for the algosim NAUADC clustering protocol.

Reimplements ``algosim/clustering_solutions.py:create_solution_group``
verbatim but swaps Llama-3.1-8B-Instruct for Claude Sonnet 4.6 as the
pairwise judge. The prompt INSTRUCTION + envelope and the decision
regex are copied verbatim from ``algosim/infer_algosim.py`` (constants
below) so we do not depend on the submodule at runtime.

Output schema matches ``algosim_data/responses/<config>.parquet`` so
``bench.eval.algosim_report`` works against the new directory unchanged:

    columns: problem_id, question, solutions, group_index, records
    records[i] = {solution_index, past_solution_index, output, response}

Usage (smoke test):
    uv run python -m bench.eval.algosim_claude_judge \\
        --configs H8P --max-tasks 5 --workers 2

Usage (full 8-config sweep):
    uv run python -m bench.eval.algosim_claude_judge \\
        --configs H8P,H9P,H7P,H10P,T15N,C,P15,A \\
        --workers 8

The script streams per-problem JSONL checkpoints alongside the final
parquet so a crash mid-config can resume without re-paying for already
completed problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from anthropic._exceptions import APIError, RateLimitError

# --- Verbatim from algosim/infer_algosim.py ---------------------------------

INSTRUCTION = (
    'Your task is to classify whether a given solution solves a problem with '
    'similar logic to the existing solution or whether it leverages a novel '
    'approach. You will be given a problem and a previous solution that has '
    'been used to solve the same problem. If the given solution leverages '
    'similar logic to the previous solution, conclude your response with the '
    'sentence "Decision: similar to the previous solution." Otherwise, '
    'conclude your response with the sentence "Decision: a novel approach." '
    'Include your reasoning for performing this task in your response. Below, '
    'the problem is provided wrapped in the <|PROBLEM|> tag, the previous '
    'solution is provided wrapped in the <|PREVIOUS SOLUTION|> tag, and the '
    'solution to be classified is provided within the <|SOLUTION|> tag.'
)

FALSE_RE = re.compile(r"([*][*])?decision:([*][*])? a novel approach([*][*])?", re.I)
TRUE_RE = re.compile(r"similar to the previous solution", re.I)


# --- API call ---------------------------------------------------------------

@dataclass
class JudgeStats:
    n_calls: int = 0
    n_cache_writes: int = 0
    n_cache_reads: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    n_no_regex_match: int = 0


def _judge_pair(
    client: Anthropic,
    *,
    model: str,
    question: str,
    previous_solution: str,
    candidate: str,
    stats: JudgeStats,
    max_retries: int = 5,
) -> tuple[bool, str]:
    """Run one pairwise judgement. Returns (is_similar, raw_response_text)."""

    prefix = (
        f"{INSTRUCTION}\n\n"
        f"<|PROBLEM|>\n{question}\n<|/PROBLEM|>\n\n"
        f"<|PREVIOUS SOLUTION|>\n{previous_solution}\n<|/PREVIOUS SOLUTION|>"
    )
    suffix = f"\n\n<|SOLUTION|>\n{candidate}\n<|/SOLUTION|>"

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prefix,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": suffix},
                        ],
                    }
                ],
            )
            break
        except RateLimitError as e:
            wait = min(60, 2 ** attempt)
            print(f"[rate-limit] sleeping {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30, 2 ** attempt)
            print(f"[api-error] sleeping {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"judge_pair: exhausted {max_retries} retries")

    text = msg.content[0].text if msg.content else ""

    # Update stats from msg.usage
    u = msg.usage
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    stats.n_calls += 1
    stats.input_tokens += u.input_tokens
    stats.output_tokens += u.output_tokens
    stats.cache_write_tokens += cw
    stats.cache_read_tokens += cr
    if cw > 0:
        stats.n_cache_writes += 1
    if cr > 0:
        stats.n_cache_reads += 1

    # algosim's decision parsing — verbatim
    if FALSE_RE.search(text):
        return False, text
    if TRUE_RE.search(text):
        return True, text
    stats.n_no_regex_match += 1
    return False, text  # algosim default


# --- Clustering loop --------------------------------------------------------

def _cluster_problem(
    client: Anthropic,
    *,
    model: str,
    problem_id: str,
    question: str,
    solutions: list[str],
    rng: random.Random,
    stats: JudgeStats,
) -> dict:
    """Reimplements algosim.clustering_solutions.create_solution_group."""

    if len(solutions) == 0:
        return {
            "problem_id": problem_id,
            "question": question,
            "solutions": solutions,
            "group_index": [],
            "records": [],
        }

    solution_indices = list(range(len(solutions)))
    solution_groups: list[list[int]] = []
    records: list[dict] = []
    while len(solution_indices) > 0:
        rep_idx = rng.sample(solution_indices, k=1)[0]
        solution_groups.append([rep_idx])
        solution_indices.remove(rep_idx)
        if len(solution_indices) == 0:
            break

        past_solution = solutions[rep_idx]
        # Sequential within a problem (so the prefix stays warm in cache)
        outputs: list[bool] = []
        for cand_idx in solution_indices:
            is_similar, response_text = _judge_pair(
                client,
                model=model,
                question=question,
                previous_solution=past_solution,
                candidate=solutions[cand_idx],
                stats=stats,
            )
            outputs.append(is_similar)
            records.append({
                "solution_index": int(cand_idx),
                "past_solution_index": int(rep_idx),
                "output": bool(is_similar),
                "response": response_text,
            })

        solution_groups[-1].extend(
            [i for i, o in zip(solution_indices, outputs) if o]
        )
        solution_indices = [
            i for i, o in zip(solution_indices, outputs) if not o
        ]

    group_index = [0] * len(solutions)
    for cluster_id, members in enumerate(solution_groups):
        for j in members:
            group_index[j] = cluster_id

    return {
        "problem_id": problem_id,
        "question": question,
        "solutions": solutions,
        "group_index": group_index,
        "records": records,
    }


# --- Per-config driver ------------------------------------------------------

def _seeded_rng(seed: int, problem_id: str) -> random.Random:
    """Deterministic per-problem RNG so smoke / full runs reproduce."""
    h = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[dict], set[str]]:
    """Return (already-completed rows, set of completed problem_ids)."""
    if not checkpoint_path.exists():
        return [], set()
    rows: list[dict] = []
    ids: set[str] = set()
    with checkpoint_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            ids.add(row["problem_id"])
    return rows, ids


def _process_config(
    client: Anthropic,
    *,
    config: str,
    requests_dir: Path,
    responses_dir: Path,
    model: str,
    workers: int,
    max_tasks: int | None,
    seed: int,
) -> JudgeStats:
    requests_path = requests_dir / f"{config}.parquet"
    output_path = responses_dir / f"{config}.parquet"
    checkpoint_path = responses_dir / f"{config}.jsonl"

    if output_path.exists():
        print(f"[{config}] already exists at {output_path}; skipping")
        return JudgeStats()

    if not requests_path.exists():
        raise SystemExit(f"[{config}] requests parquet not found: {requests_path}")

    responses_dir.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_parquet(requests_path)
    if max_tasks is not None:
        df_in = df_in.head(max_tasks).copy()
    print(f"[{config}] loaded {len(df_in)} problems from {requests_path}")

    # Resume from checkpoint if present
    prior_rows, done_ids = _load_checkpoint(checkpoint_path)
    if prior_rows:
        print(f"[{config}] resuming with {len(prior_rows)} problems already done")

    todo = [row for _, row in df_in.iterrows() if row["problem_id"] not in done_ids]
    if not todo:
        print(f"[{config}] nothing to do (all problems in checkpoint)")
    else:
        print(f"[{config}] {len(todo)} problems to process with {workers} workers")

    stats = JudgeStats()
    # Append-mode handle for checkpointing each completed problem.
    with checkpoint_path.open("a") as ckpt_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _cluster_problem,
                    client,
                    model=model,
                    problem_id=row["problem_id"],
                    question=row["question"],
                    solutions=list(row["solutions"]),
                    rng=_seeded_rng(seed, row["problem_id"]),
                    stats=stats,
                ): row["problem_id"]
                for row in todo
            }
            n_done = 0
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[{config}] problem {pid} FAILED: {e}", file=sys.stderr)
                    continue
                # Persist as JSON (parquet doesn't accept dicts via json.dumps,
                # but JSONL is fine for checkpointing).
                ckpt_f.write(json.dumps(result) + "\n")
                ckpt_f.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(todo):
                    print(
                        f"[{config}] {n_done}/{len(todo)} | "
                        f"calls={stats.n_calls} "
                        f"cache_reads={stats.n_cache_reads} "
                        f"cache_writes={stats.n_cache_writes} "
                        f"no_regex={stats.n_no_regex_match}"
                    )

    # Consolidate checkpoint -> parquet
    final_rows, _ = _load_checkpoint(checkpoint_path)
    df_out = pd.DataFrame.from_records(final_rows)
    df_out.to_parquet(output_path, index=False)
    print(f"[{config}] wrote {len(df_out)} rows to {output_path}")
    return stats


# --- Cost estimator ---------------------------------------------------------

# Sonnet 4.6 pricing (Apr 2026)
_INPUT_PRICE_PER_M = 3.00
_OUTPUT_PRICE_PER_M = 15.00
_CACHE_WRITE_MULT = 1.25  # writes cost 1.25x base input
_CACHE_READ_MULT = 0.1    # reads cost 0.1x base input


def _estimate_cost(stats: JudgeStats) -> float:
    # cache_write_tokens and cache_read_tokens are counted separately
    # in the Anthropic response; input_tokens excludes them.
    base_input = stats.input_tokens
    write_cost = stats.cache_write_tokens * _CACHE_WRITE_MULT * _INPUT_PRICE_PER_M / 1_000_000
    read_cost = stats.cache_read_tokens * _CACHE_READ_MULT * _INPUT_PRICE_PER_M / 1_000_000
    input_cost = base_input * _INPUT_PRICE_PER_M / 1_000_000
    output_cost = stats.output_tokens * _OUTPUT_PRICE_PER_M / 1_000_000
    return write_cost + read_cost + input_cost + output_cost


# --- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True,
                        help="Comma-separated config keys (e.g., H8P,T15N,P15)")
    parser.add_argument("--requests-dir", type=Path,
                        default=Path("algosim_data/requests"))
    parser.add_argument("--responses-dir", type=Path,
                        default=Path("algosim_data/responses_claude"))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel problems per config")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit problems per config (for smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY env var is required")

    client = Anthropic()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    total_stats = JudgeStats()
    for cfg in configs:
        t0 = time.time()
        cfg_stats = _process_config(
            client,
            config=cfg,
            requests_dir=args.requests_dir,
            responses_dir=args.responses_dir,
            model=args.model,
            workers=args.workers,
            max_tasks=args.max_tasks,
            seed=args.seed,
        )
        dt = time.time() - t0
        cost = _estimate_cost(cfg_stats)
        print(f"[{cfg}] done in {dt:.1f}s | "
              f"calls={cfg_stats.n_calls} | "
              f"input={cfg_stats.input_tokens} cache_w={cfg_stats.cache_write_tokens} "
              f"cache_r={cfg_stats.cache_read_tokens} output={cfg_stats.output_tokens} | "
              f"~${cost:.2f}")

        # accumulate
        total_stats.n_calls += cfg_stats.n_calls
        total_stats.n_cache_writes += cfg_stats.n_cache_writes
        total_stats.n_cache_reads += cfg_stats.n_cache_reads
        total_stats.cache_write_tokens += cfg_stats.cache_write_tokens
        total_stats.cache_read_tokens += cfg_stats.cache_read_tokens
        total_stats.input_tokens += cfg_stats.input_tokens
        total_stats.output_tokens += cfg_stats.output_tokens
        total_stats.n_no_regex_match += cfg_stats.n_no_regex_match

    total_cost = _estimate_cost(total_stats)
    print(f"\n[TOTAL] calls={total_stats.n_calls} "
          f"cache_reads={total_stats.n_cache_reads} "
          f"cache_writes={total_stats.n_cache_writes} "
          f"no_regex={total_stats.n_no_regex_match} | "
          f"~${total_cost:.2f}")


if __name__ == "__main__":
    main()
