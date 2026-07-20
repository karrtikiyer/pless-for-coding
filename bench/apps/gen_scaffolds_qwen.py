"""Generate algorithm scaffolds with Qwen3-8B (thinking ON) — self-scaffold arm.

Companion to bench/apps/gen_scaffolds.py (Claude Opus). Same 26 never-solved
APPS ATCODER-interview tasks, the SAME system prompt (a structured algorithm,
no code), but the scaffold author is Qwen3-8B itself with thinking ENABLED. The
point is to compare Qwen's own scaffolds against Opus's — i.e. how much of the
transfer effect (if any) is Opus-specific reasoning vs. any structured plan.

Design notes:
  * System prompt is imported verbatim from gen_scaffolds so the only variable
    is the model. If Qwen ignores the "no code" rule or returns empties, tune
    the prompt via --extra-system (kept a separate knob so the fair-comparison
    default stays untouched).
  * Qwen3-8B thinking-mode sampling (verified against the HF model card,
    2026-07): temperature 0.6, top_p 0.95, top_k 20, no greedy. The card
    recommends up to 32768 output tokens; on MPS that is impractical, so
    --max-new-tokens defaults to 16384 and truncated-empty scaffolds (thinking
    consumed the whole budget) are surfaced by the validation gate.
  * The raw generation is ``<think>...</think>`` + the answer; the scaffold is
    the post-``</think>`` text (via generator._strip_think_content).

Usage (run AFTER the MPS GPU is free — do not run concurrently with a Qwen
generation job):
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m bench.apps.gen_scaffolds_qwen \\
        --model Qwen/Qwen3-8B --source ATCODER --difficulty interview \\
        --out results/scaffold_transfer/scaffolds_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bench.apps.dataset import load_apps
from bench.apps.gen_scaffolds import (
    SYSTEM_PROMPT,
    TASK_IDS,
    _build_user_content,
    _load_checkpoint,
    _looks_like_code,
)
from bench.generator import (
    _strip_think_content,
    generate_samples_standard,
    load_model_and_tokenizer,
)


def _build_scaffold_prompt(tokenizer, problem, extra_system: str = "") -> str:
    """Chat-template prompt asking for a structured algorithm, thinking ON.

    Mirrors bench.apps.prompts.format_prompt_apps_instruct's tokenize handling
    but uses the scaffold-generation system prompt (not the code-writing one).
    """
    system = SYSTEM_PROMPT + (("\n\n" + extra_system) if extra_system else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(problem)},
    ]
    if getattr(tokenizer, "_qwen_direct_tokenize", False):
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, enable_thinking=True,
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def _generate_one(model, tokenizer, problem, *, temperature, top_p, top_k,
                  max_new_tokens, extra_system) -> tuple[str, str]:
    """Return (scaffold, raw_with_thinking) for one problem."""
    prompt = _build_scaffold_prompt(tokenizer, problem, extra_system)
    raw = generate_samples_standard(
        model=model, tokenizer=tokenizer, prompt_text=prompt,
        n_samples=1, max_new_tokens=max_new_tokens,
        temperature=temperature, stop_strings=None,
        top_p=top_p, top_k=top_k, hf_batch_size=1,
    )[0]
    scaffold = _strip_think_content(raw)
    return scaffold, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--source", default="ATCODER")
    ap.add_argument("--difficulty", default="interview")
    ap.add_argument("--out", type=Path,
                    default=Path("results/scaffold_transfer/scaffolds_qwen.jsonl"))
    # Qwen3-8B thinking-mode recommended settings (HF model card, verified).
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=16384,
                    help="Shared thinking+answer budget. Card suggests up to 32768; "
                         "capped lower for MPS. Truncated-empty scaffolds (thinking ate "
                         "the budget) are reported by the validation gate.")
    ap.add_argument("--min-scaffold-chars", type=int, default=200)
    ap.add_argument("--extra-system", default="",
                    help="Optional Qwen-specific system-prompt addendum (kept separate "
                         "so the fair-comparison default matches the Opus prompt).")
    ap.add_argument("--task-ids", type=int, nargs="+", default=None)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    wanted = set(args.task_ids if args.task_ids is not None else TASK_IDS)
    problems = {
        p.problem_id: p
        for p in load_apps(source=args.source, difficulty=args.difficulty)
        if p.problem_id in wanted
    }
    missing = wanted - set(problems)
    if missing:
        print(f"[warn] task_ids not found: {sorted(missing)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(args.out)
    todo = [p for tid, p in sorted(problems.items()) if tid not in done]
    if done:
        print(f"Resuming: {len(done)} Qwen scaffolds already in {args.out}")
    print(f"Generating {len(todo)} Qwen scaffolds (thinking ON, model={args.model}, "
          f"t{args.temperature}/p{args.top_p}/k{args.top_k}, cap {args.max_new_tokens})")

    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)

    n_empty = 0
    with args.out.open("a") as fh:
        for problem in tqdm(todo, desc="qwen-scaffolds"):
            try:
                scaffold, raw = _generate_one(
                    model, tokenizer, problem,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens, extra_system=args.extra_system,
                )
            except Exception as e:
                tqdm.write(f"task {problem.problem_id} FAILED: {e}")
                continue
            flags = []
            if _looks_like_code(scaffold):
                flags.append("code")
            if len(scaffold.strip()) < args.min_scaffold_chars:
                flags.append("empty")
                n_empty += 1
            fh.write(json.dumps({
                "task_id": problem.problem_id,
                "scaffold": scaffold,
                "raw_with_thinking": raw,
                "model": args.model,
                "flags": flags,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            fh.flush()
            tqdm.write(f"[done] task {problem.problem_id}: {len(scaffold)} chars "
                       f"(raw {len(raw)}){' FLAGS=' + ','.join(flags) if flags else ''}")

    # Validation summary (report, don't hard-fail — this is an exploratory arm).
    rows = [json.loads(l) for l in args.out.open() if l.strip()]
    leaked = sorted(r["task_id"] for r in rows if _looks_like_code(r["scaffold"]))
    empty = sorted(r["task_id"] for r in rows
                   if len(r["scaffold"].strip()) < args.min_scaffold_chars)
    print(f"\n[summary] {len(rows)} scaffolds | code-like: {leaked} | "
          f"empty/short: {empty}")
    if empty:
        print(f"[note] {len(empty)} scaffolds empty (thinking likely hit the "
              f"{args.max_new_tokens}-token cap). Re-run those task_ids with a "
              f"higher --max-new-tokens.", file=sys.stderr)


if __name__ == "__main__":
    main()
