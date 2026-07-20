"""Generate external algorithm scaffolds with Claude Opus for the transfer study.

For each of the 26 never-solved APPS ATCODER-interview tasks, ask Claude Opus for
a STRUCTURED ALGORITHM (a flowchart in words) — numbered steps, named data
structures, complexity — with NO code. Qwen3-8B then implements from the scaffold
with its own thinking OFF (see bench/apps/prompts.format_prompt_apps_scaffold and
the ``--scaffold-file`` runner flag).

The scaffold's altitude is the whole experiment: if Opus leaks code, a Qwen pass
proves transcription, not reasoning transfer. Guardrails: a forbidding system
prompt, a post-generation code-token heuristic with one re-request, and a
pre-exit validation gate that refuses to declare success if any scaffold trips
the heuristic.

Reuses the Anthropic client pattern from bench/eval/algosim_claude_judge.py
(client init, ANTHROPIC_API_KEY guard, retry, ThreadPoolExecutor, JSONL
checkpoint/resume). Opus 4.8 specifics (verified via the claude-api skill):
model id ``claude-opus-4-8``; ``temperature``/``top_p``/``top_k`` and
``budget_tokens`` are rejected (400) — do NOT send them; thinking is
``{"type": "adaptive"}`` and depth is set via ``output_config.effort``; with
thinking on, the response's first content block may be a thinking block, so we
scan for the ``text`` block rather than reading ``content[0]``.

Usage:
    uv run python -m bench.apps.gen_scaffolds \\
        --model claude-opus-4-8 --source ATCODER --difficulty interview \\
        --out results/scaffold_transfer/scaffolds.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# NOTE: the ``anthropic`` SDK is imported lazily (inside the functions that call
# it), not at module top — so importing this module for its shared constants
# (SYSTEM_PROMPT, TASK_IDS, _looks_like_code, ...) from bench.apps.gen_scaffolds_qwen
# does NOT require anthropic to be installed (e.g. in a vLLM-only env).
from bench.apps.dataset import load_apps

# The 26 APPS ATCODER-interview task_ids never solved by any of 23 full-252
# Qwen3-8B configs (all thinking-ON). See docs/theory/todos.md.
TASK_IDS = [
    117, 257, 326, 370, 454, 455, 512, 661, 929, 962, 1122, 1175, 1223,
    1368, 1469, 1471, 1581, 1717, 2374, 2390, 2500, 2503, 2659, 2715, 2749, 2886,
]

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are an expert competitive programmer and algorithm designer. Given a "
    "programming problem, produce a STRUCTURED ALGORITHM — a flowchart in words "
    "— that a competent programmer could implement without further insight.\n\n"
    "Output ONLY these sections:\n"
    "1. Restatement: one or two sentences on exactly what to compute.\n"
    "2. Data structures: the named structures to use, described in words "
    "(e.g. \"a prefix-sum array `pre` of length n+1\").\n"
    "3. Algorithm: numbered, imperative steps. Include how to READ the input "
    "from standard input and how to FORMAT the output to standard output.\n"
    "4. Edge cases: the tricky inputs to handle.\n"
    "5. Complexity: time and space.\n\n"
    "STRICTLY FORBIDDEN: Python or any programming-language syntax, code blocks, "
    "triple-backtick fences, function/variable definitions written as code, "
    "library calls, or line-by-line source. Describe every operation in prose, "
    "not code. If you are about to write code, describe it in words instead."
)

# Heuristic for detecting code leakage in a scaffold (see module docstring).
# Deliberately conservative — matches only near-unambiguous Python, NOT prose.
# Earlier clauses (``\bwhile\b.*:$`` and a bare ``^from ``) fired on normal
# algorithm prose ("Repeat while the queue is not empty:", "From the leftmost
# element, ...") and are gone; see tests/test_gen_scaffolds_code_heuristic.py.
_CODE_FENCE = re.compile(r"```")
_CODE_TOKENS = re.compile(
    r"(?m)("
    r"^\s*(def |class |import )"          # def/class/import statement at line start
    r"|\bfrom\s+\w+\s+import\b"           # from X import Y (not prose 'from the ...')
    r"|\bprint\s*\("                      # print(
    r"|\bfor\s+\w+\s+in\s+range\s*\("     # for i in range(
    r"|=\s*input\s*\("                    # = input(
    r")"
)


def _looks_like_code(text: str) -> bool:
    return bool(_CODE_FENCE.search(text) or _CODE_TOKENS.search(text))


# --- Opus pricing (per claude-api skill, 2026): $5 / $25 per MTok -----------
_INPUT_PRICE_PER_M = 5.00
_OUTPUT_PRICE_PER_M = 25.00
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.1


@dataclass
class GenStats:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    n_rerequests: int = 0
    n_code_leaks: int = 0


def _estimate_cost(s: GenStats) -> float:
    return (
        s.input_tokens * _INPUT_PRICE_PER_M / 1_000_000
        + s.cache_write_tokens * _CACHE_WRITE_MULT * _INPUT_PRICE_PER_M / 1_000_000
        + s.cache_read_tokens * _CACHE_READ_MULT * _INPUT_PRICE_PER_M / 1_000_000
        + s.output_tokens * _OUTPUT_PRICE_PER_M / 1_000_000
    )


def _text_of(msg) -> str:
    """Extract the answer text; with adaptive thinking, content[0] may be a
    thinking block, so scan for the first ``text`` block."""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def _accumulate(stats: GenStats, msg) -> None:
    u = msg.usage
    stats.n_calls += 1
    stats.input_tokens += u.input_tokens
    stats.output_tokens += u.output_tokens
    stats.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
    stats.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0


def _call_opus(client, *, model, user_content, stats, max_tokens, max_retries=5) -> str:
    """One Opus call. No sampling params (rejected on 4.8); adaptive thinking.

    Imports the anthropic exception types lazily (see the module-top note).

    Streams and reads the final message: with adaptive thinking, the budget is
    shared between reasoning and the answer, so ``max_tokens`` must be large
    enough that hard problems don't spend the whole budget thinking and return
    an empty text block (observed at 8192). Streaming also avoids the SDK's
    non-streaming timeout guard at large ``max_tokens``.
    """
    from anthropic._exceptions import APIError, RateLimitError
    for attempt in range(max_retries):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                msg = stream.get_final_message()
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
        raise RuntimeError(f"gen_scaffold: exhausted {max_retries} retries")
    _accumulate(stats, msg)
    return _text_of(msg)


def _build_user_content(problem) -> str:
    parts = ["Problem:", problem.question.strip()]
    if problem.starter_code.strip():
        parts += [
            "",
            "Interface hint (describe how to use it in words; do NOT echo it as code):",
            problem.starter_code.strip(),
        ]
    return "\n".join(parts)


def _generate_scaffold(client, *, model, problem, stats, max_tokens) -> str:
    """Generate one scaffold; re-request once if the first output leaks code."""
    user_content = _build_user_content(problem)
    text = _call_opus(client, model=model, user_content=user_content, stats=stats,
                      max_tokens=max_tokens)
    if _looks_like_code(text):
        print(f"[code-leak] task {problem.problem_id}: re-requesting without code",
              file=sys.stderr)
        stats.n_rerequests += 1
        text = _call_opus(
            client, model=model,
            user_content=(user_content + "\n\nYour previous answer contained code. "
                          "Rewrite the algorithm with NO code — words only, no "
                          "syntax, no code fences."),
            stats=stats, max_tokens=max_tokens,
        )
        if _looks_like_code(text):
            stats.n_code_leaks += 1
    return text


def _load_checkpoint(path: Path) -> set[int]:
    """Return task_ids already present in the output JSONL (resume support)."""
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(int(json.loads(line)["task_id"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", default="ATCODER")
    ap.add_argument("--difficulty", default="interview")
    ap.add_argument("--out", type=Path,
                    default=Path("results/scaffold_transfer/scaffolds.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32000,
                    help="Shared thinking+answer budget. Must be large enough that "
                         "adaptive thinking on hard problems doesn't consume the whole "
                         "budget and return an empty scaffold (8192 was too small).")
    ap.add_argument("--min-scaffold-chars", type=int, default=200,
                    help="Validation floor: a scaffold with fewer non-whitespace "
                         "chars is treated as a generation failure.")
    ap.add_argument("--task-ids", type=int, nargs="+", default=None,
                    help="Override the default 26 never-solved task_ids.")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY env var is required")

    wanted = set(args.task_ids if args.task_ids is not None else TASK_IDS)
    problems = {
        p.problem_id: p
        for p in load_apps(source=args.source, difficulty=args.difficulty)
        if p.problem_id in wanted
    }
    missing = wanted - set(problems)
    if missing:
        print(f"[warn] {len(missing)} requested task_ids not found in "
              f"{args.source}/{args.difficulty}: {sorted(missing)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(args.out)
    todo = [p for tid, p in sorted(problems.items()) if tid not in done]
    if done:
        print(f"Resuming: {len(done)} scaffolds already in {args.out}")
    print(f"Generating {len(todo)} scaffolds with {args.workers} workers "
          f"(model={args.model})")

    from anthropic import Anthropic
    client = Anthropic()
    stats = GenStats()
    with args.out.open("a") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_generate_scaffold, client, model=args.model,
                            problem=p, stats=stats,
                            max_tokens=args.max_tokens): p.problem_id
                for p in todo
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    scaffold = fut.result()
                except Exception as e:
                    print(f"task {tid} FAILED: {e}", file=sys.stderr)
                    continue
                fh.write(json.dumps({
                    "task_id": tid,
                    "scaffold": scaffold,
                    "model": args.model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                fh.flush()
                print(f"[done] task {tid} ({len(scaffold)} chars)")

    print(f"\ncalls={stats.n_calls} re-requests={stats.n_rerequests} "
          f"input={stats.input_tokens} output={stats.output_tokens} "
          f"cache_r={stats.cache_read_tokens} cache_w={stats.cache_write_tokens} "
          f"~${_estimate_cost(stats):.2f}")

    # Validation gate: re-scan every scaffold on disk; refuse to declare success
    # silently on either failure mode — code leakage (a leaked solution would
    # reduce a Qwen pass to transcription) or an empty/short scaffold (adaptive
    # thinking ate the whole token budget → degenerate treatment prompt).
    leaked, empty = [], []
    with args.out.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _looks_like_code(row["scaffold"]):
                leaked.append(row["task_id"])
            if len(row["scaffold"].strip()) < args.min_scaffold_chars:
                empty.append(row["task_id"])
    if leaked or empty:
        if leaked:
            print(f"\n[VALIDATION FAILED] {len(leaked)} scaffolds contain code-like "
                  f"content: {sorted(leaked)}.", file=sys.stderr)
        if empty:
            print(f"\n[VALIDATION FAILED] {len(empty)} scaffolds are empty/short "
                  f"(<{args.min_scaffold_chars} chars): {sorted(empty)}. Likely the "
                  f"token budget was consumed by thinking — raise --max-tokens and "
                  f"regenerate these task_ids.", file=sys.stderr)
        raise SystemExit(1)
    print(f"[validation OK] {args.out}: no code-like content, no empty scaffolds")


if __name__ == "__main__":
    main()
