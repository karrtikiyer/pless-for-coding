"""Control analysis: are verbatim statement-loop patterns present ONLY in
truncated/failed traces, or also in COMPLETED (closed </think>) and even CORRECT
ones? If completed-correct traces also contain >=6-cycle verbatim loops, then
"looping" is not by itself diagnostic of failure — and a completed-recovered loop
becomes a stronger Fig-4 control than the paper's non-repeating "Normal" baseline.

Cross-tabulates every sample by:
  completion  : closed </think>  vs truncated
  statement-loop : has a >=6-cycle verbatim period>=10 loop (same detector as the
                   Fig-3b/4 screener) vs not
  correctness : per-sample pass_results from the cot_efficiency metrics json

Reuses the EXACT loop detector + anchor finder from loop_collapse_screen.py, so
"has statement loop" here means the same thing as trace selection there.

Usage:
  HF_HUB_OFFLINE=1 uv run python scripts/loop_collapse_control.py \
      --model Qwen/Qwen3-8B \
      --jsonl  results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl \
      --metrics results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json \
      --out results/loop_collapse_replication/Qwen--Qwen3-8B/control_loop_vs_completion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.signal_diagnostic import simulate_onset, LOOP_PARAMS  # noqa: E402
from scripts.loop_collapse_screen import (  # noqa: E402
    extract_think, find_loop_anchor, MAX_CTX_TOKENS,
)

MIN_CYCLES = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    lp = LOOP_PARAMS[args.model]
    n, k, window = lp["n"], lp["k"], lp["window"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    # per-sample correctness: pass_results[si] aligns with samples_with_thinking[si]
    m = json.load(open(args.metrics))
    passmap = {t["task_id"]: t.get("pass_results", []) for t in m["per_task"]}

    cells = Counter()          # (completion, statement-loop, correct) -> count
    onset_cells = Counter()    # (completion, any-n-gram-onset, correct) -> count
    completed_loop_correct = []  # the interesting cases (for a possible Fig-4 arm)
    n_scanned = 0

    with open(args.jsonl) as f:
        for li, line in enumerate(f):
            if args.max_records is not None and li >= args.max_records:
                break
            d = json.loads(line)
            tid = d["task_id"]
            pr = passmap.get(tid, [])
            for si, raw in enumerate(d.get("samples_with_thinking", [])):
                think_str, complete = extract_think(raw)
                if not think_str.strip():
                    continue
                n_scanned += 1
                ids = tok.encode(think_str, add_special_tokens=False)[:MAX_CTX_TOKENS]
                onset = simulate_onset(ids, n, k, window)
                has_loop = False
                cyc = 0
                if onset is not None:
                    anc = find_loop_anchor(ids, onset, n, k, window)
                    if anc is not None and anc["n_cycles"] >= MIN_CYCLES:
                        has_loop = True
                        cyc = anc["n_cycles"]
                correct = bool(pr[si]) if si < len(pr) else False
                comp = "completed" if complete else "truncated"
                corr = "correct" if correct else "wrong"
                cells[(comp, "loop" if has_loop else "noloop", corr)] += 1
                onset_cells[(comp, "ngram" if onset is not None else "noNgram", corr)] += 1
                if comp == "completed" and has_loop and correct:
                    completed_loop_correct.append(
                        {"task_id": tid, "sample_idx": si, "n_cycles": cyc,
                         "period": anc["period_median"],
                         "unit": tok.decode(anc["gram"])[:200]})
            if li % 25 == 0:
                print(f"  scanned {li+1} tasks ({n_scanned} samples)", flush=True)

    # report
    print(f"\n=== {args.model}: loop-presence x completion x correctness "
          f"(statement loop = >= {MIN_CYCLES} verbatim cycles) ===")
    print(f"scanned {n_scanned} samples")
    hdr = f"{'completion':>10} {'loop':>7} {'correct':>8} {'count':>6}"
    print(hdr); print("-" * len(hdr))
    for comp in ("completed", "truncated"):
        for loop in ("loop", "noloop"):
            for corr in ("correct", "wrong"):
                print(f"{comp:>10} {loop:>7} {corr:>8} {cells[(comp,loop,corr)]:>6}")

    # the headline question: completed+statement-loop, split by correctness
    cl_correct = cells[("completed", "loop", "correct")]
    cl_wrong = cells[("completed", "loop", "wrong")]
    print(f"\nHEADLINE: completed traces that contain a >= {MIN_CYCLES}-cycle verbatim "
          f"statement loop: {cl_correct + cl_wrong}  "
          f"(of which CORRECT: {cl_correct}, wrong: {cl_wrong})")
    print(f"  → statement-loop pattern {'IS' if cl_correct>0 else 'is NOT'} present in "
          f"completed+correct traces.")

    # broader picture: ANY n-gram onset (transient/recovered repeats), incl. <6-cycle
    print(f"\n--- broader: ANY n-gram onset (n={n},k={k},w={window}; incl. short/transient) ---")
    hdr2 = f"{'completion':>10} {'ngram':>8} {'correct':>8} {'count':>6}"
    print(hdr2); print("-" * len(hdr2))
    for comp in ("completed", "truncated"):
        for ng in ("ngram", "noNgram"):
            for corr in ("correct", "wrong"):
                print(f"{comp:>10} {ng:>8} {corr:>8} {onset_cells[(comp,ng,corr)]:>6}")
    print(f"  completed+ngram+CORRECT: {onset_cells[('completed','ngram','correct')]}  "
          f"(transient/recovered repeats that still solved the task)")
    if completed_loop_correct:
        print("\n  examples (completed + statement-loop + CORRECT):")
        for e in completed_loop_correct[:8]:
            print(f"    task {e['task_id']}[{e['sample_idx']}] cyc={e['n_cycles']} "
                  f"P={e['period']:.0f}: {e['unit'][:120]!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "n_scanned": n_scanned,
        "cells": {f"{a}|{b}|{c}": v for (a, b, c), v in cells.items()},
        "onset_cells": {f"{a}|{b}|{c}": v for (a, b, c), v in onset_cells.items()},
        "completed_loop_correct_examples": completed_loop_correct,
    }, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
