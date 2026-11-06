"""Cross-check bench/eval/humaneval_official.py against the official openai/human-eval package (plan §7).

For each sample, our grader builds program = task_prompt + middle + tail with tail = "\\n" + test + "\\n" +
check(entry_point); the official harness builds prompt + completion + "\\n" + test + "\\n" + check(...). So the
official `completion` is exactly our `middle`, and both graders must agree sample by sample.

Run locally, without adding the package to the project:
    uv run --with human-eval==1.0.3 python scripts/humaneval_regen_crosscheck_official.py \
        --files results/humaneval_regen_2026_11_smoke/*/*.jsonl --max-programs 400

Scope: the official completion is derived from OUR build_program, so this validates execution (harness,
composition order, exit handling), not our chat-reply normalization; audit that separately on real replies.

Exits 1 if the official harness is inert (its exec line commented out — the package ships it disabled in some
versions) or if any sample disagrees.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.eval import humaneval_official as ours  # noqa: E402


def official_completion(record: dict, sample: str) -> str:
    program, _ = ours.build_program(record, sample)
    prompt = record["task_prompt"]
    tail = "\n" + record["test"] + "\n" + f"check({record['entry_point']})"
    assert program.startswith(prompt) and program.endswith(tail)
    return program[len(prompt): len(program) - len(tail)]


def official_problem(record: dict) -> dict:
    return {"task_id": record["task_id"], "prompt": record["task_prompt"], "test": record["test"],
            "entry_point": record["entry_point"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", type=Path, required=True)
    p.add_argument("--max-programs", type=int, default=400)
    p.add_argument("--timeout", type=float, default=ours.DEFAULT_TIMEOUT)
    args = p.parse_args(argv)
    from human_eval.execution import check_correctness

    # The official harness must actually execute code: a wrong completion has to fail.
    probe = {"task_id": "probe", "prompt": "def f():\n", "test": "def check(c):\n    assert c() == 1\n", "entry_point": "f"}
    if check_correctness(probe, "    return 2\n", args.timeout)["passed"]:
        print("official harness is inert (exec disabled in this human_eval build) — cross-check invalid")
        return 1

    n = agree = 0
    disagreements = []
    for f in args.files:
        for line in open(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            for sample in rec["samples"]:
                if n >= args.max_programs:
                    break
                ours_ok = ours.run_program(ours.build_program(rec, sample)[0], args.timeout)[0]
                off_ok = check_correctness(official_problem(rec), official_completion(rec, sample), args.timeout)["passed"]
                n += 1
                agree += ours_ok == off_ok
                if ours_ok != off_ok:
                    disagreements.append((str(f), rec["task_id"], ours_ok, off_ok))
    print(f"programs compared: {n}; agree: {agree}; disagree: {n - agree}")
    for d in disagreements[:20]:
        print("  disagree:", d)
    return 0 if n and agree == n else 1


if __name__ == "__main__":
    sys.exit(main())
