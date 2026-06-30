"""Taxonomy of ALL truncated p-less traces — what fraction are the verbatim STATEMENT
loops we studied (Fig 3b/4), vs other kinds? Answers: "if we detect statement loops,
do we cover most truncations?"

Categories (per truncated sample, in order):
  no_onset            : n-gram detector never fired (no verbatim ≥k-repeat at all)
  paraphrastic        : onset fired, but NO clean period ≤ PMAX (repetition is
                        non-verbatim/drifting — the n-gram caught a local repeat but the
                        region isn't globally periodic)
  single_token (P=1)  : a single token repeated (degenerate)
  short_fragment 2-9  : a <10-token unit repeated (e.g. " = product_{")
  statement_reflective: 10≤P≤PMAX, prose + impasse/reflection marker (Wait/perhaps/stuck/
                        Let me think…) — THE kind we studied
  statement_other     : 10≤P≤PMAX, but NOT reflective prose (repeating code/formula/eqn lines)
  long_period >PMAX   : period exceeds the statement cap

Usage:
  HF_HUB_OFFLINE=1 uv run python scripts/loop_collapse_categorize.py \
      --model Qwen/Qwen3-8B --jsonl <pless jsonl> --out <out.json> [--max-records N]
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from scripts.signal_diagnostic import simulate_onset, LOOP_PARAMS  # noqa: E402
from scripts.loop_collapse_screen import extract_think  # noqa: E402

PMAX = 800
REGION = 1600
MATCH = 0.85
MARK = re.compile(r"\b(wait|perhaps|maybe|alternatively|stuck|hmm|not sure|let me think|"
                  r"give up|unable|i think i|hold on|i'm not sure)\b", re.I)


def fundamental_period(arr, dmin=1, dmax=PMAX, thresh=MATCH):
    n = len(arr); dmax = min(dmax, n - 1)
    for d in range(dmin, dmax + 1):
        if float(np.mean(arr[: n - d] == arr[d:])) >= thresh:
            return d
    return None


def prose(u): return sum(c.isalpha() or c.isspace() for c in u) / max(len(u), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()
    n, k, window = LOOP_PARAMS[args.model].values()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    cats = Counter(); examples = {}
    n_trunc = 0
    for li, line in enumerate(open(args.jsonl)):
        if args.max_records and li >= args.max_records:
            break
        d = json.loads(line)
        for si, raw in enumerate(d.get("samples_with_thinking", [])):
            think, complete = extract_think(raw)
            if complete or not think.strip():
                continue
            n_trunc += 1
            ids = tok.encode(think, add_special_tokens=False)[:38000]
            onset = simulate_onset(ids, n, k, window)
            if onset is None:
                cat = "no_onset"; unit = ""
            else:
                arr = np.asarray(ids)
                reg = arr[onset: min(len(ids), onset + REGION)]
                P = fundamental_period(reg) if len(reg) > 4 else None
                if P is None:
                    cat = "paraphrastic"; unit = tok.decode(ids[onset:onset + 120])
                elif P == 1:
                    cat = "single_token"; unit = tok.decode(reg[:1].tolist())
                elif P < 10:
                    cat = "short_fragment"; unit = tok.decode(reg[:P].tolist())
                elif P <= PMAX:
                    unit = tok.decode(reg[:P].tolist())
                    cat = ("statement_reflective" if prose(unit) > 0.78 and MARK.search(unit)
                           else "statement_other")
                else:
                    cat = "long_period"; unit = tok.decode(reg[:200].tolist())
            cats[cat] += 1
            examples.setdefault(cat, [])
            if len(examples[cat]) < 5:
                examples[cat].append({"task_id": d["task_id"], "sample_idx": si,
                                      "unit": unit[:160]})
        if li % 25 == 0:
            print(f"  scanned {li+1} tasks, {n_trunc} truncated", flush=True)

    order = ["statement_reflective", "statement_other", "short_fragment", "single_token",
             "paraphrastic", "long_period", "no_onset"]
    print(f"\n=== {args.model}: truncation taxonomy (n_truncated={n_trunc}) ===")
    for c in order:
        v = cats.get(c, 0)
        print(f"  {c:>22}: {v:>5}  ({100*v/max(n_trunc,1):4.1f}%)")
    sr = cats.get("statement_reflective", 0); so = cats.get("statement_other", 0)
    print(f"\n  STATEMENT loops total (reflective+other, 10≤P≤{PMAX}): "
          f"{sr+so} ({100*(sr+so)/max(n_trunc,1):.1f}% of truncations)")
    print(f"  of which REFLECTIVE (what Fig3b/4 studied): {sr} ({100*sr/max(n_trunc,1):.1f}%)")
    print("\n  examples per category:")
    for c in order:
        for e in examples.get(c, [])[:3]:
            print(f"    [{c}] {e['task_id']}[{e['sample_idx']}]: {e['unit']!r}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": args.model, "n_truncated": n_trunc, "counts": dict(cats),
         "examples": examples}, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
