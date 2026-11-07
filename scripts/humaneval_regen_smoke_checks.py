"""Smoke-test checks for the 2026-11 HumanEval regeneration (plan §8 step 3). Runs locally, CPU only.

Reads the smoke outputs written by run_regen_humaneval_8models.sh (smoke + smoke-rerun) and reports:

  hard checks (exit 1 on failure)
    * every record has the configured number of samples
    * base models, unstripped prompt: >= 99% of completions start their body with exactly 4 spaces
    * base models, --prompt-variant stripped (prompt ends in the docstring's closing quotes), see STRIPPED_* below:
      >= 90% of the non-empty completions that start on a new line have a 4-space first body line;
      >= 50% of the non-empty completions start on a new line; not every completion is empty
    * prompts carry exactly one BOS (models whose tokenizer has a BOS)
    * the backend stopped: no raw text contains a stop sequence when the backend applied stop strings
    * every config (greedy, beam and seeded sampling) is identical between the two smoke copies (raw text)
    * truly sampled configs (T@0.7, top-p) do not return 10 identical samples for every task
  reported only
    * stripped prompt: counts of completions starting on a new line (4-space body / other indent), continuing on
      the closing-quotes line, and empty
    * completions that hit max_new_tokens (from the generation-side finish metadata)
    * whether the backend applied stop strings (False for models whose vLLM detokenizer is broken)
    * median seconds per task, per model x config (input to the time budget)

Usage:
    uv run python scripts/humaneval_regen_smoke_checks.py \
        --smoke-root results/humaneval_regen_2026_11_smoke --rerun-root results/humaneval_regen_2026_11_smoke_rerun
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

DETERMINISTIC = {"greedy", "beam4", "greedy_hf"}
MUST_VARY = {"temp_t0.7", "top_p0.9_t1.0", "top_p0.95_t0.2"}   # real sampling: 10 samples must not all coincide

# Stripped prompt (prompt.strip(), ends in the docstring's closing quotes): a completion either starts on a new
# line ("\n    body"), continues on the quotes line (' x = 1\n    ...': legitimate model behaviour, graded as
# written), or is empty (also legitimate; fails its tests). Only the new-line starts are comparable with the
# unstripped 4-space check, since their body follows a newline exactly as in the unstripped prompt.
#
# STRIPPED_NEWLINE_4SP_MIN = 0.90. Real stripped data (results/humaneval_regen_2026_11_diag, greedy and
# top_p0.95_t0.2 for OCI / Qwen 1.5B / 3B; 3614 new-line starts) has 100.0% 4-space bodies; the overall shares
# of 98.2% / 94.5% (Qwen greedy) fall short only through quotes-line continuations and empties. A real
# indentation bug (tokenizer / prompt-boundary whitespace error, e.g. a 3-space body) is systematic and drives
# the share toward 0%, so 90% catches it with a wide margin, while leaving room for sampled configs at T=1.0
# on 5-task smokes (50 samples: up to 5 odd first lines, e.g. a comment at another indent, do not fail).
# STRIPPED_NEWLINE_SHARE_MIN = 0.50. A whitespace-losing decode ("\n    x" read as "x", the vLLM detokenizer
# failure seen for OCI) would turn every completion into an apparent quotes-line continuation and bypass the
# 4-space check; real data has >= 95.1% new-line starts among non-empty completions (Qwen 3B greedy 155/163),
# so a majority of quotes-line starts is flagged.
STRIPPED_NEWLINE_4SP_MIN = 0.90
STRIPPED_NEWLINE_SHARE_MIN = 0.50


def at_cap(metas: list[dict] | None) -> int | None:
    """Samples that hit max_new_tokens, from the generation-side metadata (not re-tokenized locally)."""
    if not metas:
        return None
    return sum(1 for m in metas if m and m.get("finish_reason") == "length")


def first_body_indent(completion: str) -> int | None:
    for line in completion.split("\n"):
        if line.strip():
            return len(line) - len(line.lstrip(" "))
    return None


def classify_start(completion: str) -> str:
    """'empty' (whitespace only), 'new_line' (first line blank: body starts on a later line) or 'same_line'
    (text on the line the completion starts on, i.e. after the stripped prompt's closing quotes)."""
    if not completion.strip():
        return "empty"
    return "new_line" if not completion.split("\n", 1)[0].strip() else "same_line"


def stripped_start_counts(completions: list[str]) -> dict[str, int]:
    c = {"new_line_4sp": 0, "new_line_other": 0, "same_line": 0, "empty": 0}
    for x in completions:
        k = classify_start(x)
        if k == "new_line":
            k = "new_line_4sp" if first_body_indent(x) == 4 else "new_line_other"
        c[k] += 1
    return c


def stripped_indent_failures(counts: dict[str, int]) -> list[str]:
    """Hard-check messages for stripped-prompt completions (empty list: pass)."""
    nl = counts["new_line_4sp"] + counts["new_line_other"]
    nonempty = nl + counts["same_line"]
    if nonempty == 0:
        return ["all completions are empty"]
    out = []
    if nl / nonempty < STRIPPED_NEWLINE_SHARE_MIN:
        out.append(f"only {nl}/{nonempty} non-empty completions start on a new line "
                   f"(< {STRIPPED_NEWLINE_SHARE_MIN:.0%}; whitespace lost at the prompt boundary?)")
    if nl and counts["new_line_4sp"] / nl < STRIPPED_NEWLINE_4SP_MIN:
        out.append(f"only {counts['new_line_4sp'] / nl:.1%} of new-line-starting completions have a 4-space first "
                   f"body line (< {STRIPPED_NEWLINE_4SP_MIN:.0%})")
    return out


def stop_hits(raw_texts: list[str], stops: list[str]) -> int:
    return sum(any(s in t for s in stops) for t in raw_texts)


def identical_share(a: dict[str, list[str]], b: dict[str, list[str]]) -> tuple[int, int]:
    shared = sorted(set(a) & set(b))
    return sum(a[t] == b[t] for t in shared), len(shared)


def first_divergence(a: list[str], b: list[str]) -> tuple[int, int] | None:
    """(sample index, character position) of the first difference between two sample lists, else None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, next((k for k, (cx, cy) in enumerate(zip(x, y)) if cx != cy), min(len(x), len(y)))
    return None if len(a) == len(b) else (min(len(a), len(b)), 0)


def seconds_per_task(timestamps: list[str]) -> float | None:
    if len(timestamps) < 2:
        return None
    t = sorted(datetime.fromisoformat(x) for x in timestamps)
    return float(statistics.median((t[i + 1] - t[i]).total_seconds() for i in range(len(t) - 1)))


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-root", type=Path, required=True)
    p.add_argument("--rerun-root", type=Path, default=None)
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.humaneval.regen import count_leading_bos, load_tokenizer

    failures, rows, starts = [], [], []
    for f in sorted(args.smoke_root.glob("*/*.jsonl")):
        recs = _load(f)
        if not recs:
            continue
        model, config, style = recs[0]["model"], recs[0]["config"], recs[0]["prompt_style"]
        variant = recs[0].get("prompt_variant", "unstripped")   # files written before the field existed
        label = f.stem   # == config for unstripped files; <config>__prompt_stripped otherwise
        tok = load_tokenizer(model)
        n_cfg = recs[0]["n"]
        bad_n = [r["task_id"] for r in recs if len(r["samples"]) != n_cfg]
        if bad_n:
            failures.append(f"{model} {config}: wrong sample count for {bad_n}")
        if tok.bos_token_id is not None:
            bad_bos = [r["task_id"] for r in recs if count_leading_bos(r["prompt_token_ids_head"], tok) != 1]
            if bad_bos:
                failures.append(f"{model} {config}: prompts without exactly one BOS: {bad_bos}")
        comps = [s for r in recs for s in r["samples"]]
        raws = [s for r in recs for s in r["raw_samples"]]
        indent4 = None
        if style == "base" and variant == "stripped":
            sc = stripped_start_counts(comps)
            starts.append((model, label, sc))
            nl = sc["new_line_4sp"] + sc["new_line_other"]
            indent4 = sc["new_line_4sp"] / nl if nl else None
            failures += [f"{model} {label}: {msg}" for msg in stripped_indent_failures(sc)]
        elif style == "base":
            ind = [first_body_indent(c) for c in comps]
            indent4 = sum(i == 4 for i in ind) / len(ind)
            if indent4 < 0.99:
                failures.append(f"{model} {config}: only {indent4:.1%} of completions start with 4 spaces")
        hits = stop_hits(raws, recs[0]["stop"] or []) if style == "base" else 0
        if hits and recs[0].get("backend_stop_strings"):
            failures.append(f"{model} {config}: backend did not stop on {hits} raw samples (stop text present)")
        mism = [m for r in recs for m in (r.get("sample_meta") or []) if m and m.get("text_matches_reference") is False]
        if mism and recs[0].get("backend_stop_strings"):
            failures.append(f"{model} {config}: vLLM text differs from the reference decoding in {len(mism)} samples "
                            "while vLLM applied the stop strings (stops unreliable)")
        capped = [at_cap(r.get("sample_meta")) for r in recs]
        capped = None if any(c is None for c in capped) else sum(capped)
        if config in MUST_VARY and n_cfg > 1 and all(len(set(r["raw_samples"])) == 1 for r in recs):
            failures.append(f"{model} {config}: samples identical within every task (seed / n>1 problem)")
        same = None
        if args.rerun_root:
            g = args.rerun_root / f.parent.name / f.name
            if g.exists():
                a = {r["task_id"]: r["raw_samples"] for r in recs}
                b = {r["task_id"]: r["raw_samples"] for r in _load(g)}
                k, n = identical_share(a, b)
                same = f"{k}/{n}"
                if k != n:
                    tid = next(t for t in sorted(set(a) & set(b)) if a[t] != b[t])
                    pos = first_divergence(a[tid], b[tid])
                    failures.append(f"{model} {config}: rerun differs on {n - k}/{n} tasks "
                                    f"(first: {tid}, sample {pos[0]}, char {pos[1]})")
        rows.append((model, label, style, len(recs), len(comps), indent4, hits, capped, same,
                     seconds_per_task([r["timestamp"] for r in recs]), recs[0].get("backend_stop_strings")))

    w = max([16] + [len(r[1]) for r in rows])   # 16 (unchanged) unless a __prompt_<variant> stem is longer
    print(f"{'model':36s} {'config':{w}s} {'style':5s} {'tasks':>5s} {'samp':>5s} {'4sp':>6s} "
          f"{'stop-in-raw':>11s} {'at-cap':>6s} {'rerun=':>7s} {'s/task':>7s} {'bk-stop':>7s}")
    for m, c, s, t, n, i4, h, a, same, spt, bks in rows:
        print(f"{m:36s} {c:{w}s} {s:5s} {t:5d} {n:5d} {('%.1f%%' % (100 * i4)) if i4 is not None else '-':>6s} "
              f"{h:11d} {a if a is not None else '-':>6} {same or '-':>7s} "
              f"{('%.1f' % spt) if spt is not None else '-':>7s} {str(bks):>7s}")
    if starts:
        print("\nSTRIPPED-PROMPT COMPLETION STARTS (4sp column above = 4-space share of new-line starts)")
        print(f"{'model':36s} {'file':{w}s} {'nl+4sp':>7s} {'nl-other':>8s} {'quotes-line':>11s} {'empty':>6s}")
        for m, c, sc in starts:
            print(f"{m:36s} {c:{w}s} {sc['new_line_4sp']:7d} {sc['new_line_other']:8d} {sc['same_line']:11d} "
                  f"{sc['empty']:6d}")
    print("\nHARD CHECKS:", "PASS" if not failures else "FAIL")
    for x in failures:
        print("  -", x)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
