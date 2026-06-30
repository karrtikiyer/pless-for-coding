"""Phase 0 (CPU) — screen p-less loop traces and locate a verbatim recurring
anchor + per-cycle positions, for replicating Circular-Reasoning Figs 3b & 4
(arXiv:2601.05693) on our naturalistic ATCODER code-reasoning loops.

What this does, per model, from the raw pless JSONL (decoded text only):
  1. Extract the <think> stream (model-aware: Qwen emits <think>…</think>;
     DeepSeek-R1-Distill has <think> in the PROMPT, so the generation is pure
     reasoning and truncation == no </think>).
  2. Keep TRUNCATED samples (never closed </think>) and re-tokenize.
  3. Run the validated streaming n-gram detector (simulate_onset) → onset_token.
  4. Find the dominant verbatim repeating n-gram near onset; its first token is
     the ANCHOR (analog of the paper's "\n\nBut"); its occurrence start indices
     are the per-cycle anchor positions. Keep traces with >= --min-cycles clean,
     roughly evenly-spaced cycles.
  5. Find a NORMAL-baseline anchor: a short n-gram recurring in the PRE-onset
     (normal-reasoning) region → gives the paper's "Normal 1/2" curves.
  6. Rank by (cycle count desc, period regularity), pick top --top per model,
     reconstruct the faithful prompt (format_prompt_apps_instruct, the exact
     generation code path) and emit a manifest JSONL for the GPU extractor.

NO GPU, NO model weights — tokenizer + n-gram detector only. Same
re-tokenization caveat as the pilot1 work (only decoded text was stored).

Usage:
  HF_HUB_OFFLINE=1 uv run python scripts/loop_collapse_screen.py \
      --model Qwen/Qwen3-8B \
      --jsonl results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl \
      --out results/loop_collapse_replication/Qwen--Qwen3-8B/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# single source of truth for the detector + per-model params
from scripts.signal_diagnostic import simulate_onset, LOOP_PARAMS  # noqa: E402
from bench.apps.dataset import load_apps_test_map  # noqa: E402
from bench.apps.prompts import format_prompt_apps_instruct  # noqa: E402

MAX_CTX_TOKENS = 38000   # absolute safety cap (prompt + think), matches pilot1
NORMAL_NGRAM = 8         # shorter n-gram for the normal-baseline anchor

# Statement-loop period bounds: we target SENTENCE-level cycles (the paper's
# Fig 3b/4 "statement loop", anchor like "\n\nBut"), NOT degenerate single-token /
# numeric runs (period 1, e.g. "999…9") which are the Fig-3a "numerical loop" case
# and give a meaningless cycle-k-vs-(k-1) comparison.
PMIN_STATEMENT = 10      # min repeating-unit length (tokens) to count as a statement loop
                         # (data-driven: period 1 = single-token, 2-9 = short fragments,
                         #  >=10 = sentence-level statement loops; see period diagnostic)
PMAX_PERIOD = 800        # max statement-cycle length (empirically nothing exceeds this)
PERIOD_MATCH_THRESH = 0.85   # token-identity fraction at lag P for "verbatim" periodicity
# Period detection runs on a LOOP-DOMINATED region starting at onset — NOT symmetric
# around it. Including pre-onset (non-loop) tokens dilutes the match fraction and
# under-detects real loops. This region is DECOUPLED from the n-gram detector window.
PERIOD_REGION = 3000


# ---------------------------------------------------------------------------
# Model-aware <think> extraction
# ---------------------------------------------------------------------------

def extract_think(raw: str) -> tuple[str, bool]:
    """Return (think_str, is_complete).

    Qwen: model emits "<think> … </think>". DeepSeek-R1-Distill: <think> lives in
    the prompt, so the stored generation is pure reasoning and completion is
    signalled solely by a "</think>" closing tag.
    """
    ts = raw.find("<think>")
    te = raw.find("</think>")
    if ts >= 0:                                   # Qwen-style: include the <think> tag
        complete = te > ts
        return (raw[ts:te] if complete else raw[ts:]), complete
    # DeepSeek-style: no opening tag in the generation
    complete = te >= 0
    return (raw[:te] if complete else raw), complete


# ---------------------------------------------------------------------------
# Anchor / cycle detection
# ---------------------------------------------------------------------------

def _detect_period(arr: np.ndarray, dmin: int, dmax: int, thresh: float) -> int | None:
    """Smallest lag d in [dmin, dmax] at which the token stream is >= `thresh`
    self-identical (arr[i] == arr[i+d]) — the fundamental statement-cycle period.
    Smallest-meeting-threshold avoids locking onto 2P/3P harmonics."""
    n = len(arr)
    dmax = min(dmax, n - 1)
    for d in range(dmin, dmax + 1):
        if float(np.mean(arr[: n - d] == arr[d:])) >= thresh:
            return d
    return None


def _longest_periodic_run(positions: list[int], P: float) -> list[int]:
    """Longest run of positions whose consecutive gaps are ~P (one per cycle)."""
    if not positions:
        return []
    best: list[int] = []
    cur = [positions[0]]
    for prev, p in zip(positions, positions[1:]):
        if 0.5 * P <= (p - prev) <= 1.5 * P:
            cur.append(p)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [p]
    return cur if len(cur) > len(best) else best


def find_loop_anchor(ids: list[int], onset: int, n: int, k: int, window: int) -> dict | None:
    """Locate a STATEMENT-level verbatim cycle around onset and return a once-per-
    cycle anchor token + its ordered per-cycle positions (think-token coordinates).

    Method: (1) find the fundamental period P by token-stream autocorrelation in
    [PMIN_STATEMENT, PMAX_PERIOD] (excludes degenerate period-1 numeric/char runs);
    (2) pick the anchor = the RAREST token within one period (a clean, distinctive
    once-per-cycle marker — the analog of the paper's "\\n\\nBut"); (3) collect that
    token's positions and keep the longest ~P-spaced run = the consecutive cycles.
    """
    arr = np.asarray(ids)
    # LOOP-DOMINATED region: from onset forward (the loop is solid post-onset). A
    # small pre-onset lead-in is included only if the post-onset tail is too short.
    hi = min(len(ids), onset + PERIOD_REGION)
    lo = onset
    if hi - lo < PMIN_STATEMENT * 2:
        lo = max(0, onset - PERIOD_REGION // 2)
    region = arr[lo:hi]
    if len(region) < PMIN_STATEMENT * 2:
        return None
    P = _detect_period(region, PMIN_STATEMENT, PMAX_PERIOD, PERIOD_MATCH_THRESH)
    if P is None:
        return None
    unit = region[:P]
    vals, counts = np.unique(unit, return_counts=True)
    # The anchor must occur EXACTLY ONCE per period, else its positions are spaced
    # at a sub-period (e.g. a comma every ~5 tokens inside a 200-token sentence) and
    # "cycles" become a sub-period artifact, not statement repetitions. Among the
    # once-per-period tokens, pick the GLOBALLY rarest → cleanest cycle markers.
    once = [int(v) for v, c in zip(vals.tolist(), counts.tolist()) if c == 1]
    if not once:
        return None                                     # no clean once-per-cycle marker
    gcount = Counter(arr.tolist())
    anchor_tok = min(once, key=lambda t: gcount[t])
    positions = [int(i) for i in np.where(arr == anchor_tok)[0]]
    run = _longest_periodic_run(positions, P)
    if len(run) < 2:
        return None
    gaps = np.diff(run)
    return {
        "anchor_token_id": anchor_tok,
        "gram": [int(x) for x in unit],                 # the repeating unit (for inspection)
        "ngram_n": int(P),
        "positions": run,
        "n_cycles": len(run),
        "period_median": float(np.median(gaps)),
        "period_cv": float(np.std(gaps) / max(np.mean(gaps), 1e-9)),
    }


def find_normal_anchor(ids: list[int], onset: int, n: int = NORMAL_NGRAM,
                       min_occ: int = 3) -> dict | None:
    """A short n-gram recurring in the PRE-onset (normal-reasoning) region, for the
    paper's "Normal 1/2" baseline. Prefer occurrences spread across the region."""
    pre_hi = max(n, onset // 2)          # first half of the pre-onset region = clearly "normal"
    region = ids[:pre_hi]
    if len(region) < n * (min_occ + 1):
        return None
    counts = Counter(tuple(region[j:j + n]) for j in range(len(region) - n + 1))
    for gram, c in counts.most_common(20):
        if c < min_occ:
            break
        positions = [i for i in range(pre_hi - n + 1) if tuple(ids[i:i + n]) == gram]
        # require they are not all adjacent (avoid a tiny local repeat)
        if len(positions) >= min_occ and (positions[-1] - positions[0]) >= n * min_occ:
            return {"anchor_token_id": int(gram[0]), "gram": list(gram),
                    "ngram_n": n, "positions": [int(p) for p in positions]}
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", default="ATCODER")
    ap.add_argument("--difficulty", default="interview")
    ap.add_argument("--top", type=int, default=5, help="max traces to emit")
    ap.add_argument("--min-cycles", type=int, default=6,
                    help="min verbatim cycles (>=6 ⇒ Repeat 1..5 curves)")
    ap.add_argument("--max-records", type=int, default=None, help="debug: cap tasks scanned")
    ap.add_argument("--catalog", type=Path, default=None,
                    help="dump ALL candidates' metadata (period, cycles, decoded unit; "
                         "no token arrays) for transparent hand-selection")
    ap.add_argument("--select", type=str, default=None,
                    help="emit manifest for EXACTLY these traces, e.g. '2656:5,117:5,1469:1' "
                         "(overrides --top ranking)")
    args = ap.parse_args()

    if args.model not in LOOP_PARAMS:
        sys.exit(f"no detector params for {args.model}; add to LOOP_PARAMS")
    lp = LOOP_PARAMS[args.model]
    n, k, window = lp["n"], lp["k"], lp["window"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tmap = load_apps_test_map(source=args.source, difficulty=args.difficulty)

    n_samples = n_trunc = n_onset = 0
    candidates: list[dict] = []

    with open(args.jsonl) as f:
        for li, line in enumerate(f):
            if args.max_records is not None and li >= args.max_records:
                break
            d = json.loads(line)
            tid = d["task_id"]
            if tid not in tmap:
                continue
            for si, raw in enumerate(d.get("samples_with_thinking", [])):
                n_samples += 1
                think_str, complete = extract_think(raw)
                if complete or not think_str.strip():
                    continue                       # only truncated loops here
                n_trunc += 1
                ids = tok.encode(think_str, add_special_tokens=False)
                if len(ids) > MAX_CTX_TOKENS:
                    ids = ids[:MAX_CTX_TOKENS]
                onset = simulate_onset(ids, n, k, window)
                if onset is None:
                    continue
                n_onset += 1
                anc = find_loop_anchor(ids, onset, n, k, window)
                if anc is None or anc["n_cycles"] < args.min_cycles:
                    continue
                candidates.append({
                    "task_id": tid, "sample_idx": si,
                    "think_token_ids": ids, "onset_token": onset,
                    **anc,
                })
            if li % 25 == 0:
                print(f"  scanned {li+1} tasks | trunc={n_trunc} onset={n_onset} "
                      f"candidates={len(candidates)}", flush=True)

    # rank: most cycles first, then most regular period (lowest CV)
    candidates.sort(key=lambda c: (-c["n_cycles"], c["period_cv"]))

    # full catalog (lightweight) for transparent hand-selection
    if args.catalog is not None:
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        with open(args.catalog, "w") as cf:
            for c in candidates:
                cf.write(json.dumps({
                    "task_id": c["task_id"], "sample_idx": c["sample_idx"],
                    "n_cycles": c["n_cycles"], "period_median": c["period_median"],
                    "period_cv": c["period_cv"],
                    "onset_token": c["onset_token"],           # for onset-matched selection
                    "n_think": len(c["think_token_ids"]),
                    "anchor_str": tok.decode([c["anchor_token_id"]]),
                    "unit_str": tok.decode(c["gram"])[:240],
                }) + "\n")
        print(f"catalog ({len(candidates)} candidates) -> {args.catalog}")

    # selection: explicit --select wins; else top-N by ranking
    if args.select:
        want = []
        for tok_sel in args.select.split(","):
            t, s = tok_sel.split(":")
            want.append((int(t), int(s)))
        by_key = {(c["task_id"], c["sample_idx"]): c for c in candidates}
        missing = [w for w in want if w not in by_key]
        if missing:
            print(f"WARNING: --select traces not among candidates: {missing}")
        picked = [by_key[w] for w in want if w in by_key]
    else:
        picked = candidates[:args.top]

    print(f"\n=== {args.model} screening summary ===")
    print(f"samples={n_samples}  truncated={n_trunc}  truncated_with_onset={n_onset}  "
          f"clean_verbatim_candidates(>= {args.min_cycles} cycles)={len(candidates)}")
    print(f"{'rank':>4} {'task':>6} {'smp':>3} {'cycles':>6} {'period':>7} {'cv':>5}  anchor")
    for r, c in enumerate(candidates[:15]):
        anchor_str = tok.decode([c["anchor_token_id"]])
        print(f"{r:>4} {c['task_id']:>6} {c['sample_idx']:>3} {c['n_cycles']:>6} "
              f"{c['period_median']:>7.1f} {c['period_cv']:>5.2f}  {anchor_str!r}")

    if not picked:
        print(f"\nNO qualifying verbatim loops for {args.model} "
              f"(>= {args.min_cycles} cycles). Fig 4 not feasible from this screen.")
        return

    # reconstruct faithful prompt + normal anchor for the picked traces, emit manifest
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.out, "w") as fout:
        for c in picked:
            prob = tmap[c["task_id"]]
            prompt_str, _ = format_prompt_apps_instruct(prob, tok, enable_thinking=True)
            prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
            normal = find_normal_anchor(c["think_token_ids"], c["onset_token"])
            rec = {
                "model": args.model,
                "task_id": c["task_id"], "sample_idx": c["sample_idx"],
                "n_prompt_tokens": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "think_token_ids": c["think_token_ids"],
                "n_think_tokens": len(c["think_token_ids"]),
                "onset_token": c["onset_token"],
                "anchor_token_id": c["anchor_token_id"],
                "anchor_str": tok.decode([c["anchor_token_id"]]),
                "anchor_gram_str": tok.decode(c["gram"]),
                "anchor_ngram_n": c["ngram_n"],
                "loop_anchor_positions": c["positions"],
                "n_cycles": c["n_cycles"],
                "period_median": c["period_median"],
                "period_cv": c["period_cv"],
                "normal_anchor_token_id": normal["anchor_token_id"] if normal else None,
                "normal_anchor_positions": normal["positions"] if normal else [],
                "normal_anchor_str": tok.decode([normal["anchor_token_id"]]) if normal else None,
            }
            # spot-check: every anchor position must carry the anchor token id
            for p in rec["loop_anchor_positions"]:
                assert rec["think_token_ids"][p] == rec["anchor_token_id"], \
                    f"{rec['task_id']}[{rec['sample_idx']}]: anchor pos {p} token mismatch"
            fout.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"\nemitted {n_written} traces -> {args.out}")
    for c in picked:
        print(f"  task {c['task_id']}[{c['sample_idx']}]: {c['n_cycles']} cycles, "
              f"anchor={tok.decode([c['anchor_token_id']])!r}")


if __name__ == "__main__":
    main()
