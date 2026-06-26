"""Pilot 1 — Phase 1: trace selection + sentence segmentation (CPU only).

Builds a deterministic MANIFEST for the hidden-state loop-detection pilot. NO GPU,
NO model weights — just tokenizer + the validated n-gram onset detector.

What it produces, per selected trace:
  - the exact think-block token ids (capped) that Phase 2 will feed the model,
  - a partition of those tokens into sentence/line units (split on .?! or newline),
    with short units (<MIN_UNIT_TOKENS) merged forward,
  - the onset sentence index (sentence containing the n-gram onset token),
  - a per-sentence label: "loop" (onset sentence and after) vs "normal" (before),
    and for clean traces all "normal".

Three groups, reported SEPARATELY downstream (no pooling — avoids the terminal-only
success bias):
  terminal  = looping_truncated with an n-gram onset (deepest loops)
  transient = looping_completed with an n-gram onset (recovered loops)
  clean     = completed, no n-gram onset (negative / FPR)

Design invariants (asserted at build time):
  I1. Every think token is assigned to exactly one sentence (a true partition).
  I2. Sentence token ranges are contiguous and cover [0, n_think) with no gaps.
  I3. After merging, every kept unit has >= MIN_UNIT_TOKENS tokens (except possibly
      one trailing unit).
  I4. For loop traces, onset_token lies inside the onset sentence's token range.

Faithfulness: the analyzed think stream INCLUDES the leading "<think>" tag and is
tokenized as one piece; the prompt is rebuilt via the exact generation code path
(format_prompt_apps_instruct, enable_thinking=True). Token indices (onset, spans)
are in this <think>-inclusive stream. Known limitation: only decoded text was
stored, so re-tokenization may differ slightly from the emitted ids.

Usage:
  HF_HUB_OFFLINE=1 uv run python scripts/pilot1_segment.py \
      --jsonl results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl \
      --model Qwen/Qwen3-8B --n-per-group 30 --out results/pilot1_hidden/manifest.jsonl
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

# reuse the SINGLE source of truth for the n-gram onset detector + params
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from scripts.signal_diagnostic import simulate_onset, LOOP_PARAMS  # noqa: E402
from bench.apps.dataset import load_apps_test_map  # noqa: E402
from bench.apps.prompts import format_prompt_apps_instruct  # noqa: E402

MIN_UNIT_TOKENS = 5          # units below this are merged forward
POST_ONSET_MARGIN = 500      # for terminal loops: cap think at onset + this
MAX_CTX_TOKENS = 38000       # absolute safety cap (prompt + think)

_SENT_SPLIT = re.compile(r"[.!?]+\s+|\n+")


# ---------------------------------------------------------------------------
# Segmentation primitives
# ---------------------------------------------------------------------------

def sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Split text into [start,end) char spans on (.?! + whitespace) OR newline.
    The delimiter is kept at the END of the preceding span. Spans tile the whole
    string with no gaps."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.end()
        if end > start:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def assign_tokens_to_sentences(
    char_spans: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
) -> list[int]:
    """Assign each token to exactly one sentence by its START char (guarantees a
    partition — no token straddles two units). Returns sentence index per token."""
    starts = [cs for cs, _ in char_spans]
    out: list[int] = []
    prev = 0
    for (a, _b) in offsets:
        j = bisect.bisect_right(starts, a) - 1
        j = max(0, min(j, len(char_spans) - 1))
        j = max(prev, j)          # S3: enforce non-decreasing (insurance vs odd offsets)
        out.append(j)
        prev = j
    return out


def token_ranges_from_assignment(sent_of_tok: list[int]) -> list[tuple[int, int]]:
    """Contiguous [tok_start, tok_end) per sentence index that actually owns tokens.
    Sentences with no tokens are dropped. Relies on sent_of_tok being non-decreasing
    (true because tokens are in order and assigned by start char)."""
    ranges: list[tuple[int, int]] = []
    if not sent_of_tok:
        return ranges
    cur = sent_of_tok[0]
    run_start = 0
    for i in range(1, len(sent_of_tok)):
        if sent_of_tok[i] != cur:
            ranges.append((run_start, i))
            run_start = i
            cur = sent_of_tok[i]
    ranges.append((run_start, len(sent_of_tok)))
    return ranges


def merge_short_units(
    ranges: list[tuple[int, int]],
    min_tokens: int = MIN_UNIT_TOKENS,
) -> list[tuple[int, int]]:
    """Merge any unit with < min_tokens tokens INTO THE NEXT unit. A too-short final
    unit is merged into the previous one instead. Preserves contiguity & coverage."""
    if not ranges:
        return ranges
    merged: list[tuple[int, int]] = []
    carry_start: int | None = None
    for (a, b) in ranges:
        s = carry_start if carry_start is not None else a
        if (b - s) < min_tokens:
            carry_start = s          # too short: roll forward into next unit
            continue
        merged.append((s, b))
        carry_start = None
    if carry_start is not None:       # trailing short remainder
        if merged:
            pa, _pb = merged[-1]
            merged[-1] = (pa, ranges[-1][1])   # fold into previous
        else:
            merged.append((carry_start, ranges[-1][1]))  # whole thing is one short unit
    return merged


# ---------------------------------------------------------------------------
# Per-trace manifest
# ---------------------------------------------------------------------------

def build_trace_record(
    tokenizer, problem, task_id, sample_idx, cls, think_str,
    loop_params: dict,
) -> dict:
    """Reconstruct the FAITHFUL prompt context, tokenize the analyzed think stream
    (which INCLUDES the leading ``<think>`` tag) once with offsets, segment, label.

    Faithfulness notes:
      * The prompt is rebuilt via the exact generation code path
        (``format_prompt_apps_instruct(..., enable_thinking=True)``), reproducing
        the chat-templated string vLLM was fed — so the prompt tokens (incl. the
        position-0 attention sink) match generation.
      * ``think_str`` is ``s[<think> : </think>]`` (or to EOS if truncated) and is
        tokenized AS ONE PIECE — never spliced — to avoid BPE-merge artifacts at
        the ``<think>\\n`` boundary.
      * Known, unavoidable approximation: only decoded TEXT was stored (not the
        original generation token ids), so re-tokenizing may differ slightly from
        the emitted ids. Same limitation as the Σpᵢ² probe; discrepancy is tiny.
    """
    # 1) faithful prompt prefix (chat-templated; matches what the model saw)
    prompt_str, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    # 2) analyzed think stream (includes "<think>\n..."), tokenized as one piece
    enc = tokenizer(think_str, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"])
    offs = [tuple(x) for x in enc["offset_mapping"]]
    assert len(ids) == len(offs), "ids/offsets length mismatch"
    if not ids:                                  # S1: empty think block has no usable signal
        raise ValueError(f"{task_id}[{sample_idx}]: empty think block")

    # onset on the analyzed stream (the ~2 <think> prefix tokens don't loop; the
    # onset index is simply shifted by them — consistent with this stream)
    onset = simulate_onset(ids, loop_params["n"], loop_params["k"], loop_params["window"])

    # cap: terminal loops only need pre-onset + margin
    if cls == "looping_truncated" and onset is not None:
        cap = onset + POST_ONSET_MARGIN
        ids, offs = ids[:cap], offs[:cap]
    # absolute safety: prompt + think must fit the model context with headroom
    think_budget = MAX_CTX_TOKENS - len(prompt_ids)
    if len(ids) > think_budget:
        ids, offs = ids[:think_budget], offs[:think_budget]
    n_think = len(ids)
    # S2: a loop trace must keep its onset inside the window, else it is silently
    # relabeled all-"normal". Fail loud rather than corrupt the group.
    if cls != "clean" and onset is not None and onset >= n_think:
        raise ValueError(
            f"{task_id}[{sample_idx}]: onset {onset} capped out of window "
            f"(n_think={n_think}); raise MAX_CTX_TOKENS or check this trace")

    # segment over the (possibly capped) char range covered by kept tokens
    char_end = offs[-1][1] if offs else 0
    spans = sentence_char_spans(think_str[:char_end])
    sent_of_tok = assign_tokens_to_sentences(spans, offs)
    raw_ranges = token_ranges_from_assignment(sent_of_tok)
    ranges = merge_short_units(raw_ranges)

    # onset sentence + labels
    onset_sentence = None
    if onset is not None and onset < n_think:
        onset_sentence = next(i for i, (a, b) in enumerate(ranges) if a <= onset < b)

    sentences = []
    for i, (a, b) in enumerate(ranges):
        if onset_sentence is None:
            label = "normal"
        else:
            label = "loop" if i >= onset_sentence else "normal"
        sentences.append({"tok_start": a, "tok_end": b, "n_tokens": b - a, "label": label})

    rec = {
        "task_id": task_id, "sample_idx": sample_idx, "cls": cls,
        "prompt_token_ids": prompt_ids, "n_prompt_tokens": len(prompt_ids),
        "think_token_ids": ids, "n_think_tokens": n_think,
        "onset_token": onset, "onset_sentence": onset_sentence,
        "sentences": sentences,
    }
    _validate(rec)
    return rec


def _validate(rec: dict) -> None:
    """Assert the design invariants I1–I4."""
    n = rec["n_think_tokens"]
    sents = rec["sentences"]
    # I2: contiguous + full coverage [0, n)
    assert sents[0]["tok_start"] == 0, f"{rec['task_id']}: first unit must start at 0"
    assert sents[-1]["tok_end"] == n, f"{rec['task_id']}: last unit must end at n_think"
    for a, b in zip(sents, sents[1:]):
        assert a["tok_end"] == b["tok_start"], f"{rec['task_id']}: gap/overlap between units"
    # I3: all but possibly the last unit are >= MIN_UNIT_TOKENS
    for s in sents[:-1]:
        assert s["n_tokens"] >= MIN_UNIT_TOKENS, f"{rec['task_id']}: short unit survived merge"
    # I4: onset token inside onset sentence
    if rec["onset_sentence"] is not None:
        os = sents[rec["onset_sentence"]]
        assert os["tok_start"] <= rec["onset_token"] < os["tok_end"], \
            f"{rec['task_id']}: onset token not inside onset sentence"
    # labels monotonic (all normal then all loop)
    labels = [s["label"] for s in sents]
    if "loop" in labels:
        first_loop = labels.index("loop")
        assert all(l == "loop" for l in labels[first_loop:]), \
            f"{rec['task_id']}: labels not monotonic normal->loop"


# ---------------------------------------------------------------------------
# Selection + driver
# ---------------------------------------------------------------------------

def main() -> None:
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="ATCODER")
    ap.add_argument("--difficulty", default="interview")
    ap.add_argument("--n-per-group", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    random.seed(args.seed)

    lp = LOOP_PARAMS[args.model]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    # problem_id -> AppsProblem, for faithful prompt reconstruction
    tmap = load_apps_test_map(source=args.source, difficulty=args.difficulty)

    # First pass: classify every sample cheaply (onset only) to pick groups. The
    # analyzed stream INCLUDES "<think>" (s[ts:...]) — same stream used downstream.
    groups: dict[str, list] = {"terminal": [], "transient": [], "clean": []}
    with open(args.jsonl) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] not in tmap:         # need the problem to rebuild the prompt
                continue
            for si, s in enumerate(d.get("samples_with_thinking", [])):
                ts = s.find("<think>"); te = s.find("</think>")
                if ts < 0:
                    continue
                complete = te > ts
                think_str = s[ts: te] if complete else s[ts:]   # INCLUDE <think>
                if not think_str.strip():        # S1: skip empty/whitespace think blocks
                    continue
                ids = tok.encode(think_str, add_special_tokens=False)
                onset = simulate_onset(ids, lp["n"], lp["k"], lp["window"])
                entry = (d["task_id"], si, think_str)
                if complete and onset is None:
                    groups["clean"].append(entry)
                elif complete and onset is not None:
                    groups["transient"].append(entry)
                elif not complete and onset is not None:
                    groups["terminal"].append(entry)
                # not-complete & no onset excluded — no usable anchor

    print("available per group:", {k: len(v) for k, v in groups.items()})
    cls_map = {"terminal": "looping_truncated", "transient": "looping_completed", "clean": "clean"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.out, "w") as fout:
        for g in ("terminal", "transient", "clean"):
            picked = random.sample(groups[g], min(args.n_per_group, len(groups[g])))
            for (tid, si, think_str) in picked:
                rec = build_trace_record(tok, tmap[tid], tid, si, cls_map[g], think_str, lp)
                fout.write(json.dumps(rec) + "\n")
                n_written += 1
            print(f"  {g}: wrote {len(picked)}")
    print(f"manifest -> {args.out}  ({n_written} traces)")


if __name__ == "__main__":
    main()
