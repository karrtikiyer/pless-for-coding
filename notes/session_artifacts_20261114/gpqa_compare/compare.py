"""Hash-only / count-only comparison of two GPQA Diamond copies.

PRIVACY CONTRACT: this script never prints or writes question / option /
answer-option text. Output is limited to counts, booleans, row indices,
lengths, column names, sha256 hashes (first 12 hex), and regex pattern
*descriptions* (character classes, not data). Exceptions are reported by
type only.

Usage:
  uv run python -I compare.py --mirror <parquet> [--official <csv>] [--out <row_hashes.csv>]
Without --official, only the mirror-side analysis runs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import statistics
import sys
import unicodedata
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_LOOSE_DROP = re.compile(r"[^\w\s]|_")  # punctuation incl. $ \ { } ; '_' too


def norm1(s: str) -> str:
    """Level 1 ('strict'): NFKC, whitespace collapsed, stripped."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(s))).strip()


def norm2(s: str) -> str:
    """Level 2 ('loose'): level 1 + lowercase, punctuation/LaTeX delimiters removed."""
    t = norm1(s).lower()
    t = _LOOSE_DROP.sub(" ", t)
    return _WS.sub(" ", t).strip()


NORMS = (("L1_strict", norm1), ("L2_loose", norm2))


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# option-marker templates (described by character class only)
# --------------------------------------------------------------------------
# Each: (description, regex with one group capturing the letter)
MARKER_PATTERNS = [
    ("linestart '[A-D])' ", re.compile(r"(?m)^[ \t]*([A-D])\)[ \t]")),
    ("linestart '([A-D])' ", re.compile(r"(?m)^[ \t]*\(([A-D])\)[ \t]")),
    ("linestart '[A-D].' ", re.compile(r"(?m)^[ \t]*([A-D])\.[ \t]")),
    ("linestart '[A-D]:' ", re.compile(r"(?m)^[ \t]*([A-D]):[ \t]")),
    ("inline '([A-D])' ", re.compile(r"\(([A-D])\)[ \t]")),
    ("inline '[A-D])' ", re.compile(r"(?<![\w(])([A-D])\)[ \t]")),
]
LETTERS = "ABCD"


def find_template(raw: str):
    """Return (pattern_index, [(letter, start, end_of_marker), ...]) for the
    first pattern whose markers in the row are exactly A,B,C,D in order
    (taking the LAST occurrence run of A..D so a stem mentioning '(A)' does
    not break it). None if no pattern conforms."""
    for pi, (_, rx) in enumerate(MARKER_PATTERNS):
        ms = [(m.group(1), m.start(), m.end()) for m in rx.finditer(raw)]
        if len(ms) < 4:
            continue
        # search for the last contiguous A,B,C,D run
        for k in range(len(ms) - 4, -1, -1):
            if "".join(x[0] for x in ms[k:k + 4]) == LETTERS:
                return pi, ms[k:k + 4]
    return None


def conforming_patterns(raw: str) -> list[int]:
    out = []
    for pi, (_, rx) in enumerate(MARKER_PATTERNS):
        letters = "".join(m.group(1) for m in rx.finditer(raw))
        if LETTERS in letters:
            out.append(pi)
    return out


def segments(raw: str):
    """Split a mirror string into (stem, {letter: option_text}) using the
    detected template. Returns None if no template conforms."""
    t = find_template(raw)
    if t is None:
        return None
    _, ms = t
    stem = raw[: ms[0][1]]
    opts = {}
    for i, (L, _s, e) in enumerate(ms):
        end = ms[i + 1][1] if i + 1 < 4 else len(raw)
        opts[L] = raw[e:end]
    return stem, opts


# --------------------------------------------------------------------------
# answer-value normalisation
# --------------------------------------------------------------------------
def answer_repr_is_safe(vals) -> bool:
    return all(re.fullmatch(r"[\(\[ ]*[A-Da-d0-9][\)\]\. ]*", str(v)) for v in vals)


def answer_to_letter(v, digit_base):
    s = re.sub(r"[^A-Za-z0-9]", "", str(v)).upper()
    if len(s) == 1 and s in LETTERS:
        return s
    if len(s) == 1 and s.isdigit() and digit_base is not None:
        i = int(s) - digit_base
        if 0 <= i < 4:
            return LETTERS[i]
    return None


def digit_base_for(vals):
    ds = {re.sub(r"[^0-9]", "", str(v)) for v in vals}
    ds.discard("")
    if not ds:
        return None
    ints = {int(d) for d in ds if len(d) == 1}
    if ints and ints <= {0, 1, 2, 3}:
        return 0
    if ints and ints <= {1, 2, 3, 4}:
        return 1
    return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def len_stats(xs):
    ls = [len(str(x)) for x in xs]
    if not ls:
        return "n/a"
    return f"min={min(ls)} median={statistics.median(ls)} max={max(ls)}"


def dup_report(texts, label):
    for lvl, fn in NORMS:
        c = Counter(sha12(fn(t)) for t in texts)
        groups = [n for n in c.values() if n > 1]
        print(f"  [{label}] duplicates@{lvl}: groups={len(groups)} "
              f"rows_in_dup_groups={sum(groups)} extra_copies={sum(g - 1 for g in groups)}")


def pick_col(cols, *cands, contains=None):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if contains:
        for c in cols:
            if all(x in c.lower() for x in contains):
                return c
    return None


# --------------------------------------------------------------------------
# mirror analysis
# --------------------------------------------------------------------------
def analyse_mirror(path):
    import pandas as pd

    print("== MIRROR ==")
    print(f"  file_sha256={file_sha256(path)}")
    df = pd.read_parquet(path)
    print(f"  rows={len(df)} cols={len(df.columns)}")
    for c in df.columns:
        print(f"  column name={c!s} dtype={df[c].dtype} nulls={int(df[c].isna().sum())}")
    qcol = pick_col(df.columns, "question", "problem", "prompt")
    acol = pick_col(df.columns, "answer", "label", "target")
    if qcol is None or acol is None:
        print("  ERROR: could not identify question/answer columns by name")
        return df, None, None
    qs = [str(x) for x in df[qcol].tolist()]
    print(f"  question_len_chars: {len_stats(qs)}")
    dup_report(qs, "mirror")

    # template markers
    pat_counts = Counter()
    any_marker = 0
    primary = Counter()
    for q in qs:
        cp = conforming_patterns(q)
        for pi in cp:
            pat_counts[pi] += 1
        if cp:
            any_marker += 1
        t = find_template(q)
        primary[t[0] if t else None] += 1
    print(f"  rows_with_embedded_A-D_option_markers (any pattern)={any_marker}/{len(qs)}")
    for pi, (desc, _) in enumerate(MARKER_PATTERNS):
        print(f"  marker_pattern[{pi}] {desc}: rows_containing_A..D={pat_counts.get(pi, 0)} "
              f"rows_where_primary(ordered A,B,C,D)={primary.get(pi, 0)}")
    print(f"  rows_with_no_conforming_template={primary.get(None, 0)}")
    # stem / segment lengths (lengths only)
    seg_ok = [segments(q) for q in qs]
    stem_lens = [len(s[0]) for s in seg_ok if s]
    opt_lens = [len(o) for s in seg_ok if s for o in s[1].values()]
    if stem_lens:
        print(f"  parsed_stem_len: min={min(stem_lens)} median={statistics.median(stem_lens)} max={max(stem_lens)}")
        print(f"  parsed_option_len: min={min(opt_lens)} median={statistics.median(opt_lens)} max={max(opt_lens)}")
    dup_report([s[0] for s in seg_ok if s], "mirror_parsed_stem_only")
    empty_opts = sum(1 for s in seg_ok if s for o in s[1].values() if not norm1(o))
    print(f"  parsed_empty_options={empty_opts}")

    # answers
    avals = [str(x) for x in df[acol].tolist()]
    c = Counter(avals)
    print(f"  answer_distinct_values={len(c)}")
    if answer_repr_is_safe(c.keys()):
        for v, n in sorted(c.items()):
            print(f"    answer_value={v!r} count={n}")
    else:
        print("    (answer values are not single letters/digits; representation withheld)")
        print(f"    counts_sorted={sorted(c.values(), reverse=True)}")
    base = digit_base_for(c.keys())
    letters = [answer_to_letter(v, base) for v in avals]
    print(f"  answer_as_letter_distribution={dict(sorted(Counter(l or 'UNMAPPED' for l in letters).items()))}")
    return df, qcol, acol


# --------------------------------------------------------------------------
# full comparison
# --------------------------------------------------------------------------
def compare(mdf, qcol, acol, official_path, out_path):
    import pandas as pd

    print("== OFFICIAL ==")
    print(f"  file_sha256={file_sha256(official_path)}")
    odf = pd.read_csv(official_path, dtype=str, keep_default_na=False)
    print(f"  rows={len(odf)} cols={len(odf.columns)}")
    print(f"  column_names={list(map(str, odf.columns))}")
    oq = pick_col(odf.columns, "Question", contains=("question",))
    oc = pick_col(odf.columns, "Correct Answer", contains=("correct", "answer"))
    oi = [pick_col(odf.columns, f"Incorrect Answer {k}") for k in (1, 2, 3)]
    print(f"  identified: question={oq!s} correct={oc!s} incorrect={oi!s}")
    if oq is None or oc is None or any(x is None for x in oi):
        print("  ERROR: could not identify official question/answer columns")
        return
    o_q = odf[oq].tolist()
    o_opts = [[r[oc], r[oi[0]], r[oi[1]], r[oi[2]]] for _, r in odf.iterrows()]
    print(f"  question_len_chars: {len_stats(o_q)}")
    dup_report(o_q, "official")
    print("  official correct-option position: N/A — official CSV stores the correct "
          "answer in its own column with no presentation order; any position "
          "distribution is independent of the mirror's option order.")

    m_q = [str(x) for x in mdf[qcol].tolist()]
    m_ans = [str(x) for x in mdf[acol].tolist()]
    base = digit_base_for(set(m_ans))
    m_letter = [answer_to_letter(v, base) for v in m_ans]
    m_norm = {lvl: [fn(q) for q in m_q] for lvl, fn in NORMS}
    m_segs = [segments(q) for q in m_q]

    rows = []
    level_counts = Counter()
    n_one = n_zero = n_multi = 0
    matched_mirror = set()
    agree = disagree = undetermined = 0
    letter_method = Counter()
    for i in range(len(odf)):
        hit, hit_lvl = [], None
        for lvl, fn in NORMS:
            stem = fn(o_q[i])
            opts = [fn(o) for o in o_opts[i]]
            if not stem:
                continue
            hit = [j for j, mq in enumerate(m_norm[lvl])
                   if stem in mq and all(o and o in mq for o in opts)]
            if hit:
                hit_lvl = lvl
                break
        if not hit:
            n_zero += 1
        elif len(hit) == 1:
            n_one += 1
        else:
            n_multi += 1
        level_counts[hit_lvl or "none"] += 1
        matched_mirror.update(hit)

        # answer-letter inference on each matched mirror row
        agrees = []
        for j in hit:
            fn = dict(NORMS)[hit_lvl]
            correct = fn(o_opts[i][0])
            others = [fn(o) for o in o_opts[i][1:]]
            inferred = None
            seg = m_segs[j]
            if seg:
                segn = {L: fn(t) for L, t in seg[1].items()}
                eq = [L for L, t in segn.items() if t == correct]
                if len(eq) == 1:
                    inferred, how = eq[0], "segment_equal"
                else:
                    # option is prefix of its segment (trailing instructions after D)
                    pre = [L for L, t in segn.items() if t.startswith(correct)
                           and not any(t.startswith(o) and len(o) > len(correct) for o in others)]
                    if len(pre) == 1:
                        inferred, how = pre[0], "segment_prefix"
            if inferred is None:
                # positional fallback: order of first occurrence after the stem
                mq = m_norm[hit_lvl][j]
                st = mq.find(fn(o_q[i])) + len(fn(o_q[i]))
                pos = [(mq.find(o, st), k) for k, o in enumerate([correct] + others)]
                if all(p >= 0 for p, _ in pos) and len({p for p, _ in pos}) == 4:
                    order = [k for _, k in sorted(pos)]
                    inferred, how = LETTERS[order.index(0)], "position_fallback"
                else:
                    how = "undetermined"
            letter_method[how] += 1
            if inferred is None or m_letter[j] is None:
                agrees.append(None)
            else:
                agrees.append(inferred == m_letter[j])
        if not hit or any(a is None for a in agrees):
            a_flag = None
        else:
            a_flag = all(agrees)
        if hit:
            if a_flag is True:
                agree += 1
            elif a_flag is False:
                disagree += 1
            else:
                undetermined += 1
        rows.append((i, sha12(norm1(o_q[i])), ";".join(map(str, hit)),
                     hit_lvl or "none",
                     "" if a_flag is None else str(a_flag).lower()))

    print("== ALIGNMENT ==")
    print(f"  official_rows={len(odf)} mirror_rows={len(m_q)}")
    print(f"  official_exactly_one_match={n_one} zero_match={n_zero} multiple_match={n_multi}")
    print(f"  match_level_needed={dict(level_counts)}")
    print(f"  mirror_rows_matched_by_no_official={len(m_q) - len(matched_mirror)}")
    uniq = [(i, int(r[2])) for i, r in enumerate(rows) if r[2] and ";" not in r[2]]
    same_idx = sum(1 for a, b in uniq if a == b)
    print(f"  identical_index_mapping={same_idx}/{len(uniq)} (among unique matches)")
    if len(uniq) >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr([a for a, _ in uniq], [b for _, b in uniq]).correlation
        print(f"  spearman_rho_official_vs_mirror_index={rho:.4f}")
    print("== ANSWERS ==")
    print(f"  letter_inference_method={dict(letter_method)}")
    print(f"  matched_rows answer_agrees={agree} answer_disagrees={disagree} undetermined={undetermined}")
    print(f"  mirror_answer_letter_distribution={dict(sorted(Counter(l or 'UNMAPPED' for l in m_letter).items()))}")

    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("official_row_index,sha12_norm_official_question,matched_mirror_indices,match_level,answer_agrees\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    os.chmod(out_path, 0o600)
    print(f"  row_hashes_written={out_path} rows={len(rows)}")

    N = len(odf)
    full = (n_one == N and len(matched_mirror) == len(m_q) == N)
    if full and same_idx == N:
        verdict = "IDENTICAL CONTENT"
    elif full:
        verdict = "SAME QUESTIONS DIFFERENT ORDER"
    elif n_one + n_multi > 0:
        verdict = f"PARTIAL OVERLAP ({n_one + n_multi}/{N})"
    else:
        verdict = "NO OVERLAP"
    print(f"VERDICT: {verdict} | answers agree={agree} disagree={disagree} undetermined={undetermined}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--official")
    ap.add_argument("--out", default="/tmp/gpqa_compare/row_hashes.csv")
    a = ap.parse_args()
    try:
        mdf, qcol, acol = analyse_mirror(a.mirror)
        if a.official and qcol:
            compare(mdf, qcol, acol, a.official, a.out)
    except Exception as e:  # never echo messages: they could contain data text
        print(f"ERROR: exception of type {type(e).__name__} (message suppressed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
