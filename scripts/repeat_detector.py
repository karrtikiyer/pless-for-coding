"""Streaming n-gram repetition detector for think-phase loop detection.

Content-agnostic: fires when ANY length-n token sequence recurs >= k times within
the recent `window` tokens. Fixed default (n=8, k=4) — NOT tuned on the truncated
traces; validate against productive traces as the negative class.
"""
from collections import deque, Counter


class RepeatDetector:
    def __init__(self, n: int = 8, k: int = 4, window: int = 400):
        self.n = n            # length of the repeating unit to watch (tokens)
        self.k = k            # how many recurrences before we call it a loop
        self.window = window  # only inspect the last `window` tokens
        self.toks: deque[int] = deque(maxlen=window)
        self.fired = False
        self.fire_pos: int | None = None   # 0-based index (in the emitted stream) where it fired
        self.onset: int | None = None      # 0-based index where the looping unit FIRST appeared
        self._step = -1                    # index of the most-recently-appended token

    def update(self, token_id: int) -> bool:
        """Append one emitted token; return True the first step the loop is detected.
        On firing, sets .fire_pos (this index) and .onset (first occurrence of the
        repeating unit) — the position to CHOP back to."""
        self._step += 1
        self.toks.append(int(token_id))
        if self.fired:
            return False
        if len(self.toks) < self.n * self.k:
            return False
        t = list(self.toks)
        counts = Counter(tuple(t[i:i + self.n]) for i in range(len(t) - self.n + 1))
        winner, cnt = counts.most_common(1)[0]
        if cnt >= self.k:
            self.fired = True
            self.fire_pos = self._step
            # map the winning n-gram's first occurrence in the window to a global index
            deque_start = self._step - len(self.toks) + 1
            first_i = next(i for i in range(len(t) - self.n + 1)
                           if tuple(t[i:i + self.n]) == winner)
            self.onset = deque_start + first_i
            return True
        return False


def scan(tokens, n: int, k: int, window: int):
    """Fast, allocation-light equivalent of replaying ``RepeatDetector(n,k,window).update``
    over ``tokens`` and reporting the FIRST fire. Returns ``(fired, fire_pos, onset)``.

    Semantics are identical to ``RepeatDetector`` (proven bit-for-bit in
    ``tests/test_repeat_detector.py`` against a brute replay), but it maintains the n-gram
    counts incrementally (add the new rightmost n-gram, drop the one that slid out of the
    window) instead of rebuilding a Counter over the whole window every token. That turns
    per-token cost from O(window) into O(n), which makes full-trace sweeps feasible with
    the SAME logic the live detector deploys — no separate strided approximation.
    """
    buf: list[int] = []
    counts: dict[tuple, int] = {}
    first_checked = False
    prev_lo = 0
    for step, tk in enumerate(tokens):
        buf.append(int(tk))
        hi = len(buf)
        lo = hi - window if hi > window else 0
        # An n-gram slides out of the window iff lo advanced past a start that had been a
        # full n-gram in the previous window.
        if lo > prev_lo and prev_lo + n <= hi - 1:
            dg = tuple(buf[prev_lo:prev_lo + n])
            c = counts.get(dg, 0) - 1
            if c <= 0:
                counts.pop(dg, None)
            else:
                counts[dg] = c
        prev_lo = lo
        new_ng = None
        if hi - lo >= n:
            new_ng = tuple(buf[hi - n:hi])
            counts[new_ng] = counts.get(new_ng, 0) + 1
        if (hi - lo) < n * k:          # same guard as RepeatDetector.update
            continue
        winner = None
        if not first_checked:
            # First step past the guard: an n-gram may have reached k while checks were
            # suppressed, so do one full max scan (matches update()'s per-step max check).
            first_checked = True
            wn, cnt = max(counts.items(), key=lambda kv: kv[1])
            if cnt >= k:
                winner = wn
        elif new_ng is not None and counts.get(new_ng, 0) >= k:
            # After the first check, max can only rise via the newly added n-gram
            # (evictions only decrease counts), so checking new_ng is sufficient.
            winner = new_ng
        if winner is not None:
            onset = next(i for i in range(lo, hi - n + 1)
                         if tuple(buf[i:i + n]) == winner)
            return True, step, onset
    return False, None, None


if __name__ == "__main__":
    # Unit test: must FIRE on a verbatim loop, must NOT fire on varied/productive text.
    def run(tokens, **kw):
        d = RepeatDetector(**kw)
        fired_at = None
        for i, tk in enumerate(tokens):
            if d.update(tk) and fired_at is None:
                fired_at = i
        return fired_at

    # 1) verbatim loop: "Now , the code ." (5-token unit) repeated 12x
    loop = [10, 11, 12, 13, 14] * 12
    fa = run(loop, n=8, k=4)
    assert fa is not None, "FAIL: did not fire on a verbatim loop"
    print(f"[ok] verbatim loop: fired at token {fa} (after ~{fa} tokens of loop)")

    # 1b) onset must point at/near the START of the loop (chop target), not the fire point
    d = RepeatDetector(n=8, k=4)
    for tk in loop:
        if d.update(tk):
            break
    assert d.onset is not None and d.onset <= 5, f"FAIL: onset {d.onset} not near loop start"
    print(f"[ok] onset = {d.onset} (chop target, near loop start; fired at {d.fire_pos})")

    # 2) productive/varied text: no short unit repeats k times
    import itertools
    varied = list(itertools.chain.from_iterable([[i, i + 1, i + 2, i + 3] for i in range(0, 400, 4)]))
    fa = run(varied, n=8, k=4)
    assert fa is None, f"FAIL: false-positive on varied text at {fa}"
    print("[ok] varied text: did NOT fire (no false positive)")

    # 3) short-unit loop "A B A B ..." (our 'inverse(6)*3=' style), 2-token unit x 30
    loop2 = [99, 100] * 30
    fa = run(loop2, n=8, k=4)
    assert fa is not None, "FAIL: did not fire on a 2-token loop"
    print(f"[ok] 2-token loop: fired at token {fa}")

    # 4) legitimate brief repetition (3 reps) must NOT fire at k=4
    brief = [7, 8, 9, 10, 11, 12, 13, 14] * 3 + list(range(50, 120))
    fa = run(brief, n=8, k=4)
    assert fa is None, f"FAIL: fired on benign 3x repetition at {fa}"
    print("[ok] benign 3x repetition: did NOT fire at k=4")

    print("\nAll RepeatDetector unit tests passed.")
