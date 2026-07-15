"""Prove scripts.repeat_detector.scan() is bit-identical to replaying RepeatDetector.update.

scan() is the fast (incremental-count) path used for offline window sweeps; update() is the
live detector the adaptive run deploys. They MUST agree on (fired, fire_pos, onset) or the
tuned window wouldn't transfer to deployment — the exact "written differently each time"
inconsistency this guards against.
"""
import random

from scripts.repeat_detector import RepeatDetector, scan


def brute(tokens, n, k, window):
    """Reference: replay the deployed update() and report the first fire."""
    d = RepeatDetector(n=n, k=k, window=window)
    for t in tokens:
        if d.update(t):
            return True, d.fire_pos, d.onset
    return False, None, None


def test_scan_matches_update_on_random_sequences():
    rng = random.Random(1234)
    for _ in range(500):
        n = rng.choice([2, 3, 4, 5, 8, 30])
        k = rng.choice([2, 3, 4, 6, 8])
        window = rng.choice([8, 20, 50, 400, 1600, 3000])
        vocab = rng.choice([2, 3, 5, 10, 50])
        L = rng.randint(0, 300)
        toks = [rng.randrange(vocab) for _ in range(L)]
        # half the time, splice in a verbatim loop (the thing the detector must catch)
        if L > 20 and rng.random() < 0.5:
            unit = [rng.randrange(vocab) for _ in range(rng.randint(1, 6))]
            loop = unit * rng.randint(2, 12)
            pos = rng.randint(0, max(0, L - len(loop)))
            toks[pos:pos + len(loop)] = loop
        assert scan(toks, n, k, window) == brute(toks, n, k, window), \
            (n, k, window, vocab, toks)


def test_scan_overlap_and_guard_deferred_edges():
    # 'AAAA…' — overlapping n-grams whose k-th occurrence can complete before n*k tokens,
    # i.e. while update()'s guard suppresses checking (the case that breaks a naive
    # new-ngram-only fast path).
    for n, k in [(2, 2), (2, 3), (3, 2), (2, 5), (4, 3)]:
        for window in [10, 50, 400]:
            for L in range(0, 40):
                toks = [7] * L
                assert scan(toks, n, k, window) == brute(toks, n, k, window), (n, k, window, L)


def test_scan_two_competing_loops():
    # two different units so the winner isn't always the most-recent n-gram
    toks = [1, 2] * 5 + [3, 4, 5] * 6 + [1, 2] * 4
    for n in (2, 3):
        for k in (3, 4, 5):
            for w in (8, 20, 100):
                assert scan(toks, n, k, w) == brute(toks, n, k, w), (n, k, w)


def test_scan_no_fire_on_clean_text():
    toks = list(range(600))
    assert scan(toks, 30, 8, 3000) == (False, None, None)
    assert brute(toks, 30, 8, 3000) == (False, None, None)
