"""TDD for the live adaptive loop-rescue control logic (scripts/adaptive_loop.py).

Model-free: an injected scripted round_fn yields pre-scripted (gen, reason, onset) tuples
and records which sampler/temp it was called with, so we can assert the base->escape
switch happens on the first loop detection and persists.
"""
from scripts.adaptive_loop import adaptive_continue

BASE, ESCAPE = "pless_a2", "pless_a5"   # sentinel sampler objects


def _scripted(scripts):
    calls = []
    it = iter(scripts)

    def round_fn(ctx, sampler, temp, budget):
        calls.append({"sampler": sampler, "temp": temp, "budget": budget, "ctx_len": len(ctx)})
        return next(it)

    round_fn.calls = calls
    return round_fn


def _run(rf, **kw):
    defaults = dict(prompt_ids=[7], round_fn=rf, base_sampler=BASE, base_temp=1.0,
                    escape_sampler=ESCAPE, escape_temp=1.0, max_total=1000, max_chops=3)
    defaults.update(kw)
    return adaptive_continue(**defaults)


def test_no_loop_stays_on_base_sampler():
    rf = _scripted([([1, 2, 3], "eos", None)])
    r = _run(rf)
    assert r["tokens"] == [1, 2, 3]
    assert r["reason"] == "eos"
    assert r["fired"] is False and r["chops"] == 0 and r["switched_at"] is None
    assert rf.calls[0]["sampler"] == BASE          # never switched


def test_first_loop_chops_to_onset_and_switches_to_escape():
    # round 1 loops at onset=2 -> keep [10,11], switch; round 2 closes
    rf = _scripted([([10, 11, 12, 13], "loop", 2), ([20, 21], "eos", None)])
    r = _run(rf)
    assert r["tokens"] == [10, 11, 20, 21]          # chop-only: NO nudge inserted
    assert r["fired"] is True and r["chops"] == 1
    assert r["switched_at"] == 2                     # switched after keeping 2 pre-loop tokens
    assert rf.calls[0]["sampler"] == BASE            # first round on base (alpha=2)
    assert rf.calls[1]["sampler"] == ESCAPE          # continuation on escape (alpha=5)


def test_escape_sampler_persists_across_rechops():
    # loops twice more after the switch, then closes; all post-switch rounds use ESCAPE
    rf = _scripted([([10, 11, 12], "loop", 1),
                    ([30, 31, 32], "loop", 1),
                    ([40, 41], "eos", None)])
    r = _run(rf)
    assert r["chops"] == 2 and r["fired"] is True
    assert rf.calls[0]["sampler"] == BASE
    assert rf.calls[1]["sampler"] == ESCAPE
    assert rf.calls[2]["sampler"] == ESCAPE          # stays on escape for every re-chop


def test_rechop_cap_respected():
    rf = _scripted([([9, 8, 7], "loop", 1)] * 6)     # loops forever
    r = _run(rf, max_chops=2)
    assert r["chops"] == 2                            # 2 chops applied, then kept whole
    assert r["reason"] == "loop"
    # kept = [9] (chop1) + [9] (chop2) + [9,8,7] (3rd loop kept whole)
    assert r["tokens"] == [9, 9, 9, 8, 7]


def test_budget_exhaustion_stops():
    rf = _scripted([([1, 2, 3, 4], "cap", None)] * 5)
    r = _run(rf, max_total=4)
    assert r["reason"] == "cap" and r["tokens"] == [1, 2, 3, 4]


def test_switched_at_is_none_when_only_non_loop():
    rf = _scripted([([1, 2], "cap", None)])
    r = _run(rf)
    assert r["switched_at"] is None and r["fired"] is False
