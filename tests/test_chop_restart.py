"""TDD for the chop+restart-nudge+escalate fair-test probe (scripts/chop_restart_alpha_compare.py).

Covers the model-free logic only — loop localization (find_loop), per-arm injection
(build_action_prefix), and the re-chop bookkeeping (chop_continue) with an injectable
round_fn so no model/GPU is needed. The model forward (decode_round) is exercised by the
smoke run, not here.
"""
from scripts.chop_restart_alpha_compare import (
    find_loop,
    build_action_prefix,
    chop_continue,
    PIVOT_NUDGE,
    RESTART_NUDGE,
)


# --- find_loop: locate the start of a repeated span -------------------------

def test_find_loop_returns_onset_on_repeated_chunk():
    text = "HELLO" * 10  # 5-char unit repeated 10x
    assert find_loop(text, chunk=5, min_repeat=3) == 0


def test_find_loop_returns_none_on_clean_text():
    text = "abcdefghijklmnopqrstuvwxyz0123456789" * 2  # each 5-char chunk repeats only 2x
    assert find_loop(text, chunk=5, min_repeat=3) is None


# --- build_action_prefix: per-arm injection ---------------------------------

def test_force_arm_injects_think_close_and_fence():
    out = build_action_prefix("THINK", "A_force")
    assert out.startswith("THINK")
    assert "</think>" in out
    assert out.rstrip().endswith("```python")


def test_chop_only_arm_injects_no_nudge():
    out = build_action_prefix("THINK", "chop_only")
    assert out == "THINK"                     # bare chop control — nothing appended
    assert "</think>" not in out              # keeps thinking, does NOT close


def test_chop_pivot_arm_injects_the_pivot_nudge():
    out = build_action_prefix("THINK", "chop_pivot")
    assert out == "THINK" + PIVOT_NUDGE
    assert "</think>" not in out              # keeps thinking, does NOT close
    assert "different approach" in out        # pivot = keep reasoning, redirect


def test_chop_restart_arm_injects_the_restart_nudge():
    out = build_action_prefix("THINK", "chop_restart")
    assert out == "THINK" + RESTART_NUDGE
    assert "</think>" not in out              # keeps thinking, does NOT close
    assert "from scratch" in out              # restart = discard reasoning, start over


def test_pivot_and_restart_nudges_are_distinct():
    assert PIVOT_NUDGE != RESTART_NUDGE
    # neither references the (chopped-away) looping text
    assert "going in circles" not in PIVOT_NUDGE
    assert "going in circles" not in RESTART_NUDGE


# --- chop_continue: re-chop bookkeeping with an injected round_fn ------------

def _scripted_round_fn(scripts):
    """Return a round_fn that yields the next scripted (gen, reason, onset) per call,
    recording the args it was called with."""
    calls = []
    it = iter(scripts)

    def round_fn(ctx, sampler, temp, budget):
        calls.append({"ctx": list(ctx), "sampler": sampler, "temp": temp, "budget": budget})
        return next(it)

    round_fn.calls = calls
    return round_fn


def test_chop_continue_no_loop_returns_round_output():
    rf = _scripted_round_fn([([1, 2, 3], "cap", None)])
    kept, reason, chops, events = chop_continue(
        prompt_ids=[7, 8], round_fn=rf, sampler=object(), temp=1.0,
        nudge_ids=[99], max_total=100, max_chops=3)
    assert kept == [1, 2, 3]
    assert reason == "cap"
    assert chops == 0
    assert events == []


def test_chop_continue_one_loop_then_eos_inserts_nudge_at_onset():
    rf = _scripted_round_fn([
        ([10, 11, 12, 13, 14], "loop", 2),  # chop back to onset=2 → keep [10,11], add nudge
        ([20, 21], "eos", None),
    ])
    kept, reason, chops, events = chop_continue(
        prompt_ids=[7], round_fn=rf, sampler=object(), temp=1.0,
        nudge_ids=[99], max_total=100, max_chops=3)
    assert kept == [10, 11, 99, 20, 21]
    assert reason == "eos"
    assert chops == 1
    assert len(events) == 1


def test_chop_continue_respects_max_chops_cap():
    rf = _scripted_round_fn([([10, 11, 12], "loop", 1)] * 5)  # loops forever
    kept, reason, chops, events = chop_continue(
        prompt_ids=[], round_fn=rf, sampler=object(), temp=1.0,
        nudge_ids=[99], max_total=1000, max_chops=2)
    # 2 chops applied, then the 3rd loop round is kept whole and we stop
    assert chops == 2
    assert reason == "loop"
    assert kept == [10, 99, 10, 99, 10, 11, 12]


def test_chop_continue_budget_exhaustion_stops():
    # round always returns a non-loop chunk of 4 tokens; max_total small → budget stop
    rf = _scripted_round_fn([([1, 2, 3, 4], "cap", None)] * 10)
    kept, reason, chops, events = chop_continue(
        prompt_ids=[], round_fn=rf, sampler=object(), temp=1.0,
        nudge_ids=[99], max_total=4, max_chops=3)
    assert reason == "cap"          # first round already fills budget, returns its reason
    assert kept == [1, 2, 3, 4]
