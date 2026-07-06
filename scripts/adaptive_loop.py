"""Pure, model-free control logic for the live adaptive loop-rescue decoder.

Deployment semantics: generate at the BASE sampler (pless alpha=2); the first time the
live n-gram detector fires, CHOP the current round back to the loop onset and SWITCH to
the ESCAPE sampler (pless_alpha=5) for the remainder; re-chop subsequent loops (staying on
escape) up to `max_chops`. No nudge (the chop-only rescue that the A28 findings selected).

The model forward lives entirely inside the injected `round_fn`, so this state machine is
unit-testable with a scripted round_fn (no GPU). Mirrors chop_continue but adds the
base->escape sampler switch on the first detection — that switch is the whole point.
"""


def adaptive_continue(prompt_ids, round_fn, base_sampler, base_temp,
                      escape_sampler, escape_temp, max_total, max_chops):
    """Drive one sample end-to-end.

    round_fn(ctx_ids:list, sampler, temp, budget) -> (gen:list[int], reason, onset)
      reason in {"loop","eos","cap"}; onset is the index within `gen` to chop back to.

    Returns a dict: {tokens, reason, chops, fired, switched_at, events}.
      fired      = did the detector ever fire (→ we switched to escape)
      switched_at= len(kept) at the moment of the first switch (None if never)
      events     = per-chop bookkeeping
    """
    kept, chops, fired, switched_at, events = [], 0, False, None, []
    sampler, temp = base_sampler, base_temp
    while True:
        budget = max_total - len(kept)
        if budget <= 0:
            return _result(kept, "budget", chops, fired, switched_at, events)
        gen, reason, onset = round_fn(list(prompt_ids) + kept, sampler, temp, budget)
        if reason == "loop" and chops < max_chops:
            kept = kept + gen[:onset]           # chop-only: keep pre-loop, no nudge
            chops += 1
            if not fired:                       # first detection → switch base→escape
                fired = True
                switched_at = len(kept)
            sampler, temp = escape_sampler, escape_temp
            events.append({"chop": chops, "onset": onset, "round_len": len(gen)})
            continue
        kept = kept + gen
        return _result(kept, reason, chops, fired, switched_at, events)


def _result(tokens, reason, chops, fired, switched_at, events):
    return {"tokens": tokens, "reason": reason, "chops": chops,
            "fired": fired, "switched_at": switched_at, "events": events}
