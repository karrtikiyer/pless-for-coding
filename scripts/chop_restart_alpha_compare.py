"""The parked FAIR TEST: chop a REAL truncated trace at its loop onset, then compare
post-detection ACTIONS on identical input (only the action differs):

  A_force     : inject </think> + code fence → write the solution NOW (extract existing).
  chop_only   : continue THINKING at pless_alpha(ALPHA) with NO nudge — isolates whether
                removing the loop + alpha-diversity alone recovers (the lit says the CHOP
                is the load-bearing part; this is the control for both nudge arms).
  chop_pivot  : inject a PIVOT nudge ("step back, try a different approach") → continue
                THINKING at pless_alpha(ALPHA). Bets the kept reasoning is an asset.
  chop_restart: inject a RESTART nudge ("discard it, reconsider from scratch") → continue
                THINKING at pless_alpha(ALPHA). Bets the kept reasoning is a liability.
  All three chop arms re-detect loops live + re-chop (cap MAX_CHOPS).

Comparison ladder: A_force vs (any chop arm) = does continuing-to-think recover what
force-extract can't?  chop_only vs chop_pivot/chop_restart = does a nudge add anything
over the bare chop?  chop_pivot vs chop_restart = keep-and-redirect vs discard-and-restart.

Why real traces (not fresh regen): chop_regen_probe.py regenerated a fresh trace from the
bare prompt with pless@a2 (which mostly never reaches a solution), so its 0/17 conflated
"action failed" with "trace never had a solution" → that comparison was retracted 2026-06-13.
Seeding from the REAL saved trace (known to loop; for recoverable tasks, known to contain a
solution) isolates the ACTION. chop_action_compare.py had the right input but continued at
alpha=2 and only ever nudged "write code now"; this script adds the alpha=5 continuation and
the restart-thinking nudge — the combination that was never run.

Honest prior (2506.10979, EMNLP 2025 — abstract re-fetched 2026-06-24): models reliably
IDENTIFY unhelpful thoughts but FAIL TO RECOVER when the bad thought is INJECTED AND LEFT IN
CONTEXT — they "naively continue the line of reasoning"; instructing them to reevaluate does
not fix it (inverse scaling: larger models worse). Crucially that setup KEEPS the bad thought;
our chop REMOVES it, so the paper most directly predicts the no-chop/nudge-only case is weak —
it does NOT foreclose chop+nudge. That is exactly why chop_only is a required control.
The paper is math/QA-style reasoning (no confirmed code/MBPP eval in the abstract).
Standing conclusion is still prevention (alpha=5 from start, A31) >> all rescue; this asks
whether detect→chop→continue recovers tasks force-</think> misses, and whether a nudge helps.

Context guard: total context (chopped prefix tokens + continuation) is capped at MAX_CTX so a
deep-chop task cannot overflow Qwen3-8B's 32768-token window (the MPS run OOMed there).

Run (smoke):  HF_HUB_OFFLINE=1 PYTHONPATH=. TASK_IDS="1226 1126" N=1 \
              uv run python scripts/chop_restart_alpha_compare.py
Run (Phase-1): see run_chop_restart_apps_qwen3.sh (14 anchored tasks, N=4, MAX_CONT=16384).
Env: TASK_IDS("" = all 40 pless truncated tasks), N(1), MAX_CONT(16384), MAX_CHOPS(3),
     ALPHA(5), ESCAPE("alpha"|"temp"), ESCAPE_TEMP(2.0), NGRAM_N(30), NGRAM_K(6),
     NGRAM_WINDOW(1200), CONFIG("pless_think_t1.0_t1.0"), MAX_CTX(32768), OUT(json path).
"""
import gc
import json
import os

import torch


def _free():
    """Release MPS/CUDA cached memory between arms/samples (long prefills are heavy)."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

from bench.generator import load_model_and_tokenizer
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_guarded_pless_sampler, make_pless_alpha_sampler
from scripts.repeat_detector import RepeatDetector

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"

# Two nudge bets, both phrased to be COHERENT with the chopped context (the chop already
# removed the looping tokens, so neither references "going in circles" — that text is gone;
# both refer only to the surviving pre-loop reasoning the model can still see):
#   PIVOT   — keep the reasoning, redirect it ("step back, try a different approach").
#   RESTART — discard the reasoning, start over ("from scratch"). The user's original framing.
PIVOT_NUDGE = (
    "\n\nWait — this line of reasoning has stalled. Let me step back and try a "
    "fundamentally different approach.\n"
)
RESTART_NUDGE = (
    "\n\nThis approach isn't working. Let me discard it and reconsider the problem "
    "from scratch with a different strategy.\n"
)
FORCE_INJECT = "\n</think>\n\n```python\n"

# Arm → nudge string. chop_only gets no nudge (the bare-chop control).
NUDGES = {"chop_only": "", "chop_pivot": PIVOT_NUDGE, "chop_restart": RESTART_NUDGE}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_chop_restart.py)                      #
# --------------------------------------------------------------------------- #
def find_loop(text, chunk=120, min_repeat=4):
    """Return the char index where a >= chunk-long span first repeats >= min_repeat
    times, else None. Same heuristic as chop_action_compare.py (so A_force and the
    restart arms chop at the IDENTICAL onset → fair comparison)."""
    for s in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[s:].count(text[s:s + chunk]) >= min_repeat:
            return s
    return None


def build_action_prefix(chopped_text, arm):
    """Append the per-arm injection to the chopped (pre-loop) trace text.

    A_force closes </think> + opens a code fence. The chop arms continue thinking, each
    with its own nudge (chop_only = none, chop_pivot = PIVOT, chop_restart = RESTART).
    """
    if arm == "A_force":
        return chopped_text + FORCE_INJECT
    return chopped_text + NUDGES[arm]


def chop_continue(prompt_ids, round_fn, sampler, temp, nudge_ids, max_total, max_chops):
    """Drive a continuation that re-chops on re-detected loops.

    round_fn(ctx_ids:list, sampler, temp, budget) -> (gen:list[int], reason, onset)
      reason in {"loop","eos","cap"}; onset is the index in `gen` to chop back to.
    On reason=="loop" (while under max_chops): keep gen[:onset] + nudge_ids and re-enter.
    Otherwise append gen and stop. Returns (kept:list[int], reason, chops, events).
    The model forward is entirely inside round_fn, so this logic is model-free/testable.
    """
    kept, chops, events = [], 0, []
    while True:
        budget = max_total - len(kept)
        if budget <= 0:
            return kept, "budget", chops, events
        ctx = list(prompt_ids) + kept
        gen, reason, onset = round_fn(ctx, sampler, temp, budget)
        if reason == "loop" and chops < max_chops:
            kept = kept + gen[:onset] + list(nudge_ids)
            chops += 1
            events.append({"chop": chops, "onset": onset, "round_len": len(gen)})
            continue
        kept = kept + gen
        return kept, reason, chops, events


# --------------------------------------------------------------------------- #
# Model-bound pieces                                                            #
# --------------------------------------------------------------------------- #
def make_safe(sampler):
    """Wrap a sampler to survive MPS bf16 NaN/inf on long-context forwards
    (no-op on clean rows; would not occur on CUDA). Mirrors chop_action_compare.py."""
    def s(probs):
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        bad = (probs.sum(-1, keepdim=True) <= 0).squeeze(-1)
        if bad.any():
            probs[bad] = 1.0
        return sampler(probs)
    return s


def decode_round(model, input_ids, sampler, temperature, max_new, eos_id, think_end_id,
                 n, k, window, detect=True):
    """Single-sequence token-by-token decode; stop early on loop detection (think phase
    only) when detect=True. Returns (gen:list[int], reason, onset_in_gen)."""
    det = RepeatDetector(n=n, k=k, window=window) if detect else None
    with torch.no_grad():
        # logits_to_keep=1: only materialize last-position logits. Without it the full
        # prefill logits (seq_len x vocab ~ GBs at 8k tokens) OOMs MPS. Mirrors HF .generate().
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    past, logits = out.past_key_values, out.logits[0, -1].float()
    del out
    gen, in_code = [], False
    for step in range(max_new):
        lg = logits / temperature if temperature != 1.0 else logits
        probs = torch.softmax(lg, dim=-1).unsqueeze(0)
        tid = int(sampler(probs.clone()).view(1))
        gen.append(tid)
        if tid == eos_id:
            return gen, "eos", None
        if tid == think_end_id:
            in_code = True
        if det is not None and not in_code and det.update(tid):
            return gen, "loop", det.onset
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[tid]], device=model.device),
                        past_key_values=past, use_cache=True, logits_to_keep=1)
        past, logits = out.past_key_values, out.logits[0, -1].float()
        del out
        # MPS does not return freed step buffers to the OS; without this the reserved
        # pool grows unbounded over a 2k-step decode (observed 51 GiB → OOM). Bound it.
        if step % 128 == 127 and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return gen, "cap", None


def get_trace(tid, config):
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == config and r["task_id"] == tid:
                return r["truncated_solution"]
    return None


def extract_status(text, arm, problem):
    """Execute the arm's output. A_force: text is code after the fence. The chop arms
    (chop_only/chop_pivot/chop_restart): text is the continuation (re-reasoning + </think>
    + code), so the code lives after the first </think> the model emits."""
    if arm == "A_force":
        sample = "```python\n" + text
    else:
        if "</think>" not in text:
            return "no_</think>", ""
        code = text.split("</think>", 1)[1]
        sample = code if "```" in code else "```python\n" + code
    res, _ = evaluate_apps_sample(sample, problem)
    return res.status, sample


def run_force(model, tok, full_text, problem, n, max_cont, sampler, eos_id, think_end_id):
    """A_force arm: generate code after the injected </think>+fence. Uses the same manual,
    memory-efficient (logits_to_keep=1) decode as the restart arms — HF generate_samples
    OOMs MPS on these long (~5-20k-token) prefills."""
    recs = []
    for i in range(n):
        _free()
        ids = torch.tensor([tok.encode(full_text)], device=model.device)
        gen, reason, _ = decode_round(model, ids, sampler, 1.0, max_cont, eos_id,
                                      think_end_id, 0, 0, 0, detect=False)
        text = tok.decode(gen, skip_special_tokens=False)
        st, _ = extract_status(text, "A_force", problem)
        recs.append({"sample": i, "end": "force", "chops": 0, "closed_think": True,
                     "exec": st, "recovered": st == "Passed", "gen_tokens": len(gen)})
    return recs


def run_restart(model, tok, arm, full_text, nudge, problem, n, max_cont, max_chops,
                cont_sampler, cont_temp, eos_id, think_end_id, n_g, k_g, w_g):
    """Chop arms: continue THINKING from (chopped trace [+ nudge]) with live re-chop.
    `nudge` is the per-arm steering string ("" for chop_only); the SAME nudge is re-injected
    on each re-chop, so a re-detected loop is treated identically to the first."""
    nudge_ids = tok.encode(nudge, add_special_tokens=False) if nudge else []

    def round_fn(ctx, sampler, temp, budget):
        ids = torch.tensor([ctx], device=model.device)
        return decode_round(model, ids, sampler, temp, budget, eos_id, think_end_id,
                            n_g, k_g, w_g)

    recs = []
    for i in range(n):
        _free()
        prompt_ids = tok.encode(full_text)
        kept, reason, chops, events = chop_continue(
            prompt_ids, round_fn, cont_sampler, cont_temp, nudge_ids, max_cont, max_chops)
        text = tok.decode(kept, skip_special_tokens=False)
        st, _ = extract_status(text, arm, problem)
        recs.append({"sample": i, "end": reason, "chops": chops,
                     "closed_think": "</think>" in text, "exec": st,
                     "recovered": st == "Passed", "gen_tokens": len(kept),
                     "chop_events": events})
    return recs


def main():
    config = os.environ.get("CONFIG", "pless_think_t1.0_t1.0")
    task_env = os.environ.get("TASK_IDS", "").strip()
    n = int(os.environ.get("N", "1"))
    max_cont = int(os.environ.get("MAX_CONT", "16384"))
    max_chops = int(os.environ.get("MAX_CHOPS", "3"))
    alpha = float(os.environ.get("ALPHA", "5"))
    escape = os.environ.get("ESCAPE", "alpha")
    escape_temp = float(os.environ.get("ESCAPE_TEMP", "2.0"))
    n_g = int(os.environ.get("NGRAM_N", "30"))
    k_g = int(os.environ.get("NGRAM_K", "6"))
    w_g = int(os.environ.get("NGRAM_WINDOW", "1200"))
    max_ctx = int(os.environ.get("MAX_CTX", "32768"))   # Qwen3-8B native window
    out = os.environ.get("OUT", "")

    if task_env:
        task_ids = [int(x) for x in task_env.split()]
    else:
        seen, task_ids = set(), []
        with open(f"{POD}/truncated_cases.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["config"] == config and r["task_id"] not in seen:
                    seen.add(r["task_id"]); task_ids.append(r["task_id"])

    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tok = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    think_end_id = tok.convert_tokens_to_ids("</think>")

    # A_force generates code with the published pless (alpha=2); the three chop arms all
    # continue thinking with the SAME escape sampler (alpha=ALPHA, default 5) — the only
    # thing that differs across the chop arms is the nudge string.
    safe_pless = make_safe(make_guarded_pless_sampler())
    if escape == "alpha":
        c_sampler, c_temp, c_tag = make_safe(make_pless_alpha_sampler(alpha)), 1.0, f"a{int(alpha)}"
    else:
        c_sampler, c_temp, c_tag = make_safe(make_guarded_pless_sampler()), escape_temp, f"T{escape_temp}"

    print(f"config={config} | tasks={len(task_ids)} | n={n} | cont={c_tag} | "
          f"arms=A_force,chop_only,chop_pivot,chop_restart | detect {n_g}-gram x{k_g}/w{w_g} | "
          f"max_chops={max_chops} cap={max_cont} max_ctx={max_ctx}", flush=True)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    results, skipped = [], []
    for tid in task_ids:
        if tid not in pmap:
            skipped.append((tid, "not_in_pmap")); continue
        problem = pmap[tid]
        trace = get_trace(tid, config)
        if trace is None:
            skipped.append((tid, "no_trace")); continue
        cut = find_loop(trace)
        if cut is None:
            skipped.append((tid, "no_loop_located")); continue
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        chopped = prefix + trace[:cut]

        # Context guard: keep (chopped prefix tokens + continuation) under the model window.
        # Deep-chop tasks (e.g. 326/1175/739, ~16k-token cut) would otherwise overflow 32768
        # and OOM/error. We clamp the per-task continuation budget rather than skip, so the
        # task still contributes whatever room remains. -512 leaves headroom for the nudge.
        base_len = len(tok.encode(chopped))
        eff_cont = min(max_cont, max_ctx - base_len - 512)
        if eff_cont < 512:
            skipped.append((tid, f"prefix_too_long({base_len}tok)")); continue
        clamp = "" if eff_cont == max_cont else f" [cont clamped {max_cont}->{eff_cont}]"
        print(f"\n=== task {tid}: cut@{cut} ({cut/len(trace):.0%} of {len(trace)} chars), "
              f"prefix={base_len}tok, {len(problem.inputs)} tests{clamp} ===", flush=True)

        def _force():
            return run_force(model, tok, build_action_prefix(chopped, "A_force"), problem,
                             n, eff_cont, safe_pless, eos_id, think_end_id)

        def _chop(arm):
            return run_restart(model, tok, arm, build_action_prefix(chopped, arm), NUDGES[arm],
                               problem, n, eff_cont, max_chops, c_sampler, c_temp,
                               eos_id, think_end_id, n_g, k_g, w_g)

        arms = {
            "A_force":     _force,
            "chop_only":   lambda: _chop("chop_only"),
            "chop_pivot":  lambda: _chop("chop_pivot"),
            "chop_restart": lambda: _chop("chop_restart"),
        }
        for arm, fn in arms.items():
            _free()
            try:
                recs = fn()
            except Exception as e:                       # one sample must not kill the run
                msg = str(e).replace("\n", " ")[:240]
                print(f"  {arm:<10} EXC {type(e).__name__}: {msg}", flush=True)
                recs = [{"sample": i, "end": f"EXC:{type(e).__name__}", "chops": -1,
                         "closed_think": False, "exec": "exc", "recovered": False,
                         "exc": msg} for i in range(n)]
            for rec in recs:
                rec.update({"task_id": tid, "arm": arm, "cut": cut})
                results.append(rec)
                print(f"  {arm:<10} s{rec['sample']}: end={rec['end']:<10} "
                      f"chops={rec['chops']} </think>={rec['closed_think']} "
                      f"exec={rec['exec']}{'  PASS' if rec['recovered'] else ''}", flush=True)
            if out:
                with open(out, "w") as fh:
                    json.dump({"meta": {"config": config, "n": n, "cont": c_tag,
                                        "skipped": skipped}, "results": results}, fh, indent=2)

    # --- summary: totals + partition by whether A_force recovered ---
    from collections import defaultdict
    by_arm = defaultdict(lambda: [0, 0])
    rec_by_task_arm = defaultdict(dict)
    for r in results:
        by_arm[r["arm"]][0] += r["recovered"]; by_arm[r["arm"]][1] += 1
        rec_by_task_arm[r["task_id"]].setdefault(r["arm"], []).append(r["recovered"])

    ARMS = ("A_force", "chop_only", "chop_pivot", "chop_restart")
    print("\n" + "=" * 64)
    print(f"FAIR TEST — {len(task_ids)-len(skipped)} tasks run, {len(skipped)} skipped {skipped}")
    print("recovered (exec==Passed) by arm  [pass-samples / total-samples | tasks-recovered]:")

    def task_rec(arm, tid):  # task recovered if any sample passed
        return any(rec_by_task_arm.get(tid, {}).get(arm, [False]))
    for arm in ARMS:
        p, t = by_arm[arm]
        ntask = sum(1 for tid in rec_by_task_arm if task_rec(arm, tid))
        print(f"  {arm:<13}: {p}/{t} | {ntask} tasks")

    # Comparison ladder (pre-registered): does any chop arm recover what A_force misses?
    print("\n  --- vs A_force (the only place a chop arm can WIN) ---")
    for arm in ("chop_only", "chop_pivot", "chop_restart"):
        win = [tid for tid in rec_by_task_arm
               if not task_rec("A_force", tid) and task_rec(arm, tid)]
        cost = [tid for tid in rec_by_task_arm
                if task_rec("A_force", tid) and not task_rec(arm, tid)]
        print(f"  {arm:<13}: A_force FAILS but {arm} RECOVERS (win): {win}")
        print(f"  {'':<13}  A_force recovers but {arm} fails (cost): {cost}")

    # Does a nudge add anything over the bare chop?
    print("\n  --- nudge effect (chop_pivot / chop_restart vs chop_only) ---")
    for arm in ("chop_pivot", "chop_restart"):
        gain = [tid for tid in rec_by_task_arm
                if task_rec(arm, tid) and not task_rec("chop_only", tid)]
        loss = [tid for tid in rec_by_task_arm
                if task_rec("chop_only", tid) and not task_rec(arm, tid)]
        print(f"  {arm:<13}: recovers where chop_only fails: {gain} | loses where chop_only wins: {loss}")
    if out:
        print(f"\n  results -> {out}")


if __name__ == "__main__":
    main()
