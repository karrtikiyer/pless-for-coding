"""Chop-and-regenerate probe: detect a think-phase loop, CHOP back to the loop onset
(removing the loop tokens so the model is no longer conditioned on them), inject a
steering nudge, and continue generating with an escalated sampler. Re-chop up to a cap.

This is the user's insight + Word Salad Chopper (arXiv:2511.00536): no sampler trick
at the loop position escapes because the context is loop-dominated; remove the loop
tokens and the conditioning is gone. Tests whether chop+nudge+escalate avoids re-entry.

Run: PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/chop_regen_probe.py
Env: TASK_IDS("930 1085"), N_SAMPLES(2), MAX_TOTAL(6144), MAX_CHOPS(3),
     ESCAPE("alpha"|"temp"), ALPHA(5), ESCAPE_TEMP(2.0), NGRAM_N(8), NGRAM_K(4)
"""
import json
import os
import torch

from bench.generator import load_model_and_tokenizer
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_guarded_pless_sampler, make_pless_alpha_sampler
from scripts.repeat_detector import RepeatDetector

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
TASK_IDS = [int(x) for x in os.environ.get("TASK_IDS", "930 1085").split()]
N = int(os.environ.get("N_SAMPLES", "2"))
MAX_TOTAL = int(os.environ.get("MAX_TOTAL", "6144"))
MAX_CHOPS = int(os.environ.get("MAX_CHOPS", "3"))
ESCAPE = os.environ.get("ESCAPE", "alpha")            # "alpha" → pless_alpha; "temp" → temp-before-pless
ALPHA = float(os.environ.get("ALPHA", "5"))
ESCAPE_TEMP = float(os.environ.get("ESCAPE_TEMP", "2.0"))
NGRAM_N = int(os.environ.get("NGRAM_N", "8"))
NGRAM_K = int(os.environ.get("NGRAM_K", "4"))
OUT = os.environ.get("OUT", "")   # if set, save per-sample results to this JSON incrementally
NUDGE = "\n\nWait, I'm going in circles. Let me step back and write the solution directly.\n"


def decode_round(model, input_ids, sampler, temperature, max_new, eos_id, think_end_id, detect):
    """Single-sequence decode; stop early on loop detection (think phase only)."""
    det = RepeatDetector(n=NGRAM_N, k=NGRAM_K) if detect else None
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[0, -1].float()
    gen, in_code = [], False
    for _ in range(max_new):
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
            out = model(input_ids=torch.tensor([[tid]], device=model.device), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1].float()
    return gen, "cap", None


def chop_and_generate(model, tok, prompt, eos_id, think_end_id, escape_sampler, escape_temp):
    prompt_ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    nudge_ids = tok.encode(NUDGE, add_special_tokens=False)
    base = make_guarded_pless_sampler()
    kept: list[int] = []
    sampler, temp, chops, events = base, 1.0, 0, []
    while True:
        budget = MAX_TOTAL - len(kept)
        if budget <= 0:
            reason = "budget"; break
        ctx = (torch.cat([prompt_ids, torch.tensor([kept], device=model.device)], dim=1)
               if kept else prompt_ids)
        gen, reason, onset = decode_round(model, ctx, sampler, temp, budget, eos_id, think_end_id, detect=True)
        if reason == "loop" and chops < MAX_CHOPS:
            kept += gen[:onset] + nudge_ids           # keep pre-loop reasoning + nudge
            sampler, temp = escape_sampler, escape_temp
            chops += 1
            events.append({"chop": chops, "onset_in_round": onset, "round_len": len(gen)})
            continue
        kept += gen
        break
    return tok.decode(kept, skip_special_tokens=False), reason, chops, events, len(kept)


def main():
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tok = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    think_end_id = tok.convert_tokens_to_ids("</think>")
    if ESCAPE == "alpha":
        escape_sampler, escape_temp, tag = make_pless_alpha_sampler(ALPHA), 1.0, f"chop+α{int(ALPHA)}"
    else:
        escape_sampler, escape_temp, tag = make_guarded_pless_sampler(), ESCAPE_TEMP, f"chop+T{ESCAPE_TEMP}"

    results = []
    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    for tid in TASK_IDS:
        problem = pmap[tid]
        prompt, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        print(f"\n=== task {tid} | {tag} | nudge | max_chops={MAX_CHOPS} cap={MAX_TOTAL} | "
              f"detect {NGRAM_N}-gram×{NGRAM_K} ===", flush=True)
        for r in range(N):
            try:
                text, reason, chops, events, ntok = chop_and_generate(
                    model, tok, prompt, eos_id, think_end_id, escape_sampler, escape_temp)
                closed = "</think>" in text
                status, code = "no_code", ""
                if closed:
                    code = text.split("</think>", 1)[1]
                    res, _ = evaluate_apps_sample(code if "```" in code else "```python\n" + code, problem)
                    status = res.status
            except Exception as e:                       # don't let one sample kill a multi-hour run
                reason, chops, events, ntok, closed, status, code = f"EXC:{type(e).__name__}", -1, [], 0, False, "exc", ""
            rec = {"task_id": tid, "sample": r, "chops": chops, "end": reason,
                   "gen_tokens": ntok, "closed_think": closed, "exec": status,
                   "recovered": status == "Passed", "chop_events": events,
                   "code": code}    # save extracted code (post-</think>) for solution-diversity comparison
            results.append(rec)
            print(f"  sample {r}: chops={chops} end={reason} gen={ntok} tok | "
                  f"</think>={closed} | exec={status} | chop_events={events}", flush=True)
            if OUT:
                with open(OUT, "w") as fh:
                    json.dump(results, fh, indent=2)

    # summary
    n_rec = sum(1 for x in results if x["recovered"])
    n_closed = sum(1 for x in results if x["closed_think"])
    from collections import Counter
    chop_hist = Counter(x["chops"] for x in results)
    print("\n" + "=" * 60)
    print(f"CHOP-AND-REGEN over {len(results)} samples ({len(TASK_IDS)} tasks × n={N}):")
    print(f"  recovered (exec=Passed): {n_rec}/{len(results)}")
    print(f"  closed </think>:         {n_closed}/{len(results)}")
    print(f"  chop-count histogram:    {dict(sorted(chop_hist.items()))}")
    if OUT:
        print(f"  results -> {OUT}")


if __name__ == "__main__":
    main()
