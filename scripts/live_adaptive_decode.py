"""Live adaptive loop-rescue decoder (HF token-by-token) — the deployable pipeline.

Generate from the BARE prompt at pless alpha=BASE_ALPHA (default 2). A live n-gram detector
runs on the think phase; the first time it fires, chop back to the loop onset and SWITCH to
pless_alpha=ESC_ALPHA (default 5) for the remainder, re-chopping subsequent loops (staying on
escape) up to MAX_CHOPS. No nudge (the chop-only rescue the A28 fair test selected).

This is the FRESH-generation deployment measurement: because per-sample looping is stochastic
(Qwen 43% / DeepSeek 77% of tasks have mixed loop rates), we cannot reuse saved traces — we
generate live and rescue whatever actually loops. Each sample also yields its plain-alpha=2
counterfactual for free: a NON-fired sample == plain alpha=2; a FIRED sample would have looped
to the cap under plain alpha=2 → baseline fail. So baseline_recovered = recovered AND NOT fired.

This SEQUENTIAL driver reuses the GPU-validated decode_round for correctness (smoke/method
validation). Batching is the throughput optimization added once the method is confirmed.

Per-model detector config (set via env by the launch script):
  Qwen3-8B            : NGRAM_N=30 NGRAM_K=6  NGRAM_WINDOW=1600
  DeepSeek-R1-Distill : NGRAM_N=30 NGRAM_K=8  NGRAM_WINDOW=3000  (candidate 40/4000/8)

Run (smoke): HF_HUB_OFFLINE=1 PYTHONPATH=. MODEL=Qwen/Qwen3-8B MAX_PROBLEMS=4 N=4 \
             NGRAM_N=30 NGRAM_K=6 NGRAM_WINDOW=1600 \
             OUT=results/_live_adaptive/qwen_smoke.json uv run python scripts/live_adaptive_decode.py
Env: MODEL, SOURCE(ATCODER), DIFFICULTY(interview), TASK_IDS(""=all), MAX_PROBLEMS(0=all),
     N(10), MAX_NEW(32768), MAX_CHOPS(3), BASE_ALPHA(2), ESC_ALPHA(5),
     NGRAM_N(30), NGRAM_K(6), NGRAM_WINDOW(1600), MAX_CTX(32768), OUT.
"""
import gc
import json
import os

import torch

from bench.generator import (load_model_and_tokenizer, _expand_past_key_values)
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_pless_alpha_sampler
from scripts.chop_restart_alpha_compare import decode_round, make_safe
from scripts.adaptive_loop import adaptive_continue
from scripts.repeat_detector import RepeatDetector
from scripts.batched_gen import batched_gen_round, batched_phase2


def _free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_think_end(tok):
    """Model-aware </think> id (Qwen3 151668, DeepSeek 128014). Replicated from
    bench.generator_vllm.resolve_think_end_id to avoid importing the vLLM module."""
    ids = tok.encode("</think>", add_special_tokens=False)
    assert len(ids) == 1, f"</think> not single-token: {ids}"
    return ids[0]


def extract_and_eval(text, problem):
    """The generation is a full bare-prompt trace: code lives after the first </think>."""
    if "</think>" not in text:
        return "no_</think>", False
    code = text.split("</think>", 1)[1]
    sample = code if "```" in code else "```python\n" + code
    res, _ = evaluate_apps_sample(sample, problem)
    return res.status, res.status == "Passed"


def batched_phase1(model, prompt_ids, n, sampler, temp, max_new, eos_id, think_end_id,
                   n_g, k_g, w_g, label="", log_every=512):
    """Phase 1: batched alpha=2 generation of n samples from the shared prompt, with per-row
    live n-gram detection during the think phase. Mirrors bench.generator.generate_samples
    (prefill once -> expand KV -> batched decode) but stops a row on the FIRST loop and
    records its onset. Rows that close </think> keep generating code (detection off) until
    eos/cap. Returns per row: {gen, reason in {eos,loop,cap}, onset, closed_think}.
    NOTE: model loop validated by GPU smoke, not unit tests (batched forward needs a model).
    """
    dev = model.device
    input_ids = torch.tensor([prompt_ids], device=dev)
    with torch.no_grad():
        pf = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    past = _expand_past_key_values(pf.past_key_values, n)
    logits = pf.logits[0, -1].float().unsqueeze(0).expand(n, -1).contiguous()

    dets = [RepeatDetector(n=n_g, k=k_g, window=w_g) for _ in range(n)]
    in_code = [False] * n
    reason = [None] * n
    onset = [None] * n
    gens = [[] for _ in range(n)]
    finished = torch.zeros(n, dtype=torch.bool, device=dev)

    for step in range(max_new):
        lg = logits / temp if temp != 1.0 else logits
        probs = torch.softmax(lg, dim=-1)
        nxt = sampler(probs.clone()).view(n)
        nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
        for i in range(n):
            if bool(finished[i]):
                continue
            tid = int(nxt[i])
            gens[i].append(tid)
            if tid == eos_id:
                finished[i] = True; reason[i] = "eos"
            elif tid == think_end_id:
                in_code[i] = True
            elif not in_code[i] and dets[i].update(tid):
                finished[i] = True; reason[i] = "loop"; onset[i] = dets[i].onset
        if log_every and step % log_every == 0:
            print(f"    {label} phase1 step {step}/{max_new} finished={int(finished.sum())}/{n}",
                  flush=True)
        if bool(finished.all()) or step == max_new - 1:
            break
        with torch.no_grad():
            out = model(input_ids=nxt.view(n, 1), past_key_values=past, use_cache=True,
                        logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
        if step % 128 == 127 and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    for i in range(n):
        if reason[i] is None:
            reason[i] = "cap"
    return [{"gen": gens[i], "reason": reason[i], "onset": onset[i],
             "closed_think": in_code[i]} for i in range(n)]


def run_task_batched(model, tok, problem, prompt_ids, n, base_s, esc_s, max_new, max_chops,
                     eos_id, think_end_id, n_g, k_g, w_g):
    """Batched Phase 1 (alpha=2 + detect) then SEQUENTIAL Phase 2 (alpha=5 chop-continue with
    re-chop) on the fired rows only. Phase 2 reuses the tested adaptive_continue + decode_round.
    """
    p1 = batched_phase1(model, prompt_ids, n, base_s, 1.0, max_new, eos_id, think_end_id,
                        n_g, k_g, w_g)

    def round_fn(ctx, sampler, temp, budget):
        ids = torch.tensor([ctx], device=model.device)
        return decode_round(model, ids, sampler, temp, budget, eos_id, think_end_id,
                            n_g, k_g, w_g)

    recs = []
    for i, r in enumerate(p1):
        if r["reason"] != "loop":                      # healthy / cap: no rescue
            toks, fired, chops, reason = r["gen"], False, 0, r["reason"]
        else:                                          # fired: chop -> continue at alpha=5
            _free()
            chopped = list(prompt_ids) + r["gen"][:r["onset"]]
            budget = max(0, max_new - r["onset"])
            out2 = adaptive_continue(chopped, round_fn, esc_s, 1.0, esc_s, 1.0,
                                     budget, max_chops - 1)
            toks = r["gen"][:r["onset"]] + out2["tokens"]
            fired, chops, reason = True, 1 + out2["chops"], out2["reason"]
        text = tok.decode(toks, skip_special_tokens=False)
        st, ok = extract_and_eval(text, problem)
        recs.append({"sample": i, "fired": fired, "chops": chops, "reason": reason,
                     "closed_think": "</think>" in text, "exec": st, "recovered": ok,
                     "baseline_recovered": ok and not fired, "gen_tokens": len(toks)})
    return recs


def run_task_fullbatch(model, tok, problem, prompt_ids, n, base_s, esc_s, max_new, max_chops,
                       eos_id, think_end_id, n_g, k_g, w_g, pad_id, round_cap,
                       phase2_batch=4, tid="?"):
    """Fully batched: batched Phase-1 (alpha=2 + detect) then BATCHED Phase-2 (alpha=5
    chop-continue) over this task's fired rows together, via batched_phase2 +
    batched_gen_round. Turns DeepSeek's ~6-7 sequential per-task continuations into batched
    rounds. Falls back to no Phase-2 work when nothing fired."""
    print(f"[task {tid}] phase1 start: n={n} prompt={len(prompt_ids)}tok cap={max_new}", flush=True)
    p1 = batched_phase1(model, prompt_ids, n, base_s, 1.0, max_new, eos_id, think_end_id,
                        n_g, k_g, w_g, label=f"[task {tid}]")
    recs = [None] * n
    fired = []
    for i, r in enumerate(p1):
        if r["reason"] != "loop":
            recs[i] = {"gen": r["gen"], "fired": False, "chops": 0, "reason": r["reason"]}
        else:
            fired.append({"idx": i, "pre": r["gen"][:r["onset"]],
                          "prefix": list(prompt_ids) + r["gen"][:r["onset"]],
                          "budget": max(1, max_new - r["onset"])})
    nchunk = (len(fired) + phase2_batch - 1) // phase2_batch if fired else 0
    print(f"[task {tid}] phase1 done: fired {len(fired)}/{n}"
          + (f" -> phase2 ({nchunk} chunk(s) of <={phase2_batch})" if fired
             else " (no rescue needed)"), flush=True)
    if fired:
        def round_fn(prefixes, mn):
            return batched_gen_round(model, prefixes, esc_s, 1.0, mn, eos_id, think_end_id,
                                     n_g, k_g, w_g, pad_id, label=f"[task {tid}] p2")

        # Cap Phase-2 batch width: B rows each carry a long context (chopped prefix +
        # continuation, up to ~32k) → KV cache ~ B x context. All 10 at once OOMs an 80GB
        # card; chunks of ~4 keep it bounded. Still batched (4x fewer forwards than sequential).
        for s in range(0, len(fired), phase2_batch):
            _free()
            batched_phase2(fired[s:s + phase2_batch], round_fn, max_chops - 1, round_cap)
        for f in fired:
            recs[f["idx"]] = {"gen": f["pre"] + f["cont"], "fired": True,
                              "chops": 1 + f["chops"], "reason": f["reason"]}

    out = []
    for i in range(n):
        rr = recs[i]
        text = tok.decode(rr["gen"], skip_special_tokens=False)
        st, ok = extract_and_eval(text, problem)
        out.append({"sample": i, "fired": rr["fired"], "chops": rr["chops"],
                    "reason": rr["reason"], "closed_think": "</think>" in text, "exec": st,
                    "recovered": ok, "baseline_recovered": ok and not rr["fired"],
                    "gen_tokens": len(rr["gen"])})
    return out


def run_task(model, tok, problem, prompt_ids, n, base_s, esc_s, max_new, max_chops,
             eos_id, think_end_id, n_g, k_g, w_g):
    def round_fn(ctx, sampler, temp, budget):
        ids = torch.tensor([ctx], device=model.device)
        return decode_round(model, ids, sampler, temp, budget, eos_id, think_end_id,
                            n_g, k_g, w_g)

    recs = []
    for i in range(n):
        _free()
        out = adaptive_continue(prompt_ids, round_fn, base_s, 1.0, esc_s, 1.0,
                                max_new, max_chops)
        text = tok.decode(out["tokens"], skip_special_tokens=False)
        st, ok = extract_and_eval(text, problem)
        recs.append({"sample": i, "fired": out["fired"], "chops": out["chops"],
                     "switched_at": out["switched_at"], "reason": out["reason"],
                     "closed_think": "</think>" in text, "exec": st,
                     "recovered": ok, "baseline_recovered": ok and not out["fired"],
                     "gen_tokens": len(out["tokens"])})
    return recs


def main():
    model_id = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    source = os.environ.get("SOURCE", "ATCODER")
    difficulty = os.environ.get("DIFFICULTY", "interview")
    task_env = os.environ.get("TASK_IDS", "").strip()
    max_problems = int(os.environ.get("MAX_PROBLEMS", "0"))
    n = int(os.environ.get("N", "10"))
    max_new = int(os.environ.get("MAX_NEW", "32768"))
    max_chops = int(os.environ.get("MAX_CHOPS", "3"))
    base_alpha = float(os.environ.get("BASE_ALPHA", "2"))
    esc_alpha = float(os.environ.get("ESC_ALPHA", "5"))
    n_g = int(os.environ.get("NGRAM_N", "30"))
    k_g = int(os.environ.get("NGRAM_K", "6"))
    w_g = int(os.environ.get("NGRAM_WINDOW", "1600"))
    max_ctx = int(os.environ.get("MAX_CTX", "32768"))
    batched = os.environ.get("BATCHED", "1") == "1"   # 1 = fully batched (default); 0 = sequential
    round_cap = int(os.environ.get("PHASE2_CAP", "16384"))  # per Phase-2 batched round budget
    phase2_batch = int(os.environ.get("PHASE2_BATCH", "4"))  # max rows per Phase-2 batch (KV cap)
    out = os.environ.get("OUT", "")

    pmap = load_apps_test_map(source=source, difficulty=difficulty)
    task_ids = ([int(x) for x in task_env.split()] if task_env else sorted(pmap))
    if max_problems:
        task_ids = task_ids[:max_problems]

    model, tok = load_model_and_tokenizer(model_id, dtype="bfloat16")
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    think_end_id = resolve_think_end(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    base_s = make_safe(make_pless_alpha_sampler(base_alpha))
    esc_s = make_safe(make_pless_alpha_sampler(esc_alpha))

    print(f"model={model_id} | {source}/{difficulty} | tasks={len(task_ids)} n={n} | "
          f"base=a{int(base_alpha)}->esc=a{int(esc_alpha)} | detect {n_g}-gram k{k_g}/w{w_g} | "
          f"max_new={max_new} max_chops={max_chops} think_end={think_end_id} pad={pad_id} | "
          f"{'FULLY BATCHED (p2 cap ' + str(round_cap) + ')' if batched else 'sequential'}", flush=True)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Resume: if OUT exists, keep its completed tasks and skip them (a restart after a crash
    # must not overwrite prior work — the incremental write uses mode "w").
    results, skipped = [], []
    done_tasks = set()
    if out and os.path.exists(out):
        try:
            prev = json.load(open(out))
            results = prev.get("results", [])
            done_tasks = {r["task_id"] for r in results}
            print(f"resume: {len(done_tasks)} tasks already in {out}, skipping them", flush=True)
        except Exception as e:
            print(f"resume: could not read {out} ({e}); starting fresh", flush=True)
            results = []
    for tid in task_ids:
        if tid in done_tasks:
            continue
        if tid not in pmap:
            skipped.append(tid); continue
        problem = pmap[tid]
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        prompt_ids = tok.encode(prefix)
        eff_new = min(max_new, max_ctx - len(prompt_ids) - 64)
        if batched:
            recs = run_task_fullbatch(model, tok, problem, prompt_ids, n, base_s, esc_s,
                                      eff_new, max_chops, eos_id, think_end_id, n_g, k_g, w_g,
                                      pad_id, round_cap, phase2_batch=phase2_batch, tid=tid)
        else:
            recs = run_task(model, tok, problem, prompt_ids, n, base_s, esc_s, eff_new,
                            max_chops, eos_id, think_end_id, n_g, k_g, w_g)
        fired = sum(r["fired"] for r in recs)
        ad9 = sum(r["recovered"] for r in recs)
        base9 = sum(r["baseline_recovered"] for r in recs)
        for r in recs:
            r["task_id"] = tid
            results.append(r)
        print(f"task {tid}: fired {fired}/{n} | recovered adaptive {ad9}/{n} vs "
              f"baseline(a2) {base9}/{n} | rescue +{ad9 - base9}", flush=True)
        if out:
            with open(out, "w") as fh:
                json.dump({"meta": {"model": model_id, "n": n, "detector": [n_g, k_g, w_g],
                                    "base_alpha": base_alpha, "esc_alpha": esc_alpha,
                                    "skipped": skipped}, "results": results}, fh)

    # --- summary: deployment pass@1 (adaptive) vs plain-a2 baseline, both from this run ---
    tot = len(results)
    ad = sum(r["recovered"] for r in results)
    base = sum(r["baseline_recovered"] for r in results)
    fired = sum(r["fired"] for r in results)
    print("\n" + "=" * 60)
    print(f"LIVE ADAPTIVE — {len(task_ids) - len(skipped)} tasks x n={n} = {tot} samples")
    print(f"  detector fired: {fired}/{tot} ({fired / max(1, tot):.1%})")
    print(f"  pass@1 plain a2 (baseline): {base}/{tot} = {base / max(1, tot):.3f}")
    print(f"  pass@1 adaptive (a2->chop->a5): {ad}/{tot} = {ad / max(1, tot):.3f}")
    print(f"  rescue gain: +{ad - base} samples ({(ad - base) / max(1, tot):+.3f} pass@1)")
    if out:
        print(f"  results -> {out}")


if __name__ == "__main__":
    main()
