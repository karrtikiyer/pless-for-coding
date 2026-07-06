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

from bench.generator import load_model_and_tokenizer
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_pless_alpha_sampler
from scripts.chop_restart_alpha_compare import decode_round, make_safe
from scripts.adaptive_loop import adaptive_continue


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
    out = os.environ.get("OUT", "")

    pmap = load_apps_test_map(source=source, difficulty=difficulty)
    task_ids = ([int(x) for x in task_env.split()] if task_env else sorted(pmap))
    if max_problems:
        task_ids = task_ids[:max_problems]

    model, tok = load_model_and_tokenizer(model_id, dtype="bfloat16")
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|im_end|>")
    think_end_id = resolve_think_end(tok)
    base_s = make_safe(make_pless_alpha_sampler(base_alpha))
    esc_s = make_safe(make_pless_alpha_sampler(esc_alpha))

    print(f"model={model_id} | {source}/{difficulty} | tasks={len(task_ids)} n={n} | "
          f"base=a{int(base_alpha)}->esc=a{int(esc_alpha)} | detect {n_g}-gram k{k_g}/w{w_g} | "
          f"max_new={max_new} max_chops={max_chops} think_end={think_end_id}", flush=True)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    results, skipped = [], []
    for tid in task_ids:
        if tid not in pmap:
            skipped.append(tid); continue
        problem = pmap[tid]
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        prompt_ids = tok.encode(prefix)
        eff_new = min(max_new, max_ctx - len(prompt_ids) - 64)
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
