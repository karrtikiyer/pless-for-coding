"""Forced-</think> recovery across ALL solvable pless-truncated tasks (local MPS, n=1).

First-pass per-task recovery screen before the GPU n=10 sampling run. For each
solvable pless-truncated ATCODER/interview task: take the pless truncated trace
with the earliest loop onset, cut there, force </think> + code fence, generate
with pless, extract + execute against the real APPS tests.

n=1 because pless is near-deterministic on code (verified: identical completions).
Single draw per task — a screen, not a rate. Saves raw generations for diagnosis.

Run: PYTHONPATH=. HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
       uv run python scripts/forced_think_all27.py
"""
import json
import torch
from collections import defaultdict

from bench.generator import load_model_and_tokenizer, generate_samples
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_guarded_pless_sampler

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
OUT = "results/_repro_loop_probe/forced_think_all27_results.json"
MAX_CODE_TOKENS = 896
TEMP = 1.0

UNSOLVABLE = {117, 280, 326, 370, 454, 455, 512, 661, 962, 1122, 1175, 1223, 1368}


def find_loop(text, chunk=120, min_repeat=4):
    for start in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[start:].count(text[start:start + chunk]) >= min_repeat:
            return start
    return None


def main():
    # ── gather pless truncated traces, grouped by task ──
    traces = defaultdict(list)  # tid -> [(sidx, text)]
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0":
                traces[r["task_id"]].append((r["sample_idx"], r["truncated_solution"]))

    solvable = sorted(set(traces) - UNSOLVABLE)
    print(f"Solvable pless-truncated tasks: {len(solvable)}", flush=True)

    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    sampler = make_guarded_pless_sampler()

    results = []
    n_recover = 0
    for k, tid in enumerate(solvable, 1):
        problem = pmap.get(tid)
        if problem is None:
            print(f"[{k}/{len(solvable)}] task {tid}: NO PROBLEM DATA — skip", flush=True)
            continue
        # pick the sample with the earliest detected loop onset; fallback 60% of longest
        best = None  # (loop_pos, sidx, text)
        for sidx, text in traces[tid]:
            lp = find_loop(text)
            if lp is not None and (best is None or lp < best[0]):
                best = (lp, sidx, text)
        if best is None:
            sidx, text = max(traces[tid], key=lambda x: len(x[1]))
            cut = int(len(text) * 0.6)
            cut_kind = "fallback60%"
        else:
            cut, sidx, text = best
            cut_kind = "loop"

        prefix, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
        full = prefix + text[:cut] + "\n</think>\n\n```python\n"
        try:
            raw = generate_samples(model, tokenizer, full, sampler,
                                   n_samples=1, max_new_tokens=MAX_CODE_TOKENS,
                                   temperature=TEMP, stop_strings=None)
            gen = raw[0]
            res, ext = evaluate_apps_sample("```python\n" + gen, problem)
            status, npass, ntot = res.status, res.n_tests_passed, res.n_tests_total
            ok = status == "Passed"
        except Exception as e:
            gen, status, npass, ntot, ok = "", f"EXC:{type(e).__name__}", 0, 0, False
        n_recover += ok
        results.append({
            "task_id": tid, "cut_kind": cut_kind, "cut": cut, "trace_len": len(text),
            "cut_pct": round(cut / len(text), 3), "sample_idx": sidx,
            "status": status, "n_tests_passed": npass, "n_tests_total": ntot,
            "recovered": ok, "n_tests": len(problem.inputs), "gen": gen,
        })
        print(f"[{k}/{len(solvable)}] task {tid}: cut@{cut} ({cut/len(text):.0%},{cut_kind}) "
              f"-> {status} {npass}/{ntot} {'RECOVER' if ok else ''}", flush=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print(f"RECOVERED {n_recover}/{len(results)} tasks (n=1, single draw each)")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
