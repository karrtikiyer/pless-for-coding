"""Forced-</think> recovery at n=10 (vLLM, single GPU) — PLESS primary + temp ceiling.

The RESCUE question: for each solvable pless-truncated ATCODER-interview task, cut the
pless trace at its loop onset, force </think> + ```python, and re-generate the CODE.
Does a correct program come out, and at what per-task rate?

Two arms per task (the gap between them is the finding):
  - PLESS (primary, faithful): code phase uses pless @ temp 1.0 — the SAME sampler the
    original run used, so this answers "if pless had stopped thinking at the loop onset,
    would IT produce correct code?" pless is only ~deterministic where the distribution
    is peaked; on uncertain code positions multiple tokens survive the Σpᵢ² threshold and
    it samples with real diversity, so n>1 is meaningful.
  - TEMP (ceiling): standard temperature sampling (0.8 / top_p 0.95). Answers "is a correct
    solution recoverable AT ALL?" A task temp recovers but pless can't ⇒ the solution is in
    the reasoning but pless's threshold can't reach it (fixable sampler problem). Neither
    recovers ⇒ reasoning insufficient.

Run via ./run_forced_think_recovery_apps_qwen3.sh (sets vLLM env + venv).
Env: ARMS ("pless temp" | "pless" | "temp"), N_SAMPLES(10), PLESS_TEMP(1.0),
     TEMP(0.8), TOP_P(0.95), MAX_CODE_TOKENS(1024), MAX_MODEL_LEN(24576),
     GPU_MEM_UTIL(0.90), MODEL, OUT.
"""
import json
import math
import os
from collections import defaultdict

from bench.generator_vllm import (
    load_engine, generate_samples_vllm, generate_samples_standard_vllm,
)
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
OUT = os.environ.get("OUT", "results/forced_think_recovery/recovery_n10.json")
N = int(os.environ.get("N_SAMPLES", "10"))
ARMS = os.environ.get("ARMS", "pless temp").split()
PLESS_TEMP = float(os.environ.get("PLESS_TEMP", "1.0"))   # faithful to the original run
TEMP = float(os.environ.get("TEMP", "0.8"))               # ceiling arm
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_CODE_TOKENS = int(os.environ.get("MAX_CODE_TOKENS", "1024"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "24576"))
GPU_MEM = float(os.environ.get("GPU_MEM_UTIL", "0.90"))

UNSOLVABLE = {117, 280, 326, 370, 454, 455, 512, 661, 962, 1122, 1175, 1223, 1368}
KS = [1, 3, 5, 10]


def find_loop(text, chunk=120, min_repeat=4):
    for start in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[start:].count(text[start:start + chunk]) >= min_repeat:
            return start
    return None


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def evaluate(gens, problem):
    pr = [evaluate_apps_sample("```python\n" + g, problem)[0].status == "Passed" for g in gens]
    c = sum(pr)
    return c, {k: round(pass_at_k(len(gens), c, k), 3) for k in KS}


def main():
    traces = defaultdict(list)
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0":
                traces[r["task_id"]].append((r["sample_idx"], r["truncated_solution"]))
    solvable = sorted(set(traces) - UNSOLVABLE)
    print(f"Solvable tasks: {len(solvable)}; arms={ARMS}; N={N} "
          f"pless@T={PLESS_TEMP} temp={TEMP}/top_p={TOP_P}", flush=True)

    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # register pless processor (needed for the pless arm); temp arm ignores it.
    engine = load_engine(MODEL, max_model_len=MAX_MODEL_LEN,
                         gpu_memory_utilization=GPU_MEM,
                         register_pless_logitsproc=("pless" in ARMS))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    for i, tid in enumerate(solvable, 1):
        problem = pmap.get(tid)
        if problem is None:
            continue
        best = None
        for sidx, text in traces[tid]:
            lp = find_loop(text)
            if lp is not None and (best is None or lp < best[0]):
                best = (lp, sidx, text)
        if best is None:
            sidx, text = max(traces[tid], key=lambda x: len(x[1]))
            cut, cut_kind = int(len(text) * 0.6), "fallback60%"
        else:
            cut, sidx, text = best
            cut_kind = "loop"

        prefix, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
        full = prefix + text[:cut] + "\n</think>\n\n```python\n"

        row = {"task_id": tid, "cut_kind": cut_kind, "cut_pct": round(cut / len(text), 3),
               "n_tests": len(problem.inputs), "n": N, "arms": {}}
        for arm in ARMS:
            if arm == "pless":
                gens = generate_samples_vllm(
                    engine, tokenizer, full, sampler_name="pless", n_samples=N,
                    max_new_tokens=MAX_CODE_TOKENS, temperature=PLESS_TEMP, stop_strings=None)
            else:  # temp
                gens = generate_samples_standard_vllm(
                    engine, tokenizer, full, n_samples=N,
                    max_new_tokens=MAX_CODE_TOKENS, temperature=TEMP, top_p=TOP_P, stop_strings=None)
            c, pak = evaluate(gens, problem)
            row["arms"][arm] = {"n_pass": c, "pass_at_k": pak}
        results.append(row)
        msg = "  ".join(f"{a}={row['arms'][a]['n_pass']}/{N}" for a in ARMS)
        print(f"[{i}/{len(solvable)}] task {tid} (cut@{cut/len(text):.0%}): {msg}", flush=True)
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=2)

    print("\n" + "=" * 60)
    for arm in ARMS:
        nrec = sum(1 for r in results if r["arms"][arm]["n_pass"] > 0)
        p1 = round(sum(r["arms"][arm]["pass_at_k"][1] for r in results) / len(results), 3)
        p10 = round(sum(r["arms"][arm]["pass_at_k"][10] for r in results) / len(results), 3)
        print(f"{arm:>6}: recovered (>=1/{N}) {nrec}/{len(results)} tasks  "
              f"mean pass@1={p1} pass@10={p10}")
    if "pless" in ARMS and "temp" in ARMS:
        pless_only = [r["task_id"] for r in results
                      if r["arms"]["pless"]["n_pass"] > 0 and r["arms"]["temp"]["n_pass"] == 0]
        temp_only = [r["task_id"] for r in results
                     if r["arms"]["temp"]["n_pass"] > 0 and r["arms"]["pless"]["n_pass"] == 0]
        print(f"  temp recovers but pless can't (sampler-reachable): {sorted(temp_only)}")
        print(f"  pless recovers but temp can't: {sorted(pless_only)}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
