"""Forced-</think> recovery across categories (taxonomy validation, local MPS).

Tests the loop-onset-recovery prediction across the R1/R2/S/L categories:
  - R1/R2 (solution/code present at loop onset) → predict RECOVERY
  - S/L  (still searching / pure loop, no solution) → predict NO recovery

Treatment only (mechanism already validated by forced_think_smoke.py Layer 0):
cut the pless truncated trace at loop onset, force </think> + code fence,
generate with pless, extract+execute against the real APPS tests.

Run: PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/forced_think_multi.py
"""
import json
import torch

from bench.generator import load_model_and_tokenizer, generate_samples
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_guarded_pless_sampler

MODEL = "Qwen/Qwen3-8B"
POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
N_CODE = 1            # pless is near-deterministic on code → n=1 is the honest count
MAX_CODE_TOKENS = 896
TEMP = 1.0

# (task_id, predicted_category, prediction). 1126 already confirmed RECOVER (2/2) in a prior run.
TASKS = [
    (558,  "R1", "recover"),
    (1085, "S",  "fail"),
    (990,  "L",  "fail"),
]


def find_loop(text, chunk=120, min_repeat=4):
    for start in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[start:].count(text[start:start + chunk]) >= min_repeat:
            return start
    return None


def get_pless_trace(tid):
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] == tid:
                return r["truncated_solution"], r["sample_idx"]
    return None, None


def main():
    print("Loading APPS test data (ATCODER/interview)...")
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    print("Loading model on MPS...")
    model, tokenizer = load_model_and_tokenizer(MODEL, dtype="bfloat16")
    sampler = make_guarded_pless_sampler()

    summary = []
    for tid, cat, pred in TASKS:
        problem = pmap[tid]
        trace, sidx = get_pless_trace(tid)
        cut = find_loop(trace)
        prefix, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
        full = prefix + trace[:cut] + "\n</think>\n\n```python\n"
        print(f"\n=== task {tid} [{cat}, predict {pred}] "
              f"cut@{cut} ({cut/len(trace):.0%} of {len(trace)}) "
              f"{len(problem.inputs)} tests ===", flush=True)
        raw = generate_samples(
            model, tokenizer, full, sampler,
            n_samples=N_CODE, max_new_tokens=MAX_CODE_TOKENS,
            temperature=TEMP, stop_strings=None,
        )
        passed = 0
        statuses = []
        for i, gen in enumerate(raw):
            res, ext = evaluate_apps_sample("```python\n" + gen, problem)
            ok = res.status == "Passed"
            passed += ok
            statuses.append(res.status)
            print(f"  code {i}: {res.status} tests={res.n_tests_passed}/{res.n_tests_total} "
                  f"gen_chars={len(gen)}", flush=True)
        verdict = "RECOVER" if passed > 0 else "no-recover"
        match = "✓" if (verdict == "RECOVER") == (pred == "recover") else "✗ MISMATCH"
        summary.append((tid, cat, pred, f"{passed}/{N_CODE}", verdict, match))
        print(f"  >>> {passed}/{N_CODE} passed — {verdict} (predicted {pred}) {match}", flush=True)
        # free KV/activation memory before the next (long-prefix) task
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'task':>5} {'cat':>3} {'predict':>8} {'pass':>5} {'verdict':>11} {'match':>10}")
    for tid, cat, pred, pf, verdict, match in summary:
        print(f"{tid:>5} {cat:>3} {pred:>8} {pf:>5} {verdict:>11} {match:>10}")


if __name__ == "__main__":
    main()
