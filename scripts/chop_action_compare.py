"""3-way post-detection ACTION comparison on REAL pod traces (avoids the fresh-regen
confound). Chop the pless truncated trace at its loop onset, then compare:
  A force_think : inject </think> + code fence → write code NOW (extract existing solution)
  B nudge       : inject a generic "stop looping" nudge → continue THINKING
  C pseudocode  : inject "write the pseudocode then the code" → continue THINKING  [user's idea]
Generate with pless, extract code (after </think> for B/C), execute against APPS tests.

Recoverable tasks only (solution exists in the pre-loop reasoning), so this isolates the
ACTION, not whether a solution was ever found.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/chop_action_compare.py
Env: TASK_IDS("1226 1126 1224"), N(1), MAX_CONT(2048)
"""
import json
import os
import torch

from bench.generator import load_model_and_tokenizer, generate_samples
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_guarded_pless_sampler
from scripts.repeat_detector import RepeatDetector

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
TASK_IDS = [int(x) for x in os.environ.get("TASK_IDS", "1226 1126 1224").split()]
N = int(os.environ.get("N", "1"))
MAX_CONT = int(os.environ.get("MAX_CONT", "2048"))

ACTIONS = {
    "A_force_think": "\n</think>\n\n```python\n",
    "B_nudge":       "\n\nWait, I'm going in circles. Let me write the solution directly.\n",
    "C_pseudocode":  "\n\nLet me stop and write the pseudocode for the algorithm step by step, "
                     "then implement it:\n",
}


def find_loop(text, chunk=120, min_repeat=4):
    for s in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[s:].count(text[s:s + chunk]) >= min_repeat:
            return s
    return None


def get_trace(tid):
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] == tid:
                return r["truncated_solution"]
    return None


def extract_status(gen, action, problem):
    if action == "A_force_think":
        sample = "```python\n" + gen
    else:
        if "</think>" not in gen:
            return "no_</think>"
        code = gen.split("</think>", 1)[1]
        sample = code if "```" in code else "```python\n" + code
    res, _ = evaluate_apps_sample(sample, problem)
    return res.status


def main():
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tok = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    _base = make_guarded_pless_sampler()

    def sampler(probs):
        # MPS bfloat16 can emit NaN/inf logits on long-context forwards → NaN probs →
        # multinomial crash. Sanitize (no-op when clean; would not occur on CUDA).
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        bad = (probs.sum(-1, keepdim=True) <= 0).squeeze(-1)
        if bad.any():
            probs[bad] = 1.0  # rare all-NaN row → uniform fallback (prevents crash)
        return _base(probs)

    results = []
    for tid in TASK_IDS:
        problem = pmap[tid]
        trace = get_trace(tid)
        cut = find_loop(trace)
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        base = prefix + trace[:cut]
        print(f"\n=== task {tid}: cut@{cut} ({cut/len(trace):.0%} of {len(trace)}), {len(problem.inputs)} tests ===", flush=True)
        for act, inject in ACTIONS.items():
            full = base + inject
            try:
                raw = generate_samples(model, tok, full, sampler, n_samples=N,
                                       max_new_tokens=MAX_CONT, temperature=1.0, stop_strings=None)
            except Exception as e:
                results.append((tid, act, 0, f"EXC:{type(e).__name__}", False, 0))
                print(f"  {act:<14} EXC: {type(e).__name__}", flush=True)
                continue
            for i, gen in enumerate(raw):
                st = extract_status(gen, act, problem)
                ok = st == "Passed"
                results.append((tid, act, i, st, ok, len(gen)))
                print(f"  {act:<14} s{i}: {st:<12} {'PASS' if ok else ''} gen={len(gen)}c", flush=True)


    print("\n" + "=" * 60)
    from collections import defaultdict
    by_act = defaultdict(lambda: [0, 0])
    for tid, act, i, st, ok, n in results:
        by_act[act][0] += ok; by_act[act][1] += 1
    print("recovered (Passed) by action:")
    for act in ACTIONS:
        p, tot = by_act[act]
        print(f"  {act:<14}: {p}/{tot}")


if __name__ == "__main__":
    main()
