"""Forced-</think> recovery smoke test (task 1226, ATCODER/interview, Qwen3-8B).

Question: if pless's truncated thinking is cut at the loop onset and we force
`</think>` + a code fence, does the model emit a PASSING program?

Three layers (per the experiment design):
  Layer 0 (positive control): take a PASSING pless_norm trace of 1226, reconstruct
    prefix + its own thinking + </think> + ```python, regenerate the code. If this
    does NOT pass, prefix-injection is broken and a null on the treatment is
    uninterpretable. Gates everything.
  Layer 1 (treatment): take the pless truncated trace, cut at loop onset, force
    </think> + ```python, generate, execute.

Faithful prefix recipe (verified): the original run used enable_thinking=True, whose
chat prompt ends with "<|im_start|>assistant\n" (model emits <think> itself). So:
    full = apply_chat_template(enable_thinking=True) + thinking[:cut] + "</think>\n\n```python\n"

Run: HF_HUB_OFFLINE=1 uv run python scripts/forced_think_smoke.py
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
TASK_ID = 1226
N_CODE = 3            # code completions per condition
MAX_CODE_TOKENS = 1024
TEMP = 1.0


def find_loop(text, chunk=120, min_repeat=4):
    for start in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[start:].count(text[start:start + chunk]) >= min_repeat:
            return start
    return None


def get_pless_truncated_trace():
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] == TASK_ID:
                return r["truncated_solution"], r["sample_idx"]
    raise RuntimeError("no pless truncated trace for 1226")


def get_norm_passing_trace():
    m = json.load(open(f"{POD}/metrics/pless_norm_think_t1.0_t1.0_metrics.json"))
    t = next(x for x in m["per_task"] if x["task_id"] == TASK_ID)
    pass_idx = [i for i, p in enumerate(t["pass_results"]) if p]
    with open(f"{POD}/pless_norm_think_t1.0_t1.0.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["task_id"] == TASK_ID:
                return r["samples_with_thinking"][pass_idx[0]], pass_idx[0]
    raise RuntimeError("no pless_norm passing trace for 1226")


def run_condition(label, model, tokenizer, problem, full_prompt, sampler):
    """Generate N_CODE code completions from full_prompt; extract+execute each."""
    print(f"\n=== {label} ===")
    print(f"  prompt chars={len(full_prompt)}  tail={full_prompt[-60:]!r}")
    raw = generate_samples(
        model, tokenizer, full_prompt, sampler,
        n_samples=N_CODE, max_new_tokens=MAX_CODE_TOKENS,
        temperature=TEMP, stop_strings=None,
    )
    passed = 0
    for i, gen in enumerate(raw):
        # generation continues AFTER "```python\n"; rebuild a fenced block for the extractor
        sample_text = "```python\n" + gen
        res, ext = evaluate_apps_sample(sample_text, problem)
        ok = res.status == "Passed"
        passed += ok
        print(f"  code {i}: status={res.status} "
              f"tests={res.n_tests_passed}/{res.n_tests_total} "
              f"extracted={ext.success} gen_chars={len(gen)}")
    print(f"  >>> {label}: {passed}/{N_CODE} PASSED")
    return passed


def main():
    print("Loading APPS test data (ATCODER/interview)...")
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    problem = pmap[TASK_ID]
    print(f"  problem {TASK_ID}: {len(problem.inputs)} test cases, fn_name={problem.fn_name}")

    print("Loading model on MPS...")
    model, tokenizer = load_model_and_tokenizer(MODEL, dtype="bfloat16")

    prefix, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
    assert isinstance(prefix, str) and prefix.endswith("<|im_start|>assistant\n"), \
        f"unexpected prefix tail: {prefix[-40:]!r}"

    sampler = make_guarded_pless_sampler()

    # ── Layer 0: positive control (passing pless_norm trace) ──
    norm_trace, nidx = get_norm_passing_trace()
    ti = norm_trace.find("</think>")
    ctrl_prompt = prefix + norm_trace[:ti] + "</think>\n\n```python\n"
    ctrl_pass = run_condition(
        f"LAYER 0 control (pless_norm sample {nidx}, cut at own </think> @{ti})",
        model, tokenizer, problem, ctrl_prompt, sampler,
    )

    # ── Layer 1: treatment (pless truncated trace, cut at loop onset) ──
    pless_trace, pidx = get_pless_truncated_trace()
    cut = find_loop(pless_trace)
    treat_prompt = prefix + pless_trace[:cut] + "\n</think>\n\n```python\n"
    treat_pass = run_condition(
        f"LAYER 1 treatment (pless sample {pidx}, cut at loop onset @{cut} "
        f"of {len(pless_trace)} = {cut/len(pless_trace):.0%})",
        model, tokenizer, problem, treat_prompt, sampler,
    )

    print("\n" + "=" * 60)
    print(f"SMOKE RESULT (task {TASK_ID}):")
    print(f"  Layer 0 control : {ctrl_pass}/{N_CODE}  "
          f"({'mechanism OK' if ctrl_pass > 0 else 'MECHANISM BROKEN — treatment null is uninterpretable'})")
    print(f"  Layer 1 treat   : {treat_pass}/{N_CODE}  "
          f"({'RECOVERY' if treat_pass > 0 else 'no recovery'})")


if __name__ == "__main__":
    main()
