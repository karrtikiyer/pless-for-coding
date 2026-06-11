"""Dump task 558's forced-</think> raw generation to diagnose the ParsingError."""
import json
import torch
from bench.generator import load_model_and_tokenizer, generate_samples
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.eval.executor import extract_python_code
from bench.sampler_bridge import make_guarded_pless_sampler

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"


def find_loop(text, chunk=120, min_repeat=4):
    for start in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[start:].count(text[start:start + chunk]) >= min_repeat:
            return start
    return None


pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
problem = pmap[558]
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")

trace = None
with open(f"{POD}/truncated_cases.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] == 558:
            trace = r["truncated_solution"]; break

cut = find_loop(trace)
prefix, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
full = prefix + trace[:cut] + "\n</think>\n\n```python\n"
print(f"cut@{cut}, prompt chars={len(full)}", flush=True)

raw = generate_samples(model, tokenizer, full, make_guarded_pless_sampler(),
                       n_samples=1, max_new_tokens=896, temperature=1.0, stop_strings=None)
gen = raw[0]
print("=" * 70)
print(f"RAW GENERATION ({len(gen)} chars):")
print(gen)
print("=" * 70)
sample_text = "```python\n" + gen
extracted = extract_python_code(sample_text)
print(f"EXTRACTED CODE ({len(extracted)} chars):")
print(repr(extracted[:500]))
print("=" * 70)
res, ext = evaluate_apps_sample(sample_text, problem)
print(f"status={res.status} ext.success={ext.success} tests={res.n_tests_passed}/{res.n_tests_total}")
print(f"stderr_excerpt={res.stderr_excerpt!r}")
