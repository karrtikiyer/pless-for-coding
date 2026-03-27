"""
Quick test of OpenCodeInterpreter-DS-1.3B on MPS.

Checks:
  1. Tokenizer chat_template exists and what it looks like
  2. Instruct format prompt for a sample MBPP task
  3. 20-task pass@1 with instruct format (n=5 samples)
  4. 20-task pass@1 with BigCode format for comparison

Run: uv run python test_oci_mps.py
"""

import json
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.eval.executor import evaluate_all, extract_python_code
from bench.prompts import format_prompt_base_bigcode, format_prompt_instruct

# BPE artifact fix — this tokenizer's decode() returns GPT-2 byte-level Unicode
# characters (e.g. Ġ=space, Ċ=newline) instead of actual bytes.  Apply the
# inverse GPT-2 byte encoder to recover proper UTF-8 text.
def _build_bpe_trans():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return str.maketrans({chr(c): chr(b) for b, c in zip(bs, cs) if c != b and c > 127})

_BPE_TRANS = _build_bpe_trans()

def fix_bpe(text: str) -> str:
    return text.translate(_BPE_TRANS)

MODEL_ID = "m-a-p/OpenCodeInterpreter-DS-1.3B"
N_TASKS = 20          # number of MBPP tasks to test
N_SAMPLES = 5         # samples per task (enough for pass@1 estimate)
MAX_NEW_TOKENS = 512

MBPP_BIGCODE_STOP = ["\nassert", "\nclass", "\nprint", '\n"""', "\nif __name__"]


# ─── helpers ─────────────────────────────────────────────────────────────────

def _truncate_at_stop(text: str, stops: list[str]) -> str:
    earliest = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1 and i < earliest:
            earliest = i
    return text[:earliest]


def generate_n(model, tokenizer, prompt, n, max_new_tokens, stop_strings=None, temperature=0.2, top_p=0.95):
    """Generate n samples for a prompt using model.generate()."""
    if isinstance(prompt, list):
        input_ids = torch.tensor([prompt], device=model.device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prompt_len = input_ids.shape[1]

    kwargs = {}
    if stop_strings:
        kwargs["stop_strings"] = stop_strings
        kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=pad_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            num_return_sequences=n,
            **kwargs,
        )

    decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    samples = []
    for i in range(n):
        full_text = tokenizer.decode(out[i], skip_special_tokens=True)
        text = full_text[len(decoded_prompt):]
        if stop_strings:
            text = _truncate_at_stop(text, stop_strings)
        samples.append(fix_bpe(text))
    return samples


def run_tasks(model, tokenizer, tasks, mode):
    """Run tasks in 'instruct' or 'bigcode' mode. Returns (records, pass@1)."""
    records = []
    for task in tasks:
        if mode == "instruct":
            prompt, code_prefix = format_prompt_instruct(task, tokenizer)
            stop_strings = None
        else:
            prompt, code_prefix = format_prompt_base_bigcode(task)
            stop_strings = MBPP_BIGCODE_STOP

        samples_raw = generate_n(model, tokenizer, prompt, N_SAMPLES,
                                  MAX_NEW_TOKENS, stop_strings)
        samples = [code_prefix + s for s in samples_raw]
        records.append({
            "task_id": task["task_id"],
            "prompt_text": task["prompt"],
            "samples": samples,
            "test_list": task["test_list"],
        })

    task_results = evaluate_all(records, "mbpp", timeout=5.0, workers=4)
    n_correct = sum(r["num_correct"] for r in task_results)
    n_total = sum(len(r["pass_results"]) for r in task_results)
    pass_at_1 = sum(
        1 - (1 - r["num_correct"] / len(r["pass_results"])) ** N_SAMPLES
        for r in task_results
        if len(r["pass_results"]) > 0
    ) / len(task_results) * 100  # rough pass@1 via inclusion-exclusion
    return records, task_results, pass_at_1


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: tokenizer only ────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Check tokenizer / chat_template")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Vocab size:      {tokenizer.vocab_size}")
    print(f"EOS token:       {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print(f"Has chat_template: {tokenizer.chat_template is not None}")
    if tokenizer.chat_template:
        print(f"chat_template (first 300 chars):\n  {tokenizer.chat_template[:300]!r}")
    else:
        print("WARNING: No chat_template — apply_chat_template will likely fail")
    print()

    # ── Step 2: what does the instruct prompt look like? ─────────────────────
    print("=" * 60)
    print("STEP 2: Sample instruct prompt")
    print("=" * 60)
    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
    dataset = dataset.map(lambda t: {"prompt": t["text"]})
    tasks = list(dataset)[:N_TASKS]

    sample_task = tasks[0]
    instruct_prompt, _ = format_prompt_instruct(sample_task, tokenizer)
    bigcode_prompt, _ = format_prompt_base_bigcode(sample_task)

    print(f"Task: {sample_task['prompt']}")
    print()
    print("── Instruct prompt ──")
    print(instruct_prompt)
    print()
    print("── BigCode prompt ──")
    print(bigcode_prompt)
    print()

    # ── Step 3: load model on MPS ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Load model on MPS")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # should route to MPS
        attn_implementation="sdpa",
        use_cache=True,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    print()

    # ── Step 4: show raw output for task 0 ───────────────────────────────────
    print("=" * 60)
    print("STEP 4: Raw generation for task 0 (both modes)")
    print("=" * 60)
    print("── Instruct output (1 sample, BPE-fixed) ──")
    prompt_i, _ = format_prompt_instruct(sample_task, tokenizer)
    out_i = generate_n(model, tokenizer, prompt_i, 1, MAX_NEW_TOKENS, stop_strings=None)
    # generate_n already applies fix_bpe; show what comes out
    print(out_i[0][:800])
    print()
    extracted_i = extract_python_code(out_i[0])
    print("  → extracted:")
    print(extracted_i[:400])
    print()
    print("── BigCode output (1 sample, BPE-fixed) ──")
    prompt_b, _ = format_prompt_base_bigcode(sample_task)
    out_b = generate_n(model, tokenizer, prompt_b, 1, MAX_NEW_TOKENS, stop_strings=MBPP_BIGCODE_STOP)
    print(out_b[0][:800])
    print()
    print("  → extracted:")
    print(extract_python_code(out_b[0])[:400])
    print()

    # ── Step 5: 20-task pass@1 ────────────────────────────────────────────────
    print("=" * 60)
    print(f"STEP 5: {N_TASKS}-task pass@1 (n={N_SAMPLES} samples each)")
    print("=" * 60)

    print(f"\nRunning INSTRUCT mode ({N_TASKS} tasks) ...")
    _, instruct_results, instruct_pass1 = run_tasks(model, tokenizer, tasks, "instruct")
    print(f"  pass@1 (instruct): {instruct_pass1:.1f}%")
    per_task_i = [(r["task_id"], r["num_correct"], len(r["pass_results"])) for r in instruct_results]
    print(f"  per-task correct: {per_task_i}")

    print(f"\nRunning BIGCODE mode ({N_TASKS} tasks) ...")
    _, bigcode_results, bigcode_pass1 = run_tasks(model, tokenizer, tasks, "bigcode")
    print(f"  pass@1 (bigcode): {bigcode_pass1:.1f}%")
    per_task_b = [(r["task_id"], r["num_correct"], len(r["pass_results"])) for r in bigcode_results]
    print(f"  per-task correct: {per_task_b}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Instruct pass@1 ({N_TASKS} tasks, n={N_SAMPLES}): {instruct_pass1:.1f}%")
    print(f"BigCode  pass@1 ({N_TASKS} tasks, n={N_SAMPLES}): {bigcode_pass1:.1f}%")
    print(f"Paper baseline (500 tasks, n=10):                 44.0%")


if __name__ == "__main__":
    main()
