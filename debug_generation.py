"""Debug script: compare float32 vs bfloat16 generation on HumanEval tasks.

Runs both dtypes and prints samples side by side for comparison.

Usage:
    .venv-debug/bin/python debug_generation.py
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
N_SAMPLES = 5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

# Exact HumanEval prompts (canonical dataset prompt field)
TASKS = {
    "HumanEval/150": (
        '\ndef x_or_y(n, x, y):\n'
        '    """A simple program which should return the value of x if n is \n'
        '    a prime number and should return the value of y otherwise.\n'
        '\n'
        '    Examples:\n'
        '    for x_or_y(7, 34, 12) == 34\n'
        '    for x_or_y(15, 8, 5) == 5\n'
        '    \n'
        '    """\n'
    ),
    "HumanEval/74": (
        "\ndef total_match(lst1, lst2):\n"
        "    '''\n"
        "    Write a function that accepts two lists of strings and returns the list that has \n"
        "    total number of chars in the all strings of the list less than the other list.\n"
        "\n"
        "    if the two lists have the same number of chars, return the first list.\n"
        "\n"
        "    Examples\n"
        "    total_match([], []) ➞ []\n"
        "    total_match(['hi', 'admin'], ['hI', 'Hi']) ➞ ['hI', 'Hi']\n"
        "    total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project']) ➞ ['hi', 'admin']\n"
        "    total_match(['hi', 'admin'], ['hI', 'hi', 'hi']) ➞ ['hI', 'hi', 'hi']\n"
        "    total_match(['4'], ['1', '2', '3', '4', '5']) ➞ ['4']\n"
        "    '''\n"
    ),
    "HumanEval/43": (
        '\n\ndef pairs_sum_to_zero(l):\n'
        '    """\n'
        '    pairs_sum_to_zero takes a list of integers as an input.\n'
        '    it returns True if there are two distinct elements in the list that\n'
        '    sum to zero, and False otherwise.\n'
        '    >>> pairs_sum_to_zero([1, 3, 5, 0])\n'
        '    False\n'
        '    >>> pairs_sum_to_zero([1, 3, -2, 1])\n'
        '    False\n'
        '    >>> pairs_sum_to_zero([1, 2, 3, 7])\n'
        '    False\n'
        '    >>> pairs_sum_to_zero([2, 4, -5, 3, 5, 7])\n'
        '    True\n'
        '    >>> pairs_sum_to_zero([1])\n'
        '    False\n'
        '    """\n'
    ),
    "HumanEval/14": (
        "from typing import List\n\n\n"
        "def all_prefixes(string: str) -> List[str]:\n"
        '    """ Return list of all prefixes from shortest to longest of the input string\n'
        "    >>> all_prefixes('abc')\n"
        "    ['a', 'ab', 'abc']\n"
        '    """\n'
    ),
    "HumanEval/128": (
        '\ndef prod_signs(arr):\n'
        '    """\n'
        '    You are given an array arr of integers and you need to return\n'
        '    sum of magnitudes of integers multiplied by product of all signs\n'
        '    of each number in the array, represented by 1, -1 or 0.\n'
        '    Note: return None for empty arr.\n'
        '\n'
        '    Example:\n'
        '    >>> prod_signs([1, 2, 2, -4]) == -9\n'
        '    >>> prod_signs([0, 1]) == 0\n'
        '    >>> prod_signs([]) == None\n'
        '    """\n'
    ),
}

STOP_STRINGS = ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint("]


def truncate_at_stop(text, stop_strings):
    earliest = len(text)
    for stop in stop_strings:
        idx = text.find(stop)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def run_generation(model, tokenizer, dtype_name):
    """Generate samples for all tasks with the given model."""
    print(f"\n{'#' * 70}")
    print(f"# DTYPE: {dtype_name}  |  TEMP: {TEMPERATURE}")
    print(f"{'#' * 70}")

    for task_id, prompt_text in TASKS.items():
        print(f"\n{'=' * 70}")
        print(f"TASK: {task_id}  |  DTYPE: {dtype_name}")
        print(f"{'=' * 70}")

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        prompt_len = input_ids.shape[1]

        print(f"Generating {N_SAMPLES} samples...")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=0,
                top_p=1.0,
                num_return_sequences=N_SAMPLES,
            )

        has_stub = 0
        for i in range(N_SAMPLES):
            generated = output[i, prompt_len:]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            truncated = truncate_at_stop(raw, STOP_STRINGS)
            is_stub = (
                truncated.strip() in ("pass", "")
                or "# your code here" in truncated.lower()
                or "# write your code" in truncated.lower()
            )
            if is_stub:
                has_stub += 1
            print(f"\n--- sample[{i}] ({len(generated)} tokens) {'[STUB]' if is_stub else ''} ---")
            print(truncated)

        print(f"\n>>> {task_id}: {has_stub}/{N_SAMPLES} stubs")


def main():
    print(f"Model: {MODEL_ID}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"\nTokenizer info:")
    print(f"  add_bos_token: {getattr(tokenizer, 'add_bos_token', 'N/A')}")
    print(f"  bos_token_id: {tokenizer.bos_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")

    attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

    # --- Run 1: float32 on CPU (doesn't fit on MPS) ---
    print(f"\nLoading model in float32 (CPU)...")
    model_f32 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
        use_cache=True,
        trust_remote_code=True,
    )
    model_f32.eval()
    run_generation(model_f32, tokenizer, "float32")
    del model_f32
    gc.collect()

    # --- Run 2: bfloat16 on MPS/CUDA ---
    print(f"\nLoading model in bfloat16...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        use_cache=True,
        trust_remote_code=True,
    )
    model_bf16.eval()
    run_generation(model_bf16, tokenizer, "bfloat16")
    del model_bf16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\n" + "=" * 70)
    print("DONE. Compare float32 vs bfloat16 results above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
