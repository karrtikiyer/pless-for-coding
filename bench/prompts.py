# Fixed 3-shot examples from MBPP train split — identical to paper's Llama2_input.jsonl
_MBPP_FEW_SHOT_EXAMPLES = [
    {
        "desc": "Write a function to find the similar elements from the given two tuple lists.",
        "tests": (
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\n"
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
        ),
        "solution": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "    res = tuple(set(test_tup1) & set(test_tup2))\n"
            "    return (res)"
        ),
    },
    {
        "desc": "Write a python function to identify non-prime numbers.",
        "tests": (
            "assert is_not_prime(2) == False\n"
            "assert is_not_prime(10) == True\n"
            "assert is_not_prime(35) == True"
        ),
        "solution": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    result = False\n"
            "    for i in range(2,int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n"
            "            result = True\n"
            "    return result"
        ),
    },
    {
        "desc": "Write a function to find squares of individual elements in a list using lambda function.",
        "tests": (
            "assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]\n"
            "assert square_nums([10,20,30])==([100,400,900])\n"
            "assert square_nums([12,15])==([144,225])"
        ),
        "solution": (
            "def square_nums(nums):\n"
            "    square_nums = list(map(lambda x: x ** 2, nums))\n"
            "    return square_nums"
        ),
    },
]


def format_prompt_base(task: dict, n_shots: int = 3) -> tuple[str, str]:
    """Completion prompt for base models, matching the paper's Llama2_input.jsonl format.

    n_shots controls how many few-shot examples to include (0 = zero-shot, default 3).
    """
    desc = task["prompt"]
    test_lines = "\n".join(task["test_list"])

    parts = []
    for ex in _MBPP_FEW_SHOT_EXAMPLES[:n_shots]:
        parts.append(
            f"You are an expert Python programmer, and here is your task: {ex['desc']} "
            f"Your code should pass these tests:\n\n{ex['tests']}\n[BEGIN]\n{ex['solution']}\n[DONE]\n\n"
        )
    parts.append(
        f"You are an expert Python programmer, and here is your task: {desc} "
        f"Your code should pass these tests:\n\n{test_lines}\n[BEGIN]\n"
    )
    return "".join(parts), ""  # code_prefix="" — model generates full def


def format_prompt_instruct(task: dict, tokenizer) -> tuple[str, str]:
    """Format a chat-template prompt for instruct models.

    Returns (full_prompt, code_prefix). For instruct models the code_prefix
    is empty since the model generates the complete function.
    """
    test_lines = "\n".join(task["test_list"])
    user_msg = (
        f"Write a Python function to solve the following task.\n\n"
        f"Task: {task['prompt']}\n\n"
        f"Your solution must pass these tests:\n```\n{test_lines}\n```\n\n"
        f"Provide only the function implementation, no explanations."
    )
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": user_msg},
    ]
    # Most tokenizers handle special tokens correctly when encoding the
    # template string, so default to the text path (preserves BOS behaviour).
    # Old Qwen tokenizers are the exception: <|im_start|>/<|im_end|> get
    # split into subwords by encode(), producing a garbled prompt.  For
    # those we tokenize directly (return_dict=False gives list[int] in
    # transformers 5.x which defaults to BatchEncoding otherwise).
    if getattr(tokenizer, '_qwen_direct_tokenize', False):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return prompt, ""


def is_instruct_model(model_id: str) -> bool:
    """Check if a model ID indicates an instruct/chat model."""
    lower = model_id.lower()
    return "instruct" in lower or "chat" in lower
