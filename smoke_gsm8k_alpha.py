#!/usr/bin/env python3
"""Local smoke test: Qwen2.5-Coder on GSM8K with the Wei 2022 8-shot CoT
prompt and pless_alpha sampling at two α values.

Purpose: BEFORE committing to a 100+ GPU-hour full sweep, verify on Mac MPS
(or any local CPU/GPU) that the experimental design will actually produce
signal. Four sanity checks at the end:

  1. Code-format slippage — does the coder model write Python despite CoT prompt?
  2. Answer-extraction rate — does the model end with "The answer is N." reliably?
  3. Accuracy floor — is pass@1 in a reasonable range (~30-60% for 7B GSM8K)?
  4. Reasoning variation — do α=2 and α=5 produce visibly different reasoning?

This is a research-only smoke. Not committed to bench/. Default size keeps
the run under ~15 min on M-series MPS.

Usage:
    uv run python smoke_gsm8k_alpha.py
    # Larger smoke:
    uv run python smoke_gsm8k_alpha.py --n-problems 5 --n-samples 5
    # Smaller, RAM-constrained Macs:
    uv run python smoke_gsm8k_alpha.py --model Qwen/Qwen2.5-Coder-3B-Instruct

The 8-shot CoT prompt below is verified verbatim from Table 20 of
Wei et al. 2022 ("Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models", arXiv:2201.11903).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


# Verified verbatim from Wei et al. 2022, Table 20 (page 35 of the PDF).
WEI_2022_GSM8K_8SHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--n-problems", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--alphas", type=str, default="2.0,5.0",
                   help="Comma-separated α values (default '2.0,5.0' — extremes of our grid)")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="smoke_gsm8k_alpha.jsonl")
    return p.parse_args()


def gold_answer_from_gsm8k(answer_field: str) -> str:
    """Extract the gold numeric answer (after '####' in GSM8K answers)."""
    m = re.search(r"####\s*([+-]?\d+(?:[.,]\d+)?)", answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def extract_predicted_answer(completion: str) -> str | None:
    """Extract numeric answer from 'The answer is N.' (Wei 2022 format)."""
    matches = re.findall(
        r"answer is\s*\$?\s*([+-]?\d+(?:[.,]\d+)?)",
        completion,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].replace(",", "").strip()
    return None


def main():
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    print(f"[smoke] model={args.model}")
    print(f"[smoke] n_problems={args.n_problems}  n_samples={args.n_samples}  "
          f"alphas={alphas}  temp={args.temperature}")

    # Load GSM8K test split
    from datasets import load_dataset
    print(f"[smoke] Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.n_problems, replace=False)
    problems = [ds[int(i)] for i in sorted(indices.tolist())]
    print(f"[smoke] Loaded {len(problems)} problems (seed={args.seed}); indices={sorted(indices.tolist())}")

    # Load model — inherits OCI byte-level BPE fix, MPS device_map="auto"
    from bench.generator import generate_samples, load_model_and_tokenizer
    from bench.sampler_bridge import make_pless_alpha_sampler

    print(f"[smoke] Loading {args.model} ... (this is the slow step on first run)")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, dtype="bfloat16")
    print(f"[smoke] Model loaded in {time.time() - t0:.1f}s on device {model.device}")

    # Generate
    out_records: list[dict] = []
    n_total = len(problems) * args.n_samples * len(alphas)
    n_done = 0
    t_start = time.time()

    # Stop when the model would start a new "Q:" exemplar — prevents
    # it from continuing the few-shot pattern after answering.
    stop_strings = ["\nQ:", "\n\nQ:"]

    for prob_idx, prob in enumerate(problems):
        question = prob["question"]
        gold = gold_answer_from_gsm8k(prob["answer"])
        prompt = WEI_2022_GSM8K_8SHOT.format(question=question)

        for alpha in alphas:
            sampler_fn = make_pless_alpha_sampler(alpha)
            try:
                completions = generate_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt,
                    sampler_fn=sampler_fn,
                    n_samples=args.n_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop_strings=stop_strings,
                )
            except Exception as exc:
                print(f"  [fail] problem={prob_idx} α={alpha}: {exc!r}")
                continue

            for sample_idx, completion in enumerate(completions):
                # generate_samples returns full text starting AFTER the prompt
                predicted = extract_predicted_answer(completion)
                correct = (predicted == gold) if predicted else False
                out_records.append({
                    "problem_idx": prob_idx,
                    "ds_index": int(indices[prob_idx]),
                    "question": question,
                    "gold_answer": gold,
                    "alpha": alpha,
                    "sample_idx": sample_idx,
                    "completion": completion,
                    "predicted_answer": predicted,
                    "correct": correct,
                    "has_python_fence": "```python" in completion,
                })

                n_done += 1
                elapsed = time.time() - t_start
                eta = (n_total - n_done) * (elapsed / max(1, n_done))
                print(f"[smoke] {n_done:3d}/{n_total}  α={alpha}  prob{prob_idx} "
                      f"sample{sample_idx}  predicted={predicted!r:>8} "
                      f"gold={gold!r:>8} {'✓' if correct else '✗'}  "
                      f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

    # Save JSONL
    out_path = Path(args.output)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records))
    print(f"\n[smoke] Wrote {len(out_records)} records to {out_path}")

    # ───── Inspection report ─────
    print("\n" + "=" * 70)
    print("SMOKE INSPECTION REPORT")
    print("=" * 70)

    n_records = len(out_records)
    if n_records == 0:
        print("No records — nothing to inspect.")
        return

    # Check 1: code-format slippage
    n_with_code = sum(1 for r in out_records if r["has_python_fence"])
    code_pct = 100 * n_with_code / n_records
    flag_code = "⚠️  HIGH" if code_pct > 10 else "✅ OK"
    print(f"\n1. Code-format slippage: {n_with_code}/{n_records} = {code_pct:.0f}%  {flag_code}")
    if code_pct > 10:
        print("    → Model is writing Python instead of reasoning. Consider switching to a non-coder model.")

    # Check 2: answer-extraction success
    n_extracted = sum(1 for r in out_records if r["predicted_answer"] is not None)
    ext_pct = 100 * n_extracted / n_records
    flag_ext = "✅ OK" if ext_pct >= 70 else "⚠️  LOW"
    print(f"\n2. Answer extraction (matches 'The answer is N.'): {n_extracted}/{n_records} = {ext_pct:.0f}%  {flag_ext}")
    if ext_pct < 70:
        print("    → Model is not following the 'The answer is N.' format reliably.")

    # Check 3: per-α accuracy
    print(f"\n3. Per-α accuracy:")
    for alpha in alphas:
        recs = [r for r in out_records if r["alpha"] == alpha]
        n_correct = sum(1 for r in recs if r["correct"])
        pct = 100 * n_correct / max(1, len(recs))
        print(f"    α={alpha}:  {n_correct}/{len(recs)} correct  (≈{pct:.0f}% pass@1)")

    # Check 4: visible variation between α arms (qualitative)
    print(f"\n4. Reasoning variation (first 120 chars of each completion):")
    for prob_idx in range(min(2, args.n_problems)):
        print(f"\n   Problem {prob_idx} (gold={out_records[0]['gold_answer'] if out_records else '?'}):")
        for alpha in alphas:
            recs = [r for r in out_records
                    if r["problem_idx"] == prob_idx and r["alpha"] == alpha]
            print(f"     α={alpha}:")
            for r in recs:
                preview = r["completion"][:120].replace("\n", " ⏎ ")
                marker = "✓" if r["correct"] else "✗"
                print(f"       {marker} {preview}...")

    print()
    print("=" * 70)
    print("Decision criteria:")
    print("  PROCEED to full sweep if:  code-slippage < 10%, extraction >= 70%,")
    print("                             pass@1 in 30-70% range, visible α-variation in #4")
    print("  RECONSIDER if any check fails — try a different model or prompt")
    print("=" * 70)


if __name__ == "__main__":
    main()
