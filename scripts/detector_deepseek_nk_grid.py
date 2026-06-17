"""DeepSeek (n,k) detector grid at window=3000 — NO GPU.

Confirms the loop-force k for DeepSeek-R1-Distill-Llama-8B (carried over from Qwen3 as
k=6 without a DeepSeek-specific check). Measures, per k at fixed n=30/window=3000:
  - FP% on genuine productive reasoning (temp completed+passed) — firing here clips good
    reasoning; want ~0.
  - catch% on truncated/looping traces — want high.
  - median fire position on truncated — higher k fires LATER (more repeats needed), so it
    can avoid clipping the productive 5-9K band.

Finding (2026-06-17): k=6 has 1.5% FP / 98.5% catch; k=8 → 0.0% FP / 93.5% catch (fire pos
barely moves: DeepSeek loops onset early ~5K regardless). => use k=8 for the DeepSeek run.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/detector_deepseek_nk_grid.py
"""
import gzip
import json
import os
from collections import Counter
from transformers import AutoTokenizer

DS = "results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
WINDOW = 3000
STEP = 200
N = 30
KS = [6, 8, 10, 12, 16]
N_SAMPLE = 200


def fire_pos(toks, k, n=N):
    if len(toks) < n:
        return None
    for end in range(n, len(toks) + 1, STEP):
        t = toks[max(0, end - WINDOW):end]
        if len(t) < n:
            continue
        if max(Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1)).values()) >= k:
            return end
    return None


def med(x):
    x = sorted(v for v in x if v is not None)
    return x[len(x) // 2] if x else None


def load(b):
    for ext in (".jsonl", ".jsonl.gz"):
        p = f"{DS}/{b}{ext}"
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return [json.loads(l) for l in f]


def pr(b):
    return {t["task_id"]: t["pass_results"]
            for t in json.load(open(f"{DS}/metrics/{b}_metrics.json"))["per_task"]}


def stride(xs, k):
    if len(xs) <= k:
        return xs
    s = len(xs) / k
    return [xs[int(i * s)] for i in range(k)]


def main():
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    trunc, succ = [], []
    for r in load("pless_think_t1.0_t1.0"):
        for sw in r.get("samples_with_thinking", []):
            if "</think>" not in sw:
                trunc.append(sw)
    for cfg in ("temp_p0.95_think_t1.0_t1.0", "temp_k20_think_t1.0_t1.0", "temp_think_t0.6_t0.6"):
        P = pr(cfg)
        for r in load(cfg):
            for i, sw in enumerate(r.get("samples_with_thinking", [])):
                if "</think>" in sw and i < len(P[r["task_id"]]) and P[r["task_id"]][i]:
                    succ.append(sw.split("</think>", 1)[0])
    trunc = [tok.encode(s, add_special_tokens=False) for s in stride(trunc, N_SAMPLE)]
    succ = [tok.encode(s, add_special_tokens=False) for s in stride(succ, N_SAMPLE)]
    print(f"DeepSeek n={N}, window={WINDOW} | {len(succ)} productive-passed, {len(trunc)} truncated\n")
    print(f"{'k':>3} | {'FP% (good fire)':>15} | {'catch% (loops)':>15} | {'med fire-pos (trunc)':>21}")
    print("-" * 64)
    for k in KS:
        fp = sum(fire_pos(t, k) is not None for t in succ) / len(succ) * 100
        tf = [fire_pos(t, k) for t in trunc]
        ca = sum(x is not None for x in tf) / len(trunc) * 100
        print(f"{k:>3} | {fp:>14.1f}% | {ca:>14.1f}% | {str(med(tf)):>21}")
    print("\nHigher k = fire LATER (more repeats needed). k=8 zeroes FP at ~5pp catch cost.")


if __name__ == "__main__":
    main()
