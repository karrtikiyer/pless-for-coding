"""Consolidated offline (n,k,window) sweep for BOTH models — choose the best detector
config for the chop->a5 live rescue. NO GPU. Reports, per (n,k,window):
  catch% = fraction of LOOPED traces (no </think>) where some n-gram repeats >=k in-window
  FP%    = fraction of PRODUCTIVE traces (closed </think>) that would (wrongly) fire
Productive proxy = closed-</think> samples (reasoning that terminated on its own).

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/detector_config_choose.py
"""
import gzip
import json
from collections import Counter

from transformers import AutoTokenizer


def load_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f]


def split_traces(path, cap=160):
    loops, prod = [], []
    for r in load_rows(path):
        for s in r.get("samples_with_thinking", []):
            (loops if "</think>" not in s else prod).append(s)

    def take(xs):
        return xs[:: max(1, len(xs) // cap)][:cap]

    return take(loops), take(prod)


def peak(tokens, n, window, step=256):
    if len(tokens) < n:
        return 0
    best = 0
    for end in range(n, len(tokens) + 1, step):
        t = tokens[max(0, end - window):end]
        if len(t) < n:
            continue
        c = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
        m = max(c.values())
        if m > best:
            best = m
    return best


def sweep(name, tok_id, path, grid_n, grid_k, grid_w):
    tok = AutoTokenizer.from_pretrained(tok_id)
    loops, prod = split_traces(path)
    Ltok = [tok.encode(s) for s in loops]
    Ptok = [tok.encode(s) for s in prod]
    print(f"\n===== {name}: {len(Ltok)} looped, {len(Ptok)} productive (proxy) =====")
    print(f"{'n':>3} {'window':>6} {'k':>3} {'catch%':>7} {'FP%':>6}")
    for n in grid_n:
        for w in grid_w:
            Lpk = [peak(t, n, w) for t in Ltok]
            Ppk = [peak(t, n, w) for t in Ptok]
            for k in grid_k:
                catch = 100 * sum(p >= k for p in Lpk) / max(1, len(Lpk))
                fp = 100 * sum(p >= k for p in Ppk) / max(1, len(Ppk))
                print(f"{n:>3} {w:>6} {k:>3} {catch:>7.1f} {fp:>6.1f}")


sweep("Qwen3-8B", "Qwen/Qwen3-8B",
      "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl",
      grid_n=[30], grid_k=[4, 5, 6, 8], grid_w=[800, 1200, 1600])
sweep("DeepSeek-R1-Distill", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
      "results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl",
      grid_n=[30, 40], grid_k=[6, 8, 10], grid_w=[1200, 3000, 4000])
