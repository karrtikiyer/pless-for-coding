"""Offline detector tuning for DeepSeek-R1-Distill-Llama-8B (NO GPU).

DeepSeek pless truncates 64.9% on ATCODER_interview. This sweeps window (and a small
n/k grid) on DeepSeek-tokenized traces to find the loop-force operating point, the same
way we did for Qwen3 — FP on productive reasoning vs catch on truncated/looping traces.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/detector_deepseek_sweep.py
"""
import json
import statistics as st
from collections import Counter
from transformers import AutoTokenizer

DS = "results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
N = 30
K = 6
STEP = 200
WINDOWS = [400, 800, 1200, 1600, 2000, 3000, 4000]
N_TRUNC = 200      # sample sizes (traces are ~33K tokens -> cap for runtime)
N_SUCC = 400


def fires(toks, window, n=N, k=K, step=STEP):
    if len(toks) < n:
        return False
    for end in range(n, len(toks) + 1, step):
        t = toks[max(0, end - window):end]
        if len(t) < n:
            continue
        if max(Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1)).values()) >= k:
            return True
    return False


def main():
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    # truncated pless samples = the looping ones (no closing </think>)
    trunc_txt = []
    for line in open(f"{DS}/pless_think_t1.0_t1.0.jsonl"):
        r = json.loads(line)
        for sw in r.get("samples_with_thinking", []):
            if "</think>" not in sw:
                trunc_txt.append(sw)

    # successful productive reasoning (completed + passed) from the temp configs — FP must stay ~0
    succ_txt = []
    for cfg in ("temp_k20_think_t1.0_t1.0", "temp_p0.95_think_t1.0_t1.0", "temp_think_t0.6_t0.6"):
        m = json.load(open(f"{DS}/metrics/{cfg}_metrics.json"))
        passed = {t["task_id"]: t["pass_results"] for t in m["per_task"]}
        for line in open(f"{DS}/{cfg}.jsonl"):
            r = json.loads(line)
            pr = passed.get(r["task_id"], [])
            for i, sw in enumerate(r.get("samples_with_thinking", [])):
                if i < len(pr) and pr[i] and "</think>" in sw:
                    succ_txt.append(sw.split("</think>", 1)[0])

    # deterministic subsample (no RNG): even stride
    def stride(xs, k):
        if len(xs) <= k:
            return xs
        step = len(xs) / k
        return [xs[int(i * step)] for i in range(k)]

    trunc = [tok.encode(s, add_special_tokens=False) for s in stride(trunc_txt, N_TRUNC)]
    succ = [tok.encode(s, add_special_tokens=False) for s in stride(succ_txt, N_SUCC)]
    print(f"DeepSeek-R1-Distill-Llama-8B | n={N} k={K} step={STEP}")
    print(f"truncated sampled {len(trunc)} (of {len(trunc_txt)}), success sampled {len(succ)} (of {len(succ_txt)})")
    print(f"trunc median len={int(st.median([len(t) for t in trunc]))} tok, "
          f"success median len={int(st.median([len(t) for t in succ]))} tok\n")

    print(f"{'window':>8} | {'FP% (good cut)':>14} | {'catch% (loops)':>15}")
    print("-" * 44)
    for w in WINDOWS:
        fp = sum(fires(t, w) for t in succ) / len(succ) * 100
        ca = sum(fires(t, w) for t in trunc) / len(trunc) * 100
        print(f"{w:>8} | {fp:>13.1f}% | {ca:>14.1f}%")
    print("\nFP% = % of GOOD reasoning wrongly cut (keep ~0). catch% = % of looping traces caught.")
    print("Compare to Qwen3 (window=1200 -> 2.2% FP / 97% catch). DeepSeek periods may differ.")


if __name__ == "__main__":
    main()
