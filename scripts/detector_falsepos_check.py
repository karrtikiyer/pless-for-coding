"""Validate detector params (n,k) WITHOUT generation: does n=30/k=6 false-fire on real
PRODUCTIVE reasoning, and does it fire LATE on genuine loops? Tokenizer only, no GPU.

False-positive rate = fraction of SUCCESSFUL reasoning traces (completed </think> + passed,
on the hard tasks) where the detector fires at all. A good detector ≈ 0% here.
On TRUNCATED traces it should fire, and late (near genuine-loop onset, not ~1.5K).

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/detector_falsepos_check.py
"""
import json
import os
from collections import Counter
from transformers import AutoTokenizer

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
UNSOLVABLE = {117, 280, 326, 370, 454, 455, 512, 661, 962, 1122, 1175, 1223, 1368}
PARAMS = [(8, 4), (30, 6), (30, 20)]   # broken / proposed / literature
WINDOW = 400
STEP = 25                               # scan throttle (matches runtime _LOOP_CHECK_EVERY ballpark)


def fire_position(tokens, n, k, window=WINDOW, step=STEP):
    """First token index at which an n-gram recurs >=k times in the last `window`, else None."""
    for end in range(n * k, len(tokens) + 1, step):
        t = tokens[max(0, end - window):end]
        if len(t) < n:
            continue
        c = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
        if c and max(c.values()) >= k:
            return end
    return None


def pct(xs, ps=(50, 90)):
    xs = sorted(x for x in xs if x is not None)
    return {p: (xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else None) for p in ps}


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # truncated traces (should fire, late)
    trunc = []
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["task_id"] not in UNSOLVABLE:
                trunc.append(tok.encode(r["truncated_solution"], add_special_tokens=False))

    # successful PRODUCTIVE reasoning (must NOT fire) — temp configs, hard tasks, passed + </think>
    trunc_tids = {json.loads(l)["task_id"] for l in open(f"{POD}/truncated_cases.jsonl")}
    hard = trunc_tids - UNSOLVABLE
    success = []
    for cfg in ("temp_k20_think_t1.0_t1.0", "temp_p0.95_think_t1.0_t1.0", "temp_think_t0.6_t0.6"):
        m = json.load(open(f"{CANON}/metrics/{cfg}_metrics.json"))
        passed = {t["task_id"]: t["pass_results"] for t in m["per_task"]}
        with open(f"{CANON}/{cfg}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["task_id"] not in hard:
                    continue
                pr = passed.get(r["task_id"], [])
                for i, swt in enumerate(r.get("samples_with_thinking", [])):
                    if i < len(pr) and pr[i] and "</think>" in swt:
                        success.append(tok.encode(swt.split("</think>", 1)[0], add_special_tokens=False))

    print(f"traces: {len(success)} successful-productive, {len(trunc)} truncated\n")
    print(f"{'(n,k)':>8} | {'FALSE-POS on success':>22} | {'fires on truncated':>20} | {'trunc fire pos p50/p90':>24}")
    print("-" * 86)
    for n, k in PARAMS:
        fp = [fire_position(t, n, k) for t in success]
        fp_rate = sum(x is not None for x in fp) / len(success)
        tf = [fire_position(t, n, k) for t in trunc]
        tf_rate = sum(x is not None for x in tf) / len(trunc)
        fpp = pct([x for x in tf if x is not None])
        print(f"{f'({n},{k})':>8} | {fp_rate*100:>20.1f}% | {tf_rate*100:>18.1f}% | "
              f"{str(fpp[50]):>10} / {str(fpp[90]):<10}")
    print("\nGood detector: ~0% false-pos on productive reasoning, high fire-rate on truncated, "
          "and trunc fire pos NOT ~1.5K (the broken n=8/k=4 runtime value).")


if __name__ == "__main__":
    main()
