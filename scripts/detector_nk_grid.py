"""Offline (n,k) grid for the loop-force detector — NO GPU, NO regeneration.

For each candidate (n,k): false-positive rate on genuine PRODUCTIVE reasoning
(firing here = cutting good reasoning -> cond-correctness loss) vs catch rate on
TRUNCATED/looping traces (firing here = the truncation we want to prevent).

Optimization vs detector_falsepos_check.py: for a fixed n we compute, per trace, the
PEAK n-gram repetition count over any window (step-sampled). A given k fires iff
peak >= k, so one pass answers every k. Window locality (last `WINDOW` tokens) is
preserved via overlapping windows (STEP < WINDOW).

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/detector_nk_grid.py
"""
import json
from collections import Counter
from transformers import AutoTokenizer

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
UNSOLVABLE = {117, 280, 326, 370, 454, 455, 512, 661, 962, 1122, 1175, 1223, 1368}
WINDOW = 400
STEP = 200                      # overlapping windows; coarse enough to be fast, fine enough for a grid
GRID_N = [12, 16, 20, 24, 30, 40]
GRID_K = [4, 5, 6, 8]


def peak_count(tokens, n, window=WINDOW, step=STEP):
    """Max count of any n-gram within any `window`-token span (step-sampled)."""
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
            if best >= max(GRID_K):   # already fires for every k -> stop early
                return best
    return best


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # truncated/looping traces — detector SHOULD fire (these cause the truncation)
    trunc = []
    for line in open(f"{POD}/truncated_cases.jsonl"):
        r = json.loads(line)
        if r["task_id"] not in UNSOLVABLE:
            trunc.append(tok.encode(r["truncated_solution"], add_special_tokens=False))
    trunc_tids = {json.loads(l)["task_id"] for l in open(f"{POD}/truncated_cases.jsonl")}
    hard = trunc_tids - UNSOLVABLE

    # successful productive reasoning — detector MUST NOT fire (firing = cut good reasoning)
    success = []
    for cfg in ("temp_k20_think_t1.0_t1.0", "temp_p0.95_think_t1.0_t1.0", "temp_think_t0.6_t0.6"):
        m = json.load(open(f"{CANON}/metrics/{cfg}_metrics.json"))
        passed = {t["task_id"]: t["pass_results"] for t in m["per_task"]}
        for line in open(f"{CANON}/{cfg}.jsonl"):
            r = json.loads(line)
            if r["task_id"] not in hard:
                continue
            pr = passed.get(r["task_id"], [])
            for i, sw in enumerate(r.get("samples_with_thinking", [])):
                if i < len(pr) and pr[i] and "</think>" in sw:
                    success.append(tok.encode(sw.split("</think>", 1)[0], add_special_tokens=False))

    print(f"traces: {len(success)} productive-success (FP target ~0), {len(trunc)} truncated (catch target high)")
    print(f"(window={WINDOW}, step={STEP})\n")

    # peak count per trace per n (the expensive part, done once)
    succ_peak = {n: [peak_count(t, n) for t in success] for n in GRID_N}
    trunc_peak = {n: [peak_count(t, n) for t in trunc] for n in GRID_N}

    print(f"{'(n,k)':>8} | " + " | ".join(f"k={k}".center(13) for k in GRID_K))
    print(f"{'':>8} | " + " | ".join("FP% / catch%".center(13) for _ in GRID_K))
    print("-" * (10 + 16 * len(GRID_K)))
    for n in GRID_N:
        cells = []
        for k in GRID_K:
            fp = sum(p >= k for p in succ_peak[n]) / len(success) * 100
            ca = sum(p >= k for p in trunc_peak[n]) / len(trunc) * 100
            cells.append(f"{fp:>4.1f}/{ca:>5.1f}")
        print(f"{('n='+str(n)):>8} | " + " | ".join(c.center(13) for c in cells))
    print("\nFP% = % of GOOD reasoning wrongly cut (keep ~0). catch% = % of looping traces caught (higher = less truncation).")
    print("Current setting n=30/k=6. A better setting has HIGHER catch% at FP% still ~0.")


if __name__ == "__main__":
    main()
