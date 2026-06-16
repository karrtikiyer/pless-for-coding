"""Derive a NON-overfitted min-think-token floor for loop-force, from data.

Compares two token-length distributions on the HARD truncated tasks:
  (1) LOOP ONSET in pless/pless_norm truncated traces — where repetition starts
      (the detector must be able to fire at/after this).
  (2) NATURAL successful-reasoning length — think-tokens of samples that closed
      </think> AND passed, from the non-truncating temp configs (clean completions).
Floor should sit ABOVE most of (2) (don't cut productive reasoning) while (1) extends
beyond it (loops still caught after the floor). If they overlap, position can't separate.

Tokenizer only — no model, no GPU.
Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/min_think_floor_analysis.py
"""
import json
import gzip
import os
from transformers import AutoTokenizer

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"   # truncated traces source
CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
UNSOLVABLE = {117, 280, 326, 370, 454, 455, 512, 661, 962, 1122, 1175, 1223, 1368}


def find_loop(text, chunk=120, min_repeat=4):
    for s in range(0, max(1, len(text) - chunk * min_repeat), 80):
        if text[s:].count(text[s:s + chunk]) >= min_repeat:
            return s
    return None


def pct(xs, ps=(10, 25, 50, 75, 90, 95)):
    xs = sorted(xs)
    if not xs:
        return {p: None for p in ps}
    return {p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in ps}


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # (1) loop-onset token positions in truncated traces (pless + pless_norm)
    onsets = []
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["task_id"] in UNSOLVABLE:
                continue
            c = find_loop(r["truncated_solution"])
            if c is not None:
                onsets.append(len(tok.encode(r["truncated_solution"][:c], add_special_tokens=False)))

    # (2) natural successful think-lengths on the SAME hard tasks, from non-truncating temp configs
    trunc_tids = set()
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            trunc_tids.add(json.loads(line)["task_id"])
    hard = trunc_tids - UNSOLVABLE

    success_len = []
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
                        think = swt.split("</think>", 1)[0]
                        success_len.append(len(tok.encode(think, add_special_tokens=False)))

    po = pct(onsets); ps = pct(success_len)
    print(f"=== LOOP ONSET tokens (truncated traces, n={len(onsets)}) ===")
    print("  " + "  ".join(f"p{p}={po[p]}" for p in (10, 25, 50, 75, 90, 95)))
    print(f"\n=== NATURAL SUCCESSFUL think-tokens on the SAME hard tasks (temp configs, n={len(success_len)}) ===")
    print("  " + "  ".join(f"p{p}={ps[p]}" for p in (10, 25, 50, 75, 90, 95)))

    print("\n=== floor candidates (protect productive reasoning AND still catch loops) ===")
    for floor in (2000, 4000, 6000, 8000, 10000, 12000):
        cut_success = sum(1 for x in success_len if x > floor)   # successful chains the floor would NOT have cut early
        # (a successful chain longer than floor is fine — detector only fires on a loop, and a
        #  successful chain has no loop; the risk is firing on a *productive* chain's incidental
        #  repetition before it finishes — so we want floor ABOVE most success lengths)
        below = sum(1 for x in success_len if x <= floor) / max(1, len(success_len))
        loops_after = sum(1 for x in onsets if x >= floor) / max(1, len(onsets))
        print(f"  floor={floor:>6}: {below*100:>4.0f}% of successful reasoning finishes by here "
              f"(safe to allow firing); {loops_after*100:>4.0f}% of loop onsets are >= floor (still catchable)")


if __name__ == "__main__":
    main()
