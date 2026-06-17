"""Three-way completion breakdown — the 'closed-but-no-code' diagnostic.

`cot_efficiency` defines completed = (closed </think> AND code extracted) and
truncated = (not closed), so they do NOT sum to 100 — the gap is samples that closed
</think> but produced no extractable code. That gap is the clean diagnostic for whether
the model HAD a solution when it stopped thinking: tiny for a model that can write code
when interrupted (Qwen3 ~0.4%), large for one whose loops are dead-ends (DeepSeek ~34-41%).

Per config: truncated% (no </think>) + closed-no-code% (</think>, extraction failed) +
completed% (</think> + code) = 100. Uses the metrics `extraction_success` list; gz-aware.

Run: PYTHONPATH=. uv run python scripts/completion_breakdown.py
"""
import gzip
import json
import os

QW = "results/loop_forcethink_qwen_w1200/Qwen--Qwen3-8B/ATCODER_interview"
QCANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
DS3K = "results/loop_forcethink_deepseek_w3000/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
DSCANON = "results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"

IDS25 = set([117, 160, 230, 257, 270, 280, 325, 326, 341, 369, 370, 417, 454, 455, 512,
             527, 541, 558, 559, 579, 587, 588, 615, 616, 661])

# (display, dir, base, id_filter_or_None)
CONFIGS = [
    ("Qwen3 pless  w1200 (full 252)",  QW,    "pless_think_t1.0_t1.0",      None),
    ("Qwen3 pless  w1200 (same 25)",   QW,    "pless_think_t1.0_t1.0",      IDS25),
    ("Qwen3 pless  BASE  (full 252)",  QCANON, "pless_think_t1.0_t1.0",     None),
    ("DeepSeek pless w3000 (25 smoke)", DS3K, "pless_think_t1.0_t1.0",      IDS25),
    ("DeepSeek pnorm w3000 (25 smoke)", DS3K, "pless_norm_think_t1.0_t1.0", IDS25),
    ("DeepSeek pless BASE  (full 252)", DSCANON, "pless_think_t1.0_t1.0",   None),
]


def load(d, base):
    for ext in (".jsonl", ".jsonl.gz"):
        p = f"{d}/{base}{ext}"
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return {json.loads(l)["task_id"]: json.loads(l) for l in f}
    return None


def breakdown(d, base, idf):
    recs = load(d, base)
    pt = json.load(open(f"{d}/metrics/{base}_metrics.json"))["per_task"]
    exk = next((k for k in pt[0] if "extract" in k.lower()), None)
    n = tr = cnc = comp = 0
    for t in pt:
        if idf and t["task_id"] not in idf:
            continue
        if t["task_id"] not in recs:
            continue
        r = recs[t["task_id"]]
        ex = t.get(exk)
        for i, sw in enumerate(r.get("samples_with_thinking", [])):
            n += 1
            closed = "</think>" in sw
            hc = bool(ex[i]) if ex and i < len(ex) else False
            if not closed:
                tr += 1
            elif hc:
                comp += 1
            else:
                cnc += 1
    return n, 100 * tr / n, 100 * cnc / n, 100 * comp / n


def main():
    print(f"{'config':<34}{'n':>6}{'trunc%':>8}{'closed-no-code%':>17}{'completed%':>12}")
    print("-" * 77)
    for name, d, base, idf in CONFIGS:
        if not os.path.exists(f"{d}/metrics/{base}_metrics.json") or load(d, base) is None:
            print(f"{name:<34}  (missing)")
            continue
        n, tr, cnc, comp = breakdown(d, base, idf)
        print(f"{name:<34}{n:>6}{tr:>8.1f}{cnc:>17.1f}{comp:>12.1f}")
    print("\nclosed-no-code% = closed </think> but produced no extractable code — the clean")
    print("diagnostic for whether the model had a solution when it stopped thinking.")


if __name__ == "__main__":
    main()
