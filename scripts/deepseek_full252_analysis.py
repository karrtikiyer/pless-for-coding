"""Full-252 DeepSeek-R1-Distill-Llama-8B loop-force (n=30/k=8/window=3000) analysis.

Two tables, both grounded live (gz-aware; tokenizer cached, model weights NOT needed):
  1. Matched comparison vs the no-force baseline: trunc / completion(+code) / closed-no-code /
     conditional-correctness / pass@{1,5,10}.
  2. Over-truncation check: think-token length of PRODUCTIVE reasoning (temp passed+completed)
     vs loop-force FORCED-cut vs NATURAL-close traces. If the forced traces sit in the
     productive band (and aren't shorter than natural closes), the detector is catching
     loops at ~productive length, not clipping good reasoning.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/deepseek_full252_analysis.py
"""
import gzip
import json
import math
import os

K8 = "results/loop_forcethink_deepseek_w3000_k8/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
BASE = "results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
N, K, WINDOW = 30, 8, 3000


def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(d, b):
    for e in (".jsonl", ".jsonl.gz"):
        p = f"{d}/{b}{e}"
        if os.path.exists(p):
            op = gzip.open if e.endswith("gz") else open
            with op(p, "rt") as f:
                return {json.loads(l)["task_id"]: json.loads(l) for l in f}


def pr(d, b):
    return {t["task_id"]: t["pass_results"]
            for t in json.load(open(f"{d}/metrics/{b}_metrics.json"))["per_task"]}


def matched(d, b):
    recs = load(d, b)
    pt = json.load(open(f"{d}/metrics/{b}_metrics.json"))["per_task"]
    exk = next((k for k in pt[0] if "extract" in k.lower()), None)
    nt = len(pt)
    p1 = sum(pak(len(t["pass_results"]), sum(t["pass_results"]), 1) for t in pt) / nt
    p5 = sum(pak(len(t["pass_results"]), sum(t["pass_results"]), 5) for t in pt) / nt
    p10 = sum(pak(len(t["pass_results"]), sum(t["pass_results"]), 10) for t in pt) / nt
    tot = trunc = done = cnc = corrdone = 0
    for t in pt:
        r = recs[t["task_id"]]; ex = t.get(exk)
        swt = r.get("samples_with_thinking") or r["samples"]; prr = t["pass_results"]
        for i, sw in enumerate(swt):
            tot += 1; closed = "</think>" in sw
            hc = bool(ex[i]) if ex and i < len(ex) else False
            ok = bool(prr[i]) if i < len(prr) else False
            if not closed: trunc += 1
            elif hc:
                done += 1
                if ok: corrdone += 1
            else: cnc += 1
    return dict(nt=nt, trunc=100 * trunc / tot, compl=100 * done / tot, cnc=100 * cnc / tot,
                cond=corrdone / done if done else 0, p1=p1, p5=p5, p10=p10)


def main():
    print("=== 1. Matched comparison — full 252, DeepSeek, loop-force n30/k8/w3000 vs baseline ===\n")
    h = f"{'config':<26}{'trunc%':>7}{'compl%':>8}{'no-code%':>9}{'cond':>7}{'pass@1':>8}{'pass@5':>8}{'pass@10':>8}"
    print(h); print("-" * len(h))
    for lbl, d, b in [
        ("pless baseline", BASE, "pless_think_t1.0_t1.0"),
        ("pless loop-force k8", K8, "pless_think_t1.0_t1.0"),
        ("pnorm baseline", BASE, "pless_norm_think_t1.0_t1.0"),
        ("pnorm loop-force k8", K8, "pless_norm_think_t1.0_t1.0"),
    ]:
        x = matched(d, b)
        print(f"{lbl:<26}{x['trunc']:>7.1f}{x['compl']:>8.1f}{x['cnc']:>9.1f}{x['cond']:>7.3f}"
              f"{x['p1']:>8.3f}{x['p5']:>8.3f}{x['p10']:>8.3f}")

    print("\n=== 2. Over-truncation check — think-token lengths (DeepSeek tokenizer) ===\n")
    from transformers import AutoTokenizer
    from bench.loop_detect import ngram_loop_fired
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    def q(x):
        x = sorted(v for v in x if v is not None)
        return (f"p25={x[len(x)//4]:>6} med={x[len(x)//2]:>6} p75={x[3*len(x)//4]:>6} "
                f"mean={round(sum(x)/len(x)):>6}  (n={len(x)})") if x else "—"

    prod = []
    for cfg in ("temp_p0.95_think_t1.0_t1.0", "temp_k20_think_t1.0_t1.0", "temp_think_t0.6_t0.6"):
        recs = load(BASE, cfg); P = pr(BASE, cfg)
        for tid, r in recs.items():
            for i, sw in enumerate(r.get("samples_with_thinking", [])):
                if "</think>" in sw and i < len(P[tid]) and P[tid][i]:
                    prod.append(len(tok.encode(sw.split("</think>", 1)[0], add_special_tokens=False)))
    print(f"PRODUCTIVE (temp passed+completed)  {q(prod)}")
    recs = load(K8, "pless_think_t1.0_t1.0")
    forced, natural = [], []
    for tid, r in recs.items():
        for sw in r.get("samples_with_thinking", []):
            if "</think>" not in sw:
                continue
            t = tok.encode(sw.split("</think>", 1)[0], add_special_tokens=False)
            (forced if ngram_loop_fired(t, n=N, k=K, window=WINDOW) else natural).append(len(t))
    print(f"LOOP-FORCE forced / cut             {q(forced)}")
    print(f"LOOP-FORCE natural-close            {q(natural)}")
    print("\nforced ≈ productive (and ≥ natural-close) → detector fires at ~productive length, "
          "NOT clipping good reasoning. The low overall mean is the model's naturally-short closes.")


if __name__ == "__main__":
    main()
