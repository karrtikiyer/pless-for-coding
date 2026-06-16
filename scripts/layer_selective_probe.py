"""Selective latent idea: intervene ONLY at high-final-entropy 'decision-point' tokens,
using the cross-layer-PERSISTENT candidate set across layers 24-29 (junk is transient →
filtered by requiring a token to survive across layers). Deterministic tokens use the
final layer (which is what pless already does).

Per think-phase position, split by FINAL-layer Shannon H into buckets, and within each:
  - final_eff#tok        : does the FINAL layer already give diversity here? (pless is
                           entropy-adaptive, so high-H tokens already keep many)
  - persist_set_size@M   : # tokens in top-k of >= M of layers 24-29 (cross-layer stable)
  - agree(final_top1 ∈ persist_set) : does the stable latent set keep the model's choice?
  - agree(L30 single)    : reference

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/layer_selective_probe.py
Env: TASK_IDS, MAX_THINK_TOK(1200), N_POS(120), TOPK(20), PERSIST_M(4)
"""
import json
import math
import os
import torch
from collections import defaultdict

from bench.generator import load_model_and_tokenizer
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
TASK_IDS = [int(x) for x in os.environ.get("TASK_IDS", "1226 930 558 1126").split()]
MAX_THINK_TOK = int(os.environ.get("MAX_THINK_TOK", "1200"))
N_POS = int(os.environ.get("N_POS", "120"))
TOPK = int(os.environ.get("TOPK", "20"))
PERSIST_M = int(os.environ.get("PERSIST_M", "4"))      # token must be in top-k of >=M of layers 24-29
BAND = list(range(24, 30))                              # 24..29 (before the L30 near-collapse)
# final-layer Shannon-H buckets (nats): deterministic vs decision points
BUCKETS = [(0.0, 0.1, "det <0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1.5, "0.5-1.5"),
           (1.5, 3.0, "1.5-3.0"), (3.0, 99, "decision >3.0")]


def bucket(h):
    for lo, hi, name in BUCKETS:
        if lo <= h < hi:
            return name
    return BUCKETS[-1][2]


def main():
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tok = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    model.eval()
    fn, lm = model.model.norm, model.lm_head
    L = model.config.num_hidden_layers

    traces = {}
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] in TASK_IDS and r["task_id"] not in traces:
                traces[r["task_id"]] = r["truncated_solution"]

    # per-bucket accumulators
    agg = defaultdict(lambda: {"n": 0, "final_eff": 0.0, "final_surv": 0.0,
                               "persist_size": 0.0, "agree_persist": 0, "agree_L30": 0})

    for tid in TASK_IDS:
        if tid not in traces:
            continue
        problem = pmap[tid]
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        pre = tok.encode(prefix, return_tensors="pt")[0]
        th = tok.encode(traces[tid], add_special_tokens=False)[:MAX_THINK_TOK]
        ids = torch.cat([pre, torch.tensor(th)]).unsqueeze(0).to(model.device)
        plen, seq = len(pre), ids.size(1)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        lo, hi = plen - 1, seq - 2
        g = torch.Generator().manual_seed(0)
        pos = torch.randint(lo, hi, (min(N_POS, hi - lo),), generator=g).unique().tolist()

        for p in pos:
            fl = lm(fn(hs[L][0, p:p+1]))[0].float()       # final logits
            fp = torch.softmax(fl, -1)
            fH = float(-(fp * fp.clamp_min(1e-12).log()).sum())
            fsig2 = float((fp * fp).sum())
            ftop1 = int(fl.argmax())
            final_surv = int((fp >= fsig2).sum())          # pless survivors on FINAL layer
            b = bucket(fH)
            a = agg[b]; a["n"] += 1
            a["final_eff"] += math.exp(fH)
            a["final_surv"] += final_surv
            # cross-layer persistence across 24-29
            cnt = defaultdict(int)
            for ell in BAND:
                tk = lm(fn(hs[ell][0, p:p+1]))[0].topk(TOPK).indices.tolist()
                for t in tk:
                    cnt[t] += 1
            persist = {t for t, c in cnt.items() if c >= PERSIST_M}
            a["persist_size"] += len(persist)
            a["agree_persist"] += int(ftop1 in persist)
            l30 = lm(fn(hs[30][0, p:p+1]))[0].topk(10).indices.tolist()
            a["agree_L30"] += int(ftop1 in l30)
        del hs, out
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"task {tid}: {len(pos)} positions", flush=True)

    total = sum(a["n"] for a in agg.values())
    print(f"\n=== by FINAL-layer entropy bucket ({total} positions; persist=top{TOPK} in >={PERSIST_M}/6 of L24-29) ===")
    print(f"{'final-H bucket':<16} {'n':>5} {'%':>5} {'final_eff#tok':>13} {'final_pless_surv':>16} "
          f"{'persist_set':>11} {'agree_persist':>13} {'agree_L30':>9}")
    for lo, hi, name in BUCKETS:
        a = agg.get(name)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        print(f"{name:<16} {n:>5} {100*n/total:>4.0f}% {a['final_eff']/n:>13.1f} "
              f"{a['final_surv']/n:>16.1f} {a['persist_size']/n:>11.1f} "
              f"{a['agree_persist']/n:>13.2f} {a['agree_L30']/n:>9.2f}")
    print("\nRead: 'decision' buckets (high final-H) = where diversity matters. If final_pless_surv")
    print("is already >1 there, pless-on-final ALREADY gives diversity (latent redundant). If")
    print("agree_persist is high there, the cross-layer-stable set keeps the right token (idea viable).")


if __name__ == "__main__":
    main()
