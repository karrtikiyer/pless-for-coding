"""Test weighted combinations of layers 24-30 (vs single layers) — does blending escape
the single-layer diversity-vs-agreement tradeoff?

For each blend scheme, on real think-phase positions across tasks, report:
  - eff #tokens (eᴴ of the blended dist) — diversity
  - agree@k: fraction of positions where the FINAL layer's top-1 token is in the blend's
    top-k — does the blend KEEP the model's correct token?
Bar to beat a single layer: high agree@10 (~>0.8, keeps the right token) AND meaningful
diversity (~3-12 eff tokens). If no blend clears it, combining doesn't help.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/layer_blend_probe.py
Env: TASK_IDS("1226 930 558 1126"), MAX_THINK_TOK(1200), N_POS(120)
"""
import json
import math
import os
import torch

from bench.generator import load_model_and_tokenizer
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
TASK_IDS = [int(x) for x in os.environ.get("TASK_IDS", "1226 930 558 1126").split()]
MAX_THINK_TOK = int(os.environ.get("MAX_THINK_TOK", "1200"))
N_POS = int(os.environ.get("N_POS", "120"))
BAND = list(range(24, 31))  # layers 24..30

# weight schemes over BAND (prob-space mixtures unless name starts with 'logit')
def schemes(band):
    n = len(band)
    up = [(l - band[0] + 1) for l in band]          # favor later layers (correctness)
    down = [(band[-1] - l + 1) for l in band]        # favor earlier layers (diversity)
    norm = lambda w: [x / sum(w) for x in w]
    return {
        "single_L26":      {26: 1.0},
        "single_L30":      {30: 1.0},
        "uniform_24_30":   {l: 1.0 / n for l in band},
        "linUP_24_30":     {l: w for l, w in zip(band, norm(up))},    # weight to later
        "linDOWN_24_30":   {l: w for l, w in zip(band, norm(down))},  # weight to earlier
        "later_28_30":     {l: 1.0 / 3 for l in (28, 29, 30)},
        "logitavg_24_30":  {l: 1.0 / n for l in band},               # logit-space (flagged below)
    }


def main():
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tok = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    model.eval()
    final_norm, lm_head = model.model.norm, model.lm_head
    n_layers = model.config.num_hidden_layers

    traces = {}
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] in TASK_IDS and r["task_id"] not in traces:
                traces[r["task_id"]] = r["truncated_solution"]

    SCH = schemes(BAND)
    acc = {name: {"effsum": 0.0, 1: 0, 5: 0, 10: 0} for name in SCH}
    npos_total = 0

    def lens(h):  # logit lens
        return lm_head(final_norm(h))

    for tid in TASK_IDS:
        if tid not in traces:
            continue
        problem = pmap[tid]
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        pre = tok.encode(prefix, return_tensors="pt")[0]
        th = tok.encode(traces[tid], add_special_tokens=False)[:MAX_THINK_TOK]
        input_ids = torch.cat([pre, torch.tensor(th)]).unsqueeze(0).to(model.device)
        plen, seq = len(pre), input_ids.size(1)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        lo, hi = plen - 1, seq - 2
        g = torch.Generator().manual_seed(0)
        pos = torch.randint(lo, hi, (min(N_POS, hi - lo),), generator=g).unique().to(model.device)
        npos_total += len(pos)

        final_top1 = lens(hs[n_layers][0, pos]).argmax(-1)        # (P,)
        # precompute per-layer probs + logits for the band (reused across schemes)
        probs = {l: torch.softmax(lens(hs[l][0, pos]).float(), -1) for l in BAND}
        logits_band = {l: lens(hs[l][0, pos]).float() for l in BAND}

        for name, w in SCH.items():
            if name.startswith("logit"):
                blended_logits = sum(w[l] * logits_band[l] for l in w)
                pb = torch.softmax(blended_logits, -1)
            else:
                pb = sum(w[l] * probs[l] for l in w)              # prob-space mixture (weights sum to 1)
            H = -(pb * pb.clamp_min(1e-12).log()).sum(-1)
            acc[name]["effsum"] += torch.exp(H).sum().item()
            topk = pb.topk(10, dim=-1).indices
            hit = (topk == final_top1.unsqueeze(1))
            acc[name][1] += hit[:, :1].any(1).sum().item()
            acc[name][5] += hit[:, :5].any(1).sum().item()
            acc[name][10] += hit.any(1).sum().item()
        del hs, out, probs, logits_band
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"task {tid}: {len(pos)} positions", flush=True)

    print(f"\n=== BLEND SCHEMES over {len(TASK_IDS)} tasks, {npos_total} positions (layers {BAND[0]}-{BAND[-1]}) ===")
    print(f"{'scheme':<16} {'~eff#tok':>9} {'agree@1':>8} {'agree@5':>8} {'agree@10':>9}")
    for name in SCH:
        a = acc[name]
        print(f"{name:<16} {a['effsum']/npos_total:>9.1f} {a[1]/npos_total:>8.2f} "
              f"{a[5]/npos_total:>8.2f} {a[10]/npos_total:>9.2f}")
    print("\nBar: a blend WINS only if agree@10 >> single-layer (L26~0.47 / L30~0.67) AND "
          "eff#tok stays meaningful (~3-12). Else combining doesn't escape the tradeoff.")


if __name__ == "__main__":
    main()
