"""Logit-lens entropy profile across Qwen3-8B layers on real think-phase tokens,
aggregated over several tasks, with a quantified "agreement" (noise) metric.

For each layer ℓ: project hidden state through final-norm + LM head (logit lens),
compute Σpᵢ² (collision energy; what pless thresholds) and Shannon H. Aggregate over
all sampled think-phase positions across TASK_IDS.

KEY metric — agreement@k: fraction of positions where the FINAL layer's top-1 token is
within layer ℓ's top-k. High agreement = layer ℓ keeps the correct token (meaningful
diversity, safe for pless-on-latent); low agreement = ℓ diverges to other/junk tokens.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/layer_entropy_profile.py
Env: TASK_IDS("1226 930 558 1126"), MAX_THINK_TOK(1200), N_POS(120)
"""
import json
import os
import torch

from bench.generator import load_model_and_tokenizer
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
TASK_IDS = [int(x) for x in os.environ.get("TASK_IDS", "1226 930 558 1126").split()]
MAX_THINK_TOK = int(os.environ.get("MAX_THINK_TOK", "1200"))
N_POS = int(os.environ.get("N_POS", "120"))
CAND_LAYERS = [int(x) for x in os.environ.get("CAND_LAYERS", "20 22 24 25 26 28 30").split()]


def main():
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tok = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    model.eval()
    final_norm, lm_head = model.model.norm, model.lm_head

    traces = {}
    with open(f"{POD}/truncated_cases.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "pless_think_t1.0_t1.0" and r["task_id"] in TASK_IDS and r["task_id"] not in traces:
                traces[r["task_id"]] = r["truncated_solution"]

    n_layers = model.config.num_hidden_layers
    V = lm_head.weight.size(0)
    # accumulators
    sig2_sum = [0.0] * (n_layers + 1)
    H_sum = [0.0] * (n_layers + 1)
    npos_total = 0
    agree = {l: {1: 0, 5: 0, 10: 0} for l in CAND_LAYERS}

    for tid in TASK_IDS:
        if tid not in traces:
            print(f"task {tid}: no trace, skip"); continue
        problem = pmap[tid]
        prefix, _ = format_prompt_apps_instruct(problem, tok, enable_thinking=True)
        prefix_ids = tok.encode(prefix, return_tensors="pt")[0]
        think_ids = tok.encode(traces[tid], add_special_tokens=False)[:MAX_THINK_TOK]
        input_ids = torch.cat([prefix_ids, torch.tensor(think_ids)]).unsqueeze(0).to(model.device)
        plen, seq = len(prefix_ids), input_ids.size(1)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        lo, hi = plen - 1, seq - 2
        g = torch.Generator().manual_seed(0)
        pos = torch.randint(lo, hi, (min(N_POS, hi - lo),), generator=g).unique().to(model.device)
        npos_total += len(pos)

        # final-layer top-1 (the model's actual choice) per position
        final_logits = lm_head(final_norm(hs[n_layers][0, pos]))
        final_top1 = final_logits.argmax(-1)                      # (P,)

        for ell in range(n_layers + 1):
            logits = lm_head(final_norm(hs[ell][0, pos]))
            p = torch.softmax(logits.float(), -1)
            sig2_sum[ell] += (p * p).sum(-1).sum().item()
            H_sum[ell] += (-(p * p.clamp_min(1e-12).log()).sum(-1)).sum().item()
            if ell in CAND_LAYERS:
                topk = logits.topk(10, dim=-1).indices                # (P,10)
                hit = (topk == final_top1.unsqueeze(1))               # (P,10)
                agree[ell][1] += hit[:, :1].any(1).sum().item()
                agree[ell][5] += hit[:, :5].any(1).sum().item()
                agree[ell][10] += hit.any(1).sum().item()
        del hs, out
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"task {tid}: profiled {len(pos)} positions", flush=True)

    print(f"\n=== AGGREGATE over {len(TASK_IDS)} tasks, {npos_total} positions "
          f"(n_layers={n_layers}, V={V}, uniform Σpᵢ²={1/V:.2e}) ===")
    print(f"{'layer':>5} {'meanΣpᵢ²':>11} {'meanH':>8} {'~eff#tok':>9}")
    import math
    for ell in range(n_layers + 1):
        s = sig2_sum[ell] / npos_total
        H = H_sum[ell] / npos_total
        tag = "embed" if ell == 0 else ("FINAL" if ell == n_layers else "")
        print(f"{ell:>5} {s:>11.4e} {H:>8.3f} {math.exp(H):>9.0f}  {tag}")

    print(f"\n=== AGREEMENT@k: fraction of positions where FINAL top-1 ∈ layer-ℓ top-k ===")
    print("(high = layer keeps the model's correct token = SAFE for pless-on-latent)")
    print(f"{'layer':>5} {'agree@1':>8} {'agree@5':>8} {'agree@10':>9}")
    for l in CAND_LAYERS:
        print(f"{l:>5} {agree[l][1]/npos_total:>8.2f} {agree[l][5]/npos_total:>8.2f} {agree[l][10]/npos_total:>9.2f}")
    print("\nVerdict heuristic: target the highest layer with meaningful diversity "
          "(~5-30 eff tokens) AND high agree@10 (keeps the right token).")


if __name__ == "__main__":
    main()
