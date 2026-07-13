"""CLI: paired HF-vs-vLLM backend-delta report for pless α=2 on identical task_ids.

Consumes two bench.eval metrics JSONs (for pass@k, via the canonical estimator) and the
two run JSONLs (for the exact no-</think> truncation rate), restricts to the shared /
requested task_ids, and prints a paired report with a bootstrap CI on the HF-vLLM gap.

Example (after the HF run + its bench.eval scoring):
  uv run python scripts/compare_backend_delta.py \
    --hf-metrics   results/_backend_delta_deepseek/.../metrics/pless_think_t1.0_t1.0_metrics.json \
    --hf-jsonl     results/_backend_delta_deepseek/.../pless_think_t1.0_t1.0.jsonl \
    --vllm-metrics results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_think_t1.0_t1.0_metrics.json \
    --vllm-jsonl   results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl
"""
import argparse
import json

from scripts.backend_delta import (
    bootstrap_gap,
    paired_task_results,
    pass_at_k,
    truncation_rate,
)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_metrics(path):
    with open(path) as f:
        return json.load(f)["per_task"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-metrics", required=True)
    ap.add_argument("--vllm-metrics", required=True)
    ap.add_argument("--hf-jsonl", required=True)
    ap.add_argument("--vllm-jsonl", required=True)
    ap.add_argument("--task-ids", type=int, nargs="+", default=None,
                    help="Restrict to these task_ids (default: all shared).")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional markdown output path.")
    args = ap.parse_args()

    hf_pt = _load_metrics(args.hf_metrics)
    v_pt = _load_metrics(args.vllm_metrics)
    ids, hf_res, v_res = paired_task_results(hf_pt, v_pt, subset=args.task_ids)

    hf_k = pass_at_k(hf_res, args.k)
    v_k = pass_at_k(v_res, args.k)

    hf_recs = _load_jsonl(args.hf_jsonl)
    v_recs = _load_jsonl(args.vllm_jsonl)
    hf_tr = truncation_rate(hf_recs, subset=ids)
    v_tr = truncation_rate(v_recs, subset=ids)

    gaps = {k: bootstrap_gap(hf_res, v_res, k=k, iters=args.iters, seed=args.seed)
            for k in args.k}

    lines = []
    lines.append(f"# Backend delta (HF vs vLLM), pless α=2 — paired on {len(ids)} task_ids\n")
    lines.append(f"Shared task_ids: {ids}\n")
    lines.append("| metric | HF | vLLM | HF−vLLM (95% CI) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| truncation (no-</think>) | {hf_tr[2]:.1%} ({hf_tr[0]}/{hf_tr[1]}) "
                 f"| {v_tr[2]:.1%} ({v_tr[0]}/{v_tr[1]}) | — |")
    for k in args.k:
        pt, lo, hi = gaps[k]
        sig = "" if lo <= 0 <= hi else "  ← excludes 0"
        lines.append(f"| pass@{k} | {hf_k[str(k)]:.3f} | {v_k[str(k)]:.3f} "
                     f"| {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig} |")
    lines.append("")
    verdict_k = 1 if 1 in args.k else args.k[0]
    _, lo, hi = gaps[verdict_k]
    if not (lo <= 0 <= hi):
        lines.append(f"**Verdict:** pass@{verdict_k} gap CI excludes 0 → backends are NOT "
                     f"equivalent for this model at matched config (the divergence is real).")
    else:
        lines.append(f"**Verdict:** pass@{verdict_k} gap CI includes 0 → no significant "
                     f"backend difference at this N; expand the subset before concluding.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
