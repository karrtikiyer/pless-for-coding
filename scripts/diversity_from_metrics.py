"""Compute diversity on ALREADY-EVALUATED results — NO code re-execution.

Pure glue: loads each config's existing metrics (per_task: task_id/num_correct/
pass_results) + its JSONL (samples), then calls the PROJECT'S OWN diversity functions
(add_structural_diversity, add_self_codebleu, compute_structural_diversity,
compute_self_codebleu_diversity). Those operate on samples+pass_results via static AST/
CodeBLEU analysis — they never run the sandbox — so this reuses the canonical metric
logic without re-executing. bench.eval couples diversity with execution and has no
report-only path; this supplies that missing entry point only.

Run: PYTHONPATH=. uv run python scripts/diversity_from_metrics.py
"""
import json
from bench.eval.metrics import (
    add_self_codebleu, compute_self_codebleu_diversity,
)
# NOTE: structural_diversity (zss tree-edit) is INTRACTABLE on APPS-CoT code — a single
# pair of ~2000-node solutions times out >60s, and there are thousands of pairs. That is
# the documented reason these runs use --skip-diversity. CodeBLEU is fast (~0.01s/pair),
# so we report cb_div only, via the project's own add_self_codebleu (no re-execution).

FULL = "results/pless_recovery_full252/Qwen--Qwen3-8B/ATCODER_interview"
CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
DEC06 = "results/decoders_t0.6/Qwen--Qwen3-8B/ATCODER_interview"

# Set via env CONFIG_SET=main|dec06 (default main). dec06 = the 4 new T0.6 decoders.
import os as _os
if _os.environ.get("CONFIG_SET") == "dec06":
    CONFIGS = [
        ("pless @T0.6",      f"{DEC06}/pless_think_t0.6_t0.6"),
        ("pless_norm @T0.6", f"{DEC06}/pless_norm_think_t0.6_t0.6"),
        ("top_p @T0.6",      f"{DEC06}/temp_p0.95_think_t0.6_t0.6"),
        ("top_k @T0.6",      f"{DEC06}/temp_k20_think_t0.6_t0.6"),
    ]
else:
    CONFIGS = [
        ("pless α=3",        f"{FULL}/pless_alpha_think_t1.0_a3.0_t1.0"),
        ("pless α=4",        f"{FULL}/pless_alpha_think_t1.0_a4.0_t1.0"),
        ("pless α=5",        f"{FULL}/pless_alpha_think_t1.0_a5.0_t1.0"),
        ("pless T2.0",       f"{FULL}/pless_think_t2.0_t2.0"),
        ("pless @α2 (base)", f"{CANON}/pless_think_t1.0_t1.0"),
        ("pless_norm @α2",   f"{CANON}/pless_norm_think_t1.0_t1.0"),
        ("temp p0.95 @T1.0", f"{CANON}/temp_p0.95_think_t1.0_t1.0"),
        ("temp k20 @T1.0",   f"{CANON}/temp_k20_think_t1.0_t1.0"),
        ("temp @T0.6",       f"{CANON}/temp_think_t0.6_t0.6"),
        ("temp p+k @T0.6",   f"{CANON}/temp_p0.95_k20_think_t0.6_t0.6"),
    ]


def main():
    print(f"{'config':<18} {'cb_div':>8} {'syntax_div':>11} {'dataflow_div':>13} {'#≥2corr':>8}")
    print("-" * 56)
    for name, stem in CONFIGS:
        d, base = stem.rsplit("/", 1)
        m = json.load(open(f"{d}/metrics/{base}_metrics.json"))
        task_results = m["per_task"]                     # task_id, num_correct, pass_results
        records = [json.loads(l) for l in open(f"{stem}.jsonl")]  # task_id, samples
        add_self_codebleu(task_results, records)         # project's own fn, static, no execution
        agg = compute_self_codebleu_diversity(task_results)
        n_ge2 = sum(1 for t in task_results if t.get("num_correct", 0) >= 2)
        print(f"{name:<18} {agg.get('codebleu_diversity',0):>8.4f} "
              f"{agg.get('syntax_match_diversity',0):>11.4f} "
              f"{agg.get('dataflow_match_diversity',0):>13.4f} {n_ge2:>8}", flush=True)


if __name__ == "__main__":
    main()
