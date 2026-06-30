"""Compute structural diversity for the 9 DeepSeek ATCODER-interview methods, WITHOUT
re-execution. Reuses the canonical metric functions (bench.eval.metrics) on the
already-extracted code of the already-known correct samples (pass_results live in the
existing metrics JSONs). Validates against alpha=5, whose struct_div (0.524) was computed
by the full eval pipeline.

Usage:
  HF_HUB_OFFLINE=1 uv run python scripts/compute_diversity_deepseek.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bench.eval.executor import extract_python_code
from bench.eval.metrics import (
    add_structural_diversity, compute_structural_diversity,
    add_self_codebleu, compute_self_codebleu_diversity,
)

B = "results/pless_cot_efficiency_vllm/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"
D = "results/pless_recovery_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"

METHODS = [
    ("temp t0.6",          f"{B}/temp_think_t0.6_t0.6.jsonl",                                           f"{B}/metrics/temp_think_t0.6_t0.6_metrics.json"),
    ("temp t1.0 topk20",   f"{B}/temp_k20_think_t1.0_t1.0.jsonl",                                       f"{B}/metrics/temp_k20_think_t1.0_t1.0_metrics.json"),
    ("temp t1.0 topp0.95", f"{B}/temp_p0.95_think_t1.0_t1.0.jsonl",                                     f"{B}/metrics/temp_p0.95_think_t1.0_t1.0_metrics.json"),
    ("temp t0.6 p0.95 k20",f"{B}/temp_p0.95_k20_think_t0.6_t0.6.jsonl",                                 f"{B}/metrics/temp_p0.95_k20_think_t0.6_t0.6_metrics.json"),
    ("pless (a=2)",        f"{B}/pless_think_t1.0_t1.0.jsonl",                                          f"{B}/metrics/pless_think_t1.0_t1.0_metrics.json"),
    ("pless_norm",         f"{B}/pless_norm_think_t1.0_t1.0.jsonl",                                     f"{B}/metrics/pless_norm_think_t1.0_t1.0_metrics.json"),
    ("pless_alpha a=3",    f"{D}/pless_alpha_think_t1.0_a3.0_t1.0.jsonl",                               f"{D}/metrics/pless_alpha_think_t1.0_a3.0_t1.0_metrics.json"),
    ("pless_alpha a=4",    f"{D}/pless_alpha_think_t1.0_a4.0_t1.0.jsonl",                               f"{D}/metrics/pless_alpha_think_t1.0_a4.0_t1.0_metrics.json"),
    ("pless_alpha a=5",    f"{D}/pless_alpha_think_t1.0_a5.0_t1.0.jsonl",                               f"{D}/metrics/pless_alpha_think_t1.0_a5.0_t1.0_metrics.json"),
]


def diversity_for(jsonl_path: str, metrics_path: str) -> tuple[float, float]:
    """Returns (struct_div, cb_div) — both over correct samples, no re-execution."""
    m = json.load(open(metrics_path))
    per_task = {t["task_id"]: t for t in m["per_task"]}
    # records: task_id + extracted code per sample (no execution; same extractor as eval)
    records = []
    for line in open(jsonl_path):
        r = json.loads(line)
        tid = r["task_id"]
        if tid not in per_task:
            continue
        extracted = [extract_python_code(s) or s for s in r["samples"]]
        records.append({"task_id": tid, "samples": extracted})
    # task_results: reuse the EXISTING pass_results + num_correct (no re-execution)
    task_results = [
        {"task_id": tid, "pass_results": per_task[tid]["pass_results"],
         "num_correct": per_task[tid]["num_correct"]}
        for tid in (rec["task_id"] for rec in records)
    ]
    add_structural_diversity(task_results, records)      # AST tree-edit (fast)
    add_self_codebleu(task_results, records)             # all-pairs CodeBLEU (slow)
    sd = compute_structural_diversity(task_results)
    cb = compute_self_codebleu_diversity(task_results)["codebleu_diversity"]
    return sd, cb


def main() -> None:
    out = {}
    print(f"{'method':>22} | {'struct_div':>10} | {'cb_div':>7} | note", flush=True)
    print("-" * 64, flush=True)
    for lab, jl, mp in METHODS:
        if not Path(jl).exists():
            print(f"{lab:>22} | {'MISSING':>10} |         | {jl}", flush=True); continue
        sd, cb = diversity_for(jl, mp)
        out[lab] = {"struct_div": sd, "cb_div": cb}
        note = ""
        if lab == "pless_alpha a=5":
            existing = json.load(open(mp)).get("structural_diversity")
            note = f"struct validate vs {existing} (delta {abs(sd - (existing or 0)):.3f})"
        print(f"{lab:>22} | {sd:>10.4f} | {cb:>7.4f} | {note}", flush=True)
    Path("results/pless_recovery_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/"
         "ATCODER_interview/diversity_all9.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
