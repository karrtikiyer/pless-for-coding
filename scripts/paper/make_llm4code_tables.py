"""Extract LLM4Code paper tables from source-of-truth metrics JSONs.

No hand-typed numbers. Every value in every table comes from a metrics JSON
via `bench.eval.headline_table.headline_table(...)`, which locks the
paper's 7-column format (pass@{1,3,5,10}, cov@{0.3,0.7}, cb_div) via
PAPER_COLUMNS defined here (decoupled from CANONICAL_COLUMNS).

Usage:
    uv run python scripts/paper/make_llm4code_tables.py

Writes markdown tables under `paper/llm4code/tables/`. Prints a gap report
to stderr listing any (model, benchmark, config) cells where the metrics
JSON is missing — those are the experiments still to run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench.eval.headline_table import headline_row, headline_table  # noqa: E402


# Explicit column set for paper tables — decoupled from CANONICAL_COLUMNS
# so the paper doesn't silently gain/lose columns when the canonical set is
# expanded for other analysis workflows.
PAPER_COLUMNS = [
    "pass@1",
    "pass@3",
    "pass@5",
    "pass@10",
    "cov@0.3",
    "cov@0.7",
    "cb_div",
]

# Hasan et al. arXiv:2507.03160 reported MBPP pass@1 numbers at top-p=0.95, T=0.2
# (reproduced from their paper; we do not have their raw metrics, so only pass@1
# is populated when the Hasan baseline row is injected into Table 3C).
HASAN_REPORTED_PASS_AT_1 = {
    "Qwen2.5-Coder-3B": 57.0,
    "Qwen2.5-Coder-1.5B": 51.0,
    "OpenCodeInterpreter-DS-1.3B": 44.0,
}

# Per-table-group caption prefix (benchmark + task count + sample count),
# keyed by table_id prefix. Keeps captions self-contained for a reader
# scanning tables out of order (prior "Metrics for <model>." gave no benchmark
# context).
_TABLE_CAPTION_PREFIX = {
    "3a_": r"MBPP-full (500 tasks, $n{=}10$)",
    "3b_": r"HumanEval (164 tasks, $n{=}10$)",
    "3c_": r"MBPP-full (500 tasks, $n{=}10$)",
    "5a_": r"APPS-252 ATCODER-interview (252 tasks, $n{=}10$, thinking-ON)",
    "a1_": r"MBPP-full (500 tasks, $n{=}10$)",
    "a2_": r"HumanEval (164 tasks, $n{=}10$)",
}


TABLES_DIR = REPO_ROOT / "paper" / "llm4code" / "tables"
# Sibling paper repo (Overleaf-linked GitHub repo) — .tex tables also copied here
# after each run so the paper repo doesn't drift from the code repo.
TABLES_DIR_PAPER = REPO_ROOT.parent / "pless-for-coding-paper" / "tables"

# Per-model, per-benchmark config: (label, metrics_json_path_relative_to_repo).
# Baselines first, then p-less family. Labels are what the table header will show.
#
# When a JSON is missing, the row is DROPPED from that table and a warning is
# logged to stderr — this keeps the paper honest about coverage without
# silently omitting rows.

MBPP: dict[str, list[tuple[str, str]]] = {
    # Canonical 6-row layout: greedy, beam4, temp@0.7, top-p@0.9, pless@0.7, pless-norm@0.7.
    # Same configs across all 7 MBPP models. Qwen2.5-Coder-7B-Instruct is dropped
    # from Table 3A because its MBPP runs used different Ts (no temp@0.7, no top-p@0.9);
    # it stays in Table 3B (HumanEval) where config coverage matches.
    "CodeLlama-7b-hf": [
        ("greedy", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/greedy_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/beam4_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/temp_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-hf/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "CodeLlama-7b-Instruct-hf": [
        ("greedy", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/greedy_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/beam4_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "OpenCodeInterpreter-DS-1.3B": [
        ("greedy", "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/greedy_bigcode_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/beam4_bigcode_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/temp_bigcode_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/top_p0.9_bigcode_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/pless_bigcode_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/pless_norm_bigcode_t0.7_metrics.json"),
    ],
    "Llama-2-7b-hf": [
        # Base model, MBPP only. In Shi et al.'s code-generation model set.
        ("greedy", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/greedy_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/beam4_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/temp_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-hf/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "Llama-2-7b-chat-hf": [
        # Instruct model, MBPP only. In Shi et al.'s code-generation model set.
        ("greedy", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/greedy_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/beam4_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/temp_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/meta-llama--Llama-2-7b-chat-hf/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "Qwen2.5-Coder-1.5B": [
        # Base small model; also in Hasan et al. small-model set (Table 3C).
        ("greedy", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/greedy_bigcode_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/beam4_bigcode_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/temp_bigcode_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/top_p0.9_bigcode_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_bigcode_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_norm_bigcode_t0.7_metrics.json"),
    ],
    "Qwen2.5-Coder-3B": [
        # Base small model; also in Hasan et al. small-model set (Table 3C).
        ("greedy", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/greedy_bigcode_t1.0_metrics.json"),
        ("beam4", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/beam4_bigcode_t1.0_metrics.json"),
        ("temp@0.7", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/temp_bigcode_t0.7_metrics.json"),
        ("top-p@0.9", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/top_p0.9_bigcode_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_bigcode_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_norm_bigcode_t0.7_metrics.json"),
    ],
    # Qwen2.5-Coder-7B-Instruct MBPP is intentionally dropped from Table 3A:
    # its MBPP runs used temp@{0.2,0.6,0.8}, top-p@0.95 (not the canonical temp@0.7 /
    # top-p@0.9), so it can't be shown at the canonical T alongside the other 7 models.
    # The model is still headline on HumanEval in Table 3B where config coverage matches.
}


# Appendix tables — additional base/instruct models we ran but did not headline.
MBPP_APPENDIX: dict[str, list[tuple[str, str]]] = {
    "Qwen-7B (base)": [
        ("greedy",           "results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/greedy_t1.0_metrics.json"),
        ("beam4",            "results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/beam4_t1.0_metrics.json"),
        ("temp@0.7",         "results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/temp_t0.7_metrics.json"),
        ("top-p@0.9",        "results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.7",       "results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7",  "results/pless_full_mbpp_results/Qwen--Qwen-7B/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "Qwen-7B-Chat": [
        ("greedy",           "results/pless_full_mbpp_results/Qwen--Qwen-7B-Chat/metrics/greedy_t1.0_metrics.json"),
        ("beam4",            "results/pless_full_mbpp_results/Qwen--Qwen-7B-Chat/metrics/beam4_t1.0_metrics.json"),
        ("temp@0.7",         "results/pless_full_mbpp_results/Qwen--Qwen-7B-Chat/metrics/temp_t0.7_metrics.json"),
        ("top-p@0.9",        "results/pless_full_mbpp_results/Qwen--Qwen-7B-Chat/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.7",       "results/pless_full_mbpp_results/Qwen--Qwen-7B-Chat/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7",  "results/pless_full_mbpp_results/Qwen--Qwen-7B-Chat/metrics/pless_norm_t0.7_metrics.json"),
    ],
}

HUMANEVAL_APPENDIX: dict[str, list[tuple[str, str]]] = {
    "Qwen2.5-Coder-7B (base)": [
        ("temp@0.7",         "results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B/metrics/temp_t0.7_metrics.json"),
        ("temp@1.0",         "results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B/metrics/temp_t1.0_metrics.json"),
        ("p-less@0.7",       "results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7",  "results/pless_human_eval_results/temprature_results/Qwen--Qwen2.5-Coder-7B/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "Codestral-22B-v0.1": [
        ("greedy",           "results/pless_human_eval_results/full_precision_results/mistralai--Codestral-22B-v0.1/metrics/greedy_metrics.json"),
        ("temp@0.7",         "results/pless_human_eval_results/full_precision_results/mistralai--Codestral-22B-v0.1/metrics/temp_0.7_metrics.json"),
        ("top-p@0.95",       "results/pless_human_eval_results/full_precision_results/mistralai--Codestral-22B-v0.1/metrics/top_p_0.95_metrics.json"),
        ("top-p@0.9",        "results/pless_human_eval_results/full_precision_results/mistralai--Codestral-22B-v0.1/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.6",       "results/pless_human_eval_results/full_precision_results/mistralai--Codestral-22B-v0.1/metrics/pless_t0.6_metrics.json"),
        ("p-less-norm@0.6",  "results/pless_human_eval_results/full_precision_results/mistralai--Codestral-22B-v0.1/metrics/pless_norm_t0.6_metrics.json"),
    ],
    "Qwen3-Coder-30B-A3B-Instruct": [
        ("greedy",           "results/pless_human_eval_results/full_precision_results/Qwen--Qwen3-Coder-30B-A3B-Instruct/metrics/greedy_metrics.json"),
        ("temp@0.7",         "results/pless_human_eval_results/full_precision_results/Qwen--Qwen3-Coder-30B-A3B-Instruct/metrics/temp_0.7_metrics.json"),
        ("top-p@0.95",       "results/pless_human_eval_results/full_precision_results/Qwen--Qwen3-Coder-30B-A3B-Instruct/metrics/top_p_0.95_metrics.json"),
        ("top-p@0.9",        "results/pless_human_eval_results/full_precision_results/Qwen--Qwen3-Coder-30B-A3B-Instruct/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.6",       "results/pless_human_eval_results/full_precision_results/Qwen--Qwen3-Coder-30B-A3B-Instruct/metrics/pless_t0.6_metrics.json"),
        ("p-less-norm@0.6",  "results/pless_human_eval_results/full_precision_results/Qwen--Qwen3-Coder-30B-A3B-Instruct/metrics/pless_norm_t0.6_metrics.json"),
    ],
}


# Table 3C — small-model MBPP comparison against Hasan et al. (arXiv:2507.03160).
# Their reported single-config baselines: Qwen2.5-Coder-3B 57.0%, 1.5B 51.0%, OCI-DS-1.3B 44.0%.
# All rows below are our own runs; the Hasan et al. baseline is stated inline in the caption.
SMALL_MODELS_2507: dict[str, list[tuple[str, str]]] = {
    "Qwen2.5-Coder-3B": [
        ("greedy",           "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/greedy_bigcode_t1.0_metrics.json"),
        ("beam4",            "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/beam4_bigcode_t1.0_metrics.json"),
        ("temp@0.7",         "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/temp_bigcode_t0.7_metrics.json"),
        ("p-less@0.6",       "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_bigcode_t0.6_metrics.json"),
        ("p-less@0.7",       "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_bigcode_t0.7_metrics.json"),
        ("p-less-norm@0.6",  "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-3B/metrics/pless_norm_bigcode_t0.6_metrics.json"),
    ],
    "Qwen2.5-Coder-1.5B": [
        ("greedy",           "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/greedy_bigcode_t1.0_metrics.json"),
        ("beam4",            "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/beam4_bigcode_t1.0_metrics.json"),
        ("temp@0.7",         "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/temp_bigcode_t0.7_metrics.json"),
        ("p-less@0.6",       "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_bigcode_t0.6_metrics.json"),
        ("p-less@0.7",       "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_bigcode_t0.7_metrics.json"),
        ("p-less-norm@0.6",  "results/pless_full_mbpp_results/Qwen--Qwen2.5-Coder-1.5B/metrics/pless_norm_bigcode_t0.6_metrics.json"),
    ],
    "OpenCodeInterpreter-DS-1.3B": [
        ("greedy",           "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/greedy_bigcode_t1.0_metrics.json"),
        ("beam4",            "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/beam4_bigcode_t1.0_metrics.json"),
        ("temp@0.7",         "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/temp_bigcode_t0.7_metrics.json"),
        ("p-less@0.7",       "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/pless_bigcode_t0.7_metrics.json"),
        ("p-less-norm@0.7",  "results/pless_full_mbpp_results/m-a-p--OpenCodeInterpreter-DS-1.3B/metrics/pless_norm_bigcode_t0.7_metrics.json"),
    ],
}

HUMANEVAL: dict[str, list[tuple[str, str]]] = {
    "CodeLlama-7b-hf": [
        ("temp@0.7", "results/pless_human_eval_results/temprature_results/codellama--CodeLlama-7b-hf/metrics/temp_t0.7_metrics.json"),
        ("temp@1.0", "results/pless_human_eval_results/temprature_results/codellama--CodeLlama-7b-hf/metrics/temp_t1.0_metrics.json"),
        ("p-less@0.7", "results/pless_human_eval_results/temprature_results/codellama--CodeLlama-7b-hf/metrics/pless_t0.7_metrics.json"),
        ("p-less-norm@0.7", "results/pless_human_eval_results/temprature_results/codellama--CodeLlama-7b-hf/metrics/pless_norm_t0.7_metrics.json"),
    ],
    "CodeLlama-7b-Instruct-hf": [
        ("greedy", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/greedy_metrics.json"),
        ("temp@0.2", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_0.2_metrics.json"),
        ("temp@0.7", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/temp_0.7_metrics.json"),
        ("top-p@0.95", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p_0.95_metrics.json"),
        ("top-p@0.9", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.6", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_t0.6_metrics.json"),
        ("p-less-norm@0.6", "results/pless_human_eval_results/full_precision_results/codellama--CodeLlama-7b-Instruct-hf/metrics/pless_norm_t0.6_metrics.json"),
    ],
    "OpenCodeInterpreter-DS-1.3B": [
        # No non-p-less baselines on HE for this model — see MISSING_EXPERIMENTS.md.
        # pless_alpha at a=2.0 is mathematically identical to p-less (both threshold
        # at the collision probability Σpᵢ²), so it renders as p-less@1.0.
        ("p-less@1.0", "results/pless_alpha_full_humaneval/m-a-p--OpenCodeInterpreter-DS-1.3B/humaneval/metrics/pless_alpha_a2.0_t1.0_metrics.json"),
    ],
    "Qwen2.5-Coder-7B-Instruct": [
        ("greedy", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/greedy_metrics.json"),
        ("temp@0.2", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/temp_0.2_metrics.json"),
        ("temp@0.7", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/temp_0.7_metrics.json"),
        ("top-p@0.95", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/top_p_0.95_metrics.json"),
        ("top-p@0.9", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/top_p0.9_t1.0_metrics.json"),
        ("p-less@0.6", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/pless_t0.6_metrics.json"),
        ("p-less-norm@0.6", "results/pless_human_eval_results/full_precision_results/Qwen--Qwen2.5-Coder-7B-Instruct/metrics/pless_norm_t0.6_metrics.json"),
    ],
}

# APPS-252 Rényi sweep for reasoning models — Section 5 tables.
# Baseline (k=2 collision entropy) is pless@t1.0 from the recovery/fixed_full252 dirs.
APPS_RENYI: dict[str, list[tuple[str, str]]] = {
    "Qwen3-8B": [
        # Vendor-recommended config per HuggingFace Qwen3-8B model card (thinking mode):
        # T=0.6, top-p=0.95, top-k=20, min-p=0. See §5.4 / Fig 5 caption for context.
        ("recommended", "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/temp_p0.95_k20_think_t0.6_t0.6_metrics.json"),
        # top-p sweep at T=1.0 (thinking-ON); shows G_k best-k exceeds even the best swept top-p
        ("top-p @0.7",  "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/temp_p0.7_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.75", "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/temp_p0.75_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.8",  "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/temp_p0.8_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.85", "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/temp_p0.85_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.9",  "results/_top_p_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/temp_p0.9_think_t1.0_t1.0_metrics.json"),
        ("p-less baseline (k=2)", "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/metrics/pless_think_t1.0_t1.0_metrics.json"),
        ("G_k=0.05", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.05_t1.0_metrics.json"),
        ("G_k=0.1", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.1_t1.0_metrics.json"),
        ("G_k=0.2", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.2_t1.0_metrics.json"),
        ("G_k=0.25", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.25_t1.0_metrics.json"),
        ("G_k=0.3", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.3_t1.0_metrics.json"),
        ("G_k=0.35", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.35_t1.0_metrics.json"),
        ("G_k=0.4", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.4_t1.0_metrics.json"),
        ("G_k=0.8", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.8_t1.0_metrics.json"),
        ("G_k=1.6", "results/_renyi_sweep_full252/Qwen--Qwen3-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k1.6_t1.0_metrics.json"),
    ],
    "DeepSeek-R1-Distill-Llama-8B": [
        # Vendor-recommended config per HuggingFace DeepSeek-R1-Distill-Llama-8B model card:
        # T=0.6, top-p=0.95. See §5.4 / Fig 5 caption for context.
        ("recommended", "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.95_think_t0.6_t0.6_metrics.json"),
        # top-p sweep at T=1.0 (thinking-ON)
        ("top-p @0.7",  "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.7_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.75", "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.75_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.8",  "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.8_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.85", "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.85_think_t1.0_t1.0_metrics.json"),
        ("top-p @0.9",  "results/_top_p_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/temp_p0.9_think_t1.0_t1.0_metrics.json"),
        ("p-less baseline (k=2)", "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_think_t1.0_t1.0_metrics.json"),
        ("G_k=0.05", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.05_t1.0_metrics.json"),
        ("G_k=0.075", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.075_t1.0_metrics.json"),
        ("G_k=0.1", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.1_t1.0_metrics.json"),
        ("G_k=0.15", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.15_t1.0_metrics.json"),
        ("G_k=0.2", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.2_t1.0_metrics.json"),
        ("G_k=0.25", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.25_t1.0_metrics.json"),
        ("G_k=0.3", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.3_t1.0_metrics.json"),
        ("G_k=0.35", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.35_t1.0_metrics.json"),
        ("G_k=0.4", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.4_t1.0_metrics.json"),
        ("G_k=0.45", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.45_t1.0_metrics.json"),
        ("G_k=0.8", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k0.8_t1.0_metrics.json"),
        ("G_k=1.6", "results/_renyi_sweep_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/metrics/pless_renyi_think_t1.0_k1.6_t1.0_metrics.json"),
    ],
}


import json  # noqa: E402


# Expected num_tasks per benchmark slug. MBPP-full is 500 (never 257 sanitized);
# HumanEval is 164; APPS-252 is 252. Rows with a different task count are DROPPED
# and a hard warning is emitted — mixing task-count populations in one table is a
# paper-integrity bug that a reviewer will catch.
_EXPECTED_NUM_TASKS: dict[str, int] = {
    "table_3a_mbpp": 500,
    "table_3b_humaneval": 164,
    "table_3c_small_models_2507": 500,
    "table_a1_mbpp_appendix": 500,
    "table_a2_humaneval_appendix": 164,
    "table_5a_apps_renyi": 252,
}


def _filter_present(rows: Iterable[tuple[str, str]], label: str, expected_num_tasks: int | None = None) -> list[tuple[Path, str]]:
    """Filter rows to those with an existing metrics JSON AND the expected num_tasks.

    If ``expected_num_tasks`` is set, rows whose JSON reports a different
    ``num_tasks`` are DROPPED with a WRONG_NUM_TASKS warning. Log gaps to stderr.
    """
    kept: list[tuple[Path, str]] = []
    for row_label, rel_path in rows:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"GAP: {label} | {row_label} | missing: {rel_path}", file=sys.stderr)
            continue
        if expected_num_tasks is not None:
            actual = json.loads(path.read_text()).get("num_tasks")
            if actual != expected_num_tasks:
                print(f"WRONG_NUM_TASKS: {label} | {row_label} | got {actual}, expected {expected_num_tasks}: {rel_path}", file=sys.stderr)
                continue
        kept.append((path, row_label))
    return kept


def _render_model_section(model: str, benchmark: str, rows: list[tuple[str, str]], expected_num_tasks: int | None = None) -> str:
    """Render a per-model section: heading + headline_table markdown."""
    label = f"{model} on {benchmark}"
    present = _filter_present(rows, label, expected_num_tasks=expected_num_tasks)
    if not present:
        return f"### {model}\n\n_No metrics on disk (or all rows dropped by num_tasks guard)._\n"
    return f"### {model}\n\n" + headline_table(present, label_header="Decoder", columns=PAPER_COLUMNS) + "\n"


_TEX_LABEL_MAP = {
    "pass@1": r"pass@1",
    "pass@3": r"pass@3",
    "pass@5": r"pass@5",
    "pass@10": r"pass@10",
    "cov@0.3": r"cov@0.3",
    "cov@0.7": r"cov@0.7",
    "cb_div": r"cb\_div",
}


def _tex_escape(s: str) -> str:
    """Minimal escape for LaTeX cells (labels + values). Handles: backslash, & % _,
    UTF-8 em-dash (→ ---), and Greek letter α (→ $\alpha$) which appear in labels/values.
    Default IEEEtran font can't render these UTF-8 codepoints even under [T1]{fontenc}.
    """
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("—", "---")   # em-dash → LaTeX em-dash
        .replace("α", r"$\alpha$")  # α → math mode
    )


def _render_model_section_tex(model: str, rows: list[tuple[str, str]], table_id: str, expected_num_tasks: int | None = None) -> str:
    """Render a per-model LaTeX booktabs table using IEEE conference style."""
    label = f"{model}"
    present = _filter_present(rows, label, expected_num_tasks=expected_num_tasks)
    if not present:
        return f"% {model}: no metrics on disk (or all rows dropped by num_tasks guard) — table skipped\n"

    col_spec = "l" + "r" * len(PAPER_COLUMNS)
    header_cells = ["Decoder"] + [_TEX_LABEL_MAP[c] for c in PAPER_COLUMNS]
    header_row = " & ".join(header_cells) + r" \\"

    body_lines = []
    for path, row_label in present:
        row = headline_row(path, row_label)
        cells = [_tex_escape(row["label"])] + [_tex_escape(row[c]) for c in PAPER_COLUMNS]
        body_lines.append("    " + " & ".join(cells) + r" \\")

    # Table 3C: inject the Hasan et al. reported-only baseline row for the 3 small
    # models we share with them. Only pass@1 is populated (they don't report cov@t
    # or diversity at matched configs); all other columns show "—". Row-level
    # attribution to Hasan et al. goes in the caption to keep the row label short
    # (long labels force \resizebox to shrink the whole table's font).
    if table_id.startswith("3c_") and model in HASAN_REPORTED_PASS_AT_1:
        hasan_cells = [r"Hasan (top-$p$=0.95, $T$=0.2)"]
        for col in PAPER_COLUMNS:
            if col == "pass@1":
                hasan_cells.append(f"{HASAN_REPORTED_PASS_AT_1[model]:.2f}\\%")
            else:
                hasan_cells.append("---")
        body_lines.append("    " + " & ".join(hasan_cells) + r" \\")

    prefix_key = table_id.split("_", 1)[0] + "_"
    prefix = _TABLE_CAPTION_PREFIX.get(prefix_key)
    if prefix is not None:
        caption = f"{prefix} --- {_tex_escape(model)}."
    else:
        caption = f"Metrics for {_tex_escape(model)}."
    if table_id.startswith("3c_") and model in HASAN_REPORTED_PASS_AT_1:
        caption += r" ``Hasan'' row reported by Hasan et al.~\cite{hasan2025slm}; other rows this work."
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{table_id}}}",
        rf"\resizebox{{\columnwidth}}{{!}}{{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def _write_both(configs: dict[str, list[tuple[str, str]]], slug: str, md_heading: str, table_id_prefix: str) -> None:
        expected = _EXPECTED_NUM_TASKS.get(slug)
        md = f"# {md_heading}\n\nCanonical headline metrics from `bench.eval.headline_table`. Auto-generated by `scripts/paper/make_llm4code_tables.py`. Rows with num_tasks != {expected} are DROPPED (see stderr WRONG_NUM_TASKS warnings).\n\n"
        tex_parts = [f"% {md_heading}\n% Auto-generated by scripts/paper/make_llm4code_tables.py — do not hand-edit.\n% Rows with num_tasks != {expected} are DROPPED.\n"]
        for model, rows in configs.items():
            md += _render_model_section(model, slug, rows, expected_num_tasks=expected) + "\n"
            # Sanitize model name to LaTeX-safe label: strip . / - / space / ( / ) / ,
            safe = model.replace('.', '').replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            tex_parts.append(_render_model_section_tex(model, rows, f"{table_id_prefix}_{safe}", expected_num_tasks=expected))
        (TABLES_DIR / f"{slug}.md").write_text(md)
        (TABLES_DIR / f"{slug}.tex").write_text("\n".join(tex_parts))

    _write_both(MBPP, "table_3a_mbpp", "Table 3A — MBPP-full (500 tasks, n=10)", "3a")
    _write_both(HUMANEVAL, "table_3b_humaneval", "Table 3B — HumanEval (164 tasks, n=10)", "3b")
    _write_both(SMALL_MODELS_2507, "table_3c_small_models_2507", "Table 3C — Small-model MBPP vs Hasan et al. arXiv:2507.03160 baseline", "3c")
    _write_both(APPS_RENYI, "table_5a_apps_renyi", "Table 5A — APPS-252 (ATCODER-interview, n=10, thinking-ON)", "5a")
    _write_both(MBPP_APPENDIX, "table_a1_mbpp_appendix", "Table A1 — MBPP appendix (additional base/instruct models)", "a1")
    _write_both(HUMANEVAL_APPENDIX, "table_a2_humaneval_appendix", "Table A2 — HumanEval appendix (additional base/instruct models)", "a2")


    # Mirror .tex + .md tables into sibling paper repo (Overleaf-linked) if it exists.
    # .tex is what main.tex \input's; .md is kept for repo-side markdown viewing.
    if TABLES_DIR_PAPER.exists():
        n = 0
        for src in list(TABLES_DIR.glob("*.tex")) + list(TABLES_DIR.glob("*.md")):
            (TABLES_DIR_PAPER / src.name).write_text(src.read_text())
            n += 1
        print(f"mirrored {n} table files to {TABLES_DIR_PAPER}", file=sys.stderr)

    print(f"Wrote {len(MBPP) + len(HUMANEVAL) + len(SMALL_MODELS_2507) + len(APPS_RENYI) + len(MBPP_APPENDIX) + len(HUMANEVAL_APPENDIX)} per-model sections across 6 md + 6 tex files in {TABLES_DIR.relative_to(REPO_ROOT)}/", file=sys.stderr)


if __name__ == "__main__":
    main()
