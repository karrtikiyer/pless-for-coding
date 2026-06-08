"""CoT token-efficiency vs accuracy — early analysis on existing Qwen3-8B data.

Mines the existing split-decoding corpus (reasoning traces with `<think>...</think>`)
to ask: within a fixed token budget, does the THINK-phase sampler / temperature move
the accuracy-vs-CoT-length Pareto frontier? Reports both token efficiency (think
length, in tokens) and accuracy/exploration (pass@1, pass@k), using the efficiency
decomposition from arXiv:2602.09805 (completion rate / conditional correctness / length).

No new generation is performed — this reuses on-disk JSONL + metrics JSON.

Usage:
    uv run python -m bench.eval.cot_efficiency \
        --results-dir results/pless_full_mbpp_results/Qwen--Qwen3-8B \
        --dataset mbpp

Verification (one config, asserts pass@1 matches the on-disk metrics):
    uv run python -m bench.eval.cot_efficiency \
        --results-dir results/pless_full_mbpp_results/Qwen--Qwen3-8B \
        --dataset mbpp --limit-files 1 --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import date
from pathlib import Path

from bench.eval.executor import strip_code_fences
from bench.eval.loader import load_results
from bench.eval.metrics import compute_cover_at_t, compute_pass_at_k
from bench.eval.split_decoding_analysis import CONFIGS, load_metrics

# Budget for the Pareto frontier: a length axis is only meaningful within one cap.
PARETO_BUDGET = 8192
NEAR_CAP_FRAC = 0.98
K_VALUES = [1, 5, 10]
T_VALUES = [0.3, 0.5]

# Filename → max_tokens, authoritative for configs the existing analysis knows about.
_BUDGET_BY_FILE = {cfg["file"]: cfg["max_tokens"] for cfg in CONFIGS.values()}

# Stable colors per think-sampler for the Pareto scatter.
_SAMPLER_COLORS = {
    "temp_standard": "#1E88E5",
    "temp_pure": "#26A69A",
    "pless": "#E53935",
    "pless_norm": "#8E24AA",
    "pless_alpha": "#FB8C00",
    "temp": "#1E88E5",
}


# ── think-span extraction & length ──────────────────────────────────────────

def extract_think_span(swt: str) -> tuple[str, bool]:
    """Return (think_text, closed) from a `samples_with_thinking` string.

    `closed` is True iff a `</think>` terminator is present (i.e. the thinking
    phase ended naturally rather than being truncated at the token cap).

    Handles two emission styles:
      * opening tag in the OUTPUT — Qwen3 generates `<think>…</think>`.
      * opening tag in the PROMPT — DeepSeek-R1-Distill injects `<think>` via its
        chat template, so the generation starts mid-reasoning and only the
        closing `</think>` appears. Then the think span is everything up to
        `</think>` (or the whole text if it never closed / was truncated).
    Note: the `start == -1` branch never fires for Qwen3 (it emits `<think>`),
    so existing Qwen results are unaffected.
    """
    swt = str(swt)
    end = swt.find("</think>")
    closed = end != -1
    start = swt.find("<think>")
    if start == -1:
        return (swt[:end] if closed else swt), closed
    start += len("<think>")
    return (swt[start:end] if end != -1 else swt[start:]), closed


def extract_think_span_fence(swt: str) -> tuple[str, bool]:
    """Return (think_text, closed) using the ```` ``` ```` code fence as delimiter.

    For *instruct* (non-reasoning) models induced into CoT via a ``<think>``
    prefill, the ``</think>`` terminator emits unreliably (~53% on
    Qwen2.5-Coder-7B-Instruct) — but the model still produces a code fence ~100%
    of the time. So the fence is a far more robust delimiter for "reasoning
    before code":

      * ``think_text`` = everything before the first ```` ``` ```` fence (a
        trailing ``</think>`` tag, if the model did emit one, is stripped so it
        doesn't inflate the span).
      * ``closed`` = a code fence is present (i.e. reasoning terminated and code
        began). No fence at all => the model rambled without producing code =>
        ``closed=False`` (the fence-delimiter analogue of truncation).

    Mirrors :func:`extract_think_span`'s ``(text, closed)`` contract so it is a
    drop-in alternative in :func:`build_sample_rows` via ``delimiter="fence"``.
    """
    swt = str(swt)
    end = swt.find("```")
    closed = end != -1
    think = swt[:end] if closed else swt
    # Drop a trailing </think> the model may have emitted just before the fence.
    tag = think.rfind("</think>")
    if tag != -1:
        think = think[:tag]
    return think, closed


def measure_lengths(text: str, tokenizer) -> tuple[int | None, int]:
    """Return (n_tokens, n_chars). n_tokens is None when no tokenizer is given."""
    n_chars = len(text)
    if tokenizer is None:
        return None, n_chars
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tokens, n_chars


# ── per-sample rows (join + classification) ─────────────────────────────────

def build_sample_rows(records, metrics, tokenizer, max_tokens,
                      has_code_by_task=None, delimiter="think"):
    """Join each samples_with_thinking[i] to its pass/fail label and classify it.

    Classification (mutually exclusive, primary signal = `closed`):
      - truncated : not closed (think phase hit the cap before the delimiter)
      - completed : closed and produced code
      - malformed : closed but no extractable code
    `near_cap` is a secondary diagnostic (tokens >= NEAR_CAP_FRAC * max_tokens),
    reported separately — not merged into the truncation count.

    `delimiter` selects how the think span and `closed` are derived:
      - "think" (default) : `</think>` terminator — for native reasoning models
        (Qwen3, DeepSeek-R1-Distill). Unchanged legacy behavior.
      - "fence"           : first ```` ``` ```` code fence — for instruct models
        induced into CoT, where `</think>` emits unreliably (~53%) but a code
        fence is ~always present. See :func:`extract_think_span_fence`.

    `has_code_by_task` (optional {task_id: [bool]}) overrides the code-presence
    check — used for APPS, where the metrics `extraction_success` list is the
    authoritative signal the executor itself used (the APPS extractor is far
    more involved than `strip_code_fences`).
    """
    extract = extract_think_span_fence if delimiter == "fence" else extract_think_span
    pass_by_task = {pt["task_id"]: pt["pass_results"] for pt in metrics["per_task"]}
    rows = []
    for rec in records:
        tid = rec["task_id"]
        swt_list = rec.get("samples_with_thinking") or []
        code_list = rec.get("samples") or []
        labels = pass_by_task.get(tid)
        if labels is None:
            # Metrics may cover only a subset of the JSONL tasks; skip the rest.
            continue
        if len(swt_list) != len(labels):
            raise ValueError(
                f"alignment violation task_id {tid}: "
                f"{len(swt_list)} thinking samples vs {len(labels)} pass labels"
            )
        code_flags = has_code_by_task.get(tid) if has_code_by_task else None
        for i, swt in enumerate(swt_list):
            think, closed = extract(swt)
            n_tok, n_chars = measure_lengths(think, tokenizer)
            if code_flags is not None:
                has_code = bool(code_flags[i])
            else:
                code = code_list[i] if i < len(code_list) else ""
                has_code = bool(strip_code_fences(str(code)).strip())
            near_cap = n_tok is not None and n_tok >= NEAR_CAP_FRAC * max_tokens
            truncated = not closed
            completed = closed and has_code
            malformed = closed and not has_code
            rows.append({
                "task_id": tid,
                "sample_idx": i,
                "think_tokens": n_tok,
                "think_chars": n_chars,
                "closed": closed,
                "near_cap": near_cap,
                "has_code": has_code,
                "completed": completed,
                "truncated": truncated,
                "malformed": malformed,
                "passed": bool(labels[i]),
            })
    return rows


def aggregate_rows(rows):
    """Efficiency decomposition (arXiv:2602.09805) over per-sample rows."""
    n = len(rows)
    if n == 0:
        return {}
    completed = [r for r in rows if r["completed"]]
    tok_all = [r["think_tokens"] for r in rows if r["think_tokens"] is not None]
    tok_done = [r["think_tokens"] for r in completed if r["think_tokens"] is not None]

    def _mean(xs):
        return statistics.fmean(xs) if xs else None

    def _median(xs):
        return statistics.median(xs) if xs else None

    def _p90(xs):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]

    n_done = len(completed)
    cond_acc = (sum(r["passed"] for r in completed) / n_done) if n_done else None
    return {
        "n_samples": n,
        "completion_rate": n_done / n,
        "truncation_rate": sum(r["truncated"] for r in rows) / n,
        "malformed_rate": sum(r["malformed"] for r in rows) / n,
        "near_cap_rate": sum(r["near_cap"] for r in rows) / n,
        "conditional_correctness": cond_acc,
        # Coherence residual: passing samples NOT classed completed (a truncated
        # trace can still contain a code block that passes). Nonzero => pass@1
        # slightly exceeds compl%×cond. Reported in the coherence audit.
        "passed_not_completed": sum(1 for r in rows if r["passed"] and not r["completed"]),
        "mean_think_tokens": _mean(tok_all),
        "median_think_tokens": _median(tok_all),
        "p90_think_tokens": _p90(tok_all),
        "mean_think_tokens_completed": _mean(tok_done),
        "median_think_tokens_completed": _median(tok_done),
        "mean_think_chars": _mean([r["think_chars"] for r in rows]),
    }


# ── config discovery & metadata ─────────────────────────────────────────────

_THINK_GLOBS = ("split_*.jsonl", "temp_think_t*.jsonl",
                "pless_think_t*.jsonl", "pless_norm_think_t*.jsonl")


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl.xz", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_config_files(results_dir: Path, extra_dirs=(), dataset="mbpp"):
    """Return [(jsonl_path, metrics_path)], deduped by stem (prefer plain .jsonl)."""
    found: dict[str, Path] = {}
    search_dirs = [results_dir, *extra_dirs]
    # APPS uses arbitrary temp-family filenames (temp_p0.95_k20_think_...); glob
    # all jsonl in the dir and let analyze_config skip files lacking thinking /
    # metrics. MBPP/HE use the fixed split/think filename families.
    patterns = ("*.jsonl",) if dataset == "apps" \
        else (*_THINK_GLOBS, "pless_alpha_think_*.jsonl")
    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in patterns:
            for p in d.glob(pattern):
                if ".entropy." in p.name:  # skip entropy sidecars
                    continue
                stem = _stem(p)
                # Prefer uncompressed when duplicates exist.
                if stem not in found or p.suffix == ".jsonl":
                    found[stem] = p
    out = []
    for stem, jsonl_path in sorted(found.items()):
        metrics_path = jsonl_path.parent / "metrics" / f"{stem}_metrics.json"
        out.append((jsonl_path, metrics_path))
    return out


def _apps_label(method: str, top_p, top_k, temp) -> str:
    """Descriptive label for a unified APPS config (temp-family by filter)."""
    if method != "temp":
        return f"{method} (t{temp})"
    parts = []
    if top_p is not None and top_p < 1.0:
        parts.append(f"top_p {top_p}")
    if top_k:
        parts.append(f"top_k {top_k}")
    filt = " + ".join(parts) if parts else "unfiltered"
    return f"temp {temp} ({filt})"


def config_meta(record: dict, jsonl_path: Path, dataset="mbpp",
                max_tokens_override=None) -> dict:
    """Extract sampler/temp/budget metadata for one config from its first record."""
    method = record.get("method", "?")
    stem = _stem(jsonl_path)

    if dataset == "apps":
        # APPS unified runs: method ∈ {temp, pless, pless_norm, pless_alpha};
        # temp-family distinguished by top_p/top_k stored in the record. Budget
        # is the run's --max-new-tokens (not in the record) → supplied by caller.
        top_p = record.get("top_p")
        top_k = record.get("top_k", 0)
        temp = record.get("temperature")
        alpha = record.get("alpha")
        label = _apps_label(method, top_p, top_k, temp)
        return {
            "file": jsonl_path.name,
            "method": method,
            "sampler_think": label,
            "temp_think": temp,
            "sampler_code": label,
            "temp_code": temp,
            "top_p": top_p,
            "top_k": top_k,
            "alpha": alpha,
            "label": label,
            "model": record.get("model"),
            "source": record.get("source"),
            "difficulty": record.get("difficulty"),
            "max_tokens": max_tokens_override,
        }

    budget = _BUDGET_BY_FILE.get(jsonl_path.name)
    if budget is None:
        # Sweep files not in the known table used the 8192 budget; the original
        # 7-config (t0.6) phase used 4096. No-think files used 512.
        if "_t0.6_" in stem or stem.endswith("_t0.6"):
            budget = 4096
        else:
            budget = PARETO_BUDGET

    alpha = None
    if method == "split":
        sampler_think = record.get("sampler_think")
        sampler_code = record.get("sampler_code")
        temp_think = record.get("temp_think")
        temp_code = record.get("temp_code")
    elif method == "pless_alpha":
        m = re.search(r"_a(\d+(?:\.\d+)?)_", stem)
        alpha = float(m.group(1)) if m else None
        sampler_think = sampler_code = "pless_alpha"
        temp_think = temp_code = record.get("temperature")
    else:  # uniform think (temp / pless / pless_norm)
        sampler_think = sampler_code = method
        temp_think = temp_code = record.get("temperature")

    return {
        "file": jsonl_path.name,
        "model": record.get("model"),
        "method": method,
        "sampler_think": sampler_think,
        "temp_think": temp_think,
        "sampler_code": sampler_code,
        "temp_code": temp_code,
        "alpha": alpha,
        "max_tokens": budget,
    }


def task_results_from_metrics(metrics: dict) -> list[dict]:
    """Shape the on-disk per_task into the {pass_results, num_correct} the
    estimators expect, so pass@k matches the metrics JSON exactly."""
    return [
        {
            "task_id": pt["task_id"],
            "num_correct": pt["num_correct"],
            "pass_results": pt["pass_results"],
            "num_distinct_correct": pt.get("num_distinct_correct", 0),
        }
        for pt in metrics["per_task"]
    ]


def analyze_config(jsonl_path: Path, metrics_path: Path, tokenizer,
                   dataset="mbpp", max_tokens=None, delimiter="think") -> dict | None:
    """Build the full per-config row (metadata + accuracy + efficiency).

    `delimiter` selects the think-span boundary (see :func:`build_sample_rows`):
    "think" (default, `</think>` — native reasoning models) or "fence" (first
    ```` ``` ```` — instruct models induced into CoT, where `</think>` emits
    unreliably).
    """
    if not metrics_path.exists():
        print(f"  SKIP {jsonl_path.name}: no metrics JSON "
              f"(run `python -m bench.eval` first)")
        return None
    records = load_results(jsonl_path)
    if not records or "samples_with_thinking" not in records[0]:
        print(f"  SKIP {jsonl_path.name}: no thinking traces")
        return None
    metrics = load_metrics(metrics_path)
    meta = config_meta(records[0], jsonl_path, dataset, max_tokens)

    # Length stats (from JSONL) and pass@k (from metrics) must cover the SAME
    # task set — restrict both to the JSONL∩metrics intersection.
    jsonl_ids = {r["task_id"] for r in records}
    task_results = [tr for tr in task_results_from_metrics(metrics)
                    if tr["task_id"] in jsonl_ids]
    dropped = len(metrics["per_task"]) - len(task_results)
    if dropped:
        print(f"    note: {jsonl_path.name} — {dropped} metrics task(s) absent "
              f"from JSONL; analyzing {len(task_results)} common tasks")
    if not task_results:
        print(f"  SKIP {jsonl_path.name}: no common tasks between JSONL and metrics")
        return None
    n_per_task = metrics.get("num_samples_per_task") or len(task_results[0]["pass_results"])
    pak = compute_pass_at_k(task_results, K_VALUES)
    cov, _ = compute_cover_at_t(task_results, T_VALUES, n_per_task)

    # APPS: use the executor's per-sample `extraction_success` as the
    # authoritative has-code signal (matches what was actually evaluated).
    has_code_by_task = None
    if dataset == "apps":
        has_code_by_task = {
            pt["task_id"]: pt["extraction_success"]
            for pt in metrics["per_task"] if "extraction_success" in pt
        } or None

    rows = build_sample_rows(records, metrics, tokenizer, meta["max_tokens"],
                             has_code_by_task=has_code_by_task, delimiter=delimiter)
    agg = aggregate_rows(rows)

    out = {**meta, "think_delimiter": delimiter, "n_tasks": len(task_results), **agg}
    for k in K_VALUES:
        out[f"pass@{k}"] = pak.get(str(k))
    for t in T_VALUES:
        out[f"cov@{t}"] = cov.get(str(t))
    out["_rows"] = rows  # kept in-memory for distribution plot; not written to CSV
    return out


# ── outputs ─────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "file", "method", "label", "source", "difficulty",
    "sampler_think", "temp_think", "sampler_code", "temp_code",
    "top_p", "top_k", "alpha", "max_tokens", "n_tasks", "n_samples", "completion_rate",
    "truncation_rate", "malformed_rate", "near_cap_rate",
    "mean_think_tokens", "median_think_tokens", "p90_think_tokens",
    "mean_think_tokens_completed", "median_think_tokens_completed",
    "mean_think_chars", "conditional_correctness",
    "pass@1", "pass@5", "pass@10", "cov@0.3", "cov@0.5",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# Above this truncation rate a config is a context-limited failure, not "short
# reasoning" — its low mean length is an artifact of hitting the cap, so it must
# not be presented as Pareto-optimal.
MAX_TRUNC_FOR_FRONTIER = 0.25


def pareto_dominant(configs: list[dict]) -> list[dict]:
    """Configs not dominated on (shorter median_think_tokens, >= pass@1).

    Length axis is the MEDIAN over all samples — robust to the cap-pinned
    truncated tail and budget-insensitive (when <50% truncate), unlike the mean
    which bakes in `n_trunc × cap`. Tolerance: dominated only if another is
    strictly shorter AND not more than 1 absolute pass@1 point worse. Configs
    above MAX_TRUNC_FOR_FRONTIER truncation are excluded.
    """
    usable = [c for c in configs
              if c.get("median_think_tokens") is not None
              and c.get("pass@1") is not None
              and (c.get("truncation_rate") or 0) <= MAX_TRUNC_FOR_FRONTIER]
    frontier = []
    for c in usable:
        dominated = False
        for o in usable:
            if o is c:
                continue
            if (o["median_think_tokens"] < c["median_think_tokens"]
                    and o["pass@1"] >= c["pass@1"] - 0.01):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=lambda c: c["median_think_tokens"])


def config_label(c: dict) -> str:
    """Human label that does NOT disguise a unified (non-split) run as a split.

    Split runs render as "think → code"; single-sampler runs render as
    "<sampler> <temp> (unified, no split)".
    """
    if c.get("label"):  # APPS configs carry an explicit descriptive label
        return c["label"]
    if c.get("alpha") is not None:
        return f"pless_alpha α={c['alpha']} (unified, no split)"
    if c.get("method") == "split":
        return (f"{c['sampler_think']} {c['temp_think']} → "
                f"{c['sampler_code']} {c['temp_code']}")
    return f"{c['sampler_think']} {c['temp_think']} (unified, no split)"


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _model_label(configs: list[dict]) -> str:
    """Model name for report/plot titles — taken from the data, not hardcoded."""
    for c in configs:
        if c.get("model"):
            return c["model"]
    return "model"


def write_report(configs: list[dict], dataset: str, path: Path,
                 tokenizer_label: str = "tokenizer") -> None:
    pareto = [c for c in configs if c["max_tokens"] == PARETO_BUDGET]
    model = _model_label(configs)
    lines = [
        f"# CoT Token-Efficiency vs Accuracy — {model} / {dataset.upper()}\n",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Configs analyzed:** {len(configs)} "
        f"({len(pareto)} at the {PARETO_BUDGET}-token budget used for the frontier)\n",
        f"Think length is measured in **tokens** (`{tokenizer_label}`). Efficiency is "
        "decomposed per arXiv:2602.09805 into completion rate, conditional correctness "
        "(pass rate among completed samples), and think length.\n",
        # ── Grounded column definitions (computed in bench/eval/cot_efficiency.py
        #    aggregate_rows + bench/eval/metrics.py; see source for exact code).
        "## Column definitions\n",
        "A *sample* is one generation; each problem has 10 samples. A sample is "
        "**completed** iff it has a closing `</think>` (the think phase finished, "
        "not truncated at the cap) AND code was extracted (for APPS, the executor's "
        "`extraction_success`); **truncated** iff it has no `</think>`.\n",
        "- **budget** — generation cap (`--max-new-tokens`); all configs here share it.",
        "- **compl%** (`completion_rate`) — % of samples that completed "
        "(`#completed / #samples`).",
        "- **trunc%** (`truncation_rate`) — % of samples with no closing `</think>` "
        "(think ran into the token cap). Primary truncation signal is the missing "
        "`</think>`, not the token count.",
        "- **cond-correctness** (`conditional_correctness`) — pass rate *among completed samples "
        "only*: `#(completed & correct) / #completed`. Answers \"given it finished "
        "reasoning, did the code pass all hidden tests?\" Equals `pass@1 / compl%`.",
        "- **mean think tok** (`mean_think_tokens`) — mean think-block length over "
        "**ALL** samples, in tokens (think = text between `<think>` and `</think>`; "
        "for prompt-injected `<think>` models, text up to `</think>`). **Includes "
        "truncated samples**, which contribute their length-at-cut (≈ the cap) — so "
        "this is inflated toward the cap for configs that truncate, and NOT "
        "comparable across configs with different trunc%.",
        "- **median (all)** (`median_think_tokens`) — median think length over **ALL** "
        "samples; the Pareto-frontier axis. Budget-insensitive (the truncated cap "
        "*value* doesn't move it) but **biased UPWARD by truncation rate**: truncated "
        "samples occupy the top ranks, so higher trunc% pushes the median to a higher "
        "percentile of the completed distribution. So a config sitting right may be "
        "there partly because it truncates more, not only because it reasons longer — "
        "read the trunc% (marker size) alongside. It is the least-misleading single "
        "length stat here (unlike median-done it won't falsely make a truncating config "
        "look short), but it is NOT fully decoupled from truncation.",
        "- **median (done)** (`median_think_tokens_completed`) — median think length "
        "over **completed samples only**. Cap-robust but **censored**: a config's "
        "longest traces were truncated out, biasing its completed-median low. Don't "
        "read it as \"who reasons shorter\" across configs with different trunc%.",
        "- **pass@1 / pass@10** — unbiased pass@k (human-eval estimator, "
        "`metrics.compute_pass_at_k`) over each problem's `(num_correct, n=10)`. "
        "pass@1 = overall fraction of single samples that pass; pass@10 (k=n=10) = "
        "fraction of problems solved by **≥1** of the 10 samples (coverage).",
        "- **cov@0.3 / cov@0.5** (CSV) — % of problems with ≥30% / ≥50% of their "
        "samples correct (`num_correct ≥ t·n`).",
        "\n**Coherence checks (should hold every run):**",
        "- `pass@1 ≈ compl% × cond-correctness` per row, with residual = "
        "`#(passed but NOT completed) / n`. Passing requires extracted code that "
        "passes the tests, which *usually* implies a closed `</think>` — but a "
        "**truncated** trace can still contain a passing code block, and such samples "
        "count in pass@1 yet not in `completed`. So pass@1 can slightly EXCEED "
        "compl%×cond; the residual is audited per-config below (0 = identity holds exactly).",
        "- `pass@10 ≥ pass@5 ≥ pass@1` per row (monotone in k).",
        "\n**No single length stat is clean under differing truncation — each is "
        "biased a different way:** `mean think tok` = avg tokens *spent* (counts "
        "truncated at the cap → biased UP + budget-dependent); `median (all)` = typical "
        "length but biased UP by truncation rate (rank effect; budget-insensitive — the "
        "frontier axis, least-misleading); `median (done)` = typical *finished* length, "
        "biased DOWN (censored — drops the truncated long tail). For a truncating "
        "config: mean ≫ median(all) > median(done). So **read trunc% (marker size) "
        "alongside any length**; a clean cross-config length comparison needs a budget "
        "where all configs complete (no truncation).\n",
        "## Per-config decomposition\n",
        "| Config (think→code) | budget | compl% | trunc% | cond-correctness | "
        "mean think tok | median (all) | median (done) | pass@1 | pass@10 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(configs, key=lambda c: (-(c["max_tokens"] or 0),
                                            -(c.get("pass@1") or 0))):
        label = config_label(c)
        lines.append(
            f"| {label} | {c['max_tokens']} "
            f"| {_fmt((c.get('completion_rate') or 0) * 100, 1)} "
            f"| {_fmt((c.get('truncation_rate') or 0) * 100, 1)} "
            f"| {_fmt(c.get('conditional_correctness'))} "
            f"| {_fmt(c.get('mean_think_tokens'), 0)} "
            f"| {_fmt(c.get('median_think_tokens'), 0)} "
            f"| {_fmt(c.get('median_think_tokens_completed'), 0)} "
            f"| {_fmt(c.get('pass@1'))} | {_fmt(c.get('pass@10'))} |"
        )

    # Coherence audit: per-config count of passing-but-not-completed (truncated)
    # samples — the residual in `pass@1 ≈ compl% × cond-correctness`.
    leaks = [(config_label(c), c.get("passed_not_completed") or 0) for c in configs]
    nonzero = [f"{lbl} ({k})" for lbl, k in leaks if k]
    if nonzero:
        lines.append(
            "\n**Coherence audit** — configs where a passing sample was truncated "
            "(counted in pass@1 but not `completed`, so pass@1 > compl%×cond by that "
            f"many samples / n): {', '.join(nonzero)}. All other configs: residual 0 "
            "(identity exact)."
        )
    else:
        lines.append(
            "\n**Coherence audit** — `pass@1 == compl% × cond-correctness` exactly for "
            "every config (no passing sample was truncated)."
        )

    lines += [
        f"\n## Pareto-dominant configs ({PARETO_BUDGET}-token budget)\n",
        f"Not dominated on (shorter median think tokens, pass@1 within 1pt); "
        f"configs with >{MAX_TRUNC_FOR_FRONTIER:.0%} truncation excluded as "
        f"context-limited failures. Length axis = `median (all)` — budget-insensitive "
        f"but biased UP by truncation rate (so a config may rank longer partly because "
        f"it truncates more); read trunc% alongside:\n",
        "| Config | median (all) | trunc% | pass@1 | pass@10 | cond-correctness |",
        "|---|---|---|---|---|---|",
    ]
    for c in pareto_dominant(pareto):
        label = config_label(c)
        lines.append(
            f"| {label} | {_fmt(c.get('median_think_tokens'), 0)} "
            f"| {_fmt((c.get('truncation_rate') or 0) * 100, 1)} "
            f"| {_fmt(c.get('pass@1'))} | {_fmt(c.get('pass@10'))} "
            f"| {_fmt(c.get('conditional_correctness'))} |"
        )

    lines += [
        "\n## Limitations\n",
        "- Single model / single difficulty; no cross-model generalization.",
        "- Samplers compared are whatever was generated for this run; greedy is "
        "excluded by design (Qwen discourages it in thinking mode).",
        "- Token counts are analysis-time estimates (tokenizer special-token handling "
        "may differ slightly from generation time).",
        f"- Truncation at the {PARETO_BUDGET} cap censors the upper tail: truncated "
        "samples are pinned near the cap (inflating the *mean* for configs that "
        "truncate) yet underestimate their true length. So length is NOT comparable "
        "across configs with different trunc% — see Column definitions.",
        "- Stochastic samplers run at a fixed temperature, not matched effective "
        "entropy, so cross-config pass@1 differences mix sampler + operating point.",
        "- Correlational across independently-generated configs (no paired seeds).",
    ]
    path.write_text("\n".join(lines) + "\n")


def make_plots(configs: list[dict], dataset: str, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir.mkdir(parents=True, exist_ok=True)
    model = _model_label(configs)
    pareto = [c for c in configs
              if c["max_tokens"] == PARETO_BUDGET
              and c.get("mean_think_tokens") is not None]

    # One distinct (color, marker) per config — keying on sampler_think collapsed
    # all APPS temp-family points to one color. Numbered tags at each point plus a
    # numbered legend make them identifiable even when dots overlap.
    palette = list(plt.cm.tab10.colors)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    def _scatter(metric: str, fname: str):
        # x = MEDIAN think tokens (robust, budget-insensitive "typical length");
        # marker SIZE ∝ truncation% (the tail/cost, decoupled into its own
        # channel so it isn't confounded into the x-position as the mean would).
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, c in enumerate(pareto):
            x, y = c.get("median_think_tokens"), c.get(metric)
            tr = (c.get("truncation_rate") or 0)
            size = 80 + 1300 * tr   # 0% -> 80, ~14% -> ~262
            ax.scatter(x, y, color=palette[i % len(palette)],
                       marker=markers[i % len(markers)], s=size,
                       edgecolors="black", linewidths=0.6, zorder=3,
                       label=f"{i + 1}. {config_label(c)} — trunc {tr*100:.0f}%")
            ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, fontweight="bold", zorder=4)
        # No "Pareto frontier" line: the x-axis (median-all) is biased upward by
        # truncation, so a connecting line would imply a clean length↔accuracy
        # tradeoff the confounded axis can't support. The 6 trunc%-sized labeled
        # points show position directly; non-dominance is in the report table.
        ax.set_xlabel("median think tokens (all samples)  —  marker size ∝ truncation %")
        ax.set_ylabel(metric)
        ax.set_title(f"CoT length vs {metric} — {model} / {dataset.upper()} "
                     f"({PARETO_BUDGET}-tok budget)")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        # Legend below the plot (labels are long); 2 columns.
        ax.legend(title="config  (bigger marker = more truncation)", fontsize=8,
                  loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
        fig.savefig(fig_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _scatter("pass@1", "cot_pareto_pass_at_1.png")
    _scatter("pass@10", "cot_pareto_pass_at_10.png")

    # Think-length distribution per sampler over ALL samples (not completed-only):
    # truncated traces pile up near the cap, so a config that fails to terminate
    # shows a visible mass at the top instead of being censored out (which made
    # the completed-only version misleadingly uniform).
    by_sampler: dict[str, list[int]] = {}
    for c in pareto:
        for r in c.get("_rows", []):
            if r["think_tokens"] is not None:
                by_sampler.setdefault(c["sampler_think"], []).append(r["think_tokens"])
    if by_sampler:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        samplers = sorted(by_sampler)
        ax.violinplot([by_sampler[s] for s in samplers], showmedians=True)
        # Annotate each violin with its truncation% (the mass at the cap).
        trunc_by_label = {c["sampler_think"]: (c.get("truncation_rate") or 0) for c in pareto}
        for i, s in enumerate(samplers, start=1):
            ax.annotate(f"trunc {trunc_by_label.get(s, 0)*100:.0f}%",
                        (i, max(by_sampler[s])), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xticks(range(1, len(samplers) + 1))
        ax.set_xticklabels(samplers, rotation=20, ha="right")
        ax.set_ylabel("think length (tokens, ALL samples; truncated pinned at cap)")
        ax.set_title(f"Think-length distribution by sampler — {model} / {dataset.upper()}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "cot_think_length_dist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── verification ────────────────────────────────────────────────────────────

def verify_config(jsonl_path: Path, metrics_path: Path, tokenizer) -> None:
    """Assert the index-alignment invariant and that recomputed pass@1 matches disk."""
    records = load_results(jsonl_path)
    metrics = load_metrics(metrics_path)
    pass_by_task = {pt["task_id"]: pt["pass_results"] for pt in metrics["per_task"]}
    checked = skipped = 0
    for rec in records:
        labels = pass_by_task.get(rec["task_id"])
        if labels is None:
            skipped += 1
            continue
        swt = rec.get("samples_with_thinking") or []
        assert len(swt) == len(labels), (
            f"alignment violation on task {rec['task_id']}: "
            f"{len(swt)} vs {len(labels)}")
        checked += 1

    pak = compute_pass_at_k(task_results_from_metrics(metrics), K_VALUES)
    on_disk = metrics["pass_at_k"]["1"]
    assert abs(pak["1"] - on_disk) < 1e-6, (
        f"pass@1 mismatch: recomputed {pak['1']} vs on-disk {on_disk}")

    # Spot-check a couple of think-length extractions.
    for rec in records[:1]:
        for swt in rec["samples_with_thinking"][:2]:
            think, closed = extract_think_span(swt)
            n_tok, n_chars = measure_lengths(think, tokenizer)
            ratio = (n_tok / n_chars) if (n_tok and n_chars) else float("nan")
            print(f"    [{jsonl_path.name}] closed={closed} "
                  f"chars={n_chars} tokens={n_tok} tok/char={ratio:.3f}")
    note = f" ({skipped} JSONL task(s) not in metrics, skipped)" if skipped else ""
    print(f"  OK {jsonl_path.name}: pass@1={pak['1']:.4f} matches on-disk "
          f"{on_disk:.4f}; alignment holds on {checked} tasks{note}.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: tokenizer '{model_id}' unavailable ({e}); "
              f"falling back to char-only lengths.")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--dataset", default="mbpp", choices=["mbpp", "humaneval", "apps"])
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Generation budget of the run (REQUIRED for --dataset apps; "
                         "used for near_cap + as the frontier budget). MBPP/HE infer it.")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--alpha-dir", type=Path, default=None,
                    help="Optional dir with pless_alpha_think_*.jsonl files")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--think-delimiter", choices=["think", "fence"],
                    default="think",
                    help="Think-span boundary. 'think' (default): </think> "
                         "terminator — native reasoning models (Qwen3, "
                         "DeepSeek-R1-Distill). 'fence': first ```python code "
                         "fence — instruct models induced into CoT, where "
                         "</think> emits unreliably (~53% on Qwen2.5-Coder).")
    ap.add_argument("--verify", action="store_true",
                    help="Run alignment + pass@1 assertions and exit.")
    args = ap.parse_args()

    if args.dataset == "apps" and args.max_tokens is None:
        ap.error("--max-tokens is required for --dataset apps "
                 "(the run's --max-new-tokens, e.g. 16384)")

    # For APPS every config shares the run budget; make it the frontier budget so
    # the report/plots include all configs (PARETO_BUDGET is the MBPP 8192 default).
    if args.dataset == "apps":
        global PARETO_BUDGET
        PARETO_BUDGET = args.max_tokens

    out_dir = args.output_dir or (args.results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None if args.no_tokens else load_tokenizer(args.tokenizer)

    extra = [args.alpha_dir] if args.alpha_dir else []
    files = discover_config_files(args.results_dir, extra, dataset=args.dataset)
    if args.limit_files:
        files = files[: args.limit_files]
    print(f"Discovered {len(files)} config file(s) under {args.results_dir}")

    if args.verify:
        for jsonl_path, metrics_path in files:
            if metrics_path.exists():
                verify_config(jsonl_path, metrics_path, tokenizer)
        return

    configs = []
    for jsonl_path, metrics_path in files:
        c = analyze_config(jsonl_path, metrics_path, tokenizer,
                           dataset=args.dataset, max_tokens=args.max_tokens,
                           delimiter=args.think_delimiter)
        if c is not None:
            configs.append(c)
            print(f"  {c['file']}: budget={c['max_tokens']} "
                  f"mean_think_tok={_fmt(c.get('mean_think_tokens'), 0)} "
                  f"pass@1={_fmt(c.get('pass@1'))} "
                  f"trunc%={_fmt((c.get('truncation_rate') or 0) * 100, 1)}")

    if not configs:
        print("No analyzable configs found.")
        return

    csv_path = out_dir / f"cot_efficiency_{args.dataset}.csv"
    report_path = out_dir / f"cot_efficiency_{args.dataset}_report.md"
    write_csv(configs, csv_path)
    tok_label = "char counts (--no-tokens)" if args.no_tokens else args.tokenizer
    write_report(configs, args.dataset, report_path, tokenizer_label=tok_label)
    make_plots(configs, args.dataset, out_dir / "figures")
    print(f"\nCSV:    {csv_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
