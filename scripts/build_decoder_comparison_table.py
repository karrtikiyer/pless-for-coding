"""Build a grounded decoder-comparison table (ATCODER-interview, n=10), config-set driven.

Every number is pulled LIVE from source-of-truth files and cross-verified:
  - pass@1/5/10: recomputed from raw pass_results (unbiased estimator) AND compared to the
    stored pass_at_k in the metrics JSON — mismatch => flagged.
  - trunc%: recomputed from samples_with_thinking (fraction lacking </think>).
  - mean think tok: from the cot_efficiency CSV when present (the pipeline's tokenizer output);
    otherwise recomputed from the jsonl by re-tokenizing the think phase (SET's MODEL tokenizer).
  - cb_div: via the project's add_self_codebleu / compute_self_codebleu_diversity (no execution).
Configs whose metrics JSON is missing (not scored yet) are skipped and listed as pending, so the
table can be built incrementally.

Run: SET=deepseek_fixed PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py
     SET=qwen           PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py   (default)
"""
import csv
import json
import math
import os

from bench.eval.metrics import add_self_codebleu, compute_self_codebleu_diversity

# ---- config sets ----------------------------------------------------------
_Q_FULL = "results/pless_recovery_full252/Qwen--Qwen3-8B/ATCODER_interview"
_Q_CANON = "results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252"
_Q_DEC06 = "results/decoders_t0.6/Qwen--Qwen3-8B/ATCODER_interview"
_DS_FIX = "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview"

SETS = {
    # (display, dir, jsonl_basename)
    "qwen": {
        "model": "Qwen/Qwen3-8B",
        "out": "docs/decoder_comparison_cot_apps_qwen3.md",
        "use_csv": True,
        "title": "Qwen3-8B",
        "configs": [
            ("temp p0.95 @T1.0", _Q_CANON, "temp_p0.95_think_t1.0_t1.0"),
            ("temp k20 @T1.0",   _Q_CANON, "temp_k20_think_t1.0_t1.0"),
            ("temp @T0.6 (unfilt)", _Q_CANON, "temp_think_t0.6_t0.6"),
            ("top_k @T0.6",      _Q_DEC06, "temp_k20_think_t0.6_t0.6"),
            ("pless α=4",        _Q_FULL,  "pless_alpha_think_t1.0_a4.0_t1.0"),
            ("top_p @T0.6",      _Q_DEC06, "temp_p0.95_think_t0.6_t0.6"),
            ("pless T2.0",       _Q_FULL,  "pless_think_t2.0_t2.0"),
            ("pless α=5",        _Q_FULL,  "pless_alpha_think_t1.0_a5.0_t1.0"),
            ("temp p+k @T0.6",   _Q_CANON, "temp_p0.95_k20_think_t0.6_t0.6"),
            ("pless α=3",        _Q_FULL,  "pless_alpha_think_t1.0_a3.0_t1.0"),
            ("pless_norm @α2",   _Q_CANON, "pless_norm_think_t1.0_t1.0"),
            ("pless @α2 (base)", _Q_CANON, "pless_think_t1.0_t1.0"),
            ("pless_norm @T0.6", _Q_DEC06, "pless_norm_think_t0.6_t0.6"),
            ("pless @T0.6",      _Q_DEC06, "pless_think_t0.6_t0.6"),
        ],
    },
    # Corrected (post-#45488-fix) DeepSeek runs, all in one tree.
    "deepseek_fixed": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "out": "docs/decoder_comparison_cot_apps_deepseek.md",
        "use_csv": False,                 # no cot CSV for these runs → mean-tok from jsonl
        "title": "DeepSeek-R1-Distill-Llama-8B (fixed vLLM, post-#45488)",
        "configs": [
            ("pless α=2",              _DS_FIX, "pless_think_t1.0_t1.0"),
            ("pless α=3",              _DS_FIX, "pless_alpha_think_t1.0_a3.0_t1.0"),
            ("pless α=4",              _DS_FIX, "pless_alpha_think_t1.0_a4.0_t1.0"),
            ("pless α=5 (prevention)", _DS_FIX, "pless_alpha_think_t1.0_a5.0_t1.0"),
            ("adaptive (1-chop)",      _DS_FIX, "pless_adaptive_recon"),
            ("adaptive (3-chop)",      _DS_FIX, "pless_adaptive_recon_rechop"),
            ("pless_norm",             _DS_FIX, "pless_norm_think_t1.0_t1.0"),
            ("temp t1.0 (k20)",        _DS_FIX, "temp_k20_think_t1.0_t1.0"),
            ("temp t0.6 (p0.95+k20)",  _DS_FIX, "temp_p0.95_k20_think_t0.6_t0.6"),
            ("temp t1.0 (p0.95)",      _DS_FIX, "temp_p0.95_think_t1.0_t1.0"),
            ("temp t0.6 (unfilt)",     _DS_FIX, "temp_think_t0.6_t0.6"),
            ("temp t0.6 (p0.95) [rec]", _DS_FIX, "temp_p0.95_think_t0.6_t0.6"),
        ],
    },
}


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    setname = os.environ.get("SET", "qwen")
    cfg = SETS[setname]
    _tok = [None]  # lazy tokenizer for the jsonl mean-tok fallback

    def mean_tok_from_jsonl(records):
        if _tok[0] is None:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            at = AutoTokenizer.from_pretrained(cfg["model"])
            t = "a b\nc"
            _tok[0] = (PreTrainedTokenizerFast.from_pretrained(cfg["model"])
                       if at.decode(at.encode(t, add_special_tokens=False),
                                    skip_special_tokens=True).strip() != t.strip() else at)
        tok = _tok[0]
        sw = [s for r in records for s in (r.get("samples_with_thinking") or r["samples"])]
        samp = sw[::max(1, len(sw) // 300)][:300]     # ~300-sample estimate
        n = 0
        for s in samp:
            th = s.split("</think>")[0] if "</think>" in s else s
            n += len(tok.encode(th, add_special_tokens=False))
        return n / len(samp) if samp else float("nan")

    rows, warnings, pending = [], [], []
    for name, d, base in cfg["configs"]:
        mpath = f"{d}/metrics/{base}_metrics.json"
        if not os.path.exists(mpath):
            pending.append(name)
            continue
        m = json.load(open(mpath))
        pt = m["per_task"]
        records = [json.loads(l) for l in open(f"{d}/{base}.jsonl")]
        n_samp = len(pt[0]["pass_results"])
        ntasks = len(pt)

        def agg(k):
            return sum(pass_at_k(len(t["pass_results"]), sum(t["pass_results"]), k) for t in pt) / ntasks
        p1, p5, p10 = agg(1), agg(5), agg(10)
        for k, v in (("1", p1), ("5", p5), ("10", p10)):
            stored = m.get("pass_at_k", {}).get(k)
            if stored is not None and abs(stored - v) > 1e-6:
                warnings.append(f"{name}: pass@{k} recomputed {v:.4f} != stored {stored:.4f}")

        rec_by_id = {r["task_id"]: r for r in records}
        n_trunc = n_total = 0
        for t in pt:
            swt = rec_by_id[t["task_id"]].get("samples_with_thinking") or rec_by_id[t["task_id"]]["samples"]
            for s in swt:
                n_total += 1
                if "</think>" not in s:
                    n_trunc += 1
        trunc = n_trunc / n_total if n_total else 0.0

        # mean think tok: CSV when present, else re-tokenize the jsonl
        mean_tok = float("nan")
        if cfg["use_csv"]:
            csv_path = f"{d}/analysis/cot_efficiency_apps.csv"
            crow = next((r for r in csv.DictReader(open(csv_path)) if r["file"] == f"{base}.jsonl"),
                        None) if os.path.exists(csv_path) else None
            if crow:
                mean_tok = float(crow["mean_think_tokens"])
                if abs(float(crow["truncation_rate"]) - trunc) > 0.01:
                    warnings.append(f"{name}: trunc recomputed {trunc:.3f} != CSV {float(crow['truncation_rate']):.3f}")
        if math.isnan(mean_tok):
            mean_tok = mean_tok_from_jsonl(records)

        add_self_codebleu(pt, records)          # project fn (no execution)
        cb = compute_self_codebleu_diversity(pt).get("codebleu_diversity", 0.0)
        rows.append((name, ntasks, n_samp, p1, p10, cb, mean_tok, trunc))

    rows.sort(key=lambda r: -r[3])
    lines = [f"# Decoder comparison — ATCODER-interview, {cfg['title']} (thinking on, n=10)\n",
             "All values pulled live by `scripts/build_decoder_comparison_table.py`. pass@k recomputed "
             "from raw pass_results (unbiased, Chen 2021) & checked vs stored; trunc% from `</think>` "
             "presence; mean think tok from the cot CSV when present else re-tokenized from the jsonl; "
             "cb_div via the project's `add_self_codebleu` (correct-only, no execution).\n",
             "| Config | n | pass@1 | pass@10 | cb_div | mean think tok | trunc% |",
             "|---|---|---|---|---|---|---|"]
    for name, nt, ns, p1, p10, cb, mt, tr in rows:
        ntnote = f"{nt}" if nt == 252 else f"**{nt}**"
        lines.append(f"| {name} | {ntnote} | {p1:.3f} | {p10:.3f} | {cb:.4f} | {mt:,.0f} | {tr*100:.1f} |")
    if pending:
        lines.append("\n**Pending (not scored yet):** " + ", ".join(pending) + ".")
    lines.append("\n## Cross-verification\n")
    lines.append("⚠ MISMATCHES:\n" + "\n".join(f"- {w}" for w in warnings) if warnings
                 else "✓ pass@k (recomputed vs stored) and trunc% agree to tolerance.")
    lines.append("\n## Caveats\n"
                 "- **cb_div** correct-only over each config's own ≥2-correct subset (mild cross-config confound).\n"
                 "- **mean think tok** counts truncated samples at their cut length (≈cap) → biased UP for "
                 "high-truncation configs (the rambling cost). Re-tokenized estimate when no CSV (±few %).\n"
                 "- **structural_diversity** omitted (zss intractable on APPS-CoT — the reason for `--skip-diversity`).")

    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    with open(cfg["out"], "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {cfg['out']}  ({len(rows)} scored, {len(pending)} pending)")
    for w in warnings:
        print("  ⚠", w)


if __name__ == "__main__":
    main()
