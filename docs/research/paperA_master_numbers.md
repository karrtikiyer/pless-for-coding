# Paper A — Master Numbers (single source of truth)

**Purpose.** Every headline number the Narrative-A manuscript cites must appear here first, pulled
**live** from a canonical (non-superseded) results file — never from memory. Regenerate before each
drafting session. Assembled 2026-07-30.

**How each block was produced (reproducible):**
- **APPS:** `SET=deepseek_fixed|qwen PYTHONPATH=. uv run python scripts/build_decoder_comparison_table.py`
  → regenerates `docs/decoder_comparison_cot_apps_{deepseek,qwen3}.md`. pass@k recomputed unbiased
  (Chen 2021) and cross-checked vs stored — both files reported **✓ agree to tolerance** this run.
- **MBPP:** `results/pless_full_mbpp_results/analysis/comparison_report.md` (500 problems, vs Yi et al.
  arXiv:2402.06925 Table 26).
- **HumanEval:** `results/pless_human_eval_results/full_precision_results/analysis/report.md`
  (164 tasks × 10).

---

## ⚠ Correction to the scoping plan (hard-truth #2 was overstated)

The plan file's hard-truth #2 said *"HumanEval consolidated: pless 0.471 < temp 0.624."* That figure
came from `results/analysis/consolidated_report.md`, whose per-config rows are **duplicated with
conflicting values** (e.g. `Qwen/Qwen-7B | pless | 0.6` appears 4× as 0.128/0.314/0.140/0.354). It is
**not usable for an aggregate** without dedup and is NOT reproducible from the clean per-model report.

The clean per-model HumanEval numbers (below) show a **pass@1-vs-pass@10 tradeoff**, not a blanket
loss: pless is competitive-or-best at pass@1 and loses at pass@10 on diversity. **Use the tradeoff
framing, not "pless loses on HumanEval."** (Do NOT cite `consolidated_report.md` or
`consolidated_summary.csv` aggregates in the paper until re-derived with dedup.)

---

## 1. APPS — reasoning-model CoT (the core of the paper)

Full tables: `docs/decoder_comparison_cot_apps_{deepseek,qwen3}.md` (regenerated live this session).
252 tasks × 10, thinking on. Key rows:

### DeepSeek-R1-Distill-Llama-8B (`results/_deepseek_fixed_full252/…`, post-#45488-fix)
| Config | pass@1 | pass@10 | trunc% |
|---|---|---|---|
| pless α=5 (prevention) | 0.483 | 0.714 | 0.3 |
| temp t1.0 (p0.95) | 0.480 | **0.726** | 0.0 |
| pless α=4 | 0.473 | 0.710 | 1.4 |
| adaptive (1-chop) | 0.457 | 0.687 | 7.1 |
| **pless α=2 (default)** | **0.392** | **0.627** | **41.8** |
| **pless_norm (default)** | **0.392** | 0.663 | **41.7** |

### Qwen3-8B (`results/pless_recovery_full252/…`, `…_cot_efficiency_vllm/…_all_252/`)
| Config | pass@1 | pass@10 | trunc% |
|---|---|---|---|
| temp p0.95 @T1.0 | **0.705** | 0.821 | 0.2 |
| pless α=4 | 0.696 | 0.821 | 1.4 |
| pless α=5 | 0.686 | 0.833 | 0.6 |
| adaptive (1-chop) | 0.682 | **0.845** | 2.7 |
| **pless α=2 (default)** | **0.625** | 0.825 | **14.5** |
| **pless_norm (default)** | 0.629 | 0.829 | 16.0 |

**Reading:** default pless (α=2) catastrophically loops on CoT (DeepSeek 41.8% / Qwen 14.5%
truncation) → worst pass@1. Raising α removes the loop (trunc → ~0–1%) and recovers pass@1, but the
best pless-α only **ties** well-tuned temperature (DeepSeek α=5 0.483 ≈ temp 0.480; Qwen temp 0.705 >
α=5 0.686). This is the "silently breaks on long CoT, α is the diagnostic knob, still only ties temp"
spine.

---

## 2. MBPP — base-model short completions (`comparison_report.md`, 500 problems, vs Yi et al.)

**Llama-2-7B (base) — pass@1 ranking vs 15 published decoding methods (rank / 19):**
| Method | pass@1 (%) | Rank |
|---|---|---|
| P-Less Norm (t=0.6) | 22.3 | **1 / 19** |
| P-Less (t=0.6) | 22.2 | 2 |
| P-Less (t=1.0) | 19.8 | 4 |
| P-Less Norm (t=1.0) | 19.1 | 7 |

**Llama-2-7B-Chat — pless beats temperature at pass@1** (P-Less t=0.6 **20.5** vs Temperature t=0.7
**17.8**), but temperature wins pass@10 (30.2 vs pless ~21.4 — the same diversity gap as HumanEval).

**Reading:** on short base-model completions, the hyperparameter-free method is genuinely
**top-ranked at pass@1** — this is the "where it helps" half.

⚠ Data-quality note: `results/pless_mbpp_results/` (older) reports Llama pless_t1.0 = 28.5 vs the
canonical `pless_full_mbpp_results/` 19.8 — different run/fix state. Use `pless_full_mbpp_results/`.
The survey-agent's "Qwen-7B pless_norm t1.0 = 35.7% rank 1/18" is from the older tree — re-pull from
the canonical corpus before citing.

---

## 3. HumanEval — the pass@1-vs-pass@10 (accuracy-vs-diversity) tradeoff (`report.md`, 164 tasks × 10)

pass@1 → pass@10 (%), selected:
| Model | pless (best variant) | temp/top_p (best) | greedy |
|---|---|---|---|
| CodeLlama-7b-Instruct | p_less 36.1 → 38.4 | temp_0.7 36.2 → **62.8** | 36.0 |
| Codestral-22B | **p_less 78.0** → 84.8 | temp_0.7 72.6 → **91.5** | 75.6 |
| Qwen2.5-Coder-7B-Instruct | pless(t0.6) **87.5** → 87.8 | temp_0.7 79.0 → **94.5** | 84.1 |
| Qwen3-Coder-30B-A3B-Instruct | pless(t0.6) 78.9 → 79.9 | temp_0.7 76.2 → **86.6** | 75.6 |

**Reading (the honest, defensible claim):** pless behaves like a **high-accuracy / low-diversity**
decoder — it matches or beats temperature and greedy at **pass@1** (Codestral p_less 78.0 > temp_0.7
72.6; Qwen2.5 pless 87.5 > greedy 84.1) but is **dominated at pass@10** because temperature/top-p
cover more distinct solutions. This is the *same* mechanism as the CoT loops (peaked → narrow
survivor set) — it just helps at pass@1 and hurts at pass@10. Unifying thread for the paper.

---

## 4. Superseded / excluded runs (do NOT cite)

| Excluded | Why |
|---|---|
| `results/pless_cot_efficiency_vllm/deepseek-ai--…/ATCODER_interview/` | pre-#45488 (mangled prompts); pass@1 ~0.16–0.33 vs fixed 0.39–0.48 |
| DeepSeek arm of `results/pless_recovery_full252/` (α=3/4/5 ≈ 0.30) | pre-fix; superseded by `_deepseek_fixed_full252` (Qwen arm stands) |
| `results/pless_cot_efficiency_vllm/Qwen--…/ATCODER_interview/` (100-task) | explicit `SUPERSEDED.md`; pre-E7 swap bug (spurious 0.508) |
| `results/pless_mbpp_results/` | older; disagrees with canonical 500-problem `pless_full_mbpp_results/` |
| `results/analysis/consolidated_{report.md,summary.csv}` aggregates | duplicated/conflicting rows — not aggregatable without dedup |
| `…/analysis_before_fix/`, `*_backup_*`, `ATCODER_introductory` (12-task) | pre-fix snapshots / smoke tests |

## 5. Provenance
All numbers above pulled 2026-07-30 from the files named in each section. APPS via live
`build_decoder_comparison_table.py` run (cross-verification ✓). No number here is from memory.
