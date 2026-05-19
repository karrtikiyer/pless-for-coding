# Bimodal-Entropy Experiment Plan

Experimental plan for measuring per-position Rényi entropy during code
generation to test the bimodal-entropy hypothesis that underpins the
theoretical claim in `renyi_alpha_pless_theory.md` §6.

---

## 1. Goal

Empirically verify or refute the claim:

> *Position-conditional next-token Rényi entropy `H_α(p_t | x_<t)` in
> code generation has a bimodal distribution: one mode near zero
> (syntactic positions) and a separated higher-entropy mode (semantic
> positions), with a visible gap in between.*

If bimodal: paper §4 ("Why p-less-α works for code") becomes a
measured property of code-token distributions, not an analogy to AdapT.
If unimodal: we report the negative finding honestly and refine §4 to
"asymmetric-by-shape adaptation regardless of bimodality."

---

## 2. What we log (per token-generation step)

Per position during sampling, write one JSONL record with:

| Field | Type | Why |
|---|---|---|
| `task_id` | int | identifies the MBPP problem |
| `sample_id` | int | which of the 3 samples (per problem) |
| `position` | int | step index within the sample |
| `token_id` | int | the sampled token's vocabulary index |
| `token_str` | str | decoded surface form (for token-class labeling later) |
| `sigma_p2` | float | `Σpᵢ²` (exact, what p-less uses) |
| `sigma_p3` | float | `Σpᵢ³` (for α=3 histogram) |
| `sigma_p5` | float | `Σpᵢ⁵` (for α=5 histogram) |
| `max_p` | float | `max(pᵢ)` (for `H_∞`) |
| `top32_probs` | list[float], len 32 | top-32 probabilities |
| `top32_indices` | list[int], len 32 | their vocabulary indices |

Per-record size: ~280 bytes. For ~45,000 positions total across both
models, ~12 MB.

---

## 3. Instrumentation diff

Two surgical changes — no regression risk to the existing samplers
when the new flag is not used.

### 3.1 `bench/generator.py`

Add an optional `entropy_log` parameter to `generate_samples(...)`:

```
def generate_samples(
    model, tokenizer, prompt_text, sampler_fn,
    n_samples, max_new_tokens, temperature, stop_strings,
    entropy_log: list[dict] | None = None,    # NEW
):
```

Inside the token-generation loop (around the existing `softmax → sample`
step), when `entropy_log is not None`:

```
probs = torch.softmax(logits[-1], dim=-1)
if entropy_log is not None:
    top_p, top_i = probs.topk(32)
    rec = {
        "position": step,
        "sigma_p2": (probs ** 2).sum().item(),
        "sigma_p3": (probs ** 3).sum().item(),
        "sigma_p5": (probs ** 5).sum().item(),
        "max_p":    probs.max().item(),
        "top32_probs":   top_p.cpu().tolist(),
        "top32_indices": top_i.cpu().tolist(),
    }
    entropy_log.append(rec)
next_token = sampler_fn(probs)
# … then fill in token_id, token_str on the rec:
if entropy_log is not None:
    rec["token_id"] = int(next_token)
    rec["token_str"] = tokenizer.decode([int(next_token)])
```

Two extra lines per sample: pass `entropy_log=[]` in, get back the
populated list. ~30 lines added total; the function signature stays
backwards-compatible (default `None`).

### 3.2 `bench/runner.py`

Add a CLI flag and pass it through:

```
parser.add_argument(
    "--log-entropy", action="store_true",
    help="Log per-position Rényi entropy stats to a sidecar "
         ".entropy.jsonl alongside the main results JSONL.",
)
```

In `main()`, when set, allocate a list per task, pass to
`generate_samples`, then write the list to a sidecar JSONL at the
same path as the result JSONL but with `.entropy.jsonl` suffix.

The sidecar path is `{out_path}.entropy.jsonl` (e.g.,
`results/pless_alpha_entropy/Qwen.../temp_t1.0.jsonl.entropy.jsonl`).

### 3.3 Plain temperature sampling — no new sampler needed

The experiment uses `--method temp --temperature 1.0` (plain
multinomial). Existing path. No interaction with the α-sweep code.

The reason for plain temperature, not p-less or p-less-α: we want to
observe the **raw distribution shape per position**, unmodified by any
truncation. The bimodality claim is about `H_α(p_t)` — the model's
output distribution — not about what the sampler keeps after pruning.

---

## 4. Run plan

| Parameter | Value | Rationale |
|---|---|---|
| Models | Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct | match the NAUADC pair |
| Benchmark | MBPP-full, 50-task random subset | reuse `results/pless_alpha_smoke/smoke_task_ids.txt` (same 50 IDs as the smoke + α=2 sanity gate) |
| Samples per problem | 3 | enough variance per position; not measuring pass@k |
| Sampler | `temp` at T=1.0 | unmodified distribution; matches existing `temp_t1.0` baselines for comparison |
| Max new tokens | 256 | typical MBPP solution length; cap to avoid runaway |
| Backend | HF | matches existing baselines; numerical equivalence with prior measurements |

Expected positions: 50 problems × 3 samples × ~150 avg tokens ≈
**~22,500 positions per model, ~45,000 total**.

Output paths:
```
results/pless_alpha_entropy/Qwen--Qwen2.5-Coder-7B-Instruct/
  temp_t1.0.jsonl                       (the standard sample output)
  temp_t1.0.jsonl.entropy.jsonl         (sidecar — the actual measurement)
results/pless_alpha_entropy/codellama--CodeLlama-7b-Instruct-hf/
  temp_t1.0.jsonl
  temp_t1.0.jsonl.entropy.jsonl
```

Run commands (one per model — can run on separate GPUs in parallel):

```bash
# Qwen
TASK_IDS=$(cat results/pless_alpha_smoke/smoke_task_ids.txt)
CUDA_VISIBLE_DEVICES=0 uv run python -m bench.runner \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --method temp --temperature 1.0 \
  --n-samples 3 --max-new-tokens 256 \
  --task-ids $TASK_IDS \
  --mbpp-config full --backend hf \
  --results-dir results/pless_alpha_entropy \
  --log-entropy

# CodeLlama
CUDA_VISIBLE_DEVICES=1 uv run python -m bench.runner \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --method temp --temperature 1.0 \
  --n-samples 3 --max-new-tokens 256 \
  --task-ids $TASK_IDS \
  --mbpp-config full --backend hf \
  --results-dir results/pless_alpha_entropy \
  --log-entropy
```

GPU time: ~30 min per model on 4090, ~15 min on H100. Total ~30 min if
running in parallel on 2 GPUs.

---

## 5. Analysis

New script: `bench/eval/analyze_entropy.py` (or similar).

### 5.1 Load + derive entropies

For each per-position record:
- `H_2 = −log(sigma_p2)` — primary x-axis
- `H_3 = (1/(1−3)) · log(sigma_p3) = −½ log(sigma_p3)`
- `H_5 = −¼ log(sigma_p5)`
- `H_∞ = −log(max_p)`
- Shannon ≈ `−Σ top32_probs · log(top32_probs)` (top-32 captures the bulk)

### 5.2 Primary figure: H₂ histograms

For each model:
- Histogram of `H_2` over all positions (linear and log y-axis)
- Hartigan's dip test for unimodality (reject → bimodal)
- Visual check for a gap or trough between two modes

Use `diptest` Python package (`pip install diptest`) for the dip test.

### 5.3 Secondary figures

1. **Per-token-class boxplot**: heuristically classify token strings into
   {whitespace, punctuation, keyword, operator, identifier-start,
   identifier-cont, numeric, string-content, other}, then boxplot
   `H_2` per class. Confirms low-entropy positions are syntactic and
   high-entropy are semantic.
2. **Rényi-α histogram overlay**: H₂, H₃, H₅ on the same axes.
   Confirms bimodality is robust across α (predicted: yes, since
   Rényi entropies are monotonically related to each other and the
   position-conditional shape doesn't depend on α).
3. **Position-in-sequence vs entropy scatter**: any periodic structure?
   (e.g., entropy spikes at start of each statement)

### 5.4 Robustness: passing-samples-only

Run eval on the generated samples (`uv run python -m bench.eval --dataset mbpp ...`).
For each sample, look up its pass/fail status. Filter the entropy
JSONL to positions from passing samples only. Re-plot H₂ histogram.

If the two histograms (all vs passing-only) agree on bimodality, the
finding is robust to the correctness confound. If they disagree,
report both and discuss.

### 5.5 Outputs

```
results/pless_alpha_entropy/analysis/
  hist_H2_qwen.png
  hist_H2_codellama.png
  hist_H2_qwen_passing_only.png
  hist_H2_codellama_passing_only.png
  boxplot_per_class_qwen.png
  boxplot_per_class_codellama.png
  hist_H2_H3_H5_overlay_qwen.png
  hist_H2_H3_H5_overlay_codellama.png
  position_vs_entropy_qwen.png
  position_vs_entropy_codellama.png
  dip_test_results.json
  summary.md       # human-readable findings + paper-ready narrative
```

---

## 6. Verification (after the run)

Before drawing any conclusions:

1. **Sidecar file written**: `temp_t1.0.jsonl.entropy.jsonl` exists,
   non-empty, parses as JSONL.
2. **Per-record fields**: every record has all 10 fields; no None / NaN.
3. **Sanity check on `sigma_p2`**: `1/V ≤ sigma_p2 ≤ 1` at every position.
   `V ≈ 152K for Qwen2.5-Coder, ≈ 32K for CodeLlama → lower bound
   `sigma_p2 ≥ 1/152K ≈ 6.6e-6` and `1/32K ≈ 3.1e-5`. Any value below
   these is a logging bug.
4. **Sanity check on `max_p`**: should equal `top32_probs[0]`.
5. **Sample length sanity**: positions per sample should be 1..256; if
   most samples hit 256 (no EOS), the prompt format is wrong.
6. **Token-class spot check**: print 20 random positions with
   `token_str` and `H_2`; visually verify whitespace/keywords have low
   H_2 and identifier-starts have high.

---

## 7. Cost & timing

| Phase | Cost |
|---|---|
| Instrumentation (`bench/generator.py` + `bench/runner.py`) | ~1 h coding + regression test |
| Run: 2 models × 50 problems × 3 samples on HF | ~30 min on 4090 each (parallel) |
| Analysis: histograms, dip test, boxplots | ~2 h Python (most time on token-class regex tuning) |
| **Total wall-clock** | **half day** |

GPU cost is negligible (~1 GPU-hour total). Storage: ~12 MB.

---

## 8. Decision rule (what we'd report)

| Hartigan dip-test outcome | Interpretation | Paper §4 framing |
|---|---|---|
| p < 0.01 on BOTH models | Strongly bimodal, both | "Code-token entropy is bimodal across models; this is the mechanism by which the α-sweep selectively loosens at semantic positions." |
| p < 0.05 on at least one | Bimodal on at least one | Same claim, with a caveat noting model variation. |
| p ≥ 0.05 on both | Unimodal | Refine to "asymmetric-by-shape adaptation" without claiming bimodality. Honest negative result. |

Additionally:
- Per-class boxplots should show **clear separation** between
  syntactic and semantic classes regardless of histogram bimodality.
  This is a weaker but still useful finding (supports the §4 claim
  even if the dip test is borderline).

---

## 9. Out of scope (this experiment)

- m-a-p OpenCodeInterpreter (only running 2 models; smaller model may
  have qualitatively different entropy distribution — interesting
  follow-up).
- HumanEval (same hypothesis should hold on any code benchmark; we
  focus on MBPP for consistency with the rest of the paper).
- Entropy under different temperatures (T=1.0 is the canonical baseline;
  bimodality at other T is interesting but a secondary follow-up).
- Direct entropy logging on top of the α-sweep generations (would
  show what α-sweep "sees" at each position; orthogonal to whether
  the underlying distribution itself is bimodal).
- Comparison to non-code text (would show code is *more* bimodal than
  prose; cool but not gating for the paper).

---

## 10. Critical files

To be modified:

- `/Users/karrtikiyer/projects/airesearch/pless-for-coding/bench/generator.py`
  — add `entropy_log` parameter to `generate_samples`, populate at each
  step when not None.
- `/Users/karrtikiyer/projects/airesearch/pless-for-coding/bench/runner.py`
  — add `--log-entropy` CLI flag, write sidecar JSONL after each task.

To be created:

- `/Users/karrtikiyer/projects/airesearch/pless-for-coding/bench/eval/analyze_entropy.py`
  — load sidecar, compute Hartigan dip test, generate figures.

To be read (no edits):

- `/Users/karrtikiyer/projects/airesearch/pless-for-coding/results/pless_alpha_smoke/smoke_task_ids.txt`
  — the 50 task IDs to reuse for consistency.

New dependency:

- `diptest` Python package (`uv add diptest`) for Hartigan's dip test.

---

## 11. Execution sequence

1. **Implement instrumentation** (`bench/generator.py`, `bench/runner.py`).
   ~1 h. Regression-test: `--method pless --temperature 1.0` without
   `--log-entropy` must produce byte-identical output to before.
2. **Smoke run** on Qwen, 5 problems, 1 sample. Verify sidecar JSONL
   schema. ~5 min.
3. **Full run** on both models in parallel (separate CUDA_VISIBLE_DEVICES).
   ~30 min total.
4. **Analysis** script. Histograms + dip test + boxplots. ~2 h.
5. **Write up** in `results/pless_alpha_entropy/analysis/summary.md`
   with figures and the decision-rule verdict.
6. **Commit + push** the new code + results + analysis.

Total: half day end-to-end if executed sequentially. ~3 h if the
analysis script can be parallelized with the GPU runs.
