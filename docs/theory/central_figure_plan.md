# Central Figure Plan — Survival Mass vs Entropy at α=2 and α=5

**Status**: Plan only. Implementation pending approval.
**Last updated**: 2026-05-25.

## Why this figure

The α-knob paper makes a mechanism claim: "α controls which positions in
the bimodal-entropy distribution admit alternatives to the top token".
Currently we have two disconnected pieces of evidence:
1. **Entropy distribution** is bimodal with a small secondary mode around
   ~0.5-0.7 nats (verified — see `results/entropy_probe/.../entropy_kde.png`).
2. **Pass@k grows monotonically with α** (verified — MBPP, HE, GSM8K).

The proposed central figure **bridges these**: at each entropy value H,
plot the fraction of token probability mass that survives the p-less
filter at α=2 vs α=5. The expected shape (unverified hypothesis until we
compute it) is:

- Low H (formulaic positions): both α=2 and α=5 keep ~100% of mass
  (top token dominates regardless).
- Mid H (the secondary-mode region): α=2 drops to a single token; α=5
  preserves alternatives.
- High H (rare flat tail): both drop, but α=2 falls faster.

If the figure shows this, it directly visualizes *why* the α-knob
produces diversity at the bimodal-distribution's high-entropy minority
positions while preserving the low-entropy syntax/structure positions —
the mechanism story compressed into one plot.

## Scope (per user choice 2026-05-25)

- **Coverage**: Code-only, both models (Qwen2.5-Coder + CodeLlama).
  GSM8K out-of-scope for the central figure (would require an entropy
  probe re-run with top-K logging — separate task if pursued later).
- **α values**: Just α=2 and α=5 (endpoints). Maximum contrast, two
  curves per model.
- **Validation**: Standard rigor — 4 cross-checks (below).

## Data sources (verified by direct inspection 2026-05-25)

| File | Records | Has top-K probs? | Has σpᵅ thresholds? | Dataset | Sampler |
|---|---|---|---|---|---|
| `results/pless_alpha_entropy/Qwen--Qwen2.5-Coder-7B-Instruct/pless_t1.0.jsonl.entropy.jsonl` | 295,444 | ✓ top-32 | ✓ σ_p², σ_p³, σ_p⁵ | MBPP | pless@T=1.0 |
| `results/pless_alpha_entropy/codellama--CodeLlama-7b-Instruct-hf/pless_t1.0.jsonl.entropy.jsonl` | 284,739 | ✓ top-32 | ✓ σ_p², σ_p³, σ_p⁵ | MBPP | pless@T=1.0 |

**Per-record fields** (verified from one record): `task_id, sample_id,
position, token_id, token_str, sigma_p2, sigma_p3, sigma_p5, max_p,
top32_probs, top32_indices`.

Note on the sampler: the recordings come from generations with the
α=2 (pless@T=1.0) sampler. The stored `top32_probs` is the *raw
softmax distribution* at the visited position — independent of which
sampler was used to sample. So the survival-vs-H curves we compute
characterize the model's underlying distribution at each position,
not a sampler artifact. The α=2 sampling only determined WHICH
positions ended up in our recording — those visited positions are the
ones the α-knob actually encounters at inference.

## Computation per record

Each record gives us a (H, p_vec, σ²p, σ⁵p) tuple to plot.

1. **Normalize top-32 to a valid pmf**: `p_norm = top32_probs / sum(top32_probs)`.
   Record `truncation_mass = 1 - sum(top32_probs)` so we can quantify
   the tail we're missing.
2. **Compute entropy H from top-32**: `H = -Σ p_norm * log(p_norm + ε)`
   in nats. Note this is approximate — true H includes the
   non-top-32 tail.
3. **Compute thresholds** (already stored): `τ_α=2 = sigma_p2`,
   `τ_α=5 = sigma_p5`.
4. **Survival mass at each α**: `survived_α = sum(p for p in p_norm
   if p ≥ τ_α)`.

Edge cases:
- If all p < τ (would zero out everything): per the production
  `make_pless_alpha_sampler` fallback, the argmax token survives.
  Set `survived = max(p_norm)` in that case for fidelity.
- If `sum(top32_probs) < 0.95`, flag the position as "high-tail"
  for the validation step.

## Aggregation

Bin H values into ~50 bins (0.0 to ~max H observed, ~0.05 nat
width). Per bin:
- `mean_survival_α=2`, `mean_survival_α=5`
- `n_positions` (bin sample count — needed for downstream noise
  assessment)
- `mean_truncation_mass` (for the top-32 leakage validation)

Plot:
- X-axis: entropy H (nats), 0 to ~ceil(max H, e.g., 4 nats)
- Y-axis: mean surviving mass (0 to 1)
- Two curves per model (α=2 and α=5), each model on its own subplot
  → 2 subplots (Qwen2.5-Coder, CodeLlama) × 2 curves each
- Light-shaded vertical band overlay: bimodal KDE of H from same data
  (so the reader sees WHERE in the entropy distribution the curves
  diverge)

## Standard-rigor validation (4 checks)

Per user-approved validation depth:

### Check 1: Recomputed σ_p² matches stored sigma_p2
For a random 500-record subsample, recompute `σ²p = sum(p² for p in
top32_probs)` (note: from the *unnormalized* top-32, since the stored
σ_p² was computed from the full softmax which gives the same answer
as long as top-32 captures most mass). Report max absolute deviation.
**Acceptance**: |Δ| < 1e-4 for ≥99% of records. Anything else means
data drift or top-32 truncation is bigger than expected — must
investigate.

### Check 2: Top-32 truncation impact
Per H bin, report `mean_truncation_mass = 1 - mean(sum(top32_probs))`.
**Acceptance**: For bins with H ≤ 1.0 nat, truncation_mass ≤ 0.01
(secondary-mode region is where most mass sits in top 5 tokens,
top-32 should be ≥99% covered). For high-H bins (>2 nat), flag
truncation explicitly in the figure annotation.

### Check 3: H recomputation cross-check
Pick 100 random records. Recompute H two ways:
- (a) From normalized top-32 (what we use)
- (b) Using independent torch.nn.functional.cross_entropy approach
  if we have access to the original full-vocab probs (we don't, so
  this becomes a self-consistency check on the math).
Since we only have top-32, the cross-check is internal: compute H
in nats and bits both ways, ensure they convert consistently.
**Acceptance**: All recomputed H values are non-negative finite floats.

### Check 4: Per-bin sample size adequacy
Per H bin, annotate `n_positions`. Bins with `n < 50` get
greyed-out / dotted curve to signal noise. Document the H-range
where the curve is statistically reliable.

## Output deliverables

- **`results/entropy_probe/_central_figure/survival_vs_entropy.png`** —
  2 subplots (one per model), 2 curves each (α=2 and α=5), KDE shading.
- **`results/entropy_probe/_central_figure/survival_vs_entropy_data.json`** —
  per-bin numerical data (so the figure is reproducible / re-styleable
  later).
- **`results/entropy_probe/_central_figure/validation_report.md`** —
  the 4 checks above with pass/fail and the actual numbers.
- **`bench/eval/entropy_survival_curves.py`** — analysis code
  (persisted module; CLI).
- **`tests/test_entropy_survival_curves.py`** — unit tests for the
  per-record computation (uses synthetic mini-distributions).

## Implementation plan

### Step 1: Tests first (TDD per project rule)
Write unit tests for the core math:
- `compute_survival(p_vec, threshold)` returns expected sum for
  hand-constructed examples.
- `compute_entropy(p_vec)` matches hand computation on simple cases.
- Edge cases: uniform distribution, single-spike distribution,
  all-below-threshold (argmax fallback).

### Step 2: Implement the analysis module
- `entropy_survival_curves.py`:
  - `load_records(jsonl_path)` — stream the .entropy.jsonl
  - `process_record(rec)` — returns (H, survived_α2, survived_α5, truncation_mass)
  - `aggregate_to_bins(records, bin_width=0.05)` — bin by H, compute means
  - `validate(records)` — runs the 4 checks
  - `plot(bins_by_model, output_path)` — render the figure
  - CLI: `uv run python -m bench.eval.entropy_survival_curves --models Qwen--Qwen2.5-Coder-7B-Instruct codellama--CodeLlama-7b-Instruct-hf --output-dir results/entropy_probe/_central_figure`

### Step 3: Run on both models in background
~5-15 min CPU on a Mac. Reads ~580k records, computes, plots.

### Step 4: Inspect outputs against the 4 validation checks
Each check has an explicit acceptance criterion. If any fails, do
NOT declare the figure valid — debug first.

### Step 5: Write a short interpretation md
- What the figure shows quantitatively (the specific numbers).
- What it grounds (mechanism story).
- Honest caveats:
  - Code-only — GSM8K side not in this figure.
  - α=2 sampler used during recording (not greedy), so we condition
    on "positions α=2 visits"; positions α=5 might visit but α=2
    doesn't are NOT in this data. (See "Out of scope" below.)
  - Top-32 truncation at high H.
  - Single seed per generation.

## Out of scope (explicit deferrals)

- **GSM8K / CoT central figure**: would need a new entropy probe with
  top-K logging on GSM8K. Estimated 1-2 GPU-hr; queues behind the
  running APPS eval. Track as a follow-up.
- **Sampling-conditioned analysis**: the recorded positions are those
  the α=2 sampler visited. To fully characterize the survival curves
  the α=5 sampler would encounter at runtime, we'd need a separate
  recording with α=5 sampling. The current analysis assumes the
  position distribution is approximately the same across α values
  (plausible since the model's softmax shape per problem prefix
  doesn't depend on the sampler, only the prefix history). This is
  an unverified assumption; we'll note it explicitly.
- **CodeLlama isn't strictly identical**: it's an older model. The
  cross-model comparison should focus on "do both show the same
  qualitative pattern?" not "are the curves quantitatively identical?"
- **Cross-α-arm survival curves**: we could compute α=2.5 and α=3.0
  curves too (σ_p³ is stored), but per user choice we limit to
  endpoints for clarity.

## Verification (end-to-end)

After implementation completes:
1. `uv run pytest tests/test_entropy_survival_curves.py -v` — all
   unit tests pass.
2. `cat results/entropy_probe/_central_figure/validation_report.md` —
   all 4 checks pass per their acceptance criteria.
3. Visually inspect `survival_vs_entropy.png` — should show the
   predicted shape (low-H plateau at 1.0, mid-H divergence, high-H
   joint decline).
4. Spot-check 5 random records: hand-verify (H, survived_α=2,
   survived_α=5) by reading `survival_vs_entropy_data.json`.

## Open questions for the user before launch

None at this stage — scope decisions taken. Ready to implement on
approval.

## Concurrency note

This analysis runs locally on Mac CPU. Won't conflict with:
- The running APPS eval (CPU-bound on a different process tree)
- The running NAUADC judge (API-bound, mostly idle CPU)

We'll launch this analysis in background as well, with periodic log
checks.
