# Paper 2 — section skeleton

**Title (placeholder until greedy results):** *When a Hyperparameter-Free Decoder Fails: Diagnosing
and Repairing p-less Sampling Loops in Reasoning-Model Code Generation*

**Arc:** failure → mechanism → taxonomy (support) → lever → honest positioning. Structure only; no
prose yet. Claims = CL1–CL5 + CL4b from `paperB_crux.md`. Every number cites a grounded source:
`docs/research/paperA_master_numbers.md` (APPS metrics), `paperA_loop_positioning.md` (taxonomy),
`paperA_renyi_nonequivalence.md` (α vs G), regenerated `docs/decoder_comparison_cot_apps_*.md`.

---

## §1 Introduction  → sets up CL1, states all claims
- The appeal of hyperparameter-free decoding (p-less: threshold Σᵢpᵢ², no knobs); the origin paper
  evaluated math/reasoning/creative, not reasoning-model *code*.
- **Hook:** on reasoning-model CoT it fails *silently* — 15–42% of traces loop to the token cap.
- Contributions (map to claims): (1) the silent failure [CL1]; (2) a decoder-side mechanism [CL2];
  (3) a code loop taxonomy — ~half paraphrastic, missed by periodicity/precursor detectors [CL3];
  (4) the α-power-sum lever that removes loops, recovers accuracy, and is most token-efficient on the
  heavy looper [CL4/CL4b]; (5) honest positioning — beats recommended settings, trails only swept temp;
  power-sum ≠ Rényi [CL5].
- Framing sentence: the value is a *calibration-free default that must not silently break* — we show
  when it does and a lever that repairs it.

## §2 Background and Related Work  → positions CL1/CL3
- p-less / p-less-norm definitions (threshold, renorm, optional pre-temp).
- Reasoning-model looping: Pipis et al. 2025 (loop at low temp — model-side cause, we cite not claim);
  Duan et al. 2026 (Circular Reasoning, hidden-state precursor, math); Xie et al. 2025 (Word Salad
  Chopper, non-verbatim, math); Yang et al. 2025 (identify-but-not-recover); DEER / s1 (reactive
  length control). **Gap:** all math/QA + detection-only; none on code, none reports a paraphrastic
  fraction, none studies a truncation sampler as cause/preventive.
- Truncation samplers & degeneration: Holtzman 2020 (mode-collapse → repetition); min-p, top-nσ,
  η-sampling (all knobbed).

## §3 Setup  → grounds everything
- Models: DeepSeek-R1-Distill-Llama-8B, Qwen3-8B. Benchmark: APPS interview, 252 tasks × 10, thinking on.
- Decoders compared: p-less (α=2), p-less-norm, p-less-α (α=3,4,5), temperature/top-p/top-k variants
  incl. each model's **recommended** setting, adaptive chop→α5.
- Metrics: pass@1/@10 (Chen 2021), **non-termination %** (loop proxy), cb_div (CodeBLEU), **mean
  thinking tokens**. Note the mean-token cap-inflation caveat for high-truncation configs.
- The α family: τ_α(p)=Σᵢpᵢ^α, α=2 ≡ p-less. **State once, plainly: this is a power-sum / frequency-
  moment threshold, not a Rényi-entropy generalization** (full relation deferred to §7/appendix).
- **[T1]** master decoder table (both models): Config | pass@1 | pass@10 | cb_div | mean tok | trunc%.

## §4 The silent failure (CL1)  → [F1], [T1]
- Default p-less (α=2) & p-less-norm loop: **41.8% / 14.5%** non-termination; worst pass@1 on DeepSeek
  (0.392 vs best temp 0.480), near-worst on Qwen (0.625).
- **The key contrast (answers the "isn't this just low-temp looping?" reviewer):** p-less loops at
  nominal **T=1.0**, where plain temperature@T1.0 truncates ~0%. So it's a *sampler-structural* failure,
  not a temperature setting. → **[F1]**.

## §5 The mechanism (CL2)  → [F2]
- Peakedness × hard threshold: on a near-deterministic step Σᵢpᵢ² → max pᵢ, so the survivor set
  collapses to the mode; p-less decodes near-greedily exactly where the model is confident, and
  re-derives the same step → loop. Inverts the "truncation cures repetition" view.
- Corroboration: Qwen3's own model card warns greedy decoding → "endless repetitions."
- Scope handoff: *why* reasoning models produce peaked distributions (RL) is cited (Pipis et al.),
  not claimed. → **[F2]** (threshold vs max-p as a distribution peaks; #survivors → 1).

## §6 Loop taxonomy (CL3) — the supporting contribution  → [F3], [T2]
- Categorize all non-terminating traces: verbatim statement loop 40.7% / 49.8%; **paraphrastic drift
  41.3% / 46.8%**; short/degenerate; no-loop.
- Paraphrastic = same idea in drifting words (periodicity self-match median 0.23–0.26; 73–74% < 0.5).
- **Honest detector story (corrected):** n-gram detection is *not* useless (it drives our verbatim-half
  rescue) but *over-fires* — 68/54 completed-correct traces also trip it; the discriminating signal is
  *sustained periodicity*, which paraphrastic loops lack, so verbatim-period and hidden-state-precursor
  methods miss them. Published precursor (Duan 2026) doesn't transfer to code (~17–20% vs 0.64–0.76;
  directional). → **[F3]**, **[T2]**.

## §7 The α-power-sum lever (CL4 + CL4b)  → [F4], [F5], [T1]
- Raising α monotonically: removes loops (trunc 41.8→0.3% / 14.5→0.6%), recovers pass@1
  (0.392→0.483 / 0.625→0.696@α4), ~halves wasted tokens (17.3k→9.4k / 13.5k→11.1k), partly restores
  diversity (cb_div 0.489→0.553 / 0.453→0.474). No per-task tuning. → **[F4]** (existing α-sweep fig).
- **Token-efficiency vs other decoders [CL4b], loop-rate-dependent:** on DeepSeek high-α pless is the
  single most token-efficient decoder (α4/α5 9.2k/9.4k < every temp/top-p/top-k 9.6–10.1k); on Qwen
  it's a tie (~11.1–11.3k). Savings come from not spending budget on loops. → **[F5]** (mean-tokens vs
  pass@1 scatter, all decoders, high-α pless highlighted; DeepSeek panel makes the point).
- Prevention (α up front) ≥ reactive detect-and-chop rescue (one paragraph; don't over-expand).

## §8 Honest positioning (CL5)  → [T3], [T4]
- **vs recommended settings:** pless α4/α5 beat each model's official recommended config with zero
  tuning — Qwen α4 0.696 > rec 0.680; DeepSeek α5 0.483 > rec 0.475. Outperformed only by a higher-temp
  config (T1.0/p0.95: Qwen 0.705, DeepSeek 0.480≈tie) that departs from the model's own recommendation.
  → **[T3]** (pless-α vs recommended vs best-swept, both models).
- **Power-sum ≠ Rényi:** τ_α and the origin paper's G_k=exp(−H_k) coincide only at α=2; opposite
  monotonicity after; filter non-equivalence. → **[T4]** (from `paperA_renyi_nonequivalence.md`; can go
  to appendix). α is a lever, not a decoding win, not uniquely principled (top-p/top-k share the
  diversity monotonicity).

## §9 Discussion & Limitations
- Scope: two 8B reasoning models, one benchmark (APPS). α ties the best swept temperature (not a win).
- We don't explain *why* RL models are peaked; paraphrastic loops remain undetected (open problem).
- No theoretical optimality claim for α.

## §10 Conclusion
- Hyperparameter-free ≠ assumption-free: a collision-probability threshold silently collapses to greedy
  on peaked reasoning distributions and loops; the α-power-sum lever repairs it and is competitive
  out-of-the-box, but the deeper lesson is that ~half of reasoning-code loops are paraphrastic and
  invisible to current detectors.

---

## Figures — status
| id | content | status |
|---|---|---|
| [F1] | trunc% (and pass@1) of p-less α2 vs temp/top-p at **T=1.0** — "loops where temp doesn't" | **generate** (numbers in [T1]) |
| [F2] | mechanism schematic: as a distribution peaks, threshold Σpᵢ²→max pᵢ, #survivors→1 | **generate** (synthetic, illustrative) |
| [F3] | loop-taxonomy stacked bars per model (verbatim/paraphrastic/short/no-loop) | **generate** (numbers in [T2]) |
| [F4] | α-sweep: pass@1 + trunc% vs α, both models, temp reference lines | **exists** (`figures/fig_apps_alpha.png`) |
| [F5] | mean-tokens vs pass@1 scatter, all decoders, high-α pless highlighted (DeepSeek) | **generate** (numbers in [T1]) |

## Tables — status
| id | content | source |
|---|---|---|
| [T1] | master decoder table both models (pass@1/@10/cb_div/mean tok/trunc%) | regenerated `decoder_comparison_cot_apps_*.md` |
| [T2] | loop taxonomy fractions per model | `paperA_loop_positioning.md` / `loop_collapse_internal_state_findings.md` |
| [T3] | pless-α vs recommended vs best-swept temp, both models | `paperA_master_numbers.md` + model cards |
| [T4] | τ_α vs G_k non-equivalence (appendix) | `paperA_renyi_nonequivalence.md` |

## Length budget (~8–9 pp; verify venue)
Intro 1 · Background 0.75 · Setup 0.75 · §4 failure 1 · §5 mechanism 1 · §6 taxonomy 1.25 ·
§7 lever 1.5 · §8 positioning 0.75 · Discussion 0.5. Appendix: [T4], detector configs, per-config
full tables, prevention-vs-rescue detail.

## Reuse from Narrative-A LaTeX
`paper/latex/main.tex` §5/§6 prose + `refs.bib` + `figures/fig_apps_alpha.png` port directly. Paper 2
is essentially Narrative-A's §5–§6 promoted to a full paper, minus the MBPP/HE "where it helps" half.
