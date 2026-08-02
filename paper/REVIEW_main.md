# Review copy — *When Does Hyperparameter-Free Decoding Break?*

**How to comment:** reference the tag, then your note. Examples:
- `¶14: too long, cut the second half`
- `¶18: soften "does not beat" → "is competitive with"`
- `[T2]: add an FSD-d row`
- `[C3]: this is the real headline — move it up`
- `global: intro is ~30% too long`

Paragraphs are `¶N` (continuous), tables `[T#]`, figures `[F#]`, contribution bullets `[C#]`.
This renders the prose in `paper/latex/main.tex`; math is shown readably. Drop comments in chat or
into `paper/REVIEW_NOTES.md`.

---

## Title
**When Does Hyperparameter-Free Decoding Break? A Rényi-α Analysis of Entropy-Threshold Sampling for Code and Reasoning**

## Abstract
**¶1** — Hyperparameter-free decoders promise to remove temperature and top-p tuning by deriving a truncation threshold directly from the token distribution. We ask *when* that promise holds for code generation. Across 13 model checkpoints on MBPP-500 and HumanEval-164, and two reasoning models on APPS competitive-programming problems, we find a sharp boundary. p-less sampling (threshold Σᵢpᵢ²) behaves as a *high-accuracy, low-diversity* decoder: competitive—occasionally best—at pass@1 on short base-model completions, dominated at pass@10 by temperature, and *silently degenerating* on long chain-of-thought, where peaked next-token distributions raise the threshold until only the mode survives, driving 15–42% of reasoning traces into non-terminating loops. We trace both failures to one mechanism—peakedness × hard thresholding—and study the Rényi-α family τ_α(p)=Σᵢpᵢ^α (α=2 recovers p-less) as a diagnostic knob, a generalization the origin paper proposes but never evaluates beyond α=2. Raising α flattens the survivor set, monotonically eliminates looping (truncation 42%→0.3%), and recovers most lost accuracy, *but does not surpass well-tuned temperature*. Our contribution is a characterization of *when* parameter-free decoding helps and breaks, the first empirical α-sweep of entropy-threshold decoding, and a reproducibility analysis showing how tokenizer/numerics bugs can fabricate decoding-method effects of up to 22pp.

## 1. Introduction
**¶2** — Code generation has converged on a small set of decoding policies (greedy/beam at deployment, temperature at evaluation) and a tradition of treating decoding as an afterthought. The benchmarks encourage this: pass@1 rewards a single confident guess, pass@10 rewards diversity, and few systems agree on which to optimize.

**¶3** — Standard truncation samplers introduce hyperparameters that interact non-obviously with model peakedness (top-p, top-k, η-sampling, min-p, top-nσ). Shi et al. (2024) document wide sensitivity across decoding methods on Llama-2-7B over MBPP/HumanEval but do not consider hyperparameter-free alternatives. Tan et al. (2025) propose p-less / p-less-norm: truncation samplers whose threshold comes from collision entropy, no tunable knobs — but evaluate only on math/reasoning/creative writing, not code, where mass often concentrates on a single correct token.

**¶4** — We give the first systematic evaluation of p-less on code, and find its behavior is governed by *distribution peakedness*, which predicts both where it helps and where it silently fails. Contributions:
- **[C1]** A when-does-it-help boundary (§4): competitive at pass@1, dominated at pass@10, catastrophic on long CoT.
- **[C2]** A mechanism (§5): peakedness × hard threshold → survivor set collapses to the mode → the pass@10 gap and the reasoning loop are two faces of one effect.
- **[C3]** The first empirical α-sweep of an entropy-threshold sampler (§6), with a self-contained non-equivalence to the origin paper's un-evaluated Rényi form.
- **[C4]** Code-specific reasoning-loop findings (§5): a paraphrastic-vs-verbatim breakdown on APPS + a verified non-transfer of a published hidden-state precursor to code.
- **[C5]** A reproducibility post-mortem (§8): three silent bugs shifted pass@1 up to 22pp and fabricated a result that reversed once fixed.

**¶5** — The headline is conditional, not promotional: p-less is a calibration-free default useful where distributions are flat and silently broken where peaked; the value is the removed hyperparameter, not raw pass@k uplift.

## 2. Background and Related Work
**¶6** *(p-less and p-less-norm)* — p-less threshold τ=Σᵢpᵢ², admit pᵢ ≥ τ; p-less-norm relaxes to (v·Σpᵢ²−1)/(v−1), admitting more tail tokens. Survivors renormalized and sampled; only an optional pre-truncation temperature, no top-k/top-p/η budget.

**¶7** *(Decoding on code)* — Shi et al. (2024) benchmark a wide panel on Llama-2-7B over MBPP/HumanEval; predates p-less; we use their Llama-2-7B MBPP numbers as an external comparison. Holtzman et al. (2020) established that likelihood-maximizing decoding collapses to the mode and produces repetitive degeneration — the mechanism our peaked-distribution failure instantiates.

**¶8** *(Reasoning loops)* — Duan et al. (2026) show loops are semantically redundant but lexically distinct and predict them from hidden states; Xie et al. (2025) note reasoning models "do not exhibit strictly verbatim repetitions" and catch them with a hidden-state probe; Pipis et al. (2025) show looping at low temperature. All of this is math/QA and detection-focused: none on code, none reports the paraphrastic fraction, none studies a truncation sampler as cause or preventive. Yang et al. (2025) show identify-but-fail-to-recover with inverse scaling; DEER and s1 control length reactively / by budget forcing.

## 3. Methodology
**¶9** *(Models & benchmarks)* — 13 checkpoints (Llama-2, Code Llama, Codestral, Qwen2.5-Coder, Qwen3-Coder; 1.3B–30B) on MBPP-500 and HumanEval-164. Reasoning regime: DeepSeek-R1-Distill-Llama-8B and Qwen3-8B on 252 APPS "interview" problems, thinking on. Instruct models use chat templates; base models bare prompts.

**¶10** *(Sampling & metrics)* — 10 samples/task; pass@k (Chen 2021 unbiased), cover@t, structural diversity (Zhang–Shasha), CodeBLEU diversity. For APPS also the non-termination (truncation) rate as loop proxy. Per-task SE ≈1.75pp (MBPP), ≈2.8pp (HumanEval); sub-2·SE differences are "directional."

**¶11** *(Rényi-α family)* — We study τ_α(p)=Σᵢpᵢ^α with α≥2; α=2 is exactly p-less.

## 4. Where p-less helps: flat distributions
**¶12** — On weaker base models p-less admits a small tractable survivor set and is competitive at pass@1. On Llama-2-7B (MBPP), merged with the Shi et al. (2024) survey, p-less-norm@0.6 is top-ranked (see [T1]), ahead of FSD-d and beam-8; plain temperature ranks 15/19.

**[T1]** — MBPP-500 pass@1, Llama-2-7B base, ranked vs the 19-method survey: p-less-norm(0.6) **22.3 (rank 1)**, p-less(0.6) 22.2 (2), p-less(1.0) 19.8 (4), p-less-norm(1.0) 19.1 (7), temperature(0.7) 13.2 (15).

**¶13** — But the pass@1 win comes with a diversity deficit — the recurring **pass@1-vs-pass@10 trade-off** ([T2]): p-less matches/beats greedy and temperature at pass@1 (Codestral 78.0 vs 72.6; Qwen2.5-Coder-7B-Instruct 87.5 vs greedy 84.1) but is dominated at pass@10 (CodeLlama 62.8 vs 38.4). p-less occupies the high-pass@1/low-diversity corner of the Pareto frontier ([F1]); it sits on but does not extend it.

**[T2]** — HumanEval-164 pass@1→pass@10, p-less(best) / temp-or-top-p(best) / greedy: CodeLlama-7B-Inst 36.1→38.4 / 36.2→**62.8** / 36.0; Codestral-22B **78.0**→84.8 / 72.6→**91.5** / 75.6; Qwen2.5-Coder-7B-Inst **87.5**→87.8 / 79.0→**94.5** / 84.1; Qwen3-Coder-30B 78.9→79.9 / 76.2→**86.6** / 75.6.

**[F1]** — Pareto: pass@1 vs structural diversity (MBPP base/chat). p-less in the high-pass@1/low-diversity corner; temperature reaches diversities p-less can't in the safe range.

## 5. Where p-less breaks: peaked distributions and long CoT
**¶14** *(temperature cliff)* — As pre-truncation temperature rises, p-less collapses rather than diversifying. On a six-point HumanEval sweep ([F2]), pass@1 is flat in T∈[0.7,1.5] but every temperature-sensitive model falls off a cliff by T=2.5. On the most peaked model (Qwen3-Coder-30B) p-less is T-immune because only the top token survives — it has degenerated to greedy.

**[F2]** — Pass@1 vs pre-truncation temperature (HumanEval, six models): sweet spot T∈[0.7,1.5], cliff by T=2.5.

**¶15** *(long CoT: silent loops)* — With default p-less (α=2), **41.8% of DeepSeek-R1-Distill and 14.5% of Qwen3-8B APPS traces never terminate** — they loop to the cap — and pass@1 is the worst of any decoder ([T3], DeepSeek 0.392). Same mechanism as the pass@10 gap: on a near-deterministic step the collision-entropy threshold approaches the top probability, the survivor set collapses to the mode, and the model re-derives the same continuation. This *inverts* the usual view of truncation as a repetition cure: a threshold tied to Σpᵢ² *amplifies* looping exactly where the model is confident.

**¶16** *(what the loops look like on code)* — Categorizing all non-terminating APPS traces: only ~41–50% are **verbatim** statement loops (Qwen 40.7%, DeepSeek 49.8%); a comparable share — **41.3% (Qwen) / 46.8% (DeepSeek)** — are **paraphrastic** (same idea in drifting words; periodicity self-match median 0.23–0.26; 73–74% below 0.5), so no token recurs and they're invisible to verbatim n-gram detection. Neither prior paper reports this. Consistently, a published hidden-state precursor (Duan et al. 2026) *does not transfer*: reimplemented on Qwen3-8B code traces it catches only ~17–20% of terminal loops before onset (vs 0.64–0.76 on synthetic verbatim loops), and its "semantic-precedes-textual" precursor is absent before onset. Because paraphrastic loops lack a clean onset, these rates are directional, not a strict numerical comparison; the robust finding is qualitative — no precursor gap to exploit.

## 6. The α knob: a diagnostic lens, not a new SOTA sampler
**¶17** *(raising α removes the loop)* — τ_α flattens the survivor set as α grows. On APPS ([F3], [T3]), α from 2→5 collapses truncation (41.8→0.3% DeepSeek; 14.5→0.6% Qwen) and recovers most lost pass@1 (0.392→0.483; 0.625→0.686), monotonically, no per-task tuning.

**[T3]** — APPS (252×10, thinking on), pass@1 / pass@10 / non-term%. **DeepSeek:** α=5 0.483/0.714/0.3; temp(1.0,p0.95) 0.480/**0.726**/0.0; α=4 0.473/0.710/1.4; adaptive 0.457/0.687/7.1; α=2 0.392/0.627/41.8; p-less-norm 0.392/0.663/41.7. **Qwen3-8B:** temp(1.0,p0.95) **0.705**/0.821/0.2; α=4 0.696/0.821/1.4; α=5 0.686/0.833/0.6; adaptive 0.682/**0.845**/2.7; α=2 0.625/0.825/14.5; p-less-norm 0.629/0.829/16.0.

**[F3]** — APPS: (a) raising α recovers pass@1 but only ties best temperature (dashed lines); (b) looping collapses as α rises from the α=2 default.

**¶18** *(honest punchline)* — The best α only *ties* well-tuned temperature: DeepSeek α=5 (0.483) ≈ temp (0.480); on Qwen3 temperature (0.705) *beats* best α (0.686). α is a diagnostic lens and a repair for a broken default, not a new SOTA sampler.

**¶19** *(relation to origin's Rényi form)* — Tan et al. (2025) App. B.5 propose a different generalization G_k=exp(−H_k)=(Σpᵢ^k)^{1/(k−1)} but run no experiments at k≠2. Our raw power-sum τ_α and their rooted G_k coincide only at order 2; for order >2 they move in **opposite directions** ([T4]): τ_α decreases (loosens) while G_k increases toward maxᵢpᵢ (tightens). On a 5-token example, sweeping order 2→8, τ_α admits progressively more (mode-only → full support, 5 distinct filters) while G_k stays on the mode at every order — no reparameterization aligns them; non-equivalent for order >2. We thus call ours *α-frequency-moment* thresholding and claim only the first *empirical* α>2 study, not the idea of a Rényi generalization.

**[T4]** — Two Rényi forms on p=[0.7,0.2,0.1], order 2/3/4/5: τ_α 0.5400/0.3520/0.2418/0.1684 (ours, decreasing); G_k 0.5400/0.5933/0.6230/0.6406 (origin, increasing → maxᵢpᵢ=0.70). Coincide at order 2.

## 7. Prevention vs. rescue
**¶20** — If a broken default (α=2) loops, prevent (run α=5 from the start) or detect-and-rescue (α=2, detect the loop, chop back to onset, continue at α=5)? Prevention wins on the high-loop model (DeepSeek 0.483 vs adaptive 0.457) and ties on the low-loop model (Qwen 0.686 vs 0.682; adaptive edges pass@10 0.845 vs 0.833) — the more a model loops, the more prevention beats rescue. Mechanistically, α=5's flatter survivor set suppresses re-entry after a chop where α=2 re-enters. But both are matched by good temperature ([T3]): the lever matters only if committed to the p-less family.

## 8. A reproducibility post-mortem
**¶21** — Three silent bugs each shifted pass@1 enough to change conclusions: (i) a transformers-v5 tokenizer routing bug (HF #45488) fed one backend whitespace-mangled prompts (`def f(a, b):`→`deff(a,b):`), depressing pass@1 up to 22pp; (ii) a blanket uniform-smoothing crash-guard silently perturbed the published p-less filter on every step; (iii) a float32 reduction could round the threshold above maxᵢpᵢ, prune every token, and NaN-crash. Before fixing (i)–(ii) an "adaptive ≫ prevention" result appeared; after fixing and unifying the backend, the ordering reversed to prevention≥adaptive (§6/§7). We document the fixes so future decoding studies on reasoning models avoid the same traps.

## 9. Discussion and Limitations
**¶22** *(where helps/breaks)* — Flat distributions (weak base models; instruct at T∈[0.6,1.0]): p-less reproduces greedy's pass@1 with small diversity at zero hyperparameter cost, tops the Shi et al. panel on Llama-2-7B. Peaked distributions (strong instruct under diversity demand; long CoT): threshold collapses to the mode → pass@10 dominated by temperature, traces loop.

**¶23** *(limitations)* — (i) p-less never *beats* well-tuned temperature; value is the removed knob, not accuracy. (ii) We do not claim α is uniquely principled: the diversity monotonicity also holds for top-p/top-k, so α is one member of a broader class; no theoretical optimality claim. (iii) Two 8B reasoning models, one benchmark (APPS). (iv) 10 samples/task; bootstrap CIs deferred. (v) Cross-survey comparisons cross independent pipelines (a 4pp temp@0.7 pipeline gap is documented).

**¶24** *(practical recommendation)* — For a model of unknown peakedness, start with p-less at T=1.0 and compare pass@1 to greedy on a small calibration set; if they match, keep p-less for its calibration-free property; if pass@1 is materially worse, or if the model produces long CoT, fall back to temperature in [0.7,1.0] or raise α to escape the peakedness trap.

## 10. Conclusion
**¶25** — p-less is a competitive but not dominant code sampler whose behavior is governed by distribution peakedness: a high-accuracy/low-diversity default that helps where distributions are flat and silently breaks—collapsing to greedy and looping—where peaked. The Rényi-α family diagnoses and repairs the failure but does not beat well-tuned temperature. The most transferable lesson may be methodological: parameter-free does not mean assumption-free, and small tokenizer/numerics bugs can fabricate decoding-method effects larger than the effects under study.

---

### Open items already flagged (you don't need to comment unless you disagree)
- **[F1]** regenerate at final DPI / legend readability.
- **Appendix** (not rendered above): migrate full 12/15-config APPS tables, T₁/T₂ decomposition, per-model MBPP/HE table, structural-diversity bars, detector config.
