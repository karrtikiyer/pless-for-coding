# p-less Sampling (Tan, Wu, Howard; arXiv:2509.23234): evaluation setup extraction

## Sources fetched (re-verify here)

| URL | What was read |
|---|---|
| https://arxiv.org/abs/2509.23234 | Abstract page via WebFetch: title, authors, submission history (v1 to v6) |
| https://arxiv.org/pdf/2509.23234 (resolves to **v6**, 44 pp.) | Downloaded with curl, converted with `pdftotext -layout`. **I read all 2831 lines, pages 1 to 44**, covering the main text, references, App. A, App. B.1 to B.6, App. C.1 to C.13 and the prompts. |
| https://arxiv.org/pdf/2509.23234v1 (34 pp.) | Checked with grep only. I read the k-order section (v1 App. B.6) and v1 Table 8 in full. |
| https://arxiv.org/pdf/2509.23234v4 (34 pp.), .../v5 (44 pp.) | Checked with grep only, for reasoning-model mentions and the k-order section. Neither was read in full. |

Unless stated otherwise, every section, table and figure number below refers to **v6**. Appendix numbering changed between versions. In v1/v4 the k-order results are **App. B.6 / Table 8** and the DeepSeek results are **Table 6**, cited in v1 as "Appendix B.4". In v5/v6 the same material is **App. C.7 / Table 9** and **App. C.5 / Table 7**.

---

## !! Correction to the project's CLAUDE.md (stated in the paper, verified)

The project CLAUDE.md says the origin paper reports **no experiments at k != 2** for the Rényi generalisation. **That is false.** Every version from v1 onward reports such experiments.

- v6 App. C.7: "Table 9 provides results comparing k-order thresholds of 0.025, 0.1, 0.4, 1.0, and 1.6 to our default p-less and p-lessnorm methods."
- The model is DeepSeek-R1-Distill-Qwen-7B. The datasets are CSQA, GPQA, GSM8K and QASC. Temperatures are τ ∈ {0.5, 0.7, 1.0, 1.5, 2.0}.
- The v1 Table 8 numbers are identical to the v6 Table 9 numbers. I diffed the two tables after normalising whitespace, and only the caption number changed.
- Authors' conclusion: "suggests that tuning p-less to a specific k-order threshold is unnecessary in most cases."
- My inference: claims that we give the "first empirical evaluation" of G_k, or that the origin paper is "theoretical only", are **not supportable** as written. A narrower claim may still hold: first evaluation of G_k on **code generation** and/or **long-CoT reasoning-model code tasks** (APPS). The paper never evaluates code (see §4).
- Possible tension (my inference): in Table 9, low k **collapses at high temperature**. For example, GSM8K at τ=2.0 scores 0.4 / 0.3 / 0.3 / 9.0 for k = 0.025 / 0.1 / 0.4 / 1.0, against 89.2 for default p-less. The project's finding is that low k (around 0.1 to 0.4) helps on APPS. Before claiming it, check that the project ran at τ ≤ 1.0. At τ ≤ 1.0, the paper's low-k results are close to p-less (Table 9 rows τ = 0.5 / 0.7 / 1.0).

---

## 6. Metadata (verified)

- **Title** (abstract page and PDF): "p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding"
- **Authors**: Runyan Tan, Shuang Wu, Phillip Howard. Affiliations in the PDF: Thoughtworks and the National University of Singapore.
- **arXiv**: 2509.23234 [cs.AI], cross-listed to cs.CL
- **Versions** (abstract page): v1 27 Sep 2025; v2 30 Sep 2025; v3 8 Oct 2025; v4 28 Oct 2025; v5 2 Feb 2026; v6 27 Feb 2026.
- **Venue**: the v6 PDF header reads "Published as a conference paper at ICLR 2026".
- **Match**: the v6 PDF abstract matches the abstract-page paraphrase (math, logical reasoning and creative writing tasks; efficiency; diversity). The PDF metadata title and authors match. The abstract page links the code to GitHub ryttry/p-less. In the PDF the code link renders as "§".
- **Code snippet**: v6 Fig. 15 shows `p_less_decode`, which computes `p = probs.square().sum(...)`, applies `mask = probs < p`, renormalises, and samples with multinomial. This is the same logic as the repo's `p-less/` submodule as described in the project CLAUDE.md.

---

## 1. Evaluation datasets / tasks (complete list)

The paper names five datasets in §4.1 and App. C.1: "five math, logical reasoning, and creative writing datasets" (§1).

| Dataset | Paper's category / description (App. C.1) | Metric | # examples stated? | Prompting (App. C.12.1) |
|---|---|---|---|---|
| GSM8K | math: "Grade School Math word problems, linguistically diverse" | accuracy | **NOT FOUND** (full read; no test-set size given) | 8-shot CoT |
| GPQA | science / graduate-level reasoning: "Graduate-Level Reasoning on expert-level science questions" | accuracy | **NOT FOUND**. The subset (diamond/main) is not stated. | zero-shot CoT |
| QASC | commonsense/fact-composition QA: "requiring valid compositions of facts using commonsense reasoning" | accuracy; n-gram diversity (Su et al. 2022) | **NOT FOUND** | 8-shot CoT |
| CommonsenseQA (CSQA) | commonsense QA: "requiring drawing from prior common sense knowledge" | accuracy | **NOT FOUND** | 8-shot CoT |
| Writing Prompts (WP) | creative writing: "Writing Prompts for open-ended creative story generation" | length-controlled win rate (AlpacaEval framework) plus human evaluation | "a subset of 100 prompts" (§4.3); human eval on 100 prompts (App. A) | instructional prompt |

- The paper groups GSM8K, GPQA, QASC and CSQA under **"math and logical reasoning"**. It treats WP as **"instruction following creative writing"** (§4.1).
- Headline metric for the four reasoning sets: "the area under the accuracy-temperature curve ... which we term AUC" (§4.2), normalised to [0, 1].
- The paper contains **no code-generation dataset** (no HumanEval, MBPP, APPS or LiveCodeBench) and no "commercial" dataset. A full-text grep for humaneval/mbpp/livecode/"code gen" returned nothing.

## 2. Models evaluated

| Model | Where used | How the paper describes it |
|---|---|---|
| Llama-2-7B (Chat) | main (Tables 1, 2, 5, 8, 10, 12; human eval) | instruction-following; "finetuned to follow instructions" (App. C.1) |
| Mistral-7B (Instruct) | main (Tables 1, 2, 3, 5, 6, 10, 11, 12, 14, 15) | same |
| Llama3-70b (Instruct) | main (Tables 1, 5, 10, 12, 13; Figs 4, 11 to 14; App. C.13) | same; "enables us to generalize our results to a significantly larger model" (App. C.1) |
| **DeepSeek-R1-Distill-Qwen-7B** | **ablation only**: §4.2 last paragraph; App. C.5 (Table 7); App. C.7 (Table 9) | **the only model the paper calls a reasoning model**: "We further ablated with the DeepSeek-R1-Distill-Qwen-7B (Guo et al., 2025) reasoning model" (§4.2). App. C.5 is titled "REASONING MODEL PERFORMANCE". |

The paper does not describe any model as "thinking", "long-CoT" or "RL-trained". The only place RL appears is the title of the cited DeepSeek-R1 reference.

## 3. (Model x dataset) combinations reported

Baselines are the main-text set from §4.1: top-p, min-p, ε-sampling, η-sampling and mirostat. Defaults (App. C.2): "p = 0.9 for Top-p and p = 0.1 for Min-p", ε = η = 0.0002, mirostat 4.0. Temperatures are 0.5, 0.7, 1.0, 1.5 and 2.0. The paper never compares against top-k, typical sampling or contrastive search.

| Model | Datasets | Methods | Temps | Samples / seeds | Result type (table) |
|---|---|---|---|---|---|
| Llama2-7b | CSQA, GPQA, GSM8K, QASC | 5 baselines + p-less + p-less-norm | 0.5 to 2.0 | "averaged across generations produced by three different random seeds" (App. C.3) | accuracy per τ (Table 5); AUC (Table 1); hyperparameter sweep (Table 8) |
| Mistral-7b | CSQA, GPQA, GSM8K, QASC | same | 0.5 to 2.0 | "one random seed due to computational constraints" (App. C.3) | Tables 1 and 5 |
| Llama3-70b | CSQA, GPQA, GSM8K, QASC | same | 0.5 to 2.0 | one seed | Tables 1 and 5 |
| Llama2-7b, Mistral-7b | WP | same | 1.0, 1.5, 2.0 | "one generation per method and temperature" on 100 prompts (§4.3) | LC win rate (Table 2) |
| Mistral-7b | all 5 datasets | greedy, beam (bs = 3, 5), p-less at τ = 1.0 | 1.0 | not stated | accuracy / diversity / win rate (Table 6) |
| Mistral-7b | GSM8K, GPQA | 5 baselines + p-less | not stated | "200 Mistral-7b generations" (§5.1) | sampling time per token (Table 3); t-tests on 100 GSM8K samples (Table 14) |
| Mistral-7b | all 5 datasets | min-p vs p-less | 0.5 to 2.5 | not stated | diversity (Table 11) |
| DeepSeek-R1-Distill-Qwen-7B | CSQA, GPQA, GSM8K, QASC | ε, η, min-p, mirostat, top-p, p-less, p-less-norm | 0.5 to 2.0 | **NOT FOUND** (no seed or sample count given) | mean accuracy (Table 7) |
| DeepSeek-R1-Distill-Qwen-7B | CSQA, GPQA, GSM8K, QASC | p-less at k ∈ {0.025, 0.1, 0.4, 1.0, 1.6} vs p-less, p-less-norm | 0.5 to 2.0 | NOT FOUND | mean accuracy (Table 9) |

**Numbers copied exactly.** Each cell shows p-less followed by the best baseline.
- Table 1 AUC, p-less: Llama2-7b GSM8K 0.267 (min-p 0.256); Mistral-7b GSM8K 0.562 (min-p 0.523); Llama3-70b GSM8K 0.932 (min-p 0.924).
- Table 7 (DeepSeek-R1-Distill-Qwen-7B), GSM8K at τ = 2.0: p-less 89.2, p-less-norm 88.6, min-p 86.1, top-p0.9 0.2, ε 6.1. CSQA at τ = 2.0: p-less-norm 67.2, p-less 66.7, min-p 61.2. GSM8K at τ = 1.0: p-less 88.7, min-p 88.7, top-p 88.2.
- Table 9, GSM8K, columns k = 0.025 / 0.1 / 0.4 / 1.0 / 1.6 / p-less / p-less-norm: τ = 1.0 gives 87.2 / 87.9 / 88.3 / 89.0 / 89.2 / 88.7 / 88.7; τ = 2.0 gives 0.4 / 0.3 / 0.3 / 9.0 / 86.2 / 89.2 / 88.6.
- Table 2 LC win rate, Llama-2-7b at τ = 2.0: p-less 65.64, min-p 48.94, top-p 0.00.

**Setup details not found anywhere**: max generation tokens for any model (DeepSeek included); whether DeepSeek's `<think>` traces were kept or truncated; inference framework; dataset splits and sizes for the four reasoning sets.

## 4. Reasoning-model decoding, long-CoT, looping/termination, code

- The reasoning-model claim (§4.2): p-less and p-less-norm "maintain strong performance across all temperature settings, including being significantly superior to the second best at temperature 2.0". App. C.5 says most other methods "exhibit significant degradation in performance at higher temperatures".
- **Long-CoT / thinking traces**: the paper does not discuss them. A full read and a grep for think/thinking found nothing.
- **Looping / repetition**: the paper does not discuss decoding loops. "Repetition" appears only as the name of the n-gram diversity metric (§5.2). What the paper does describe for baselines at high τ is the opposite failure, gibberish and degeneracy (App. C.12.2).
- **Termination / length**: App. C.9.1 says p-less "reduces competition to sampling tokens that will lead to generation termination". Table 12 reports mean generation length for the three non-reasoning models only. The paper reports no generation lengths for DeepSeek.
- **Code**: **the paper does not evaluate code generation.** No HumanEval, MBPP, APPS or LiveCodeBench appears in the full v6 text. The only code is the implementation snippet in Fig. 15.

## 5. Appendix B.5: Rényi-order generalisation

- What it proposes (v6 B.5): "a generalized k-order threshold G[P]k within the formalism of Rényi entropy", defined as G[P]_k = 1 / exp(H_k(p)) (Eq. 17). The appendix argues that G increases with k, reaches 1/n as k → 0 ("uniform sampling"), and reaches max p_i as k → ∞ ("greedy decoding"). §3.3 says p-less "corresponds to the exponential of the negative Rényi entropy of order 2".
- B.5 itself contains only the definition and limit behaviour. It has no experiments, theorems or tables.
- **Experiments at k != 2 are reported in a separate appendix**: v6 App. C.7 / Table 9, which is v1 App. B.6 / Table 8. The k values tested were 0.025, 0.1, 0.4, 1.0 and 1.6, all below 2. No k > 2 was tested. I read the whole of B.5 and C.7.
- My inference: the paper does not write G_k in the (Σp^k)^{1/(k-1)} form. That form is mathematically equivalent to exp(−H_k) by the standard Rényi definition in Eq. 7. The columns are labelled "p-less0.025" and so on, and the text calls them "k-order thresholds of 0.025 ...". I read this as k = 0.025 and so on.

---

## Limitations / threats to validity

- **Stated in the paper**: Mistral-7b and Llama3-70b use one seed "due to computational constraints". Beam search beats p-less on GSM8K (App. C.4.1). App. C.13 describes two p-less failure patterns on Llama3-70b: entropy spikes during arithmetic, and ambiguous questions.
- **My inference**:
  - The reasoning-model evidence is a single 7B distilled model, with no seed count, no token budget, and no disclosure of how CoT traces were handled.
  - Dataset sizes are never given, so the AUC differences can't be judged against sampling noise. Many gaps are below 0.01 AUC.
  - All baseline hyperparameters are fixed defaults in the main tables. The sweep in Table 8 covers Llama2-7b only.

## Reproducibility

The code repo is ryttry/p-less (abstract page), and the sampler appears in Fig. 15. Prompts are given in full in App. C.12.4. Seeds are partially specified (3 for Llama2-7b, 1 for the others), with no seed values given. Max tokens, splits and sizes are not given.

## Relevance to this project (my inference)

1. Paper B must not say that the origin paper did not test k != 2. The defensible novelty is G_k on **code** and on **long-CoT reasoning models at long token budgets**, which the paper never tested, including looping and termination, which the paper never discusses.
2. The origin paper's DeepSeek-R1-Distill-Qwen-7B results are on QA and math only. Our APPS DeepSeek results are therefore the first code evidence for that model family under p-less. That is a candidate claim, and lit-scout should confirm that no later work does this.
3. Cite v6 (ICLR 2026) with the v6 numbering: App. C.7 / Table 9.
