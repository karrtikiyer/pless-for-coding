# Non-coding reasoning datasets for p-less / G_k efficiency+performance study

Scout pass on 2026-11-13. Every fact below comes from a page fetched in this session (URL given). Anything not fetched is marked "not verified". Difficulty notes that go beyond reported numbers are marked [inference].

## Model-card / report scores used for column "reported scores"

- Ministral-3-3B-Reasoning-2512 (https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512). The Reasoning table reports exactly 4 benchmarks: AIME25 0.721, AIME24 0.775, GPQA Diamond 0.534, LiveCodeBench 0.548. License apache-2.0.
- Qwen3-8B. The HF card (https://huggingface.co/Qwen/Qwen3-8B) lists no reasoning benchmarks. It recommends temperature 0.6, top-p 0.95, top-k 20, and output of 32,768 tokens (38,912 for complex math). Scores come from Qwen3 Tech Report Table 17, thinking mode (https://arxiv.org/html/2505.09388): MATH-500 97.4, AIME'24 76.0, AIME'25 67.3, ZebraLogic 84.8, AutoLogi 89.1, GPQA-Diamond 62.0, MMLU-Redux 87.5, LiveBench 2024-11-25 67.1. Eval max output 32,768 tokens (AIME 38,912).
- DeepSeek-R1-Distill-Llama-8B (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B): AIME 2024 pass@1 50.4, cons@64 80.0, MATH-500 89.1, GPQA Diamond 49.0, LiveCodeBench 39.6. Max generation 32,768 tokens, T=0.6, top-p 0.95, 64 samples.

## Ranked shortlist

| Rank | Dataset | Domain | #problems | Answer type | License | Reported scores (Ministral-3B-R / Qwen3-8B / R1-Llama-8B) | Saturation / difficulty | Token-cost note | Source URLs |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Competition-math pool: AIME 2024 + AIME 2025 + AIME 2026 (+ HMMT Feb 2025, AMC 2023) | Math | 30 + 30 + 30 (+30 HMMT, +40 AMC23) = 90, or 160 with all five | Integer (AIME). LaTeX string (HMMT). Short string, mostly integer (AMC23) | AIME25/26 and HMMT: CC BY-NC-SA 4.0. AIME24 (H4) and AMC23: not stated on card | AIME24: 0.775 / 76.0 / 50.4. AIME25: 0.721 / 67.3 / n.r. AIME26/HMMT/AMC: n.r. | Mid-range for all three models, so discriminative. AIME 2026 is the newest edition here (post-cutoff for these 2025 models [inference]). Size is below 300 even pooled, so the bootstrap is weak | Qwen gives these 38,912 tokens, versus 32,768 for other tasks (Qwen report). Longest-CoT candidate | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 ; https://huggingface.co/datasets/MathArena/aime_2025 ; https://huggingface.co/datasets/MathArena/aime_2026 ; https://huggingface.co/datasets/MathArena/hmmt_feb_2025 ; https://huggingface.co/datasets/math-ai/amc23 |
| 2 | GPQA Diamond | Grad-level science (bio/phys/chem) | 198 (main 448, extended 546) | 4-way multiple choice (letter) | CC BY 4.0 (HF). MIT (GitHub repo). Gated: you agree not to reveal examples online | 0.534 / 62.0 / 49.0 | Mid-range on every card, best-discriminating non-math set. Under 300 problems | not verified | https://huggingface.co/datasets/Idavidrein/gpqa ; https://arxiv.org/html/2311.12022 ; https://github.com/idavidrein/gpqa |
| 3 | OlympiadBench, text-only English math (OE_TO_maths_en_COMP) | Olympiad math | 674 | Open-ended numeric/expression/tuple, some with unit and error tolerance | Apache-2.0 | n.r. on any card | Not verified for these models. Likely harder than MATH-500 [inference]. Used in efficiency papers (AES, LoCo) | not verified | https://huggingface.co/datasets/Hothan/OlympiadBench |
| 4 | MATH-500 | Math | 500 | LaTeX short answer (boxed) | not stated on card | n.r. / 97.4 / 89.1 | Near-saturated on accuracy for Qwen3-8B. Still useful for token/termination efficiency. A standard overthinking testbed | not verified | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| 5 | BBEH (BIG-Bench Extra Hard), full or mini | Diverse non-math logic/reasoning | 4,520 full; 460 mini | Short answer, scored by repo's evaluate.py | Code Apache-2.0, data CC-BY 4.0. Repo archived | n.r. | Hard: best reasoning model harmonic mean 44.8%, general-purpose 9.8% (arXiv abstract). Floor risk for the 3B model [inference] | not verified | https://github.com/google-deepmind/bbeh ; https://arxiv.org/abs/2502.19187 (HF card returned 401) |
| 6 | ZebraLogic (grid_mode) | Logic-grid puzzles | 1,000 grid (+3,259 mc_mode) | Filled JSON grid (grid) or MC | not stated on card | n.r. / 84.8 / n.r. | Accuracy falls as puzzle size grows ("curse of complexity"), so you can stratify by puzzle size. Grid parsing burden is moderate | not verified | https://huggingface.co/datasets/allenai/ZebraLogicBench ; https://arxiv.org/abs/2502.01100 |
| 7 | MMLU-Pro (subsample, e.g. 300-1,000) | 14-domain knowledge+reasoning | 12,032 test | 10-option MC (A-J) | MIT | n.r. | not verified for these models | not verified | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| 8 | MuSR | Narrative multi-step soft reasoning | 756 (250 murder / 256 object placement / 250 team allocation) | MC index (murder split has 2 choices, so 50% chance) | CC BY 4.0 | n.r. | Low chance floor hurts resolution [inference]. Long inputs (3.8k-7.3k chars) | not verified | https://huggingface.co/datasets/TAUR-Lab/MuSR |
| 9 | GSM8K (test) | Grade-school math | 1,319 | Integer after `####` | MIT | n.r. | Likely saturated [inference]. Main use is measuring overthinking on easy items (Chen et al. 2412.21187) | not verified | https://huggingface.co/datasets/openai/gsm8k |
| 10 | LiveBench reasoning | Zebra / spatial / 3rd task | 200 | Short string / integer, auto-graded | not stated | n.r. / 67.1 (LiveBench overall, not the reasoning split) / n.r. | Release dates 2024-06-24 to 2024-11-25, so older than these models | not verified | https://huggingface.co/datasets/livebench/reasoning |

n.r. = not reported on the fetched card or report.

## Why this ranking

Ranks 1 and 2 are the only sets where all three models have published scores in the mid range (roughly 50-78%). That means p-less loosening or tightening can move pass@k in either direction. They also produce the longest chain-of-thought (Qwen raises its cap for AIME), which is exactly where p-less looping and termination% matter. The weakness of both is size: 90-160 problems pooled, and 198 for GPQA. That is why OlympiadBench (674, Apache, rule-gradable) is ranked 3 to give the bootstrap power. MATH-500 is kept as an efficiency-only arm because Qwen3-8B scores 97.4 on it. BBEH and ZebraLogic are the strongest non-math options. BBEH may sit near the floor for the 3B model, and ZebraLogic is near the ceiling for Qwen3-8B (84.8). Ranks 7-10 are fallbacks.

Recent efficiency papers use these datasets:
- AES (arXiv 2611.11901): GSM8K, MATH-500, AMC, AIME24, AIME25, OlympiadBench, plus LSAT and GPQA out-of-domain, on R1-Distill-Qwen-1.5B/7B.
- LoCo (arXiv 2611.04814): "MATH-500, AMC 2023, Olympiad, AIME 2024, and AIME 2025".
- Chen et al. overthinking (arXiv 2412.21187): "such as GSM8K, MATH500, GPQA, and AIME".

## Could NOT verify / dropped

- Omni-MATH (4,428, Apache-2.0, https://huggingface.co/datasets/KbsdJames/Omni-MATH) exists. Dropped because the card recommends a model-based judge (Omni-Judge), so grading is not objective exact-match.
- BBH (lukaemon/bbh, 27 tasks, 6,511 rows): exists, but the license is not stated and saturation is not verified. Not shortlisted.
- FOLIO (yale-nlp/FOLIO, MIT): gated. Splits and answer format not visible. Not shortlisted.
- PlanBench, ProofWriter, ARC-AGI, AutoLogi as a dataset, LSAT: not fetched, so UNVERIFIED.
- Per-sample token costs for any dataset: not verified. Only generation caps were found.
- Licenses for MATH-500 (H4), AIME 2024 (H4), AMC23, ZebraLogic, and LiveBench reasoning: not stated on the fetched cards.
- Top-nσ (arXiv 2411.07641) says only "four reasoning-focused datasets". The names are not verified.
- Abstracts of ChainPrune (2608.21860) and RATIO (2609.39801) name no datasets.
- BBEH HF card returned 401. Facts are from GitHub and arXiv instead.
