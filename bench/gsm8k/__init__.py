"""GSM8K pass@k + diversity sweep for the α-knob mechanism test.

Question this module exists to answer: does the α-knob produce monotonic
pass@10 + diversity gains on math reasoning (GSM8K), like it does on code
(MBPP, HumanEval)? Together with the per-token entropy probe (which
already shows GSM8K is bimodal), this lets us test the bimodality→
α-effect mechanism story on a non-code domain at scale.

Built as a self-contained module that:
  * uses the verified Wei et al. 2022 8-shot CoT prompt (Table 20 of
    arXiv:2201.11903, verified verbatim from the PDF on 2026-05-22),
  * generates via bench.generator + bench.sampler_bridge (the same
    HF-backend pipeline used for MBPP/HumanEval — no vLLM here for
    numerical equivalence with our existing code-side data),
  * extracts "The answer is N." per Wei 2022's answer convention,
  * computes pass@k via the existing Chen et al. 2021 unbiased
    estimator from bench/eval/metrics.py:compute_pass_at_k,
  * computes diversity via pairwise BLEU-4 on the reasoning text
    (text before "The answer is"), conditional on correctness,
    deduplicated — mirroring the convention of
    bench/eval/metrics.py:add_self_codebleu so the GSM8K diversity
    numbers are directly comparable to our code-side CodeBLEU diversity.
"""
