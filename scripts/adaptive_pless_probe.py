"""Adaptive-pless local probe: detect think-phase looping via n-gram repetition,
then ESCALATE the sampler from pless (collision threshold) to pless_alpha(α=5) for
that sequence — to break the loop mid-generation without forcing </think>.

Tests the core question: does bumping α AFTER a loop has started actually escape it?

Run: PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/adaptive_pless_probe.py
Env: TASK_IDS ("930 1085"), N_SAMPLES(2), MAX_TOKENS(6144), ALPHA(5),
     NGRAM_N(8), NGRAM_K(4)
"""
import json
import os
import torch

from bench.generator import load_model_and_tokenizer, generate_samples
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.dataset import load_apps_test_map
from bench.eval.apps_executor import evaluate_apps_sample
from bench.sampler_bridge import make_guarded_pless_sampler, make_pless_alpha_sampler
from scripts.repeat_detector import RepeatDetector

POD = "results/pless_cot_efficiency_hf/Qwen--Qwen3-8B/ATCODER_interview"
TASK_IDS = [int(x) for x in os.environ.get("TASK_IDS", "930 1085").split()]
N = int(os.environ.get("N_SAMPLES", "2"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "6144"))
ALPHA = float(os.environ.get("ALPHA", "5"))
NGRAM_N = int(os.environ.get("NGRAM_N", "8"))
NGRAM_K = int(os.environ.get("NGRAM_K", "4"))


class AdaptiveSampler:
    """sampler_fn for generate_samples. Per-row: start pless, switch that row to
    pless_alpha(ALPHA) the step after its n-gram detector fires."""
    def __init__(self, n_rows, alpha, n, k, think_end_id):
        self.base = make_guarded_pless_sampler()
        self.escape = make_pless_alpha_sampler(alpha)
        self.det = [RepeatDetector(n=n, k=k) for _ in range(n_rows)]
        self.escaped = [False] * n_rows
        self.in_code = [False] * n_rows          # stop escalating once </think> emitted
        self.think_end_id = think_end_id
        self.N = n_rows

    def __call__(self, probs):                    # probs: (N, vocab)
        out = torch.empty((self.N, 1), dtype=torch.long, device=probs.device)
        for r in range(self.N):
            s = self.escape if self.escaped[r] else self.base
            out[r] = s(probs[r:r+1].clone())
        # record emitted tokens; flip escaped on detection (think phase only)
        for r in range(self.N):
            tok = int(out[r].item())
            if tok == self.think_end_id:
                self.in_code[r] = True
            if not self.escaped[r] and not self.in_code[r]:
                if self.det[r].update(tok):
                    self.escaped[r] = True
        return out


def main():
    pmap = load_apps_test_map(source="ATCODER", difficulty="interview")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B", dtype="bfloat16")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    for tid in TASK_IDS:
        problem = pmap[tid]
        prompt, _ = format_prompt_apps_instruct(problem, tokenizer, enable_thinking=True)
        sampler = AdaptiveSampler(N, ALPHA, NGRAM_N, NGRAM_K, think_end_id)
        print(f"\n=== task {tid} | adaptive pless(α2→α{int(ALPHA)} on {NGRAM_N}-gram×{NGRAM_K}) "
              f"| n={N} cap={MAX_TOKENS} ===", flush=True)
        texts, ids_list, prompt_len = generate_samples(
            model, tokenizer, prompt, sampler, n_samples=N,
            max_new_tokens=MAX_TOKENS, temperature=1.0, stop_strings=None,
            return_token_ids=True,
        )
        for r in range(N):
            det = sampler.det[r]
            txt = texts[r]
            closed = "</think>" in txt
            n_gen = len(ids_list[r]) - prompt_len
            # did the loop break post-escape? re-scan tokens AFTER fire for another loop
            broke = None
            if det.fired:
                post = [int(x) for x in ids_list[r][prompt_len + det.fire_pos:]]
                d2 = RepeatDetector(n=NGRAM_N, k=NGRAM_K)
                refired = any(d2.update(t) for t in post)
                broke = not refired
            # execute (code only present if </think> closed)
            status = "no_code"
            if closed:
                code = txt.split("</think>", 1)[1]
                res, _ = evaluate_apps_sample("```python\n" + code if "```" not in code else code, problem)
                status = res.status
            print(f"  sample {r}: gen={n_gen} tok | fired={det.fired}"
                  f"{' @'+str(det.fire_pos) if det.fired else ''} | escaped={sampler.escaped[r]}"
                  f" | loop_broke={broke} | </think>={closed} | exec={status}", flush=True)


if __name__ == "__main__":
    main()
