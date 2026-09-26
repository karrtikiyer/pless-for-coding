# LCB v6 Stratified-60 — Progress Report (Qwen3-8B bf16)

Live comparison doc, updated as each (config, platform) pair completes 30/30.
Dataset: LiveCodeBench v6, 60 problems stratified 10/10/10 (easy/medium/hard)
across AtCoder + LeetCode. n=10 samples/task, cap=25,000 tokens, seed=42
(except qwen_recipe: T=0.6; all others T=1.0).

## AtCoder (30 tasks)

### pless_topmass

```
┌───────────────┬─────────────────────────────────────────┐
│    Metric     │                  Value                   │
├───────────────┼─────────────────────────────────────────┤
│ pass@1        │ 0.693                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@3        │ 0.754                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@5        │ 0.775                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@10       │ 0.800                                    │
├───────────────┼─────────────────────────────────────────┤
│ Termination   │ 91.3% (274/300 samples closed </think>)  │
├───────────────┼─────────────────────────────────────────┤
│ Mean tokens   │ 11,642                                   │
├───────────────┼─────────────────────────────────────────┤
│ Easy solved   │ 10/10                                    │
├───────────────┼─────────────────────────────────────────┤
│ Medium solved │ 9/10                                     │
├───────────────┼─────────────────────────────────────────┤
│ Hard solved   │ 5/10                                     │
└───────────────┴─────────────────────────────────────────┘
```

### pless_default (canonical, original authors)

```
┌───────────────┬─────────────────────────────────────────┐
│    Metric     │                  Value                   │
├───────────────┼─────────────────────────────────────────┤
│ pass@1        │ 0.637                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@3        │ 0.727                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@5        │ 0.747                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@10       │ 0.767                                    │
├───────────────┼─────────────────────────────────────────┤
│ Termination   │ 74.0% (222/300 samples closed </think>)  │
├───────────────┼─────────────────────────────────────────┤
│ Mean tokens   │ 12,814                                   │
├───────────────┼─────────────────────────────────────────┤
│ Easy solved   │ 10/10                                    │
├───────────────┼─────────────────────────────────────────┤
│ Medium solved │ 9/10                                     │
├───────────────┼─────────────────────────────────────────┤
│ Hard solved   │ 4/10                                     │
└───────────────┴─────────────────────────────────────────┘
```

**Note:** pless_default's termination (74.0%) is notably worse than pless_topmass's
(91.3%) despite similar pass@k — canonical p-less rambles substantially more on this
stratified sample, consistent with the peaked-distribution / rambling-loop finding
from the APPS-16 entropy probe earlier in this project.

### gk_0_1 (G_k Rényi, k=0.1)

```
┌───────────────┬─────────────────────────────────────────┐
│    Metric     │                  Value                   │
├───────────────┼─────────────────────────────────────────┤
│ pass@1        │ 0.707                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@3        │ 0.797                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@5        │ 0.816                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@10       │ 0.833                                    │
├───────────────┼─────────────────────────────────────────┤
│ Termination   │ 91.3% (274/300 samples closed </think>)  │
├───────────────┼─────────────────────────────────────────┤
│ Mean tokens   │ 11,419                                   │
├───────────────┼─────────────────────────────────────────┤
│ Easy solved   │ 10/10                                    │
├───────────────┼─────────────────────────────────────────┤
│ Medium solved │ 9/10                                     │
├───────────────┼─────────────────────────────────────────┤
│ Hard solved   │ 6/10                                     │
└───────────────┴─────────────────────────────────────────┘
```

**Best config so far** — pass@10=0.833, ahead of pless_topmass (0.800).

### pless_effk

```
┌───────────────┬─────────────────────────────────────────┐
│    Metric     │                  Value                   │
├───────────────┼─────────────────────────────────────────┤
│ pass@1        │ 0.707                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@3        │ 0.775                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@5        │ 0.792                                    │
├───────────────┼─────────────────────────────────────────┤
│ pass@10       │ 0.800                                    │
├───────────────┼─────────────────────────────────────────┤
│ Termination   │ 91.7% (275/300 samples closed </think>)  │
├───────────────┼─────────────────────────────────────────┤
│ Mean tokens   │ 11,842                                   │
├───────────────┼─────────────────────────────────────────┤
│ Easy solved   │ 10/10                                    │
├───────────────┼─────────────────────────────────────────┤
│ Medium solved │ 9/10                                     │
├───────────────┼─────────────────────────────────────────┤
│ Hard solved   │ 5/10                                     │
└───────────────┴─────────────────────────────────────────┘
```

## AtCoder leaderboard so far (pass@10)

| Rank | Config | pass@10 | Termination |
|---|---|---|---|
| 1 | gk_0_1 | 0.833 | 91.3% |
| 2 | pless_topmass | 0.800 | 91.3% |
| 2 | pless_effk | 0.800 | 91.7% |
| 4 | pless_default | 0.767 | 74.0% |

## LeetCode (30 tasks)

_Pending — platform-outer loop means LeetCode starts only after all 5 configs in
each wave finish their AtCoder pass. ETA: GPU1 ~2 days, GPU0 ~1.5 days from
2026-09-25 (see conversation for live pace)._

Directional signal only (not this stratified sample): a separate Laptop smoke test
on 3 hard LeetCode problems (FP8, Qwen recipe, n=3) found 2/2 completed tasks at
100% termination, max 15,747/25,000 tokens used — no evidence yet that the cap is
insufficient, but n=2 tasks is a small basis for confidence.

## Configs still pending (both platforms)

pless_gini, pless_top1half, top_nsigma (GPU1 wave, in progress: pless_gini);
gk_0_35, qwen_recipe, top_nsigma_1_5 (GPU0 wave, in progress: gk_0_35);
pless_default_t1_5, pless_default_t2_0 (queued, not yet launched — waiting for a
free GPU slot).
