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

### gk_0_35 (G_k Rényi, k=0.35)

```
┌───────────────┬─────────────────────────────────────────┐
│    Metric     │                  Value                   │
├───────────────┼─────────────────────────────────────────┤
│ pass@1        │ 0.700                                     │
├───────────────┼─────────────────────────────────────────┤
│ pass@3        │ 0.774                                     │
├───────────────┼─────────────────────────────────────────┤
│ pass@5        │ 0.794                                     │
├───────────────┼─────────────────────────────────────────┤
│ pass@10       │ 0.800                                     │
├───────────────┼─────────────────────────────────────────┤
│ Termination   │ 92.0% (276/300 samples closed </think>)  │
├───────────────┼─────────────────────────────────────────┤
│ Mean tokens   │ 11,679                                    │
├───────────────┼─────────────────────────────────────────┤
│ Easy solved   │ 10/10                                     │
├───────────────┼─────────────────────────────────────────┤
│ Medium solved │ 9/10                                      │
├───────────────┼─────────────────────────────────────────┤
│ Hard solved   │ 5/10                                      │
└───────────────┴─────────────────────────────────────────┘
```

### pless_gini

```
┌───────────────┬─────────────────────────────────────────┐
│    Metric     │                  Value                   │
├───────────────┼─────────────────────────────────────────┤
│ pass@1        │ 0.703                                     │
├───────────────┼─────────────────────────────────────────┤
│ pass@3        │ 0.760                                     │
├───────────────┼─────────────────────────────────────────┤
│ pass@5        │ 0.780                                     │
├───────────────┼─────────────────────────────────────────┤
│ pass@10       │ 0.800                                     │
├───────────────┼─────────────────────────────────────────┤
│ Termination   │ 90.3% (271/300 samples closed </think>)  │
├───────────────┼─────────────────────────────────────────┤
│ Mean tokens   │ 11,672                                    │
├───────────────┼─────────────────────────────────────────┤
│ Easy solved   │ 10/10                                     │
├───────────────┼─────────────────────────────────────────┤
│ Medium solved │ 9/10                                      │
├───────────────┼─────────────────────────────────────────┤
│ Hard solved   │ 5/10                                      │
└───────────────┴─────────────────────────────────────────┘
```

## Diversity (structural = AST tree-edit-distance, higher=more diverse;
## self-CodeBLEU, lower=more diverse) — 4 configs with full diversity computed

**Easy (n=10 tasks/config, all with >=2 correct):**

| Config | struct_div | self_codebleu |
|---|---|---|
| pless_topmass | 0.2335 | 0.4980 |
| pless_default | 0.2312 | 0.4279 |
| gk_0_1 | **0.2666** | 0.4448 |
| pless_effk | 0.2612 | **0.4211** |

**Medium (n=9-10 tasks/config):**

| Config | struct_div | self_codebleu |
|---|---|---|
| pless_topmass | 0.3073 | 0.4307 |
| pless_default | 0.2230 | **0.3443** |
| gk_0_1 | **0.3356** | 0.4636 |
| pless_effk | 0.3082 | 0.4242 |

**Hard (n=3-5 tasks/config, NOT matched across configs — treat as suggestive
only, each average is over a different task subset):**

| Config | struct_div | self_codebleu | n tasks |
|---|---|---|---|
| pless_topmass | 0.4787 | 0.5240 | 4 |
| pless_default | 0.4321 | 0.4301 | 3 |
| gk_0_1 | 0.3867 | 0.4709 | 5 |
| pless_effk | 0.3914 | 0.4738 | 5 |

**Note:** structural diversity and self-CodeBLEU disagree on which config is
"most diverse" on medium (gk_0_1 highest structural but lowest CodeBLEU
diversity; pless_default the reverse) — they measure genuinely different
things (algorithm/AST shape vs surface-token similarity), not a contradiction
in the data.

## AtCoder leaderboard so far (pass@10)

| Rank | Config | pass@10 | Termination |
|---|---|---|---|
| 1 | gk_0_1 | 0.833 | 91.3% |
| 2 | pless_topmass | 0.800 | 91.3% |
| 2 | pless_effk | 0.800 | 91.7% |
| 2 | gk_0_35 | 0.800 | 92.0% |
| 2 | pless_gini | 0.800 | 90.3% |
| 6 | pless_default | 0.767 | 74.0% |

## LeetCode (30 tasks)

_Pending — platform-outer loop means LeetCode starts only after all 5 configs in
each wave finish their AtCoder pass. ETA: GPU1 ~2 days, GPU0 ~1.5 days from
2026-09-25 (see conversation for live pace)._

Directional signal only (not this stratified sample): a separate Laptop smoke test
on 3 hard LeetCode problems (FP8, Qwen recipe, n=3) found 2/2 completed tasks at
100% termination, max 15,747/25,000 tokens used — no evidence yet that the cap is
insufficient, but n=2 tasks is a small basis for confidence.

## Configs still pending (both platforms)

top_nsigma (GPU1 wave, in progress: pless_top1half);
top_nsigma_1_5 (GPU0 wave, in progress: qwen_recipe);
pless_default_t1_5, pless_default_t2_0 (queued, not yet launched — waiting for a
free GPU slot).
