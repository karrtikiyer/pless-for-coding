# Qwen3-8B Split Decoding — Final Analysis (500 common tasks)

**Date:** 2026-04-27  
**Dataset:** MBPP-full  
**Tasks compared:** 500 / 500 (common across all configs)

## Comparison Table

| Config | Method | Think | Split | pass@1 | pass@3 | pass@5 | pass@10 | Solved | Trunc% |
|--------|--------|-------|-------|--------|--------|--------|---------|--------|--------|
| **C** | temp_think 0.6 | Yes | No | 0.7380 | 0.7993 | 0.8168 | 0.8340 | 417/500 | 22.7% |
| **F** | split: temp→pless | Yes | Yes | 0.7336 | 0.7956 | 0.8123 | 0.8320 | 416/500 | 23.2% |
| **G** | split: temp→pless_norm | Yes | Yes | 0.7318 | 0.7976 | 0.8156 | 0.8340 | 417/500 | 23.3% |
| **E** | pless_norm_think 0.6 | Yes | No | 0.7186 | 0.7835 | 0.8025 | 0.8220 | 411/500 | 24.6% |
| **D** | pless_think 0.6 | Yes | No | 0.7178 | 0.7768 | 0.7949 | 0.8160 | 408/500 | 24.8% |
| **B** | pless 0.7 | No | No | 0.6694 | 0.6733 | 0.6739 | 0.6740 | 337/500 | — |
| **A** | temp 0.7 | No | No | 0.6616 | 0.7037 | 0.7189 | 0.7340 | 367/500 | — |

## Key Comparisons

**Does split (temp think → pless code) beat uniform temp?**
- C (temp_think 0.6): pass@1 = 0.7380
- F (split: temp→pless): pass@1 = 0.7336
- Delta: **-0.0044** (-0.4pp)
- Head-to-head: C wins 69, F wins 53, ties 378

**Does split (temp think → pless_norm code) beat uniform temp?**
- C (temp_think 0.6): pass@1 = 0.7380
- G (split: temp→pless_norm): pass@1 = 0.7318
- Delta: **-0.0062** (-0.6pp)
- Head-to-head: C wins 80, G wins 53, ties 367

**Does temp for thinking help vs uniform pless?**
- D (pless_think 0.6): pass@1 = 0.7178
- F (split: temp→pless): pass@1 = 0.7336
- Delta: **+0.0158** (+1.6pp)
- Head-to-head: D wins 54, F wins 80, ties 366

**Does temp for thinking help vs uniform pless_norm?**
- E (pless_norm_think 0.6): pass@1 = 0.7186
- G (split: temp→pless_norm): pass@1 = 0.7318
- Delta: **+0.0132** (+1.3pp)
- Head-to-head: E wins 58, G wins 75, ties 367

**How much does thinking help for temp?**
- A (temp 0.7): pass@1 = 0.6616
- C (temp_think 0.6): pass@1 = 0.7380
- Delta: **+0.0764** (+7.6pp)
- Head-to-head: A wins 80, C wins 118, ties 302

**How much does thinking help for pless?**
- B (pless 0.7): pass@1 = 0.6694
- D (pless_think 0.6): pass@1 = 0.7178
- Delta: **+0.0484** (+4.8pp)
- Head-to-head: B wins 89, D wins 93, ties 318

## Truncation Analysis (thinking configs only)

| Config | Method | Truncated | Rate | All-trunc tasks |
|--------|--------|-----------|------|-----------------|
| C | temp_think 0.6 | 1133/5000 | 22.7% | 63 |
| D | pless_think 0.6 | 1240/5000 | 24.8% | 73 |
| E | pless_norm_think 0.6 | 1232/5000 | 24.6% | 70 |
| F | split: temp→pless | 1161/5000 | 23.2% | 69 |
| G | split: temp→pless_norm | 1164/5000 | 23.3% | 62 |

## Diversity Metrics

| Config | Method | pass@1 | pass@10 | struct_div | codebleu_div | ngram_div | dataflow_div |
|--------|--------|--------|---------|-----------|-------------|-----------|-------------|
| **C** | temp_think 0.6 | 0.7380 | 0.8340 | 0.1667 | 0.3544 | 0.4579 | 0.1879 |
| **F** | split: temp→pless | 0.7336 | 0.8320 | 0.1686 | 0.3393 | 0.4350 | 0.1885 |
| **G** | split: temp→pless_norm | 0.7318 | 0.8340 | 0.1713 | 0.3431 | 0.4427 | 0.1833 |
| **E** | pless_norm_think 0.6 | 0.7186 | 0.8220 | 0.1244 | 0.2447 | 0.3133 | 0.1375 |
| **D** | pless_think 0.6 | 0.7178 | 0.8160 | 0.1305 | 0.2559 | 0.3266 | 0.1475 |
| **B** | pless 0.7 | 0.6694 | 0.6740 | 0.0067 | 0.0152 | 0.0182 | 0.0109 |
| **A** | temp 0.7 | 0.6616 | 0.7340 | 0.0573 | 0.1372 | 0.1768 | 0.0684 |

## Observations

1. **Best pass@1:** C (temp_think 0.6) at 0.7380
2. **Split decoding trades ~0.4pp pass@1 for convergence at pass@10:** F=0.832, G=0.834 vs C=0.834
3. **Split decoding is +1.6pp ahead of uniform pless thinking (D)** — temp for thinking matters
4. **Diversity comes from thinking, not code:** F retains 101% of C's structural diversity (0.1686 vs 0.1667), while D drops to 0.1305
5. **Pless without thinking is a dead end:** near-zero diversity (0.0067), no pass@1-to-pass@10 lift (0.669 → 0.674)
6. **Thinking is the dominant factor:** +7.6pp pass@1 (A→C)