# Qwen3-8B Split Decoding — Early Analysis (226 common tasks)

**Date:** 2026-04-29  
**Dataset:** MBPP-full  
**Tasks compared:** 226 / 500 (common across all configs)

## Comparison Table

| Config | Method | Think | Split | Tokens | pass@1 | pass@3 | pass@5 | pass@10 | Solved | Trunc% |
|--------|--------|-------|-------|--------|--------|--------|--------|---------|--------|--------|
| **H8** | split: temp 1.5 → pless 1.5 | Yes | Yes | 8192 | 0.8412 | 0.8898 | 0.9003 | 0.9115 | 206/226 | 12.7% |
| **T15** | split: temp 1.5 → temp 1.5 (baseline) | Yes | Yes | 8192 | 0.8407 | 0.8847 | 0.8959 | 0.9115 | 206/226 | 13.0% |
| **H1** | split: temp 0.7 → pless 1.0 | Yes | Yes | 8192 | 0.8367 | 0.8802 | 0.8914 | 0.9027 | 204/226 | 13.8% |
| **H3** | split: temp 0.7 → pless 2.0 | Yes | Yes | 8192 | 0.8363 | 0.8796 | 0.8901 | 0.8982 | 203/226 | 13.5% |
| **H9** | split: temp 1.5 → pless 2.0 | Yes | Yes | 8192 | 0.8350 | 0.8796 | 0.8909 | 0.9027 | 204/226 | 13.3% |
| **H5** | split: temp 0.8 → pless 1.5 | Yes | Yes | 8192 | 0.8336 | 0.8782 | 0.8931 | 0.9071 | 205/226 | 13.5% |
| **H2** | split: temp 0.7 → pless 1.5 | Yes | Yes | 8192 | 0.8332 | 0.8800 | 0.8960 | 0.9159 | 207/226 | 13.5% |
| **R8** | split: pless 2.0 → temp 0.8 | Yes | Yes | 8192 | 0.8305 | 0.8774 | 0.8942 | 0.9159 | 207/226 | 14.2% |
| **H6** | split: temp 0.8 → pless 2.0 | Yes | Yes | 8192 | 0.8301 | 0.8737 | 0.8871 | 0.9027 | 204/226 | 13.6% |
| **R7** | split: pless 2.0 → temp 0.7 | Yes | Yes | 8192 | 0.8288 | 0.8758 | 0.8899 | 0.9027 | 204/226 | 14.0% |
| **H4** | split: temp 0.8 → pless 1.0 | Yes | Yes | 8192 | 0.8274 | 0.8735 | 0.8859 | 0.8982 | 203/226 | 13.9% |
| **H7** | split: temp 1.5 → pless 1.0 | Yes | Yes | 8192 | 0.8274 | 0.8839 | 0.8978 | 0.9115 | 206/226 | 13.5% |
| **R9** | split: pless 2.0 → temp 0.9 | Yes | Yes | 8192 | 0.8274 | 0.8769 | 0.8915 | 0.9071 | 205/226 | 14.6% |
| **C** | temp_think 0.6 | Yes | No | 4096 | 0.7252 | 0.7827 | 0.7980 | 0.8142 | 184/226 | 24.8% |
| **F** | split: temp→pless | Yes | Yes | 4096 | 0.7155 | 0.7782 | 0.7950 | 0.8142 | 184/226 | 25.9% |
| **G** | split: temp→pless_norm | Yes | Yes | 4096 | 0.7142 | 0.7829 | 0.8022 | 0.8230 | 186/226 | 26.1% |
| **E** | pless_norm_think 0.6 | Yes | No | 4096 | 0.6996 | 0.7613 | 0.7778 | 0.7920 | 179/226 | 27.8% |
| **D** | pless_think 0.6 | Yes | No | 4096 | 0.6987 | 0.7560 | 0.7754 | 0.8009 | 181/226 | 28.1% |
| **B** | pless 0.7 | No | No | 512 | 0.6562 | 0.6591 | 0.6593 | 0.6593 | 149/226 | — |
| **A** | temp 0.7 | No | No | 512 | 0.6491 | 0.6888 | 0.7025 | 0.7212 | 163/226 | — |

## Key Comparisons

**Does split (temp think → pless code) beat uniform temp?**
- C (temp_think 0.6): pass@1 = 0.7252
- F (split: temp→pless): pass@1 = 0.7155
- Delta: **-0.0097** (-1.0pp)
- Head-to-head: C wins 34, F wins 23, ties 169

**Does split (temp think → pless_norm code) beat uniform temp?**
- C (temp_think 0.6): pass@1 = 0.7252
- G (split: temp→pless_norm): pass@1 = 0.7142
- Delta: **-0.0111** (-1.1pp)
- Head-to-head: C wins 39, G wins 22, ties 165

**Does temp for thinking help vs uniform pless?**
- D (pless_think 0.6): pass@1 = 0.6987
- F (split: temp→pless): pass@1 = 0.7155
- Delta: **+0.0168** (+1.7pp)
- Head-to-head: D wins 25, F wins 34, ties 167

**Does temp for thinking help vs uniform pless_norm?**
- E (pless_norm_think 0.6): pass@1 = 0.6996
- G (split: temp→pless_norm): pass@1 = 0.7142
- Delta: **+0.0146** (+1.5pp)
- Head-to-head: E wins 24, G wins 36, ties 166

**How much does thinking help for temp?**
- A (temp 0.7): pass@1 = 0.6491
- C (temp_think 0.6): pass@1 = 0.7252
- Delta: **+0.0761** (+7.6pp)
- Head-to-head: A wins 38, C wins 52, ties 136

**How much does thinking help for pless?**
- B (pless 0.7): pass@1 = 0.6562
- D (pless_think 0.6): pass@1 = 0.6987
- Delta: **+0.0425** (+4.2pp)
- Head-to-head: B wins 41, D wins 42, ties 143

**Does raising temp-think (0.6→0.7) and pless temp-code (0.6→1.0) help?**
- F (split: temp→pless): pass@1 = 0.7155
- H1 (split: temp 0.7 → pless 1.0): pass@1 = 0.8367
- Delta: **+0.1212** (+12.1pp)
- Head-to-head: F wins 5, H1 wins 72, ties 149

**Effect of increasing pless temp-code 1.0 → 1.5 (more permissive pless)**
- H1 (split: temp 0.7 → pless 1.0): pass@1 = 0.8367
- H2 (split: temp 0.7 → pless 1.5): pass@1 = 0.8332
- Delta: **-0.0035** (-0.4pp)
- Head-to-head: H1 wins 22, H2 wins 22, ties 182

**Effect of increasing pless temp-code 1.5 → 2.0 (most permissive pless)**
- H2 (split: temp 0.7 → pless 1.5): pass@1 = 0.8332
- H3 (split: temp 0.7 → pless 2.0): pass@1 = 0.8363
- Delta: **+0.0031** (+0.3pp)
- Head-to-head: H2 wins 22, H3 wins 25, ties 179

**Best uniform-temp baseline vs new sweep top (temp 0.7 → pless 1.0)**
- C (temp_think 0.6): pass@1 = 0.7252
- H1 (split: temp 0.7 → pless 1.0): pass@1 = 0.8367
- Delta: **+0.1115** (+11.2pp)
- Head-to-head: C wins 7, H1 wins 71, ties 148

## Truncation Analysis (thinking configs only)

| Config | Method | Tokens | Truncated | Rate | All-trunc tasks |
|--------|--------|--------|-----------|------|-----------------|
| C | temp_think 0.6 | 4096 | 560/2260 | 24.8% | 33 |
| D | pless_think 0.6 | 4096 | 635/2260 | 28.1% | 39 |
| E | pless_norm_think 0.6 | 4096 | 628/2260 | 27.8% | 39 |
| F | split: temp→pless | 4096 | 585/2260 | 25.9% | 36 |
| G | split: temp→pless_norm | 4096 | 589/2260 | 26.1% | 31 |
| H1 | split: temp 0.7 → pless 1.0 | 8192 | 311/2260 | 13.8% | 14 |
| H2 | split: temp 0.7 → pless 1.5 | 8192 | 304/2260 | 13.5% | 9 |
| H3 | split: temp 0.7 → pless 2.0 | 8192 | 305/2260 | 13.5% | 10 |
| H4 | split: temp 0.8 → pless 1.0 | 8192 | 315/2260 | 13.9% | 13 |
| H5 | split: temp 0.8 → pless 1.5 | 8192 | 305/2260 | 13.5% | 11 |
| H6 | split: temp 0.8 → pless 2.0 | 8192 | 308/2260 | 13.6% | 13 |
| H7 | split: temp 1.5 → pless 1.0 | 8192 | 306/2260 | 13.5% | 12 |
| H8 | split: temp 1.5 → pless 1.5 | 8192 | 286/2260 | 12.7% | 12 |
| H9 | split: temp 1.5 → pless 2.0 | 8192 | 301/2260 | 13.3% | 12 |
| R7 | split: pless 2.0 → temp 0.7 | 8192 | 316/2260 | 14.0% | 11 |
| R8 | split: pless 2.0 → temp 0.8 | 8192 | 322/2260 | 14.2% | 10 |
| R9 | split: pless 2.0 → temp 0.9 | 8192 | 330/2260 | 14.6% | 12 |

## Diversity Metrics

| Config | Method | pass@1 | pass@10 | struct_div | codebleu_div | ngram_div | dataflow_div |
|--------|--------|--------|---------|-----------|-------------|-----------|-------------|
| **H8** | split: temp 1.5 → pless 1.5 | 0.8412 | 0.9115 | 0.2042 | 0.3845 | 0.5022 | 0.2101 |
| **T15** | split: temp 1.5 → temp 1.5 (baseline) | 0.8407 | 0.9115 | 0.2195 | 0.4089 | 0.5285 | 0.2263 |
| **H1** | split: temp 0.7 → pless 1.0 | 0.8367 | 0.9027 | 0.1919 | 0.3653 | 0.4711 | 0.2049 |
| **H3** | split: temp 0.7 → pless 2.0 | 0.8363 | 0.8982 | 0.1845 | 0.3526 | 0.4570 | 0.1963 |
| **H9** | split: temp 1.5 → pless 2.0 | 0.8350 | 0.9027 | 0.2004 | 0.3933 | 0.5075 | 0.2183 |
| **H5** | split: temp 0.8 → pless 1.5 | 0.8336 | 0.9071 | 0.1878 | 0.3570 | 0.4583 | 0.2059 |
| **H2** | split: temp 0.7 → pless 1.5 | 0.8332 | 0.9159 | 0.1927 | 0.3674 | 0.4766 | 0.1942 |
| **R8** | split: pless 2.0 → temp 0.8 | 0.8305 | 0.9159 | 0.1696 | 0.3075 | 0.3947 | 0.1810 |
| **H6** | split: temp 0.8 → pless 2.0 | 0.8301 | 0.9027 | 0.1890 | 0.3663 | 0.4682 | 0.2098 |
| **R7** | split: pless 2.0 → temp 0.7 | 0.8288 | 0.9027 | 0.1716 | 0.2985 | 0.3790 | 0.1799 |
| **H4** | split: temp 0.8 → pless 1.0 | 0.8274 | 0.8982 | 0.1897 | 0.3607 | 0.4663 | 0.2043 |
| **H7** | split: temp 1.5 → pless 1.0 | 0.8274 | 0.9115 | 0.2013 | 0.3788 | 0.4888 | 0.2109 |
| **R9** | split: pless 2.0 → temp 0.9 | 0.8274 | 0.9071 | 0.1692 | 0.3246 | 0.4175 | 0.1807 |
| **C** | temp_think 0.6 | 0.7252 | 0.8142 | 0.1667 | 0.3544 | 0.4579 | 0.1879 |
| **F** | split: temp→pless | 0.7155 | 0.8142 | 0.1686 | 0.3393 | 0.4350 | 0.1885 |
| **G** | split: temp→pless_norm | 0.7142 | 0.8230 | 0.1713 | 0.3431 | 0.4427 | 0.1833 |
| **E** | pless_norm_think 0.6 | 0.6996 | 0.7920 | 0.1244 | 0.2447 | 0.3133 | 0.1375 |
| **D** | pless_think 0.6 | 0.6987 | 0.8009 | 0.1305 | 0.2559 | 0.3266 | 0.1475 |
| **B** | pless 0.7 | 0.6562 | 0.6593 | 0.0067 | 0.0152 | 0.0182 | 0.0109 |
| **A** | temp 0.7 | 0.6491 | 0.7212 | 0.0573 | 0.1372 | 0.1768 | 0.0684 |

## Observations

1. **Best pass@1:** H8 (split: temp 1.5 → pless 1.5) at 0.8412
2. **Split decoding trades ~1.0pp pass@1 for convergence at pass@10:** F=0.814, G=0.823 vs C=0.814
3. **Split decoding is +1.7pp ahead of uniform pless thinking (D)** — temp for thinking matters
4. **Diversity comes from thinking, not code:** F retains 101% of C's structural diversity (0.1686 vs 0.1667), while D drops to 0.1305
5. **Pless without thinking is a dead end:** near-zero diversity (0.0067), no pass@1-to-pass@10 lift (0.656 → 0.659)
6. **Thinking is the dominant factor:** +7.6pp pass@1 (A→C)
7. **Sweep configs (H1–H3) leap ahead by ~+11.2pp pass@1 over C** — but **confounded with token budget**: H1–H3 used 8192 tokens (13.8% truncation), C used 4096 (24.8%). Re-running C/F at 8192 is required to isolate the temp/pless effect.
8. **pless temp-code (1.0/1.5/2.0) barely moves pass@1 at temp-think=0.7:** spread of 0.35pp across H1–H3. Diversity is also flat — the threshold p = Σ probs² is mostly insensitive in this range.

**Note:** Partial data for T15=226, R7=264, R8=264, R9=275 tasks. Final results may shift.