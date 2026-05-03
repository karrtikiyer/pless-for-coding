# Qwen3-8B Split Decoding — Final Analysis (500 common tasks)

**Date:** 2026-05-03  
**Dataset:** MBPP-full  
**Tasks compared:** 500 / 500 (common across all configs)

## Comparison Table

| Config | Method | Think | Split | Tokens | pass@1 | pass@3 | pass@5 | pass@10 | Solved | Trunc% |
|--------|--------|-------|-------|--------|--------|--------|--------|---------|--------|--------|
| **T15** | split: temp 1.5 → temp 1.5 (baseline) | Yes | Yes | 8192 | 0.8502 | 0.8969 | 0.9083 | 0.9200 | 460/500 | 11.3% |
| **H9** | split: temp 1.5 → pless 2.0 | Yes | Yes | 8192 | 0.8456 | 0.8931 | 0.9038 | 0.9140 | 457/500 | 11.5% |
| **H8** | split: temp 1.5 → pless 1.5 | Yes | Yes | 8192 | 0.8454 | 0.8946 | 0.9070 | 0.9180 | 459/500 | 11.3% |
| **H5** | split: temp 0.8 → pless 1.5 | Yes | Yes | 8192 | 0.8452 | 0.8929 | 0.9071 | 0.9200 | 460/500 | 11.5% |
| **H3** | split: temp 0.7 → pless 2.0 | Yes | Yes | 8192 | 0.8438 | 0.8924 | 0.9054 | 0.9160 | 458/500 | 11.7% |
| **H1** | split: temp 0.7 → pless 1.0 | Yes | Yes | 8192 | 0.8436 | 0.8905 | 0.9029 | 0.9140 | 457/500 | 11.6% |
| **H6** | split: temp 0.8 → pless 2.0 | Yes | Yes | 8192 | 0.8428 | 0.8899 | 0.9035 | 0.9180 | 459/500 | 11.5% |
| **H2** | split: temp 0.7 → pless 1.5 | Yes | Yes | 8192 | 0.8426 | 0.8910 | 0.9036 | 0.9160 | 458/500 | 11.7% |
| **H7** | split: temp 1.5 → pless 1.0 | Yes | Yes | 8192 | 0.8426 | 0.8943 | 0.9069 | 0.9180 | 459/500 | 11.7% |
| **H4** | split: temp 0.8 → pless 1.0 | Yes | Yes | 8192 | 0.8416 | 0.8915 | 0.9040 | 0.9160 | 458/500 | 11.7% |
| **H10** | split: temp 1.5 → pless 3.0 | Yes | Yes | 8192 | 0.8378 | 0.8872 | 0.8999 | 0.9120 | 456/500 | 12.2% |
| **R8** | split: pless 2.0 → temp 0.8 | Yes | Yes | 8192 | 0.8364 | 0.8878 | 0.9049 | 0.9242 | 244/500 | 12.0% |
| **R9** | split: pless 2.0 → temp 0.9 | Yes | Yes | 8192 | 0.8349 | 0.8859 | 0.9011 | 0.9164 | 252/500 | 12.2% |
| **R7** | split: pless 2.0 → temp 0.7 | Yes | Yes | 8192 | 0.8326 | 0.8843 | 0.8995 | 0.9129 | 241/500 | 12.1% |
| **P15** | uniform pless 1.5 (thinking) | Yes | No | 8192 | 0.8242 | 0.8721 | 0.8847 | 0.8980 | 449/500 | 13.5% |
| **H8P** | split: temp(pure) 1.5 → pless 1.5 | Yes | Yes | 8192 | 0.8108 | 0.8687 | 0.8834 | 0.8980 | 449/500 | 15.7% |
| **H9P** | split: temp(pure) 1.5 → pless 2.0 | Yes | Yes | 8192 | 0.8068 | 0.8708 | 0.8888 | 0.9080 | 454/500 | 15.7% |
| **H7P** | split: temp(pure) 1.5 → pless 1.0 | Yes | Yes | 8192 | 0.8052 | 0.8689 | 0.8900 | 0.9100 | 455/500 | 16.2% |
| **H10P** | split: temp(pure) 1.5 → pless 3.0 | Yes | Yes | 8192 | 0.8030 | 0.8646 | 0.8846 | 0.9060 | 453/500 | 16.0% |
| **T15P** | split: temp(pure) 1.5 → temp(pure) 1.5 | Yes | Yes | 8192 | 0.8012 | 0.8551 | 0.8684 | 0.8820 | 441/500 | 16.5% |
| **T15N** | uniform temp 1.5 (native, thinking) | Yes | No | 8192 | 0.7992 | 0.8567 | 0.8721 | 0.8880 | 444/500 | 16.1% |
| **C** | temp_think 0.6 | Yes | No | 4096 | 0.7380 | 0.7993 | 0.8168 | 0.8340 | 417/500 | 22.7% |
| **F** | split: temp→pless | Yes | Yes | 4096 | 0.7336 | 0.7956 | 0.8123 | 0.8320 | 416/500 | 23.2% |
| **G** | split: temp→pless_norm | Yes | Yes | 4096 | 0.7318 | 0.7976 | 0.8156 | 0.8340 | 417/500 | 23.3% |
| **E** | pless_norm_think 0.6 | Yes | No | 4096 | 0.7186 | 0.7835 | 0.8025 | 0.8220 | 411/500 | 24.6% |
| **D** | pless_think 0.6 | Yes | No | 4096 | 0.7178 | 0.7768 | 0.7949 | 0.8160 | 408/500 | 24.8% |
| **B** | pless 0.7 | No | No | 512 | 0.6694 | 0.6733 | 0.6739 | 0.6740 | 337/500 | — |
| **A** | temp 0.7 | No | No | 512 | 0.6616 | 0.7037 | 0.7189 | 0.7340 | 367/500 | — |
| **H11P** | split: temp(pure) 2.0 → pless 3.0 | Yes | Yes | 8192 | 0.4578 | 0.6553 | 0.7325 | 0.8020 | 401/500 | 55.7% |
| **H12P** | split: temp(pure) 2.5 → pless 3.0 | Yes | Yes | 8192 | 0.2416 | 0.4977 | 0.6206 | 0.7360 | 368/500 | 56.0% |

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

**Does raising temp-think (0.6→0.7) and pless temp-code (0.6→1.0) help?**
- F (split: temp→pless): pass@1 = 0.7336
- H1 (split: temp 0.7 → pless 1.0): pass@1 = 0.8436
- Delta: **+0.1100** (+11.0pp)
- Head-to-head: F wins 15, H1 wins 154, ties 331

**Effect of increasing pless temp-code 1.0 → 1.5 (more permissive pless)**
- H1 (split: temp 0.7 → pless 1.0): pass@1 = 0.8436
- H2 (split: temp 0.7 → pless 1.5): pass@1 = 0.8426
- Delta: **-0.0010** (-0.1pp)
- Head-to-head: H1 wins 52, H2 wins 52, ties 396

**Effect of increasing pless temp-code 1.5 → 2.0 (most permissive pless)**
- H2 (split: temp 0.7 → pless 1.5): pass@1 = 0.8426
- H3 (split: temp 0.7 → pless 2.0): pass@1 = 0.8438
- Delta: **+0.0012** (+0.1pp)
- Head-to-head: H2 wins 43, H3 wins 50, ties 407

**Best uniform-temp baseline vs new sweep top (temp 0.7 → pless 1.0)**
- C (temp_think 0.6): pass@1 = 0.7380
- H1 (split: temp 0.7 → pless 1.0): pass@1 = 0.8436
- Delta: **+0.1056** (+10.6pp)
- Head-to-head: C wins 14, H1 wins 151, ties 335

## Truncation Analysis (thinking configs only)

| Config | Method | Tokens | Truncated | Rate | All-trunc tasks |
|--------|--------|--------|-----------|------|-----------------|
| C | temp_think 0.6 | 4096 | 1133/5000 | 22.7% | 63 |
| D | pless_think 0.6 | 4096 | 1240/5000 | 24.8% | 73 |
| E | pless_norm_think 0.6 | 4096 | 1232/5000 | 24.6% | 70 |
| F | split: temp→pless | 4096 | 1161/5000 | 23.2% | 69 |
| G | split: temp→pless_norm | 4096 | 1164/5000 | 23.3% | 62 |
| H1 | split: temp 0.7 → pless 1.0 | 8192 | 581/5000 | 11.6% | 25 |
| H2 | split: temp 0.7 → pless 1.5 | 8192 | 586/5000 | 11.7% | 22 |
| H3 | split: temp 0.7 → pless 2.0 | 8192 | 584/5000 | 11.7% | 19 |
| H4 | split: temp 0.8 → pless 1.0 | 8192 | 587/5000 | 11.7% | 23 |
| H5 | split: temp 0.8 → pless 1.5 | 8192 | 577/5000 | 11.5% | 22 |
| H6 | split: temp 0.8 → pless 2.0 | 8192 | 577/5000 | 11.5% | 25 |
| H7 | split: temp 1.5 → pless 1.0 | 8192 | 585/5000 | 11.7% | 23 |
| H8 | split: temp 1.5 → pless 1.5 | 8192 | 566/5000 | 11.3% | 23 |
| H9 | split: temp 1.5 → pless 2.0 | 8192 | 573/5000 | 11.5% | 24 |
| R7 | split: pless 2.0 → temp 0.7 | 8192 | 603/5000 | 12.1% | 25 |
| R8 | split: pless 2.0 → temp 0.8 | 8192 | 598/5000 | 12.0% | 24 |
| R9 | split: pless 2.0 → temp 0.9 | 8192 | 608/5000 | 12.2% | 25 |

## Diversity Metrics

| Config | Method | pass@1 | pass@10 | struct_div | codebleu_div | ngram_div | dataflow_div |
|--------|--------|--------|---------|-----------|-------------|-----------|-------------|
| **T15** | split: temp 1.5 → temp 1.5 (baseline) | 0.8502 | 0.9200 | 0.2052 | 0.3856 | 0.5011 | 0.2097 |
| **H9** | split: temp 1.5 → pless 2.0 | 0.8456 | 0.9140 | 0.2004 | 0.3933 | 0.5075 | 0.2183 |
| **H8** | split: temp 1.5 → pless 1.5 | 0.8454 | 0.9180 | 0.2042 | 0.3845 | 0.5022 | 0.2101 |
| **H5** | split: temp 0.8 → pless 1.5 | 0.8452 | 0.9200 | 0.1878 | 0.3570 | 0.4583 | 0.2059 |
| **H3** | split: temp 0.7 → pless 2.0 | 0.8438 | 0.9160 | 0.1845 | 0.3526 | 0.4570 | 0.1963 |
| **H1** | split: temp 0.7 → pless 1.0 | 0.8436 | 0.9140 | 0.1919 | 0.3653 | 0.4711 | 0.2049 |
| **H6** | split: temp 0.8 → pless 2.0 | 0.8428 | 0.9180 | 0.1890 | 0.3663 | 0.4682 | 0.2098 |
| **H2** | split: temp 0.7 → pless 1.5 | 0.8426 | 0.9160 | 0.1927 | 0.3674 | 0.4766 | 0.1942 |
| **H7** | split: temp 1.5 → pless 1.0 | 0.8426 | 0.9180 | 0.2013 | 0.3788 | 0.4888 | 0.2109 |
| **H4** | split: temp 0.8 → pless 1.0 | 0.8416 | 0.9160 | 0.1897 | 0.3607 | 0.4663 | 0.2043 |
| **H10** | split: temp 1.5 → pless 3.0 | 0.8378 | 0.9120 | 0.2064 | 0.3895 | 0.5116 | 0.2006 |
| **R8** | split: pless 2.0 → temp 0.8 | 0.8364 | 0.9242 | 0.1696 | 0.3075 | 0.3947 | 0.1810 |
| **R9** | split: pless 2.0 → temp 0.9 | 0.8349 | 0.9164 | 0.1692 | 0.3246 | 0.4175 | 0.1807 |
| **R7** | split: pless 2.0 → temp 0.7 | 0.8326 | 0.9129 | 0.1716 | 0.2985 | 0.3790 | 0.1799 |
| **P15** | uniform pless 1.5 (thinking) | 0.8242 | 0.8980 | 0.1589 | 0.2958 | 0.3778 | 0.1704 |
| **H8P** | split: temp(pure) 1.5 → pless 1.5 | 0.8108 | 0.8980 | 0.2057 | 0.3843 | 0.5013 | 0.2054 |
| **H9P** | split: temp(pure) 1.5 → pless 2.0 | 0.8068 | 0.9080 | 0.2168 | 0.4005 | 0.5195 | 0.2203 |
| **H7P** | split: temp(pure) 1.5 → pless 1.0 | 0.8052 | 0.9100 | 0.2141 | 0.3978 | 0.5189 | 0.2156 |
| **H10P** | split: temp(pure) 1.5 → pless 3.0 | 0.8030 | 0.9060 | 0.2021 | 0.3890 | 0.5048 | 0.2137 |
| **T15P** | split: temp(pure) 1.5 → temp(pure) 1.5 | 0.8012 | 0.8820 | 0.2078 | 0.3903 | 0.5104 | 0.2102 |
| **T15N** | uniform temp 1.5 (native, thinking) | 0.7992 | 0.8880 | 0.1999 | 0.3839 | 0.4967 | 0.2104 |
| **C** | temp_think 0.6 | 0.7380 | 0.8340 | 0.1667 | 0.3544 | 0.4579 | 0.1879 |
| **F** | split: temp→pless | 0.7336 | 0.8320 | 0.1686 | 0.3393 | 0.4350 | 0.1885 |
| **G** | split: temp→pless_norm | 0.7318 | 0.8340 | 0.1713 | 0.3431 | 0.4427 | 0.1833 |
| **E** | pless_norm_think 0.6 | 0.7186 | 0.8220 | 0.1244 | 0.2447 | 0.3133 | 0.1375 |
| **D** | pless_think 0.6 | 0.7178 | 0.8160 | 0.1305 | 0.2559 | 0.3266 | 0.1475 |
| **B** | pless 0.7 | 0.6694 | 0.6740 | 0.0067 | 0.0152 | 0.0182 | 0.0109 |
| **A** | temp 0.7 | 0.6616 | 0.7340 | 0.0573 | 0.1372 | 0.1768 | 0.0684 |
| **H11P** | split: temp(pure) 2.0 → pless 3.0 | 0.4578 | 0.8020 | 0.2008 | 0.3699 | 0.4898 | 0.1749 |
| **H12P** | split: temp(pure) 2.5 → pless 3.0 | 0.2416 | 0.7360 | 0.2006 | 0.3272 | 0.4219 | 0.1680 |

## Observations

1. **Best pass@1:** T15 (split: temp 1.5 → temp 1.5 (baseline)) at 0.8502
2. **Split decoding trades ~0.4pp pass@1 for convergence at pass@10:** F=0.832, G=0.834 vs C=0.834
3. **Split decoding is +1.6pp ahead of uniform pless thinking (D)** — temp for thinking matters
4. **Diversity comes from thinking, not code:** F retains 101% of C's structural diversity (0.1686 vs 0.1667), while D drops to 0.1305
5. **Pless without thinking is a dead end:** near-zero diversity (0.0067), no pass@1-to-pass@10 lift (0.669 → 0.674)
6. **Thinking is the dominant factor:** +7.6pp pass@1 (A→C)
7. **Sweep configs (H1–H3) leap ahead by ~+10.6pp pass@1 over C** — but **confounded with token budget**: H1–H3 used 8192 tokens (11.6% truncation), C used 4096 (22.7%). Re-running C/F at 8192 is required to isolate the temp/pless effect.
8. **pless temp-code (1.0/1.5/2.0) barely moves pass@1 at temp-think=0.7:** spread of 0.12pp across H1–H3. Diversity is also flat — the threshold p = Σ probs² is mostly insensitive in this range.