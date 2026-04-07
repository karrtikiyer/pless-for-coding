# HumanEval Temperature Sweep Results

**Models:** CodeLlama-7b, CodeLlama-7b-Instruct, Codestral-22B, Qwen2.5-Coder-7B, Qwen2.5-Coder-7B-Instruct, Qwen3-Coder-30B  
**Methods:** pless, pless_norm, temp  
**Temperatures:** 0.7, 1.0, 1.5, 2.0, 2.5, 3.0  
**Samples per task:** 10

## Summary Table

| Model | Method | Temp | pass@1 | pass@5 | pass@10 | cover@0.5 | diversity |
|-------|--------|------|--------|--------|---------|-----------|-----------|
| CodeLlama-7b | pless | 0.7 | 3.4% | 9.3% | 12.2% | 2.4% | 0.0764 |
| CodeLlama-7b | pless | 1.0 | 3.4% | 11.9% | 17.1% | 0.6% | 0.0616 |
| CodeLlama-7b | pless | 1.5 | 2.9% | 11.6% | 18.3% | 0.0% | 0.2057 |
| CodeLlama-7b | pless | 2.0 | 1.6% | 7.4% | 14.0% | 0.0% | 0.4400 |
| CodeLlama-7b | pless | 2.5 | 0.1% | 0.3% | 0.6% | 0.0% | 0.0000 |
| CodeLlama-7b | pless | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| CodeLlama-7b | pless_norm | 0.7 | 4.1% | 10.8% | 14.0% | 3.0% | 0.0595 |
| CodeLlama-7b | pless_norm | 1.0 | 2.9% | 11.3% | 17.7% | 0.0% | 0.1221 |
| CodeLlama-7b | pless_norm | 1.5 | 2.9% | 13.1% | 22.6% | 0.0% | 0.1045 |
| CodeLlama-7b | pless_norm | 2.0 | 2.1% | 9.3% | 16.5% | 0.0% | 0.3962 |
| CodeLlama-7b | pless_norm | 2.5 | 0.1% | 0.6% | 1.2% | 0.0% | 0.0000 |
| CodeLlama-7b | pless_norm | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| CodeLlama-7b | temp | 0.7 | 4.2% | 17.3% | 28.0% | 0.0% | 0.2079 |
| CodeLlama-7b | temp | 1.0 | 1.8% | 7.8% | 13.4% | 0.0% | 0.0000 |
| CodeLlama-7b-Instruct | pless | 0.7 | 27.4% | 31.0% | 31.1% | 28.0% | 0.0092 |
| CodeLlama-7b-Instruct | pless | 1.0 | 27.1% | 31.5% | 32.3% | 29.3% | 0.0115 |
| CodeLlama-7b-Instruct | pless | 1.5 | 26.8% | 33.8% | 34.8% | 29.3% | 0.0080 |
| CodeLlama-7b-Instruct | pless | 2.0 | 27.0% | 41.8% | 46.3% | 26.8% | 0.0716 |
| CodeLlama-7b-Instruct | pless | 2.5 | 19.1% | 44.5% | 56.1% | 19.5% | 0.2283 |
| CodeLlama-7b-Instruct | pless | 3.0 | 4.9% | 18.9% | 30.5% | 1.2% | 0.1183 |
| CodeLlama-7b-Instruct | pless_norm | 0.7 | 27.7% | 30.6% | 31.1% | 28.7% | 0.0082 |
| CodeLlama-7b-Instruct | pless_norm | 1.0 | 27.3% | 31.7% | 32.3% | 29.9% | 0.0114 |
| CodeLlama-7b-Instruct | pless_norm | 1.5 | 26.6% | 33.2% | 34.1% | 28.0% | 0.0126 |
| CodeLlama-7b-Instruct | pless_norm | 2.0 | 24.6% | 38.8% | 43.3% | 25.6% | 0.0407 |
| CodeLlama-7b-Instruct | pless_norm | 2.5 | 18.3% | 44.2% | 53.7% | 15.2% | 0.1863 |
| CodeLlama-7b-Instruct | pless_norm | 3.0 | 3.7% | 14.4% | 23.2% | 0.6% | 0.1298 |
| CodeLlama-7b-Instruct | temp | 0.7 | 25.9% | 44.4% | 51.8% | 27.4% | 0.0854 |
| CodeLlama-7b-Instruct | temp | 1.0 | 24.7% | 46.5% | 56.1% | 25.0% | 0.1325 |
| Codestral-22B | pless | 0.7 | 5.7% | 12.2% | 14.0% | 3.7% | 0.2280 |
| Codestral-22B | pless | 1.0 | 7.0% | 17.7% | 22.0% | 4.9% | 0.2789 |
| Codestral-22B | pless | 1.5 | 10.8% | 36.3% | 51.8% | 2.4% | 0.3990 |
| Codestral-22B | pless | 2.0 | 15.4% | 47.6% | 65.2% | 6.7% | 0.6230 |
| Codestral-22B | pless | 2.5 | 2.1% | 8.9% | 14.6% | 0.0% | 0.2485 |
| Codestral-22B | pless | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Codestral-22B | pless_norm | 0.7 | 5.8% | 11.7% | 13.4% | 4.9% | 0.2000 |
| Codestral-22B | pless_norm | 1.0 | 7.2% | 18.6% | 23.8% | 3.7% | 0.3104 |
| Codestral-22B | pless_norm | 1.5 | 11.7% | 38.2% | 52.4% | 3.7% | 0.4373 |
| Codestral-22B | pless_norm | 2.0 | 15.4% | 46.9% | 61.6% | 6.7% | 0.4813 |
| Codestral-22B | pless_norm | 2.5 | 2.0% | 8.0% | 12.8% | 0.0% | 0.3155 |
| Codestral-22B | pless_norm | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Codestral-22B | temp | 0.7 | 15.6% | 51.1% | 71.3% | 5.5% | 0.4725 |
| Codestral-22B | temp | 1.0 | 15.7% | 52.8% | 75.0% | 4.9% | 0.5163 |
| Qwen2.5-Coder-7B | pless | 0.7 | 56.3% | 67.8% | 70.1% | 57.9% | 0.0970 |
| Qwen2.5-Coder-7B | pless | 1.0 | 55.9% | 72.9% | 76.2% | 59.8% | 0.1725 |
| Qwen2.5-Coder-7B | pless | 1.5 | 50.9% | 77.5% | 81.7% | 56.1% | 0.4234 |
| Qwen2.5-Coder-7B | pless | 2.0 | 36.4% | 73.4% | 83.5% | 36.0% | 0.5651 |
| Qwen2.5-Coder-7B | pless | 2.5 | 1.8% | 6.5% | 9.8% | 0.6% | 0.0000 |
| Qwen2.5-Coder-7B | pless | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Qwen2.5-Coder-7B | pless_norm | 0.7 | 57.7% | 68.1% | 69.5% | 62.2% | 0.0778 |
| Qwen2.5-Coder-7B | pless_norm | 1.0 | 56.3% | 72.8% | 76.2% | 61.6% | 0.2181 |
| Qwen2.5-Coder-7B | pless_norm | 1.5 | 53.0% | 80.2% | 86.0% | 58.5% | 0.3827 |
| Qwen2.5-Coder-7B | pless_norm | 2.0 | 37.1% | 73.3% | 82.3% | 37.8% | 0.5526 |
| Qwen2.5-Coder-7B | pless_norm | 2.5 | 1.9% | 6.8% | 10.4% | 0.6% | 0.0000 |
| Qwen2.5-Coder-7B | pless_norm | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Qwen2.5-Coder-7B | temp | 0.7 | 49.7% | 80.9% | 89.0% | 54.3% | 0.4720 |
| Qwen2.5-Coder-7B | temp | 1.0 | 34.6% | 72.9% | 85.4% | 35.4% | 0.4938 |
| Qwen2.5-Coder-7B-Instruct | pless | 0.7 | 84.8% | 88.7% | 89.0% | 85.4% | 0.0094 |
| Qwen2.5-Coder-7B-Instruct | pless | 1.0 | 85.4% | 88.8% | 89.0% | 87.2% | 0.0158 |
| Qwen2.5-Coder-7B-Instruct | pless | 1.5 | 84.5% | 88.3% | 88.4% | 87.2% | 0.0485 |
| Qwen2.5-Coder-7B-Instruct | pless | 2.0 | 82.4% | 91.2% | 92.1% | 86.6% | 0.1610 |
| Qwen2.5-Coder-7B-Instruct | pless | 2.5 | 64.3% | 90.7% | 93.9% | 76.2% | 0.3535 |
| Qwen2.5-Coder-7B-Instruct | pless | 3.0 | 19.9% | 53.7% | 66.5% | 12.2% | 0.2963 |
| Qwen2.5-Coder-7B-Instruct | pless_norm | 0.7 | 84.8% | 88.6% | 89.0% | 86.6% | 0.0102 |
| Qwen2.5-Coder-7B-Instruct | pless_norm | 1.0 | 84.6% | 88.2% | 88.4% | 86.6% | 0.0153 |
| Qwen2.5-Coder-7B-Instruct | pless_norm | 1.5 | 83.9% | 88.5% | 89.0% | 87.2% | 0.0523 |
| Qwen2.5-Coder-7B-Instruct | pless_norm | 2.0 | 82.1% | 91.4% | 93.3% | 86.6% | 0.1743 |
| Qwen2.5-Coder-7B-Instruct | pless_norm | 2.5 | 63.6% | 91.5% | 94.5% | 74.4% | 0.3505 |
| Qwen2.5-Coder-7B-Instruct | pless_norm | 3.0 | 19.6% | 55.9% | 70.7% | 11.6% | 0.3054 |
| Qwen2.5-Coder-7B-Instruct | temp | 0.7 | 80.6% | 90.4% | 91.5% | 85.4% | 0.1831 |
| Qwen2.5-Coder-7B-Instruct | temp | 1.0 | 77.9% | 90.3% | 91.5% | 84.8% | 0.2518 |
| Qwen3-Coder-30B | pless | 0.7 | 75.2% | 78.0% | 78.0% | 78.0% | 0.0027 |
| Qwen3-Coder-30B | pless | 1.0 | 75.5% | 78.0% | 78.0% | 78.0% | 0.0024 |
| Qwen3-Coder-30B | pless | 1.5 | 75.3% | 78.0% | 78.0% | 77.4% | 0.0024 |
| Qwen3-Coder-30B | pless | 2.0 | 76.2% | 78.0% | 78.0% | 78.0% | 0.0086 |
| Qwen3-Coder-30B | pless | 2.5 | 75.4% | 78.3% | 78.7% | 77.4% | 0.0127 |
| Qwen3-Coder-30B | pless | 3.0 | 75.3% | 78.9% | 79.3% | 78.0% | 0.0309 |
| Qwen3-Coder-30B | pless_norm | 0.7 | 75.1% | 78.0% | 78.0% | 77.4% | 0.0017 |
| Qwen3-Coder-30B | pless_norm | 1.0 | 75.5% | 78.0% | 78.0% | 78.0% | 0.0025 |
| Qwen3-Coder-30B | pless_norm | 1.5 | 75.7% | 78.0% | 78.0% | 78.0% | 0.0030 |
| Qwen3-Coder-30B | pless_norm | 2.0 | 75.6% | 78.0% | 78.0% | 78.0% | 0.0088 |
| Qwen3-Coder-30B | pless_norm | 2.5 | 75.3% | 78.0% | 78.0% | 77.4% | 0.0148 |
| Qwen3-Coder-30B | pless_norm | 3.0 | 76.0% | 79.0% | 79.3% | 78.0% | 0.0386 |
| Qwen3-Coder-30B | temp | 0.7 | 76.0% | 79.5% | 79.9% | 77.4% | 0.0886 |
| Qwen3-Coder-30B | temp | 1.0 | 75.8% | 80.7% | 81.1% | 78.7% | 0.1045 |

## Per-Model Analysis

### CodeLlama-7b

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 3.4% | 3.4% | 2.9% | 1.6% | 0.1% | 0.0% |
| pless_norm | 4.1% | 2.9% | 2.9% | 2.1% | 0.1% | 0.0% |
| temp | 4.2% | 1.8% | — | — | — | — |

### CodeLlama-7b-Instruct

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 27.4% | 27.1% | 26.8% | 27.0% | 19.1% | 4.9% |
| pless_norm | 27.7% | 27.3% | 26.6% | 24.6% | 18.3% | 3.7% |
| temp | 25.9% | 24.7% | — | — | — | — |

### Codestral-22B

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 5.7% | 7.0% | 10.8% | 15.4% | 2.1% | 0.0% |
| pless_norm | 5.8% | 7.2% | 11.7% | 15.4% | 2.0% | 0.0% |
| temp | 15.6% | 15.7% | — | — | — | — |

### Qwen2.5-Coder-7B

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 56.3% | 55.9% | 50.9% | 36.4% | 1.8% | 0.0% |
| pless_norm | 57.7% | 56.3% | 53.0% | 37.1% | 1.9% | 0.0% |
| temp | 49.7% | 34.6% | — | — | — | — |

### Qwen2.5-Coder-7B-Instruct

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 84.8% | 85.4% | 84.5% | 82.4% | 64.3% | 19.9% |
| pless_norm | 84.8% | 84.6% | 83.9% | 82.1% | 63.6% | 19.6% |
| temp | 80.6% | 77.9% | — | — | — | — |

### Qwen3-Coder-30B

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 75.2% | 75.5% | 75.3% | 76.2% | 75.4% | 75.3% |
| pless_norm | 75.1% | 75.5% | 75.7% | 75.6% | 75.3% | 76.0% |
| temp | 76.0% | 75.8% | — | — | — | — |

## Key Findings

### Best Method at T=2.0

- **CodeLlama-7b**: **pless_norm** (pass@1=2.1%)
- **CodeLlama-7b-Instruct**: **pless** (pass@1=27.0%)
- **Codestral-22B**: **pless** (pass@1=15.4%)
- **Qwen2.5-Coder-7B**: **pless_norm** (pass@1=37.1%)
- **Qwen2.5-Coder-7B-Instruct**: **pless** (pass@1=82.4%)
- **Qwen3-Coder-30B**: **pless** (pass@1=76.2%)

### Temperature Robustness (pass@1 drop from T=0.7 to T=2.0)

- CodeLlama-7b / pless: 3.4% → 1.6% (Δ=+1.8pp)
- CodeLlama-7b / pless_norm: 4.1% → 2.1% (Δ=+2.1pp)
- CodeLlama-7b-Instruct / pless: 27.4% → 27.0% (Δ=+0.4pp)
- CodeLlama-7b-Instruct / pless_norm: 27.7% → 24.6% (Δ=+3.1pp)
- Codestral-22B / pless: 5.7% → 15.4% (Δ=-9.7pp)
- Codestral-22B / pless_norm: 5.8% → 15.4% (Δ=-9.6pp)
- Qwen2.5-Coder-7B / pless: 56.3% → 36.4% (Δ=+19.9pp)
- Qwen2.5-Coder-7B / pless_norm: 57.7% → 37.1% (Δ=+20.7pp)
- Qwen2.5-Coder-7B-Instruct / pless: 84.8% → 82.4% (Δ=+2.4pp)
- Qwen2.5-Coder-7B-Instruct / pless_norm: 84.8% → 82.1% (Δ=+2.6pp)
- Qwen3-Coder-30B / pless: 75.2% → 76.2% (Δ=-1.0pp)
- Qwen3-Coder-30B / pless_norm: 75.1% → 75.6% (Δ=-0.5pp)

### Standard Temperature Baselines (T=0.7 and T=1.0)

- CodeLlama-7b T=0.7: temp=4.2%, best pless (pless_norm)=4.1% (Δ=-0.1pp)
- CodeLlama-7b T=1.0: temp=1.8%, best pless (pless)=3.4% (Δ=+1.6pp)
- CodeLlama-7b-Instruct T=0.7: temp=25.9%, best pless (pless_norm)=27.7% (Δ=+1.8pp)
- CodeLlama-7b-Instruct T=1.0: temp=24.7%, best pless (pless_norm)=27.3% (Δ=+2.6pp)
- Codestral-22B T=0.7: temp=15.6%, best pless (pless_norm)=5.8% (Δ=-9.8pp)
- Codestral-22B T=1.0: temp=15.7%, best pless (pless_norm)=7.2% (Δ=-8.5pp)
- Qwen2.5-Coder-7B T=0.7: temp=49.7%, best pless (pless_norm)=57.7% (Δ=+8.0pp)
- Qwen2.5-Coder-7B T=1.0: temp=34.6%, best pless (pless_norm)=56.3% (Δ=+21.7pp)
- Qwen2.5-Coder-7B-Instruct T=0.7: temp=80.6%, best pless (pless)=84.8% (Δ=+4.1pp)
- Qwen2.5-Coder-7B-Instruct T=1.0: temp=77.9%, best pless (pless)=85.4% (Δ=+7.5pp)
- Qwen3-Coder-30B T=0.7: temp=76.0%, best pless (pless)=75.2% (Δ=-0.8pp)
- Qwen3-Coder-30B T=1.0: temp=75.8%, best pless (pless)=75.5% (Δ=-0.3pp)
