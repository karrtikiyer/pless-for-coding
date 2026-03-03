# HumanEval Temperature Sweep Results

**Models:** CodeLlama-7b, Codestral-22B, Qwen2.5-Coder-7B, Qwen3-Coder-30B  
**Methods:** pless, pless_norm, temp  
**Temperatures:** 0.7, 1.0, 1.5, 2.0, 2.5, 3.0  
**Samples per task:** 10

## Summary Table

| Model | Method | Temp | pass@1 | pass@5 | pass@10 | cover@0.5 | diversity |
|-------|--------|------|--------|--------|---------|-----------|-----------|
| CodeLlama-7b | pless | 0.7 | 3.4% | 9.3% | 12.2% | 2.4% | 0.0764 |
| CodeLlama-7b | pless | 1.0 | 3.4% | 12.0% | 17.1% | 0.6% | 0.0840 |
| CodeLlama-7b | pless | 1.5 | 2.9% | 11.6% | 18.3% | 0.0% | 0.2057 |
| CodeLlama-7b | pless | 2.0 | 1.6% | 7.4% | 14.0% | 0.0% | 0.4400 |
| CodeLlama-7b | pless | 2.5 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| CodeLlama-7b | pless | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| CodeLlama-7b | pless_norm | 0.7 | 4.1% | 10.8% | 14.0% | 3.0% | 0.0595 |
| CodeLlama-7b | pless_norm | 1.0 | 3.0% | 11.9% | 18.9% | 0.6% | 0.1300 |
| CodeLlama-7b | pless_norm | 1.5 | 2.9% | 13.1% | 22.6% | 0.0% | 0.1045 |
| CodeLlama-7b | pless_norm | 2.0 | 2.1% | 9.3% | 16.5% | 0.0% | 0.3962 |
| CodeLlama-7b | pless_norm | 2.5 | 0.1% | 0.6% | 1.2% | 0.0% | 0.0000 |
| CodeLlama-7b | pless_norm | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| CodeLlama-7b | temp | 0.7 | 4.4% | 18.3% | 29.9% | 0.0% | 0.1386 |
| CodeLlama-7b | temp | 1.0 | 1.7% | 7.6% | 13.4% | 0.0% | 0.0000 |
| Codestral-22B | pless | 0.7 | 5.9% | 12.3% | 14.0% | 4.3% | 0.2280 |
| Codestral-22B | pless | 1.0 | 7.2% | 18.1% | 22.6% | 4.9% | 0.3030 |
| Codestral-22B | pless | 1.5 | 10.5% | 36.4% | 53.0% | 2.4% | 0.3636 |
| Codestral-22B | pless | 2.0 | 15.0% | 47.3% | 64.0% | 5.5% | 0.6146 |
| Codestral-22B | pless | 2.5 | 2.2% | 9.0% | 14.0% | 0.0% | 0.2563 |
| Codestral-22B | pless | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Codestral-22B | pless_norm | 0.7 | 5.8% | 11.7% | 13.4% | 4.9% | 0.2000 |
| Codestral-22B | pless_norm | 1.0 | 7.2% | 18.4% | 23.8% | 4.3% | 0.2929 |
| Codestral-22B | pless_norm | 1.5 | 11.5% | 38.7% | 53.0% | 3.0% | 0.4450 |
| Codestral-22B | pless_norm | 2.0 | 14.8% | 46.3% | 62.2% | 6.7% | 0.5035 |
| Codestral-22B | pless_norm | 2.5 | 2.0% | 7.7% | 12.2% | 0.0% | 0.3155 |
| Codestral-22B | pless_norm | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Codestral-22B | temp | 0.7 | 15.2% | 51.1% | 70.7% | 4.3% | 0.4797 |
| Codestral-22B | temp | 1.0 | 15.1% | 51.1% | 72.6% | 4.3% | 0.5200 |
| Qwen2.5-Coder-7B | pless | 0.7 | 57.4% | 68.3% | 70.1% | 58.5% | 0.0956 |
| Qwen2.5-Coder-7B | pless | 1.0 | 57.1% | 73.2% | 76.2% | 59.8% | 0.1761 |
| Qwen2.5-Coder-7B | pless | 1.5 | 51.6% | 78.0% | 82.9% | 56.7% | 0.4199 |
| Qwen2.5-Coder-7B | pless | 2.0 | 33.7% | 71.5% | 82.3% | 31.7% | 0.5535 |
| Qwen2.5-Coder-7B | pless | 2.5 | 1.7% | 6.4% | 9.8% | 0.6% | 0.0000 |
| Qwen2.5-Coder-7B | pless | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Qwen2.5-Coder-7B | pless_norm | 0.7 | 58.7% | 68.2% | 69.5% | 62.2% | 0.0782 |
| Qwen2.5-Coder-7B | pless_norm | 1.0 | 57.5% | 73.2% | 76.8% | 61.6% | 0.2076 |
| Qwen2.5-Coder-7B | pless_norm | 1.5 | 53.7% | 79.7% | 84.8% | 59.8% | 0.3837 |
| Qwen2.5-Coder-7B | pless_norm | 2.0 | 33.4% | 71.3% | 81.7% | 31.1% | 0.5106 |
| Qwen2.5-Coder-7B | pless_norm | 2.5 | 1.7% | 6.1% | 9.1% | 0.6% | 0.0000 |
| Qwen2.5-Coder-7B | pless_norm | 3.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0000 |
| Qwen2.5-Coder-7B | temp | 0.7 | 49.6% | 81.2% | 89.0% | 53.0% | 0.4738 |
| Qwen2.5-Coder-7B | temp | 1.0 | 33.0% | 71.4% | 84.1% | 32.9% | 0.4797 |
| Qwen3-Coder-30B | pless | 0.7 | 78.4% | 78.7% | 78.7% | 78.7% | 0.0028 |
| Qwen3-Coder-30B | pless | 1.0 | 78.4% | 78.7% | 78.7% | 78.7% | 0.0024 |
| Qwen3-Coder-30B | pless | 1.5 | 78.4% | 78.7% | 78.7% | 78.7% | 0.0023 |
| Qwen3-Coder-30B | pless | 2.0 | 78.4% | 78.7% | 78.7% | 78.7% | 0.0085 |
| Qwen3-Coder-30B | pless | 2.5 | 78.0% | 78.9% | 79.3% | 78.0% | 0.0126 |
| Qwen3-Coder-30B | pless | 3.0 | 78.3% | 79.5% | 79.9% | 78.7% | 0.0305 |
| Qwen3-Coder-30B | pless_norm | 0.7 | 78.3% | 78.6% | 78.7% | 78.0% | 0.0016 |
| Qwen3-Coder-30B | pless_norm | 1.0 | 78.5% | 78.7% | 78.7% | 78.7% | 0.0025 |
| Qwen3-Coder-30B | pless_norm | 1.5 | 78.4% | 78.7% | 78.7% | 78.7% | 0.0029 |
| Qwen3-Coder-30B | pless_norm | 2.0 | 78.4% | 78.7% | 78.7% | 78.7% | 0.0087 |
| Qwen3-Coder-30B | pless_norm | 2.5 | 78.0% | 78.6% | 78.7% | 78.0% | 0.0144 |
| Qwen3-Coder-30B | pless_norm | 3.0 | 78.5% | 79.7% | 79.9% | 79.3% | 0.0380 |
| Qwen3-Coder-30B | temp | 0.7 | 78.4% | 80.2% | 80.5% | 78.0% | 0.0892 |
| Qwen3-Coder-30B | temp | 1.0 | 78.8% | 81.0% | 81.1% | 79.3% | 0.1050 |

## Per-Model Analysis

### CodeLlama-7b

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 3.4% | 3.4% | 2.9% | 1.6% | 0.0% | 0.0% |
| pless_norm | 4.1% | 3.0% | 2.9% | 2.1% | 0.1% | 0.0% |
| temp | 4.4% | 1.7% | — | — | — | — |

### Codestral-22B

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 5.9% | 7.2% | 10.5% | 15.0% | 2.2% | 0.0% |
| pless_norm | 5.8% | 7.2% | 11.5% | 14.8% | 2.0% | 0.0% |
| temp | 15.2% | 15.1% | — | — | — | — |

### Qwen2.5-Coder-7B

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 57.4% | 57.1% | 51.6% | 33.7% | 1.7% | 0.0% |
| pless_norm | 58.7% | 57.5% | 53.7% | 33.4% | 1.7% | 0.0% |
| temp | 49.6% | 33.0% | — | — | — | — |

### Qwen3-Coder-30B

| Method | T=0.7 | T=1.0 | T=1.5 | T=2.0 | T=2.5 | T=3.0 |
|--------|-------|-------|-------|-------|-------|-------|
| pless | 78.4% | 78.4% | 78.4% | 78.4% | 78.0% | 78.3% |
| pless_norm | 78.3% | 78.5% | 78.4% | 78.4% | 78.0% | 78.5% |
| temp | 78.4% | 78.8% | — | — | — | — |

## Key Findings

### Best Method at T=2.0

- **CodeLlama-7b**: **pless_norm** (pass@1=2.1%)
- **Codestral-22B**: **pless** (pass@1=15.0%)
- **Qwen2.5-Coder-7B**: **pless** (pass@1=33.7%)
- **Qwen3-Coder-30B**: **pless** (pass@1=78.4%)

### Temperature Robustness (pass@1 drop from T=0.7 to T=2.0)

- CodeLlama-7b / pless: 3.4% → 1.6% (Δ=+1.8pp)
- CodeLlama-7b / pless_norm: 4.1% → 2.1% (Δ=+2.1pp)
- Codestral-22B / pless: 5.9% → 15.0% (Δ=-9.1pp)
- Codestral-22B / pless_norm: 5.8% → 14.8% (Δ=-9.0pp)
- Qwen2.5-Coder-7B / pless: 57.4% → 33.7% (Δ=+23.7pp)
- Qwen2.5-Coder-7B / pless_norm: 58.7% → 33.4% (Δ=+25.3pp)
- Qwen3-Coder-30B / pless: 78.4% → 78.4% (Δ=+0.0pp)
- Qwen3-Coder-30B / pless_norm: 78.3% → 78.4% (Δ=-0.1pp)

### Standard Temperature Baselines (T=0.7 and T=1.0)

- CodeLlama-7b T=0.7: temp=4.4%, best pless (pless_norm)=4.1% (Δ=-0.2pp)
- CodeLlama-7b T=1.0: temp=1.7%, best pless (pless)=3.4% (Δ=+1.6pp)
- Codestral-22B T=0.7: temp=15.2%, best pless (pless)=5.9% (Δ=-9.4pp)
- Codestral-22B T=1.0: temp=15.1%, best pless (pless)=7.2% (Δ=-7.9pp)
- Qwen2.5-Coder-7B T=0.7: temp=49.6%, best pless (pless_norm)=58.7% (Δ=+9.1pp)
- Qwen2.5-Coder-7B T=1.0: temp=33.0%, best pless (pless_norm)=57.5% (Δ=+24.5pp)
- Qwen3-Coder-30B T=0.7: temp=78.4%, best pless (pless)=78.4% (Δ=+0.0pp)
- Qwen3-Coder-30B T=1.0: temp=78.8%, best pless (pless_norm)=78.5% (Δ=-0.2pp)
