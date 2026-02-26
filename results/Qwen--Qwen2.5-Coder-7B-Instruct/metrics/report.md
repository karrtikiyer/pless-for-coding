## MBPP Results (257 tasks, 10 samples/task)

| Model | Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (distinct) | cover@0.3 | cover@0.3 (distinct) | cover@0.5 | cover@0.5 (distinct) | cover@0.7 | cover@0.7 (distinct) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder-7B-Instruct | pless (t=1.0) | 84.3 | 85.6 | 86.0 | 86.4 | 86.4 | 86.4 | 85.6 | 5.8 | 85.2 | 1.2 | 83.7 | 0.0 |
| Qwen2.5-Coder-7B-Instruct | pless_norm (t=1.0) | 84.0 | 85.8 | 86.3 | 86.8 | 86.8 | 86.8 | 85.6 | 5.4 | 84.4 | 0.8 | 83.3 | 0.0 |

*pass@k values are percentages. cover@t shows % of tasks where the fraction of correct samples >= t.*
