## MBPP Results (257 tasks, 10 samples/task)

| Model | Method | pass@1 | pass@3 | pass@5 | pass@10 | cover@0.1 | cover@0.1 (distinct) | cover@0.3 | cover@0.3 (distinct) | cover@0.5 | cover@0.5 (distinct) | cover@0.7 | cover@0.7 (distinct) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | pless (t=1.0) | 66.0 | 75.8 | 78.5 | 81.3 | 81.3 | 81.3 | 74.7 | 36.6 | 70.0 | 14.8 | 62.6 | 2.3 |
| Qwen2.5-7B | pless_norm (t=1.0) | 65.5 | 76.0 | 78.5 | 81.3 | 81.3 | 81.3 | 75.5 | 40.1 | 70.4 | 14.0 | 62.3 | 3.1 |

*pass@k values are percentages. cover@t shows % of tasks where the fraction of correct samples >= t.*
