# Phase A prompt comparison — Deepseek-6.7B-Instruct on CODEFORCES problem_id=4000

Generated 2026-05-26 to verify the claim "our paper-replica injects the same
prompt the paper used". Compares three prompt formats for the same APPS
problem.

## Token counts (Deepseek-Coder tokenizer)

| Source | Token count | Chars |
|---|---:|---:|
| (A) Paper's stored prompt (HF `sh0416/outputs-apps`) | 684 | 2,214 |
| (B) Bigcode-eval-harness default `apps.py::get_prompt()` | 578 | 1,741 |
| (C) Our `bench/apps/prompts.py::format_prompt_apps_instruct()` | 651 | 2,087 |

## What we inject during --paper-replica-model

Our paper-replica injection uses **(A) verbatim**. So at the *string* level we
match the paper. Tokenizer adds no BOS for this prompt (verified
`add_special_tokens=True/False` produce identical token sequences starting
with `'You'`, not `<|begin_of_sentence|>`).

What we did NOT verify: that bigcode-evaluation-harness with paper's specific
configuration produces (A). Paper's stored prompt is **not** the default
output of bigcode-eval-harness's APPS task — that produces (B). So paper used
a custom prompt formatter (likely a fork or pre-processing step that wraps
the raw APPS question in Deepseek's chat template format).

---

## (A) Paper's stored prompt — from `sh0416/outputs-apps`

```
You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
You are given an unweighted tree with $n$ vertices. Recall that a tree is a connected undirected graph without cycles.

Your task is to choose three distinct vertices $a, b, c$ on this tree such that the number of edges which belong to at least one of the simple paths between $a$ and $b$, $b$ and $c$, or $a$ and $c$ is the maximum possible. See the notes section for a better understanding.

The simple path is the path that visits each vertex at most once.


-----Input-----

The first line contains one integer number $n$ ($3 \le n \le 2 \cdot 10^5$) — the number of vertices in the tree. 

Next $n - 1$ lines describe the edges of the tree in form $a_i, b_i$ ($1 \le a_i$, $b_i \le n$, $a_i \ne b_i$). It is guaranteed that given graph is a tree.


-----Output-----

In the first line print one integer $res$ — the maximum number of edges which belong to at least one of the simple paths between $a$ and $b$, $b$ and $c$, or $a$ and $c$.

In the second line print three integers $a, b, c$ such that $1 \le a, b, c \le n$ and $a \ne, b \ne c, a \ne c$.

If there are several answers, you can print any.


-----Example-----
Input
8
1 2
2 3
3 4
4 5
4 6
3 7
3 8

Output
5
1 8 6



-----Note-----

The picture corresponding to the first example (and another one correct answer):

[Image]

If you choose vertices $1, 5, 6$ then the path between $1$ and $5$ consists of edges $(1, 2), (2, 3), (3, 4), (4, 5)$, the path between $1$ and $6$ consists of edges $(1, 2), (2, 3), (3, 4), (4, 6)$ and the path between $5$ and $6$ consists of edges $(4, 5), (4, 6)$. The union of these paths is $(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)$ so the answer is $5$. It can be shown that there is no better answer.

Your task is to implement the solution. Input will be given through standard input and your solution produces correct output to standard output. Do not include test cases in your solution.
### Response:

```

---

## (B) Bigcode-evaluation-harness's default APPS `get_prompt(doc)` output

Source: https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/apps.py

```

QUESTION:
You are given an unweighted tree with $n$ vertices. Recall that a tree is a connected undirected graph without cycles.

Your task is to choose three distinct vertices $a, b, c$ on this tree such that the number of edges which belong to at least one of the simple paths between $a$ and $b$, $b$ and $c$, or $a$ and $c$ is the maximum possible. See the notes section for a better understanding.

The simple path is the path that visits each vertex at most once.


-----Input-----

The first line contains one integer number $n$ ($3 \le n \le 2 \cdot 10^5$) — the number of vertices in the tree. 

Next $n - 1$ lines describe the edges of the tree in form $a_i, b_i$ ($1 \le a_i$, $b_i \le n$, $a_i \ne b_i$). It is guaranteed that given graph is a tree.


-----Output-----

In the first line print one integer $res$ — the maximum number of edges which belong to at least one of the simple paths between $a$ and $b$, $b$ and $c$, or $a$ and $c$.

In the second line print three integers $a, b, c$ such that $1 \le a, b, c \le n$ and $a \ne, b \ne c, a \ne c$.

If there are several answers, you can print any.


-----Example-----
Input
8
1 2
2 3
3 4
4 5
4 6
3 7
3 8

Output
5
1 8 6



-----Note-----

The picture corresponding to the first example (and another one correct answer):

[Image]

If you choose vertices $1, 5, 6$ then the path between $1$ and $5$ consists of edges $(1, 2), (2, 3), (3, 4), (4, 5)$, the path between $1$ and $6$ consists of edges $(1, 2), (2, 3), (3, 4), (4, 6)$ and the path between $5$ and $6$ consists of edges $(4, 5), (4, 6)$. The union of these paths is $(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)$ so the answer is $5$. It can be shown that there is no better answer.
Use Standard Input format
ANSWER:

```

---

## (C) Our pipeline's default (without `--paper-replica-model`)

Source: `bench/apps/prompts.py::format_prompt_apps_instruct`

```
<｜begin▁of▁sentence｜>You are a helpful coding assistant. Write clean, correct Python programs.### Instruction:
Solve the following programming problem in Python. The program must read input from standard input and write its answer to standard output. Provide only the complete Python program in a single ```python ... ``` code block, with no surrounding explanation.

Problem:
You are given an unweighted tree with $n$ vertices. Recall that a tree is a connected undirected graph without cycles.

Your task is to choose three distinct vertices $a, b, c$ on this tree such that the number of edges which belong to at least one of the simple paths between $a$ and $b$, $b$ and $c$, or $a$ and $c$ is the maximum possible. See the notes section for a better understanding.

The simple path is the path that visits each vertex at most once.


-----Input-----

The first line contains one integer number $n$ ($3 \le n \le 2 \cdot 10^5$) — the number of vertices in the tree. 

Next $n - 1$ lines describe the edges of the tree in form $a_i, b_i$ ($1 \le a_i$, $b_i \le n$, $a_i \ne b_i$). It is guaranteed that given graph is a tree.


-----Output-----

In the first line print one integer $res$ — the maximum number of edges which belong to at least one of the simple paths between $a$ and $b$, $b$ and $c$, or $a$ and $c$.

In the second line print three integers $a, b, c$ such that $1 \le a, b, c \le n$ and $a \ne, b \ne c, a \ne c$.

If there are several answers, you can print any.


-----Example-----
Input
8
1 2
2 3
3 4
4 5
4 6
3 7
3 8

Output
5
1 8 6



-----Note-----

The picture corresponding to the first example (and another one correct answer):

[Image]

If you choose vertices $1, 5, 6$ then the path between $1$ and $5$ consists of edges $(1, 2), (2, 3), (3, 4), (4, 5)$, the path between $1$ and $6$ consists of edges $(1, 2), (2, 3), (3, 4), (4, 6)$ and the path between $5$ and $6$ consists of edges $(4, 5), (4, 6)$. The union of these paths is $(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)$ so the answer is $5$. It can be shown that there is no better answer.
### Response:

```

---

## Verdict

| Observation | Status |
|---|---|
| (A) vs (B) differ substantially | Paper used a custom prompt formatter, not bigcode's default |
| (A) vs (C) differ substantially | Our default APPS formatter is different from paper's |
| (A) is what we inject when `--paper-replica-model` is set | ✓ verified |
| Tokenization adds no surprise BOS or special tokens for (A) | ✓ verified |
| (C) is what we use without `--paper-replica-model` | ✓ confirmed |

**Take-away for Phase A**: our paper-replica injection is the right path to
control for prompt-format effects. We feed the model the string paper
recorded as its prompt. The remaining pass@10 gap to paper's 0.1993 is
*not* due to prompt-string differences — it lives in the generation
framework (paper: HF transformers via bigcode-eval-harness; us: vLLM 0.21).
