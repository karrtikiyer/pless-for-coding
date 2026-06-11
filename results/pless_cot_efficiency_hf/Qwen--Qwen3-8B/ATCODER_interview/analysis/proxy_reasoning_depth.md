# Proxy Analysis: Was Pless Reasoning Ready When Passing Configs Committed?

## Method

For each of the 27 pless truncated samples where a solution signal appeared before the
repetitive loop (the "signal-before-loop" subset), we ask:

**When the best-performing config on this task committed to writing code (closed `</think>`),
what was pless's thinking at that same depth?**

### Steps
1. Per task: find the config with the highest n_correct across 10 samples (not a fixed config).
2. From that config's passing samples: compute avg thinking chars at `</think>` — the reference depth.
3. Extract pless's thinking text at that reference depth.
4. Classify: was pless already looping, actively forming a solution, still searching, or hadn't even reached that depth?

This is more principled than the earlier char-ratio approach, which:
- used a single fixed config (temp_p0.95) regardless of which config best solved the task
- used a linguistic signal ("now, code") that is unreliable in truncated text

---

## Per-task reference config and classification

| task_id | best_cfg | ref_depth | pless_total | depth% | category | note |
| --- | --- | --- | --- | --- | --- | --- |
| 417 | temp_p0.95_think_t1.0_t1.0 | 55,915 | 124,343 | 45% | C: still_searching | Stuck, trying new approach, no concrete algorithm yet |
| 558 | pless_norm_think_t1.0_t1.0 | 30,412 | 104,533 | 29% | **B: active_solution** | Has concrete Python code written at ref depth |
| 616 | temp_k20_think_t1.0_t1.0 | 23,902 | 135,037 | 18% | A: loop_already_running | Loop started before ref depth |
| 927 | temp_p0.95_k20_think_t0.6_t0.6 | 33,938 | 106,051 | 32% | A: loop_already_running | "Now, the code." repeated dozens of times |
| 990 | temp_k20_think_t1.0_t1.0 | 35,700 | 91,528 | 39% | A: loop_already_running | "No, the original problem says: 1≤M≤..." repeating |
| 1085 | temp_p0.95_k20_think_t0.6_t0.6 | 50,663 | 142,687 | 36% | A: loop_already_running | "Let me think of the process as follows." repeating |
| 1086 | temp_k20_think_t1.0_t1.0 | 54,201 | 41,539 | 130% | D: ref_exceeds_pless | Pless never reached ref depth — total thinking shorter |
| 1125 | temp_k20_think_t1.0_t1.0 | 46,287 | 90,141 | 51% | C: still_searching | Reasoning about bit manipulation — not looping, no solution |
| 1126 | pless_norm_think_t1.0_t1.0 | 36,242 | 107,166 | 34% | A: loop_already_running | "Now, code: Now, the code:" loop |
| 1171 | pless_norm_think_t1.0_t1.0 | 31,712 | 103,726 | 31% | A: loop_already_running | "right = V[-j:] if j !=0 else []" repeating |
| 1178 | temp_p0.95_k20_think_t0.6_t0.6 | 79,829 | 130,761 | 61% | C: still_searching | Still trying different approaches, no concrete plan |
| 1224 | temp_k20_think_t1.0_t1.0 | 22,734 | 98,569 | 23% | A: loop_already_running | "Now, code seems correct. Now, what about..." loop |
| 1226 | pless_norm_think_t1.0_t1.0 | 18,189 | 131,220 | 14% | A: loop_already_running | "But the code is written as:" repeating at 14% depth |
| 1328 | temp_p0.95_think_t1.0_t1.0 | 38,442 | 143,485 | 27% | C: still_searching | Still reasoning about DP, not looped, no concrete solution |

---

## Summary

| Category | N | Tasks | Interpretation |
| --- | --- | --- | --- |
| A: loop_already_running | 8 | 616,927,990,1085,1126,1171,1224,1226 | Pless entered repetitive loop BEFORE the depth where passing configs committed |
| B: active_solution | 1 | 558 | Pless had concrete code at ref depth — genuine overthinking case |
| C: still_searching | 4 | 417,1125,1178,1328 | Pless hadn't found the solution yet — still actively reasoning |
| D: ref_exceeds_pless | 1 | 1086 | Pless was truncated before even reaching the reference depth |

Note: task 370 excluded (best config also got 0 correct — no valid reference).

---

## Key finding (revised from char-ratio approach)

**Only 1 of 14 classifiable tasks (task 558) shows evidence of a genuine solution in pless
at the point where passing configs would have committed.**

The other 13:
- **8 tasks**: pless was already looping before the reference depth — the repetitive loop
  started *earlier* than when passing configs found their answer. There was no solution to
  recover at the commit point; the loop had already consumed it.
- **4 tasks**: pless was still actively searching — no solution formed, no loop yet.
  These needed more thinking, not less.
- **1 task**: pless ran out of tokens before reaching the reference depth at all.

## Implication for forced-`</think>` experiment

Injecting `</think>` at the reference depth would only recover **task 558** — the one case
where pless had a concrete solution in progress at that point. For the 8 loop-running cases,
the loop was already underway before the injection point, so the model state is incoherent.
For the 4 still-searching cases, the reasoning was incomplete — injection produces wrong code.

The right intervention for the 8 loop cases is **early loop detection** (A28 — detect loop onset
and inject `</think>` at *that* point, before the loop consumes the budget).
