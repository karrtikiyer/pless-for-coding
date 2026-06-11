# Truncated Task Solvability (51 tasks)

## Unsolvable tasks — 0 correct across all methods and all k (13 tasks)

| task_id | truncated_by | question (first 300 chars) |
| --- | --- | --- |
| 117 | both | There is a grass field that stretches infinitely. In this field, there is a negligibly small cow. Let (x, y) denote the point that is x\ \mathrm{cm} south and y\ \mathrm{cm} east of the point where the cow stands now. The cow itself is standing at (0, 0). There are also N north-south lines and M eas |
| 280 | both | There are N camels numbered 1 through N. The weight of Camel i is w_i. You will arrange the camels in a line and make them cross a bridge consisting of M parts. Before they cross the bridge, you can choose their order in the line - it does not have to be Camel 1, 2, \ldots, N from front to back - an |
| 326 | both | We have N strings of lowercase English letters: S_1, S_2, \cdots, S_N. Takahashi wants to make a string that is a palindrome by choosing one or more of these strings - the same string can be chosen more than once - and concatenating them in some order of his choice. The cost of using the string S_i  |
| 370 | both | Jumbo Takahashi will play golf on an infinite two-dimensional grid. The ball is initially at the origin (0, 0), and the goal is a grid point (a point with integer coordinates) (X, Y). In one stroke, Jumbo Takahashi can perform the following operation:  - Choose a grid point whose Manhattan distance  |
| 454 | both | Let us define the oddness of a permutation p = {p_1,\ p_2,\ ...,\ p_n} of {1,\ 2,\ ...,\ n} as \sum_{i = 1}^n \|i - p_i\|. Find the number of permutations of {1,\ 2,\ ...,\ n} of oddness k, modulo 10^9+7.  -----Constraints-----  - All values in input are integers.  - 1 \leq n \leq 50  - 0 \leq k \leq  |
| 455 | both | Snuke is introducing a robot arm with the following properties to his factory:  - The robot arm consists of m sections and m+1 joints. The sections are numbered 1, 2, ..., m, and the joints are numbered 0, 1, ..., m. Section i connects Joint i-1 and Joint i. The length of Section i is d_i.  - For ea |
| 512 | both | There is a building with 2N floors, numbered 1, 2, \ldots, 2N from bottom to top. The elevator in this building moved from Floor 1 to Floor 2N just once. On the way, N persons got on and off the elevator. Each person i (1 \leq i \leq N) got on at Floor A_i and off at Floor B_i. Here, 1 \leq A_i < B_ |
| 661 | both | Construct a sequence a = {a_1,\ a_2,\ ...,\ a_{2^{M + 1}}} of length 2^{M + 1} that satisfies the following conditions, if such a sequence exists.  - Each integer between 0 and 2^M - 1 (inclusive) occurs twice in a.  - For any i and j (i < j) such that a_i = a_j, the formula a_i \ xor \ a_{i + 1} \  |
| 962 | both | Given is a directed graph G with N vertices and M edges.  The vertices are numbered 1 to N, and the i-th edge is directed from Vertex A_i to Vertex B_i.  It is guaranteed that the graph contains no self-loops or multiple edges. Determine whether there exists an induced subgraph (see Notes) of G such |
| 1122 | both | You are going to hold a competition of one-to-one game called AtCoder Janken. (Janken is the Japanese name for Rock-paper-scissors.)N players will participate in this competition, and they are given distinct integers from 1 through N. The arena has M playing fields for two players. You need to assig |
| 1175 | both | Given are integers L and R. Find the number, modulo 10^9 + 7, of pairs of integers (x, y) (L \leq x \leq y \leq R) such that the remainder when y is divided by x is equal to y \mbox{ XOR } x.What is \mbox{ XOR }?  The XOR of integers A and B, A \mbox{ XOR } B, is defined as follows:   - When A \mbox |
| 1223 | both | Given is a permutation P of \{1, 2, \ldots, N\}. For a pair (L, R) (1 \le L \lt R \le N), let X_{L, R} be the second largest value among P_L, P_{L+1}, \ldots, P_R. Find \displaystyle \sum_{L=1}^{N-1} \sum_{R=L+1}^{N} X_{L,R}.  -----Constraints-----  -  2 \le N \le 10^5   -  1 \le P_i \le N   -  P_i  |
| 1368 | pless_only | You are given N items.  The value of the i-th item (1 \leq i \leq N) is v_i.  Your have to select at least A and at most B of these items.  Under this condition, find the maximum possible arithmetic mean of the values of selected items.  Additionally, find the number of ways to select items so that  |

## Solvable tasks — best pass@k and which method (38 tasks)

### pass@1: best method per task

| task_id | truncated_by | best_pass@1 | best_method(s) | pless | pless_norm | temp_k20_t1 | temp_p0.95_k20_t0.6 | temp_p0.95_t1 | temp_t0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | norm_only | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 90.0 | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 341 | norm_only | 80.0 | pless | 70.0 | 80.0 | 60.0 | 60.0 | 60.0 | 70.0 |
| 369 | norm_only | 10.0 | temp_p0.95_t1, temp_t0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 10.0 |
| 417 | both | 60.0 | temp_p0.95_t1 | 30.0 | 30.0 | 30.0 | 20.0 | 60.0 | 40.0 |
| 558 | pless_only | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 579 | norm_only | 10.0 | pless | 0.0 | 10.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 588 | norm_only | 70.0 | temp_p0.95_t1 | 20.0 | 10.0 | 50.0 | 40.0 | 70.0 | 40.0 |
| 616 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 70.0 | 60.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 711 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 60.0 | 80.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 739 | pless_only | 80.0 | temp_k20_t1, temp_p0.95_t1 | 70.0 | 70.0 | 80.0 | 70.0 | 80.0 | 50.0 |
| 793 | both | 100.0 | temp_p0.95_k20_t0.6 | 70.0 | 50.0 | 90.0 | 100.0 | 90.0 | 90.0 |
| 827 | norm_only | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| 927 | both | 100.0 | temp_p0.95_k20_t0.6 | 60.0 | 90.0 | 90.0 | 100.0 | 80.0 | 70.0 |
| 930 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_t0.6 | 70.0 | 60.0 | 100.0 | 100.0 | 90.0 | 100.0 |
| 990 | both | 90.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 40.0 | 40.0 | 90.0 | 90.0 | 90.0 | 80.0 |
| 991 | norm_only | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1037 | both | 70.0 | pless, temp_p0.95_k20_t0.6, temp_t0.6 | 30.0 | 70.0 | 20.0 | 70.0 | 20.0 | 70.0 |
| 1085 | both | 100.0 | temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 70.0 | 50.0 | 80.0 | 100.0 | 100.0 | 100.0 |
| 1086 | pless_only | 20.0 | temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 10.0 | 10.0 | 20.0 | 10.0 | 20.0 | 20.0 |
| 1087 | pless_only | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 90.0 | 90.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| 1090 | pless_only | 90.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 90.0 | 80.0 | 90.0 | 90.0 | 90.0 | 80.0 |
| 1125 | pless_only | 80.0 | temp_k20_t1 | 40.0 | 50.0 | 80.0 | 60.0 | 50.0 | 50.0 |
| 1126 | pless_only | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 100.0 | 80.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| 1171 | pless_only | 100.0 | pless_norm, temp_p0.95_k20_t0.6, temp_t0.6 | 100.0 | 90.0 | 90.0 | 100.0 | 80.0 | 100.0 |
| 1178 | both | 30.0 | temp_p0.95_k20_t0.6 | 10.0 | 20.0 | 10.0 | 30.0 | 20.0 | 0.0 |
| 1224 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 20.0 | 10.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1226 | pless_only | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1273 | norm_only | 30.0 | pless, temp_t0.6 | 0.0 | 30.0 | 10.0 | 20.0 | 20.0 | 30.0 |
| 1274 | norm_only | 80.0 | temp_t0.6 | 60.0 | 60.0 | 60.0 | 70.0 | 70.0 | 80.0 |
| 1277 | pless_only | 70.0 | pless_norm, temp_p0.95_t1 | 70.0 | 40.0 | 60.0 | 60.0 | 70.0 | 50.0 |
| 1328 | both | 100.0 | temp_p0.95_t1, temp_t0.6 | 90.0 | 60.0 | 90.0 | 80.0 | 100.0 | 100.0 |
| 1329 | pless_only | 90.0 | temp_p0.95_k20_t0.6 | 70.0 | 60.0 | 60.0 | 90.0 | 80.0 | 80.0 |
| 1369 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_t0.6 | 90.0 | 90.0 | 100.0 | 100.0 | 80.0 | 100.0 |
| 1370 | norm_only | 20.0 | pless, temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 10.0 | 20.0 | 20.0 | 10.0 | 20.0 | 20.0 |
| 1373 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 30.0 | 60.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1374 | both | 80.0 | temp_p0.95_t1 | 60.0 | 20.0 | 60.0 | 50.0 | 80.0 | 70.0 |
| 1426 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 80.0 | 80.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1428 | norm_only | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### pass@3: best method per task

| task_id | truncated_by | best_pass@3 | best_method(s) | pless | pless_norm | temp_k20_t1 | temp_p0.95_k20_t0.6 | temp_p0.95_t1 | temp_t0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 341 | norm_only | 100.0 | pless | 99.2 | 100.0 | 96.7 | 96.7 | 96.7 | 99.2 |
| 369 | norm_only | 30.0 | temp_p0.95_t1, temp_t0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 30.0 | 30.0 |
| 417 | both | 96.7 | temp_p0.95_t1 | 70.8 | 70.8 | 70.8 | 53.3 | 96.7 | 83.3 |
| 558 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 579 | norm_only | 30.0 | pless | 0.0 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 588 | norm_only | 99.2 | temp_p0.95_t1 | 53.3 | 30.0 | 91.7 | 83.3 | 99.2 | 83.3 |
| 616 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 99.2 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 711 | both | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 739 | pless_only | 100.0 | temp_k20_t1, temp_p0.95_t1 | 99.2 | 99.2 | 100.0 | 99.2 | 100.0 | 91.7 |
| 793 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 99.2 | 91.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 827 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 927 | both | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 | 99.2 |
| 930 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 99.2 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 990 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 83.3 | 83.3 | 100.0 | 100.0 | 100.0 | 100.0 |
| 991 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1037 | both | 99.2 | pless, temp_p0.95_k20_t0.6, temp_t0.6 | 70.8 | 99.2 | 53.3 | 99.2 | 53.3 | 99.2 |
| 1085 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 99.2 | 91.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1086 | pless_only | 53.3 | temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 30.0 | 30.0 | 53.3 | 30.0 | 53.3 | 53.3 |
| 1087 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1090 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1125 | pless_only | 100.0 | temp_k20_t1 | 83.3 | 91.7 | 100.0 | 96.7 | 91.7 | 91.7 |
| 1126 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1171 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1178 | both | 70.8 | temp_p0.95_k20_t0.6 | 30.0 | 53.3 | 30.0 | 70.8 | 53.3 | 0.0 |
| 1224 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 53.3 | 30.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1226 | pless_only | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 91.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1273 | norm_only | 70.8 | pless, temp_t0.6 | 0.0 | 70.8 | 30.0 | 53.3 | 53.3 | 70.8 |
| 1274 | norm_only | 100.0 | temp_t0.6 | 96.7 | 96.7 | 96.7 | 99.2 | 99.2 | 100.0 |
| 1277 | pless_only | 99.2 | pless_norm, temp_p0.95_t1 | 99.2 | 83.3 | 96.7 | 96.7 | 99.2 | 91.7 |
| 1328 | both | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1329 | pless_only | 100.0 | temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 99.2 | 96.7 | 96.7 | 100.0 | 100.0 | 100.0 |
| 1369 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1370 | norm_only | 53.3 | pless, temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 30.0 | 53.3 | 53.3 | 30.0 | 53.3 | 53.3 |
| 1373 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 70.8 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1374 | both | 100.0 | temp_p0.95_t1 | 96.7 | 53.3 | 96.7 | 91.7 | 100.0 | 99.2 |
| 1426 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1428 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### pass@5: best method per task

| task_id | truncated_by | best_pass@5 | best_method(s) | pless | pless_norm | temp_k20_t1 | temp_p0.95_k20_t0.6 | temp_p0.95_t1 | temp_t0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 341 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 369 | norm_only | 50.0 | temp_p0.95_t1, temp_t0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 50.0 |
| 417 | both | 100.0 | temp_p0.95_t1 | 91.7 | 91.7 | 91.7 | 77.8 | 100.0 | 97.6 |
| 558 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 579 | norm_only | 50.0 | pless | 0.0 | 50.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 588 | norm_only | 100.0 | temp_p0.95_t1 | 77.8 | 50.0 | 99.6 | 97.6 | 100.0 | 97.6 |
| 616 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 711 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 739 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.6 |
| 793 | both | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| 827 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 927 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 930 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 990 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 97.6 | 97.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| 991 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1037 | both | 100.0 | pless, temp_p0.95_k20_t0.6, temp_t0.6 | 91.7 | 100.0 | 77.8 | 100.0 | 77.8 | 100.0 |
| 1085 | both | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1086 | pless_only | 77.8 | temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 50.0 | 50.0 | 77.8 | 50.0 | 77.8 | 77.8 |
| 1087 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1090 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1125 | pless_only | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6 | 97.6 | 99.6 | 100.0 | 100.0 | 99.6 | 99.6 |
| 1126 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1171 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1178 | both | 91.7 | temp_p0.95_k20_t0.6 | 50.0 | 77.8 | 50.0 | 91.7 | 77.8 | 0.0 |
| 1224 | both | 100.0 | temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 77.8 | 50.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1226 | pless_only | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1273 | norm_only | 91.7 | pless, temp_t0.6 | 0.0 | 91.7 | 50.0 | 77.8 | 77.8 | 91.7 |
| 1274 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1277 | pless_only | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 100.0 | 97.6 | 100.0 | 100.0 | 100.0 | 99.6 |
| 1328 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1329 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1369 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1370 | norm_only | 77.8 | pless, temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 50.0 | 77.8 | 77.8 | 50.0 | 77.8 | 77.8 |
| 1373 | both | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 91.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1374 | both | 100.0 | pless_norm, temp_k20_t1, temp_p0.95_t1, temp_t0.6 | 100.0 | 77.8 | 100.0 | 99.6 | 100.0 | 100.0 |
| 1426 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1428 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### pass@10: best method per task

| task_id | truncated_by | best_pass@10 | best_method(s) | pless | pless_norm | temp_k20_t1 | temp_p0.95_k20_t0.6 | temp_p0.95_t1 | temp_t0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 341 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 369 | norm_only | 100.0 | temp_p0.95_t1, temp_t0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| 417 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 558 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 579 | norm_only | 100.0 | pless | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 588 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 616 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 711 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 739 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 793 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 827 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 927 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 930 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 990 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 991 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1037 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1085 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1086 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1087 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1090 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1125 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1126 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1171 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1178 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| 1224 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1226 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1273 | norm_only | 100.0 | pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1274 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1277 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1328 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1329 | pless_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1369 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1370 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1373 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1374 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1426 | both | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1428 | norm_only | 100.0 | pless_norm, pless, temp_k20_t1, temp_p0.95_k20_t0.6, temp_p0.95_t1, temp_t0.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

