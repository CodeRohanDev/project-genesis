# Published baseline: CLEAR (2026-10-24)

CLEAR (Rolnick et al., 2019, "Experience Replay for Continual Learning") is the standard replay method for continual
RL and, like our Life learner, needs no task boundaries. Adapted to our PPO code (`--clear`, class Clear in ppo.py):
a fixed-size reservoir of ALL past experience (no competence gating; same budget 8192 as Life) storing observation,
action, behaviour action-probabilities, behaviour value and advantage; every PPO minibatch adds, on a replay batch,
an off-policy clipped-importance policy-gradient term (stands in for V-trace) + coef x (policy cloning KL + value
cloning MSE). Not the original IMPALA/V-trace implementation.

Tuning (tuning seeds 101-103, C1 setup, rule fixed before running: best mean over the 12 final tests):
coef 0.1 -> 88.6%, 0.3 -> 92.7%, **1.0 -> 94.1%** (chosen; it is the top of the grid, a larger value might do better).
Final: run_clear.sh MODE=final COEF=1.0, seeds 1-10, code 7899b59.

## C1: Empty -> DoorKey -> Lava, mixed sizes, 600k steps/task (12 tests incl. unseen sizes)
| Method | mean of 12 tests | unseen sizes | Lava learned / median steps to 80% |
|---|---|---|---|
| no protection | 35.3 ± 10.9 | 28.1 ± 10.2 | 3/10, 288k |
| told boundaries + distill | 94.3 ± 3.9 | 90.6 ± 5.0 | 9/10, 47k |
| **CLEAR** | 92.1 ± 2.9 | 86.9 ± 4.1 | 10/10, 159k |
| blind, reservoir only (O_replay) | 97.0 ± 1.3 | 94.2 ± 2.4 | 10/10, 82k |
| **blind, ours (O_full)** | **97.1 ± 1.5** | **94.3 ± 3.0** | 10/10, 62k |
| joint (all at once) | 96.1 ± 1.4 | 92.3 ± 2.8 | - |
Ours - CLEAR (paired over seeds): **+5.0 ± 3.1** points.

## L1: 8 stages, 6 task kinds, 500k steps/stage (11 trained tests)
| Method | final mean | mean over stages of "tasks learned so far" | stages reaching 80% (of 8) |
|---|---|---|---|
| no protection | 32.4 ± 9.6 | 46.6 ± 4.3 | 6.7 |
| **CLEAR** | 66.6 ± 2.8 | 76.3 ± 2.3 | 7.1 |
| told boundaries + distill | 92.3 ± 1.7 | 92.1 ± 2.1 | 8.0 |
| **blind, ours (L_blind)** | **92.5 ± 1.6** | **92.2 ± 2.3** | 8.0 |
| joint (all at once, 4M) | 80.7 ± 10.1 | - | - |
Ours - CLEAR final (paired): **+25.9 ± 2.9** points.
Per-test final (E8, E12u, D8, D12u, L7, L10u, Obst8, Obst10u, Cross9, Cross11u, Unlock, KeyCorridor*):
CLEAR 100 96 100 91 93 57 38 25 24 9 100 1 | ours 100 100 100 92 99 69 88 85 96 89 100 0.

## Reading
- CLEAR retains the early tasks well but learns the later, new kinds of task (Obstacles, Crossing) poorly: 7.1/8
  stages reached 80%, and Lava (3rd task) took 159k steps vs 62k for ours. Replaying all past experience with strong
  cloning (coef 1.0, chosen on the 3-task setup) holds the policy to old behaviour; competence-gated memory replays only
  mastered behaviour and our surprise reset reopens the policy for a new task.
- Caveats: PPO adaptation, not original CLEAR; coef tuned on C1 and reused for L1 (same rule as for our method, whose
  settings were also fixed earlier); coef 1.0 was the grid edge.
